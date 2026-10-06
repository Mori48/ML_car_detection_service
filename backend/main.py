from pathlib import Path
from contextlib import asynccontextmanager
import aiofiles
import sys
from fastapi.concurrency import run_in_threadpool
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from backend.db.database import init_db, get_db
from backend.db.models import VideoProcessingLog, ProcessingStatus
from backend.schemas import VideoProcessResponse, VideoHistoryItem
from backend.tracker import process_video
from backend.config import OUTPUT_DIR, UPLOAD_DIR
import asyncio
from sqlalchemy import update
from fastapi import WebSocket, WebSocketDisconnect
from backend.tracker import websocket_video_generator

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()  
    print("База данных инициализирована.")

    #  Очистка зависших записей 
    async for db in get_db():
        try:
            stmt = (
                update(VideoProcessingLog)
                .where(VideoProcessingLog.status == ProcessingStatus.PROCESSING)
                .values(status=ProcessingStatus.FAILED)
            )
            await db.execute(stmt)
            await db.commit()
            print("База данных очищена от зависших сессий.")
        except Exception as e:
            print(f"Ошибка при очистке БД: {e}")
        break  # Выходим после первой успешной сессии
        
    yield

app = FastAPI(
    title = "Car detection",
    lifespan=lifespan
    )

app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")

@app.get("/health", tags = ["Helth Check"])
async def health_check():
    return {"status":"ok",
            "service":"car_detection"}


@app.get("/history",response_model=list[VideoHistoryItem])
async def get_history(
    db: AsyncSession =  Depends(get_db)
    ):
    request = select(VideoProcessingLog).order_by(VideoProcessingLog.created_at.desc())
    result = await db.execute(request)
    logs = result.scalars().all()
    return logs

@app.post("/process_video", response_model=VideoProcessResponse, tags=["Tracking"])
async def procces_video_endpoint(
    file: UploadFile = File(...),
    db:AsyncSession = Depends(get_db)
    ):
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(
            status_code=400,
            detail="Неподдерживаемый формат. Загрузите видео .mp4, .avi или .mov"
        )
    input_path = UPLOAD_DIR / file.filename
    async with aiofiles.open(input_path,"wb") as buffer:
        while content := await file.read(1024 * 1024):
            await buffer.write(content)
    await file.close()
    record = VideoProcessingLog(filename= file.filename,
                                input_path = str(input_path),
                                status = ProcessingStatus.PROCESSING)

    db.add(record)
    await db.commit()
    await db.refresh(record)

    output_path = OUTPUT_DIR / f"processed_{record.id}_{file.filename}"
    try:
        record.vehicles_count = await run_in_threadpool(process_video, input_path, output_path)
        record.output_path = str(output_path)
        record.status = ProcessingStatus.COMPLETED
    except Exception as e:
        record.status = ProcessingStatus.FAILED
        await db.commit()
        raise HTTPException(
                    status_code=500,
                    detail=f"Ошибка при обработке видео: {e}"
                )
    await db.commit()
    await db.refresh(record)
    return record


# async def db_update_wrapper(generator, db: AsyncSession, record: VideoProcessingLog):
#     try:
#         async for item in generator:
#             if isinstance(item, bytes):
#                 # Если это кадр видео — пробрасываем его дальше в браузер
#                 yield item
#             elif isinstance(item, int):
#                 # Если пришло число машин — видео закончилось, сохраняем в БД!
#                 record.vehicles_count = item
#                 record.status = ProcessingStatus.COMPLETED
#                 await db.commit()
#     except Exception as e:
#         record.status = ProcessingStatus.FAILED
#         await db.commit()
#         print(f"Ошибка потока: {e}")


# @app.post("/stream_video", tags=["Streaming"])
# async def stream_video_endpoint(
#     file: UploadFile = File(...),
#     db: AsyncSession = Depends(get_db)
# ):
#     if not file.filename.lower().endswith((".mp4", ".avi", ".mov")):
#         raise HTTPException(
#             status_code=400,
#             detail="Неподдерживаемый формат. Загрузите видео .mp4, .avi или .mov"
#         )
        
#     input_path = UPLOAD_DIR / file.filename
#     async with aiofiles.open(input_path, "wb") as buffer:
#         while content := await file.read(1024 * 1024):
#             await buffer.write(content)
#     await file.close()

#     record = VideoProcessingLog(
#         filename=file.filename,
#         input_path=str(input_path),
#         status=ProcessingStatus.PROCESSING
#     )
#     db.add(record)
#     await db.commit()
#     await db.refresh(record)

#     return StreamingResponse(
#         db_update_wrapper(stream_video_generator(input_path), db, record),
#         media_type="multipart/x-mixed-replace; boundary=frame"
#     )
    


@app.post("/upload_for_stream", tags=["Streaming"])
async def upload_for_stream_endpoint(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    # 1. Проверяем формат файла
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(
            status_code=400,
            detail="Неподдерживаемый формат. Загрузите видео .mp4, .avi или .mov"
        )
        
    # 2. Сохраняем файл на диск
    input_path = UPLOAD_DIR / file.filename
    async with aiofiles.open(input_path, "wb") as buffer:
        while content := await file.read(1024 * 1024):
            await buffer.write(content)
    await file.close()

    # 3. Создаем запись в БД
    record = VideoProcessingLog(
        filename=file.filename,
        input_path=str(input_path),
        status=ProcessingStatus.PROCESSING
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)

    # 4. Возвращаем ID, чтобы фронтенд знал, к какому сокету подключаться
    return {"record_id": record.id, "filename": file.filename}


@app.websocket("/ws/stream/{record_id}")
async def websocket_stream_endpoint(
    websocket: WebSocket, 
    record_id: int, 
    db: AsyncSession = Depends(get_db)
):
    # 1. Принимаем WebSocket соединение
    await websocket.accept()
    
    # 2. Ищем запись в БД по переданному ID
    request = select(VideoProcessingLog).where(VideoProcessingLog.id == record_id)
    result = await db.execute(request)
    record = result.scalar_one_or_none()
    
    if not record:
        await websocket.close(code=1008, reason="Видео не найдено")
        return

    try:
        # 3. Запускаем наш генератор чистых байтов
        generator = websocket_video_generator(record.input_path)
        
        async for item in generator:
            if isinstance(item, bytes):
                # Если пришли байты картинки — пуляем их прямо в сокет
                await websocket.send_bytes(item)
            elif isinstance(item, int):
                # Если пришло число (конец видео) — обновляем количество машин в БД
                record.vehicles_count = item
                record.status = ProcessingStatus.COMPLETED
                await db.commit()
                
        # Нормально закрываем сокет по завершении видео
        await websocket.close(code=1000)
        
    except WebSocketDisconnect:
        print(f"Клиент отключился от стрима видео ID {record_id}")
        record.status = ProcessingStatus.FAILED
        await db.commit()
    except Exception as e:
        print(f"Ошибка WebSocket потока: {e}")
        record.status = ProcessingStatus.FAILED
        await db.commit()
        await websocket.close(code=1011)
