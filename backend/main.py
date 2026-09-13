from pathlib import Path
from contextlib import asynccontextmanager
import aiofiles
import sys

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import gradio as gr
from backend.db.database import init_db, get_db
from backend.db.models import VideoProcessingLog, ProcessingStatus
from backend.schemas import VideoProcessResponse, VideoHistoryItem
from backend.tracker import process_video
from backend.config import OUTPUT_DIR, UPLOAD_DIR
import asyncio
from frontend.app import demo
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@asynccontextmanager
async def lifespan(app: FastAPI ):
    await init_db()
    yield

app = FastAPI(
    title = "Car detection",
    lifespan=lifespan
    )

app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")
app = gr.mount_gradio_app(app, demo, path="/gradio")

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
        record.vehicles_count = process_video(input_path, output_path)
        record.output_path = str(output_path)
        record.status = ProcessingStatus.COMPLETED
    except Exception as e:
        record.status = ProcessingStatus.FAILED
        await db.commit()
        raise HTTPException(
                    status_code=500,
                    detail="Ошибка при обработке видео: {e}"
                )
    await db.commit()
    await db.refresh(record)
    return record



    



