import gradio as gr
import requests
import asyncio
from pathlib import Path
import sys
import io
from PIL import Image
import websockets
import os, tempfile
API_URL = os.getenv("API_URL", "http://localhost:8000")
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def process_video_ui(video_path: str):
    if not video_path:
        return "Пожалуйста, загрузите видеофайл.", None, fetch_history_ui()

    try:
        filename = Path(video_path).name

        with open(video_path, "rb") as f:
            files = {"file": (filename, f, "video/mp4")}
            response = requests.post(f"{API_URL}/process_video", files=files)

        if response.status_code == 200:
            data = response.json()

            vehicles_count = data.get("vehicles_count", 0)
            status = data.get("status", "UNKNOWN")
            output_rel_path = data.get("output_path")

            video_file_path = None
            if output_rel_path:
                clean_filename = Path(output_rel_path).name
                r = requests.get(f"{API_URL}/static/{clean_filename}", timeout=300)
                local = Path(tempfile.gettempdir()) / clean_filename
                local.write_bytes(r.content)
                video_file_path = str(local)

            status_msg = f"Успешно обработано! Статус: {status}"
            count_msg = f"Насчитано транспортных средств: {vehicles_count}"
            updated_history = fetch_history_ui()

            return (
                f"{status_msg}\n{count_msg}",
                video_file_path,  
                updated_history,
            )
        else:
            try:
                detail = response.json().get("detail", "Неизвестная ошибка")
            except Exception:
                detail = response.text
            return (
                f"Ошибка сервера ({response.status_code}): {detail}",
                None,
                fetch_history_ui(),
            )
    except Exception as e:
        return (
            f"Ошибка при подключении к бэкенду: {e}",
            None,
            fetch_history_ui(),
        )


async def stream_video_ui(video_path: str):
    """
    Асинхронный генератор, который загружает видео и читает потоковый ответ через WebSocket.
    """
    if not video_path:
        yield None, "Пожалуйста, загрузите видеофайл.", fetch_history_ui()
        return

    try:
        filename = Path(video_path).name
        
        # 1. Сигнализируем UI о начале работы
        yield None, "Загрузка видео на сервер...", gr.skip()

        # Выполняем загрузку файла в отдельном потоке (to_thread), 
        # чтобы блокирующий requests не вешал асинхронный event loop Gradio
        def upload_file():
            with open(video_path, "rb") as f:
                files = {"file": (filename, f, "video/mp4")}
                return requests.post(f"{API_URL}/upload_for_stream", files=files)
                
        response = await asyncio.to_thread(upload_file)

        if response.status_code != 200:
            yield None, f"Ошибка загрузки: {response.text}", gr.skip()
            return
            
        record_id = response.json().get("record_id")

        # 2. Формируем URL для WebSocket
        # Заменяем http/https на ws/wss
        ws_url = API_URL.replace("http://", "ws://").replace("https://", "wss://") + f"/ws/stream/{record_id}"
        
        yield None, f"Подключение к стриму (ID: {record_id})...", gr.skip()

        # 3. Подключаемся по WebSocket и читаем кадры
        async with websockets.connect(ws_url) as websocket:
            while True:
                try:
                    message = await websocket.recv()
                    
                    if isinstance(message, bytes):
                        # Превращаем сырые байты (JPEG) в картинку для Gradio
                        img = Image.open(io.BytesIO(message))
                        yield img, "Стрим идет (Real-time)...", gr.skip()
                    else:
                        # Если сервер прислал текст (например, итоговое число машин)
                        pass
                        
                except websockets.exceptions.ConnectionClosed:
                    # Сокет закрыт (видео закончилось)
                    break

        # 4. Когда цикл закончился, обновляем таблицу из БД (статус поменяется на COMPLETED)
        yield gr.skip(), "Стрим успешно завершен!", fetch_history_ui()

    except Exception as e:
        yield None, f"Сбой стриминга: {e}", gr.skip()


def fetch_history_ui():
    try:
        response = requests.get(f"{API_URL}/history")
        if response.status_code == 200:
            history_data = response.json()
            table_data = []
            for item in history_data:
                table_data.append(
                    [
                        item.get("id"),
                        item.get("filename"),
                        item.get("vehicles_count"),
                        item.get("status"),
                        item.get("created_at"),
                    ]
                )
            return table_data
        else:
            return []
    except Exception:
        return []

with gr.Blocks(title="Vehicle Tracking & Detection") as demo:
    gr.Markdown("# 🚗 Детекция и трекинг транспортных средств")
    
    with gr.Tabs():
        # === Вкладка 1: Классический батч-анализ ===
        with gr.Tab("Глубокая аналитика (Офлайн)"):
            gr.Markdown("Загрузите видео. Сервер полностью обработает его, сохранит результат на диск и выдаст готовый файл.")
            with gr.Row():
                with gr.Column():
                    input_video_batch = gr.Video(label="Исходное видео")
                    btn_submit_batch = gr.Button("Запустить обработку", variant="primary")

                with gr.Column():
                    status_output_batch = gr.Textbox(label="Статус и результаты", interactive=False)
                    output_video_batch = gr.Video(label="Готовое видео")
                    
        # === Вкладка 2: Новый стриминг ===
        with gr.Tab("Потоковый анализ (Real-time)"):
            gr.Markdown("Видео обрабатывается и транслируется кадр за кадром без сохранения на жесткий диск. Максимальная производительность.")
            with gr.Row():
                with gr.Column():
                    input_video_stream = gr.Video(label="Исходное видео")
                    btn_submit_stream = gr.Button("Запустить стрим", variant="primary")

                with gr.Column():
                    status_output_stream = gr.Textbox(label="Статус потока", interactive=False)
                    # Используем gr.Image для отображения потока кадров
                    output_image_stream = gr.Image(label="Live Трансляция", interactive=False)

    gr.Markdown("---")
    gr.Markdown("### 📜 История обработок")

    btn_refresh = gr.Button("Обновить историю")
    history_table = gr.Dataframe(
        headers=["ID", "Имя файла", "Кол-во авто", "Статус", "Дата создания"],
        datatype=["number", "str", "number", "str", "str"],
        interactive=False,
    )

    # Привязываем кнопки к функциям
    btn_submit_batch.click(
        fn=process_video_ui,
        inputs=[input_video_batch],
        outputs=[status_output_batch, output_video_batch, history_table],
    )

    btn_submit_stream.click(
        fn=stream_video_ui,
        inputs=[input_video_stream],
        # Обновляем картинку, статус и таблицу истории
        outputs=[output_image_stream, status_output_stream, history_table],
    )

    btn_refresh.click(
        fn=fetch_history_ui,
        inputs=[],
        outputs=[history_table],
    )

    demo.load(
        fn=fetch_history_ui,
        inputs=[],
        outputs=[history_table],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=[tempfile.gettempdir()],
    )