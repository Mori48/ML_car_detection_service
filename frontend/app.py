import gradio as gr
import requests
import asyncio
from pathlib import Path
import sys
import io
from PIL import Image
from backend.config import OUTPUT_DIR, API_URL

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
                video_file_path = str((OUTPUT_DIR / clean_filename).resolve())

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

def stream_video_ui(video_path: str):
    """
    Генератор, который читает потоковый ответ от FastAPI и покадрово обновляет UI
    """
    if not video_path:
        yield None, "Пожалуйста, загрузите видеофайл.", fetch_history_ui()
        return

    try:
        filename = Path(video_path).name
        
        # Сигнализируем UI о начале работы
        yield None, "Установка соединения...", gr.skip()

        with open(video_path, "rb") as f:
            files = {"file": (filename, f, "video/mp4")}
            # Важно: stream=True позволяет читать ответ сервера по частям, не дожидаясь конца
            response = requests.post(f"{API_URL}/stream_video", files=files, stream=True)

        if response.status_code != 200:
            yield None, f"Ошибка потока: {response.status_code}", gr.skip()
            return

        bytes_data = b''
        # Читаем поток байтов кусками (чанками)
        for chunk in response.iter_content(chunk_size=8192):
            bytes_data += chunk
            # Ищем маркеры начала (FF D8) и конца (FF D9) JPEG-файла
            a = bytes_data.find(b'\xff\xd8')
            b = bytes_data.find(b'\xff\xd9')
            
            if a != -1 and b != -1:
                jpg_bytes = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                
                try:
                    # Превращаем байты в картинку для Gradio
                    img = Image.open(io.BytesIO(jpg_bytes))
                    # Выдаем кадр в интерфейс. gr.skip() значит "не трогать таблицу истории"
                    yield img, "Стрим идет (Real-time)...", gr.skip()
                except Exception:
                    continue

        # Когда цикл закончился (видео кончилось), обновляем таблицу из БД
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
        allowed_paths=[str(OUTPUT_DIR.resolve())],
    )