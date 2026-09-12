import gradio as gr
import requests
import asyncio
from pathlib import Path
import sys
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
    gr.Markdown(
        "Загрузите видео с движением машин. Сервер обработает его с помощью YOLOv8 и вернет результат."
    )

    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Исходное видео")
            btn_submit = gr.Button("Запустить обработку", variant="primary")

        with gr.Column():
            status_output = gr.Textbox(
                label="Статус и результаты", interactive=False
            )
            output_video = gr.Video(label="Обработанное видео (с трекингом)")

    gr.Markdown("---")
    gr.Markdown("### 📜 История обработок")

    btn_refresh = gr.Button("Обновить историю")
    history_table = gr.Dataframe(
        headers=["ID", "Имя файла", "Кол-во авто", "Статус", "Дата создания"],
        datatype=["number", "str", "number", "str", "str"],
        interactive=False,
    )

    btn_submit.click(
        fn=process_video_ui,
        inputs=[input_video],
        outputs=[status_output, output_video, history_table],
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
        server_name="127.0.0.1",
        server_port=7860,
        allowed_paths=[str(OUTPUT_DIR.resolve())],
    )