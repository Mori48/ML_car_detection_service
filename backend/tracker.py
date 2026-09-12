import cv2
from ultralytics import YOLO
from  pathlib import Path
import logging
import subprocess
import os
import imageio_ffmpeg as ffmpeg
from backend.config import MODEL_PATH, YOLO_CONFIDENCE




logging.basicConfig(
    level=logging.WARNING,                     
    filemode="w",                             # "a" — добавлять в конец файла, "w" — перезаписывать при каждом запуске
    format="%(asctime)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s", # Шаблон строки
    encoding="utf-8"                          # Чтобы не было проблем с русским языком
)

def process_video(input_path , output_path ):
    if not MODEL_PATH.exists():
        logging.error(f"Файл весов не найден по пути {MODEL_PATH}")
        return
    if not input_path.exists():
        logging.error(f"Видеофайл не найден по пути {input_path}")
        return

    model = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(str(input_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    temp_output_path = str(output_path).replace(".mp4", "_temp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(temp_output_path), fourcc, fps, (width, height))

    unique_vehicle_id_set = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            source=frame,
            persist=True,
            conf=YOLO_CONFIDENCE,
            verbose=False
            
        )
        if results[0].boxes is not None and results[0].boxes.id is not None:
            unique_vehicle_id_set.update(results[0].boxes.id.int().cpu().tolist())
            annotated_frame = results[0].plot()
        else:
            annotated_frame = frame
        cv2.putText(
        annotated_frame,
        f"Unique Vehicles: {len(unique_vehicle_id_set)}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
        )
        out.write(annotated_frame)
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    try:
        ffmpeg_exe = ffmpeg.get_ffmpeg_exe()

        ffmpeg_cmd = [
            ffmpeg_exe,
            "-y",
            "-i", temp_output_path,
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "23",
            str(output_path)
        ]
        subprocess.run(
            ffmpeg_cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
    except Exception as e:
        logging.warning(
            f"Не удалось перекодировать с помощью FFmpeg: {e}. Переименовываем временный файл."
        )
        if os.path.exists(temp_output_path):
            if os.path.exists(str(output_path)):
                os.remove(str(output_path))
            os.rename(temp_output_path, str(output_path))
    
    return len(unique_vehicle_id_set)





    