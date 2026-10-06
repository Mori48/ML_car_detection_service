import cv2
from ultralytics import YOLO
from  pathlib import Path
import logging
import subprocess
import os, torch
import imageio_ffmpeg as ffmpeg
from backend.config import MODEL_PATH, YOLO_CONFIDENCE, ENGINE_PATH, BASE_DIR
import asyncio
import time
import threading
import queue

logging.basicConfig(
    level=logging.WARNING,                     
    filemode="w",                             # "a" — добавлять в конец файла, "w" — перезаписывать при каждом запуске
    format="%(asctime)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s", 
    encoding="utf-8"                          
)

CUDA = torch.cuda.is_available()
DEVICE = 0 if CUDA else "cpu"
USE_TRT = CUDA and os.getenv("USE_TENSORRT", "false") == "true"

def load_model():
    if USE_TRT:
        try:
            if not ENGINE_PATH.exists():
                exported = YOLO(str(MODEL_PATH)).export(
                    format="engine", half=True, device=0, imgsz=480)
                Path(exported).replace(ENGINE_PATH)
            return YOLO(str(ENGINE_PATH), task="detect")
        except Exception as e:
            logging.warning(f"TensorRT недоступен, откат на PyTorch: {e}")
    return YOLO(str(MODEL_PATH), task="detect")

model = load_model()


def process_video(input_path , output_path ):
    if not MODEL_PATH.exists():
        logging.error(f"Файл весов не найден по пути {MODEL_PATH}")
        return
    if not input_path.exists():
        logging.error(f"Видеофайл не найден по пути {input_path}")
        return


    cap = cv2.VideoCapture(str(input_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    temp_output_path = str(output_path).replace(".mp4", "_temp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(temp_output_path), fourcc, fps, (width, height))

    track_history = {}
    valid_unique_vehicles = set()
    MIN_FRAMES_TO_CONFIRM = 15
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        
        results = model.track(
            source=frame,
            persist=True,
            conf=YOLO_CONFIDENCE,
            device=DEVICE,
            verbose=False,
            tracker="botsort.yaml",
            agnostic_nms=True,
            classes=[0, 1, 3],
        )
        tracker = model.predictor.trackers[0]

        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            current_ids = results[0].boxes.id.int().cpu().tolist()
            for track_id in current_ids:
                # Увеличиваем счетчик жизни для каждого ID
                track_history[track_id] = track_history.get(track_id, 0) + 1
                
                # Если ID продержался достаточно долго, добавляем в финальный счетчик
                if track_history[track_id] >= MIN_FRAMES_TO_CONFIRM:
                    valid_unique_vehicles.add(track_id)
                    
            annotated_frame = results[0].plot()
        else:
            annotated_frame = frame
        cv2.putText(
        annotated_frame,
        f"Unique Vehicles: {len(valid_unique_vehicles)}",
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
    
    return len(valid_unique_vehicles)


class ThreadedCamera:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(str(source))
        self.q = queue.Queue(maxsize=30)
        self.stopped = False
        self.t = threading.Thread(target=self.update, daemon=True)
        self.t.start()

    def update(self):
        while not self.stopped:
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    break
                self.q.put(frame)
            else:
                time.sleep(0.01)
                
    def read(self):
        while True:
            try:
                return True, self.q.get(timeout=0.5)
            except queue.Empty:
                if self.stopped:
                    return False, None

    def release(self):
        self.stopped = True
        self.cap.release()



async def websocket_video_generator(input_path):
    cap = ThreadedCamera(str(input_path))
    fps_smooth = 0.0

    prev_time = time.time()
    track_history = {}
    valid_unique_vehicles = set()
    MIN_FRAMES_TO_CONFIRM = 15 
    frame_count = 0
    last_boxes = []
    last_ids = []
    
    while True:
        ret, frame = await asyncio.to_thread(cap.read)
        if not ret:
            break
            
        frame = cv2.resize(frame, (854, 480))
        frame_count += 1
        
        results = await asyncio.to_thread(
        model.track,
        source=frame,
        persist=True,
        conf=YOLO_CONFIDENCE,
        imgsz=480,
        device=DEVICE,          
        max_det=35,
        verbose=False,
        tracker="botsort.yaml",
        agnostic_nms=True,
        classes=[0, 1, 3],
    )


        if results[0].boxes is not None and results[0].boxes.id is not None:
            last_boxes = results[0].boxes.xyxy.int().cpu().tolist()
            last_ids = results[0].boxes.id.int().cpu().tolist()
            last_confs = results[0].boxes.conf.float().cpu().tolist()
            for track_id in last_ids:
                track_history[track_id] = track_history.get(track_id, 0) + 1
                if track_history[track_id] >= MIN_FRAMES_TO_CONFIRM:
                    valid_unique_vehicles.add(track_id)
        else:
            last_boxes, last_ids, last_confs = [], [], []

        annotated_frame = frame.copy()

        # ОТРИСОВКА 
        for box, track_id, conf in zip(last_boxes, last_ids, last_confs):
            # Рисуем только подтвержденные машины, чтобы убрать артефакты и мерцание
            if track_history.get(track_id, 0) >= MIN_FRAMES_TO_CONFIRM:
                x1, y1, x2, y2 = box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 144, 30), 2)
                cv2.putText(
                    annotated_frame, 
                    f"ID: {track_id} Conf: {conf:.2f}",
                    (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 144, 30), 
                    2 
                )

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        if fps_smooth == 0.0:
            fps_smooth = fps
        else:
            fps_smooth = 0.9 * fps_smooth + 0.1 * fps
            
        cv2.putText(
            annotated_frame,
            f"Unique Vehicles: {len(valid_unique_vehicles)}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        cv2.putText(
            annotated_frame,
            f"FPS: {fps_smooth:.1f}", 
            (30, 90),          
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),      
            2
        )

        # Сжимаем кадр в JPEG 
        _, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        frame_bytes = buffer.tobytes()
        
        yield frame_bytes
        
        process_time = time.time() - curr_time
        delay = max(0.001, 0.03 - process_time)
        await asyncio.sleep(0.001)


    yield len(valid_unique_vehicles)
    cap.release()  