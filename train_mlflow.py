import os
from pathlib import Path
import mlflow
from ultralytics import YOLO
from backend.config import MODEL_PATH

experiment_name = "BDD100K_Car_Detection"
mlflow.set_experiment(experiment_name)
os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name

def train_yolo():
    # 1. Указываем адрес локального MLflow сервера
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # 3. Пути к данным и текущим весам
    yaml_path = r"F:\ML_car_detection_service\data\data.yaml"
    
    
    model = YOLO(str(MODEL_PATH) if os.path.exists(str(MODEL_PATH) ) else "yolov8n.pt")

    print("=== Старт дообучения YOLOv8 с авто-логированием MLflow ===")

    # Запускаем обучение БЕЗ 'with mlflow.start_run()'
    results = model.train(
        data=yaml_path,
        epochs=30,
        imgsz=640,
        batch=8,
        workers=0,
        name="yolov8n_fine_tune_run_v2",
        save=True,
        mosaic = 1,
        mixup = 0.15,
        lr0=0.001
    )

    print("\n=== Обучение завершено! ===")

if __name__ == "__main__":
    train_yolo()