import os
from pathlib import Path



BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "videos"
OUTPUT_DIR = DATA_DIR / "outputvideos"
MODEL_PATH = BASE_DIR / "data" / "weights" / "best.pt"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

API_URL = "http://127.0.0.1:8000"

DB_PATH = Path(__file__).resolve().parent.parent.parent / "app.db"
DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"

YOLO_CONFIDENCE = 0.1