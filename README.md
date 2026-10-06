# Car Detection Service

End-to-end computer vision service for detecting and tracking vehicles in video using YOLO.

The application provides:

- REST API for offline video processing
- WebSocket streaming of annotated frames in real time
- Gradio web interface
- PostgreSQL persistence for processing history and vehicle counts
- Optional TensorRT inference on NVIDIA GPUs (the engine is built automatically on first start)
- CPU and GPU deployment modes selected through Docker Compose files

The project is containerized and consists of three services (frontend, backend, database) communicating through an internal Docker network.

## Tech Stack

**Backend:** Python 3.10, FastAPI, Uvicorn, OpenCV, PyTorch, Ultralytics YOLO, TensorRT (GPU mode), WebSockets, SQLAlchemy, asyncpg

**Frontend:** Gradio

**Database:** PostgreSQL 15

**Experiment tracking (training):** MLflow

**Infrastructure:** Docker, Docker Compose

## Architecture

```text
┌─────────────────────┐
│       Browser       │
└──────────┬──────────┘
           │ HTTP
           ▼
┌─────────────────────┐
│      Frontend       │
│  Gradio UI :7860    │
└──────────┬──────────┘
           │ REST API / WebSocket
           ▼
┌─────────────────────┐      ┌─────────────────────┐
│      Backend        │      │      Database       │
│ FastAPI + YOLO      │◄────►│     PostgreSQL      │
│ OpenCV + WebSockets │      │        :5432        │
│        :8000        │      │                     │
└─────────────────────┘      └─────────────────────┘
```

### Backend (`car_backend`, port 8000)

- accepts video uploads (`.mp4`, `.avi`, `.mov`);
- runs YOLO detection with multi-object tracking;
- offline mode: processes the whole video and serves the annotated result from `/static`;
- streaming mode: sends JPEG frames with boxes, track IDs and FPS over a WebSocket;
- counts unique vehicles and stores processing history in PostgreSQL;
- marks interrupted sessions as failed on startup.

### Frontend (`car_frontend`, port 7860)

- offline analysis tab: upload a video, get the annotated result;
- real-time tab: live annotated stream with FPS and vehicle counter;
- processing history table.

The frontend communicates with the backend only over HTTP/WebSocket (`API_URL`), so it does not share code or files with the backend.

### Database (`car_db`, port 5432, localhost only)

Stores processing history, file metadata, processing status and vehicle counts. Data is persisted in a Docker volume.

## Model

| Item | Value |
|:---|:---|
| Architecture | YOLO (Ultralytics) |
| Training dataset | [FILL: dataset name, e.g. BDD100K] |
| Detected classes | [FILL: names of classes 0, 1, 3 used in `tracker.py`] |
| Training | 15 epochs, runs tracked in MLflow |

Validation metrics after 15 epochs:

| Metric | Score |
|:---|---:|
| Precision | 0.554 |
| Recall | 0.408 |
| mAP@50 | 0.427 |
| mAP@50:95 | 0.265 |

### Inference modes

| Mode | Backend | FPS |
|:---|:---|---:|
| CPU | PyTorch (`.pt`) | [FILL] |
| GPU | PyTorch (`.pt`) | [FILL] |
| GPU | TensorRT (`.engine`, FP16) | [FILL] |

Measured on [FILL: GPU, e.g. NVIDIA RTX 3050 Laptop 4 GB] with [FILL: test video, resolution], streaming mode, `imgsz=480`.

### TensorRT

In GPU mode the service exports the weights to a TensorRT engine on first start and caches it in a Docker volume (`engine_cache`). Engines are specific to the GPU and TensorRT version, so they are never stored in the repository (`*.engine` is git-ignored). If the engine cannot be loaded, the service falls back to PyTorch.

## Tracking

Tracking uses the Ultralytics **BoT-SORT** tracker (`botsort.yaml`).

A track is counted as a unique vehicle only after it has been observed for at least 15 frames. This suppresses short-lived false positives and ID flicker. In streaming mode only confirmed tracks are drawn.

## Project Structure

```text
ML_car_detection_service/
├── backend/
│   ├── db/                  # SQLAlchemy models and database setup
│   ├── main.py              # FastAPI app, REST and WebSocket endpoints
│   ├── tracker.py           # model loading, detection, tracking, streaming
│   ├── config.py
│   └── schemas.py
├── frontend/
│   ├── app.py               # Gradio interface
│   └── config.py
├── data/                    # model weights, uploads, processed videos
├── requirements/
│   ├── back.txt
│   └── front.txt
├── docker/
│   ├── Dockerfile.backend       # CPU image
│   ├── Dockerfile.backend.gpu   # CUDA + TensorRT image
│   ├── Dockerfile.frontend
│   ├── docker-compose.yaml
│   ├── docker-compose.gpu.yaml  # GPU override
│   └── .env.example
└── README.md
```

## Installation and Setup

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (on Windows with WSL 2 integration)
- [Git](https://git-scm.com/)
- For GPU mode: an NVIDIA GPU with a recent driver and Docker GPU support. Check it with:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### 1. Clone the repository

```bash
git clone https://github.com/Mori48/ML_car_detection_service.git
cd ML_car_detection_service/docker
```

### 2. Configure the environment

Create `docker/.env` from the example and set your own database password:

```bash
cp .env.example .env        # Windows (cmd/PowerShell): copy .env.example .env
```

```text
POSTGRES_USER=admin
POSTGRES_PASSWORD=change_me
POSTGRES_DB=car_detection
```

### 3a. Run on CPU

```bash
docker compose up -d --build
```

### 3b. Run on GPU (TensorRT)

```bash
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up -d --build
```

On the first GPU start the TensorRT engine is built, which can take several minutes. Follow progress with:

```bash
docker compose logs -f backend
```

Subsequent starts load the cached engine in seconds.

### 4. Open the application

| Service | URL |
|:---|:---|
| Web interface | http://localhost:7860 |
| API docs (Swagger) | http://localhost:8000/docs |
| Health check | http://localhost:8000/health |

## API

| Method | Endpoint | Description |
|:---|:---|:---|
| `POST` | `/process_video` | Upload a video, process it offline, return count and result path |
| `POST` | `/upload_for_stream` | Upload a video for streaming, returns `record_id` |
| `WS` | `/ws/stream/{record_id}` | Stream annotated JPEG frames, final message is the vehicle count |
| `GET` | `/history` | Processing history |
| `GET` | `/health` | Service health |
| `GET` | `/static/{file}` | Processed videos |

Interactive documentation: http://localhost:8000/docs

## Useful Docker Commands

Run from the `docker/` directory (add `-f docker-compose.yaml -f docker-compose.gpu.yaml` in GPU mode).

```bash
docker compose ps                  # container status
docker compose logs -f backend     # follow backend logs
docker compose restart backend     # restart a service
docker compose down                # stop, keep database data
docker compose down -v             # stop and delete database data (warning!)
```

Rebuild the TensorRT engine (for example after replacing the weights):

```bash
docker volume rm docker_engine_cache
```

## Known Limitations

- Tracker state is shared across requests, so the service is designed for one stream at a time.
- CPU mode is intended for functional testing; real-time throughput requires a GPU.
- Model quality is limited by short training (15 epochs); see metrics above.

## Notes

- Frontend and backend communicate through the Docker Compose network.
- PostgreSQL data is persisted in a Docker volume.
- TensorRT acceleration requires a compatible NVIDIA GPU and Docker GPU configuration.
- Secrets are read from `docker/.env`, which is git-ignored.
