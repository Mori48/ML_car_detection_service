# Car Detection Service

End-to-end computer vision service for automatic car detection in images and videos using YOLO. The application provides a REST API for model inference and a Gradio web interface for interactive usage.

The project is containerized with Docker and consists of separate frontend and backend services communicating through an internal Docker network.

## Tech Stack

**Backend:** Python 3.10, FastAPI, Uvicorn, OpenCV, PyTorch, Ultralytics YOLO

**Frontend:** Gradio

**Infrastructure:** Docker, Docker Compose

## Architecture

The application consists of two independent services:

```text
┌─────────────────────┐
│      Browser        │
│   Gradio UI :7860   │
└──────────┬──────────┘
           │ HTTP
           ▼
┌─────────────────────┐
│      Frontend       │
│       Gradio        │
└──────────┬──────────┘
           │ REST API
           ▼
┌─────────────────────┐
│      Backend        │
│ FastAPI + YOLO      │
│ OpenCV + PyTorch    │
└─────────────────────┘
```

### Backend

**`car_backend` — port 8000**

FastAPI REST API responsible for:

* receiving image and video files;
* running YOLO inference;
* processing media with OpenCV;
* returning detection results and metadata.

### Frontend

**`car_frontend` — port 7860**

Gradio web interface that allows users to:

* upload images and videos;
* run car detection;
* view processed results.

## Model Performance

The YOLO model was trained for 15 epochs and evaluated on the validation set.

| Metric    | Score |
| --------- | ----: |
| Precision | 0.554 |
| Recall    | 0.408 |
| mAP@50    | 0.427 |
| mAP@50:95 | 0.265 |

The reported metrics correspond to the validation set after 15 training epochs.

## Project Structure

```text
ML_car_detection_service/
│
├── backend/
│   └── ...
│
├── frontend/
│   └── ...
│
├── data/
│   └── ...
│
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
│
├── requirements.txt
└── README.md
```

## Installation and Setup

### Prerequisites

Make sure you have installed:

* Docker Desktop
* Git

On Windows, Docker Desktop should have WSL 2 integration enabled.

### 1. Clone the repository

```bash
git clone https://github.com/Mori48/ML_car_detection_service.git
cd ML_car_detection_service
```

### 2. Build and start the services

```bash
docker compose -f docker/docker-compose.yml up -d --build
```

The first build may take some time because the backend image installs PyTorch and the required ML dependencies.

### 3. Check container status

```bash
docker compose -f docker/docker-compose.yml ps
```

Both services should have a running status.

### 4. Open the application

Gradio web interface:

```text
http://localhost:7860
```

FastAPI Swagger documentation:

```text
http://localhost:8000/docs
```

FastAPI ReDoc documentation:

```text
http://localhost:8000/redoc
```

## API

The backend exposes a REST API through FastAPI.

Interactive API documentation is available at:

```text
http://localhost:8000/docs
```

The Swagger UI can be used to inspect available endpoints and send requests directly to the service.

## Useful Docker Commands

### View logs

```bash
docker compose -f docker/docker-compose.yml logs
```

View logs for a specific service:

```bash
docker compose -f docker/docker-compose.yml logs backend
docker compose -f docker/docker-compose.yml logs frontend
```

Follow logs in real time:

```bash
docker compose -f docker/docker-compose.yml logs -f
```

### Stop the application

```bash
docker compose -f docker/docker-compose.yml down
```

### Rebuild the application

```bash
docker compose -f docker/docker-compose.yml up -d --build
```

### Restart the services

```bash
docker compose -f docker/docker-compose.yml restart
```

## Notes

The application currently runs inference on CPU. GPU acceleration can be configured separately depending on the target environment and Docker/NVIDIA runtime configuration.


