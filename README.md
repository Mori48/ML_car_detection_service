# Car Detection Service

End-to-end computer vision service for automatic car detection and real-time tracking in images and videos using YOLO.

The application provides:

- REST API for batch model inference
- WebSockets for real-time video streaming
- Gradio web interface for interactive usage
- PostgreSQL persistence for processing history and vehicle counts
- TensorRT-optimized inference for real-time processing

The project is containerized with Docker and consists of frontend, backend, and PostgreSQL database services communicating through an internal Docker network.

## Tech Stack

**Backend**

- Python 3.10
- FastAPI
- Uvicorn
- OpenCV
- PyTorch
- Ultralytics YOLO
- TensorRT
- WebSockets
- SQLAlchemy
- asyncpg

**Frontend**

- Gradio

**Database**

- PostgreSQL 15

**Infrastructure**

- Docker
- Docker Compose

## Architecture

The application consists of three independent services:

```text
┌─────────────────────┐
│       Browser       │
│   Gradio UI :7860   │
└──────────┬──────────┘
           │ HTTP / WS
           ▼
┌─────────────────────┐
│      Frontend       │
│       Gradio        │
└──────────┬──────────┘
           │ REST API / WS
           ▼
┌─────────────────────┐      ┌─────────────────────┐
│      Backend        │      │      Database       │
│ FastAPI + YOLO(TRT) │◄────►│     PostgreSQL      │
│ OpenCV + WebSockets │      │        :5432        │
└─────────────────────┘      └─────────────────────┘
```

### Backend

**`car_backend` — port 8000**

The FastAPI backend is responsible for:

- receiving image and video files;
- running optimized YOLO inference using PyTorch/TensorRT;
- handling real-time video streaming via WebSockets;
- performing object tracking;
- processing media with OpenCV;
- saving processing history, metadata, and vehicle counts to PostgreSQL.

### Frontend

**`car_frontend` — port 7860**

The Gradio web interface allows users to:

- upload images and videos;
- connect to live video streams;
- view real-time bounding boxes;
- view processed results and processing logs.

### Database

**`car_db` — port 5432**

PostgreSQL stores:

- processing history;
- processing metadata;
- detected vehicle counts.

Database data is persisted using Docker volumes and is therefore preserved across normal container restarts.

## Model Performance & Optimization

The YOLO model was trained for **15 epochs** and evaluated on the validation set.

| Metric | Score |
|:---|---:|
| Precision | 0.554 |
| Recall | 0.408 |
| mAP@50 | 0.427 |
| mAP@50:95 | 0.265 |

The reported metrics correspond to the validation set after 15 training epochs.

### TensorRT Optimization

For real-time inference, the trained YOLO model can be converted to TensorRT `.engine` format.

TensorRT is used to reduce inference overhead and improve throughput compared with running the model directly from PyTorch weights.

### Tracking

The service supports object tracking for video streams.

The default tracking configuration uses **ByteTrack**, selected for its balance between tracking stability and real-time performance.

Experiments were also performed with custom **BoT-SORT** configurations, including:

- extended track buffers;
- appearance-based ReID;
- custom matching thresholds;
- different tracking parameters for handling temporary vehicle occlusions.

For the current operational dataset, ByteTrack provided the best stability-to-speed ratio.

The experimental tracker configuration files are preserved in the repository to document the tracking optimization process.

## Project Structure

```text
ML_car_detection_service/
│
├── backend/
│   ├── db/
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

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Git](https://git-scm.com/)

On Windows, Docker Desktop should have **WSL 2 integration** enabled.

For TensorRT GPU acceleration, an NVIDIA GPU and the corresponding Docker NVIDIA runtime configuration are required.

### 1. Clone the repository

```bash
git clone https://github.com/Mori48/ML_car_detection_service.git
cd ML_car_detection_service
```

### 2. Build and start the services

```bash
docker compose -f docker/docker-compose.yml up -d --build
```

The first build may take some time because the backend image installs PyTorch and the required machine learning dependencies.

### 3. Check container status

```bash
docker compose -f docker/docker-compose.yml ps
```

The backend, frontend, and database containers should be running.

### 4. Open the application

**Gradio web interface**

```text
http://localhost:7860
```

**FastAPI Swagger documentation**

```text
http://localhost:8000/docs
```

**FastAPI ReDoc documentation**

```text
http://localhost:8000/redoc
```

## API

The backend exposes a REST API through FastAPI.

Interactive API documentation is available through Swagger UI:

```text
http://localhost:8000/docs
```

Swagger UI can be used to inspect available endpoints and send requests directly to the service.

FastAPI also provides ReDoc documentation:

```text
http://localhost:8000/redoc
```

## Useful Docker Commands

### View logs

View logs for all services:

```bash
docker compose -f docker/docker-compose.yml logs
```

View logs for a specific service:

```bash
docker compose -f docker/docker-compose.yml logs backend
docker compose -f docker/docker-compose.yml logs frontend
docker compose -f docker/docker-compose.yml logs db
```

Follow logs in real time:

```bash
docker compose -f docker/docker-compose.yml logs -f
```

### Stop the application

Stop all services while preserving database volumes:

```bash
docker compose -f docker/docker-compose.yml down
```

### Stop the application and remove database volumes

> Warning: this removes persisted database data.

```bash
docker compose -f docker/docker-compose.yml down -v
```

### Rebuild the application

Use this after changing application code or dependencies:

```bash
docker compose -f docker/docker-compose.yml up -d --build
```

### Restart the services

```bash
docker compose -f docker/docker-compose.yml restart
```

## Notes

- The application is designed as a containerized multi-service system.
- Frontend and backend communicate through the Docker Compose network.
- PostgreSQL data is persisted using Docker volumes.
- TensorRT acceleration requires a compatible NVIDIA GPU and Docker GPU configuration.
- Custom tracker configurations are retained in the repository as part of the project's experimental tracking work.
