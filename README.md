#  Car Detection Service (FastAPI + Gradio + YOLO)

Микросервисное веб-приложение для автоматической детекции автомобилей на изображениях и видео. Проект развернут в изолированных Docker-контейнерах и готов к локальному или серверному запуску.

## Технологический стек
Backend: Python 3.10, FastAPI, Uvicorn, OpenCV, PyTorch, Ultralytics YOLO

Frontend: Gradio

Infrastructure: Docker, Docker Compose, WSL 2

## Архитектура проекта
Приложение состоит из двух независимых сервисов, взаимодействующих внутри внутренней сети Docker (ml_car_detection_service_default):

car_backend (Port 8000): FastAPI REST API. Принимает медиафайлы, выполняет инференс нейросетевой модели YOLO, возвращает разметку и метаданные детекции.

car_frontend (Port 7860): Пользовательский веб-интерфейс на Gradio. Позволяет загружать изображения/видео через браузер и просматривать результаты детекции.


## Быстрый запуск (Docker Compose)
Предварительные требования
Установленный Docker Desktop (с включенной поддержкой WSL 2 на Windows).

Свободный диск с объемом от 10 ГБ.

1. Клонирование репозитория
Bash
git clone [https://github.com/Mori48/ML_car_detection_service.git](https://github.com/Mori48/ML_car_detection_service.git)
cd ML_car_detection_service
2. Сборка и запуск контейнеров
Запустите проект в фоновом (detached) режиме:

    Bash
docker-compose up -d --build

3. Проверка статуса
Убедитесь, что оба контейнера находятся в статусе Up:

    Bash
docker ps

## Доступ к сервисам
После успешного запуска сервисы доступны по следующим адресам:

Веб-интерфейс (Gradio UI): [http://localhost:7860](http://localhost:7860)

Интерактивная документация API (Swagger UI): [http://localhost:8000/docs](http://localhost:8000/docs)

Альтернативная документация API (ReDoc): [http://localhost:8000/redoc](http://localhost:8000/redoc)