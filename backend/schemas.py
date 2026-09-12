from pydantic import BaseModel, Field
from datetime import datetime
from backend.db.models import ProcessingStatus


class VideoProcessResponse(BaseModel):
    id: int = Field(description="ID записи в БД")
    filename: str = Field(description="Имя файла")
    vehicles_count: int = Field(ge=0, description="Количество найденных авто")
    status: ProcessingStatus = Field(description="Статус обработки")
    output_path: str | None = Field(default=None, description="Путь к обработанному видео")

    class Config:
        from_attributes = True 

class VideoHistoryItem(BaseModel):
    id: int
    filename: str
    vehicles_count: int
    status: ProcessingStatus
    created_at: datetime

    class Config:
        from_attributes = True