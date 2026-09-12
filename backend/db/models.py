import enum
from datetime import datetime, timezone
from sqlalchemy import String, Enum
from sqlalchemy.orm import Mapped, mapped_column
from backend.db.database import Base


class ProcessingStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoProcessingLog(Base):
    __tablename__ = "video_processing_history"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    input_path: Mapped[str] = mapped_column(String, nullable=False)
    output_path: Mapped[str | None] = mapped_column(String, nullable=True, default=None)
    vehicles_count: Mapped[int] = mapped_column(default=0)
    
    status: Mapped[ProcessingStatus] = mapped_column(
        Enum(ProcessingStatus), 
        default=ProcessingStatus.PENDING, 
        nullable=False
    )
    
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )