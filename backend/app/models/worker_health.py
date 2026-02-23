"""WorkerHealth model for tracking worker status and capacity."""
from datetime import datetime
from sqlalchemy import String, Integer, DateTime, Index
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base


class WorkerHealth(Base):
    """WorkerHealth model for tracking worker status and capacity."""
    
    __tablename__ = "worker_health"
    
    # Primary key
    worker_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    
    # Worker info
    hostname: Mapped[str] = mapped_column(String(255), nullable=False)
    capacity: Mapped[int] = mapped_column(Integer, nullable=False)  # Max concurrent jobs
    active_jobs: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    
    # Health tracking
    last_heartbeat: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="active")  # "active", "dead", "draining"
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index("idx_worker_status", "status"),
        Index("idx_worker_last_heartbeat", "last_heartbeat"),
    )
    
    def __repr__(self) -> str:
        return f"<WorkerHealth(worker_id={self.worker_id}, status={self.status}, active_jobs={self.active_jobs}/{self.capacity})>"

