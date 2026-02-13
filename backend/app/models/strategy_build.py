"""StrategyBuild model for Oculus platform."""
from datetime import datetime
from uuid import uuid4
from sqlalchemy import String, Integer, Float, DateTime, Text, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class StrategyBuild(Base):
    """StrategyBuild model for tracking strategy build progress."""
    
    __tablename__ = "strategy_builds"
    
    # Primary key
    uuid: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    
    # Build info
    status: Mapped[str] = mapped_column(String(50), nullable=False)  # "queued", "building", "designing", "training", "extracting_rules", "optimizing", "building_docker", "complete", "failed"
    phase: Mapped[str | None] = mapped_column(String(100), nullable=True)
    logs: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Build metrics
    tokens_consumed: Mapped[float] = mapped_column(Float, default=0.0)
    iteration_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Foreign keys
    strategy_id: Mapped[str] = mapped_column(String(36), ForeignKey("strategies.uuid"), nullable=False)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.uuid"), nullable=False)
    
    # Timestamps
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    
    # Relationships
    strategy: Mapped["Strategy"] = relationship("Strategy", back_populates="builds", foreign_keys=[strategy_id])
    user: Mapped["User"] = relationship("User", foreign_keys=[user_id])
    iterations: Mapped[list["BuildIteration"]] = relationship(
        "BuildIteration", back_populates="build", foreign_keys="BuildIteration.build_id",
        order_by="BuildIteration.iteration_number"
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_build_strategy_id", "strategy_id"),
        Index("idx_build_user_id", "user_id"),
        Index("idx_build_status", "status"),
    )
    
    def __repr__(self) -> str:
        return f"<StrategyBuild(uuid={self.uuid}, strategy_id={self.strategy_id}, status={self.status})>"

