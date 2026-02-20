"""Strategy model for Oculus platform."""
from datetime import datetime
from uuid import uuid4
from sqlalchemy import String, Integer, Float, DateTime, Text, JSON, ForeignKey, Index, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class Strategy(Base):
    """Strategy model for AI trading strategies."""
    
    __tablename__ = "strategies"
    
    # Primary key
    uuid: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    
    # Strategy info
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="draft")
    
    # Strategy details
    strategy_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    symbols: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    target_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    backtest_results: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    # Docker info
    docker_registry: Mapped[str | None] = mapped_column(String(255), nullable=True)
    docker_image_url: Mapped[str | None] = mapped_column(String(255), nullable=True)
    
    # Marketplace
    marketplace_listed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    marketplace_price: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Metrics
    version: Mapped[int] = mapped_column(Integer, default=1)
    subscriber_count: Mapped[int] = mapped_column(Integer, default=0)
    rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # Foreign key
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.uuid"), nullable=False)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user: Mapped["User"] = relationship("User", foreign_keys=[user_id])
    chat_history: Mapped[list["ChatHistory"]] = relationship("ChatHistory", back_populates="strategy")
    builds: Mapped[list["StrategyBuild"]] = relationship("StrategyBuild", back_populates="strategy")
    licenses: Mapped[list["License"]] = relationship("License", back_populates="strategy")
    ratings: Mapped[list["Rating"]] = relationship("Rating", back_populates="strategy")
    
    # Indexes
    __table_args__ = (
        Index("idx_strategy_user_id", "user_id"),
        Index("idx_strategy_status", "status"),
    )
    
    def __repr__(self) -> str:
        return f"<Strategy(uuid={self.uuid}, name={self.name}, user_id={self.user_id})>"

