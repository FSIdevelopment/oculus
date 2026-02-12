"""Rating model for Oculus platform."""
from datetime import datetime
from uuid import uuid4
from sqlalchemy import String, Integer, DateTime, Text, ForeignKey, Index, UniqueConstraint, CheckConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class Rating(Base):
    """Rating model for strategy ratings and reviews."""
    
    __tablename__ = "ratings"
    
    # Primary key
    uuid: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    
    # Rating info
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    review_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Foreign keys
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.uuid"), nullable=False)
    strategy_id: Mapped[str] = mapped_column(String(36), ForeignKey("strategies.uuid"), nullable=False)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user: Mapped["User"] = relationship("User", foreign_keys=[user_id])
    strategy: Mapped["Strategy"] = relationship("Strategy", back_populates="ratings", foreign_keys=[strategy_id])
    
    # Constraints
    __table_args__ = (
        UniqueConstraint("user_id", "strategy_id", name="uq_rating_user_strategy"),
        CheckConstraint("rating >= 1 AND rating <= 5", name="ck_rating_range"),
        Index("idx_rating_user_id", "user_id"),
        Index("idx_rating_strategy_id", "strategy_id"),
    )
    
    def __repr__(self) -> str:
        return f"<Rating(uuid={self.uuid}, user_id={self.user_id}, strategy_id={self.strategy_id}, rating={self.rating})>"

