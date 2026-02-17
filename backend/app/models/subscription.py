"""Subscription model for Oculus platform."""
from datetime import datetime
from uuid import uuid4
from sqlalchemy import String, DateTime, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class Subscription(Base):
    """Subscription model for strategy subscriptions."""
    
    __tablename__ = "subscriptions"
    
    # Primary key
    uuid: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    
    # Subscription info
    status: Mapped[str] = mapped_column(String(50), default="active")
    
    # Stripe info
    stripe_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    
    # Foreign keys
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.uuid"), nullable=False)
    strategy_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("strategies.uuid"), nullable=True)
    license_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("licenses.uuid"), nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user: Mapped["User"] = relationship("User", foreign_keys=[user_id])
    strategy: Mapped["Strategy | None"] = relationship("Strategy", foreign_keys=[strategy_id])
    license: Mapped["License | None"] = relationship("License", back_populates="subscriptions", foreign_keys=[license_id])
    
    # Indexes
    __table_args__ = (
        Index("idx_subscription_user_id", "user_id"),
        Index("idx_subscription_strategy_id", "strategy_id"),
        Index("idx_subscription_license_id", "license_id"),
        Index("idx_subscription_status", "status"),
    )
    
    def __repr__(self) -> str:
        return f"<Subscription(uuid={self.uuid}, user_id={self.user_id}, status={self.status})>"

