"""License model for Oculus platform."""
from datetime import datetime
from uuid import uuid4
from sqlalchemy import String, Integer, DateTime, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class License(Base):
    """License model for strategy licensing."""

    __tablename__ = "licenses"

    # Primary key
    uuid: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))

    # License info
    status: Mapped[str] = mapped_column(String(50), default="active")
    license_type: Mapped[str] = mapped_column(String(50), nullable=False)  # "monthly", "annual"
    strategy_version: Mapped[int | None] = mapped_column(Integer, nullable=True)  # Strategy version number for docker registry

    # License details
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    webhook_url: Mapped[str | None] = mapped_column(String(255), nullable=True)
    subscription_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    
    # Foreign keys
    strategy_id: Mapped[str] = mapped_column(String(36), ForeignKey("strategies.uuid"), nullable=False)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.uuid"), nullable=False)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    
    # Relationships
    strategy: Mapped["Strategy"] = relationship("Strategy", back_populates="licenses", foreign_keys=[strategy_id])
    user: Mapped["User"] = relationship("User", foreign_keys=[user_id])
    subscriptions: Mapped[list["Subscription"]] = relationship("Subscription", back_populates="license")
    
    # Indexes
    __table_args__ = (
        Index("idx_license_strategy_id", "strategy_id"),
        Index("idx_license_user_id", "user_id"),
        Index("idx_license_status", "status"),
    )
    
    def __repr__(self) -> str:
        return f"<License(uuid={self.uuid}, strategy_id={self.strategy_id}, user_id={self.user_id})>"

