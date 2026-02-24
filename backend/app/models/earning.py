"""Earning model for tracking creator payouts from strategy license payments."""
from datetime import datetime
from uuid import uuid4

from sqlalchemy import String, Integer, DateTime, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Earning(Base):
    """Records each successful strategy license invoice payment.

    Created by the invoice.payment_succeeded webhook handler.
    All amounts stored in cents (USD).
    """
    __tablename__ = "earnings"

    uuid: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    strategy_id: Mapped[str] = mapped_column(String(36), ForeignKey("strategies.uuid"), nullable=False)
    creator_user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.uuid"), nullable=False)
    amount_gross: Mapped[int] = mapped_column(Integer, nullable=False)
    amount_creator: Mapped[int] = mapped_column(Integer, nullable=False)
    amount_platform: Mapped[int] = mapped_column(Integer, nullable=False)
    stripe_invoice_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    stripe_subscription_id: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    strategy = relationship("Strategy", foreign_keys=[strategy_id])
    creator = relationship("User", foreign_keys=[creator_user_id])

    __table_args__ = (
        Index("idx_earning_creator_user_id", "creator_user_id"),
        Index("idx_earning_strategy_id", "strategy_id"),
    )

    def __repr__(self) -> str:
        return f"<Earning(uuid={self.uuid}, creator_user_id={self.creator_user_id}, strategy_id={self.strategy_id})>"
