"""Earning model for tracking creator payouts from strategy license payments."""
from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Index
from sqlalchemy.orm import relationship

from app.database import Base


class Earning(Base):
    """Records each successful strategy license invoice payment.

    Created by the invoice.payment_succeeded webhook handler.
    All amounts stored in cents (USD).
    """
    __tablename__ = "earnings"

    uuid = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    strategy_id = Column(String(36), ForeignKey("strategies.uuid"), nullable=False)
    creator_user_id = Column(String(36), ForeignKey("users.uuid"), nullable=False)
    amount_gross = Column(Integer, nullable=False)    # cents — full invoice amount
    amount_creator = Column(Integer, nullable=False)  # cents — 65% share
    amount_platform = Column(Integer, nullable=False) # cents — 35% share
    stripe_invoice_id = Column(String(255), nullable=False, unique=True)
    stripe_subscription_id = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    strategy = relationship("Strategy", foreign_keys=[strategy_id])
    creator = relationship("User", foreign_keys=[creator_user_id])

    __table_args__ = (
        Index("ix_earnings_creator_user_id", "creator_user_id"),
        Index("ix_earnings_strategy_id", "strategy_id"),
    )
