"""Pydantic schemas for subscription endpoints."""
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


class SubscriptionCreateRequest(BaseModel):
    """Schema for subscribing to a marketplace strategy."""
    strategy_id: str = Field(..., min_length=1)


class SubscriptionResponse(BaseModel):
    """Schema for subscription detail response."""
    uuid: str
    status: str
    user_id: str
    strategy_id: str
    stripe_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class SubscriptionListResponse(BaseModel):
    """Schema for paginated subscription list response."""
    items: list[SubscriptionResponse]
    total: int
    skip: int
    limit: int


class StrategyEarning(BaseModel):
    """Per-strategy earnings breakdown."""
    strategy_id: str
    strategy_name: str
    total_earned: float   # dollars
    payment_count: int


class EarningsResponse(BaseModel):
    """Schema for creator earnings dashboard."""
    total_earned: float         # lifetime earnings in dollars (from DB)
    balance_available: float    # available in Stripe Connect account (dollars)
    balance_pending: float      # pending in Stripe Connect account (dollars)
    by_strategy: list[StrategyEarning]

    class Config:
        from_attributes = True
