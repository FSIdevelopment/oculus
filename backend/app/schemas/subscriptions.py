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


class EarningsResponse(BaseModel):
    """Schema for creator earnings dashboard."""
    total_earnings: float
    pending_earnings: float
    paid_out_earnings: float
    subscriber_count: int
    active_subscriptions: int
    strategies: list[dict]  # List of strategies with earnings breakdown
    
    class Config:
        from_attributes = True

