"""Pydantic schemas for strategy endpoints."""
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field


class StrategyCreate(BaseModel):
    """Schema for creating a new strategy."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    config: dict = Field(default_factory=dict)


class StrategyUpdate(BaseModel):
    """Schema for updating strategy metadata."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    config: Optional[dict] = None


class BuildResponse(BaseModel):
    """Schema for strategy build details."""
    uuid: str
    status: str
    phase: Optional[str] = None
    tokens_consumed: float
    iteration_count: int
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class StrategyResponse(BaseModel):
    """Schema for strategy detail response."""
    uuid: str
    name: str
    description: Optional[str] = None
    status: str
    strategy_type: Optional[str] = None
    symbols: Optional[dict] = None
    target_return: Optional[float] = None
    backtest_results: Optional[dict] = None
    docker_registry: Optional[str] = None
    docker_image_url: Optional[str] = None
    version: int
    subscriber_count: int
    rating: Optional[float] = None
    user_id: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class StrategyListResponse(BaseModel):
    """Schema for paginated strategy list response."""
    items: List[StrategyResponse]
    total: int
    skip: int
    limit: int

