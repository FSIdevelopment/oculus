"""Schemas for rating endpoints."""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class RatingCreate(BaseModel):
    """Schema for creating a rating."""
    
    score: int = Field(..., ge=1, le=5, description="Rating score 1-5")
    review: Optional[str] = Field(None, max_length=1000, description="Optional review text")


class RatingUpdate(BaseModel):
    """Schema for updating a rating."""
    
    score: Optional[int] = Field(None, ge=1, le=5)
    review: Optional[str] = Field(None, max_length=1000)


class RatingResponse(BaseModel):
    """Schema for rating response."""
    
    uuid: str
    user_id: str
    strategy_id: str
    rating: int
    review_text: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class RatingListResponse(BaseModel):
    """Schema for paginated rating list."""
    
    ratings: List[RatingResponse]
    total: int
    page: int
    page_size: int
    total_pages: int

