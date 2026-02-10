"""Schemas for chat history endpoints."""
from datetime import datetime
from pydantic import BaseModel, Field


class ChatMessageCreate(BaseModel):
    """Schema for creating a chat message."""
    
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., min_length=1, description="Message content")


class ChatMessageResponse(BaseModel):
    """Schema for chat message response."""
    
    uuid: str
    strategy_id: str
    user_id: str
    role: str
    content: str
    created_at: datetime
    
    class Config:
        from_attributes = True

