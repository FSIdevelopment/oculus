"""Schemas for balance and transaction endpoints."""
from datetime import datetime
from typing import List
from pydantic import BaseModel, Field


class TransactionResponse(BaseModel):
    """Schema for a single transaction."""
    
    uuid: str
    user_id: str
    tokens: float
    transaction_type: str
    description: str | None
    created_at: datetime
    
    class Config:
        from_attributes = True


class TransactionListResponse(BaseModel):
    """Schema for paginated transaction list."""
    
    transactions: List[TransactionResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class BalanceResponse(BaseModel):
    """Schema for current balance response."""
    
    user_id: str
    balance: float
    
    class Config:
        from_attributes = True

