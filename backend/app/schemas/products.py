"""Schemas for product endpoints."""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class ProductCreate(BaseModel):
    """Schema for creating a product."""
    
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    price_cents: int = Field(..., gt=0, description="Price in cents")
    product_type: str = Field(..., description="'token_package' or 'license'")
    token_amount: Optional[int] = Field(None, gt=0, description="Number of tokens (for token_package)")
    stripe_price_id: Optional[str] = Field(None, description="Stripe price ID")


class ProductUpdate(BaseModel):
    """Schema for updating a product."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    price_cents: Optional[int] = Field(None, gt=0)
    token_amount: Optional[int] = Field(None, gt=0)


class ProductResponse(BaseModel):
    """Schema for product response."""
    
    uuid: str
    name: str
    description: Optional[str]
    price: float
    product_type: str
    token_amount: Optional[int]
    stripe_product_id: Optional[str]
    stripe_price_id: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True

