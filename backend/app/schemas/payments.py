"""Schemas for payment endpoints."""
from typing import Optional
from pydantic import BaseModel, Field


class PaymentIntentRequest(BaseModel):
    """Request to create a Stripe PaymentIntent."""
    
    product_id: str = Field(..., description="UUID of the product to purchase")
    idempotency_key: Optional[str] = Field(None, description="Idempotency key for deduplication")


class PaymentIntentResponse(BaseModel):
    """Response with PaymentIntent details for frontend."""
    
    client_secret: str = Field(..., description="Stripe PaymentIntent client secret")
    publishable_key: str = Field(..., description="Stripe publishable key")
    amount: int = Field(..., description="Amount in cents")
    currency: str = Field(default="usd", description="Currency code")
    product_id: str = Field(..., description="Product UUID")
    product_name: str = Field(..., description="Product name")


class PaymentConfirmRequest(BaseModel):
    """Request to confirm a payment."""
    
    payment_intent_id: str = Field(..., description="Stripe PaymentIntent ID")
    product_id: str = Field(..., description="Product UUID")


class PaymentConfirmResponse(BaseModel):
    """Response after payment confirmation."""
    
    success: bool = Field(..., description="Whether payment was successful")
    message: str = Field(..., description="Status message")
    tokens_added: Optional[int] = Field(None, description="Number of tokens added")
    new_balance: Optional[float] = Field(None, description="User's new token balance")


class WebhookEventRequest(BaseModel):
    """Stripe webhook event payload."""
    
    id: str
    type: str
    data: dict
    created: int

