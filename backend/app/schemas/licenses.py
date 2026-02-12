"""Pydantic schemas for license endpoints."""
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


class LicensePurchaseRequest(BaseModel):
    """Schema for purchasing a license."""
    license_type: str = Field(..., pattern="^(monthly|annual)$")
    strategy_id: str = Field(..., min_length=1)


class LicenseRenewRequest(BaseModel):
    """Schema for renewing a license."""
    license_type: str = Field(..., pattern="^(monthly|annual)$")


class LicenseWebhookUpdate(BaseModel):
    """Schema for updating license webhook URL."""
    webhook_url: str = Field(..., min_length=1, max_length=255)


class LicenseResponse(BaseModel):
    """Schema for license detail response."""
    uuid: str
    status: str
    license_type: str
    strategy_id: str
    user_id: str
    webhook_url: Optional[str] = None
    created_at: datetime
    expires_at: datetime
    
    class Config:
        from_attributes = True


class LicenseListResponse(BaseModel):
    """Schema for paginated license list response."""
    items: list[LicenseResponse]
    total: int
    skip: int
    limit: int


class LicenseValidationResponse(BaseModel):
    """Schema for license validation response (returned to containers)."""
    valid: bool
    license_id: str
    user_id: str
    strategy_id: str
    expires_at: datetime
    webhook_url: Optional[str] = None
    # Data provider keys (if license is active)
    polygon_api_key: Optional[str] = None
    alphavantage_api_key: Optional[str] = None
    
    class Config:
        from_attributes = True

