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


class LicenseCheckoutResponse(BaseModel):
    """Schema returned when a Stripe checkout session is created for a license."""
    checkout_url: str
    session_id: str


class LicenseSetupIntentResponse(BaseModel):
    """Schema returned when a Stripe SetupIntent is created for in-app license payment."""
    client_secret: str
    publishable_key: str


class LicenseSubscribeRequest(BaseModel):
    """Schema for creating a subscription with a confirmed payment method."""
    license_type: str = Field(..., pattern="^(monthly|annual)$")
    payment_method_id: str = Field(..., min_length=1)


class LicensePriceResponse(BaseModel):
    """Calculated license prices for a strategy based on backtest performance."""
    monthly_price: int
    annual_price: int
    performance_score: float  # curved score in [0.0, 1.0]


class LicenseResponse(BaseModel):
    """Schema for license detail response."""
    uuid: str
    status: str
    license_type: str
    strategy_id: str
    user_id: str
    strategy_version: Optional[int] = None
    webhook_url: Optional[str] = None
    strategy_name: Optional[str] = None
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


class AtlasLicenseValidationResponse(BaseModel):
    """Schema for Atlas Trade AI license validation response.

    Returns all fields required by Atlas Trade AI to confirm a license is valid
    and to identify the strategy being traded.
    """
    is_active: bool
    license_id: str
    license_status: str
    user_uuid: str
    strategy_name: str
    strategy_description: Optional[str] = None
    strategy_version: Optional[int] = None
    backtest_results: Optional[dict] = None
    expires_at: datetime
    registry_url: Optional[str] = None
    docker_image_name: Optional[str] = None
    strategy_folder: Optional[str] = None

    class Config:
        from_attributes = True

