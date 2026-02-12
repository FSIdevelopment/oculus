"""Schemas for Stripe Connect onboarding endpoints."""
from typing import Optional
from pydantic import BaseModel, Field


class ConnectOnboardResponse(BaseModel):
    """Response with Stripe Account Link URL for onboarding."""
    
    url: str = Field(..., description="Stripe Account Link URL for hosted onboarding")


class ConnectStatusResponse(BaseModel):
    """Response with Stripe Connect onboarding status."""
    
    status: str = Field(..., description="Onboarding status: not_started | pending | complete")
    account_id: Optional[str] = Field(None, description="Stripe Connect account ID if exists")


class ConnectDashboardLinkResponse(BaseModel):
    """Response with Stripe Express Dashboard login link."""
    
    url: str = Field(..., description="Stripe Express Dashboard login URL")

