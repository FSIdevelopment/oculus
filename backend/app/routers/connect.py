"""Stripe Connect onboarding router for marketplace payouts."""
import stripe
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.config import settings
from app.models.user import User
from app.auth.dependencies import get_current_active_user
from app.schemas.connect import (
    ConnectOnboardResponse, ConnectStatusResponse, ConnectDashboardLinkResponse
)

router = APIRouter()

# Configure Stripe
stripe.api_key = settings.STRIPE_SECRET_KEY


@router.post("/api/connect/onboard", response_model=ConnectOnboardResponse)
async def create_onboard_link(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a Stripe Express Account and Account Link for onboarding.
    
    - If user has no stripe_connect_account_id, creates a new Express account
    - Creates an Account Link for hosted onboarding
    - Returns the URL for frontend to redirect to
    """
    try:
        # Check if user already has a Connect account
        if not current_user.stripe_connect_account_id:
            # Create a new Stripe Express account
            account = stripe.Account.create(
                type="express",
                email=current_user.email,
                country="US"
            )
            current_user.stripe_connect_account_id = account.id
            await db.flush()
        
        # Create Account Link for onboarding
        account_link = stripe.AccountLink.create(
            account=current_user.stripe_connect_account_id,
            type="account_onboarding",
            return_url=f"{settings.FRONTEND_URL}/dashboard/account/connect/return",
            refresh_url=f"{settings.FRONTEND_URL}/dashboard/account?tab=earnings"
        )
        
        return ConnectOnboardResponse(url=account_link.url)
    
    except stripe.error.StripeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create onboarding link: {str(e)}"
        )


@router.get("/api/connect/status", response_model=ConnectStatusResponse)
async def get_connect_status(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get the current user's Stripe Connect onboarding status.
    
    - Returns "not_started" if no account exists
    - Returns "pending" if account exists but not fully onboarded
    - Returns "complete" if charges_enabled and details_submitted
    """
    try:
        if not current_user.stripe_connect_account_id:
            return ConnectStatusResponse(status="not_started", account_id=None)
        
        # Retrieve account details from Stripe
        account = stripe.Account.retrieve(current_user.stripe_connect_account_id)
        
        # Check if account is fully onboarded
        if account.charges_enabled and account.details_submitted:
            return ConnectStatusResponse(
                status="complete",
                account_id=current_user.stripe_connect_account_id
            )
        else:
            return ConnectStatusResponse(
                status="pending",
                account_id=current_user.stripe_connect_account_id
            )
    
    except stripe.error.StripeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to retrieve account status: {str(e)}"
        )


@router.get("/api/connect/dashboard-link", response_model=ConnectDashboardLinkResponse)
async def get_dashboard_link(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get a login link to the Stripe Express Dashboard.
    
    - Only works for fully onboarded accounts
    - Returns a URL that logs the user into their Stripe dashboard
    """
    try:
        if not current_user.stripe_connect_account_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No Stripe Connect account found"
            )
        
        # Create login link
        login_link = stripe.Account.create_login_link(
            current_user.stripe_connect_account_id
        )
        
        return ConnectDashboardLinkResponse(url=login_link.url)
    
    except stripe.error.StripeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create dashboard link: {str(e)}"
        )

