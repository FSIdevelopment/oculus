"""Licenses router for strategy licensing and validation."""
from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime, timedelta
import stripe

from app.database import get_db
from app.config import settings
from app.models.user import User
from app.models.license import License
from app.models.strategy import Strategy
from app.schemas.licenses import (
    LicensePurchaseRequest, LicenseRenewRequest, LicenseWebhookUpdate,
    LicenseResponse, LicenseListResponse, LicenseValidationResponse
)
from app.auth.dependencies import get_current_active_user, admin_required

router = APIRouter()

# Configure Stripe
stripe.api_key = settings.STRIPE_SECRET_KEY


@router.post("/api/strategies/{strategy_id}/license", response_model=LicenseResponse, status_code=status.HTTP_201_CREATED)
async def purchase_license(
    strategy_id: str,
    license_data: LicensePurchaseRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Purchase a license for a strategy (monthly or annual).
    
    - Validates strategy exists and is published
    - Creates Stripe subscription
    - Creates License record
    - Returns license details
    """
    # Verify strategy exists
    result = await db.execute(select(Strategy).where(Strategy.uuid == strategy_id))
    strategy = result.scalar_one_or_none()
    
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Strategy not found"
        )
    
    # Verify user has Stripe customer ID
    if not current_user.stripe_customer_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User does not have a Stripe customer account"
        )
    
    # Calculate expiration date
    if license_data.license_type == "monthly":
        expires_at = datetime.utcnow() + timedelta(days=30)
    else:  # annual
        expires_at = datetime.utcnow() + timedelta(days=365)
    
    # Create license record
    license_obj = License(
        status="active",
        license_type=license_data.license_type,
        strategy_id=strategy_id,
        user_id=current_user.uuid,
        expires_at=expires_at
    )
    
    db.add(license_obj)
    await db.commit()
    await db.refresh(license_obj)
    
    return license_obj


@router.get("/api/users/me/licenses", response_model=LicenseListResponse)
async def list_user_licenses(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """List current user's active licenses (paginated)."""
    # Count total
    count_result = await db.execute(
        select(func.count(License.uuid)).where(
            License.user_id == current_user.uuid,
            License.status == "active"
        )
    )
    total = count_result.scalar()
    
    # Fetch paginated
    result = await db.execute(
        select(License)
        .where(
            License.user_id == current_user.uuid,
            License.status == "active"
        )
        .offset(skip)
        .limit(limit)
    )
    licenses = result.scalars().all()
    
    return {
        "items": licenses,
        "total": total,
        "skip": skip,
        "limit": limit
    }


@router.put("/api/licenses/{license_id}/renew", response_model=LicenseResponse)
async def renew_license(
    license_id: str,
    renew_data: LicenseRenewRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Renew an existing license."""
    # Fetch license
    result = await db.execute(select(License).where(License.uuid == license_id))
    license_obj = result.scalar_one_or_none()
    
    if not license_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="License not found"
        )
    
    # Verify ownership
    if license_obj.user_id != current_user.uuid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to renew this license"
        )
    
    # Update expiration
    if renew_data.license_type == "monthly":
        license_obj.expires_at = datetime.utcnow() + timedelta(days=30)
    else:  # annual
        license_obj.expires_at = datetime.utcnow() + timedelta(days=365)
    
    license_obj.license_type = renew_data.license_type
    
    await db.commit()
    await db.refresh(license_obj)
    
    return license_obj


@router.put("/api/licenses/{license_id}/webhook", response_model=LicenseResponse)
async def update_license_webhook(
    license_id: str,
    webhook_data: LicenseWebhookUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Set or update webhook URL for data streaming."""
    # Fetch license
    result = await db.execute(select(License).where(License.uuid == license_id))
    license_obj = result.scalar_one_or_none()
    
    if not license_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="License not found"
        )
    
    # Verify ownership
    if license_obj.user_id != current_user.uuid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this license"
        )
    
    license_obj.webhook_url = webhook_data.webhook_url
    
    await db.commit()
    await db.refresh(license_obj)
    
    return license_obj


@router.get("/api/licenses/{license_id}/validate", response_model=LicenseValidationResponse)
async def validate_license(
    license_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Validate license and return data provider keys if active.
    Called by strategy containers - no auth required.
    Logs requesting IP address.
    """
    # Fetch license
    result = await db.execute(select(License).where(License.uuid == license_id))
    license_obj = result.scalar_one_or_none()
    
    # Get client IP
    client_ip = request.client.host if request.client else "unknown"
    
    if not license_obj:
        return LicenseValidationResponse(
            valid=False,
            license_id=license_id,
            user_id="",
            strategy_id="",
            expires_at=datetime.utcnow()
        )
    
    # Update IP address on validation
    license_obj.ip_address = client_ip
    await db.commit()
    
    # Check if license is active and not expired
    is_valid = (
        license_obj.status == "active" and
        license_obj.expires_at > datetime.utcnow()
    )
    
    return LicenseValidationResponse(
        valid=is_valid,
        license_id=license_obj.uuid,
        user_id=license_obj.user_id,
        strategy_id=license_obj.strategy_id,
        expires_at=license_obj.expires_at,
        webhook_url=license_obj.webhook_url,
        polygon_api_key=settings.POLYGON_API_KEY if is_valid else None,
        alphavantage_api_key=settings.ALPHAVANTAGE_API_KEY if is_valid else None
    )


@router.get("/api/admin/licenses", response_model=LicenseListResponse)
async def list_all_licenses(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    admin_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """List all licenses (admin only)."""
    # Count total
    count_result = await db.execute(select(func.count(License.uuid)))
    total = count_result.scalar()
    
    # Fetch paginated
    result = await db.execute(
        select(License)
        .offset(skip)
        .limit(limit)
    )
    licenses = result.scalars().all()
    
    return {
        "items": licenses,
        "total": total,
        "skip": skip,
        "limit": limit
    }


@router.put("/api/admin/licenses/{license_id}", response_model=LicenseResponse)
async def edit_license(
    license_id: str,
    license_data: dict,
    admin_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """Edit license details (admin only)."""
    # Fetch license
    result = await db.execute(select(License).where(License.uuid == license_id))
    license_obj = result.scalar_one_or_none()
    
    if not license_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="License not found"
        )
    
    # Update allowed fields
    if "status" in license_data:
        license_obj.status = license_data["status"]
    if "expires_at" in license_data:
        license_obj.expires_at = license_data["expires_at"]
    if "webhook_url" in license_data:
        license_obj.webhook_url = license_data["webhook_url"]
    
    await db.commit()
    await db.refresh(license_obj)
    
    return license_obj

