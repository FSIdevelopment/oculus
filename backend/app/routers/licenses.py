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
    LicenseResponse, LicenseListResponse, LicenseValidationResponse,
    LicenseCheckoutResponse, LicensePriceResponse,
    LicenseSetupIntentResponse, LicenseSubscribeRequest,
    AtlasLicenseValidationResponse
)
from app.services.pricing import calculate_license_price
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

    # Create license record with current strategy version
    license_obj = License(
        status="active",
        license_type=license_data.license_type,
        strategy_id=strategy_id,
        strategy_version=strategy.version,  # Store the current strategy version
        user_id=current_user.uuid,
        expires_at=expires_at
    )
    
    db.add(license_obj)
    await db.commit()
    await db.refresh(license_obj)
    
    return license_obj


@router.get("/api/strategies/{strategy_id}/license/price", response_model=LicensePriceResponse)
async def get_license_price(
    strategy_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Return the calculated monthly and annual license prices for a strategy.

    Prices are derived from the strategy's latest backtest results using a
    normalised, weighted, power-curve formula.  Strategies with no backtest
    data are returned at the minimum price.  No authentication required so
    the frontend can call this before the user decides to purchase.
    """
    result = await db.execute(select(Strategy).where(Strategy.uuid == strategy_id))
    strategy = result.scalar_one_or_none()

    if not strategy:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Strategy not found")

    if not strategy.backtest_results:
        # No backtest data — fall back to the floor price
        return LicensePriceResponse(
            monthly_price=settings.LICENSE_MONTHLY_MIN_PRICE,
            annual_price=settings.LICENSE_MONTHLY_MIN_PRICE * settings.LICENSE_ANNUAL_MULTIPLIER,
            performance_score=0.0,
        )

    monthly, annual, score = calculate_license_price(strategy.backtest_results)
    return LicensePriceResponse(
        monthly_price=monthly,
        annual_price=annual,
        performance_score=score,
    )


@router.post("/api/strategies/{strategy_id}/license/checkout", response_model=LicenseCheckoutResponse, status_code=status.HTTP_200_OK)
async def create_license_checkout(
    strategy_id: str,
    license_data: LicensePurchaseRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a Stripe Checkout Session to purchase a strategy license.

    - Validates the strategy exists
    - Creates or reuses a Stripe customer for the user
    - Creates a Stripe Checkout Session in subscription mode
    - Returns the hosted checkout URL to redirect the user to
    - The license record is created by the Stripe webhook after payment succeeds
    """
    # Verify strategy exists
    result = await db.execute(select(Strategy).where(Strategy.uuid == strategy_id))
    strategy = result.scalar_one_or_none()

    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Strategy not found"
        )

    # Look up strategy creator's Connect account for revenue routing
    creator_result = await db.execute(select(User).where(User.uuid == strategy.user_id))
    creator = creator_result.scalar_one_or_none()
    creator_connect_id = creator.stripe_connect_account_id if creator else None

    # Require Stripe to be configured
    if not settings.STRIPE_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Stripe is not configured on this server"
        )

    # Calculate dynamic price from strategy backtest performance
    if strategy.backtest_results:
        monthly_price, _, _ = calculate_license_price(strategy.backtest_results)
    else:
        monthly_price = settings.LICENSE_MONTHLY_MIN_PRICE

    if license_data.license_type == "monthly":
        unit_amount = int(monthly_price * 100)  # Stripe uses cents
        interval = "month"
        price_label = f"${monthly_price}/month"
    else:
        annual_total = monthly_price * settings.LICENSE_ANNUAL_MULTIPLIER
        unit_amount = int(annual_total * 100)
        interval = "year"
        price_label = f"${annual_total}/year"

    # Create Stripe customer if the user doesn't have one
    if not current_user.stripe_customer_id:
        try:
            customer = stripe.Customer.create(
                email=current_user.email,
                name=current_user.name
            )
            current_user.stripe_customer_id = customer.id
            await db.commit()
        except stripe.error.StripeError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to create Stripe customer: {str(e)}"
            )

    # Create Stripe Checkout Session using inline price_data (no pre-created Price ID needed)
    try:
        session = stripe.checkout.Session.create(
            customer=current_user.stripe_customer_id,
            mode="subscription",
            line_items=[{
                "price_data": {
                    "currency": "usd",
                    "product_data": {
                        "name": f"Oculus Strategy License — {strategy.name}",
                        "description": (
                            f"{license_data.license_type.capitalize()} license for "
                            f"the {strategy.name} strategy ({price_label})"
                        ),
                    },
                    "unit_amount": unit_amount,
                    "recurring": {"interval": interval},
                },
                "quantity": 1,
            }],
            metadata={
                "event_type": "license_purchase",
                "user_id": current_user.uuid,
                "strategy_id": strategy_id,
                "license_type": license_data.license_type,
            },
            success_url=(
                f"{settings.FRONTEND_URL}/dashboard/strategies"
                "?license_success=1&session_id={CHECKOUT_SESSION_ID}"
            ),
            cancel_url=f"{settings.FRONTEND_URL}/dashboard/strategies?license_cancelled=1",
            subscription_data={
                "metadata": {"strategy_id": strategy_id},
                **({"transfer_data": {"destination": creator_connect_id}, "application_fee_percent": 35} if creator_connect_id else {}),
            },
        )
    except stripe.error.StripeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create checkout session: {str(e)}"
        )

    return LicenseCheckoutResponse(
        checkout_url=session.url,
        session_id=session.id
    )


@router.post("/api/licenses/setup-intent", response_model=LicenseSetupIntentResponse, status_code=status.HTTP_200_OK)
async def create_license_setup_intent(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a Stripe SetupIntent so the frontend can collect a payment method
    in-app using Stripe Elements.  The confirmed payment method is then passed
    to POST /api/strategies/{id}/license/subscribe to create the subscription.
    """
    if not settings.STRIPE_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Stripe is not configured on this server"
        )

    # Create Stripe customer if the user doesn't have one
    if not current_user.stripe_customer_id:
        try:
            customer = stripe.Customer.create(
                email=current_user.email,
                name=current_user.name
            )
            current_user.stripe_customer_id = customer.id
            await db.commit()
        except stripe.error.StripeError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to create Stripe customer: {str(e)}"
            )

    try:
        setup_intent = stripe.SetupIntent.create(
            customer=current_user.stripe_customer_id,
            payment_method_types=["card"],
            usage="off_session",  # stored for recurring subscription billing
        )
    except stripe.error.StripeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create setup intent: {str(e)}"
        )

    return LicenseSetupIntentResponse(
        client_secret=setup_intent.client_secret,
        publishable_key=settings.STRIPE_PUBLISHABLE_KEY,
    )


@router.post("/api/strategies/{strategy_id}/license/subscribe", response_model=LicenseResponse, status_code=status.HTTP_201_CREATED)
async def create_license_subscription(
    strategy_id: str,
    subscribe_data: LicenseSubscribeRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a Stripe Subscription and issue a License using a payment method
    that was already confirmed via a SetupIntent.

    - Attaches the payment method to the Stripe customer as the default
    - Creates a Stripe Subscription with inline price_data (dynamic pricing)
    - Creates a License record immediately (no webhook dependency)
    """
    # Verify strategy exists
    result = await db.execute(select(Strategy).where(Strategy.uuid == strategy_id))
    strategy = result.scalar_one_or_none()

    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Strategy not found"
        )

    # Look up strategy creator's Connect account for revenue routing
    creator_result = await db.execute(select(User).where(User.uuid == strategy.user_id))
    creator = creator_result.scalar_one_or_none()
    creator_connect_id = creator.stripe_connect_account_id if creator else None

    if not current_user.stripe_customer_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User does not have a Stripe customer account"
        )

    # Calculate dynamic price from backtest performance
    if strategy.backtest_results:
        monthly_price, _, _ = calculate_license_price(strategy.backtest_results)
    else:
        monthly_price = settings.LICENSE_MONTHLY_MIN_PRICE

    if subscribe_data.license_type == "annual":
        unit_amount = int(monthly_price * settings.LICENSE_ANNUAL_MULTIPLIER * 100)
        interval = "year"
    else:
        unit_amount = int(monthly_price * 100)
        interval = "month"

    try:
        # Set as the customer's default payment method
        stripe.Customer.modify(
            current_user.stripe_customer_id,
            invoice_settings={"default_payment_method": subscribe_data.payment_method_id}
        )

        # stripe.Subscription.create does not support product_data inside price_data;
        # a Product must be created first and its ID passed as `product`.
        stripe_product = stripe.Product.create(
            name=f"Oculus Strategy License — {strategy.name}",
        )

        # Create the subscription referencing the product ID
        subscription = stripe.Subscription.create(
            customer=current_user.stripe_customer_id,
            items=[{
                "price_data": {
                    "currency": "usd",
                    "product": stripe_product.id,
                    "unit_amount": unit_amount,
                    "recurring": {"interval": interval},
                }
            }],
            default_payment_method=subscribe_data.payment_method_id,
            metadata={
                "event_type": "license_purchase",
                "user_id": current_user.uuid,
                "strategy_id": strategy_id,
                "license_type": subscribe_data.license_type,
            },
            **({"transfer_data": {"destination": creator_connect_id}} if creator_connect_id else {}),
            **({"application_fee_percent": 35} if creator_connect_id else {}),
        )
    except stripe.error.StripeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create subscription: {str(e)}"
        )

    # Calculate expiration and create license record immediately
    if subscribe_data.license_type == "annual":
        expires_at = datetime.utcnow() + timedelta(days=365)
    else:
        expires_at = datetime.utcnow() + timedelta(days=30)

    license_obj = License(
        status="active",
        license_type=subscribe_data.license_type,
        strategy_id=strategy_id,
        strategy_version=strategy.version,  # Store the current strategy version
        user_id=current_user.uuid,
        subscription_id=subscription.id,
        expires_at=expires_at,
    )
    db.add(license_obj)
    await db.commit()
    await db.refresh(license_obj)

    return license_obj


@router.get("/api/users/me/licenses", response_model=LicenseListResponse)
async def list_my_licenses(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """List all licenses belonging to the current user."""
    # Total count
    count_result = await db.execute(
        select(func.count(License.uuid)).where(License.user_id == current_user.uuid)
    )
    total = count_result.scalar_one()

    # Fetch licenses with strategy join
    result = await db.execute(
        select(License, Strategy.name.label("strategy_name"))
        .join(Strategy, License.strategy_id == Strategy.uuid, isouter=True)
        .where(License.user_id == current_user.uuid)
        .order_by(License.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    rows = result.all()

    items = []
    for license_obj, strategy_name in rows:
        item = LicenseResponse(
            uuid=license_obj.uuid,
            status=license_obj.status,
            license_type=license_obj.license_type,
            strategy_id=license_obj.strategy_id,
            user_id=license_obj.user_id,
            strategy_version=license_obj.strategy_version,
            webhook_url=license_obj.webhook_url,
            strategy_name=strategy_name,
            created_at=license_obj.created_at,
            expires_at=license_obj.expires_at,
        )
        items.append(item)

    return LicenseListResponse(items=items, total=total, skip=skip, limit=limit)


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



@router.delete("/api/licenses/{license_id}", response_model=LicenseResponse)
async def cancel_license(
    license_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Cancel a license subscription at period end.

    - Marks the Stripe subscription to cancel_at_period_end=True
    - Sets license status to 'cancelling'
    - The license remains active until expires_at
    """
    result = await db.execute(
        select(License).where(License.uuid == license_id)
    )
    license_obj = result.scalar_one_or_none()

    if not license_obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="License not found")

    if license_obj.user_id != current_user.uuid:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")

    if license_obj.status not in ("active",):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel a license with status '{license_obj.status}'"
        )

    # Cancel Stripe subscription at period end (if one exists)
    if license_obj.subscription_id:
        try:
            stripe.Subscription.modify(
                license_obj.subscription_id,
                cancel_at_period_end=True
            )
        except stripe.error.StripeError as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Stripe error: {str(e)}"
            )

    license_obj.status = "cancelling"
    await db.commit()
    await db.refresh(license_obj)

    # Fetch strategy_name for response
    strat_result = await db.execute(
        select(Strategy.name).where(Strategy.uuid == license_obj.strategy_id)
    )
    strategy_name = strat_result.scalar_one_or_none()

    return LicenseResponse(
        uuid=license_obj.uuid,
        status=license_obj.status,
        license_type=license_obj.license_type,
        strategy_id=license_obj.strategy_id,
        user_id=license_obj.user_id,
        strategy_version=license_obj.strategy_version,
        webhook_url=license_obj.webhook_url,
        strategy_name=strategy_name,
        created_at=license_obj.created_at,
        expires_at=license_obj.expires_at,
    )

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


@router.get("/api/licenses/{license_id}/atlas-validate", response_model=AtlasLicenseValidationResponse)
async def atlas_validate_license(
    license_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Validate a license for Atlas Trade AI.

    Called by Atlas Trade AI to confirm a license is active before allowing
    trading.  No user authentication is required — the license UUID itself
    serves as the credential.

    Returns:
        - is_active: True only when status is "active" AND the license has not expired
        - license_id: the license UUID
        - license_status: raw status field (e.g. "active", "expired", "cancelled")
        - user_uuid: UUID of the user who holds the license
        - strategy_name: display name of the licensed strategy
        - strategy_description: description of the strategy (may be None)
        - backtest_results: full backtest results JSON (may be None)
        - expires_at: UTC datetime the license expires
    """
    # Fetch license with related strategy in a single joined query
    result = await db.execute(
        select(License, Strategy)
        .join(Strategy, License.strategy_id == Strategy.uuid)
        .where(License.uuid == license_id)
    )
    row = result.one_or_none()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="License not found"
        )

    license_obj, strategy = row

    # A license is truly active only when both the status field says "active"
    # AND the expiry date has not yet passed.
    is_active = (
        license_obj.status == "active"
        and license_obj.expires_at > datetime.utcnow()
    )

    # Derive Docker image tag name from strategy name (same logic as DockerBuilder)
    import re as _re
    _tag_name = _re.sub(r"[^a-z0-9-]", "", strategy.name.lower().replace("_", "-").replace(" ", "-")).strip("-")[:128] or "strategy"
    _derived_registry_url = f"{settings.DO_REGISTRY_URL}/{_tag_name}:latest"
    # Prefer the stored docker_image_url (has exact version), fall back to derived :latest
    _registry_url = strategy.docker_image_url or _derived_registry_url

    return AtlasLicenseValidationResponse(
        is_active=is_active,
        license_id=license_obj.uuid,
        license_status=license_obj.status,
        user_uuid=license_obj.user_id,
        strategy_name=strategy.name,
        strategy_description=strategy.description,
        strategy_version=license_obj.strategy_version or strategy.version,  # Use license version or fall back to current strategy version
        backtest_results=strategy.backtest_results,
        expires_at=license_obj.expires_at,
        registry_url=_registry_url,
        docker_image_name=_registry_url,
        strategy_folder=_tag_name,
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

