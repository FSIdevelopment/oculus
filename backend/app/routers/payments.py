"""Payments router for Stripe payment processing."""
import stripe
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timedelta

from app.database import get_db
from app.config import settings
from app.models.user import User
from app.models.product import Product
from app.models.purchase import Purchase
from app.models.license import License
from app.auth.dependencies import get_current_active_user
from app.services.balance import add_tokens
from app.schemas.payments import (
    PaymentIntentRequest, PaymentIntentResponse,
    PaymentConfirmRequest, PaymentConfirmResponse
)

router = APIRouter()

# Configure Stripe
stripe.api_key = settings.STRIPE_SECRET_KEY


@router.post("/api/payments/create-intent", response_model=PaymentIntentResponse)
async def create_payment_intent(
    request_data: PaymentIntentRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a Stripe PaymentIntent for a token product.
    
    - Validates product exists and is a token package
    - Creates PaymentIntent with idempotency key
    - Returns client secret + publishable key for frontend
    """
    # Fetch product
    result = await db.execute(
        select(Product).where(Product.uuid == request_data.product_id)
    )
    product = result.scalar_one_or_none()
    
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product not found"
        )
    
    if product.product_type != "token_package":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only token products can be purchased"
        )
    
    # Ensure user has Stripe customer ID
    if not current_user.stripe_customer_id:
        try:
            customer = stripe.Customer.create(
                email=current_user.email,
                name=current_user.name
            )
            current_user.stripe_customer_id = customer.id
            await db.flush()
        except stripe.error.StripeError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to create Stripe customer: {str(e)}"
            )
    
    # Create PaymentIntent
    try:
        amount_cents = int(product.price * 100)
        idempotency_key = request_data.idempotency_key or f"{current_user.uuid}_{product.uuid}_{datetime.utcnow().timestamp()}"
        
        payment_intent = stripe.PaymentIntent.create(
            amount=amount_cents,
            currency="usd",
            customer=current_user.stripe_customer_id,
            metadata={
                "user_id": current_user.uuid,
                "product_id": product.uuid,
                "product_name": product.name
            },
            idempotency_key=idempotency_key
        )
    except stripe.error.StripeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create payment intent: {str(e)}"
        )
    
    return PaymentIntentResponse(
        client_secret=payment_intent.client_secret,
        publishable_key=settings.STRIPE_PUBLISHABLE_KEY,
        amount=amount_cents,
        currency="usd",
        product_id=product.uuid,
        product_name=product.name
    )


@router.post("/api/payments/confirm", response_model=PaymentConfirmResponse)
async def confirm_payment(
    request_data: PaymentConfirmRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Confirm payment and add tokens to user balance.
    
    - Verifies PaymentIntent succeeded
    - Checks for duplicate purchases (idempotency)
    - Adds tokens to user balance
    - Creates Purchase record
    """
    # Fetch product
    result = await db.execute(
        select(Product).where(Product.uuid == request_data.product_id)
    )
    product = result.scalar_one_or_none()
    
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product not found"
        )
    
    # Verify PaymentIntent
    try:
        payment_intent = stripe.PaymentIntent.retrieve(request_data.payment_intent_id)
    except stripe.error.StripeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to retrieve payment intent: {str(e)}"
        )
    
    if payment_intent.status != "succeeded":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Payment not succeeded. Status: {payment_intent.status}"
        )
    
    # Check for duplicate purchase (idempotency)
    result = await db.execute(
        select(Purchase).where(
            Purchase.user_id == current_user.uuid,
            Purchase.product_id == product.uuid,
            Purchase.stripe_id == request_data.payment_intent_id
        )
    )
    existing_purchase = result.scalar_one_or_none()
    
    if existing_purchase:
        # Return success with existing balance
        return PaymentConfirmResponse(
            success=True,
            message="Payment already processed",
            tokens_added=product.token_amount,
            new_balance=current_user.balance
        )
    
    # Add tokens to user balance
    try:
        await add_tokens(
            db,
            current_user.uuid,
            product.token_amount,
            f"Token purchase: {product.name}"
        )
    except HTTPException:
        raise
    
    # Create Purchase record
    purchase = Purchase(
        user_id=current_user.uuid,
        product_id=product.uuid,
        product_type=product.product_type,
        stripe_id=request_data.payment_intent_id,
        purchased_at=datetime.utcnow()
    )
    db.add(purchase)
    await db.commit()
    
    return PaymentConfirmResponse(
        success=True,
        message="Payment confirmed and tokens added",
        tokens_added=product.token_amount,
        new_balance=current_user.balance
    )


@router.post("/api/webhooks/stripe")
async def stripe_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Handle Stripe webhook events.
    
    - Verifies webhook signature
    - Processes payment.intent.succeeded events
    - Adds tokens to user balance
    """
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    
    if not sig_header:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing stripe-signature header"
        )
    
    # Verify webhook signature
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, settings.STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid payload"
        )
    except stripe.error.SignatureVerificationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid signature"
        )
    
    # Handle checkout.session.completed for license subscription purchases
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        metadata = session.get("metadata", {})

        if metadata.get("event_type") == "license_purchase":
            user_id = metadata.get("user_id")
            strategy_id = metadata.get("strategy_id")
            license_type = metadata.get("license_type", "monthly")
            subscription_id = session.get("subscription")

            if user_id and strategy_id:
                # Prevent duplicate license creation for the same session
                existing_result = await db.execute(
                    select(License).where(License.subscription_id == session["id"])
                )
                if existing_result.scalar_one_or_none():
                    return {"status": "already_processed"}

                # Calculate expiration
                if license_type == "annual":
                    expires_at = datetime.utcnow() + timedelta(days=365)
                else:
                    expires_at = datetime.utcnow() + timedelta(days=30)

                license_obj = License(
                    status="active",
                    license_type=license_type,
                    strategy_id=strategy_id,
                    user_id=user_id,
                    subscription_id=session["id"],
                    expires_at=expires_at,
                )
                db.add(license_obj)
                try:
                    await db.commit()
                    return {"status": "license_created"}
                except Exception as e:
                    await db.rollback()
                    return {"status": "error", "message": str(e)}

        return {"status": "ignored"}

    # Handle payment.intent.succeeded event
    if event["type"] == "payment_intent.succeeded":
        payment_intent = event["data"]["object"]
        user_id = payment_intent["metadata"].get("user_id")
        product_id = payment_intent["metadata"].get("product_id")
        
        if not user_id or not product_id:
            return {"status": "ignored"}
        
        # Check for duplicate
        result = await db.execute(
            select(Purchase).where(
                Purchase.user_id == user_id,
                Purchase.product_id == product_id,
                Purchase.stripe_id == payment_intent["id"]
            )
        )
        existing = result.scalar_one_or_none()
        
        if existing:
            return {"status": "already_processed"}
        
        # Fetch product
        result = await db.execute(
            select(Product).where(Product.uuid == product_id)
        )
        product = result.scalar_one_or_none()
        
        if product:
            try:
                await add_tokens(
                    db,
                    user_id,
                    product.token_amount,
                    f"Token purchase via webhook: {product.name}"
                )
                
                # Create Purchase record
                purchase = Purchase(
                    user_id=user_id,
                    product_id=product_id,
                    product_type=product.product_type,
                    stripe_id=payment_intent["id"],
                    purchased_at=datetime.utcnow()
                )
                db.add(purchase)
                await db.commit()
                
                return {"status": "processed"}
            except Exception as e:
                await db.rollback()
                return {"status": "error", "message": str(e)}
    
    return {"status": "ignored"}

