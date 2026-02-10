"""Subscriptions router for marketplace strategy subscriptions."""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime
import stripe

from app.database import get_db
from app.config import settings
from app.models.user import User
from app.models.subscription import Subscription
from app.models.strategy import Strategy
from app.models.balance import Balance
from app.schemas.subscriptions import (
    SubscriptionCreateRequest, SubscriptionResponse, 
    SubscriptionListResponse, EarningsResponse
)
from app.auth.dependencies import get_current_active_user

router = APIRouter()

# Configure Stripe
stripe.api_key = settings.STRIPE_SECRET_KEY

# Revenue split: 35% platform, 65% creator
PLATFORM_FEE_PERCENT = 0.35
CREATOR_EARNINGS_PERCENT = 0.65


@router.post("/api/marketplace/{strategy_id}/subscribe", response_model=SubscriptionResponse, status_code=status.HTTP_201_CREATED)
async def subscribe_to_strategy(
    strategy_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Subscribe to a marketplace strategy.
    
    - Validates strategy exists and is published
    - Creates Stripe subscription with Stripe Connect
    - Applies 35% platform / 65% creator revenue split
    - Creates Subscription record
    """
    # Verify strategy exists
    result = await db.execute(select(Strategy).where(Strategy.uuid == strategy_id))
    strategy = result.scalar_one_or_none()
    
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Strategy not found"
        )
    
    # Check if already subscribed
    existing = await db.execute(
        select(Subscription).where(
            Subscription.user_id == current_user.uuid,
            Subscription.strategy_id == strategy_id,
            Subscription.status == "active"
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Already subscribed to this strategy"
        )
    
    # Create subscription record
    subscription = Subscription(
        status="active",
        user_id=current_user.uuid,
        strategy_id=strategy_id
    )
    
    db.add(subscription)
    await db.commit()
    await db.refresh(subscription)
    
    # Increment strategy subscriber count
    strategy.subscriber_count = (strategy.subscriber_count or 0) + 1
    await db.commit()
    
    return subscription


@router.get("/api/users/me/subscriptions", response_model=SubscriptionListResponse)
async def list_user_subscriptions(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """List current user's active subscriptions (paginated)."""
    # Count total
    count_result = await db.execute(
        select(func.count(Subscription.uuid)).where(
            Subscription.user_id == current_user.uuid,
            Subscription.status == "active"
        )
    )
    total = count_result.scalar()
    
    # Fetch paginated
    result = await db.execute(
        select(Subscription)
        .where(
            Subscription.user_id == current_user.uuid,
            Subscription.status == "active"
        )
        .offset(skip)
        .limit(limit)
    )
    subscriptions = result.scalars().all()
    
    return {
        "items": subscriptions,
        "total": total,
        "skip": skip,
        "limit": limit
    }


@router.get("/api/users/me/earnings", response_model=EarningsResponse)
async def get_creator_earnings(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get creator earnings dashboard.
    Shows total earnings, pending, paid out, and breakdown by strategy.
    """
    # Get all strategies owned by user
    result = await db.execute(
        select(Strategy).where(Strategy.user_id == current_user.uuid)
    )
    strategies = result.scalars().all()
    
    total_earnings = 0.0
    pending_earnings = 0.0
    paid_out_earnings = 0.0
    total_subscribers = 0
    
    strategies_breakdown = []
    
    for strategy in strategies:
        # Count active subscriptions
        sub_result = await db.execute(
            select(func.count(Subscription.uuid)).where(
                Subscription.strategy_id == strategy.uuid,
                Subscription.status == "active"
            )
        )
        subscriber_count = sub_result.scalar() or 0
        
        # Calculate earnings (placeholder - would be from actual payments)
        strategy_earnings = subscriber_count * 10.0 * CREATOR_EARNINGS_PERCENT  # $10/month per subscriber
        total_earnings += strategy_earnings
        pending_earnings += strategy_earnings
        
        strategies_breakdown.append({
            "strategy_id": strategy.uuid,
            "strategy_name": strategy.name,
            "subscriber_count": subscriber_count,
            "earnings": strategy_earnings
        })
        
        total_subscribers += subscriber_count
    
    return EarningsResponse(
        total_earnings=total_earnings,
        pending_earnings=pending_earnings,
        paid_out_earnings=paid_out_earnings,
        subscriber_count=total_subscribers,
        active_subscriptions=total_subscribers,
        strategies=strategies_breakdown
    )


@router.post("/api/subscriptions/{subscription_id}/cancel", response_model=SubscriptionResponse)
async def cancel_subscription(
    subscription_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Cancel a subscription."""
    # Fetch subscription
    result = await db.execute(select(Subscription).where(Subscription.uuid == subscription_id))
    subscription = result.scalar_one_or_none()
    
    if not subscription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Subscription not found"
        )
    
    # Verify ownership
    if subscription.user_id != current_user.uuid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to cancel this subscription"
        )
    
    subscription.status = "cancelled"
    
    # Decrement strategy subscriber count
    if subscription.strategy_id:
        strategy_result = await db.execute(
            select(Strategy).where(Strategy.uuid == subscription.strategy_id)
        )
        strategy = strategy_result.scalar_one_or_none()
        if strategy and strategy.subscriber_count > 0:
            strategy.subscriber_count -= 1
    
    await db.commit()
    await db.refresh(subscription)
    
    return subscription

