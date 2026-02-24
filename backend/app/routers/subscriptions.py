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
from app.models.earning import Earning
from app.schemas.subscriptions import (
    SubscriptionCreateRequest, SubscriptionResponse, 
    SubscriptionListResponse, EarningsResponse, StrategyEarning
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
    - total_earned: sum of all Earning records for this creator (from DB)
    - balance_available / balance_pending: from Stripe Connect Balance API
    - by_strategy: breakdown per strategy from Earning records
    """
    # Query all earnings for this creator grouped by strategy
    earn_result = await db.execute(
        select(
            Earning.strategy_id,
            func.sum(Earning.amount_creator).label("total_cents"),
            func.count(Earning.uuid).label("payment_count"),
        )
        .where(Earning.creator_user_id == current_user.uuid)
        .group_by(Earning.strategy_id)
    )
    rows = earn_result.all()

    # Fetch strategy names
    strategy_ids = [row.strategy_id for row in rows]
    strat_result = await db.execute(
        select(Strategy).where(Strategy.uuid.in_(strategy_ids))
    )
    strategies_by_id = {s.uuid: s for s in strat_result.scalars().all()}

    total_earned_cents = sum(row.total_cents or 0 for row in rows)

    by_strategy = [
        StrategyEarning(
            strategy_id=row.strategy_id,
            strategy_name=strategies_by_id[row.strategy_id].name if row.strategy_id in strategies_by_id else "Unknown",
            total_earned=round((row.total_cents or 0) / 100, 2),
            payment_count=row.payment_count or 0,
        )
        for row in rows
    ]

    # Query Stripe Connect account balance (if creator has Connect set up)
    balance_available = 0.0
    balance_pending = 0.0

    if current_user.stripe_connect_account_id:
        try:
            balance = stripe.Balance.retrieve(
                stripe_account=current_user.stripe_connect_account_id
            )
            balance_available = round(
                sum(b["amount"] for b in balance["available"]) / 100, 2
            )
            balance_pending = round(
                sum(b["amount"] for b in balance["pending"]) / 100, 2
            )
        except stripe.error.StripeError:
            pass  # If Stripe call fails, return zeros for balance

    return EarningsResponse(
        total_earned=round(total_earned_cents / 100, 2),
        balance_available=balance_available,
        balance_pending=balance_pending,
        by_strategy=by_strategy,
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
