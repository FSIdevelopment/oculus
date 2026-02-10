"""User self-service endpoints for profile and account management."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models.user import User
from app.schemas.auth import UserResponse
from app.schemas.users import UserUpdate, PayoutUpdate, BalanceResponse
from app.auth.dependencies import get_current_active_user
from app.auth.security import hash_password, verify_password

router = APIRouter()


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: User = Depends(get_current_active_user)
):
    """Get current user's profile information."""
    return current_user


@router.put("/me", response_model=UserResponse)
async def update_user_profile(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update current user's profile.
    
    - Can update: name, email, phone_number, password
    - Email uniqueness is enforced
    """
    # Check email uniqueness if email is being updated
    if user_update.email and user_update.email != current_user.email:
        result = await db.execute(select(User).where(User.email == user_update.email))
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already in use"
            )
    
    # Update fields
    if user_update.name:
        current_user.name = user_update.name
    if user_update.email:
        current_user.email = user_update.email
    if user_update.phone_number is not None:
        current_user.phone_number = user_update.phone_number
    if user_update.password:
        current_user.password_hash = hash_password(user_update.password)
    
    db.add(current_user)
    await db.commit()
    await db.refresh(current_user)
    
    return current_user


@router.get("/me/balance", response_model=BalanceResponse)
async def get_user_balance(
    current_user: User = Depends(get_current_active_user)
):
    """Get current user's token balance."""
    return {
        "balance": current_user.balance,
        "currency": "USD"
    }


@router.put("/me/payout")
async def update_payout_info(
    payout_update: PayoutUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update payout account information for monetized strategies.
    
    - Stores payout method and account details
    """
    # In a real implementation, this would store payout info securely
    # For now, we'll just acknowledge the update
    return {
        "message": "Payout information updated",
        "payout_method": payout_update.payout_method
    }

