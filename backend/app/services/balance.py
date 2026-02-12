"""Service functions for atomic token balance operations."""
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.models.user import User
from app.models.balance import Balance


async def deduct_tokens(
    db: AsyncSession,
    user_id: str,
    amount: float,
    description: str
) -> Balance:
    """
    Atomically deduct tokens from user balance.
    
    Raises HTTPException 402 if insufficient balance.
    Uses SELECT FOR UPDATE to prevent race conditions.
    """
    # Get user with lock
    result = await db.execute(
        select(User).where(User.uuid == user_id).with_for_update()
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    if user.balance < amount:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Insufficient token balance"
        )
    
    # Deduct tokens
    user.balance -= amount
    
    # Create transaction record
    transaction = Balance(
        tokens=amount,
        transaction_type="deduction",
        description=description,
        user_id=user_id,
        status="completed"
    )
    
    db.add(transaction)
    await db.flush()
    
    return transaction


async def add_tokens(
    db: AsyncSession,
    user_id: str,
    amount: float,
    description: str
) -> Balance:
    """
    Atomically add tokens to user balance.
    
    Uses SELECT FOR UPDATE to prevent race conditions.
    """
    # Get user with lock
    result = await db.execute(
        select(User).where(User.uuid == user_id).with_for_update()
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Add tokens
    user.balance += amount
    
    # Create transaction record
    transaction = Balance(
        tokens=amount,
        transaction_type="purchase",
        description=description,
        user_id=user_id,
        status="completed"
    )
    
    db.add(transaction)
    await db.flush()
    
    return transaction

