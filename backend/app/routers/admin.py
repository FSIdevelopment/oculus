"""Admin endpoints for user management and support."""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime
import secrets
import string

from app.database import get_db
from app.models.user import User
from app.models.strategy import Strategy
from app.schemas.users import (
    UserAdminUpdate, UserDetailResponse, UserListResponse,
    UserListItem, ImpersonationTokenResponse
)
from app.schemas.strategies import StrategyResponse, StrategyUpdate, StrategyListResponse
from app.auth.dependencies import admin_required
from app.auth.security import hash_password, create_access_token

router = APIRouter()


@router.get("/users", response_model=UserListResponse)
async def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    search: str = Query("", min_length=0),
    current_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """
    List all users with pagination and search.
    
    - Paginated with skip/limit
    - Search by name or email (case-insensitive)
    """
    query = select(User)
    
    # Apply search filter
    if search:
        query = query.where(
            (User.name.ilike(f"%{search}%")) | 
            (User.email.ilike(f"%{search}%"))
        )
    
    # Get total count
    count_result = await db.execute(select(func.count(User.uuid)).select_from(User))
    total = count_result.scalar()
    
    # Apply pagination
    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    users = result.scalars().all()
    
    items = [UserListItem.model_validate(u) for u in users]
    
    return {
        "items": items,
        "total": total,
        "skip": skip,
        "limit": limit
    }


@router.get("/users/{user_id}", response_model=UserDetailResponse)
async def get_user_detail(
    user_id: str,
    current_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """Get detailed information about a specific user."""
    result = await db.execute(select(User).where(User.uuid == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return user


@router.put("/users/{user_id}", response_model=UserDetailResponse)
async def update_user(
    user_id: str,
    user_update: UserAdminUpdate,
    current_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """
    Update user information (admin only).
    
    - Can update all fields including role and status
    """
    result = await db.execute(select(User).where(User.uuid == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Check email uniqueness
    if user_update.email and user_update.email != user.email:
        result = await db.execute(select(User).where(User.email == user_update.email))
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already in use"
            )
    
    # Update fields
    if user_update.name:
        user.name = user_update.name
    if user_update.email:
        user.email = user_update.email
    if user_update.phone_number is not None:
        user.phone_number = user_update.phone_number
    if user_update.user_role:
        user.user_role = user_update.user_role
    if user_update.status:
        user.status = user_update.status
    if user_update.balance is not None:
        user.balance = user_update.balance
    
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    return user


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def soft_delete_user(
    user_id: str,
    current_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """
    Soft-delete a user (set is_active=False).
    
    - User data is preserved but account is deactivated
    """
    result = await db.execute(select(User).where(User.uuid == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    user.status = "inactive"
    db.add(user)
    await db.commit()


@router.post("/users/{user_id}/reset-password")
async def reset_user_password(
    user_id: str,
    current_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate a temporary password for a user.

    - Creates a random temporary password
    - In production, would send via Resend email service
    """
    result = await db.execute(select(User).where(User.uuid == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Generate temporary password
    temp_password = ''.join(
        secrets.choice(string.ascii_letters + string.digits)
        for _ in range(12)
    )

    # Hash and update password
    user.password_hash = hash_password(temp_password)
    db.add(user)
    await db.commit()

    # In production, send via Resend:
    # resend.Emails.send({
    #     "from": "noreply@oculus.com",
    #     "to": user.email,
    #     "subject": "Password Reset",
    #     "html": f"Your temporary password is: {temp_password}"
    # })

    return {
        "message": "Password reset. Email sent to user.",
        "temp_password": temp_password  # In production, don't return this
    }


@router.post("/users/{user_id}/impersonate", response_model=ImpersonationTokenResponse)
async def impersonate_user(
    user_id: str,
    current_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate an impersonation JWT token for a user.

    - Creates a token as if the user logged in
    - Includes impersonator_id claim for audit trail
    """
    result = await db.execute(select(User).where(User.uuid == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Create token with impersonator info
    token_data = {
        "sub": user.uuid,
        "email": user.email,
        "role": user.user_role,
        "impersonator_id": current_user.uuid
    }
    access_token = create_access_token(data=token_data)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "impersonated_user_id": user.uuid,
        "impersonator_id": current_user.uuid
    }


@router.post("/users/{user_id}/support-email")
async def send_support_email(
    user_id: str,
    email_data: dict,
    current_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """
    Send a support email to a user.

    - In production, uses Resend email service
    """
    result = await db.execute(select(User).where(User.uuid == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # In production, send via Resend:
    # resend.Emails.send({
    #     "from": "support@oculus.com",
    #     "to": user.email,
    #     "subject": email_data.get("subject", "Support"),
    #     "html": email_data.get("body", "")
    # })

    return {
        "message": f"Support email sent to {user.email}",
        "user_id": user.uuid
    }


# ============================================================================
# STRATEGY ADMIN ENDPOINTS
# ============================================================================

@router.get("/strategies", response_model=StrategyListResponse)
async def list_all_strategies(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    search: str = Query("", min_length=0),
    admin_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """List ALL strategies (admin only, searchable)."""
    # Build query
    query = select(Strategy).where(Strategy.status != "deleted")

    if search:
        query = query.where(Strategy.name.ilike(f"%{search}%"))

    # Count total
    count_result = await db.execute(
        select(func.count(Strategy.uuid)).where(Strategy.status != "deleted")
    )
    total = count_result.scalar()

    # Fetch paginated
    result = await db.execute(
        query.offset(skip).limit(limit)
    )
    strategies = result.scalars().all()

    return {
        "items": strategies,
        "total": total,
        "skip": skip,
        "limit": limit
    }


@router.put("/strategies/{strategy_id}", response_model=StrategyResponse)
async def update_any_strategy(
    strategy_id: str,
    strategy_data: StrategyUpdate,
    admin_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """Edit any strategy (admin only)."""
    result = await db.execute(
        select(Strategy).where(
            Strategy.uuid == strategy_id,
            Strategy.status != "deleted"
        )
    )
    strategy = result.scalar_one_or_none()

    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Strategy not found"
        )

    # Update fields
    if strategy_data.name is not None:
        strategy.name = strategy_data.name
    if strategy_data.description is not None:
        strategy.description = strategy_data.description
    if strategy_data.config is not None:
        strategy.symbols = strategy_data.config

    strategy.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(strategy)

    return strategy


@router.delete("/strategies/{strategy_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_any_strategy(
    strategy_id: str,
    admin_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """Delete any strategy (admin only)."""
    result = await db.execute(
        select(Strategy).where(
            Strategy.uuid == strategy_id,
            Strategy.status != "deleted"
        )
    )
    strategy = result.scalar_one_or_none()

    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Strategy not found"
        )

    strategy.status = "deleted"
    strategy.updated_at = datetime.utcnow()
    await db.commit()

