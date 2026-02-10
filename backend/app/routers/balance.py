"""Balance and transaction history router."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func
from app.database import get_db
from app.models.balance import Balance
from app.auth.dependencies import get_current_active_user
from app.models.user import User
from app.schemas.balance import BalanceResponse, TransactionResponse, TransactionListResponse

router = APIRouter()


@router.get("/api/users/me/balance", response_model=BalanceResponse)
async def get_current_balance(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current token balance for the authenticated user."""
    return BalanceResponse(
        user_id=current_user.uuid,
        balance=current_user.balance
    )


@router.get("/api/users/me/balance/history", response_model=TransactionListResponse)
async def get_balance_history(
    page: int = 1,
    page_size: int = 50,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get transaction history for the authenticated user (paginated, newest first).
    """
    # Get total count
    count_result = await db.execute(
        select(func.count(Balance.uuid)).where(Balance.user_id == current_user.uuid)
    )
    total = count_result.scalar()
    
    # Get transactions
    offset = (page - 1) * page_size
    result = await db.execute(
        select(Balance)
        .where(Balance.user_id == current_user.uuid)
        .order_by(desc(Balance.created_at))
        .offset(offset)
        .limit(page_size)
    )
    transactions = result.scalars().all()
    
    total_pages = (total + page_size - 1) // page_size
    
    return TransactionListResponse(
        transactions=[
            TransactionResponse(
                uuid=t.uuid,
                user_id=t.user_id,
                tokens=t.tokens,
                transaction_type=t.transaction_type,
                description=t.description,
                created_at=t.created_at
            )
            for t in transactions
        ],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages
    )

