"""Chat history router for strategy conversations."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from app.database import get_db
from app.models.chat_history import ChatHistory
from app.models.strategy import Strategy
from app.auth.dependencies import get_current_active_user
from app.models.user import User
from app.schemas.chat import ChatMessageCreate, ChatMessageResponse

router = APIRouter()


@router.post("/api/strategies/{strategy_id}/chat", response_model=ChatMessageResponse)
async def save_chat_message(
    strategy_id: str,
    message: ChatMessageCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Save a chat message for a strategy.
    Only the strategy owner can save messages.
    """
    # Verify strategy exists and user owns it
    result = await db.execute(
        select(Strategy).where(Strategy.uuid == strategy_id)
    )
    strategy = result.scalar_one_or_none()
    
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Strategy not found"
        )
    
    if strategy.user_id != current_user.uuid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only strategy owner can save messages"
        )
    
    # Create chat message
    chat_msg = ChatHistory(
        message=message.content,
        message_type=message.role,
        strategy_id=strategy_id,
        user_id=current_user.uuid
    )
    
    db.add(chat_msg)
    await db.commit()
    await db.refresh(chat_msg)
    
    return ChatMessageResponse(
        uuid=chat_msg.uuid,
        strategy_id=chat_msg.strategy_id,
        user_id=chat_msg.user_id,
        role=chat_msg.message_type,
        content=chat_msg.message,
        created_at=chat_msg.created_at,
        metadata=chat_msg.content
    )


@router.get("/api/strategies/{strategy_id}/chat", response_model=list[ChatMessageResponse])
async def get_chat_history(
    strategy_id: str,
    page: int = 1,
    page_size: int = 500,  # Increased from 50 to 500 to support builds with many iterations
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get chat history for a strategy (paginated, chronological order).
    Strategy owner or admin can access chat history.
    """
    # Verify strategy exists
    result = await db.execute(
        select(Strategy).where(Strategy.uuid == strategy_id)
    )
    strategy = result.scalar_one_or_none()

    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Strategy not found"
        )

    # Check if user owns the strategy or is an admin
    if strategy.user_id != current_user.uuid and current_user.user_role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only strategy owner or admin can access chat history"
        )

    # Get chat history
    offset = (page - 1) * page_size
    result = await db.execute(
        select(ChatHistory)
        .where(ChatHistory.strategy_id == strategy_id)
        .order_by(ChatHistory.created_at)
        .offset(offset)
        .limit(page_size)
    )
    messages = result.scalars().all()

    return [
        ChatMessageResponse(
            uuid=msg.uuid,
            strategy_id=msg.strategy_id,
            user_id=msg.user_id,
            role=msg.message_type,
            content=msg.message,
            created_at=msg.created_at,
            metadata=msg.content
        )
        for msg in messages
    ]

