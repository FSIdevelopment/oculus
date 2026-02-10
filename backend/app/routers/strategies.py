"""Strategy CRUD router for user endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime

from app.database import get_db
from app.models.user import User
from app.models.strategy import Strategy
from app.models.strategy_build import StrategyBuild
from app.schemas.strategies import (
    StrategyCreate, StrategyUpdate, StrategyResponse, 
    StrategyListResponse, BuildResponse
)
from app.auth.dependencies import get_current_active_user

router = APIRouter()


@router.post("", response_model=StrategyResponse, status_code=status.HTTP_201_CREATED)
async def create_strategy(
    strategy_data: StrategyCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new strategy record."""
    # Check for duplicate name for this user
    result = await db.execute(
        select(Strategy).where(
            Strategy.user_id == current_user.uuid,
            Strategy.name == strategy_data.name,
            Strategy.status != "deleted"
        )
    )
    existing = result.scalar_one_or_none()
    
    name = strategy_data.name
    if existing:
        # Append last 6 chars of user UUID
        suffix = current_user.uuid[-6:]
        name = f"{strategy_data.name}-{suffix}"
    
    # Create strategy - store config in symbols field
    new_strategy = Strategy(
        name=name,
        description=strategy_data.description,
        symbols=strategy_data.config,
        user_id=current_user.uuid,
        status="draft"
    )
    
    db.add(new_strategy)
    await db.commit()
    await db.refresh(new_strategy)
    
    return new_strategy


@router.get("", response_model=StrategyListResponse)
async def list_strategies(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """List current user's strategies (paginated)."""
    # Count total
    count_result = await db.execute(
        select(func.count(Strategy.uuid)).where(
            Strategy.user_id == current_user.uuid,
            Strategy.status != "deleted"
        )
    )
    total = count_result.scalar()
    
    # Fetch paginated
    result = await db.execute(
        select(Strategy)
        .where(
            Strategy.user_id == current_user.uuid,
            Strategy.status != "deleted"
        )
        .offset(skip)
        .limit(limit)
    )
    strategies = result.scalars().all()
    
    return {
        "items": strategies,
        "total": total,
        "skip": skip,
        "limit": limit
    }


@router.get("/{strategy_id}", response_model=StrategyResponse)
async def get_strategy(
    strategy_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get strategy detail."""
    result = await db.execute(
        select(Strategy).where(
            Strategy.uuid == strategy_id,
            Strategy.user_id == current_user.uuid,
            Strategy.status != "deleted"
        )
    )
    strategy = result.scalar_one_or_none()
    
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Strategy not found"
        )
    
    return strategy


@router.put("/{strategy_id}", response_model=StrategyResponse)
async def update_strategy(
    strategy_id: str,
    strategy_data: StrategyUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Update strategy metadata (owner only)."""
    result = await db.execute(
        select(Strategy).where(
            Strategy.uuid == strategy_id,
            Strategy.user_id == current_user.uuid,
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


@router.delete("/{strategy_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_strategy(
    strategy_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Soft-delete strategy (owner only)."""
    result = await db.execute(
        select(Strategy).where(
            Strategy.uuid == strategy_id,
            Strategy.user_id == current_user.uuid,
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


@router.get("/{strategy_id}/builds", response_model=dict)
async def get_strategy_builds(
    strategy_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get build history for a strategy (paginated)."""
    # Verify strategy ownership
    result = await db.execute(
        select(Strategy).where(
            Strategy.uuid == strategy_id,
            Strategy.user_id == current_user.uuid,
            Strategy.status != "deleted"
        )
    )
    strategy = result.scalar_one_or_none()
    
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Strategy not found"
        )
    
    # Count total builds
    count_result = await db.execute(
        select(func.count(StrategyBuild.uuid)).where(
            StrategyBuild.strategy_id == strategy_id
        )
    )
    total = count_result.scalar()
    
    # Fetch paginated builds
    builds_result = await db.execute(
        select(StrategyBuild)
        .where(StrategyBuild.strategy_id == strategy_id)
        .offset(skip)
        .limit(limit)
    )
    builds = builds_result.scalars().all()
    
    return {
        "items": [BuildResponse.model_validate(b) for b in builds],
        "total": total,
        "skip": skip,
        "limit": limit
    }

