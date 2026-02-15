"""Strategy CRUD router for user endpoints."""
from typing import Optional
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
    StrategyListResponse, BuildResponse, DockerInfoResponse
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
    
    # Extract strategy fields from config dict and store in proper columns
    config = strategy_data.config or {}
    strategy_type = config.get("strategy_type")
    target_return = config.get("target_return")
    symbols = config.get("symbols", [])

    # Ensure symbols is stored as a JSON list, not the full config dict
    if not isinstance(symbols, list):
        symbols = []

    new_strategy = Strategy(
        name=name,
        description=strategy_data.description,
        strategy_type=strategy_type,
        target_return=target_return,
        symbols=symbols,
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
    has_completed_build: Optional[bool] = Query(None),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """List current user's strategies (paginated)."""
    # Build base query conditions
    base_conditions = [
        Strategy.user_id == current_user.uuid,
        Strategy.status != "deleted"
    ]

    # Add has_completed_build filter if specified
    if has_completed_build is True:
        # Subquery to find strategies with at least one completed build
        completed_build_subquery = (
            select(StrategyBuild.strategy_id)
            .where(StrategyBuild.status == "complete")
            .distinct()
        )
        base_conditions.append(Strategy.uuid.in_(completed_build_subquery))

    # Count total
    count_result = await db.execute(
        select(func.count(Strategy.uuid)).where(*base_conditions)
    )
    total = count_result.scalar()

    # Fetch paginated
    result = await db.execute(
        select(Strategy)
        .where(*base_conditions)
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


@router.get("/marketplace", response_model=StrategyListResponse)
async def list_marketplace_strategies(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    strategy_type: str = Query("", min_length=0),
    sort_by: str = Query("created_at", regex="^(created_at|rating|subscriber_count|target_return)$"),
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
    db: AsyncSession = Depends(get_db)
):
    """List all published strategies for the marketplace (public, no auth required)."""
    # Build query for published strategies
    query = select(Strategy).where(
        Strategy.status == "complete",
        Strategy.docker_image_url.isnot(None)  # Only strategies with built Docker images
    )

    # Filter by strategy type if provided
    if strategy_type:
        query = query.where(Strategy.strategy_type == strategy_type)

    # Apply sorting
    if sort_by == "created_at":
        sort_column = Strategy.created_at
    elif sort_by == "rating":
        sort_column = Strategy.rating
    elif sort_by == "subscriber_count":
        sort_column = Strategy.subscriber_count
    else:  # target_return
        sort_column = Strategy.target_return

    if sort_order == "asc":
        query = query.order_by(sort_column.asc())
    else:
        query = query.order_by(sort_column.desc())

    # Count total (with strategy_type filter)
    count_query = select(func.count(Strategy.uuid)).where(
        Strategy.status == "complete",
        Strategy.docker_image_url.isnot(None)
    )
    if strategy_type:
        count_query = count_query.where(Strategy.strategy_type == strategy_type)
    count_result = await db.execute(count_query)
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


@router.get("/marketplace/{strategy_id}", response_model=StrategyResponse)
async def get_marketplace_strategy(strategy_id: str, db: AsyncSession = Depends(get_db)):
    """Get a single marketplace strategy (public, no auth)."""
    result = await db.execute(
        select(Strategy).where(
            Strategy.uuid == strategy_id,
            Strategy.status == "complete",
            Strategy.docker_image_url.isnot(None)
        )
    )
    strategy = result.scalar_one_or_none()
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return strategy


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


@router.get("/{strategy_id}/docker", response_model=DockerInfoResponse)
async def get_docker_info(
    strategy_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get Docker pull instructions and usage guide for a strategy."""
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

    if not strategy.docker_image_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Strategy Docker image not yet built"
        )

    image_url = strategy.docker_image_url
    pull_command = f"docker pull {image_url}"
    run_command = (
        f"docker run -e LICENSE_ID=<your_license_id> "
        f"-e WEBHOOK_URL=<your_webhook_url> {image_url}"
    )

    terms_of_use = """
This strategy container is proprietary and licensed for authorized use only.
By running this container, you agree to:
1. Use this strategy only for authorized trading purposes
2. Not reverse-engineer, modify, or redistribute this strategy
3. Maintain confidentiality of strategy logic and parameters
4. Comply with all applicable financial regulations
5. Accept all trading risks and losses

For full terms, visit: https://oculusalgorithms.com/terms
"""

    return {
        "image_url": image_url,
        "pull_command": pull_command,
        "run_command": run_command,
        "environment_variables": {
            "LICENSE_ID": "Your strategy license ID (required)",
            "WEBHOOK_URL": "URL to stream strategy output (optional)",
            "OCULUS_API_URL": "Oculus API endpoint (default: https://api.oculusalgorithms.com)",
        },
        "terms_of_use": terms_of_use,
    }

