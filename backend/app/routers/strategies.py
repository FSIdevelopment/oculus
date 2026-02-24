"""Strategy CRUD router for user endpoints."""
import asyncio
import json
import logging
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_
from datetime import datetime

from app.database import get_db
from app.models.user import User
from app.models.strategy import Strategy
from app.models.strategy_build import StrategyBuild
from app.models.build_iteration import BuildIteration
from app.schemas.strategies import (
    StrategyCreate, StrategyUpdate, StrategyResponse,
    StrategyListResponse, BuildResponse, StrategyInternalResponse,
    MarketplaceListRequest
)
from app.auth.dependencies import get_current_active_user
from app.config import settings

logger = logging.getLogger(__name__)

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
    search: str = Query("", min_length=0),
    strategy_type: str = Query("", min_length=0),
    status: str = Query("", min_length=0),
    sort_by: str = Query("created_at", regex="^(name|created_at|strategy_type|rating|subscriber_count)$"),
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """List current user's strategies (paginated, filterable, sortable)."""
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

    # Add search filter (name)
    if search:
        base_conditions.append(Strategy.name.ilike(f"%{search}%"))

    # Add strategy_type filter
    if strategy_type:
        base_conditions.append(Strategy.strategy_type == strategy_type)

    # Add status filter
    if status:
        base_conditions.append(Strategy.status == status)

    # Build query
    query = select(Strategy).where(*base_conditions)

    # Apply sorting
    if sort_by == "name":
        sort_column = Strategy.name
    elif sort_by == "strategy_type":
        sort_column = Strategy.strategy_type
    elif sort_by == "rating":
        sort_column = Strategy.rating
    elif sort_by == "subscriber_count":
        sort_column = Strategy.subscriber_count
    else:  # created_at
        sort_column = Strategy.created_at

    if sort_order == "asc":
        query = query.order_by(sort_column.asc())
    else:
        query = query.order_by(sort_column.desc())

    # Count total
    count_result = await db.execute(
        select(func.count(Strategy.uuid)).where(*base_conditions)
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
        Strategy.marketplace_listed == True  # Only explicitly listed strategies
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
        Strategy.marketplace_listed == True
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
            Strategy.marketplace_listed == True
        )
    )
    strategy = result.scalar_one_or_none()
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")

    # Attach latest complete build ID so the frontend can show the Strategy Explainer
    build_result = await db.execute(
        select(StrategyBuild.uuid)
        .where(StrategyBuild.strategy_id == strategy_id, StrategyBuild.status == "complete")
        .order_by(StrategyBuild.completed_at.desc())
        .limit(1)
    )
    latest_build_id = build_result.scalar_one_or_none()

    response = StrategyResponse.model_validate(strategy)
    response.latest_build_id = latest_build_id
    return response


@router.put("/{strategy_id}/marketplace", response_model=StrategyResponse)
async def update_marketplace_listing(
    strategy_id: str,
    listing_data: MarketplaceListRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List or unlist a strategy on the marketplace (owner only).
    Strategy must be complete and have a built Docker image to be listed.
    """
    result = await db.execute(
        select(Strategy).where(
            Strategy.uuid == strategy_id,
            Strategy.user_id == current_user.uuid,
            Strategy.status != "deleted"
        )
    )
    strategy = result.scalar_one_or_none()

    if not strategy:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Strategy not found")

    # Can only list if strategy build is complete
    if listing_data.listed and strategy.status != "complete":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Strategy must be complete before listing on marketplace"
        )

    strategy.marketplace_listed = listing_data.listed
    if listing_data.price is not None:
        strategy.marketplace_price = listing_data.price
    strategy.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(strategy)
    return strategy


@router.get("/{strategy_id}", response_model=StrategyResponse)
async def get_strategy(
    strategy_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get strategy detail. Admins can view any strategy; regular users can only view their own."""
    # Admins can view any strategy
    if current_user.user_role == "admin":
        result = await db.execute(
            select(Strategy).where(
                Strategy.uuid == strategy_id,
                Strategy.status != "deleted"
            )
        )
    else:
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
    """Get build history for a strategy (paginated). Admins can view any strategy's builds; regular users can only view their own."""
    # Verify strategy ownership (admins can view any strategy)
    if current_user.user_role == "admin":
        result = await db.execute(
            select(Strategy).where(
                Strategy.uuid == strategy_id,
                Strategy.status != "deleted"
            )
        )
    else:
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


@router.post("/{strategy_id}/retrain", response_model=BuildResponse)
async def retrain_strategy(
    strategy_id: str,
    target_return: float = Query(10.0, description="Target return percentage"),
    timeframe: str = Query("1d", description="Trading timeframe"),
    max_iterations: int = Query(5, ge=1, le=20, description="Max build iterations"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Trigger a retrain (new build) for an existing strategy.

    Increments the strategy version and launches a new build loop in the background.
    Returns BuildResponse immediately with status='queued'.
    Admins can retrain any strategy; regular users can only retrain their own.
    """
    import redis.asyncio as redis_async
    from app.routers.builds import _run_build_loop

    # Fetch and validate strategy (admins can retrain any strategy)
    if current_user.user_role == "admin":
        result = await db.execute(
            select(Strategy).where(
                Strategy.uuid == strategy_id,
                Strategy.status != "deleted",
            )
        )
    else:
        result = await db.execute(
            select(Strategy).where(
                Strategy.uuid == strategy_id,
                Strategy.user_id == current_user.uuid,
                Strategy.status != "deleted",
            )
        )
    strategy = result.scalar_one_or_none()
    if not strategy:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Strategy not found")

    # Determine which user's tokens to use
    # For admin retrains, use the strategy owner's tokens
    # For regular users, use their own tokens
    if current_user.user_role == "admin":
        # Fetch the strategy owner for token validation
        owner_result = await db.execute(
            select(User).where(User.uuid == strategy.user_id)
        )
        strategy_owner = owner_result.scalar_one_or_none()
        if not strategy_owner:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Strategy owner not found"
            )
        token_user = strategy_owner
    else:
        token_user = current_user

    # Validate token balance
    if token_user.balance < settings.TOKENS_PER_ITERATION:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"Insufficient tokens. Need at least {settings.TOKENS_PER_ITERATION}. Balance: {token_user.balance}",
        )

    # Increment strategy version to signal a new build is starting
    strategy.version = strategy.version + 1
    strategy.updated_at = datetime.utcnow()

    # Create a new build record
    # Use the strategy owner's user_id for the build, not the admin's
    build = StrategyBuild(
        strategy_id=strategy_id,
        user_id=strategy.user_id,
        status="queued",
        phase="initializing",
        started_at=datetime.utcnow(),
        max_iterations=max_iterations,
    )
    db.add(build)
    await db.commit()
    await db.refresh(build)

    build_id = build.uuid
    user_id = strategy.user_id  # Use strategy owner's ID for the build
    strategy_name = strategy.name

    # Clear any stale stop signal from previous builds
    try:
        redis_client = await redis_async.from_url(settings.REDIS_URL)
        await redis_client.delete(f"oculus:build:stop:{build_id}")
        await redis_client.close()
    except Exception as exc:
        logger.warning("Could not clear stop signal for retrain build %s: %s", build_id, exc)

    # Resolve symbols list (handle both list and dict formats)
    raw_symbols = strategy.symbols
    if isinstance(raw_symbols, dict):
        strategy_symbols = raw_symbols.get("symbols", [])
    elif isinstance(raw_symbols, str):
        # Handle case where symbols might be stored as space-separated string
        strategy_symbols = raw_symbols.split()
    elif isinstance(raw_symbols, list):
        strategy_symbols = raw_symbols
    else:
        strategy_symbols = []

    # Collect all completed iterations from all previous builds on this strategy
    # so the AI can learn from every prior attempt when designing the new build.
    # Builds are fetched in chronological order (oldest → newest via started_at) so
    # that the LLM always sees the full timeline: Build #1 (all its iters) → Build #2
    # (all its iters) → … rather than a timestamp-interleaved soup.
    prior_builds_result = await db.execute(
        select(StrategyBuild)
        .where(
            StrategyBuild.strategy_id == strategy_id,
            StrategyBuild.status == "complete",
        )
        .order_by(StrategyBuild.started_at)  # chronological build order, oldest first
    )
    prior_builds = prior_builds_result.scalars().all()

    prior_iteration_history = []
    for prior_build in prior_builds:
        # Fetch this build's iterations in iteration order so the LLM reads them
        # as a coherent sequence: iter 0 → iter 1 → iter 2 → …
        iter_result = await db.execute(
            select(BuildIteration)
            .where(
                BuildIteration.build_id == prior_build.uuid,
                BuildIteration.status == "complete",
            )
            .order_by(BuildIteration.iteration_number)
        )
        past_iterations = iter_result.scalars().all()

        for past_iter in past_iterations:
            bt = past_iter.backtest_results or {}
            entry_feats = [r.get("feature", "") for r in (past_iter.entry_rules or [])]
            exit_feats = [r.get("feature", "") for r in (past_iter.exit_rules or [])]
            feats = past_iter.features or {}
            top_feat_names = [
                f.get("name", f.get("feature", ""))
                for f in (feats.get("top_features", []) if isinstance(feats, dict) else [])[:5]
            ]
            prior_iteration_history.append({
                "iteration": past_iter.iteration_number,
                "build_id": str(prior_build.uuid),
                "total_return": bt.get("total_return", 0),
                "win_rate": bt.get("win_rate", 0),
                "sharpe_ratio": bt.get("sharpe_ratio", 0),
                "max_drawdown": bt.get("max_drawdown", 0),
                "total_trades": bt.get("total_trades", 0),
                "best_model": (past_iter.best_model or {}).get("name", "?"),
                "top_features": top_feat_names,
                "entry_features": entry_feats[:5],
                "exit_features": exit_feats[:5],
            })

    logger.info(
        "Retrain for strategy %s: found %d prior iterations from %d completed builds",
        strategy_id, len(prior_iteration_history), len(prior_builds),
    )

    # Launch the build loop as a background task
    asyncio.create_task(
        _run_build_loop(
            build_id=build_id,
            strategy_id=strategy_id,
            user_id=user_id,
            strategy_name=strategy_name,
            strategy_symbols=strategy_symbols,
            description=strategy.description or "",
            target_return=target_return,
            timeframe=timeframe,
            max_iterations=max_iterations,
            strategy_type=strategy.strategy_type,
            prior_iteration_history=prior_iteration_history if prior_iteration_history else None,
            customizations="",  # Not available for retrain
            thoughts="",  # Not available for retrain
        )
    )

    return BuildResponse(
        uuid=build.uuid,
        strategy_id=build.strategy_id,
        status=build.status,
        phase=build.phase,
        tokens_consumed=build.tokens_consumed,
        iteration_count=build.iteration_count,
        max_iterations=build.max_iterations,
        started_at=build.started_at,
        completed_at=build.completed_at,
        strategy_type=strategy.strategy_type,
        strategy_name=strategy.name,
    )


@router.get("/{strategy_id}/builds/{build_id}/backtest")
async def get_build_best_backtest(
    strategy_id: str,
    build_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Return the best iteration's backtest results for a specific completed build.

    'Best' is the completed iteration with the highest total_return_pct.
    Admins can view any strategy's backtest; regular users can only view their own.
    """
    # Verify strategy ownership (admins can view any strategy)
    if current_user.user_role == "admin":
        strat_result = await db.execute(
            select(Strategy).where(
                Strategy.uuid == strategy_id,
                Strategy.status != "deleted",
            )
        )
    else:
        strat_result = await db.execute(
            select(Strategy).where(
                Strategy.uuid == strategy_id,
                Strategy.user_id == current_user.uuid,
                Strategy.status != "deleted",
            )
        )
    if not strat_result.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Strategy not found")

    # Verify build belongs to this strategy
    build_result = await db.execute(
        select(StrategyBuild).where(
            StrategyBuild.uuid == build_id,
            StrategyBuild.strategy_id == strategy_id,
        )
    )
    build = build_result.scalar_one_or_none()
    if not build:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Build not found")

    # Fetch all completed iterations for this build
    iter_result = await db.execute(
        select(BuildIteration).where(
            BuildIteration.build_id == build_id,
            BuildIteration.status == "complete",
        )
    )
    iterations = iter_result.scalars().all()

    if not iterations:
        return {"backtest_results": None, "iteration_number": None}

    # Pick the iteration with the highest total_return_pct
    def _return_score(it: BuildIteration) -> float:
        br = it.backtest_results or {}
        return float(br.get("total_return_pct") or br.get("total_return") or 0)

    best = max(iterations, key=_return_score)
    return {"backtest_results": best.backtest_results, "iteration_number": best.iteration_number}


@router.get("/{strategy_id}/builds/{build_id}/config")
async def get_build_config(
    strategy_id: str,
    build_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Return the config.json generated for a specific build.

    Reads from the database (BuildIteration.strategy_files) instead of the filesystem
    to support production environments where files don't persist.
    Admins can view any strategy's config; regular users can only view their own.
    """
    # Verify strategy ownership (admins can view any strategy)
    if current_user.user_role == "admin":
        strat_result = await db.execute(
            select(Strategy).where(
                Strategy.uuid == strategy_id,
                Strategy.status != "deleted",
            )
        )
    else:
        strat_result = await db.execute(
            select(Strategy).where(
                Strategy.uuid == strategy_id,
                Strategy.user_id == current_user.uuid,
                Strategy.status != "deleted",
            )
        )
    if not strat_result.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Strategy not found")

    # Verify build belongs to this strategy
    build_result = await db.execute(
        select(StrategyBuild).where(
            StrategyBuild.uuid == build_id,
            StrategyBuild.strategy_id == strategy_id,
        )
    )
    build = build_result.scalar_one_or_none()
    if not build:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Build not found")

    # Get the best iteration for this build (or latest completed iteration)
    # Priority: 1) iteration with best backtest results, 2) latest completed iteration
    iteration_result = await db.execute(
        select(BuildIteration)
        .where(
            BuildIteration.build_id == build_id,
            BuildIteration.status == "complete",
        )
        .order_by(BuildIteration.iteration_number.desc())
        .limit(1)
    )
    iteration = iteration_result.scalar_one_or_none()

    if not iteration:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No completed iterations found for this build",
        )

    # Extract config.json from strategy_files
    if not iteration.strategy_files or "config.json" not in iteration.strategy_files:
        # Fallback to filesystem for local development
        config_path = Path(__file__).resolve().parent.parent.parent / "strategy_outputs" / build_id / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config_data = json.load(f)
                return {"config": config_data}
            except Exception as exc:
                logger.error("Failed to read config from filesystem for build %s: %s", build_id, exc)

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Config file not found for this build",
        )

    try:
        # Parse config.json from database
        config_json_str = iteration.strategy_files["config.json"]
        config_data = json.loads(config_json_str)
        return {"config": config_data}
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse config JSON for build %s: %s", build_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Invalid config JSON in database"
        )
    except Exception as exc:
        logger.error("Failed to read config for build %s: %s", build_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read config file"
        )


@router.get("/{strategy_id}/builds/{build_id}/readme")
async def get_build_readme(
    strategy_id: str,
    build_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Return the README.md generated for a specific build.

    The README is stored in the database for production compatibility.
    Admins can view any strategy's README; regular users can view their own
    or any marketplace-listed strategy's README.
    """
    # Admins can view any strategy; owners can view their own; any authenticated
    # user can view the readme of a marketplace-listed strategy.
    if current_user.user_role == "admin":
        strat_result = await db.execute(
            select(Strategy).where(
                Strategy.uuid == strategy_id,
                Strategy.status != "deleted",
            )
        )
    else:
        strat_result = await db.execute(
            select(Strategy).where(
                Strategy.uuid == strategy_id,
                Strategy.status != "deleted",
                or_(
                    Strategy.user_id == current_user.uuid,
                    Strategy.marketplace_listed == True,
                ),
            )
        )
    if not strat_result.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Strategy not found")

    # Fetch build and verify it belongs to this strategy
    build_result = await db.execute(
        select(StrategyBuild).where(
            StrategyBuild.uuid == build_id,
            StrategyBuild.strategy_id == strategy_id,
        )
    )
    build = build_result.scalar_one_or_none()

    if not build:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Build not found")

    # Return README from database
    if not build.readme:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="README not found for this build",
        )

    return {"readme": build.readme}


@router.get("/{strategy_id}/internal", response_model=StrategyInternalResponse)
async def get_strategy_internal(
    strategy_id: str,
    api_key: str = Query(..., description="Internal API key for SignalSynk"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get strategy with Docker info for internal use by SignalSynk.
    This endpoint is for internal system use only and requires an API key.
    """
    # TODO: Validate API key against a secure internal API key
    # For now, this is a placeholder - you should implement proper API key validation
    from app.config import settings
    from app.models.build_iteration import BuildIteration
    import json

    # You'll need to add SIGNALSYNK_API_KEY to your config
    # if api_key != settings.SIGNALSYNK_API_KEY:
    #     raise HTTPException(
    #         status_code=status.HTTP_401_UNAUTHORIZED,
    #         detail="Invalid API key"
    #     )

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

    if not strategy.docker_image_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Strategy Docker image not yet built"
        )

    # Get the latest successful build iteration to extract config.json and full backtest results
    config_data = None
    full_backtest_results = None

    build_result = await db.execute(
        select(StrategyBuild)
        .where(
            StrategyBuild.strategy_id == strategy_id,
            StrategyBuild.status == "complete"
        )
        .order_by(StrategyBuild.completed_at.desc())
        .limit(1)
    )
    latest_build = build_result.scalar_one_or_none()

    if latest_build:
        # Get the latest successful iteration from this build
        iteration_result = await db.execute(
            select(BuildIteration)
            .where(
                BuildIteration.build_id == latest_build.uuid,
                BuildIteration.status == "complete"
            )
            .order_by(BuildIteration.iteration_number.desc())
            .limit(1)
        )
        latest_iteration = iteration_result.scalar_one_or_none()

        if latest_iteration:
            # Extract config.json from strategy_files
            if latest_iteration.strategy_files:
                config_json_str = latest_iteration.strategy_files.get("config.json")
                if config_json_str:
                    try:
                        config_data = json.loads(config_json_str)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse config.json for strategy {strategy_id}")

            # Get FULL backtest results from BuildIteration (includes trades, equity_curve, etc.)
            if latest_iteration.backtest_results:
                full_backtest_results = latest_iteration.backtest_results
            else:
                # Fallback to strategy.backtest_results if iteration doesn't have it
                full_backtest_results = strategy.backtest_results

    # Create response with config and full backtest results
    return StrategyInternalResponse(
        uuid=strategy.uuid,
        name=strategy.name,
        description=strategy.description,
        status=strategy.status,
        strategy_type=strategy.strategy_type,
        symbols=strategy.symbols,
        target_return=strategy.target_return,
        backtest_results=full_backtest_results,  # Full backtest results from BuildIteration
        config=config_data,
        docker_registry=strategy.docker_registry,
        docker_image_url=strategy.docker_image_url,
        version=strategy.version,
        subscriber_count=strategy.subscriber_count,
        rating=strategy.rating,
        user_id=strategy.user_id,
        created_at=strategy.created_at,
        updated_at=strategy.updated_at
    )

