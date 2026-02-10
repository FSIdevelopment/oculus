"""Router for strategy build orchestration endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime
import json

from app.database import get_db
from app.models.user import User
from app.models.strategy import Strategy
from app.models.strategy_build import StrategyBuild
from app.auth.dependencies import get_current_active_user
from app.schemas.strategies import BuildResponse
from app.services.build_orchestrator import BuildOrchestrator

router = APIRouter()


class BuildTriggerRequest:
    """Request to trigger a strategy build."""
    strategy_id: str
    description: str = ""
    target_return: float = 10.0


@router.post("/api/builds/trigger", response_model=BuildResponse)
async def trigger_build(
    strategy_id: str,
    description: str = "",
    target_return: float = 10.0,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Trigger a new strategy build.
    
    - Validates strategy ownership
    - Creates StrategyBuild record
    - Initializes build orchestrator
    - Starts LLM design phase
    - Returns build UUID for tracking
    """
    # Fetch strategy
    result = await db.execute(
        select(Strategy).where(
            (Strategy.uuid == strategy_id) & (Strategy.user_id == current_user.uuid)
        )
    )
    strategy = result.scalar_one_or_none()
    
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Strategy not found"
        )
    
    # Create build record
    build = StrategyBuild(
        strategy_id=strategy_id,
        user_id=current_user.uuid,
        status="queued",
        phase="initializing",
        started_at=datetime.utcnow(),
    )
    
    db.add(build)
    await db.flush()
    
    # Initialize orchestrator
    orchestrator = BuildOrchestrator(db, current_user, build)
    await orchestrator.initialize()
    
    try:
        # Start LLM design phase
        build.phase = "designing"
        build.status = "building"
        
        design = await orchestrator.design_strategy(
            strategy_name=strategy.name,
            symbols=strategy.symbols or [],
            description=description or strategy.description or "",
            timeframe="1d",  # TODO: Get from strategy
            target_return=target_return,
        )
        
        # Store design in build logs
        build.logs = json.dumps({"design": design})
        
        # Dispatch training job
        build.phase = "training"
        job_id = await orchestrator.dispatch_training_job(
            design, strategy.symbols or [], "1d"
        )
        
        # Update build
        build.logs = json.dumps({
            "design": design,
            "job_id": job_id,
            "logs": orchestrator.logs,
        })
        
        await db.commit()
        
        return BuildResponse(
            uuid=build.uuid,
            strategy_id=build.strategy_id,
            status=build.status,
            phase=build.phase,
            tokens_consumed=build.tokens_consumed,
            iteration_count=build.iteration_count,
            started_at=build.started_at,
            completed_at=build.completed_at,
        )
        
    except Exception as e:
        build.status = "failed"
        build.logs = json.dumps({"error": str(e), "logs": orchestrator.logs})
        await db.commit()
        await orchestrator.cleanup()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Build failed: {str(e)}"
        )


@router.get("/api/builds/{build_id}", response_model=BuildResponse)
async def get_build_status(
    build_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get build status and progress.
    
    Returns current phase, status, tokens consumed, and iteration count.
    """
    result = await db.execute(
        select(StrategyBuild).where(
            (StrategyBuild.uuid == build_id) & (StrategyBuild.user_id == current_user.uuid)
        )
    )
    build = result.scalar_one_or_none()
    
    if not build:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Build not found"
        )
    
    return BuildResponse(
        uuid=build.uuid,
        strategy_id=build.strategy_id,
        status=build.status,
        phase=build.phase,
        tokens_consumed=build.tokens_consumed,
        iteration_count=build.iteration_count,
        started_at=build.started_at,
        completed_at=build.completed_at,
    )


@router.websocket("/ws/builds/{build_id}/logs")
async def websocket_build_logs(websocket: WebSocket, build_id: str):
    """
    WebSocket endpoint for streaming build logs.

    Streams real-time logs as the build progresses through phases.
    Note: Token validation should be done via query parameter or header.
    """
    await websocket.accept()

    try:
        # TODO: Implement real-time log streaming from Redis
        # For now, send a placeholder message
        await websocket.send_json({
            "type": "status",
            "message": f"Connected to build {build_id}",
        })

        # Keep connection open
        while True:
            # Receive ping/pong to keep connection alive
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")

    except WebSocketDisconnect:
        pass

