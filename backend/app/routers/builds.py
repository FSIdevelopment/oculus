"""Router for strategy build orchestration endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime
import json
import asyncio
import logging
from redis.exceptions import ConnectionError as RedisConnectionError

from app.database import get_db, AsyncSessionLocal
from app.models.user import User
from app.models.strategy import Strategy
from app.models.strategy_build import StrategyBuild
from app.models.chat_history import ChatHistory
from app.auth.dependencies import get_current_active_user
from app.schemas.strategies import BuildResponse
from app.services.build_orchestrator import BuildOrchestrator
from app.services.docker_builder import DockerBuilder
from app.services.balance import deduct_tokens
from app.config import settings

logger = logging.getLogger(__name__)

# Estimated token cost for a build — used for server-side balance validation
ESTIMATED_TOKEN_COST = 50

router = APIRouter()


async def _save_chat_history(
    db: AsyncSession,
    user_id: str,
    strategy_id: str,
    user_message: str,
    assistant_message: str,
) -> None:
    """Save user and assistant messages to ChatHistory."""
    # Save user message
    user_chat = ChatHistory(
        message=user_message,
        message_type="user",
        content={"type": "prompt"},
        strategy_id=strategy_id,
        user_id=user_id,
    )
    db.add(user_chat)

    # Save assistant message
    assistant_chat = ChatHistory(
        message=assistant_message,
        message_type="assistant",
        content={"type": "design"},
        strategy_id=strategy_id,
        user_id=user_id,
    )
    db.add(assistant_chat)

    await db.flush()


class BuildTriggerRequest:
    """Request to trigger a strategy build."""
    strategy_id: str
    description: str = ""
    target_return: float = 10.0


async def _run_build_loop(
    build_id: str,
    strategy_id: str,
    user_id: str,
    strategy_name: str,
    strategy_symbols: list,
    description: str,
    target_return: float,
    timeframe: str = "1d",
):
    """
    Background task that runs the full build iteration loop.

    Uses its own DB session since the request-scoped session will be closed
    by the time this coroutine executes.
    """
    async with AsyncSessionLocal() as bg_db:
        # Re-fetch build and user with the background session
        result = await bg_db.execute(
            select(StrategyBuild).where(StrategyBuild.uuid == build_id)
        )
        build = result.scalar_one_or_none()
        if not build:
            return

        result = await bg_db.execute(
            select(User).where(User.uuid == user_id)
        )
        user = result.scalar_one_or_none()
        if not user:
            build.status = "failed"
            build.logs = json.dumps({"error": "User not found"})
            await bg_db.commit()
            return

        # Initialize orchestrator with background session
        orchestrator = BuildOrchestrator(bg_db, user, build)
        await orchestrator.initialize()

        try:
            # Configuration
            max_iterations = getattr(settings, 'MAX_BUILD_ITERATIONS', 5)
            design = None
            training_results = None
            iteration_logs = []

            # Iteration loop: design → train → refine → repeat
            for iteration in range(max_iterations):
                build.iteration_count = iteration

                if iteration == 0:
                    # First iteration: initial design
                    build.phase = "designing"
                    build.status = "building"

                    design = await orchestrator.design_strategy(
                        strategy_name=strategy_name,
                        symbols=strategy_symbols,
                        description=description,
                        timeframe=timeframe,
                        target_return=target_return,
                    )

                    # Save initial design to ChatHistory
                    await _save_chat_history(
                        bg_db, user_id, strategy_id,
                        orchestrator._build_design_prompt(
                            strategy_name, strategy_symbols,
                            description, timeframe, target_return, ""
                        ),
                        json.dumps(design)
                    )

                    iteration_logs.append(f"Iteration {iteration}: Initial design created")
                else:
                    # Subsequent iterations: refine based on training results
                    build.phase = "refining"

                    design = await orchestrator.refine_strategy(
                        previous_design=design,
                        backtest_results=training_results,
                        iteration=iteration,
                    )

                    # Save refinement to ChatHistory
                    refinement_prompt = orchestrator._build_refinement_prompt(
                        design, training_results, iteration
                    )
                    await _save_chat_history(
                        bg_db, user_id, strategy_id,
                        refinement_prompt,
                        json.dumps(design)
                    )

                    iteration_logs.append(f"Iteration {iteration}: Strategy refined")

                # Dispatch training job
                build.phase = "training"
                await bg_db.commit()

                job_id = await orchestrator.dispatch_training_job(
                    design, strategy_symbols, timeframe
                )

                # Listen for training results
                training_results = await orchestrator.listen_for_results(job_id)
                iteration_logs.append(f"Iteration {iteration}: Training completed")

                # Check if target metrics are met
                total_return = training_results.get('total_return', 0)
                if total_return >= target_return:
                    # Build Docker image before marking complete
                    build.phase = "building_docker"
                    await bg_db.commit()

                    # Re-fetch strategy for Docker build
                    strat_result = await bg_db.execute(
                        select(Strategy).where(Strategy.uuid == strategy_id)
                    )
                    strategy_obj = strat_result.scalar_one_or_none()

                    # TODO: Get strategy output directory from training results
                    # For now, use a placeholder path
                    strategy_output_dir = f"/tmp/strategy_{strategy_id}"

                    docker_builder = DockerBuilder(bg_db)
                    success = await docker_builder.build_and_push(
                        strategy_obj, build, strategy_output_dir
                    )

                    if success:
                        iteration_logs.append(f"Iteration {iteration}: Docker image built and pushed")
                        build.phase = "completed"
                    else:
                        iteration_logs.append(f"Iteration {iteration}: Docker build failed")
                        build.phase = "completed"

                    build.status = "complete"
                    iteration_logs.append(
                        f"Iteration {iteration}: Target achieved ({total_return}% >= {target_return}%)"
                    )
                    break
                elif iteration < max_iterations - 1:
                    iteration_logs.append(
                        f"Iteration {iteration}: Below target ({total_return}% < {target_return}%), refining..."
                    )

            # Final update
            build.completed_at = datetime.utcnow()
            build.logs = json.dumps({
                "design": design,
                "training_results": training_results,
                "iteration_logs": iteration_logs,
                "logs": orchestrator.logs,
            })

            await bg_db.commit()

        except Exception as e:
            # Mark build as failed — do NOT crash the background task
            build.status = "failed"
            build.logs = json.dumps({"error": str(e), "logs": orchestrator.logs})
            await bg_db.commit()
        finally:
            await orchestrator.cleanup()


@router.post("/api/builds/trigger", response_model=BuildResponse)
async def trigger_build(
    strategy_id: str,
    description: str = "",
    target_return: float = 10.0,
    timeframe: str = "1d",
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Trigger a new strategy build with LLM iteration loop.

    - Validates strategy ownership
    - Validates token balance (requires >= ESTIMATED_TOKEN_COST)
    - Creates StrategyBuild record and commits it
    - Launches the build loop as a background task
    - Returns BuildResponse immediately with status="queued"
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

    # Server-side token balance validation — deduct tokens before starting build
    await deduct_tokens(
        db,
        current_user.uuid,
        ESTIMATED_TOKEN_COST,
        f"Strategy build for '{strategy.name}'"
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
    await db.commit()

    # Capture values needed by the background task before the request session closes
    build_id = build.uuid
    user_id = current_user.uuid
    strategy_name = strategy.name

    # Backward-compatible symbols guard: if symbols is a dict (old data),
    # extract the "symbols" list from it; if it's already a list, use as-is
    raw_symbols = strategy.symbols
    if isinstance(raw_symbols, dict):
        strategy_symbols = raw_symbols.get("symbols", [])
    elif isinstance(raw_symbols, list):
        strategy_symbols = raw_symbols
    else:
        strategy_symbols = []

    strategy_description = description or strategy.description or ""

    # Launch the build loop as a background task
    asyncio.create_task(
        _run_build_loop(
            build_id=build_id,
            strategy_id=strategy_id,
            user_id=user_id,
            strategy_name=strategy_name,
            strategy_symbols=strategy_symbols,
            description=strategy_description,
            target_return=target_return,
            timeframe=timeframe,
        )
    )

    # Return immediately so the frontend can redirect to the build tracking page
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
    WebSocket endpoint for streaming build logs and progress.

    Subscribes to Redis pub/sub channel for real-time progress updates
    and relays them to the connected client.
    Note: Token validation should be done via query parameter or header.
    """
    import redis.asyncio as redis

    await websocket.accept()

    redis_client = None
    pubsub = None

    try:
        # Connect to Redis and subscribe to progress channel
        redis_client = await redis.from_url(settings.REDIS_URL)
        pubsub = redis_client.pubsub()

        progress_channel = f"oculus:training:progress:build:{build_id}"
        await pubsub.subscribe(progress_channel)

        # Send connection confirmation
        await websocket.send_json({
            "type": "status",
            "message": f"Connected to build {build_id}",
            "channel": progress_channel,
        })

        # Listen for messages from Redis and relay to client
        while True:
            # Check for incoming WebSocket messages (ping/pong)
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(), timeout=1.0
                )
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                pass

            # Check for Redis messages; catch mid-stream Redis disconnects
            try:
                message = await pubsub.get_message()
            except RedisConnectionError:
                logger.warning(
                    "Redis connection lost while listening for build %s logs",
                    build_id,
                )
                # Notify the client and close gracefully
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Lost connection to build log stream. Please reconnect.",
                    })
                    await websocket.close()
                except Exception:
                    pass
                break

            if message and message["type"] == "message":
                # Relay Redis message to client
                try:
                    progress_data = json.loads(message["data"])
                except (json.JSONDecodeError, TypeError):
                    progress_data = {"raw": str(message["data"])}

                await websocket.send_json({
                    "type": "progress",
                    "data": progress_data,
                })

            # Small delay to prevent busy-waiting
            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        pass
    finally:
        # Clean up Redis connection; handle cases where Redis already disconnected
        try:
            if pubsub:
                await pubsub.unsubscribe()
                await pubsub.close()
        except RedisConnectionError:
            logger.debug("Redis already disconnected during pub/sub cleanup for build %s", build_id)
        except Exception:
            logger.debug("Unexpected error during pub/sub cleanup for build %s", build_id)
        try:
            if redis_client:
                await redis_client.close()
        except RedisConnectionError:
            logger.debug("Redis already disconnected during client cleanup for build %s", build_id)
        except Exception:
            logger.debug("Unexpected error during client cleanup for build %s", build_id)

