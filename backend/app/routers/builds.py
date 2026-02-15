"""Router for strategy build orchestration endpoints."""
import os
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime
import json
import asyncio
import logging
import time
import redis.asyncio as redis_async
from redis.exceptions import ConnectionError as RedisConnectionError

from app.database import get_db, AsyncSessionLocal
from app.models.user import User
from app.models.strategy import Strategy
from app.models.strategy_build import StrategyBuild
from app.models.build_iteration import BuildIteration
from app.models.chat_history import ChatHistory
from app.auth.dependencies import get_current_active_user
from app.schemas.strategies import BuildResponse, BuildIterationResponse, BuildListItem, BuildListResponse
from pydantic import ValidationError
from app.services.build_orchestrator import BuildOrchestrator
from app.services.docker_builder import DockerBuilder
from app.services.balance import deduct_tokens
from app.config import settings
from sqlalchemy import select as sa_select

logger = logging.getLogger(__name__)

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


async def _publish_progress(build_id: str, data: dict) -> None:
    """Publish a progress update to the Redis channel for a build (best-effort)."""
    try:
        redis_client = await redis_async.from_url(settings.REDIS_URL)
        await redis_client.publish(
            f"oculus:training:progress:build:{build_id}",
            json.dumps(data),
        )
        await redis_client.close()
    except Exception:
        pass  # Redis notification is best-effort


async def _save_thinking_history(
    db: AsyncSession,
    user_id: str,
    strategy_id: str,
    thinking_text: str,
) -> None:
    """Save Claude thinking content to ChatHistory as a 'thinking' message."""
    if not thinking_text:
        return
    thinking_chat = ChatHistory(
        message=thinking_text,
        message_type="thinking",
        content={"type": "thinking"},
        strategy_id=strategy_id,
        user_id=user_id,
    )
    db.add(thinking_chat)
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
    max_iterations: int = 5,
    strategy_type: str | None = None,
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
            build.completed_at = datetime.utcnow()
            build.logs = json.dumps({"error": "User not found"})
            await bg_db.commit()
            return

        # Initialize orchestrator with background session
        orchestrator = BuildOrchestrator(bg_db, user, build)
        try:
            await orchestrator.initialize()
        except Exception as e:
            # Redis connection failed — mark build as failed and return early
            logger.error(
                f"Redis connection failed for build {build_id}: {e}",
                extra={
                    "error": "Redis connection failed — check REDIS_URL / ngrok tunnel",
                    "details": str(e),
                    "build_id": build_id,
                }
            )
            build.status = "failed"
            build.phase = "complete"
            build.completed_at = datetime.utcnow()
            await bg_db.commit()
            return

        try:
            design = None
            training_results = None
            iteration_logs = []

            # Iteration loop: design → train → refine → repeat
            for iteration in range(max_iterations):
                # Check for stop signal via Redis key
                try:
                    stop_client = await redis_async.from_url(settings.REDIS_URL)
                    stop_flag = await stop_client.get(f"oculus:build:stop:{build_id}")
                    await stop_client.close()
                    if stop_flag:
                        build.status = "stopped"
                        build.phase = "stopped"
                        build.completed_at = datetime.utcnow()
                        build.logs = json.dumps({
                            "message": "Build stopped by user",
                            "iterations_completed": iteration,
                            "iteration_logs": iteration_logs,
                        })
                        # Mark current iteration as stopped if one exists
                        if 'iteration_record' in locals() and iteration_record:
                            iteration_record.status = "failed"
                            iteration_record.duration_seconds = time.time() - iter_start_time
                        await bg_db.commit()
                        await _publish_progress(build_id, {
                            "phase": "stopped",
                            "iteration": iteration,
                            "message": "Build stopped by user",
                            "tokens_consumed": build.tokens_consumed,
                            "iteration_count": build.iteration_count,
                            "max_iterations": max_iterations,
                        })
                        logger.info("Build %s stopped by user at iteration %d", build_id, iteration)
                        return
                except Exception:
                    pass  # Redis check is best-effort

                # Deduct tokens for this iteration
                try:
                    await deduct_tokens(
                        bg_db,
                        user_id,
                        settings.TOKENS_PER_ITERATION,
                        f"Build iteration {iteration + 1} for '{strategy_name}'"
                    )
                    build.tokens_consumed += settings.TOKENS_PER_ITERATION
                    await bg_db.commit()
                except HTTPException as e:
                    if e.status_code == 402:
                        # Insufficient balance — stop build gracefully
                        build.status = "stopped"
                        build.phase = "insufficient_tokens"
                        build.completed_at = datetime.utcnow()
                        build.logs = json.dumps({
                            "error": "Insufficient token balance",
                            "message": f"Build stopped at iteration {iteration + 1}. Please purchase more tokens to continue.",
                            "iterations_completed": iteration,
                            "tokens_consumed": build.tokens_consumed,
                        })
                        await bg_db.commit()
                        await _publish_progress(build_id, {
                            "phase": "stopped",
                            "status": "insufficient_tokens",
                            "message": f"Build stopped — insufficient token balance. {iteration} iterations completed.",
                            "tokens_consumed": build.tokens_consumed,
                            "iteration_count": build.iteration_count,
                            "max_iterations": max_iterations,
                        })
                        logger.info("Build %s stopped at iteration %d due to insufficient tokens", build_id, iteration + 1)
                        return
                    raise  # Re-raise non-402 errors

                # Create BuildIteration record for this iteration
                iter_start_time = time.time()
                iteration_record = BuildIteration(
                    build_id=build_id,
                    user_id=user_id,
                    iteration_number=iteration,
                    status="pending",
                )
                bg_db.add(iteration_record)
                await bg_db.commit()
                await bg_db.refresh(iteration_record)
                iteration_uuid = iteration_record.uuid

                if iteration == 0:
                    # First iteration: initial design
                    build.phase = "designing"
                    build.status = "building"
                    iteration_record.status = "designing"
                    await bg_db.commit()
                    await _publish_progress(build_id, {
                        "phase": "designing",
                        "iteration": iteration,
                        "message": "Designing initial strategy...",
                        "tokens_consumed": build.tokens_consumed,
                        "iteration_count": build.iteration_count,
                        "max_iterations": max_iterations,
                    })

                    llm_result = await orchestrator.design_strategy(
                        strategy_name=strategy_name,
                        symbols=strategy_symbols,
                        description=description,
                        timeframe=timeframe,
                        target_return=target_return,
                        strategy_type=strategy_type,
                    )
                    design = llm_result["design"]
                    thinking_text = llm_result.get("thinking", "")

                    # Store LLM context in iteration record
                    iteration_record.llm_design = design
                    iteration_record.llm_thinking = thinking_text
                    await bg_db.commit()

                    # Save initial design to ChatHistory
                    await _save_chat_history(
                        bg_db, user_id, strategy_id,
                        orchestrator._build_design_prompt(
                            strategy_name, strategy_type, strategy_symbols,
                            description, timeframe, target_return, "", ""
                        ),
                        json.dumps(design)
                    )
                    # Save thinking content to ChatHistory
                    await _save_thinking_history(bg_db, user_id, strategy_id, thinking_text)

                    # Publish thinking via Redis
                    if thinking_text:
                        await _publish_progress(build_id, {
                            "phase": "designing",
                            "iteration": iteration,
                            "type": "thinking",
                            "thinking": thinking_text,
                            "tokens_consumed": build.tokens_consumed,
                            "iteration_count": build.iteration_count,
                            "max_iterations": max_iterations,
                        })

                    iteration_logs.append(f"Iteration {iteration}: Initial design created")
                else:
                    # Subsequent iterations: refine based on training results
                    build.phase = "refining"
                    iteration_record.status = "designing"
                    await bg_db.commit()
                    await _publish_progress(build_id, {
                        "phase": "refining",
                        "iteration": iteration,
                        "message": f"Refining strategy (iteration {iteration})...",
                        "tokens_consumed": build.tokens_consumed,
                        "iteration_count": build.iteration_count,
                        "max_iterations": max_iterations,
                    })

                    llm_result = await orchestrator.refine_strategy(
                        previous_design=design,
                        backtest_results=training_results,
                        iteration=iteration,
                        strategy_type=strategy_type,
                    )
                    design = llm_result["design"]
                    thinking_text = llm_result.get("thinking", "")

                    # Store LLM context in iteration record
                    iteration_record.llm_design = design
                    iteration_record.llm_thinking = thinking_text
                    await bg_db.commit()

                    # Save refinement to ChatHistory
                    refinement_prompt = orchestrator._build_refinement_prompt(
                        design, training_results, iteration, ""
                    )
                    await _save_chat_history(
                        bg_db, user_id, strategy_id,
                        refinement_prompt,
                        json.dumps(design)
                    )
                    # Save thinking content to ChatHistory
                    await _save_thinking_history(bg_db, user_id, strategy_id, thinking_text)

                    # Publish thinking via Redis
                    if thinking_text:
                        await _publish_progress(build_id, {
                            "phase": "refining",
                            "iteration": iteration,
                            "type": "thinking",
                            "thinking": thinking_text,
                            "tokens_consumed": build.tokens_consumed,
                            "iteration_count": build.iteration_count,
                            "max_iterations": max_iterations,
                        })

                    iteration_logs.append(f"Iteration {iteration}: Strategy refined")

                # Dispatch training job
                build.phase = "training"
                iteration_record.status = "training"
                await bg_db.commit()
                await _publish_progress(build_id, {
                    "phase": "training",
                    "iteration": iteration,
                    "message": f"Training model (iteration {iteration})...",
                    "tokens_consumed": build.tokens_consumed,
                    "iteration_count": build.iteration_count,
                    "max_iterations": max_iterations,
                })

                job_id = await orchestrator.dispatch_training_job(
                    design, strategy_symbols, timeframe, iteration_uuid=iteration_uuid
                )

                # Listen for training results
                training_results = await orchestrator.listen_for_results(job_id)
                iteration_logs.append(f"Iteration {iteration}: Training completed")

                # Persist training results to BuildIteration record
                iteration_record.status = "complete" if training_results.get("status") == "success" else "failed"
                iteration_record.duration_seconds = time.time() - iter_start_time
                iteration_record.training_config = training_results.get("design_config")
                iteration_record.optimal_label_config = training_results.get("optimal_label_config")
                iteration_record.features = training_results.get("features")
                iteration_record.hyperparameter_results = training_results.get("hyperparameter_results")
                iteration_record.nn_training_results = training_results.get("nn_training_results")
                iteration_record.lstm_training_results = training_results.get("lstm_training_results")
                iteration_record.model_evaluations = training_results.get("model_evaluations")
                iteration_record.best_model = training_results.get("best_model")
                iteration_record.backtest_results = training_results.get("backtest_results")

                # Extract rules from the result
                extracted = training_results.get("extracted_rules", {})
                iteration_record.entry_rules = extracted.get("entry_rules")
                iteration_record.exit_rules = extracted.get("exit_rules")

                await bg_db.commit()

                # Mark iteration as complete (increment AFTER work is persisted)
                build.iteration_count = iteration + 1
                await bg_db.commit()

                # Non-blocking: update creation guide with lessons from this iteration
                try:
                    await orchestrator.update_creation_guide(
                        db=bg_db,
                        strategy_type=strategy_type,
                        design=design,
                        training_results=training_results,
                        backtest_results=training_results.get("backtest_results", {}),
                    )
                    await bg_db.commit()
                except Exception:
                    pass  # update_creation_guide already logs warnings internally

                # Publish per-iteration training results via Redis
                await _publish_progress(build_id, {
                    "phase": "training_complete",
                    "iteration": iteration,
                    "iteration_uuid": iteration_uuid,
                    "message": f"Training completed (iteration {iteration})",
                    "results": {
                        "total_return": (training_results.get("backtest_results") or {}).get("total_return"),
                        "win_rate": (training_results.get("backtest_results") or {}).get("win_rate"),
                        "sharpe_ratio": (training_results.get("backtest_results") or {}).get("sharpe_ratio"),
                        "model": (training_results.get("best_model") or {}).get("name"),
                        "best_model": training_results.get("best_model"),
                        "model_metrics": training_results.get("model_metrics"),
                        "optimal_label_config": training_results.get("optimal_label_config"),
                    },
                    "tokens_consumed": build.tokens_consumed,
                    "iteration_count": build.iteration_count,
                    "max_iterations": max_iterations,
                })

                # Check if target metrics are met
                total_return = (training_results.get('backtest_results') or {}).get('total_return', 0)
                if total_return >= target_return:
                    # Build Docker image before marking complete
                    build.phase = "building_docker"
                    await bg_db.commit()
                    await _publish_progress(build_id, {
                        "phase": "building_docker",
                        "iteration": iteration,
                        "message": "Building Docker image...",
                        "tokens_consumed": build.tokens_consumed,
                        "iteration_count": build.iteration_count,
                        "max_iterations": max_iterations,
                    })

                    # Re-fetch strategy for Docker build
                    strat_result = await bg_db.execute(
                        select(Strategy).where(Strategy.uuid == strategy_id)
                    )
                    strategy_obj = strat_result.scalar_one_or_none()

                    # --- Persist strategy files from worker to disk ---
                    strategy_output_dir = str(
                        Path(__file__).resolve().parent.parent.parent
                        / "strategy_outputs"
                        / build_id
                    )
                    strategy_files = training_results.get("strategy_files", {})
                    try:
                        output_path = Path(strategy_output_dir)
                        await asyncio.to_thread(output_path.mkdir, parents=True, exist_ok=True)
                        for filename, content in strategy_files.items():
                            file_path = output_path / filename
                            await asyncio.to_thread(file_path.write_text, content)
                        iteration_logs.append(
                            f"Iteration {iteration}: Persisted {len(strategy_files)} strategy file(s) to {strategy_output_dir}"
                        )
                    except Exception as e:
                        logger.warning("Failed to persist strategy files for build %s: %s", build_id, e)
                        iteration_logs.append(f"Iteration {iteration}: Strategy file persistence failed: {e}")

                    # --- Persist backtest_results.json ---
                    try:
                        backtest_results = training_results.get("backtest_results", {})
                        backtest_json_path = Path(strategy_output_dir) / "backtest_results.json"
                        await asyncio.to_thread(
                            backtest_json_path.write_text,
                            json.dumps(backtest_results, indent=2)
                        )
                        iteration_logs.append("Persisted backtest_results.json")
                    except Exception as e:
                        logger.warning("Failed to persist backtest_results.json for build %s: %s", build_id, e)

                    # --- Generate README via Claude ---
                    try:
                        readme_content = await orchestrator.generate_strategy_readme(
                            strategy_name=strategy_name,
                            symbols=strategy_symbols,
                            design=design,
                            backtest_results=training_results.get("backtest_results", {}),
                            model_metrics=training_results.get("model_metrics", {}),
                            config=design,
                            strategy_type=strategy_type or "",
                        )
                        # Write README to strategy output directory
                        readme_path = Path(strategy_output_dir) / "README.md"
                        await asyncio.to_thread(readme_path.write_text, readme_content)
                        iteration_logs.append(f"Iteration {iteration}: README generated ({len(readme_content)} chars)")
                    except Exception as e:
                        logger.warning("README generation failed for build %s: %s", build_id, e)
                        iteration_logs.append(f"Iteration {iteration}: README generation failed: {e}")

                    # --- Docker build ---
                    docker_builder = DockerBuilder(bg_db)
                    success = await docker_builder.build_and_push(
                        strategy_obj, build, strategy_output_dir
                    )

                    if success:
                        iteration_logs.append(f"Iteration {iteration}: Docker image built and pushed")
                        build.phase = "complete"
                    else:
                        iteration_logs.append(f"Iteration {iteration}: Docker build failed")
                        build.phase = "complete"

                    build.status = "complete"
                    iteration_logs.append(
                        f"Iteration {iteration}: Target achieved ({total_return}% >= {target_return}%)"
                    )
                    await _publish_progress(build_id, {
                        "phase": "complete",
                        "iteration": iteration,
                        "message": f"Build complete — target achieved ({total_return}% >= {target_return}%)",
                        "tokens_consumed": build.tokens_consumed,
                        "iteration_count": build.iteration_count,
                        "max_iterations": max_iterations,
                    })
                    break
                elif iteration < max_iterations - 1:
                    iteration_logs.append(
                        f"Iteration {iteration}: Below target ({total_return}% < {target_return}%), refining..."
                    )

            # --- Best iteration selection (target not met after all iterations) ---
            # Only run if the target was NOT met (build.status != "complete")
            if build.status != "complete":
                # Query all completed BuildIteration records for this build
                iter_result = await bg_db.execute(
                    sa_select(BuildIteration)
                    .where(BuildIteration.build_id == build_id, BuildIteration.status == "complete")
                    .order_by(BuildIteration.iteration_number)
                )
                completed_iterations = iter_result.scalars().all()

                if completed_iterations:
                    # Find the best iteration by total_return (from backtest_results)
                    best_iter = max(
                        completed_iterations,
                        key=lambda it: (it.backtest_results or {}).get("total_return", float("-inf"))
                    )
                    best_return = (best_iter.backtest_results or {}).get("total_return", 0)
                    best_iter_num = best_iter.iteration_number

                    iteration_logs.append(
                        f"All {len(completed_iterations)} iterations complete. Best: iteration {best_iter_num} ({best_return}% return)"
                    )

                    # Publish selection decision via WebSocket
                    await _publish_progress(build_id, {
                        "phase": "selecting_best",
                        "message": f"Selecting best iteration: #{best_iter_num} with {best_return}% return",
                        "best_iteration": best_iter_num,
                        "best_return": best_return,
                        "tokens_consumed": build.tokens_consumed,
                        "iteration_count": build.iteration_count,
                        "max_iterations": max_iterations,
                    })

                    # Generate selection thinking via Claude
                    try:
                        selection_thinking = await orchestrator.generate_selection_thinking(
                            completed_iterations=completed_iterations,
                            best_iteration=best_iter,
                            target_return=target_return,
                            strategy_type=strategy_type or "",
                        )
                        await _publish_progress(build_id, {
                            "phase": "selecting_best",
                            "message": "Design thinking: iteration selection",
                            "thinking": selection_thinking,
                            "tokens_consumed": build.tokens_consumed,
                            "iteration_count": build.iteration_count,
                            "max_iterations": max_iterations,
                        })
                    except Exception as e:
                        logger.warning("Selection thinking failed for build %s: %s", build_id, e)

                    # Use the best iteration's design and training results for Docker build
                    design = best_iter.llm_design or design
                    training_results = best_iter.backtest_results or training_results

                    # --- Persist strategy files, generate README, build Docker ---
                    # (Same logic as the target-met path at lines 442-504)
                    build.phase = "building_docker"
                    await bg_db.commit()
                    await _publish_progress(build_id, {
                        "phase": "building_docker",
                        "iteration": best_iter_num,
                        "message": "Building Docker image for best iteration...",
                        "tokens_consumed": build.tokens_consumed,
                        "iteration_count": build.iteration_count,
                        "max_iterations": max_iterations,
                    })

                    # Persist strategy files from best iteration's training result
                    strategy_output_dir = str(
                        Path(__file__).resolve().parent.parent.parent / "strategy_outputs" / build_id
                    )
                    # Get strategy files: prefer from the best iteration's stored data
                    strategy_files = (best_iter.training_config or {}).get("strategy_files", {})
                    if not strategy_files:
                        strategy_files = (training_results or {}).get("strategy_files", {})
                    try:
                        output_path = Path(strategy_output_dir)
                        await asyncio.to_thread(output_path.mkdir, parents=True, exist_ok=True)
                        for filename, content in strategy_files.items():
                            file_path = output_path / filename
                            await asyncio.to_thread(file_path.write_text, content)
                        iteration_logs.append(f"Persisted {len(strategy_files)} strategy file(s)")
                    except Exception as e:
                        logger.warning("Failed to persist strategy files: %s", e)

                    # --- Persist backtest_results.json ---
                    try:
                        backtest_results = best_iter.backtest_results or {}
                        backtest_json_path = Path(strategy_output_dir) / "backtest_results.json"
                        await asyncio.to_thread(
                            backtest_json_path.write_text,
                            json.dumps(backtest_results, indent=2)
                        )
                        iteration_logs.append("Persisted backtest_results.json")
                    except Exception as e:
                        logger.warning("Failed to persist backtest_results.json: %s", e)

                    # Generate README
                    try:
                        readme_content = await orchestrator.generate_strategy_readme(
                            strategy_name=strategy_name,
                            symbols=strategy_symbols,
                            design=design,
                            backtest_results=best_iter.backtest_results or {},
                            model_metrics=(best_iter.best_model or {}),
                            config=design,
                            strategy_type=strategy_type or "",
                        )
                        readme_path = Path(strategy_output_dir) / "README.md"
                        await asyncio.to_thread(readme_path.write_text, readme_content)
                    except Exception as e:
                        logger.warning("README generation failed: %s", e)

                    # Docker build
                    strat_result = await bg_db.execute(
                        select(Strategy).where(Strategy.uuid == strategy_id)
                    )
                    strategy_obj = strat_result.scalar_one_or_none()
                    docker_builder = DockerBuilder(bg_db)
                    success = await docker_builder.build_and_push(strategy_obj, build, strategy_output_dir)

                    if success:
                        iteration_logs.append("Docker image built and pushed for best iteration")
                    else:
                        iteration_logs.append("Docker build failed for best iteration")

                    build.status = "complete"
                    build.phase = "complete"
                    await _publish_progress(build_id, {
                        "phase": "complete",
                        "message": f"Build complete — best iteration: #{best_iter_num} ({best_return}% return, target was {target_return}%)",
                        "tokens_consumed": build.tokens_consumed,
                        "iteration_count": build.iteration_count,
                        "max_iterations": max_iterations,
                    })
                else:
                    # No completed iterations at all — mark as failed
                    build.status = "failed"
                    build.phase = "complete"
                    iteration_logs.append("No iterations completed successfully")

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
            build.completed_at = datetime.utcnow()
            build.logs = json.dumps({"error": str(e), "logs": orchestrator.logs})
            # Mark current iteration as failed if one was in progress
            if 'iteration_record' in locals() and iteration_record:
                iteration_record.status = "failed"
                if 'iter_start_time' in locals():
                    iteration_record.duration_seconds = time.time() - iter_start_time
            await bg_db.commit()
        finally:
            await orchestrator.cleanup()


@router.post("/api/builds/trigger", response_model=BuildResponse)
async def trigger_build(
    strategy_id: str,
    description: str = "",
    target_return: float = 10.0,
    timeframe: str = "1d",
    max_iterations: int = 5,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Trigger a new strategy build with LLM iteration loop.

    - Validates strategy ownership
    - Validates token balance (requires >= TOKENS_PER_ITERATION for at least one iteration)
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

    # Validate user has enough tokens for at least one iteration
    if current_user.balance < settings.TOKENS_PER_ITERATION:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"Insufficient token balance. You need at least {settings.TOKENS_PER_ITERATION} tokens per iteration. Current balance: {current_user.balance}"
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
    strat_type = strategy.strategy_type  # e.g. "Momentum", "Mean Reversion", etc.

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
            max_iterations=max_iterations,
            strategy_type=strat_type,
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
        strategy_type=strategy.strategy_type,
        strategy_name=strategy.name,
    )


@router.get("/api/builds/pricing")
async def get_build_pricing():
    """Return current build pricing configuration (public, no auth required)."""
    return {
        "tokens_per_iteration": settings.TOKENS_PER_ITERATION,
    }


@router.get("/api/builds", response_model=BuildListResponse)
async def list_builds(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List all builds for the authenticated user (paginated).

    Excludes completed builds (status='complete') as they belong on the strategies page.
    Returns builds ordered by started_at DESC (latest first).
    Each build includes the strategy_name from the joined Strategy table.
    """
    # Count total builds (excluding completed)
    count_result = await db.execute(
        select(func.count(StrategyBuild.uuid)).where(
            StrategyBuild.user_id == current_user.uuid,
            StrategyBuild.status != "complete"
        )
    )
    total = count_result.scalar()

    # Fetch paginated builds with strategy name and type
    result = await db.execute(
        select(
            StrategyBuild,
            Strategy.name.label("strategy_name"),
            Strategy.strategy_type.label("strategy_type")
        )
        .join(Strategy, StrategyBuild.strategy_id == Strategy.uuid)
        .where(
            StrategyBuild.user_id == current_user.uuid,
            StrategyBuild.status != "complete"
        )
        .order_by(StrategyBuild.started_at.desc())
        .offset(skip)
        .limit(limit)
    )
    rows = result.all()

    # Build response items
    items = []
    for build, strategy_name, strategy_type in rows:
        item_dict = {
            "uuid": build.uuid,
            "strategy_id": build.strategy_id,
            "status": build.status,
            "phase": build.phase,
            "tokens_consumed": build.tokens_consumed,
            "iteration_count": build.iteration_count,
            "started_at": build.started_at,
            "completed_at": build.completed_at,
            "strategy_name": strategy_name,
            "strategy_type": strategy_type,
        }
        items.append(BuildListItem(**item_dict))

    return BuildListResponse(
        items=items,
        total=total,
        skip=skip,
        limit=limit,
    )


@router.get("/api/builds/{build_id}/iterations")
async def get_build_iterations(
    build_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get all BuildIteration records for a build.

    Returns iterations ordered by iteration_number ascending.
    Requires authentication and verifies user owns the build.
    """
    # Fetch build with strategy relationship to check ownership
    result = await db.execute(
        select(StrategyBuild)
        .where(StrategyBuild.uuid == build_id)
    )
    build = result.scalar_one_or_none()

    if not build:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Build not found"
        )

    # Fetch strategy to verify ownership
    strat_result = await db.execute(
        select(Strategy).where(Strategy.uuid == build.strategy_id)
    )
    strategy = strat_result.scalar_one_or_none()

    if not strategy or strategy.user_id != current_user.uuid:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Build not found"
        )

    # Fetch all iterations for this build, ordered by iteration_number
    iterations_result = await db.execute(
        select(BuildIteration)
        .where(BuildIteration.build_id == build_id)
        .order_by(BuildIteration.iteration_number.asc())
    )
    iterations = iterations_result.scalars().all()

    # Convert to response schema
    iteration_responses = []
    for iteration in iterations:
        try:
            iteration_responses.append(BuildIterationResponse.model_validate(iteration))
        except ValidationError as e:
            logger.error(
                f"Failed to validate iteration {iteration.uuid} for build {build_id}: {e}. "
                f"Iteration data: iteration_number={iteration.iteration_number}, "
                f"entry_rules type={type(iteration.entry_rules)}, "
                f"exit_rules type={type(iteration.exit_rules)}"
            )
            # Skip this iteration instead of crashing the whole endpoint
            continue

    return {
        "iterations": iteration_responses,
        "total": len(iteration_responses),
        "build_id": build_id,
    }


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

    # Fetch strategy for name and type
    strat_result = await db.execute(
        select(Strategy).where(Strategy.uuid == build.strategy_id)
    )
    strategy = strat_result.scalar_one_or_none()

    return BuildResponse(
        uuid=build.uuid,
        strategy_id=build.strategy_id,
        status=build.status,
        phase=build.phase,
        tokens_consumed=build.tokens_consumed,
        iteration_count=build.iteration_count,
        started_at=build.started_at,
        completed_at=build.completed_at,
        strategy_type=strategy.strategy_type if strategy else None,
        strategy_name=strategy.name if strategy else None,
    )



@router.post("/api/builds/{build_id}/stop", response_model=BuildResponse)
async def stop_build(
    build_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Stop an active build gracefully.

    Sets a Redis stop key that the build loop checks at each iteration.
    Only builds with status 'building' or 'queued' can be stopped.
    """
    # Fetch build and validate ownership
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

    if build.status not in ("building", "queued"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot stop build with status '{build.status}'. Only 'building' or 'queued' builds can be stopped."
        )

    # Set Redis stop key with 1-hour TTL
    try:
        redis_client = await redis_async.from_url(settings.REDIS_URL)
        await redis_client.set(f"oculus:build:stop:{build_id}", "1", ex=3600)
        await redis_client.close()
    except Exception as e:
        logger.warning("Failed to set Redis stop key for build %s: %s", build_id, e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not signal build to stop. Please try again."
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


@router.post("/api/builds/{build_id}/restart", response_model=BuildResponse)
async def restart_build(
    build_id: str,
    max_iterations: int = 5,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Restart a failed or stopped build by creating a new build record.

    Creates a new StrategyBuild and launches a fresh build loop.
    Only builds with status 'failed' or 'stopped' can be restarted.
    """
    # Fetch original build and validate ownership
    result = await db.execute(
        select(StrategyBuild).where(
            (StrategyBuild.uuid == build_id) & (StrategyBuild.user_id == current_user.uuid)
        )
    )
    original_build = result.scalar_one_or_none()

    if not original_build:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Build not found"
        )

    if original_build.status not in ("failed", "stopped"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot restart build with status '{original_build.status}'. Only 'failed' or 'stopped' builds can be restarted."
        )

    # Validate user has enough tokens for at least one iteration
    if current_user.balance < settings.TOKENS_PER_ITERATION:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"Insufficient token balance. You need at least {settings.TOKENS_PER_ITERATION} tokens per iteration. Current balance: {current_user.balance}"
        )

    # Fetch the strategy for the build
    strat_result = await db.execute(
        select(Strategy).where(
            (Strategy.uuid == original_build.strategy_id) & (Strategy.user_id == current_user.uuid)
        )
    )
    strategy = strat_result.scalar_one_or_none()

    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Strategy not found"
        )

    # Create a NEW build record (don't reuse the failed/stopped one)
    new_build = StrategyBuild(
        strategy_id=original_build.strategy_id,
        user_id=current_user.uuid,
        status="queued",
        phase="initializing",
        started_at=datetime.utcnow(),
    )
    db.add(new_build)
    await db.commit()

    # Capture values for background task
    new_build_id = new_build.uuid
    user_id = current_user.uuid
    strategy_name = strategy.name

    raw_symbols = strategy.symbols
    if isinstance(raw_symbols, dict):
        strategy_symbols = raw_symbols.get("symbols", [])
    elif isinstance(raw_symbols, list):
        strategy_symbols = raw_symbols
    else:
        strategy_symbols = []

    strategy_description = strategy.description or ""
    strat_type = strategy.strategy_type  # e.g. "Momentum", "Mean Reversion", etc.

    # Launch the build loop as a background task
    asyncio.create_task(
        _run_build_loop(
            build_id=new_build_id,
            strategy_id=original_build.strategy_id,
            user_id=user_id,
            strategy_name=strategy_name,
            strategy_symbols=strategy_symbols,
            description=strategy_description,
            target_return=strategy.target_return or 10.0,
            timeframe="1d",
            max_iterations=max_iterations,
            strategy_type=strat_type,
        )
    )

    return BuildResponse(
        uuid=new_build.uuid,
        strategy_id=new_build.strategy_id,
        status=new_build.status,
        phase=new_build.phase,
        tokens_consumed=new_build.tokens_consumed,
        iteration_count=new_build.iteration_count,
        started_at=new_build.started_at,
        completed_at=new_build.completed_at,
        strategy_type=strategy.strategy_type,
        strategy_name=strategy.name,
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
        try:
            redis_client = await redis.from_url(settings.REDIS_URL)
            pubsub = redis_client.pubsub()

            progress_channel = f"oculus:training:progress:build:{build_id}"
            await pubsub.subscribe(progress_channel)
        except (RedisConnectionError, ConnectionRefusedError, OSError) as e:
            logger.warning("Cannot connect to Redis for build %s WebSocket: %s", build_id, e)
            await websocket.send_json({
                "type": "error",
                "message": "Build log streaming temporarily unavailable. Build is still running.",
            })
            await websocket.close()
            return

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


