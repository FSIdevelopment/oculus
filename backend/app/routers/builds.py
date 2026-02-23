"""Router for strategy build orchestration endpoints."""
import os
import sys
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any, Set
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
from app.services.balance import deduct_tokens, check_balance
from app.config import settings
from sqlalchemy import select as sa_select
from app.services.build_queue import BuildQueueManager
from app.services.checkpoint_manager import CheckpointManager
from app.services.priority_manager import PriorityManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Strong references to background tasks to prevent garbage collection.
# Tasks are added on creation and removed via a done callback when they complete.
_background_tasks: Set[asyncio.Task] = set()


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


async def _get_queue_position(build_id: str) -> int | None:
    """Get the current queue position for a build.

    Args:
        build_id: Build ID

    Returns:
        Queue position (1-based) or None if not in queue
    """
    try:
        redis_client = await redis_async.from_url(settings.REDIS_URL, decode_responses=True)
        queue_manager = BuildQueueManager(redis_client)
        position = await queue_manager.get_queue_position(build_id)
        await redis_client.close()
        return position
    except Exception as e:
        logger.warning(f"Failed to get queue position for build {build_id}: {e}")
        return None


async def _publish_progress(build_id: str, data: dict, max_retries: int = 3) -> None:
    """Publish a progress update to the Redis channel for a build with retry logic.

    Note: Uses a timeout to avoid blocking if Redis is busy with long-running training jobs.
    """
    # Add queue position to progress data if build is queued
    if data.get("phase") in ["initializing", "queued"] or data.get("status") == "queued":
        queue_position = await _get_queue_position(build_id)
        if queue_position is not None:
            data["queue_position"] = queue_position
    for attempt in range(max_retries):
        redis_client = None
        try:
            redis_client = await redis_async.from_url(
                settings.REDIS_URL,
                socket_keepalive=True,
                socket_connect_timeout=5,
                socket_timeout=10,
                retry_on_timeout=True,
            )
            # Use timeout to avoid blocking if Redis is busy with training
            await asyncio.wait_for(
                redis_client.publish(
                    f"oculus:training:progress:build:{build_id}",
                    json.dumps(data),
                ),
                timeout=3.0  # 3 second timeout for publish
            )
            await redis_client.close()
            return  # Success
        except asyncio.TimeoutError:
            # Redis is busy (likely processing training), log but don't retry aggressively
            logger.debug(
                f"Redis publish timed out for build {build_id} (likely busy with training), skipping this update"
            )
            return  # Skip this update rather than retrying
        except (RedisConnectionError, redis_async.TimeoutError, OSError) as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Failed to publish progress for build {build_id} (attempt {attempt + 1}/{max_retries}): {e}"
                )
                await asyncio.sleep(0.5 * (2 ** attempt))  # Exponential backoff
            else:
                logger.error(
                    f"Failed to publish progress for build {build_id} after {max_retries} attempts: {e}"
                )
        except Exception as e:
            logger.error(f"Unexpected error publishing progress for build {build_id}: {e}")
            break
        finally:
            if redis_client:
                try:
                    await redis_client.close()
                except Exception:
                    pass


async def _save_thinking_history(
    db: AsyncSession,
    user_id: str,
    strategy_id: str,
    thinking_text: str,
    iteration: int | None = None,
) -> None:
    """Save Claude thinking content to ChatHistory as a 'thinking' message."""
    if not thinking_text:
        return
    content_metadata = {"type": "thinking"}
    if iteration is not None:
        content_metadata["iteration"] = iteration
    thinking_chat = ChatHistory(
        message=thinking_text,
        message_type="thinking",
        content=content_metadata,
        strategy_id=strategy_id,
        user_id=user_id,
    )
    db.add(thinking_chat)
    await db.flush()


async def _save_progress_message(
    db: AsyncSession,
    user_id: str,
    strategy_id: str,
    message: str,
    phase: str,
    iteration: int | None = None,
) -> None:
    """Save a progress step message to ChatHistory as a 'progress' message."""
    if not message:
        return
    content_metadata = {"type": "progress", "phase": phase}
    if iteration is not None:
        content_metadata["iteration"] = iteration
    progress_chat = ChatHistory(
        message=message,
        message_type="progress",
        content=content_metadata,
        strategy_id=strategy_id,
        user_id=user_id,
    )
    db.add(progress_chat)
    await db.flush()


class BuildTriggerRequest:
    """Request to trigger a strategy build."""
    strategy_id: str
    description: str = ""
    target_return: float = 10.0


async def _build_strategy_container(
    build_id: str,
    build: StrategyBuild,
    bg_db: AsyncSession,
    orchestrator: BuildOrchestrator,
    strategy_name: str,
    strategy_symbols: list,
    design: dict,
    training_results: dict,
    strategy_type: str,
    iteration: int,
    max_iterations: int,
    iteration_logs: list,
    iteration_id: str,
    timeframe: str = "1d",
) -> str:
    """
    Build the strategy container files: copy static templates,
    generate custom files via LLM, verify with backtest, generate README.

    Returns the strategy_output_dir path.
    """
    # Step 1: Create output directory
    strategy_output_dir = str(
        Path(__file__).resolve().parent.parent.parent / "strategy_outputs" / build_id
    )
    output_path = Path(strategy_output_dir)
    await asyncio.to_thread(output_path.mkdir, parents=True, exist_ok=True)

    # Step 2: Copy static template files
    await _publish_progress(build_id, {
        "phase": "building_docker",
        "iteration": iteration,
        "message": "Copying template files...",
        "tokens_consumed": build.tokens_consumed,
        "iteration_count": build.iteration_count,
        "max_iterations": max_iterations,
    })

    template_dir = Path(__file__).resolve().parent.parent.parent / "templates" / "strategy_container"
    for static_file in ["main.py", "Dockerfile", "data_provider.py", "requirements.txt"]:
        src = template_dir / static_file
        dst = output_path / static_file
        await asyncio.to_thread(shutil.copy2, str(src), str(dst))

    # Step 3: Write worker-generated files when available, otherwise fall back to
    # template + LLM generation.
    worker_strategy_files = {}
    if isinstance(training_results, dict):
        worker_strategy_files = training_results.get("strategy_files") or {}
    worker_strategy_files = (
        worker_strategy_files if isinstance(worker_strategy_files, dict) else {}
    )

    worker_has_bundle = all(
        isinstance(worker_strategy_files.get(k), str) and worker_strategy_files.get(k).strip()
        for k in ("config.json", "strategy.py", "risk_manager.py", "backtest.py")
    )
    allow_llm_regeneration = not worker_has_bundle

    if worker_has_bundle:
        await _publish_progress(
            build_id,
            {
                "phase": "building_docker",
                "iteration": iteration,
                "message": "Writing worker-generated strategy files...",
                "tokens_consumed": build.tokens_consumed,
                "iteration_count": build.iteration_count,
                "max_iterations": max_iterations,
            },
        )
        # Write all worker-provided files into the output dir (defensive filtering to
        # avoid path traversal).
        for filename, content in worker_strategy_files.items():
            if not isinstance(filename, str) or not filename.strip():
                continue
            if "/" in filename or "\\" in filename:
                continue
            if not isinstance(content, str):
                continue
            (output_path / filename).write_text(content)

        logger.info(
            "Wrote %d worker strategy_files (build_id=%s, iteration_id=%s)",
            len(worker_strategy_files),
            build_id,
            iteration_id,
        )

    # Generate (or load) config.json
    await _publish_progress(
        build_id,
        {
            "phase": "building_docker",
            "iteration": iteration,
            "message": "Generating strategy configuration...",
            "tokens_consumed": build.tokens_consumed,
            "iteration_count": build.iteration_count,
            "max_iterations": max_iterations,
        },
    )

    worker_config_json = (
        worker_strategy_files.get("config.json")
        if isinstance(worker_strategy_files.get("config.json"), str)
        else None
    )
    if worker_config_json and worker_config_json.strip():
        config_json = worker_config_json
        logger.info(
            "Using worker-provided config.json (build_id=%s, iteration_id=%s)",
            build_id,
            iteration_id,
        )
    else:
        config_json = await orchestrator.generate_strategy_config(
            strategy_name=strategy_name,
            symbols=strategy_symbols,
            design=design,
            training_results=training_results,
            strategy_type=strategy_type or "",
            timeframe=timeframe,
        )

    # Parse config for downstream use and inject build_info metadata
    try:
        config = json.loads(config_json)
    except json.JSONDecodeError:
        config = {}

    # Defense-in-depth: normalize percent fields to decimal fractions (0-1)
    # even if the LLM produced whole-percent values.
    try:
        config = orchestrator.normalize_config_percent_fractions(config)
    except Exception:
        # Never fail a build solely due to normalization.
        pass

    # Inject build_info metadata (iteration_number is 1-indexed)
    config["build_info"] = {
        "build_id": build_id,
        "iteration_id": iteration_id,
        "iteration_number": iteration + 1,
        "strategy_uuid": build.strategy_id,  # strategy UUID for full traceability
    }

    # Force-override critical config fields (defense-in-depth)
    config["strategy_name"] = strategy_name
    # Also fix the nested strategy.name — the worker uses its internal job ID here.
    if "strategy" not in config:
        config["strategy"] = {}
    config["strategy"]["name"] = strategy_name
    if "trading" not in config:
        config["trading"] = {}
    config["trading"]["symbols"] = strategy_symbols
    config["trading"]["timeframe"] = timeframe
    # Write once with injected metadata + corrected values
    (output_path / "config.json").write_text(json.dumps(config, indent=2))

    # If we didn't get the full worker bundle, generate the remaining core files.
    if not worker_has_bundle:
        # Generate strategy.py
        await _publish_progress(
            build_id,
            {
                "phase": "building_docker",
                "iteration": iteration,
                "message": "Generating strategy code...",
                "tokens_consumed": build.tokens_consumed,
                "iteration_count": build.iteration_count,
                "max_iterations": max_iterations,
            },
        )
        strategy_code = await orchestrator.generate_strategy_code(
            strategy_name=strategy_name,
            symbols=strategy_symbols,
            design=design,
            training_results=training_results,
            strategy_type=strategy_type or "",
            config=config,
        )
        (output_path / "strategy.py").write_text(strategy_code)

        # Generate risk_manager.py
        await _publish_progress(
            build_id,
            {
                "phase": "building_docker",
                "iteration": iteration,
                "message": "Generating risk manager...",
                "tokens_consumed": build.tokens_consumed,
                "iteration_count": build.iteration_count,
                "max_iterations": max_iterations,
            },
        )
        risk_manager_code = await orchestrator.generate_risk_manager_code(
            strategy_name=strategy_name,
            design=design,
            training_results=training_results,
            strategy_type=strategy_type or "",
            config=config,
        )
        (output_path / "risk_manager.py").write_text(risk_manager_code)

        # Generate backtest.py
        await _publish_progress(
            build_id,
            {
                "phase": "building_docker",
                "iteration": iteration,
                "message": "Generating backtest code...",
                "tokens_consumed": build.tokens_consumed,
                "iteration_count": build.iteration_count,
                "max_iterations": max_iterations,
            },
        )
        backtest_code = await orchestrator.generate_backtest_code(
            strategy_name=strategy_name,
            symbols=strategy_symbols,
            design=design,
            training_results=training_results,
            strategy_type=strategy_type or "",
            config=config,
        )
        (output_path / "backtest.py").write_text(backtest_code)

    # Pre-seed backtest_results.json from the worker's training results.
    # The worker already ran a full backtest during training; these results are
    # authoritative.  We normalise the worker's raw field names (total_return,
    # win_rate, max_drawdown) to the *_pct schema expected by all downstream
    # consumers (comparison logic, DB persistence, UI display).  The
    # verification subprocess below may overwrite this file if it succeeds and
    # produces better-structured output — but the pre-seed guarantees a valid
    # file is always present even if the subprocess fails.
    if worker_has_bundle and training_results:
        worker_backtest = training_results.get("backtest_results")
        if worker_backtest and isinstance(worker_backtest, dict):
            # Map worker raw field names → _pct schema.  Fields that are
            # already absent in the worker output get safe zero defaults so
            # all required schema keys are always present.
            total_trades = int(worker_backtest.get("total_trades", 0))
            win_rate_raw = float(worker_backtest.get("win_rate", 0.0))
            winning_trades = round(total_trades * win_rate_raw / 100) if total_trades else 0

            seeded = {
                "strategy_id": build.strategy_id,
                "strategy_name": strategy_name,
                # _pct schema fields
                "total_return_pct": float(worker_backtest.get("total_return", 0.0)),
                "max_drawdown_pct": float(worker_backtest.get("max_drawdown", 0.0)),
                "win_rate_pct": win_rate_raw,
                "sharpe_ratio": float(worker_backtest.get("sharpe_ratio", 0.0)),
                "profit_factor": float(worker_backtest.get("profit_factor", 0.0)),
                # Trade statistics
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": total_trades - winning_trades,
                "avg_trade_pnl": float(worker_backtest.get("avg_trade_pnl", 0.0)),
                "best_trade_pnl": float(worker_backtest.get("best_trade_pnl", 0.0)),
                "worst_trade_pnl": float(worker_backtest.get("worst_trade_pnl", 0.0)),
                "avg_trade_duration_hours": float(
                    worker_backtest.get("avg_trade_duration_hours", 0.0)
                ),
                # Arrays — worker may not provide these; default to empty lists
                "trades": worker_backtest.get("trades", []),
                "equity_curve": worker_backtest.get("equity_curve", []),
                "daily_returns": worker_backtest.get("daily_returns", []),
                "weekly_returns": worker_backtest.get("weekly_returns", []),
                "monthly_returns": worker_backtest.get("monthly_returns", []),
                "yearly_returns": worker_backtest.get("yearly_returns", []),
            }

            seed_path = output_path / "backtest_results.json"
            seed_path.write_text(json.dumps(seeded, indent=2, default=str))
            logger.info(
                "Pre-seeded backtest_results.json from worker training results "
                "(total_return_pct=%.2f, total_trades=%d, build_id=%s)",
                seeded["total_return_pct"],
                seeded["total_trades"],
                build_id,
            )

    # Step 4: Verification backtest
    # When the worker provided the full file bundle it has already validated its
    # own generated code during training.  Skip the subprocess comparison — it
    # only applies to LLM-generated code where we need an independent check.
    if worker_has_bundle:
        verification_passed = True
        iteration_logs.append(
            "Verification skipped — worker-provided bundle is pre-validated."
        )
        logger.info(
            "Verification skipped for worker-provided bundle (build_id=%s)",
            build_id,
        )

    # Verification backtest loop (max 3 retries) — only for LLM-generated code.
    # worker_has_bundle builds already set verification_passed = True above.
    if not worker_has_bundle:
        max_verification_retries = 3
        verification_passed = False
    else:
        max_verification_retries = 0  # loop body never executes


    for attempt in range(1, max_verification_retries + 1):
        await _publish_progress(build_id, {
            "phase": "building_docker",
            "iteration": iteration,
            "message": f"Running verification backtest (attempt {attempt}/{max_verification_retries})...",
            "tokens_consumed": build.tokens_consumed,
            "iteration_count": build.iteration_count,
            "max_iterations": max_iterations,
        })

        # Run backtest.py as subprocess
        env = os.environ.copy()
        env["BACKTEST_MODE"] = "true"
        env["BACKTEST_PERIOD"] = "1Y"
        env["STRATEGY_ID"] = build.strategy_id
        env["LOG_LEVEL"] = "WARNING"

        process = await asyncio.create_subprocess_exec(
            sys.executable, "backtest.py",
            cwd=strategy_output_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            iteration_logs.append(f"Verification backtest timed out (attempt {attempt})")
            if attempt < max_verification_retries and allow_llm_regeneration:
                # Regenerate strategy.py with timeout feedback
                strategy_code = await orchestrator.generate_strategy_code(
                    strategy_name=strategy_name,
                    symbols=strategy_symbols,
                    design=design,
                    training_results=training_results,
                    strategy_type=strategy_type or "",
                    config=config,
                )
                (output_path / "strategy.py").write_text(strategy_code)
            continue

        if process.returncode != 0:
            error_output = stderr.decode() if stderr else "Unknown error"
            iteration_logs.append(f"Verification backtest failed (attempt {attempt}): {error_output[:500]}")
            if attempt < max_verification_retries and allow_llm_regeneration:
                strategy_code = await orchestrator.generate_strategy_code(
                    strategy_name=strategy_name,
                    symbols=strategy_symbols,
                    design=design,
                    training_results=training_results,
                    strategy_type=strategy_type or "",
                    config=config,
                )
                (output_path / "strategy.py").write_text(strategy_code)
            continue

        # Read backtest results
        results_file = output_path / "backtest_results.json"
        if results_file.exists():
            try:
                generated_results = json.loads(results_file.read_text())
            except json.JSONDecodeError:
                iteration_logs.append(f"Verification: invalid JSON in backtest_results.json (attempt {attempt})")
                continue

            # Force-override identity fields in backtest results
            generated_results["strategy_id"] = build.strategy_id
            generated_results["strategy_name"] = strategy_name
            # Write corrected results back
            results_file.write_text(json.dumps(generated_results, indent=2, default=str))

            # Compare with training results
            comparison = orchestrator.compare_backtest_results(generated_results, training_results)

            if comparison["passed"]:
                verification_passed = True
                iteration_logs.append(f"Verification passed (attempt {attempt}): {comparison['details']}")
                break
            else:
                iteration_logs.append(f"Verification failed (attempt {attempt}): {comparison['details']}")
                if attempt < max_verification_retries and allow_llm_regeneration:
                    # Regenerate strategy with feedback
                    strategy_code = await orchestrator.generate_strategy_code(
                        strategy_name=strategy_name,
                        symbols=strategy_symbols,
                        design=design,
                        training_results=training_results,
                        strategy_type=strategy_type or "",
                        config=config,
                    )
                    (output_path / "strategy.py").write_text(strategy_code)
        else:
            iteration_logs.append(f"Verification: no backtest_results.json produced (attempt {attempt})")
            if attempt < max_verification_retries and allow_llm_regeneration:
                backtest_code = await orchestrator.generate_backtest_code(
                    strategy_name=strategy_name,
                    symbols=strategy_symbols,
                    design=design,
                    training_results=training_results,
                    strategy_type=strategy_type or "",
                    config=config,
                )
                (output_path / "backtest.py").write_text(backtest_code)

    if not verification_passed:
        iteration_logs.append("Verification: all attempts failed, proceeding with best available code")

    # Step 5: Generate README
    try:
        await _publish_progress(build_id, {
            "phase": "building_docker",
            "iteration": iteration,
            "message": "Generating README...",
            "tokens_consumed": build.tokens_consumed,
            "iteration_count": build.iteration_count,
            "max_iterations": max_iterations,
        })
        readme = await orchestrator.generate_strategy_readme(
            strategy_name=strategy_name,
            symbols=strategy_symbols,
            design=design,
            backtest_results=training_results.get("backtest_results", {}),
            model_metrics=training_results.get("model_metrics", {}),
            config=config,
            strategy_type=strategy_type or "",
        )
        # Write to filesystem for Docker build
        (output_path / "README.md").write_text(readme)
        # Save to database for production retrieval
        build.readme = readme
        await bg_db.commit()
    except Exception as e:
        logger.warning("README generation failed: %s", e)

    # Step 6: Safeguard - Re-copy main.py template to prevent any accidental overwrites
    # This ensures the 428-line template is always intact before Docker build
    try:
        main_py_template = template_dir / "main.py"
        main_py_output = output_path / "main.py"
        await asyncio.to_thread(shutil.copy2, str(main_py_template), str(main_py_output))

        # Verify the copied main.py
        main_py_size = await asyncio.to_thread(os.path.getsize, str(main_py_output))
        main_py_content = await asyncio.to_thread(main_py_output.read_text)
        has_strategy_runner = "class StrategyRunner" in main_py_content

        logger.info(
            f"main.py verified: {main_py_size} bytes, StrategyRunner class present: {has_strategy_runner}"
        )

        # If verification fails, log warning but don't fail the build
        if main_py_size < 10000 or not has_strategy_runner:
            logger.warning(
                f"main.py verification failed: size={main_py_size} bytes (expected >10KB), "
                f"has_StrategyRunner={has_strategy_runner}"
            )
    except Exception as e:
        logger.warning("main.py safeguard copy/verification failed: %s", e)

    return strategy_output_dir


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
    start_iteration: int = 0,
    prior_iteration_history: list | None = None,
    customizations: str = "",
    thoughts: str = "",
):
    """
    Background task that runs the full build iteration loop.

    Args:
        start_iteration: Iteration number to start from (default 0). Used for resuming builds.
        customizations: Required instructions that the AI must follow.
        thoughts: Optional considerations for the AI to think about.

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

            # When resuming (start_iteration > 0), load the last completed iteration's data
            if start_iteration > 0:
                last_iter_num = start_iteration - 1
                result = await bg_db.execute(
                    select(BuildIteration).where(
                        (BuildIteration.build_id == build_id) &
                        (BuildIteration.iteration_number == last_iter_num)
                    )
                )
                last_iteration = result.scalar_one_or_none()
                if last_iteration and last_iteration.llm_design:
                    design = last_iteration.llm_design
                    logger.info(f"Resuming build {build_id} from iteration {start_iteration}, loaded previous design")
                    # Load backtest results as training_results for continuity.
                    # IMPORTANT: include strategy_files so that on restart the build
                    # uses the worker-generated files rather than re-generating them
                    # via the LLM (which can corrupt percentage values, etc.).
                    if last_iteration.backtest_results:
                        training_results = {
                            "backtest_results": last_iteration.backtest_results,
                            "best_model": last_iteration.best_model.get("name") if last_iteration.best_model else "Unknown",
                            "strategy_files": last_iteration.strategy_files or {},
                        }
                    else:
                        # Initialize empty training_results if no backtest results available
                        training_results = {
                            "backtest_results": {},
                            "best_model": "Unknown",
                            "strategy_files": {},
                        }

            # Iteration loop: design → train → refine → repeat
            # Starts from start_iteration to support resuming stopped/failed builds
            async def check_stop_signal():
                """Check if build should be stopped. Returns True if stop requested."""
                stop_client = None
                try:
                    stop_client = await redis_async.from_url(settings.REDIS_URL)
                    stop_flag = await stop_client.get(f"oculus:build:stop:{build_id}")
                    return bool(stop_flag)
                except Exception as e:
                    logger.warning(f"Failed to check stop signal: {e}")
                    return False
                finally:
                    if stop_client:
                        try:
                            await stop_client.close()
                        except Exception:
                            pass  # Ignore cleanup errors

            for iteration in range(start_iteration, max_iterations):
                # Check for stop signal at start of iteration
                try:
                    if await check_stop_signal():
                        build.status = "stopped"
                        build.phase = "stopped"
                        build.completed_at = datetime.utcnow()
                        build.logs = json.dumps({
                            "message": "Build stopped by user",
                            "iterations_completed": iteration,
                            "iteration_logs": iteration_logs,
                            "timeframe": timeframe,
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

                # Check balance before starting iteration (fail fast)
                has_balance = await check_balance(
                    bg_db,
                    user_id,
                    settings.TOKENS_PER_ITERATION
                )
                if not has_balance:
                    # Insufficient balance — stop build gracefully
                    build.status = "stopped"
                    build.phase = "insufficient_tokens"
                    build.completed_at = datetime.utcnow()
                    build.logs = json.dumps({
                        "error": "Insufficient token balance",
                        "message": f"Build stopped at iteration {iteration + 1}. Please purchase more tokens to continue.",
                        "iterations_completed": iteration,
                        "tokens_consumed": build.tokens_consumed,
                        "timeframe": timeframe,
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

                # Create or reuse BuildIteration record for this iteration
                # When restarting, an iteration record may already exist
                iter_start_time = time.time()
                result = await bg_db.execute(
                    select(BuildIteration).where(
                        (BuildIteration.build_id == build_id) &
                        (BuildIteration.iteration_number == iteration)
                    )
                )
                iteration_record = result.scalar_one_or_none()

                if iteration_record:
                    # Reuse existing iteration record and reset its status
                    iteration_record.status = "pending"
                    iteration_record.duration_seconds = None
                    iteration_record.updated_at = datetime.utcnow()
                else:
                    # Create new iteration record
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

                # Set build status to "building" for all iterations (not just iteration 0)
                # Only set to "building" if not already "stopping"
                if build.status != "stopping":
                    build.status = "building"
                    await bg_db.commit()

                if iteration == 0:
                    # First iteration: initial design
                    build.phase = "designing"
                    iteration_record.status = "designing"
                    await bg_db.commit()

                    # Save checkpoint before designing
                    await CheckpointManager.save_checkpoint(
                        bg_db, build_id, "designing", iteration=iteration
                    )

                    await _publish_progress(build_id, {
                        "phase": "designing",
                        "status": "building",  # Include status to update frontend
                        "iteration": iteration,
                        "message": "Designing initial strategy...",
                        "tokens_consumed": build.tokens_consumed,
                        "iteration_count": build.iteration_count,
                        "max_iterations": max_iterations,
                    })
                    await _save_progress_message(
                        bg_db, user_id, strategy_id,
                        "Designing initial strategy...",
                        "designing",
                        iteration
                    )

                    # Check stop signal before LLM call
                    if await check_stop_signal():
                        build.status = "stopped"
                        build.phase = "stopped"
                        build.completed_at = datetime.utcnow()
                        iteration_record.status = "stopped"
                        await bg_db.commit()
                        logger.info("Build %s stopped before LLM design", build_id)
                        return

                    llm_result = await orchestrator.design_strategy(
                        strategy_name=strategy_name,
                        symbols=strategy_symbols,
                        description=description,
                        timeframe=timeframe,
                        target_return=target_return,
                        strategy_type=strategy_type,
                        iteration=iteration,
                        prior_iteration_history=prior_iteration_history,
                        customizations=customizations,
                        thoughts=thoughts,
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
                            description, timeframe, target_return, "", "",
                            customizations=customizations, thoughts=thoughts
                        ),
                        json.dumps(design)
                    )
                    # Save thinking content to ChatHistory (split into separate records)
                    thinking_blocks = llm_result.get("thinking_blocks", [])
                    if not thinking_blocks:
                        # Backwards compat: fall back to single concatenated string
                        if thinking_text:
                            thinking_blocks = [thinking_text]
                    for block in thinking_blocks:
                        await _save_thinking_history(bg_db, user_id, strategy_id, block, iteration)

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
                        "status": "building",  # Include status to update frontend
                        "iteration": iteration,
                        "message": f"Refining strategy (iteration {iteration + 1})...",
                        "tokens_consumed": build.tokens_consumed,
                        "iteration_count": build.iteration_count,
                        "max_iterations": max_iterations,
                    })
                    await _save_progress_message(
                        bg_db, user_id, strategy_id,
                        f"Refining strategy (iteration {iteration + 1})...",
                        "refining",
                        iteration
                    )

                    # Build iteration history from completed iterations (Item 5)
                    iter_history_result = await bg_db.execute(
                        sa_select(BuildIteration)
                        .where(
                            BuildIteration.build_id == build_id,
                            BuildIteration.status == "complete",
                            BuildIteration.iteration_number < iteration,
                        )
                        .order_by(BuildIteration.iteration_number)
                    )
                    past_iterations = iter_history_result.scalars().all()
                    iteration_history = []
                    for past_iter in past_iterations:
                        bt = past_iter.backtest_results or {}
                        # Extract feature names from entry/exit rules
                        entry_feats = [r.get("feature", "") for r in (past_iter.entry_rules or [])]
                        exit_feats = [r.get("feature", "") for r in (past_iter.exit_rules or [])]
                        # Extract top feature names
                        feats = past_iter.features or {}
                        top_feat_names = [
                            f.get("name", f.get("feature", "")) for f in (feats.get("top_features", []) if isinstance(feats, dict) else [])[:5]
                        ]
                        iteration_history.append({
                            "iteration": past_iter.iteration_number,
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

                    # Run refinement + creation guide update in parallel (both are LLM calls)
                    async def _guide_update_safe():
                        """Update creation guide with its own DB session; never raises."""
                        try:
                            async with AsyncSessionLocal() as guide_db:
                                await orchestrator.update_creation_guide(
                                    db=guide_db,
                                    strategy_type=strategy_type,
                                    design=design,
                                    training_results=training_results,
                                    backtest_results=training_results.get("backtest_results", {}),
                                )
                                await guide_db.commit()
                        except Exception as e:
                            logger.warning(f"Creation guide update failed (non-fatal): {e}")

                    refine_task = orchestrator.refine_strategy(
                        previous_design=design,
                        backtest_results=training_results.get("backtest_results", {}),
                        iteration=iteration,
                        strategy_type=strategy_type,
                        training_results=training_results,
                        iteration_history=iteration_history if iteration_history else None,
                        description=description,
                    )
                    guide_task = _guide_update_safe()
                    results = await asyncio.gather(refine_task, guide_task)
                    llm_result = results[0]
                    design = llm_result["design"]
                    thinking_text = llm_result.get("thinking", "")

                    # Store LLM context in iteration record
                    iteration_record.llm_design = design
                    iteration_record.llm_thinking = thinking_text
                    await bg_db.commit()

                    # Save refinement to ChatHistory
                    refinement_prompt = orchestrator._build_refinement_prompt(
                        design, training_results.get("backtest_results", {}), iteration, "", strategy_type or "",
                        training_results=training_results, iteration_history=iteration_history if iteration_history else None,
                    )
                    await _save_chat_history(
                        bg_db, user_id, strategy_id,
                        refinement_prompt,
                        json.dumps(design)
                    )
                    # Save thinking content to ChatHistory (split into separate records)
                    thinking_blocks = llm_result.get("thinking_blocks", [])
                    if not thinking_blocks:
                        # Backwards compat: fall back to single concatenated string
                        if thinking_text:
                            thinking_blocks = [thinking_text]
                    for block in thinking_blocks:
                        await _save_thinking_history(bg_db, user_id, strategy_id, block, iteration)

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

                # Check stop signal before dispatching training
                if await check_stop_signal():
                    build.status = "stopped"
                    build.phase = "stopped"
                    build.completed_at = datetime.utcnow()
                    iteration_record.status = "stopped"
                    await bg_db.commit()
                    logger.info("Build %s stopped before training dispatch", build_id)
                    return

                # Dispatch training job
                build.phase = "training"
                iteration_record.status = "training"
                await bg_db.commit()

                # Save checkpoint before training
                await CheckpointManager.save_checkpoint(
                    bg_db, build_id, "training", iteration=iteration
                )

                await _publish_progress(build_id, {
                    "phase": "training",
                    "status": "building",  # Include status to update frontend
                    "iteration": iteration,
                    "message": f"Training model (iteration {iteration + 1})...",
                    "tokens_consumed": build.tokens_consumed,
                    "iteration_count": build.iteration_count,
                    "max_iterations": max_iterations,
                })
                await _save_progress_message(
                    bg_db, user_id, strategy_id,
                    f"Training model (iteration {iteration + 1})...",
                    "training",
                    iteration
                )

                job_id = await orchestrator.dispatch_training_job(
                    design, strategy_symbols, timeframe,
                    iteration_uuid=iteration_uuid,
                    iteration_number=iteration
                )

                # Listen for training results (with stop signal checking every 5s)
                try:
                    training_results = await orchestrator.listen_for_results(job_id, stop_check_callback=check_stop_signal)
                    iteration_logs.append(f"Iteration {iteration}: Training completed")

                    # Check for invalid symbols error (only on first iteration)
                    if training_results.get("error_type") == "invalid_symbols" and iteration == 0:
                        error_msg = training_results.get("error", "Invalid symbols detected")
                        logger.error(f"Build {build_id} failed due to invalid symbols: {error_msg}")

                        # Mark build and iteration as failed
                        build.status = "failed"
                        build.phase = "validation_failed"
                        build.completed_at = datetime.utcnow()
                        iteration_record.status = "failed"
                        iteration_record.duration_seconds = time.time() - iter_start_time

                        # Store error details
                        build.logs = json.dumps({
                            "error": error_msg,
                            "error_type": "invalid_symbols",
                            "message": "Build failed due to invalid symbol(s). Please check the ticker symbols and try again.",
                            "iteration": iteration,
                            "timeframe": timeframe,
                        })

                        await bg_db.commit()

                        # Publish error to frontend
                        await _publish_progress(build_id, {
                            "phase": "failed",
                            "status": "validation_failed",
                            "error": error_msg,
                            "error_type": "invalid_symbols",
                            "message": f"{error_msg}. Please check the ticker symbols and try again.",
                        })

                        logger.info(f"Build {build_id} terminated due to invalid symbols")
                        return

                    # Diagnostic: log whether strategy_files are present in worker response
                    sf = training_results.get("strategy_files") if isinstance(training_results, dict) else None
                    if sf and isinstance(sf, dict):
                        logger.info(
                            f"Worker returned {len(sf)} strategy_files: {list(sf.keys())} "
                            f"(build_id={build_id}, iteration={iteration})"
                        )
                    else:
                        logger.warning(
                            f"Worker returned NO strategy_files in training_results "
                            f"(build_id={build_id}, iteration={iteration})"
                        )
                except asyncio.CancelledError:
                    # Build stopped during training
                    build.status = "stopped"
                    build.phase = "stopped"
                    build.completed_at = datetime.utcnow()
                    iteration_record.status = "stopped"
                    await bg_db.commit()
                    await _publish_progress(build_id, {
                        "phase": "stopped",
                        "message": "Build stopped by user during training",
                    })
                    logger.info("Build %s stopped during training", build_id)
                    return

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

                # Persist worker-generated strategy files (filename -> source string) for later reuse.
                # This prevents unit drift (e.g. 0.05 -> 5.0) when rebuilding outputs from DB.
                iteration_record.strategy_files = training_results.get("strategy_files")

                # Extract rules from the result
                extracted = training_results.get("extracted_rules", {})
                iteration_record.entry_rules = extracted.get("entry_rules")
                iteration_record.exit_rules = extracted.get("exit_rules")

                await bg_db.commit()

                # Mark iteration as complete (increment AFTER work is persisted)
                build.iteration_count = iteration + 1
                await bg_db.commit()

                # Deduct tokens AFTER iteration completion
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
                        # Edge case: balance changed between check and deduct
                        # Log warning but don't fail the build (iteration already completed)
                        logger.warning(
                            "Build %s iteration %d completed but token deduction failed (balance changed). "
                            "Tokens consumed may be inconsistent.",
                            build_id, iteration + 1
                        )
                    else:
                        raise  # Re-raise non-402 errors

                # Save checkpoint after iteration completes
                await CheckpointManager.save_checkpoint(
                    bg_db, build_id, "iteration_complete", iteration=iteration,
                    data={"total_return": (training_results.get("backtest_results") or {}).get("total_return")}
                )

                # Publish per-iteration training results via Redis
                await _publish_progress(build_id, {
                    "phase": "training_complete",
                    "iteration": iteration,
                    "iteration_uuid": iteration_uuid,
                    "message": f"Training completed (iteration {iteration + 1})",
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
                await _save_progress_message(
                    bg_db, user_id, strategy_id,
                    f"Training completed (iteration {iteration + 1})",
                    "training_complete",
                    iteration
                )

                # Check if target metrics are met
                total_return = (training_results.get('backtest_results') or {}).get('total_return', 0)
                if total_return >= target_return:
                    # --- NEW: Template-based strategy container build ---
                    build.phase = "building_docker"
                    await bg_db.commit()

                    # Save checkpoint before building Docker
                    await CheckpointManager.save_checkpoint(
                        bg_db, build_id, "building_docker", iteration=iteration
                    )

                    await _publish_progress(build_id, {
                        "phase": "building_docker",
                        "iteration": iteration,
                        "message": "Writing algorithm files...",
                        "tokens_consumed": build.tokens_consumed,
                        "iteration_count": build.iteration_count,
                        "max_iterations": max_iterations,
                    })

                    strategy_output_dir = await _build_strategy_container(
                        build_id=build_id,
                        build=build,
                        bg_db=bg_db,
                        orchestrator=orchestrator,
                        strategy_name=strategy_name,
                        strategy_symbols=strategy_symbols,
                        design=design,
                        training_results=training_results,
                        strategy_type=strategy_type or "",
                        iteration=iteration,
                        max_iterations=max_iterations,
                        iteration_logs=iteration_logs,
                        iteration_id=iteration_record.uuid,
                        timeframe=timeframe,
                    )

                    # Transition to testing phase (58.4 will implement the testing logic)
                    build.phase = "testing_algorithm"
                    await bg_db.commit()
                    await _publish_progress(build_id, {
                        "phase": "testing_algorithm",
                        "iteration": iteration,
                        "message": "Testing algorithm container...",
                        "tokens_consumed": build.tokens_consumed,
                        "iteration_count": build.iteration_count,
                        "max_iterations": max_iterations,
                    })

                    # --- Docker build, test, and push ---
                    strat_result = await bg_db.execute(
                        select(Strategy).where(Strategy.uuid == strategy_id)
                    )
                    strategy_obj = strat_result.scalar_one_or_none()
                    docker_builder = DockerBuilder(bg_db)

                    # Build the Docker image
                    image_tag = await docker_builder.build_image(strategy_obj, build, strategy_output_dir)

                    if image_tag:
                        # Test the container
                        test_result = await docker_builder.test_container(image_tag, strategy_output_dir)

                        if test_result["passed"]:
                            # Push to Digital Ocean Container Registry
                            push_success = await docker_builder.push_image(image_tag, build)
                            if push_success:
                                iteration_logs.append(f"Iteration {iteration}: Docker image built, tested, and pushed")
                            else:
                                iteration_logs.append(f"Iteration {iteration}: Docker push failed")
                        else:
                            iteration_logs.append(f"Iteration {iteration}: Container test failed: {test_result.get('error', 'Unknown')}")
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
                    await _save_progress_message(
                        bg_db, user_id, strategy_id,
                        f"Build complete — target achieved ({total_return}% >= {target_return}%)",
                        "complete",
                        iteration
                    )
                    break
                elif iteration < max_iterations - 1:
                    iteration_logs.append(
                        f"Iteration {iteration}: Below target ({total_return}% < {target_return}%), refining..."
                    )

                    # --- Diminishing returns detection (Item 12) ---
                    # If we have 3+ completed iterations, check if performance has plateaued
                    if iteration >= 2:
                        recent_returns = []
                        for past_log in iteration_logs:
                            # Parse returns from logs like "Iteration X: Below target (Y% < Z%)"
                            pass  # We'll check from the DB instead

                        dim_result = await bg_db.execute(
                            sa_select(BuildIteration)
                            .where(
                                BuildIteration.build_id == build_id,
                                BuildIteration.status == "complete",
                            )
                            .order_by(BuildIteration.iteration_number.desc())
                            .limit(3)
                        )
                        recent_iters = dim_result.scalars().all()

                        if len(recent_iters) >= 3:
                            recent_returns = [
                                (it.backtest_results or {}).get("total_return", 0)
                                for it in recent_iters
                            ]
                            # Check if improvement has stalled (< 1% absolute improvement across last 3)
                            max_recent = max(recent_returns)
                            min_recent = min(recent_returns)
                            improvement_range = abs(max_recent - min_recent)

                            if improvement_range < 1.0 and iteration >= 3:
                                logger.info(
                                    "Build %s: diminishing returns detected at iteration %d "
                                    "(last 3 returns: %s, range: %.2f%%)",
                                    build_id, iteration, recent_returns, improvement_range
                                )
                                iteration_logs.append(
                                    f"Iteration {iteration}: Diminishing returns detected "
                                    f"(last 3 returns vary by only {improvement_range:.2f}%) — "
                                    f"stopping early to use best iteration"
                                )
                                await _publish_progress(build_id, {
                                    "phase": "diminishing_returns",
                                    "iteration": iteration,
                                    "message": f"Diminishing returns detected — performance plateaued "
                                               f"(range: {improvement_range:.2f}% across last 3 iterations)",
                                    "tokens_consumed": build.tokens_consumed,
                                    "iteration_count": build.iteration_count,
                                    "max_iterations": max_iterations,
                                })
                                break  # Exit build loop, will fall through to best-iteration selection

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
                    # Composite scoring: weighted combination of return, sharpe, win_rate, drawdown
                    def _composite_score(it) -> float:
                        bt = it.backtest_results or {}
                        total_return = bt.get("total_return", 0)
                        sharpe = bt.get("sharpe_ratio", 0)
                        win_rate = bt.get("win_rate", 0)
                        max_drawdown = abs(bt.get("max_drawdown", 0))
                        total_trades = bt.get("total_trades", 0)

                        # Normalize components to comparable scales
                        # Return: use directly (already percentage)
                        return_score = total_return

                        # Sharpe: scale by 10 to make it comparable to return %
                        sharpe_score = sharpe * 10

                        # Win rate: 50% is break-even, reward above that
                        win_score = (win_rate - 50) * 0.5  # e.g. 60% -> 5 points

                        # Drawdown penalty: higher drawdown = lower score
                        drawdown_penalty = max_drawdown * 0.3  # e.g. 10% drawdown -> -3 points

                        # Trade count bonus: penalize too few trades (unreliable)
                        trade_bonus = min(total_trades / 10, 5)  # max 5 points for 50+ trades

                        return return_score + sharpe_score + win_score - drawdown_penalty + trade_bonus

                    best_iter = max(completed_iterations, key=_composite_score)
                    best_return = (best_iter.backtest_results or {}).get("total_return", 0)
                    best_iter_num = best_iter.iteration_number
                    best_score = _composite_score(best_iter)

                    bt = best_iter.backtest_results or {}
                    best_sharpe = bt.get("sharpe_ratio", 0)
                    best_win_rate = bt.get("win_rate", 0)
                    best_drawdown = bt.get("max_drawdown", 0)
                    iteration_logs.append(
                        f"All {len(completed_iterations)} iterations complete. "
                        f"Best: iteration {best_iter_num} (score={best_score:.1f}, "
                        f"return={best_return}%, sharpe={best_sharpe}, "
                        f"win_rate={best_win_rate}%, drawdown={best_drawdown}%)"
                    )

                    # Publish selection decision via WebSocket
                    await _publish_progress(build_id, {
                        "phase": "selecting_best",
                        "message": f"Selecting best iteration: #{best_iter_num + 1} with {best_return}% return (composite score: {best_score:.1f})",
                        "best_iteration": best_iter_num,
                        "best_return": best_return,
                        "tokens_consumed": build.tokens_consumed,
                        "iteration_count": build.iteration_count,
                        "max_iterations": max_iterations,
                    })
                    await _save_progress_message(
                        bg_db, user_id, strategy_id,
                        f"Selecting best iteration: #{best_iter_num + 1} with {best_return}% return",
                        "selecting_best",
                        None
                    )

                    # Generate selection thinking via Claude
                    try:
                        selection_thinking_result = await orchestrator.generate_selection_thinking(
                            completed_iterations=completed_iterations,
                            best_iteration=best_iter,
                            target_return=target_return,
                            strategy_type=strategy_type or "",
                        )
                        selection_thinking_text = selection_thinking_result.get("thinking", "")
                        selection_thinking_blocks = selection_thinking_result.get("thinking_blocks", [])

                        await _publish_progress(build_id, {
                            "phase": "selecting_best",
                            "message": "Design thinking: iteration selection",
                            "thinking": selection_thinking_text,
                            "tokens_consumed": build.tokens_consumed,
                            "iteration_count": build.iteration_count,
                            "max_iterations": max_iterations,
                        })

                        # Save selection thinking to ChatHistory (split into separate records)
                        if not selection_thinking_blocks:
                            # Backwards compat: fall back to single concatenated string
                            if selection_thinking_text:
                                selection_thinking_blocks = [selection_thinking_text]
                        for block in selection_thinking_blocks:
                            await _save_thinking_history(bg_db, user_id, strategy_id, block, None)
                    except Exception as e:
                        logger.warning("Selection thinking failed for build %s: %s", build_id, e)

                    # Use the best iteration's design and training results for Docker build
                    design = best_iter.llm_design or design
                    # Reconstruct full training_results from BuildIteration columns
                    # (best_iter.backtest_results alone only has performance metrics;
                    # we also need extracted_rules, best_model, features, etc.)
                    training_results = {
                        "status": "success",
                        "backtest_results": best_iter.backtest_results or {},
                        "best_model": best_iter.best_model or {},
                        "extracted_rules": {
                            "entry_rules": best_iter.entry_rules or [],
                            "exit_rules": best_iter.exit_rules or [],
                        },
                        "features": best_iter.features or {},
                        "model_evaluations": best_iter.model_evaluations or {},
                        "model_metrics": (best_iter.best_model or {}).get("metrics", {}),
                        "strategy_files": best_iter.strategy_files or {},
                    }

                    # --- NEW: Template-based strategy container build ---
                    build.phase = "building_docker"
                    await bg_db.commit()
                    await _publish_progress(build_id, {
                        "phase": "building_docker",
                        "iteration": best_iter_num,
                        "message": "Writing algorithm files for best iteration...",
                        "tokens_consumed": build.tokens_consumed,
                        "iteration_count": build.iteration_count,
                        "max_iterations": max_iterations,
                    })

                    strategy_output_dir = await _build_strategy_container(
                        build_id=build_id,
                        build=build,
                        bg_db=bg_db,
                        orchestrator=orchestrator,
                        strategy_name=strategy_name,
                        strategy_symbols=strategy_symbols,
                        design=design,
                        training_results=training_results,
                        strategy_type=strategy_type or "",
                        iteration=best_iter_num,
                        max_iterations=max_iterations,
                        iteration_logs=iteration_logs,
                        iteration_id=best_iter.uuid,
                        timeframe=timeframe,
                    )

                    # Transition to testing phase
                    build.phase = "testing_algorithm"
                    await bg_db.commit()
                    await _publish_progress(build_id, {
                        "phase": "testing_algorithm",
                        "iteration": best_iter_num,
                        "message": "Testing algorithm container for best iteration...",
                        "tokens_consumed": build.tokens_consumed,
                        "iteration_count": build.iteration_count,
                        "max_iterations": max_iterations,
                    })

                    # --- Docker build, test, and push ---
                    strat_result = await bg_db.execute(
                        select(Strategy).where(Strategy.uuid == strategy_id)
                    )
                    strategy_obj = strat_result.scalar_one_or_none()
                    docker_builder = DockerBuilder(bg_db)

                    # Build the Docker image
                    image_tag = await docker_builder.build_image(strategy_obj, build, strategy_output_dir)

                    if image_tag:
                        # Test the container
                        test_result = await docker_builder.test_container(image_tag, strategy_output_dir)

                        if test_result["passed"]:
                            # Push to Docker Hub
                            push_success = await docker_builder.push_image(image_tag, build)
                            if push_success:
                                iteration_logs.append("Docker image built, tested, and pushed for best iteration")
                            else:
                                iteration_logs.append("Docker push failed for best iteration")
                        else:
                            iteration_logs.append(f"Container test failed for best iteration: {test_result.get('error', 'Unknown')}")
                    else:
                        iteration_logs.append("Docker build failed for best iteration")

                    build.status = "complete"
                    build.phase = "complete"
                    await _publish_progress(build_id, {
                        "phase": "complete",
                        "message": f"Build complete — best iteration: #{best_iter_num + 1} ({best_return}% return, target was {target_return}%)",
                        "tokens_consumed": build.tokens_consumed,
                        "iteration_count": build.iteration_count,
                        "max_iterations": max_iterations,
                    })
                    await _save_progress_message(
                        bg_db, user_id, strategy_id,
                        f"Build complete — best iteration: #{best_iter_num + 1} ({best_return}% return, target was {target_return}%)",
                        "complete",
                        None
                    )
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
                "timeframe": timeframe,
            })

            # Update the parent strategy when the build completes successfully.
            # Sets status to "complete" and surfaces the backtest results so the
            # strategy tile can display annual return, max drawdown, and win rate.
            if build.status == "complete":
                strat_update_result = await bg_db.execute(
                    select(Strategy).where(Strategy.uuid == strategy_id)
                )
                strategy_record = strat_update_result.scalar_one_or_none()
                if strategy_record:
                    strategy_record.status = "complete"
                    strategy_record.backtest_results = (training_results or {}).get("backtest_results") or {}
                    strategy_record.updated_at = datetime.utcnow()
                    logger.info(
                        "Strategy %s status updated to 'complete' with backtest_results (build %s)",
                        strategy_id, build_id,
                    )

            await bg_db.commit()

        except asyncio.CancelledError:
            # Task was cancelled (e.g., server shutdown) — update build status and re-raise
            logger.info("Build %s task was cancelled", build_id)
            try:
                build.status = "stopped"
                build.phase = "stopped"
                build.completed_at = datetime.utcnow()
                await bg_db.commit()
            except Exception as commit_error:
                logger.error("Failed to update build status on cancellation: %s", commit_error)
            raise  # Re-raise so the task is properly marked as cancelled

        except Exception as e:
            # Mark build as failed — do NOT crash the background task
            # Log with full traceback so failures are visible in server logs
            current_iteration = locals().get('iteration', 'unknown')
            logger.error(
                "Build %s failed at iteration %s with unhandled exception: %s",
                build_id, current_iteration, e,
                exc_info=True,
            )
            try:
                build.status = "failed"
                build.completed_at = datetime.utcnow()
                build.logs = json.dumps({"error": str(e), "logs": orchestrator.logs, "timeframe": timeframe})
                # Mark current iteration as failed if one was in progress
                if 'iteration_record' in locals() and iteration_record:
                    iteration_record.status = "failed"
                    if 'iter_start_time' in locals():
                        iteration_record.duration_seconds = time.time() - iter_start_time
                await bg_db.commit()
            except Exception as commit_error:
                logger.error("Failed to commit error state for build %s: %s", build_id, commit_error)
        finally:
            # Always clean up the orchestrator (Redis connections, etc.)
            try:
                await orchestrator.cleanup()
            except Exception as cleanup_error:
                logger.error("Error during orchestrator cleanup for build %s: %s", build_id, cleanup_error)


async def start_build_background(
    build_id: str,
    user_id: str,
    strategy_id: str,
    max_iterations: int = 5,
):
    """
    Public function to start a build in the background.
    Used by the scheduler to start queued builds.

    Args:
        build_id: The build UUID
        user_id: The user UUID
        strategy_id: The strategy UUID
        max_iterations: Maximum number of iterations to run
    """
    # Create a new database session for this background task
    async with AsyncSessionLocal() as db:
        try:
            # Fetch the build
            result = await db.execute(
                select(StrategyBuild).where(StrategyBuild.uuid == build_id)
            )
            build = result.scalar_one_or_none()

            if not build:
                logger.error(f"Build {build_id} not found")
                return

            # Fetch the strategy
            strat_result = await db.execute(
                select(Strategy).where(Strategy.uuid == strategy_id)
            )
            strategy = strat_result.scalar_one_or_none()

            if not strategy:
                logger.error(f"Strategy {strategy_id} not found for build {build_id}")
                build.status = "failed"
                build.phase = "complete"
                build.completed_at = datetime.utcnow()
                await db.commit()
                return

            # Extract strategy data
            strategy_name = strategy.name
            raw_symbols = strategy.symbols
            if isinstance(raw_symbols, dict):
                strategy_symbols = raw_symbols.get("symbols", [])
            elif isinstance(raw_symbols, str):
                strategy_symbols = raw_symbols.split()
            elif isinstance(raw_symbols, list):
                strategy_symbols = raw_symbols
            else:
                strategy_symbols = []

            strategy_description = strategy.description or ""
            strat_type = strategy.strategy_type
            target_return = strategy.target_return or 10.0
            timeframe = "1d"  # Default timeframe

            # Clear any existing stop signal
            try:
                redis_client = await redis_async.from_url(settings.REDIS_URL)
                await redis_client.delete(f"oculus:build:stop:{build_id}")
                await redis_client.close()
            except Exception as e:
                logger.error(f"Failed to clear stop signal for build {build_id}: {e}")

            # Launch the build loop as a background task
            task = asyncio.create_task(
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
                    customizations="",
                    thoughts="",
                ),
                name=f"build-{build_id}",
            )
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)

            logger.info(f"Started build {build_id} in background")

        except Exception as e:
            logger.error(f"Failed to start build {build_id}: {e}", exc_info=True)


@router.post("/api/builds/trigger", response_model=BuildResponse)
async def trigger_build(
    strategy_id: str,
    customizations: str = "",
    thoughts: str = "",
    description: str = "",  # Deprecated, kept for backward compatibility
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

    # Create build record with status "building" to ensure dashboard shows it immediately
    # when worker picks it up (worker might start before background task runs)
    build = StrategyBuild(
        strategy_id=strategy_id,
        user_id=current_user.uuid,
        status="building",  # Changed from "queued" to "building"
        phase="initializing",
        started_at=datetime.utcnow(),
        max_iterations=max_iterations,
    )

    db.add(build)
    await db.commit()

    # Capture values needed by the background task before the request session closes
    build_id = build.uuid
    user_id = current_user.uuid
    strategy_name = strategy.name

    # Clear any existing stop signal from previous builds (prevents immediate stops)
    try:
        redis_client = await redis_async.from_url(settings.REDIS_URL)
        deleted_count = await redis_client.delete(f"oculus:build:stop:{build_id}")
        await redis_client.close()
        logger.info(f"Cleared stop signal for build {build_id} (deleted {deleted_count} keys)")
    except Exception as e:
        logger.error(f"Failed to clear stop signal for build {build_id}: {e}", exc_info=True)

    # Backward-compatible symbols guard: if symbols is a dict (old data),
    # extract the "symbols" list from it; if it's already a list, use as-is
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

    strategy_description = description or strategy.description or ""
    strat_type = strategy.strategy_type  # e.g. "Momentum", "Mean Reversion", etc.

    # Add build to queue for tracking and recovery
    try:
        # Get user priority
        user_priority = await PriorityManager.get_user_priority(db, current_user.uuid)
        priority_desc = PriorityManager.get_priority_description(user_priority)
        logger.info(f"Build {build_id} priority: {user_priority} ({priority_desc})")

        redis_client = await redis_async.from_url(settings.REDIS_URL, decode_responses=True)
        queue_manager = BuildQueueManager(redis_client)
        queue_position = await queue_manager.enqueue_build(db, build_id, priority=user_priority)
        logger.info(f"Build {build_id} added to queue at position {queue_position}")
        await redis_client.close()
    except Exception as e:
        logger.error(f"Failed to enqueue build {build_id}: {e}", exc_info=True)
        # Continue anyway - build will still run, just won't be tracked in queue

    # Launch the build loop as a background task.
    # Store a strong reference in _background_tasks to prevent garbage collection.
    # The done callback removes the reference when the task completes.
    task = asyncio.create_task(
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
            customizations=customizations,
            thoughts=thoughts,
        ),
        name=f"build-{build_id}",
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    # Return immediately so the frontend can redirect to the build tracking page
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
            "queue_position": build.queue_position,
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
    Admins can view any build's iterations; regular users can only view their own.
    """
    # Fetch build
    result = await db.execute(
        select(StrategyBuild).where(StrategyBuild.uuid == build_id)
    )
    build = result.scalar_one_or_none()

    if not build:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Build not found"
        )

    # Check ownership (admins can view any build)
    if current_user.user_role != "admin":
        # Fetch strategy to verify ownership for non-admin users
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
    Admins can view any build; regular users can only view their own builds.
    """
    # Fetch build
    result = await db.execute(
        select(StrategyBuild).where(StrategyBuild.uuid == build_id)
    )
    build = result.scalar_one_or_none()

    if not build:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Build not found"
        )

    # Check ownership (admins can view any build)
    if current_user.user_role != "admin" and build.user_id != current_user.uuid:
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
        max_iterations=build.max_iterations,
        started_at=build.started_at,
        completed_at=build.completed_at,
        strategy_type=strategy.strategy_type if strategy else None,
        strategy_name=strategy.name if strategy else None,
        queue_position=build.queue_position,
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
    Admins can stop any build; regular users can only stop their own builds.
    """
    # Fetch build
    result = await db.execute(
        select(StrategyBuild).where(StrategyBuild.uuid == build_id)
    )
    build = result.scalar_one_or_none()

    if not build:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Build not found"
        )

    # Check ownership (admins can stop any build)
    if current_user.user_role != "admin" and build.user_id != current_user.uuid:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Build not found"
        )

    if build.status not in ("building", "queued", "stopping"):
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

    # Update build status to "stopping" in DB
    build.status = "stopping"
    await db.commit()
    await db.refresh(build)

    logger.info("Stop signal sent for build %s", build_id)

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
    )


@router.post("/api/builds/{build_id}/restart", response_model=BuildResponse)
async def restart_build(
    build_id: str,
    max_iterations: int = 5,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Resume a failed or stopped build from where it left off.

    Reuses the same build record and continues from the last completed iteration.
    Only builds with status 'failed' or 'stopped' can be restarted.
    """
    # Fetch original build and validate ownership
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

    if build.status not in ("failed", "stopped"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot restart build with status '{build.status}'. Only 'failed' or 'stopped' builds can be restarted."
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
            (Strategy.uuid == build.strategy_id) & (Strategy.user_id == current_user.uuid)
        )
    )
    strategy = strat_result.scalar_one_or_none()

    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Strategy not found"
        )

    # Reuse the same build record - reset status and clear completed_at
    # Set to "building" immediately so dashboard shows it when worker picks it up
    build.status = "building"  # Changed from "queued" to "building"
    build.phase = "resuming"
    build.completed_at = None
    build.started_at = datetime.utcnow()  # Reset started_at to prevent timeout checks from failing
    # Keep max_iterations from the original build, or update if provided
    if max_iterations != build.max_iterations:
        build.max_iterations = max_iterations
    await db.commit()

    # Capture values for background task
    user_id = current_user.uuid
    strategy_name = strategy.name
    start_iteration = build.iteration_count  # Resume from where we left off

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

    strategy_description = strategy.description or ""
    strat_type = strategy.strategy_type  # e.g. "Momentum", "Mean Reversion", etc.

    # Recover the original timeframe from the build's logs
    recovered_timeframe = "1d"  # fallback default
    try:
        if build.logs:
            logs_data = json.loads(build.logs)
            recovered_timeframe = logs_data.get("timeframe", recovered_timeframe)
    except (json.JSONDecodeError, TypeError):
        pass

    logger.info(f"Resuming build {build_id} from iteration {start_iteration} of {build.max_iterations} (timeframe={recovered_timeframe})")

    # Clear any existing stop signal from previous builds (prevents immediate stops)
    try:
        redis_client = await redis_async.from_url(settings.REDIS_URL)
        deleted_count = await redis_client.delete(f"oculus:build:stop:{build_id}")
        await redis_client.close()
        logger.info(f"Cleared stop signal for build {build_id} (deleted {deleted_count} keys)")
    except Exception as e:
        logger.error(f"Failed to clear stop signal for build {build_id}: {e}", exc_info=True)

    # Launch the build loop as a background task, resuming from last completed iteration.
    # Store a strong reference in _background_tasks to prevent garbage collection.
    # The done callback removes the reference when the task completes.
    task = asyncio.create_task(
        _run_build_loop(
            build_id=build_id,
            strategy_id=build.strategy_id,
            user_id=user_id,
            strategy_name=strategy_name,
            strategy_symbols=strategy_symbols,
            description=strategy_description,
            target_return=strategy.target_return or 10.0,
            timeframe=recovered_timeframe,
            max_iterations=build.max_iterations,
            strategy_type=strat_type,
            start_iteration=start_iteration,  # Resume from here
            customizations="",  # Not available for resumed builds
            thoughts="",  # Not available for resumed builds
        ),
        name=f"build-{build_id}",
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

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


@router.websocket("/api/ws/builds/{build_id}/logs")
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
        last_ping_time = asyncio.get_event_loop().time()
        ping_interval = 30  # Send ping every 30 seconds to keep connection alive
        redis_health_check_interval = 60  # Check Redis health every 60 seconds
        last_redis_health_check = asyncio.get_event_loop().time()

        while True:
            current_time = asyncio.get_event_loop().time()

            # Send periodic ping to keep WebSocket alive
            if current_time - last_ping_time >= ping_interval:
                try:
                    await websocket.send_json({"type": "ping"})
                    last_ping_time = current_time
                except Exception as e:
                    logger.warning(f"Failed to send WebSocket ping for build {build_id}: {e}")
                    break

            # Periodic Redis health check and reconnection
            if current_time - last_redis_health_check >= redis_health_check_interval:
                try:
                    # Use timeout to avoid blocking during long-running training jobs
                    await asyncio.wait_for(redis_client.ping(), timeout=2.0)
                    last_redis_health_check = current_time
                except asyncio.TimeoutError:
                    # Redis is busy (likely processing training), skip health check
                    logger.debug(f"Redis health check timed out for build {build_id} (likely busy with training)")
                    last_redis_health_check = current_time  # Reset timer to avoid repeated timeouts
                except (RedisConnectionError, redis.TimeoutError, OSError) as e:
                    logger.warning(f"Redis health check failed for build {build_id}, reconnecting: {e}")
                    try:
                        # Close old connections
                        if pubsub:
                            await pubsub.unsubscribe()
                            await pubsub.close()
                        if redis_client:
                            await redis_client.close()

                        # Reconnect
                        redis_client = await redis.from_url(settings.REDIS_URL)
                        pubsub = redis_client.pubsub()
                        await pubsub.subscribe(progress_channel)
                        last_redis_health_check = current_time

                        logger.info(f"Successfully reconnected Redis for build {build_id}")
                    except Exception as reconnect_error:
                        logger.error(f"Failed to reconnect Redis for build {build_id}: {reconnect_error}")
                        await websocket.send_json({
                            "type": "error",
                            "message": "Lost connection to build log stream. Please refresh the page.",
                        })
                        break

            # Check for incoming WebSocket messages (ping/pong)
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(), timeout=1.0
                )
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                logger.warning(f"WebSocket receive error for build {build_id}: {e}")
                break

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

                # Lift the human-readable message field to the top level so the
                # frontend's `if (newLog.message)` check picks it up and renders
                # it in the Design Thinking section.
                await websocket.send_json({
                    "type": "progress",
                    "message": progress_data.get("message"),
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


