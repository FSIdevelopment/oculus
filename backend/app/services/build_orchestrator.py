"""Build orchestrator service for strategy design and training."""
import json
import re
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import anthropic
import redis.asyncio as redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

from app.config import settings
from app.models.strategy_build import StrategyBuild
from app.models.strategy_creation_guide import StrategyCreationGuide
from app.models.user import User
from app.services.balance import deduct_tokens
from app.services.build_history_service import BuildHistoryService, FeatureTrackerService


class BuildOrchestrator:
    """Orchestrates strategy build process: LLM design, training, and iteration."""

    def __init__(self, db: AsyncSession, user: User, build: StrategyBuild):
        self.db = db
        self.user = user
        self.build = build
        self.client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.redis_client: Optional[redis.Redis] = None
        self.logs: List[str] = []
        logger.info("BuildOrchestrator initialized")
        logger.info(f"BuildOrchestrator initialized with build: {self.build.uuid}")
        logger.info(f"BuildOrchestrator initialized with user: {self.user.uuid}")

    async def initialize(self):
        """Initialize Redis connection via Ngrok tunnel with connection pooling."""
        self.redis_client = await redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_keepalive=True,
            socket_connect_timeout=10,
            socket_timeout=30,
            retry_on_timeout=True,
            health_check_interval=30,
        )

    async def cleanup(self):
        """Clean up resources."""
        if self.redis_client:
            await self.redis_client.close()

    def _log(self, message: str):
        """Add message to logs."""
        timestamp = datetime.utcnow().isoformat()
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        print(log_entry)

    async def _publish_thinking_delta(self, thinking_content: str, block_number: int, is_complete: bool = False, iteration: int = 0):
        """Publish thinking content (delta or complete) via Redis for WebSocket relay to frontend.

        Args:
            thinking_content: The thinking text to publish
            block_number: Which thinking block this belongs to
            is_complete: True if this is the final content for this block
            iteration: Current iteration number (for clearing thinking blocks between iterations)
        """
        try:
            # Create a fresh Redis connection for this publish (same pattern as _publish_progress)
            redis_client = await redis.from_url(settings.REDIS_URL)

            channel = f"oculus:training:progress:build:{self.build.uuid}"
            message = {
                "type": "thinking",
                "block_number": block_number,
                "content": thinking_content,
                "is_complete": is_complete,
                "iteration": iteration,
                "timestamp": datetime.utcnow().isoformat(),
                "phase": "llm_thinking"
            }

            # Publish to Redis (will be relayed to frontend via WebSocket)
            await redis_client.publish(channel, json.dumps(message))
            await redis_client.close()

            status = "complete" if is_complete else "streaming"
            logger.info(f"✓ Published thinking block {block_number} ({status}, {len(thinking_content)} chars)")
        except Exception as e:
            # Best-effort - don't crash build if publishing fails
            logger.warning(f"Failed to publish thinking delta: {e}")

    async def design_strategy(
        self,
        strategy_name: str,
        symbols: List[str],
        description: str,
        timeframe: str,
        target_return: float,
        strategy_type: Optional[str] = None,
        iteration: int = 0,
        prior_iteration_history: Optional[List[Dict[str, Any]]] = None,
        customizations: str = "",
        thoughts: str = "",
    ) -> Dict[str, Any]:
        """Call Claude API to design strategy.

        Returns a dict with keys:
        - "design": the parsed strategy design JSON
        - "thinking": concatenated thinking block text

        Args:
            customizations: Required instructions that the AI must follow.
            thoughts: Optional considerations for the AI to think about.
        """
        self._log(f"Starting LLM strategy design for {strategy_name}")

        # Get relevant past builds for context
        asset_class = self._infer_asset_class(symbols)
        relevant_builds = await BuildHistoryService.get_relevant_builds(
            self.db, self.user.uuid, symbols, asset_class, max_results=3
        )

        # Build context from past builds
        context = self._format_build_context(relevant_builds)

        # Load creation guide from database
        creation_guide = await self._load_creation_guide(strategy_type)

        # Create prompt
        prompt = self._build_design_prompt(
            strategy_name, strategy_type, symbols, description, timeframe, target_return, context,
            creation_guide, prior_iteration_history=prior_iteration_history,
            customizations=customizations, thoughts=thoughts
        )

        try:
            logger.debug(f"Sending design prompt to Claude: {prompt}")
            response = self.client.messages.create(
                model="claude-opus-4-6",
                max_tokens=settings.CLAUDE_MAX_TOKENS,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )

            # Extract design and thinking from response
            result = await self._parse_design_response(response, iteration)
            desc_preview = result["design"].get("strategy_description", "N/A")[:100]
            self._log(f"Strategy design completed: {desc_preview}")
            return result

        except anthropic.APIError as e:
            self._log(f"Claude API error in design_strategy: {e}")
            logger.error("Claude API error in design_strategy (build=%s): %s", self.build.uuid, e, exc_info=True)
            raise
        except Exception as e:
            self._log(f"Unexpected error in design_strategy: {e}")
            logger.error("Unexpected error in design_strategy (build=%s): %s", self.build.uuid, e, exc_info=True)
            raise

    async def dispatch_training_job(
        self,
        design: Dict[str, Any],
        symbols: List[str],
        timeframe: str,
        iteration_uuid: str = None,
        iteration_number: int = None,
    ) -> str:
        """Dispatch training job to Redis queue if workers are available.

        If no workers are available, the build status will be set to 'waiting_for_worker'
        and a separate scheduler job will dispatch it when capacity becomes available.

        Returns:
            job_id: The job identifier
        """
        from app.services.worker_health import WorkerHealthManager

        # Check worker availability
        worker_manager = WorkerHealthManager()
        workers = await worker_manager.get_active_workers(self.db)
        total_capacity = sum(w.capacity for w in workers)
        active_jobs = sum(w.active_jobs for w in workers)
        available_capacity = total_capacity - active_jobs

        if available_capacity <= 0:
            # No workers available - set status to waiting_for_worker
            self._log("No workers available - entering training queue (waiting_for_worker)")
            self.build.status = "waiting_for_worker"
            self.build.phase = "Waiting for worker availability"
            await self.db.commit()

            # Store job payload in Redis for later dispatch
            job_id = f"build:{self.build.uuid}:iter:{iteration_uuid or 'default'}"
            job_payload = {
                "build_id": self.build.uuid,
                "user_id": self.user.uuid,
                "design": design,
                "symbols": symbols,
                "timeframe": timeframe,
                "timestamp": datetime.utcnow().isoformat(),
                "iteration_uuid": iteration_uuid,
                "iteration_number": iteration_number,
                "source": "oculus",
            }
            await self.redis_client.set(f"pending:{job_id}", json.dumps(job_payload), ex=86400)
            return job_id

        # Workers available - dispatch immediately
        self._log(f"Dispatching training job to Redis (available capacity: {available_capacity})")

        job_payload = {
            "build_id": self.build.uuid,
            "user_id": self.user.uuid,
            "design": design,
            "symbols": symbols,
            "timeframe": timeframe,
            "timestamp": datetime.utcnow().isoformat(),
            "iteration_uuid": iteration_uuid,
            "iteration_number": iteration_number,
            "source": "oculus",
        }

        # Push to Redis queue
        job_id = f"build:{self.build.uuid}:iter:{iteration_uuid or 'default'}"
        # Clean up any stale result from previous runs
        await self.redis_client.delete(f"result:{job_id}")
        await self.redis_client.set(job_id, json.dumps(job_payload), ex=86400)  # 24h TTL
        await self.redis_client.lpush(settings.TRAINING_QUEUE, job_id)

        self._log(f"Job dispatched with ID: {job_id}")
        return job_id

    async def _ensure_redis_connection(self):
        """Ensure Redis connection is healthy, reconnect if needed.

        Note: This uses a timeout on ping to avoid blocking during long-running
        training jobs where Redis may be busy.
        """
        try:
            if self.redis_client is None:
                logger.warning("Redis client is None, reconnecting...")
                self.redis_client = await redis.from_url(
                    settings.REDIS_URL,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_keepalive=True,
                    socket_connect_timeout=10,
                    socket_timeout=30,
                    retry_on_timeout=True,
                    health_check_interval=30,
                )
                return

            # Test connection with ping (with timeout to avoid blocking)
            await asyncio.wait_for(self.redis_client.ping(), timeout=2.0)
        except asyncio.TimeoutError:
            # Redis is busy (likely processing training), connection is probably fine
            logger.debug("Redis ping timed out (likely busy), assuming connection is OK")
            return
        except (redis.ConnectionError, redis.TimeoutError, OSError) as e:
            logger.warning(f"Redis connection lost, reconnecting: {e}")
            try:
                if self.redis_client:
                    await self.redis_client.close()
            except Exception:
                pass
            self.redis_client = await redis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                socket_keepalive=True,
                socket_connect_timeout=10,
                socket_timeout=30,
                retry_on_timeout=True,
                health_check_interval=30,
            )
            # Verify new connection works (with timeout)
            try:
                await asyncio.wait_for(self.redis_client.ping(), timeout=2.0)
            except asyncio.TimeoutError:
                logger.debug("Redis ping timed out after reconnect (likely busy), continuing anyway")

    async def listen_for_results(self, job_id: str, timeout: int = 3600, stop_check_callback=None) -> Dict[str, Any]:
        """Listen for training results from Redis with automatic reconnection.

        Args:
            job_id: Training job identifier
            timeout: Maximum wait time in seconds
            stop_check_callback: Optional async function that returns True if build should stop
        """
        self._log(f"Listening for results on {job_id}")

        result_key = f"result:{job_id}"
        start_time = datetime.utcnow()
        consecutive_errors = 0
        max_consecutive_errors = 10
        last_health_check = datetime.utcnow()
        health_check_interval = 300  # Only check health every 5 minutes to avoid blocking during training

        while True:
            try:
                # Only check Redis health periodically (not on every poll) to avoid blocking
                # during long-running training jobs where Redis may be busy
                elapsed_since_health_check = (datetime.utcnow() - last_health_check).total_seconds()
                if elapsed_since_health_check > health_check_interval:
                    try:
                        # Use a short timeout for health check to avoid blocking
                        await asyncio.wait_for(self.redis_client.ping(), timeout=2.0)
                        last_health_check = datetime.utcnow()
                        logger.debug(f"Redis health check passed for job {job_id}")
                    except asyncio.TimeoutError:
                        # Redis is busy (likely processing training), skip health check
                        logger.debug(f"Redis health check timed out (likely busy with training), continuing to poll")
                        last_health_check = datetime.utcnow()  # Reset timer to avoid repeated timeouts
                    except Exception as e:
                        logger.warning(f"Redis health check failed: {e}, will attempt reconnection")
                        await self._ensure_redis_connection()
                        last_health_check = datetime.utcnow()

                result_data = await self.redis_client.get(result_key)
                if result_data:
                    result = json.loads(result_data)
                    self._log(f"Results received: {result.get('status', 'unknown')}")
                    # Clean up result key to prevent stale data in future re-runs
                    await self.redis_client.delete(result_key)
                    return result

                # Reset error counter on successful poll
                consecutive_errors = 0

            except (redis.ConnectionError, redis.TimeoutError, OSError) as e:
                consecutive_errors += 1
                logger.warning(
                    f"Redis error while polling for results (attempt {consecutive_errors}/{max_consecutive_errors}): {e}"
                )

                if consecutive_errors >= max_consecutive_errors:
                    raise RuntimeError(
                        f"Failed to poll Redis after {max_consecutive_errors} consecutive errors. "
                        f"Last error: {e}"
                    )

                # Exponential backoff: wait longer after each failure
                backoff_delay = min(2 ** consecutive_errors, 30)  # Cap at 30 seconds
                logger.info(f"Waiting {backoff_delay}s before retry...")
                await asyncio.sleep(backoff_delay)

                # Try to reconnect after error
                try:
                    await self._ensure_redis_connection()
                except Exception as reconnect_error:
                    logger.error(f"Failed to reconnect to Redis: {reconnect_error}")

                continue

            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode result JSON for {job_id}: {e}")
                # Clean up corrupted result and raise error
                await self.redis_client.delete(result_key)
                raise RuntimeError(f"Corrupted result data in Redis for job {job_id}")

            # Check for stop signal
            if stop_check_callback and await stop_check_callback():
                logger.info(f"Build stopped while waiting for training results: {job_id}")
                raise asyncio.CancelledError("Build stopped by user")

            # Check timeout
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            if elapsed > timeout:
                raise TimeoutError(f"Training job {job_id} timed out after {timeout}s")

            await asyncio.sleep(5)  # Poll every 5 seconds

    async def refine_strategy(
        self,
        previous_design: Dict[str, Any],
        backtest_results: Dict[str, Any],
        iteration: int,
        strategy_type: Optional[str] = None,
        training_results: Optional[Dict[str, Any]] = None,
        iteration_history: Optional[List[Dict[str, Any]]] = None,
        description: str = "",
    ) -> Dict[str, Any]:
        """Refine strategy based on backtest results.

        Returns a dict with keys:
        - "design": the parsed strategy design JSON
        - "thinking": concatenated thinking block text
        """
        self._log(f"Refining strategy (iteration {iteration})")

        # Load creation guide from database
        creation_guide = await self._load_creation_guide(strategy_type)

        prompt = self._build_refinement_prompt(
            previous_design, backtest_results, iteration, creation_guide, strategy_type or "",
            training_results=training_results, iteration_history=iteration_history,
            description=description,
        )

        try:
            response = self.client.messages.create(
                model="claude-opus-4-6",
                max_tokens=settings.CLAUDE_MAX_TOKENS,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )

            result = await self._parse_design_response(response, iteration)
            self._log(f"Strategy refined (iteration {iteration})")
            return result

        except anthropic.APIError as e:
            self._log(f"Claude API error during refinement (iteration {iteration}): {e}")
            logger.error(
                "Claude API error in refine_strategy (build=%s, iteration=%d): %s",
                self.build.uuid, iteration, e, exc_info=True,
            )
            raise
        except Exception as e:
            self._log(f"Unexpected error during refinement (iteration {iteration}): {e}")
            logger.error(
                "Unexpected error in refine_strategy (build=%s, iteration=%d): %s",
                self.build.uuid, iteration, e, exc_info=True,
            )
            raise

    async def deduct_tokens(self, amount: float, description: str):
        """Deduct tokens from user balance."""
        await deduct_tokens(self.db, self.user.uuid, amount, description)
        self._log(f"Deducted {amount} tokens: {description}")

    def _infer_asset_class(self, symbols: List[str]) -> str:
        """Infer asset class from symbols."""
        asset_class_map = {
            "semiconductors": {"NVDA", "AMD", "AVGO", "TSM", "QCOM"},
            "tech": {"AAPL", "MSFT", "GOOGL", "META", "AMZN"},
            "energy": {"XLE", "XOP", "CVX", "XOM"},
            "index": {"SPY", "QQQ", "IWM", "DIA"},
        }

        symbol_set = set(s.upper() for s in symbols)
        for asset_class, class_symbols in asset_class_map.items():
            if symbol_set & class_symbols:
                return asset_class
        return "mixed"

    def _format_build_context(self, builds: List) -> str:
        """Format past builds as context for Claude."""
        if not builds:
            return "No previous builds available."

        lines = []
        for b in builds:
            lines.append(f"- {b.strategy_name}: {b.asset_class}, symbols={b.symbols}")
            if b.backtest_results:
                lines.append(f"  Return: {b.backtest_results.get('total_return', 'N/A')}%")

        return "\n".join(lines)

    # Path to the default creation guide markdown (sections 1-5 seed the "general" row)
    _GUIDE_MD_PATH = Path(__file__).resolve().parent.parent.parent.parent / "strategy_designer_ai" / "STRATEGY_CREATION_GUIDE.md"

    @staticmethod
    def _extract_general_sections(md_text: str) -> str:
        """Extract sections 1-5 from STRATEGY_CREATION_GUIDE.md (everything before section 6)."""
        marker = "## 6."
        idx = md_text.find(marker)
        if idx == -1:
            # No section 6 found — return the whole file
            return md_text.strip()
        return md_text[:idx].strip()

    async def _load_creation_guide(self, strategy_type: Optional[str]) -> str:
        """Load creation guide content from the database for inclusion in LLM prompts.

        Queries two rows from StrategyCreationGuide:
        1. strategy_type = "general" — cross-cutting lessons for all strategies
        2. strategy_type = <build's type> — type-specific knowledge (e.g. "Momentum")

        If the "general" row doesn't exist, seeds it from STRATEGY_CREATION_GUIDE.md sections 1-5.
        If the strategy-type row doesn't exist, uses empty string (Task 44.5 will create it).

        Returns formatted guide text ready for prompt inclusion, or empty string if nothing available.
        """
        # --- Fetch "general" guide row ---
        result = await self.db.execute(
            select(StrategyCreationGuide).where(
                StrategyCreationGuide.strategy_type == "general"
            )
        )
        general_row = result.scalar_one_or_none()

        # Seed "general" row from markdown file if it doesn't exist
        if general_row is None:
            try:
                if self._GUIDE_MD_PATH.exists():
                    md_content = self._GUIDE_MD_PATH.read_text()
                    general_content = self._extract_general_sections(md_content)
                    if general_content:
                        general_row = StrategyCreationGuide(
                            strategy_type="general",
                            content=general_content,
                            version=1,
                        )
                        self.db.add(general_row)
                        await self.db.flush()
            except Exception as e:
                self._log(f"Warning: Could not seed general creation guide: {e}")

        general_text = general_row.content if general_row else ""

        # --- Fetch strategy-type-specific guide row ---
        type_text = ""
        if strategy_type and strategy_type != "general":
            result = await self.db.execute(
                select(StrategyCreationGuide).where(
                    StrategyCreationGuide.strategy_type == strategy_type
                )
            )
            type_row = result.scalar_one_or_none()
            if type_row:
                type_text = type_row.content

        # --- Concatenate: general first, then strategy-specific ---
        parts = []
        if general_text:
            parts.append(general_text)
        if type_text:
            parts.append(f"\n### Strategy-Type-Specific Knowledge ({strategy_type})\n\n{type_text}")

        if not parts:
            return ""

        combined = "\n\n".join(parts)
        return (
            "\n\n## Strategy Creation Guide (Institutional Knowledge)\n"
            "The following guide contains proven patterns and settings from "
            "previously successful strategies. Follow these lessons:\n\n"
            f"{combined}\n"
        )

    # Available feature names produced by the ML pipeline.
    # Claude must select from these exact names for priority_features and rules.
    AVAILABLE_FEATURES = [
        # Momentum indicators
        "RSI_7", "RSI_14", "RSI_21",
        "STOCH_K_14", "STOCH_D_14", "STOCH_K_21", "STOCH_D_21",
        "WILLIAMS_R",
        "MACD", "MACD_SIGNAL", "MACD_HIST",
        "ROC_5", "ROC_10", "ROC_20",
        "MFI", "CCI",
        # Trend indicators
        "EMA_8", "EMA_13", "EMA_21", "EMA_50", "EMA_100", "EMA_200",
        "PRICE_VS_EMA_8", "PRICE_VS_EMA_13", "PRICE_VS_EMA_21",
        "PRICE_VS_EMA_50", "PRICE_VS_EMA_100", "PRICE_VS_EMA_200",
        "ADX", "PLUS_DI", "MINUS_DI", "DI_DIFF",
        # Volatility indicators
        "ATR", "ATR_PCT",
        "BB_UPPER", "BB_LOWER", "BB_PCT_B", "BB_WIDTH",
        "KC_UPPER", "KC_LOWER", "KC_PCT",
        # Volume indicators
        "VOLUME_RATIO", "OBV_DIFF", "CMF", "PRICE_VS_VWAP",
        # Pattern/change features
        "RSI_14_CHG_1", "STOCH_K_14_CHG_1", "MACD_HIST_CHG_1",
        "CCI_CHG_1", "MFI_CHG_1", "ADX_CHG_1",
        "RSI_14_CHG_3", "STOCH_K_14_CHG_3", "MACD_HIST_CHG_3",
        # Macro features
        "HIGH_52W_PCT", "LOW_52W_PCT",
        "MOM_1M", "MOM_2M", "MOM_3M", "MOM_6M", "MOM_1Y",
        # Crossover/signal features
        "EMA_CROSS_8_21", "EMA_CROSS_21_50", "EMA_CROSS_50_200",
        "MACD_CROSS", "STOCH_CROSS",
        "RSI_BULLISH_DIV", "RSI_BEARISH_DIV",
        "BB_SQUEEZE", "BREAKOUT_20D", "BREAKOUT_50D",
        "GAP_SIGNAL", "CONSEC_UP", "CONSEC_DOWN",
        "EMA_STACK_SCORE", "PRICE_ACCEL", "TREND_SLOPE",
        "HIGHER_HIGH", "HIGHER_LOW",
        "RSI_OVERSOLD_ZONE", "RSI_OVERBOUGHT_ZONE",
        "STOCH_OVERSOLD_ZONE", "STOCH_OVERBOUGHT_ZONE",
        # Feature interaction terms (cross-indicator signals)
        "RSI_VOL_INTERACTION", "ADX_MOM_INTERACTION",
        "BB_RSI_INTERACTION", "VOLUME_PRICE_CONVICTION",
        "TREND_QUALITY", "STOCH_CCI_OVERSOLD",
        "MACD_RSI_CONFIRM",
    ]

    # Full JSON schema template for the LLM output format.
    DESIGN_JSON_SCHEMA = """{
  "strategy_description": "Brief description of the strategy approach",
  "strategy_rationale": "Detailed rationale for why this strategy should work for these symbols",
  "forward_returns_days_options": [5, 10, 15],
  "profit_threshold_options": [2.0, 3.0, 5.0],
  "recommended_n_iter_search": 50,
  "priority_features": ["RSI_14", "MACD_HIST", "ADX", "BB_PCT_B"],
  "feature_rationale": {
    "RSI_14": "Why this feature matters for this strategy",
    "MACD_HIST": "..."
  },
  "entry_rules": [
    {
      "feature": "RSI_14",
      "operator": "<=",
      "threshold": 35.0,
      "description": "Enter when RSI indicates oversold conditions"
    }
  ],
  "entry_score_threshold": 3,
  "exit_rules": [
    {
      "feature": "RSI_14",
      "operator": ">=",
      "threshold": 70.0,
      "description": "Exit when RSI indicates overbought conditions"
    }
  ],
  "exit_score_threshold": 3,
  "recommended_stop_loss": 0.05,
  "recommended_trailing_stop": 0.03,
  "recommended_max_positions": 3,
  "recommended_position_size": 0.10,
  "lookback_window": 200,
  "ml_training_config": {
    "config_search_model": "LightGBM",
    "train_neural_networks": true,
    "train_lstm": true,
    "cv_folds": 3,
    "nn_epochs": 100,
    "nn_batch_size": 64,
    "nn_learning_rate": 0.001,
    "nn_hidden_layers": [[128, 64], [256, 128, 64]],
    "lstm_hidden_size": 128,
    "lstm_num_layers": 2,
    "lstm_dropout": 0.2,
    "lstm_sequence_length": 100,
    "lstm_epochs": 250,
    "lstm_batch_size": 32
  }
}"""

    def _build_design_prompt(
        self,
        strategy_name: str,
        strategy_type: str,
        symbols: List[str],
        description: str,
        timeframe: str,
        target_return: float,
        context: str,
        creation_guide: str = "",
        prior_iteration_history: Optional[List[Dict[str, Any]]] = None,
        customizations: str = "",
        thoughts: str = "",
    ) -> str:
        """Build initial strategy design prompt with full features list, JSON schema, and rules."""
        features_list = "\n".join(f"  - {f}" for f in self.AVAILABLE_FEATURES)

        # Build prior iteration history section if retrain context is available.
        # Entries arrive pre-sorted: oldest build first, iterations within each
        # build in ascending order (guaranteed by the strategies.py retrain endpoint).
        # We assign sequential numbers (Build #1, Build #2, …) so the LLM can refer
        # to specific prior builds without having to parse raw UUIDs.
        prior_history_section = ""
        if prior_iteration_history:
            prior_history_section = "\n## Prior Build History (from previous builds — DO NOT repeat failed approaches)\n"
            prior_history_section += "This strategy has been built before. Builds are listed oldest → newest.\n"
            prior_history_section += "Learn from what worked and what failed. Build on the best results.\n\n"

            # First pass: assign a sequential number to each distinct build UUID
            # in the order they appear (which is chronological after the query fix).
            prior_build_seq: dict[str, int] = {}
            seq_counter = 0
            for hist in prior_iteration_history:
                bid = hist.get("build_id")
                if bid and bid not in prior_build_seq:
                    seq_counter += 1
                    prior_build_seq[bid] = seq_counter

            # Second pass: render each entry with a human-readable build label
            for hist in prior_iteration_history:
                iter_num = hist.get("iteration", "?")
                bid = hist.get("build_id")
                build_label = f"Build #{prior_build_seq[bid]}" if bid and bid in prior_build_seq else "Build"
                iter_return = hist.get("total_return", 0)
                iter_win_rate = hist.get("win_rate", 0)
                iter_sharpe = hist.get("sharpe_ratio", 0)
                iter_trades = hist.get("total_trades", 0)
                iter_drawdown = hist.get("max_drawdown", 0)
                iter_features = hist.get("top_features", [])
                iter_model = hist.get("best_model", "?")

                prior_history_section += f"### {build_label}, Iteration {iter_num}:\n"
                prior_history_section += (
                    f"  Return: {iter_return}% | Win Rate: {iter_win_rate}% | "
                    f"Sharpe: {iter_sharpe} | Drawdown: {iter_drawdown}% | Trades: {iter_trades}\n"
                )
                prior_history_section += f"  Best Model: {iter_model}\n"
                if iter_features:
                    prior_history_section += f"  Top Features: {', '.join(iter_features[:5])}\n"
                iter_entry = hist.get("entry_features", [])
                iter_exit = hist.get("exit_features", [])
                if iter_entry:
                    prior_history_section += f"  Entry Features Used: {', '.join(iter_entry[:5])}\n"
                if iter_exit:
                    prior_history_section += f"  Exit Features Used: {', '.join(iter_exit[:5])}\n"
                prior_history_section += "\n"

        # Build user instructions section
        user_instructions_section = ""
        if customizations or thoughts:
            user_instructions_section = "\n## ⚠️ User Instructions ⚠️\n"

            if customizations:
                user_instructions_section += f"""
### REQUIRED Customizations (MUST FOLLOW):
{customizations}

**CRITICAL**: These customizations are MANDATORY REQUIREMENTS that you MUST follow EXACTLY.
These are NOT suggestions - they are explicit constraints that define what the user wants.
You MUST incorporate every aspect of these customizations into your strategy design.
If the user specifies particular indicators, entry/exit conditions, risk parameters, or any other requirements,
you MUST include them EXACTLY as requested. Ignoring or modifying these requirements is UNACCEPTABLE.
Treat these customizations with the HIGHEST priority - even higher than general best practices.
"""

            if thoughts:
                user_instructions_section += f"""
### Thoughts to Consider (Optional Guidance):
{thoughts}

These are things the user wants you to consider when building the strategy.
While not strictly required, you should take these thoughts into account and let them inform your design decisions.
"""

        return f"""You are an expert quantitative trading strategy designer. Your role is to design
highly profitable algorithmic trading strategies using technical indicators and ML-optimized parameters.

## Task
Design a trading strategy with these specifications:
- Name: {strategy_name}
- Strategy Type: {strategy_type}
- Symbols: {', '.join(symbols)}
- Timeframe: {timeframe}
- Target Return: {target_return}%
{user_instructions_section}

## Previous Builds Context
{context}
{prior_history_section}
## Available Feature Names for ML Training
The ML pipeline calculates the following features from price data. You MUST select from these
exact feature names when specifying priority_features and entry/exit rules:

{features_list}

## Output Format
You MUST respond with a JSON block wrapped in ```json ... ``` tags containing your strategy design.
The JSON must follow this exact schema:

```json
{self.DESIGN_JSON_SCHEMA}
```

## Rules for entry/exit:
- Valid operators: "<=", ">=", "<", ">", "=="
- Feature names MUST match exactly from the available features list
- Design 6-10 entry rules and 4-8 exit rules
- entry_score_threshold = how many entry rules must be true simultaneously to trigger entry
- exit_score_threshold = how many exit rules must be true simultaneously to trigger exit
- Keep risk parameters realistic and AS DECIMALS (not percentages):
  - stop_loss: 0.02 to 0.15 (meaning 2% to 15%). Example: 0.05 = 5% stop loss
  - trailing_stop: 0.01 to 0.10 (meaning 1% to 10%). Example: 0.03 = 3% trailing stop
  - position_size: 0.05 to 0.50 (meaning 5% to 50% of portfolio per position)
  - Think about realistic risk: a 5-8% stop loss and 3-5% trailing stop is typical for swing trading
  - Do NOT set stop_loss or trailing_stop as whole numbers like 5 or 10 — use 0.05 or 0.10
- lookback_window: Number of trading days to wait before trading starts (for indicator stabilization)
  - Must be at least 200 for strategies using 200-day EMA or 52-week indicators
  - Use 30-150 for shorter-term strategies that only use 30-day indicators
  - This ensures trend indicators are fully established before any trades are placed

## ML Training Configuration:
- config_search_model: Which model to use for label configuration search (testing forward_days/profit_threshold combos)
  - Options: "LightGBM" (fastest, default), "RandomForest", "GradientBoosting", "XGBoost", "MLP", "LSTM"
  - LightGBM is recommended for speed (~2s per config). LSTM is most accurate for sequential data but ~3min per config.
  - If your strategy relies heavily on sequential patterns, LSTM may find better label configs
- You decide which ML models to train: set train_neural_networks and train_lstm to true/false
- Configure neural network architecture: nn_hidden_layers (list of layer size lists), nn_epochs, nn_batch_size, nn_learning_rate
- Configure LSTM: lstm_hidden_size, lstm_num_layers, lstm_dropout, lstm_sequence_length, lstm_epochs, lstm_batch_size
- Set cv_folds for cross-validation (3 or 5)
- Available models always trained: RandomForest, GradientBoosting, XGBoost, LightGBM
- Optional models you control: MLP neural networks (train_neural_networks), LSTM (train_lstm)
- Think about whether the data is sequential (favor LSTM) or tabular (favor tree models)

## HARD CONSTRAINTS (DO NOT VIOLATE)
- The strategy type "{strategy_type}" is FIXED and chosen by the user. You MUST NOT change it.
- ALL entry rules, exit rules, features, and indicators MUST be consistent with a {strategy_type} strategy.
- Do NOT redesign this as a different strategy type (e.g., do not turn a "Volume Profile" strategy into a "Momentum Breakout" strategy).
- If you believe the strategy type is suboptimal for the given symbols/target, you may note this in strategy_rationale, but you MUST still design within the {strategy_type} framework.

## Important:
- Base your strategy design on proven technical analysis principles
- The ML pipeline will find the optimal weights - you provide the structure and rules
- Think deeply about what indicators work best for the specific symbols and strategy type
{creation_guide}"""

    def _build_refinement_prompt(
        self,
        previous_design: Dict[str, Any],
        backtest_results: Dict[str, Any],
        iteration: int,
        creation_guide: str = "",
        strategy_type: str = "",
        training_results: Optional[Dict[str, Any]] = None,
        iteration_history: Optional[List[Dict[str, Any]]] = None,
        description: str = "",
    ) -> str:
        """Build strategy refinement prompt with full schema and analysis guidance.

        Now includes ML-extracted rules, feature importance, iteration history,
        and trade-level feedback to close the information loop between
        LLM design and ML training.
        """
        # Extract key metrics from backtest results for the prompt
        total_return = backtest_results.get("total_return", 0)
        win_rate = backtest_results.get("win_rate", 0)
        max_drawdown = backtest_results.get("max_drawdown", 0)
        num_trades = backtest_results.get("total_trades", 0)
        sharpe_ratio = backtest_results.get("sharpe_ratio", 0)

        features_list = "\n".join(f"  - {f}" for f in self.AVAILABLE_FEATURES)

        # --- Build ML-extracted rules section (Items 1 & 2) ---
        ml_rules_section = ""
        feature_importance_section = ""
        if training_results:
            # Extracted rules from ML training
            extracted = training_results.get("extracted_rules", {})
            ml_entry_rules = extracted.get("entry_rules", [])
            ml_exit_rules = extracted.get("exit_rules", [])

            if ml_entry_rules or ml_exit_rules:
                ml_rules_section = "\n## ML-Extracted Rules (from actual training)\n"
                ml_rules_section += "These rules were discovered by the ML pipeline during training.\n"
                ml_rules_section += "They represent what the model actually learned works — USE THESE as your primary guide.\n\n"

                if ml_entry_rules:
                    ml_rules_section += "### ML Entry Rules:\n"
                    for i, rule in enumerate(ml_entry_rules[:10], 1):
                        feat = rule.get("feature", "?")
                        op = rule.get("operator", "?")
                        thresh = rule.get("threshold", "?")
                        imp = rule.get("importance", 0)
                        ml_rules_section += f"  {i}. {feat} {op} {thresh} (importance: {imp:.4f})\n"

                if ml_exit_rules:
                    ml_rules_section += "\n### ML Exit Rules:\n"
                    for i, rule in enumerate(ml_exit_rules[:8], 1):
                        feat = rule.get("feature", "?")
                        op = rule.get("operator", "?")
                        thresh = rule.get("threshold", "?")
                        imp = rule.get("importance", 0)
                        ml_rules_section += f"  {i}. {feat} {op} {thresh} (importance: {imp:.4f})\n"

            # Feature importance rankings
            features_data = training_results.get("features", {})
            top_features = features_data.get("top_features", []) if isinstance(features_data, dict) else []
            if top_features:
                feature_importance_section = "\n## Feature Importance Rankings (from ML training)\n"
                feature_importance_section += "These features had the highest predictive power during training:\n\n"
                for i, feat in enumerate(top_features[:15], 1):
                    # Worker uses "name" key, rule extractor uses "feature" key
                    name = feat.get("name", feat.get("feature", "?"))
                    imp = feat.get("importance", 0)
                    feature_importance_section += f"  {i}. {name}: {imp:.4f}\n"
                feature_importance_section += "\nPRIORITIZE features with high importance scores in your design.\n"
                feature_importance_section += "Features NOT in this list had low predictive power — avoid relying on them.\n"

            # Best model info — worker may send a plain string (model name) or a
            # dict {"name": ..., "metrics": ...}.  Normalise to dict form here.
            _best_model_raw = training_results.get("best_model", {})
            if isinstance(_best_model_raw, dict):
                best_model = _best_model_raw
            elif _best_model_raw:
                best_model = {"name": str(_best_model_raw), "metrics": {}}
            else:
                best_model = {}
            if best_model:
                model_name = best_model.get("name", "Unknown")
                model_metrics = best_model.get("metrics", {})
                ml_rules_section += f"\n### Best ML Model: {model_name}\n"
                if model_metrics:
                    ml_rules_section += f"  - F1: {model_metrics.get('f1', 0):.4f}\n"
                    ml_rules_section += f"  - Precision: {model_metrics.get('precision', 0):.4f}\n"
                    ml_rules_section += f"  - Recall: {model_metrics.get('recall', 0):.4f}\n"
                    ml_rules_section += f"  - AUC: {model_metrics.get('auc', 0):.4f}\n"

        # --- Build iteration history section (Item 5) ---
        history_section = ""
        if iteration_history:
            history_section = "\n## Iteration History (DO NOT repeat failed approaches)\n"
            history_section += (
                "Each entry belongs to a distinct build attempt. "
                "'Current Build' means the iterations already run in THIS build. "
                "'Prior Build #N' means a previous, separate build run.\n"
                "Learn from what worked and what failed across ALL builds.\n\n"
            )

            # Assign sequential numbers to prior builds so the LLM can refer to them
            # unambiguously. Prior build entries carry a 'build_id' but NOT
            # 'is_current_build'. Current-build entries carry both.
            prior_build_seq: dict[str, int] = {}
            seq_counter = 0
            for hist in iteration_history:
                bid = hist.get("build_id")
                if bid and not hist.get("is_current_build") and bid not in prior_build_seq:
                    seq_counter += 1
                    prior_build_seq[bid] = seq_counter

            for hist in iteration_history:
                iter_num = hist.get("iteration", "?")
                iter_return = hist.get("total_return", 0)
                iter_win_rate = hist.get("win_rate", 0)
                iter_sharpe = hist.get("sharpe_ratio", 0)
                iter_trades = hist.get("total_trades", 0)
                iter_drawdown = hist.get("max_drawdown", 0)
                iter_features = hist.get("top_features", [])
                iter_model = hist.get("best_model", "?")

                # Build a human-readable label so the LLM can tell builds apart
                bid = hist.get("build_id")
                if hist.get("is_current_build"):
                    build_label = "Current Build"
                elif bid and bid in prior_build_seq:
                    build_label = f"Prior Build #{prior_build_seq[bid]}"
                else:
                    build_label = "Prior Build"

                history_section += f"### {build_label}, Iteration {iter_num}:\n"
                history_section += f"  Return: {iter_return}% | Win Rate: {iter_win_rate}% | "
                history_section += f"Sharpe: {iter_sharpe} | Drawdown: {iter_drawdown}% | Trades: {iter_trades}\n"
                history_section += f"  Best Model: {iter_model}\n"
                if iter_features:
                    history_section += f"  Top Features: {', '.join(iter_features[:5])}\n"

                # Entry/exit rules tried
                iter_entry = hist.get("entry_features", [])
                iter_exit = hist.get("exit_features", [])
                if iter_entry:
                    history_section += f"  Entry Features Used: {', '.join(iter_entry[:5])}\n"
                if iter_exit:
                    history_section += f"  Exit Features Used: {', '.join(iter_exit[:5])}\n"
                history_section += "\n"

        # --- Build trade-level feedback section (Item 9) ---
        trade_feedback_section = ""
        if backtest_results:
            trades = backtest_results.get("trades", [])
            if trades:
                winning = [t for t in trades if t.get("pnl", 0) > 0]
                losing = [t for t in trades if t.get("pnl", 0) <= 0]

                trade_feedback_section = "\n## Trade-Level Feedback\n"
                trade_feedback_section += f"Total: {len(trades)} trades | "
                trade_feedback_section += f"Winners: {len(winning)} | Losers: {len(losing)}\n\n"

                if winning:
                    avg_win = sum(t.get("pnl", 0) for t in winning) / len(winning)
                    max_win = max(t.get("pnl", 0) for t in winning)
                    avg_hold_win = sum(t.get("hold_days", 0) for t in winning) / len(winning)
                    trade_feedback_section += f"### Winners: Avg PnL: {avg_win:.2f}% | Max: {max_win:.2f}% | Avg Hold: {avg_hold_win:.1f} days\n"

                if losing:
                    avg_loss = sum(t.get("pnl", 0) for t in losing) / len(losing)
                    max_loss = min(t.get("pnl", 0) for t in losing)
                    avg_hold_loss = sum(t.get("hold_days", 0) for t in losing) / len(losing)
                    trade_feedback_section += f"### Losers: Avg PnL: {avg_loss:.2f}% | Max: {max_loss:.2f}% | Avg Hold: {avg_hold_loss:.1f} days\n"

                # Analyze exit reasons if available
                exit_reasons = {}
                for t in trades:
                    reason = t.get("exit_reason", "unknown")
                    exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
                if exit_reasons:
                    trade_feedback_section += "\n### Exit Reason Distribution:\n"
                    for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
                        trade_feedback_section += f"  - {reason}: {count} trades\n"

                # Show worst trades for analysis
                if losing:
                    trade_feedback_section += "\n### Worst 3 Trades (learn from these):\n"
                    worst = sorted(losing, key=lambda t: t.get("pnl", 0))[:3]
                    for t in worst:
                        trade_feedback_section += (
                            f"  - {t.get('symbol', '?')}: {t.get('pnl', 0):.2f}% | "
                            f"Hold: {t.get('hold_days', '?')}d | "
                            f"Exit: {t.get('exit_reason', '?')}\n"
                        )

        # Create a filtered version of backtest_results that excludes large arrays
        # Keep: metrics, period returns (daily/weekly/monthly/yearly)
        # Remove: trades, equity_curve, ohlc_data (these can be hundreds of thousands of tokens)
        filtered_results = {
            k: v for k, v in backtest_results.items()
            if k not in ['trades', 'equity_curve', 'ohlc_data']
        }

        # Build user description section if provided
        user_description_section = ""
        if description:
            user_description_section = f"""
## ⚠️ CRITICAL: User Customization Requirements ⚠️
User Description/Comments: {description}

**MANDATORY INSTRUCTION**: The user's description/comments above are CRITICAL REQUIREMENTS that you MUST follow.
These are NOT suggestions - they are explicit constraints and customizations that define what the user wants.
You MUST incorporate every aspect of the user's description into your strategy refinements to the best of your ability.
If the user specifies particular indicators, entry/exit conditions, risk parameters, or any other requirements,
you MUST include them EXACTLY as requested. Ignoring or modifying user requirements is UNACCEPTABLE.
Treat the user's description with the HIGHEST priority - even higher than general best practices.
"""

        return f"""You are an expert quantitative trading strategy designer refining a strategy based on backtest results.

## Previous Design (Iteration {iteration})
{json.dumps(previous_design, indent=2)}

## Strategy Type
- Strategy Type: {strategy_type}
{user_description_section}
## Backtest Results (Iteration {iteration})
- Total Return: {total_return}%
- Win Rate: {win_rate}%
- Max Drawdown: {max_drawdown}%
- Sharpe Ratio: {sharpe_ratio}
- Total Trades: {num_trades}

Full results (excluding large arrays):
{json.dumps(filtered_results, indent=2)}
{ml_rules_section}
{feature_importance_section}
{history_section}
{trade_feedback_section}
## Available Feature Names
You MUST select from these exact feature names:

{features_list}

## CRITICAL INSTRUCTIONS
1. Analyze what specifically failed or underperformed in the previous designs (all iterations) and why.
2. If features were tried before and didn't work well, use DIFFERENT features.
3. If thresholds were too tight or too loose, adjust them significantly (not just +-0.5%).
4. Consider different features, thresholds, and ML configurations — but NEVER change the core strategy type.
5. If the strategy performance is much greater than the previous iteration, fine tune the strategy further keeping the features but adjusting the thresholds and risk parameters to optimize the strategy.
6. PRIORITIZE features that had high ML importance scores. The ML model already tested these features against real data.
7. If ML-extracted rules are provided, use them as your PRIMARY guide for threshold values. These are data-driven, not theoretical.
8. Review the iteration history and AVOID repeating approaches that produced poor results.
9. If trade-level feedback shows most losses come from a specific pattern (e.g., stop-loss exits), address that specifically.

## Analysis Questions (answer these in your thinking before designing)
1. What pattern do you see in the results? Is the strategy improving or stuck?
2. Which features contributed most to winning trades vs losing trades? (Check ML feature importance!)
3. Are the entry rules too loose (too many bad entries) or too tight (missing opportunities)?
4. Are the exit rules triggering too early (cutting winners) or too late (letting losers run)?
5. What HASN'T been tried yet that could work better? (Check iteration history!)
6. Should the ML training parameters (forward_returns_days, profit_thresholds) be changed?
7. What does the trade-level feedback tell you about the strategy's weaknesses?

## Output Format
You MUST respond with a JSON block wrapped in ```json ... ``` tags.
The JSON must follow this exact schema:

```json
{self.DESIGN_JSON_SCHEMA}
```

## Rules for entry/exit:
- Valid operators: "<=", ">=", "<", ">", "=="
- Feature names MUST match exactly from the available features list
- Design 2-10 entry rules and 3-8 exit rules
- entry_score_threshold = how many entry rules must be true simultaneously to trigger entry
- exit_score_threshold = how many exit rules must be true simultaneously to trigger exit
- Keep risk parameters AS DECIMALS (not percentages): stop_loss 0.02-0.15, trailing_stop 0.01-0.10, position_size 0.05-0.50
- lookback_window: at least 200 for strategies using 200-day EMA or 52-week indicators

## ML Training Configuration:
- config_search_model: "LightGBM" (fastest), "RandomForest", "GradientBoosting", "XGBoost", "MLP", "LSTM"
- Set train_neural_networks and train_lstm to true/false based on strategy needs
- Configure NN and LSTM hyperparameters as needed
- Think about whether the data is sequential (favor LSTM) or tabular (favor tree models)

## HARD CONSTRAINTS (DO NOT VIOLATE)
- The strategy type "{strategy_type}" is FIXED and chosen by the user. You MUST NOT change it.
- ALL entry rules, exit rules, features, and indicators MUST be consistent with a {strategy_type} strategy.
- "Completely revised" means new features, thresholds, ML config, and risk parameters — NOT a different strategy type.
- Do NOT redesign this as a different strategy type. Optimize WITHIN the {strategy_type} framework.

Make meaningful changes to parameters, features, and rules — don't just tweak numbers slightly. Return a revised strategy design that stays true to the {strategy_type} approach.
{creation_guide}"""

    def _validate_design_rules(self, design: dict) -> bool:
        """Check that the design has non-empty entry and exit rules."""
        entry_rules = design.get("entry_rules", [])
        exit_rules = design.get("exit_rules", [])
        if not isinstance(entry_rules, list) or len(entry_rules) == 0:
            logger.warning("Design has empty entry_rules")
            return False
        if not isinstance(exit_rules, list) or len(exit_rules) == 0:
            logger.warning("Design has empty exit_rules")
            return False
        return True

    async def _parse_design_response(self, response, iteration: int = 0) -> Dict[str, Any]:
        """Parse Claude response to extract design JSON and thinking content.

        Handles both streaming and non-streaming responses.

        Returns a dict with keys:
        - "design": the parsed strategy design JSON
        - "thinking": concatenated thinking block text (empty string if none)
        - "thinking_blocks": list of individual thinking blocks
        """
        text = ""
        thinking_parts: List[str] = []

        # Check if response is a stream or a complete Message
        if hasattr(response, 'content'):
            # Non-streaming response - has .content attribute
            for block in response.content:
                if block.type == "thinking" and hasattr(block, "thinking"):
                    thinking_parts.append(block.thinking)
                elif hasattr(block, "text"):
                    text += block.text
        else:
            # Streaming response - iterate through events
            current_block_type = None
            current_block_content = ""
            last_published_length = 0
            publish_chunk_size = 200  # Publish every 200 characters to avoid too many tiny messages
            logger.info("🔄 Starting to process streaming response...")

            for event in response:
                # New content block starting
                if event.type == "content_block_start":
                    current_block_type = event.content_block.type
                    current_block_content = ""
                    last_published_length = 0
                    logger.info(f"📦 New content block started: {current_block_type}")

                # Content within a block
                elif event.type == "content_block_delta":
                    if event.delta.type == "thinking_delta":
                        current_block_content += event.delta.thinking

                        # Publish incremental updates for thinking blocks as they stream in
                        if len(current_block_content) - last_published_length >= publish_chunk_size:
                            await self._publish_thinking_delta(
                                current_block_content,
                                len(thinking_parts) + 1,  # Block number (will be finalized when complete)
                                is_complete=False,
                                iteration=iteration
                            )
                            last_published_length = len(current_block_content)

                    elif event.delta.type == "text_delta":
                        current_block_content += event.delta.text

                # Block finished - save it
                elif event.type == "content_block_stop":
                    if current_block_type == "thinking":
                        # Save thinking block
                        thinking_parts.append(current_block_content)
                        logger.info(f"🧠 Thinking block #{len(thinking_parts)} completed ({len(current_block_content)} chars)")
                        self._log(f"Thinking block completed ({len(current_block_content)} chars)")

                        # Publish final complete version
                        await self._publish_thinking_delta(
                            current_block_content,
                            len(thinking_parts),
                            is_complete=True,
                            iteration=iteration
                        )
                    elif current_block_type == "text":
                        text += current_block_content
                        logger.debug(f"📝 Text block completed ({len(current_block_content)} chars)")
                    current_block_type = None
                    current_block_content = ""
                    last_published_length = 0

            logger.info(f"✅ Streaming complete: {len(thinking_parts)} thinking blocks, {len(text)} chars of text")

        thinking_text = "\n".join(thinking_parts)

        # Step 1: Try to match ```json ... ``` code fence (most reliable)
        design = None
        json_fence_match = re.search(r"```json\s*([\s\S]*?)```", text)
        if json_fence_match:
            try:
                design = json.loads(json_fence_match.group(1).strip())
                # Validate rules are non-empty
                if self._validate_design_rules(design):
                    return {"design": design, "thinking": thinking_text, "thinking_blocks": thinking_parts}
            except json.JSONDecodeError:
                pass

        # Step 2: Try matching outermost JSON object (less greedy)
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                design = json.loads(json_match.group())
                if self._validate_design_rules(design):
                    return {"design": design, "thinking": thinking_text, "thinking_blocks": thinking_parts}
                # Design parsed but rules are empty — will retry below
            except json.JSONDecodeError:
                pass

        # If we got a design but rules are empty, retry once
        if design and not self._validate_design_rules(design):
            logger.warning("Parsed design has blank rules — retrying AI once...")
            try:
                retry_response = self.client.messages.create(
                    model="claude-opus-4-6",
                    max_tokens=settings.CLAUDE_MAX_TOKENS,
                    thinking={"type": "adaptive"},
                    messages=[{
                        "role": "user",
                        "content": f"Your previous response had empty entry_rules and/or exit_rules arrays. "
                                   f"Please provide ONLY a JSON block with the complete strategy design including "
                                   f"6-10 entry_rules and 4-8 exit_rules. Use features from: {', '.join(self.AVAILABLE_FEATURES[:20])}..."
                    }],
                )
                retry_text = ""
                for block in retry_response.content:
                    if hasattr(block, "text"):
                        retry_text += block.text
                retry_fence = re.search(r"```json\s*([\s\S]*?)```", retry_text)
                if retry_fence:
                    retry_design = json.loads(retry_fence.group(1).strip())
                    if self._validate_design_rules(retry_design):
                        return {"design": retry_design, "thinking": thinking_text, "thinking_blocks": thinking_parts}
            except Exception as e:
                logger.warning("Rules retry failed: %s", e)

        # Log when fallback is used
        logger.warning("Using fallback design — Claude response could not be parsed: %s", text[:200])

        # Return default design with full schema if parsing fails
        return {
            "design": {
                "strategy_description": "Default design",
                "strategy_rationale": "Could not parse Claude response",
                "forward_returns_days_options": [2, 3, 5, 7, 10, 15, 20, 30],
                "profit_threshold_options": [1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
                "recommended_n_iter_search": 50,
                "priority_features": [],
                "feature_rationale": {},
                "entry_rules": [
                    {"feature": "RSI_14", "operator": "<=", "threshold": 35},
                    {"feature": "MACD_HIST", "operator": ">=", "threshold": 0},
                    {"feature": "EMA_21", "operator": "<=", "threshold": 1.02},
                    {"feature": "BB_PCT_B", "operator": "<=", "threshold": 0.3},
                    {"feature": "MOM_1M", "operator": ">=", "threshold": 0},
                    {"feature": "VOLUME_RATIO", "operator": ">=", "threshold": 1.0},
                ],
                "entry_score_threshold": 3,
                "exit_rules": [
                    {"feature": "RSI_14", "operator": ">=", "threshold": 70},
                    {"feature": "MACD_HIST", "operator": "<=", "threshold": 0},
                    {"feature": "BB_PCT_B", "operator": ">=", "threshold": 0.85},
                    {"feature": "MOM_1M", "operator": "<=", "threshold": -2},
                ],
                "exit_score_threshold": 3,
                "recommended_stop_loss": 0.05,
                "recommended_trailing_stop": 0.03,
                "recommended_max_positions": 3,
                "recommended_position_size": 0.10,
                "lookback_window": 200,
                "ml_training_config": {
                    "config_search_model": "LightGBM",
                    "train_neural_networks": True,
                    "train_lstm": True,
                    "cv_folds": 3,
                    "nn_epochs": 100,
                    "nn_batch_size": 64,
                    "nn_learning_rate": 0.001,
                    "nn_hidden_layers": [[128, 64], [256, 128, 64]],
                    "lstm_hidden_size": 128,
                    "lstm_num_layers": 2,
                    "lstm_dropout": 0.2,
                    "lstm_sequence_length": 100,
                    "lstm_epochs": 250,
                    "lstm_batch_size": 32,
                },
            },
            "thinking": thinking_text,
            "thinking_blocks": thinking_parts,
        }


    async def update_creation_guide(
        self,
        db: AsyncSession,
        strategy_type: Optional[str],
        design: Dict[str, Any],
        training_results: Dict[str, Any],
        backtest_results: Dict[str, Any],
    ) -> None:
        """Update creation guide rows with lessons learned from the latest iteration.

        Loads TWO rows: "general" + strategy-type specific. Calls Claude to produce
        updated content for both. Creates the strategy-type row on-demand if it doesn't
        exist yet. Increments `version` on each saved row.

        This method is non-blocking: failures are logged as warnings and never crash
        the build loop.
        """
        try:
            # --- Load "general" guide row ---
            result = await db.execute(
                select(StrategyCreationGuide).where(
                    StrategyCreationGuide.strategy_type == "general"
                )
            )
            general_row = result.scalar_one_or_none()
            general_content = general_row.content if general_row else ""

            # --- Load or prepare strategy-type row ---
            effective_type = strategy_type if strategy_type and strategy_type != "general" else None
            type_row = None
            type_content = ""
            if effective_type:
                result = await db.execute(
                    select(StrategyCreationGuide).where(
                        StrategyCreationGuide.strategy_type == effective_type
                    )
                )
                type_row = result.scalar_one_or_none()
                type_content = type_row.content if type_row else ""

            # --- Build Claude prompt ---
            bt = backtest_results or {}
            entry_rules_str = json.dumps(design.get("entry_rules", []), indent=2)
            exit_rules_str = json.dumps(design.get("exit_rules", []), indent=2)
            # best_model may be a dict {"name": ..., "metrics": ...} or a plain string (model name)
            _best_model_raw = (training_results or {}).get("best_model", "Unknown")
            best_model_name = (
                _best_model_raw.get("name", "Unknown")
                if isinstance(_best_model_raw, dict)
                else (_best_model_raw or "Unknown")
            )

            prompt = f"""You are maintaining a Strategy Creation Guide that helps an AI design winning trading strategies.
This guide is referenced BEFORE starting any new strategy build, so it must contain actionable lessons.

## Current General Guide
{general_content if general_content else "(empty — first entry)"}

## Current Strategy-Type Guide ({effective_type or "N/A"})
{type_content if type_content else "(empty — first entry for this type)"}

## New Iteration Results to Incorporate

Strategy Type: {strategy_type or "general"}
Design Description: {design.get("strategy_description", "N/A")}
Priority Features: {design.get("priority_features", [])}
Entry Rules: {entry_rules_str}
Exit Rules: {exit_rules_str}
Total Return: {bt.get("total_return", 0)}%
Win Rate: {bt.get("win_rate", 0)}%
Max Drawdown: {bt.get("max_drawdown", 0)}%
Best Model: {best_model_name}

ML Config: {json.dumps(design.get("ml_training_config", {}), indent=2)}

## Instructions
Update BOTH guides with lessons from this iteration. Return your response with exactly two sections:

<GENERAL>
(Complete updated general guide content — cross-cutting lessons for all strategy types)
</GENERAL>

<SPECIFIC>
(Complete updated strategy-type-specific guide content for "{effective_type or "general"}")
</SPECIFIC>

Keep content concise and actionable. Each lesson should be directly applicable."""

            # --- Call Claude ---
            response = self.client.messages.create(
                model="claude-opus-4-6",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
            )
            text_blocks = [b for b in response.content if hasattr(b, "text")]
            raw_text = "\n".join(b.text for b in text_blocks)

            # --- Parse structured output ---
            general_match = re.search(r"<GENERAL>(.*?)</GENERAL>", raw_text, re.DOTALL)
            specific_match = re.search(r"<SPECIFIC>(.*?)</SPECIFIC>", raw_text, re.DOTALL)

            new_general = general_match.group(1).strip() if general_match else None
            new_specific = specific_match.group(1).strip() if specific_match else None

            # --- Persist general row ---
            if new_general:
                if general_row:
                    general_row.content = new_general
                    general_row.version += 1
                else:
                    general_row = StrategyCreationGuide(
                        strategy_type="general",
                        content=new_general,
                        version=1,
                    )
                    db.add(general_row)

            # --- Persist strategy-type row ---
            if new_specific and effective_type:
                if type_row:
                    type_row.content = new_specific
                    type_row.version += 1
                else:
                    # Create on-demand
                    type_row = StrategyCreationGuide(
                        strategy_type=effective_type,
                        content=new_specific,
                        version=1,
                    )
                    db.add(type_row)

            await db.flush()
            self._log("Creation guide updated successfully")

        except Exception as e:
            # Non-blocking — log warning but never crash the build
            logger.warning("Creation guide update failed (non-blocking): %s", e)
            self._log(f"Warning: Creation guide update failed: {e}")

    async def generate_selection_thinking(
        self,
        completed_iterations: List[Any],
        best_iteration: Any,
        target_return: float,
        strategy_type: str = "",
    ) -> Dict[str, Any]:
        """Generate Claude's explanation of which iteration was selected and why.

        Args:
            completed_iterations: List of all completed BuildIteration records
            best_iteration: The BuildIteration record that was selected as best
            target_return: The target return percentage that was not met
            strategy_type: The type of strategy being built

        Returns:
            A dict with keys:
            - "thinking": concatenated thinking text
            - "thinking_blocks": list of individual thinking blocks
        """
        # Build a summary of all iterations for Claude
        iterations_summary = []
        for it in completed_iterations:
            bt = it.backtest_results or {}
            iterations_summary.append({
                "iteration": it.iteration_number,
                "total_return": bt.get("total_return", 0),
                "win_rate": bt.get("win_rate", 0),
                "max_drawdown": bt.get("max_drawdown", 0),
                "sharpe_ratio": bt.get("sharpe_ratio", "N/A"),
                "total_trades": bt.get("total_trades", 0),
            })

        best_bt = best_iteration.backtest_results or {}

        prompt = f"""You are reviewing the results of a multi-iteration strategy build process. The target return of {target_return}% was not achieved in any iteration, so we need to select the best performing iteration to deploy.

The strategy type is: {strategy_type}

Here are all completed iterations and their backtest results:

{json.dumps(iterations_summary, indent=2)}

The best iteration by total return is iteration {best_iteration.iteration_number} with {best_bt.get('total_return', 0):.2f}% return.

Please provide a brief explanation (2-3 sentences) of:
1. Why this iteration was selected as the best
2. What the key performance metrics show
3. Any notable trade-offs or considerations

Keep it concise and focused on the decision rationale."""

        try:
            response = self.client.messages.create(
                model="claude-opus-4-6",
                max_tokens=settings.CLAUDE_MAX_TOKENS,
                thinking={
                    "type": "adaptive",
                },
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract thinking and text blocks
            thinking_blocks = [b for b in response.content if hasattr(b, "type") and b.type == "thinking"]
            text_blocks = [b for b in response.content if hasattr(b, "text")]

            thinking_parts = [b.thinking for b in thinking_blocks] if thinking_blocks else []
            thinking_text = "\n".join(thinking_parts)
            response_text = "\n".join(b.text for b in text_blocks)

            # Combine thinking and response
            full_thinking = f"{thinking_text}\n\n{response_text}" if thinking_text else response_text

            self._log(f"Selection thinking generated ({len(full_thinking)} chars)")
            return {
                "thinking": full_thinking,
                "thinking_blocks": thinking_parts,
            }

        except Exception as e:
            logger.warning("Selection thinking generation failed: %s", e)
            self._log(f"Warning: Selection thinking generation failed: {e}")
            # Return a simple fallback explanation
            fallback_text = f"Selected iteration {best_iteration.iteration_number} as it achieved the highest total return of {best_bt.get('total_return', 0):.2f}% among all {len(completed_iterations)} completed iterations."
            return {
                "thinking": fallback_text,
                "thinking_blocks": [fallback_text],
            }

    async def generate_strategy_readme(
        self,
        strategy_name: str,
        symbols: List[str],
        design: Dict[str, Any],
        backtest_results: Dict[str, Any],
        model_metrics: Dict[str, Any],
        config: Dict[str, Any],
        strategy_type: str = "",
    ) -> str:
        """Generate a comprehensive README markdown for a completed strategy.

        Calls Claude to produce a human-readable README summarising the strategy,
        its design, performance metrics, and usage instructions.

        Returns the README markdown string, or a minimal fallback on failure.
        """
        # Unpack design fields with safe defaults
        ml_config = design.get("ml_training_config", {})
        entry_rules = json.dumps(design.get("entry_rules", []), indent=2)
        exit_rules = json.dumps(design.get("exit_rules", []), indent=2)
        bt = backtest_results or {}

        prompt = f"""Write a comprehensive README.md for a trading strategy. Include these sections:
1. Strategy name, description, and rationale
2. How the strategy works (entry/exit logic explained in plain English)
3. Technical indicators used and why
4. ML models used to discover optimal settings
5. Key parameters and their values
6. Backtest performance results
7. Risk management settings
8. How to run the strategy

Strategy: {strategy_name}
Strategy Type: {strategy_type}
Symbols: {', '.join(symbols)}
Description: {design.get("strategy_description", "N/A")}
Rationale: {design.get("strategy_rationale", "N/A")}

Entry Rules:
{entry_rules}
Entry Score Threshold: {design.get("entry_score_threshold", "N/A")}

Exit Rules:
{exit_rules}
Exit Score Threshold: {design.get("exit_score_threshold", "N/A")}

Priority Features: {design.get("priority_features", [])}

ML Training Config:
- Neural Networks (MLP): {'Enabled' if ml_config.get('train_neural_networks') else 'Disabled'}
  - Architectures: {ml_config.get('nn_hidden_layers', 'N/A')}
  - Epochs: {ml_config.get('nn_epochs', 'N/A')}, Batch Size: {ml_config.get('nn_batch_size', 'N/A')}, LR: {ml_config.get('nn_learning_rate', 'N/A')}
- LSTM: {'Enabled' if ml_config.get('train_lstm') else 'Disabled'}
  - Hidden: {ml_config.get('lstm_hidden_size', 'N/A')}, Layers: {ml_config.get('lstm_num_layers', 'N/A')}
  - Epochs: {ml_config.get('lstm_epochs', 'N/A')}, Seq Length: {ml_config.get('lstm_sequence_length', 'N/A')}
- Cross-Validation Folds: {ml_config.get('cv_folds', 'N/A')}
- Forward Return Days: {design.get('forward_returns_days_options', 'N/A')}
- Profit Thresholds: {design.get('profit_threshold_options', 'N/A')}

Best Model: {model_metrics.get('model_name', 'Unknown')}
Model F1: {model_metrics.get('f1', 0):.4f}
Precision: {model_metrics.get('precision', 0):.4f}
Recall: {model_metrics.get('recall', 0):.4f}

Backtest Results:
- Total Return: {bt.get('total_return', 0):.2f}%
- Win Rate: {bt.get('win_rate', 0):.1f}%
- Max Drawdown: {bt.get('max_drawdown', 0):.2f}%
- Sharpe Ratio: {bt.get('sharpe_ratio', 'N/A')}

Risk Management:
- Stop Loss: {design.get('recommended_stop_loss', 0) * 100:.1f}%
- Trailing Stop: {design.get('recommended_trailing_stop', 0) * 100:.1f}%
- Max Positions: {design.get('recommended_max_positions', 'N/A')}
- Position Size: {design.get('recommended_position_size', 0) * 100:.1f}%

Config:
{json.dumps(config, indent=2)[:2000]}

Return ONLY the markdown content, no code fences wrapping it."""

        try:
            response = self.client.messages.create(
                model="claude-opus-4-6",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
            )
            text_blocks = [b for b in response.content if hasattr(b, "text")]
            readme = "\n".join(b.text for b in text_blocks)
            self._log(f"README generated ({len(readme)} chars)")
            return readme
        except Exception as e:
            logger.warning("README generation failed: %s", e)
            self._log(f"Warning: README generation failed: {e}")
            return f"# {strategy_name}\n\nAuto-generated strategy for {', '.join(symbols)}.\n"

    def _read_template_file(self, filename: str) -> str:
        """Read a template file from the strategy container templates directory."""
        template_dir = Path(__file__).resolve().parent.parent.parent / "templates" / "strategy_container"
        template_path = template_dir / filename
        return template_path.read_text()

    @staticmethod
    def _ensure_async_interface(code: str) -> str:
        """Post-process generated strategy code to guarantee async method signatures.

        The LLM occasionally emits synchronous `def calculate_indicators` or
        `def check_entry_conditions` despite being told not to.  This method:
        1. Converts both methods to `async def` if they are not already.
        2. Fixes `check_entry_conditions` signature when the LLM uses the
           wrong parameter list (e.g. `(self, df: pd.DataFrame)`).

        The backtest runner always calls these with `await`, so they MUST be
        coroutines.  The method intentionally only patches these two names to
        minimise the risk of unintended side-effects.
        """
        import re

        # --- Fix calculate_indicators: ensure it is async ---
        # Only matches `def` (not already-correct `async def`), so idempotent.
        code = re.sub(
            r'\n(    )def (calculate_indicators\s*\(self)',
            r'\n\1async def \2',
            code,
        )

        # --- Fix check_entry_conditions: ensure it is async ---
        code = re.sub(
            r'\n(    )def (check_entry_conditions\s*\()',
            r'\n\1async def \2',
            code,
        )

        # --- Fix check_entry_conditions parameter list when it is wrong ---
        # Wrong form:  check_entry_conditions(self, df: pd.DataFrame) -> ...
        # Correct form: check_entry_conditions(self, symbol: str, indicators: Dict, current_date=None) -> Optional[str]:
        code = re.sub(
            r'(async def check_entry_conditions\s*\(\s*self\s*,\s*)df\s*:\s*pd\.DataFrame[^)]*\)',
            r'\1symbol: str, indicators: Dict, current_date=None) -> Optional[str]',
            code,
        )

        return code

    async def generate_strategy_code(
        self,
        strategy_name: str,
        symbols: List[str],
        design: Dict[str, Any],
        training_results: Dict[str, Any],
        strategy_type: str,
        config: Dict[str, Any],
    ) -> str:
        """Generate complete custom strategy.py file using Claude.

        Args:
            strategy_name: Name of the strategy
            symbols: List of trading symbols
            design: Strategy design dict with entry/exit rules, features, indicators
            training_results: Training results with backtest metrics
            strategy_type: Type of strategy (momentum, mean reversion, etc.)
            config: Strategy configuration dict

        Returns:
            Complete Python source code for strategy.py
        """
        self._log(f"Generating strategy code for {strategy_name}")

        try:
            # Read template file for reference
            template_code = self._read_template_file("strategy.py")

            # Build prompt — prefer ML-extracted rules over LLM-designed rules
            extracted = training_results.get("extracted_rules", {})
            entry_rules_data = extracted.get("entry_rules") or design.get("entry_rules", [])
            exit_rules_data = extracted.get("exit_rules") or design.get("exit_rules", [])
            entry_rules = json.dumps(entry_rules_data, indent=2)
            exit_rules = json.dumps(exit_rules_data, indent=2)
            features = design.get("priority_features", [])
            indicators = design.get("indicators", {})
            rules_source = "ML training pipeline" if extracted.get("entry_rules") else "LLM design"

            prompt = f"""You are generating a complete custom trading strategy file (strategy.py) for a production trading system.

## Strategy Specifications
- Name: {strategy_name}
- Type: {strategy_type}
- Symbols: {', '.join(symbols)}
- Description: {design.get("strategy_description", "N/A")}

## Entry Rules (from {rules_source} — implement these EXACTLY)
{entry_rules}
Entry Score Threshold: {design.get("entry_score_threshold", 3)}

## Exit Rules (from {rules_source} — implement these EXACTLY)
{exit_rules}
Exit Score Threshold: {design.get("exit_score_threshold", 3)}

## Priority Features
{features}

## Training Results
The ML training achieved:
- Total Return: {training_results.get('backtest_results', {}).get('total_return', 0)}%
- Win Rate: {training_results.get('backtest_results', {}).get('win_rate', 0)}%
- Best Model: {training_results.get('best_model', 'Unknown')}

## CRITICAL REQUIREMENTS

1. **EXACT RULES**: The entry/exit rules above are the EXACT rules that produced the training results.
   You MUST implement them precisely — use the exact feature names, operators, and thresholds.
   Do NOT substitute different indicators or change thresholds. Each rule has a "feature" name
   (e.g. "MOMENTUM_63D", "ADX", "EMA_21"), an "operator" (e.g. "<=", ">="), and a "threshold" value.
   Your calculate_indicators() method MUST compute every feature referenced in the rules.

2. **Class Structure**: The class MUST be named `TradingStrategy` with this exact API:
   - `__init__(self)` - Load config from config.json
   - `analyze(self, symbols: List[str]) -> List[Dict]` - Main analysis method that returns signals

3. **ASYNC METHODS — NON-NEGOTIABLE**: The following TWO methods MUST use `async def` (not `def`):

   ```python
   async def calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
       \"\"\"Calculate technical indicators. Returns a dict of indicator values.\"\"\"
       ...  # your implementation here

   async def check_entry_conditions(
       self,
       symbol: str,
       indicators: Dict,
       current_date=None
   ) -> Optional[str]:
       \"\"\"Check entry/exit conditions. Returns 'BUY', 'SELL', or None.\"\"\"
       ...  # your implementation here
   ```

   - `calculate_indicators` receives a **DataFrame** (OHLCV slice) and returns a **dict** of computed indicator values.
   - `check_entry_conditions` receives `symbol` (str), `indicators` (pre-computed dict from calculate_indicators),
     and optional `current_date`. It returns the string `'BUY'`, `'SELL'`, or `None` — NOT a dict.
   - The backtest runner calls these with `await`, so they MUST be declared `async def`.

4. **Imports**: You MUST use these imports:
   - `from data_provider import DataProvider` for fetching market data
   - `import json` to load config.json
   - Standard libraries: pandas, numpy, ta (technical analysis)
   - `from typing import Dict, Any, List, Optional`

5. **Data Fetching**: Use `DataProvider` class, NOT direct API calls:
   ```python
   data_provider = DataProvider()
   df = data_provider.get_historical_data(symbol, interval=self.config['trading']['timeframe'])
   ```

6. **Strategy Type Consideration**: This is a {strategy_type} strategy.
   - Momentum strategies: Faster entry signals, ride trends
   - Mean reversion: Wait for extremes, enter on reversals
   - Adjust indicator thresholds and timing accordingly

7. **Return Format**: The `analyze()` method must return a list of signal dicts:
   ```python
   [{{
       "symbol": "AAPL",
       "action": "BUY" or "SELL" or "HOLD",
       "confidence": 0.0-1.0,
       "indicators": {{"RSI": 35.2, "MACD": 0.5, ...}},
       "reasoning": "Brief explanation"
   }}]
   ```

## Template Reference
Here is the template strategy.py structure to follow:

```python
{template_code[:3000]}
... (template continues)
```

## Instructions
Generate a COMPLETE, PRODUCTION-READY strategy.py file that:
1. Implements the EXACT entry/exit rules listed above (same features, operators, thresholds)
2. Calculates ALL technical indicators referenced in the rules
3. Uses the DataProvider for data fetching
4. Follows the exact class structure and API
5. Includes proper error handling
6. Returns signals in the correct format

Return ONLY the raw Python source code. Do NOT wrap it in markdown code fences."""

            response = self.client.messages.create(
                model="claude-opus-4-6",
                max_tokens=settings.CLAUDE_MAX_TOKENS,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract text from response (skip ThinkingBlock objects which have
            # a .thinking attribute but no .text attribute)
            code = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    code += block.text

            # Validate the generated code is non-trivial and contains the
            # required TradingStrategy class.  A bare truthy check is not
            # sufficient — the LLM can return a near-empty text block (e.g. a
            # single newline) that passes `if code:` but produces an invalid
            # strategy.py.
            if len(code) > 500 and "class TradingStrategy" in code:
                # Post-process: guarantee async interface regardless of what
                # the LLM emitted (convert sync methods, fix signatures).
                code = self._ensure_async_interface(code)
                self._log(f"Strategy code generated ({len(code)} chars)")
                return code
            else:
                logger.warning(
                    "Strategy code validation failed: length=%d, has_class=%s",
                    len(code),
                    "class TradingStrategy" in code,
                )
                raise ValueError(
                    f"Generated strategy code failed validation: "
                    f"length={len(code)}, "
                    f"has TradingStrategy class: {'class TradingStrategy' in code}"
                )

        except Exception as e:
            logger.error("Strategy code generation failed: %s", e)
            self._log(f"Warning: Strategy code generation failed: {e}")
            # Return template as fallback (already has correct async interface)
            return self._read_template_file("strategy.py")

    async def generate_strategy_config(
        self,
        strategy_name: str,
        symbols: List[str],
        design: Dict[str, Any],
        training_results: Dict[str, Any],
        strategy_type: str,
        timeframe: str = "1d",
    ) -> str:
        """Generate custom config.json file using Claude.

        Args:
            strategy_name: Name of the strategy
            symbols: List of trading symbols
            design: Strategy design dict
            training_results: Training results
            strategy_type: Type of strategy
            timeframe: Trading timeframe (e.g. "1h", "1d", "5m")

        Returns:
            Valid JSON string for config.json
        """
        self._log(f"Generating config.json for {strategy_name}")

        try:
            # Read template for reference
            template_config = self._read_template_file("config.json")

            # Extract design parameters
            indicators = design.get("indicators", {})
            risk_params = {
                # IMPORTANT: all *_pct fields in config.json are stored as decimal fractions
                # (e.g. 0.05 means 5%). Runtime code may accept whole-percent inputs for
                # backward compatibility, but the canonical config representation is 0-1.
                "stop_loss_pct": design.get("recommended_stop_loss", 0.05),
                "trailing_stop_pct": design.get("recommended_trailing_stop", 0.03),
                "max_position_loss_pct": 0.01,
                "max_daily_loss_pct": 0.02,
            }

            # Collect feature names from extracted rules for the indicators section
            extracted = training_results.get("extracted_rules", {})
            entry_rules_data = extracted.get("entry_rules") or design.get("entry_rules", [])
            exit_rules_data = extracted.get("exit_rules") or design.get("exit_rules", [])
            rule_features = set()
            for rule in entry_rules_data + exit_rules_data:
                if isinstance(rule, dict) and "feature" in rule:
                    rule_features.add(rule["feature"])
            rule_features_str = ", ".join(sorted(rule_features)) if rule_features else "N/A"

            prompt = f"""Generate a custom config.json file for a trading strategy.

## Strategy Details
- Name: {strategy_name}
- Type: {strategy_type}
- Symbols: {', '.join(symbols)}
- Timeframe: {timeframe}
- Indicator features used in rules: {rule_features_str}

## Template Schema
Use this template structure as reference:
{template_config}

## Requirements
1. Set "strategy_name" to "{strategy_name}"
2. Set "description" to a precise, indicator-specific sentence using this pattern:
   "A {strategy_type} strategy that uses {rule_features_str} to identify trading opportunities on a {timeframe} timeframe for {', '.join(symbols)}."
   Do NOT use generic phrases like "Signal Synk", "template strategy", or "customize this".
3. Set "symbols" to {json.dumps(symbols)}
4. IMPORTANT: Set "timeframe" to "{timeframe}" in the trading section
5. Populate "indicators" section with parameters relevant to these features used in trading rules:
   {rule_features_str}
6. Set "risk_management" parameters:
   IMPORTANT: All *_pct fields MUST be decimal fractions (0-1), NOT whole percent points.
   Examples:
   - 0.05 means 5%
   - 0.10 means 10%
   Do NOT output 5.0 to mean 5%.
   - stop_loss_pct: {risk_params['stop_loss_pct']:.4f}
   - trailing_stop_pct: {risk_params['trailing_stop_pct']:.4f}
   - max_position_loss_pct: {risk_params['max_position_loss_pct']:.4f}
   - max_daily_loss_pct: {risk_params['max_daily_loss_pct']:.4f}
7. Set "max_positions" to {design.get('recommended_max_positions', 3)}
8. Set "position_size_pct" to {design.get('recommended_position_size', 0.10):.4f} (decimal fraction 0-1)
9. Keep the schedule and logging sections from the template

Return ONLY valid JSON. Do NOT wrap in markdown code fences."""

            response = self.client.messages.create(
                model="claude-opus-4-6",
                max_tokens=settings.CLAUDE_MAX_TOKENS,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract text
            config_json = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    config_json += block.text

            # Validate JSON
            if config_json:
                parsed = json.loads(config_json)  # Will raise if invalid
                # Defense-in-depth: normalize any percent-like fields to decimal fractions.
                # This prevents 0.05 -> 5.0 style unit drift from breaking verification backtests.
                parsed = self.normalize_config_percent_fractions(parsed)
                config_json = json.dumps(parsed, indent=2)
                self._log(f"Config JSON generated ({len(config_json)} chars)")
                return config_json
            else:
                raise ValueError("No config generated")

        except Exception as e:
            logger.error("Config generation failed: %s", e)
            self._log(f"Warning: Config generation failed: {e}")
            return self._read_template_file("config.json")

    @staticmethod
    def _coerce_pct_fraction(value: Any, *, default: float) -> float:
        """Coerce a percent-like value into a decimal fraction (0-1).

        Accepted inputs:
        - Decimal fractions: 0.05
        - Whole percent points: 5.0 (interpreted as 5%)
        - Strings: "5%", "0.05", "5"

        This helper is intentionally conservative: it only performs unit conversion
        for values whose absolute magnitude is > 1.
        """
        if value is None:
            return default

        # Avoid treating booleans as numbers
        if isinstance(value, bool):
            return default

        if isinstance(value, str):
            s = value.strip()
            if not s:
                return default
            try:
                if s.endswith("%"):
                    num = float(s[:-1].strip())
                    return num / 100.0
                return float(s)
            except Exception:
                return default

        try:
            v = float(value)
        except Exception:
            return default

        # If user/LLM provided percent points (e.g. 5.0 meaning 5%), convert.
        if abs(v) > 1.0:
            v = v / 100.0

        # Clamp to a sane range to prevent accidental 500% style values.
        if v > 1.0:
            return 1.0
        if v < -1.0:
            return -1.0

        return v

    @classmethod
    def normalize_config_percent_fractions(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize known percent fields in a strategy config to decimal fractions.

        This is used as a post-processing step on LLM-produced config.json to
        ensure canonical units even if the model outputs whole-percent numbers.

        Only normalizes percent-like fields in the `risk_management` section and
        `trading.position_size_pct`.
        """
        if not isinstance(config, dict):
            return config

        cfg = dict(config)  # shallow copy

        # trading.position_size_pct
        trading = cfg.get("trading")
        if isinstance(trading, dict) and "position_size_pct" in trading:
            trading2 = dict(trading)
            trading2["position_size_pct"] = cls._coerce_pct_fraction(
                trading.get("position_size_pct"),
                default=0.10,
            )
            cfg["trading"] = trading2


        # risk_management: normalize all keys ending with _pct
        risk = cfg.get("risk_management")
        if isinstance(risk, dict):
            risk2 = dict(risk)
            for k, v in list(risk2.items()):
                if isinstance(k, str) and k.endswith("_pct"):
                    # Provide reasonable defaults per key.
                    defaults = {
                        "stop_loss_pct": 0.05,
                        "take_profit_pct": 0.10,
                        "trailing_stop_pct": 0.03,
                        "max_daily_loss_pct": 0.02,
                        "max_position_loss_pct": 0.01,
                        "max_drawdown_pct": 0.10,
                    }
                    risk2[k] = cls._coerce_pct_fraction(v, default=defaults.get(k, 0.0))

            # Also normalize a few commonly-used percent-like keys that do not
            # follow the *_pct naming convention (worker-generated configs).
            #
            # Examples:
            # - max_daily_loss: 5.0  -> 0.05
            # - max_drawdown: 20     -> 0.20
            extra_defaults = {
                "max_daily_loss": 0.02,
                "max_drawdown": 0.10,
                # This is typically a fraction of equity (0-1). Treating whole
                # numbers as percent points is a safe defense-in-depth.
                "max_position_size": 0.10,
            }
            for key, default in extra_defaults.items():
                if key in risk2:
                    risk2[key] = cls._coerce_pct_fraction(risk2.get(key), default=default)
            cfg["risk_management"] = risk2

        return cfg

    async def generate_risk_manager_code(
        self,
        strategy_name: str,
        design: Dict[str, Any],
        training_results: Dict[str, Any],
        strategy_type: str,
        config: Dict[str, Any],
    ) -> str:
        """Return the canonical risk_manager.py template verbatim.

        The template already reads all risk parameters (stop_loss_pct,
        trailing_stop_pct, max_daily_loss_pct, max_drawdown_pct, etc.) from
        the config dict at runtime, so no LLM customisation is required.

        Using the template verbatim guarantees that the full interface expected
        by backtest.py is always present:
            start_new_day, register_position, unregister_position,
            update_position_price, check_position_risk,
            check_daily_loss_limit, check_drawdown_limit, can_open_new_position.

        Previous LLM-based generation was producing simplified variants that
        omitted methods from the interface, causing AttributeError at runtime.
        """
        template_code = self._read_template_file("risk_manager.py")
        self._log(
            f"risk_manager.py: using canonical template verbatim "
            f"({len(template_code)} chars)"
        )
        return template_code

    async def generate_backtest_code(
        self,
        strategy_name: str,
        symbols: List[str],
        design: Dict[str, Any],
        training_results: Dict[str, Any],
        strategy_type: str,
        config: Dict[str, Any],
    ) -> str:
        """Generate custom backtest.py file using Claude.

        Args:
            strategy_name: Name of the strategy
            symbols: List of trading symbols
            design: Strategy design dict
            training_results: Training results
            strategy_type: Type of strategy
            config: Strategy configuration

        Returns:
            Complete Python source code for backtest.py
        """
        self._log(f"Generating backtest.py for {strategy_name}")

        try:
            # Read template (it's 822 lines, so we'll include structure overview)
            template_code = self._read_template_file("backtest.py")

            prompt = f"""Generate a custom backtest.py file for a trading strategy.

## Strategy Details
- Name: {strategy_name}
- Type: {strategy_type}
- Symbols: {', '.join(symbols)}

## Required Data Structures

You MUST use these exact dataclass definitions in your backtest.py:

```python
@dataclass
class Trade:
    \"\"\"Represents a single trade in the backtest\"\"\"
    entry_date: str
    exit_date: Optional[str]
    symbol: str
    action: str  # BUY or SELL
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    pnl: float
    pnl_pct: float
    duration_hours: float
    indicators: Dict[str, Any]
    exit_reason: str = 'SIGNAL'  # SIGNAL, STOP_LOSS, TAKE_PROFIT, TRAILING_STOP, END_OF_BACKTEST

@dataclass
class BacktestResults:
    \"\"\"Complete backtest results structure - stored in database\"\"\"
    strategy_id: str
    strategy_name: str
    period: str
    start_date: str
    end_date: str

    # Performance metrics (all required)
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate_pct: float
    profit_factor: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_pnl: float
    best_trade_pnl: float
    worst_trade_pnl: float
    avg_trade_duration_hours: float

    # Trade data (stored as JSON in database)
    trades: List[Dict[str, Any]]

    # Equity curve data (stored as JSON in database)
    equity_curve: List[Dict[str, float]]

    # Period performance breakdown (stored as JSON in database)
    daily_returns: List[Dict[str, Any]]
    weekly_returns: List[Dict[str, Any]]
    monthly_returns: List[Dict[str, Any]]
    yearly_returns: List[Dict[str, Any]]

    # OHLC data for candlestick charts (stored as JSON)
    ohlc_data: List[Dict[str, Any]]

    # Metadata
    completed_at: str
    duration_seconds: float

    # Risk management statistics (optional, has default)
    risk_stats: Dict[str, Any] = None
```

## CRITICAL REQUIREMENTS

1. **Imports**: Must import `from strategy import TradingStrategy`, `from risk_manager import RiskManager`, and `from data_provider import DataProvider`

2. **Main Function**: Must have a `main()` function that runs the backtest

3. **Results File**: Must write results to `backtest_results.json` with ALL fields from BacktestResults dataclass

4. **Complete Field Requirements**: The backtest_results.json MUST include ALL of these fields:

   **Basic Info**:
   - strategy_id, strategy_name, period, start_date, end_date

   **Performance Metrics** (all required):
   - total_return_pct, max_drawdown_pct, sharpe_ratio, win_rate_pct, profit_factor

   **Trade Statistics**:
   - total_trades, winning_trades, losing_trades
   - avg_trade_pnl, best_trade_pnl, worst_trade_pnl, avg_trade_duration_hours

   **Trade Data** (trades):
   - List of trade dicts, each with: entry_date, exit_date, symbol, action, entry_price, exit_price, quantity, pnl, pnl_pct, duration_hours, indicators, exit_reason

   **Equity Curve** (equity_curve):
   - List of {{"date": "YYYY-MM-DD", "value": float}} dicts tracking portfolio value over time

   **Period Returns**:
   - daily_returns: List of {{"period": "YYYY-MM-DD", "value": float, "return_pct": float}}
   - weekly_returns: List of {{"period": "YYYY-Www", "value": float, "return_pct": float}}
   - monthly_returns: List of {{"period": "YYYY-MM", "value": float, "return_pct": float}}
   - yearly_returns: List of {{"period": "YYYY", "value": float, "return_pct": float}}

   **OHLC Data** (ohlc_data):
   - List of {{"date": "YYYY-MM-DD", "open": float, "high": float, "low": float, "close": float, "volume": int}}

   **Metadata**:
   - completed_at: ISO 8601 timestamp string
   - duration_seconds: float

   **Risk Stats** (risk_stats):
   - Dict with exit_reason_counts mapping (e.g., {{"SIGNAL": 50, "STOP_LOSS": 10, "TAKE_PROFIT": 5}})
   - Include any other risk management metrics

5. **Example JSON Output Structure**:
```json
{{
    "strategy_id": "from config",
    "strategy_name": "{strategy_name}",
    "period": "2y",
    "start_date": "2022-01-01",
    "end_date": "2024-01-01",
    "total_return_pct": 15.0,
    "max_drawdown_pct": -8.5,
    "sharpe_ratio": 1.5,
    "win_rate_pct": 55.0,
    "profit_factor": 1.8,
    "total_trades": 150,
    "winning_trades": 82,
    "losing_trades": 68,
    "avg_trade_pnl": 250.0,
    "best_trade_pnl": 2000.0,
    "worst_trade_pnl": -800.0,
    "avg_trade_duration_hours": 120.0,
    "trades": [
        {{
            "entry_date": "2022-01-15T10:30:00",
            "exit_date": "2022-01-20T15:45:00",
            "symbol": "AAPL",
            "action": "BUY",
            "entry_price": 150.0,
            "exit_price": 155.0,
            "quantity": 100,
            "pnl": 500.0,
            "pnl_pct": 3.33,
            "duration_hours": 125.25,
            "indicators": {{"rsi": 45.2, "macd": 0.5}},
            "exit_reason": "SIGNAL"
        }}
    ],
    "equity_curve": [
        {{"date": "2022-01-01", "value": 100000.0}},
        {{"date": "2022-01-02", "value": 100500.0}}
    ],
    "daily_returns": [
        {{"period": "2022-01-01", "value": 100000.0, "return_pct": 0.0}},
        {{"period": "2022-01-02", "value": 100500.0, "return_pct": 0.5}}
    ],
    "weekly_returns": [
        {{"period": "2022-W01", "value": 101000.0, "return_pct": 1.0}}
    ],
    "monthly_returns": [
        {{"period": "2022-01", "value": 103500.0, "return_pct": 3.5}}
    ],
    "yearly_returns": [
        {{"period": "2022", "value": 107000.0, "return_pct": 7.0}}
    ],
    "ohlc_data": [
        {{"date": "2022-01-01", "open": 150.0, "high": 155.0, "low": 149.0, "close": 154.0, "volume": 1000000}}
    ],
    "completed_at": "2024-01-01T12:00:00Z",
    "duration_seconds": 45.2,
    "risk_stats": {{
        "exit_reason_counts": {{"SIGNAL": 82, "STOP_LOSS": 50, "TAKE_PROFIT": 18}}
    }}
}}
```

## Strategy Type Considerations
- Strategy type is {strategy_type}
- Momentum: Longer holding periods, trend-following metrics
- Mean Reversion: Shorter holding periods, reversal metrics
- Adjust performance expectations based on strategy type

Return ONLY the raw Python source code. Do NOT wrap in markdown code fences."""

            response = self.client.messages.create(
                model="claude-opus-4-6",
                max_tokens=settings.CLAUDE_MAX_TOKENS,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract code (skip ThinkingBlock objects which lack .text)
            code = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    code += block.text

            # Validate the code is non-trivial and contains the required class.
            if len(code) > 500 and "class BacktestRunner" in code:
                self._log(f"Backtest code generated ({len(code)} chars)")
                return code
            else:
                logger.warning(
                    "Backtest code validation failed: length=%d, has_class=%s",
                    len(code),
                    "class BacktestRunner" in code,
                )
                raise ValueError(
                    f"Generated backtest code failed validation: "
                    f"length={len(code)}, "
                    f"has BacktestRunner class: {'class BacktestRunner' in code}"
                )

        except Exception as e:
            logger.error("Backtest generation failed: %s", e)
            self._log(f"Warning: Backtest generation failed: {e}")
            return self._read_template_file("backtest.py")

    def compare_backtest_results(self, generated_results: dict, training_results: dict) -> dict:
        """Compare generated backtest results with training results.

        Args:
            generated_results: Results from the verification backtest
            training_results: Original training results

        Returns:
            Dict with:
            - passed: bool - True if generated results match or exceed training
            - details: str - Human-readable comparison summary
            - metrics: dict - Side-by-side comparison of key metrics
        """
        # Extract backtest results from training_results (may be nested)
        training_bt = training_results.get('backtest_results', training_results)

        # Extract metrics with safe defaults
        gen_return = generated_results.get('total_return_pct', 0)
        train_return = training_bt.get('total_return', training_bt.get('total_return_pct', 0))

        gen_sharpe = generated_results.get('sharpe_ratio', 0)
        train_sharpe = training_bt.get('sharpe_ratio', 0)

        gen_win_rate = generated_results.get('win_rate_pct', 0)
        train_win_rate = training_bt.get('win_rate', training_bt.get('win_rate_pct', 0))

        # Pass if total return is within 10% of training or better
        return_threshold = train_return * 0.9  # 90% of training return
        passed = gen_return >= return_threshold

        metrics = {
            'total_return': {'generated': gen_return, 'training': train_return},
            'sharpe_ratio': {'generated': gen_sharpe, 'training': train_sharpe},
            'win_rate': {'generated': gen_win_rate, 'training': train_win_rate},
        }

        details = f"Generated return: {gen_return:.2f}% vs training: {train_return:.2f}% (threshold: {return_threshold:.2f}%)"

        self._log(f"Backtest comparison: {'PASSED' if passed else 'FAILED'} - {details}")

        return {'passed': passed, 'details': details, 'metrics': metrics}
