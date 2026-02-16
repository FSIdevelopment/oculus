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

    async def initialize(self):
        """Initialize Redis connection via Ngrok tunnel."""
        self.redis_client = await redis.from_url(settings.REDIS_URL)

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

    async def design_strategy(
        self,
        strategy_name: str,
        symbols: List[str],
        description: str,
        timeframe: str,
        target_return: float,
        strategy_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Call Claude API to design strategy.

        Returns a dict with keys:
        - "design": the parsed strategy design JSON
        - "thinking": concatenated thinking block text
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
            creation_guide
        )

        try:
            response = self.client.messages.create(
                model="claude-opus-4-6",
                max_tokens=settings.CLAUDE_MAX_TOKENS,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract design and thinking from response
            result = self._parse_design_response(response)
            desc_preview = result["design"].get("strategy_description", "N/A")[:100]
            self._log(f"Strategy design completed: {desc_preview}")
            return result

        except anthropic.APIError as e:
            self._log(f"Claude API error: {e}")
            raise

    async def dispatch_training_job(
        self,
        design: Dict[str, Any],
        symbols: List[str],
        timeframe: str,
        iteration_uuid: str = None,
    ) -> str:
        """Dispatch training job to Redis via REDIS_URL."""
        self._log("Dispatching training job to Redis")

        job_payload = {
            "build_id": self.build.uuid,
            "user_id": self.user.uuid,
            "design": design,
            "symbols": symbols,
            "timeframe": timeframe,
            "timestamp": datetime.utcnow().isoformat(),
            "iteration_uuid": iteration_uuid,
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

    async def listen_for_results(self, job_id: str, timeout: int = 3600) -> Dict[str, Any]:
        """Listen for training results from Redis."""
        self._log(f"Listening for results on {job_id}")

        result_key = f"result:{job_id}"
        start_time = datetime.utcnow()

        while True:
            result_data = await self.redis_client.get(result_key)
            if result_data:
                result = json.loads(result_data)
                self._log(f"Results received: {result.get('status', 'unknown')}")
                # Clean up result key to prevent stale data in future re-runs
                await self.redis_client.delete(result_key)
                return result

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
            previous_design, backtest_results, iteration, creation_guide, strategy_type or ""
        )

        try:
            response = self.client.messages.create(
                model="claude-opus-4-6",
                max_tokens=settings.CLAUDE_MAX_TOKENS,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": prompt}],
            )

            result = self._parse_design_response(response)
            self._log(f"Strategy refined (iteration {iteration})")
            return result

        except anthropic.APIError as e:
            self._log(f"Claude API error during refinement: {e}")
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
    ) -> str:
        """Build initial strategy design prompt with full features list, JSON schema, and rules."""
        features_list = "\n".join(f"  - {f}" for f in self.AVAILABLE_FEATURES)

        return f"""You are an expert quantitative trading strategy designer. Your role is to design
highly profitable algorithmic trading strategies using technical indicators and ML-optimized parameters.

## Task
Design a trading strategy with these specifications:
- Name: {strategy_name}
- Strategy Type: {strategy_type}
- Symbols: {', '.join(symbols)}
- Description: {description}
- Timeframe: {timeframe}
- Target Return: {target_return}%

## Previous Builds Context
{context}

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
    ) -> str:
        """Build strategy refinement prompt with full schema and analysis guidance."""
        # Extract key metrics from backtest results for the prompt
        total_return = backtest_results.get("total_return", 0)
        win_rate = backtest_results.get("win_rate", 0)
        max_drawdown = backtest_results.get("max_drawdown", 0)
        num_trades = backtest_results.get("total_trades", 0)

        features_list = "\n".join(f"  - {f}" for f in self.AVAILABLE_FEATURES)

        return f"""You are an expert quantitative trading strategy designer refining a strategy based on backtest results.

## Previous Design (Iteration {iteration})
{json.dumps(previous_design, indent=2)}

## Strategy Type
- Strategy Type: {strategy_type}

## Backtest Results (Iteration {iteration})
- Total Return: {total_return}%
- Win Rate: {win_rate}%
- Max Drawdown: {max_drawdown}%
- Total Trades: {num_trades}

Full results:
{json.dumps(backtest_results, indent=2)}

## Available Feature Names
You MUST select from these exact feature names:

{features_list}

## CRITICAL INSTRUCTIONS
1. Analyze what specifically failed or underperformed in the previous design and why.
2. DO NOT repeat approaches that have already been tried. Each iteration MUST try something meaningfully different.
3. If features were tried before and didn't work well, use DIFFERENT features.
4. If thresholds were too tight or too loose, adjust them significantly (not just +-5%).
5. Consider fundamentally different strategy approaches if incremental changes aren't working.

## Analysis Questions (answer these in your thinking before designing)
1. What pattern do you see in the results? Is the strategy improving or stuck?
2. Which features contributed most to winning trades vs losing trades?
3. Are the entry rules too loose (too many bad entries) or too tight (missing opportunities)?
4. Are the exit rules triggering too early (cutting winners) or too late (letting losers run)?
5. What HASN'T been tried yet that could work better?
6. Should the ML training parameters (forward_returns_days, profit_thresholds) be changed?

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

Make meaningful changes — don't just tweak numbers slightly. Return a COMPLETELY REVISED strategy design.
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

    def _parse_design_response(self, response) -> Dict[str, Any]:
        """Parse Claude response to extract design JSON and thinking content.

        Returns a dict with keys:
        - "design": the parsed strategy design JSON
        - "thinking": concatenated thinking block text (empty string if none)
        """
        # Extract text and thinking blocks from response
        text = ""
        thinking_parts: List[str] = []
        for block in response.content:
            if block.type == "thinking" and hasattr(block, "thinking"):
                thinking_parts.append(block.thinking)
            elif hasattr(block, "text"):
                text += block.text

        thinking_text = "\n".join(thinking_parts)

        # Step 1: Try to match ```json ... ``` code fence (most reliable)
        design = None
        json_fence_match = re.search(r"```json\s*([\s\S]*?)```", text)
        if json_fence_match:
            try:
                design = json.loads(json_fence_match.group(1).strip())
                # Validate rules are non-empty
                if self._validate_design_rules(design):
                    return {"design": design, "thinking": thinking_text}
            except json.JSONDecodeError:
                pass

        # Step 2: Try matching outermost JSON object (less greedy)
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                design = json.loads(json_match.group())
                if self._validate_design_rules(design):
                    return {"design": design, "thinking": thinking_text}
                # Design parsed but rules are empty — will retry below
            except json.JSONDecodeError:
                pass

        # If we got a design but rules are empty, retry once
        if design and not self._validate_design_rules(design):
            logger.warning("Parsed design has blank rules — retrying Claude once")
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
                        return {"design": retry_design, "thinking": thinking_text}
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
Best Model: {training_results.get("best_model", "Unknown")}

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
    ) -> str:
        """Generate Claude's explanation of which iteration was selected and why.

        Args:
            completed_iterations: List of all completed BuildIteration records
            best_iteration: The BuildIteration record that was selected as best
            target_return: The target return percentage that was not met
            strategy_type: The type of strategy being built

        Returns:
            A thinking string explaining the selection decision
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

            thinking_text = "\n".join(b.thinking for b in thinking_blocks) if thinking_blocks else ""
            response_text = "\n".join(b.text for b in text_blocks)

            # Combine thinking and response
            full_thinking = f"{thinking_text}\n\n{response_text}" if thinking_text else response_text

            self._log(f"Selection thinking generated ({len(full_thinking)} chars)")
            return full_thinking

        except Exception as e:
            logger.warning("Selection thinking generation failed: %s", e)
            self._log(f"Warning: Selection thinking generation failed: {e}")
            # Return a simple fallback explanation
            return f"Selected iteration {best_iteration.iteration_number} as it achieved the highest total return of {best_bt.get('total_return', 0):.2f}% among all {len(completed_iterations)} completed iterations."

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

            # Build prompt
            entry_rules = json.dumps(design.get("entry_rules", []), indent=2)
            exit_rules = json.dumps(design.get("exit_rules", []), indent=2)
            features = design.get("priority_features", [])
            indicators = design.get("indicators", {})

            prompt = f"""You are generating a complete custom trading strategy file (strategy.py) for a production trading system.

## Strategy Specifications
- Name: {strategy_name}
- Type: {strategy_type}
- Symbols: {', '.join(symbols)}
- Description: {design.get("strategy_description", "N/A")}

## Entry Rules
{entry_rules}
Entry Score Threshold: {design.get("entry_score_threshold", 3)}

## Exit Rules
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

1. **Class Structure**: The class MUST be named `TradingStrategy` with this exact API:
   - `__init__(self)` - Load config from config.json
   - `analyze(self, symbols: List[str]) -> List[Dict]` - Main analysis method that returns signals

2. **Imports**: You MUST use these imports:
   - `from data_provider import DataProvider` for fetching market data
   - `import json` to load config.json
   - Standard libraries: pandas, numpy, ta (technical analysis)

3. **Data Fetching**: Use `DataProvider` class, NOT direct API calls:
   ```python
   data_provider = DataProvider()
   df = data_provider.get_historical_data(symbol, interval=self.config['trading']['timeframe'])
   ```

4. **Strategy Type Consideration**: This is a {strategy_type} strategy.
   - Momentum strategies: Faster entry signals, ride trends
   - Mean reversion: Wait for extremes, enter on reversals
   - Adjust indicator thresholds and timing accordingly

5. **Return Format**: The `analyze()` method must return a list of signal dicts:
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
1. Implements the entry/exit rules from the design
2. Calculates the required technical indicators
3. Uses the DataProvider for data fetching
4. Follows the exact class structure and API
5. Includes proper error handling
6. Returns signals in the correct format

Return ONLY the raw Python source code. Do NOT wrap it in markdown code fences."""

            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=settings.CLAUDE_MAX_TOKENS,
                thinking={"type": "adaptive", "budget_tokens": settings.CLAUDE_MAX_TOKENS - 2000},
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract text from response
            code = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    code += block.text

            if code:
                self._log(f"Strategy code generated ({len(code)} chars)")
                return code
            else:
                raise ValueError("No code generated in response")

        except Exception as e:
            logger.warning("Strategy code generation failed: %s", e)
            self._log(f"Warning: Strategy code generation failed: {e}")
            # Return template as fallback
            return self._read_template_file("strategy.py")

    async def generate_strategy_config(
        self,
        strategy_name: str,
        symbols: List[str],
        design: Dict[str, Any],
        training_results: Dict[str, Any],
        strategy_type: str,
    ) -> str:
        """Generate custom config.json file using Claude.

        Args:
            strategy_name: Name of the strategy
            symbols: List of trading symbols
            design: Strategy design dict
            training_results: Training results
            strategy_type: Type of strategy

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
                "stop_loss_pct": design.get("recommended_stop_loss", 0.05) * 100,
                "trailing_stop_pct": design.get("recommended_trailing_stop", 0.03) * 100,
                "max_position_loss_pct": 1.0,
                "max_daily_loss_pct": 2.0,
            }

            prompt = f"""Generate a custom config.json file for a trading strategy.

## Strategy Details
- Name: {strategy_name}
- Type: {strategy_type}
- Symbols: {', '.join(symbols)}

## Template Schema
Use this template structure as reference:
{template_config}

## Requirements
1. Set "strategy_name" to "{strategy_name}"
2. Set "symbols" to {json.dumps(symbols)}
3. Populate "indicators" section with parameters from the design:
   - RSI period: {design.get('entry_rules', [{}])[0].get('threshold', 14) if design.get('entry_rules') else 14}
   - SMA/EMA periods from the priority features
   - Any other indicator parameters mentioned in the design
4. Set "risk_management" parameters:
   - stop_loss_pct: {risk_params['stop_loss_pct']:.1f}
   - trailing_stop_pct: {risk_params['trailing_stop_pct']:.1f}
   - max_position_loss_pct: {risk_params['max_position_loss_pct']:.1f}
   - max_daily_loss_pct: {risk_params['max_daily_loss_pct']:.1f}
5. Set "max_positions" to {design.get('recommended_max_positions', 3)}
6. Set "position_size_pct" to {design.get('recommended_position_size', 0.10) * 100:.0f}
7. Keep the schedule, data_source, and logging sections from the template

Return ONLY valid JSON. Do NOT wrap in markdown code fences."""

            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=settings.CLAUDE_MAX_TOKENS,
                thinking={"type": "adaptive", "budget_tokens": settings.CLAUDE_MAX_TOKENS - 2000},
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract text
            config_json = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    config_json += block.text

            # Validate JSON
            if config_json:
                json.loads(config_json)  # Will raise if invalid
                self._log(f"Config JSON generated ({len(config_json)} chars)")
                return config_json
            else:
                raise ValueError("No config generated")

        except Exception as e:
            logger.warning("Config generation failed: %s", e)
            self._log(f"Warning: Config generation failed: {e}")
            return self._read_template_file("config.json")

    async def generate_risk_manager_code(
        self,
        strategy_name: str,
        design: Dict[str, Any],
        training_results: Dict[str, Any],
        strategy_type: str,
        config: Dict[str, Any],
    ) -> str:
        """Generate custom risk_manager.py file using Claude.

        Args:
            strategy_name: Name of the strategy
            design: Strategy design dict with risk parameters
            training_results: Training results
            strategy_type: Type of strategy
            config: Strategy configuration

        Returns:
            Complete Python source code for risk_manager.py
        """
        self._log(f"Generating risk_manager.py for {strategy_name}")

        try:
            # Read template
            template_code = self._read_template_file("risk_manager.py")

            # Extract risk parameters
            stop_loss = design.get("recommended_stop_loss", 0.05)
            trailing_stop = design.get("recommended_trailing_stop", 0.03)
            max_drawdown = training_results.get('backtest_results', {}).get('max_drawdown', 10)

            prompt = f"""Generate a custom risk_manager.py file for a trading strategy.

## Strategy Details
- Name: {strategy_name}
- Type: {strategy_type}
- Stop Loss: {stop_loss * 100:.1f}%
- Trailing Stop: {trailing_stop * 100:.1f}%
- Max Drawdown (from training): {max_drawdown:.1f}%

## Risk Parameters from Design
- Stop Loss: {stop_loss} (as decimal)
- Trailing Stop: {trailing_stop} (as decimal)
- Max Positions: {design.get('recommended_max_positions', 3)}
- Position Size: {design.get('recommended_position_size', 0.10)}

## Strategy Type Considerations
This is a {strategy_type} strategy. Adjust risk tolerances accordingly:
- Momentum: May need wider stops to avoid whipsaws
- Mean Reversion: Tighter stops, faster exits
- Trend Following: Trailing stops are critical

## Template Reference
Here is the template risk_manager.py structure:

```python
{template_code[:2000]}
... (template continues)
```

## CRITICAL REQUIREMENTS
1. Class MUST be named `RiskManager`
2. Must match the template's interface (same methods and signatures)
3. Implement position sizing based on the design parameters
4. Implement stop loss and trailing stop logic
5. Include max drawdown protection
6. Load config from config.json

Return ONLY the raw Python source code. Do NOT wrap in markdown code fences."""

            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=settings.CLAUDE_MAX_TOKENS,
                thinking={"type": "adaptive", "budget_tokens": settings.CLAUDE_MAX_TOKENS - 2000},
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract code
            code = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    code += block.text

            if code:
                self._log(f"Risk manager code generated ({len(code)} chars)")
                return code
            else:
                raise ValueError("No code generated")

        except Exception as e:
            logger.warning("Risk manager generation failed: %s", e)
            self._log(f"Warning: Risk manager generation failed: {e}")
            return self._read_template_file("risk_manager.py")

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

## Template Structure Overview
The template backtest.py is 822 lines and includes:
- Imports: strategy, risk_manager, data_provider
- BacktestEngine class with position tracking
- Performance metrics calculation
- Results written to backtest_results.json

Key structure from template:
```python
{template_code[:1500]}
... (continues with backtest logic, metrics calculation)
```

## CRITICAL REQUIREMENTS
1. Must import: `from strategy import TradingStrategy` and `from risk_manager import RiskManager`
2. Must use: `from data_provider import DataProvider` for fetching historical data
3. Must write results to `backtest_results.json` with these fields:
   - total_return_pct
   - win_rate_pct
   - sharpe_ratio
   - max_drawdown_pct
   - total_trades
   - winning_trades
   - losing_trades
4. Must have a `main()` function that runs the backtest
5. Strategy type is {strategy_type} - adjust backtest parameters accordingly

## Strategy Type Considerations
- Momentum: Longer holding periods, trend-following metrics
- Mean Reversion: Shorter holding periods, reversal metrics
- Adjust performance expectations based on strategy type

Return ONLY the raw Python source code. Do NOT wrap in markdown code fences."""

            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=settings.CLAUDE_MAX_TOKENS,
                thinking={"type": "adaptive", "budget_tokens": settings.CLAUDE_MAX_TOKENS - 2000},
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract code
            code = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    code += block.text

            if code:
                self._log(f"Backtest code generated ({len(code)} chars)")
                return code
            else:
                raise ValueError("No code generated")

        except Exception as e:
            logger.warning("Backtest generation failed: %s", e)
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
