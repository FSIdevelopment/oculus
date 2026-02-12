"""Build orchestrator service for strategy design and training."""
import json
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import anthropic
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.strategy_build import StrategyBuild
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

        # Create prompt
        prompt = self._build_design_prompt(
            strategy_name, symbols, description, timeframe, target_return, context
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
        }
        
        # Push to Redis queue
        job_id = f"build:{self.build.uuid}"
        await self.redis_client.set(job_id, json.dumps(job_payload), ex=86400)  # 24h TTL
        await self.redis_client.lpush("training_queue", job_id)
        
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
    ) -> Dict[str, Any]:
        """Refine strategy based on backtest results.

        Returns a dict with keys:
        - "design": the parsed strategy design JSON
        - "thinking": concatenated thinking block text
        """
        self._log(f"Refining strategy (iteration {iteration})")

        prompt = self._build_refinement_prompt(
            previous_design, backtest_results, iteration
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
        symbols: List[str],
        description: str,
        timeframe: str,
        target_return: float,
        context: str,
    ) -> str:
        """Build initial strategy design prompt with full features list, JSON schema, and rules."""
        features_list = "\n".join(f"  - {f}" for f in self.AVAILABLE_FEATURES)

        return f"""You are an expert quantitative trading strategy designer. Your role is to design
highly profitable algorithmic trading strategies using technical indicators and ML-optimized parameters.

## Task
Design a trading strategy with these specifications:
- Name: {strategy_name}
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
"""
    
    def _build_refinement_prompt(
        self,
        previous_design: Dict[str, Any],
        backtest_results: Dict[str, Any],
        iteration: int,
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
- Design 6-10 entry rules and 4-8 exit rules
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
"""
    
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

        # Find JSON in response
        import re
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                design = json.loads(json_match.group())
                return {"design": design, "thinking": thinking_text}
            except json.JSONDecodeError:
                pass

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
                "entry_rules": [],
                "entry_score_threshold": 3,
                "exit_rules": [],
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

