#!/usr/bin/env python3
"""
LLM Strategy Designer - Uses Claude API to design trading strategies.

When --LLM flag is provided, this module calls Claude (claude-opus-4-6 with
extended thinking) to design the strategy before ML training. Claude outputs:
- Entry/exit indicator rules and thresholds
- High-level strategy design
- Features and parameters for ML training

The ML pipeline then trains models to find optimal weights for Claude's design.

Author: SignalSynk AI
"""

import os
import json
import time
import re
import importlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

import requests

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class LLMStrategyDesign:
    """Structured output from Claude's strategy design."""

    # Strategy overview
    strategy_description: str = ""
    strategy_rationale: str = ""

    # ML Training Parameters (maps to EnhancedMLConfig)
    forward_returns_days_options: List[int] = field(default_factory=lambda: [2, 3, 5, 7, 10, 15, 20, 30])
    profit_threshold_options: List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
    recommended_n_iter_search: int = 50

    # Feature Selection (subset of the 60+ available indicators)
    priority_features: List[str] = field(default_factory=list)
    feature_rationale: Dict[str, str] = field(default_factory=dict)

    # Entry Rules
    entry_rules: List[Dict[str, Any]] = field(default_factory=list)
    entry_score_threshold: int = 3

    # Exit Rules
    exit_rules: List[Dict[str, Any]] = field(default_factory=list)
    exit_score_threshold: int = 3

    # Risk Parameters
    recommended_stop_loss: float = 0.05
    recommended_trailing_stop: float = 0.03
    recommended_max_positions: int = 3
    recommended_position_size: float = 0.1

    # Lookback window — minimum bars before trading starts (for indicator stabilization)
    lookback_window: int = 200

    # ML Model Training Configuration
    config_search_model: str = "LightGBM"  # Model for label config search
    train_neural_networks: bool = True
    train_lstm: bool = True
    cv_folds: int = 3
    nn_epochs: int = 100
    nn_batch_size: int = 64
    nn_learning_rate: float = 0.001
    nn_hidden_layers: List[List[int]] = field(default_factory=lambda: [[128, 64], [256, 128, 64]])
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_sequence_length: int = 100
    lstm_epochs: int = 250
    lstm_batch_size: int = 32

    # Raw Claude response for debugging
    raw_response: str = ""


# =============================================================================
# AlphaVantage Tool Integration
# =============================================================================

class AlphaVantageTools:
    """Tool definitions and execution for AlphaVantage data access via Claude API."""

    API_KEY = "PLACEHOLDER_ROTATE_ME"
    BASE_URL = "https://www.alphavantage.co/query"
    _last_call_time = 0

    @staticmethod
    def get_tool_definitions() -> List[Dict]:
        """Return tool definitions for Claude API."""
        return [
            {
                "name": "get_stock_overview",
                "description": "Get fundamental data and company overview for a stock symbol. Returns key metrics like market cap, P/E ratio, EPS, sector, industry, and recent price data.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock ticker symbol (e.g., NVDA, AMD)"}
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "get_technical_indicator",
                "description": "Get a specific technical indicator time series for a stock. Available indicators: SMA, EMA, RSI, MACD, BBANDS, STOCH, ADX, CCI, AROON, MFI, OBV, AD, VWAP, ATR, WILLR.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock ticker symbol"},
                        "function": {
                            "type": "string",
                            "description": "Indicator function name: SMA, EMA, RSI, MACD, BBANDS, STOCH, ADX, CCI, AROON, MFI, OBV, AD, VWAP, ATR, WILLR"
                        },
                        "interval": {
                            "type": "string",
                            "enum": ["daily", "weekly", "monthly"],
                            "description": "Time interval"
                        },
                        "time_period": {
                            "type": "integer",
                            "description": "Number of periods for the indicator (e.g., 14 for RSI)"
                        },
                        "series_type": {
                            "type": "string",
                            "enum": ["close", "open", "high", "low"],
                            "description": "Price series to use"
                        }
                    },
                    "required": ["symbol", "function", "interval"]
                }
            },
            {
                "name": "get_daily_prices",
                "description": "Get daily OHLCV price data for a stock. Returns Open, High, Low, Close, Volume for the last 100 trading days (compact) or 20+ years (full).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock ticker symbol"},
                        "outputsize": {
                            "type": "string",
                            "enum": ["compact", "full"],
                            "description": "compact=last 100 data points, full=full 20+ year history"
                        }
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "get_sector_performance",
                "description": "Get real-time and historical S&P 500 sector performance across multiple timeframes (1D, 5D, 1M, 3M, YTD, 1Y, 3Y, 5Y, 10Y).",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        ]

    @classmethod
    def _rate_limit(cls):
        """Enforce AlphaVantage rate limit (30 calls/minute)."""
        now = time.time()
        elapsed = now - cls._last_call_time
        if elapsed < 5:
            time.sleep(5 - elapsed)
        cls._last_call_time = time.time()

    @classmethod
    def execute_tool(cls, tool_name: str, tool_input: Dict) -> str:
        """Execute a tool call and return the result as a string."""
        cls._rate_limit()

        try:
            if tool_name == "get_stock_overview":
                params = {
                    "function": "OVERVIEW",
                    "symbol": tool_input["symbol"],
                    "apikey": cls.API_KEY
                }
            elif tool_name == "get_technical_indicator":
                params = {
                    "function": tool_input["function"],
                    "symbol": tool_input["symbol"],
                    "interval": tool_input["interval"],
                    "apikey": cls.API_KEY
                }
                if "time_period" in tool_input:
                    params["time_period"] = tool_input["time_period"]
                if "series_type" in tool_input:
                    params["series_type"] = tool_input["series_type"]
            elif tool_name == "get_daily_prices":
                params = {
                    "function": "TIME_SERIES_DAILY",
                    "symbol": tool_input["symbol"],
                    "outputsize": tool_input.get("outputsize", "compact"),
                    "apikey": cls.API_KEY
                }
            elif tool_name == "get_sector_performance":
                params = {
                    "function": "SECTOR",
                    "apikey": cls.API_KEY
                }
            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})

            response = requests.get(cls.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Truncate large responses to avoid token bloat
            result_str = json.dumps(data, indent=2)
            if len(result_str) > 8000:
                # For price data, keep only recent entries
                if "Time Series (Daily)" in data:
                    ts = data["Time Series (Daily)"]
                    keys = sorted(ts.keys(), reverse=True)[:30]  # Last 30 days
                    data["Time Series (Daily)"] = {k: ts[k] for k in keys}
                    data["_note"] = "Truncated to last 30 trading days"
                    result_str = json.dumps(data, indent=2)
                elif len(result_str) > 8000:
                    result_str = result_str[:8000] + "\n... (truncated)"

            return result_str

        except requests.RequestException as e:
            return json.dumps({"error": f"API request failed: {str(e)}"})
        except Exception as e:
            return json.dumps({"error": f"Tool execution failed: {str(e)}"})


# =============================================================================
# Available Features Reference
# =============================================================================

# These are the exact feature names that FeatureEngineer.calculate_all_indicators()
# and add_pattern_features() produce. Claude must select from these.
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


# =============================================================================
# LLM Strategy Designer
# =============================================================================

class LLMStrategyDesigner:
    """Uses Claude API to design trading strategies before ML training."""

    # Claude model for strategy design
    MODEL = "claude-opus-4-6"
    MAX_TOKENS = 16000
    THINKING_BUDGET = 10000
    MAX_TOOL_CALLS = 60  # Limit tool calls to avoid excessive API usage

    def __init__(self, strategy_name: str, symbols: List[str],
                 description: str, timeframe: str, target: float,
                 history_manager=None, feature_tracker=None):
        if not HAS_ANTHROPIC:
            raise ImportError(
                "anthropic package is required for --LLM flag. "
                "Install with: pip install anthropic"
            )

        self.strategy_name = strategy_name
        self.symbols = symbols
        self.description = description
        self.timeframe = timeframe
        self.target = target
        self.client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY from env
        self.indicators_reference = self._load_indicators_reference()
        self.design_history: List[Dict[str, Any]] = []  # Track all previous designs

        # Cross-build memory: history of past strategy builds and feature effectiveness
        from build_history import BuildHistoryManager, FeatureTracker, infer_asset_class
        self.history_manager = history_manager or BuildHistoryManager()
        self.feature_tracker = feature_tracker or FeatureTracker()
        self.asset_class = infer_asset_class(symbols)

    def _load_indicators_reference(self) -> str:
        """Load indicator names and descriptions for Claude's reference."""
        try:
            spec = importlib.util.spec_from_file_location(
                "technical_indicators_index",
                Path(__file__).parent / "technical_indicators_index.py"
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            indicators = mod.INDICATORS

            # Build a concise reference
            lines = ["Available Technical Indicators:"]
            for key, info in indicators.items():
                name = info.get("name", key)
                indicator_type = info.get("type", "unknown")
                default_period = info.get("default_period", "N/A")
                signals = info.get("signals", {})
                buy_signal = ""
                sell_signal = ""
                for market_type, sigs in signals.items():
                    if isinstance(sigs, dict):
                        buy_signal = sigs.get("buy", "")
                        sell_signal = sigs.get("sell", "")
                        break

                lines.append(
                    f"- {key} ({name}): type={indicator_type}, "
                    f"period={default_period}, "
                    f"buy='{buy_signal[:80]}', sell='{sell_signal[:80]}'"
                )

            return "\n".join(lines)

        except Exception as e:
            print(f"  Warning: Could not load indicators reference: {e}")
            return "Indicators reference unavailable."

    def _load_creation_guide(self) -> str:
        """Load STRATEGY_CREATION_GUIDE.md if it exists, for inclusion in prompts."""
        guide_path = Path(__file__).parent / "STRATEGY_CREATION_GUIDE.md"
        try:
            if guide_path.exists():
                content = guide_path.read_text()
                if content.strip():
                    return (
                        "\n## Strategy Creation Guide (Lessons from Previous Builds)\n"
                        "The following guide contains proven patterns and settings from "
                        "previously successful strategies. Follow these lessons:\n\n"
                        f"{content}\n"
                    )
        except Exception as e:
            print(f"  Warning: Could not load creation guide: {e}")
        return ""

    def _build_history_context(self) -> str:
        """Load relevant past build data for inclusion in prompts."""
        try:
            relevant_builds = self.history_manager.get_relevant_builds(
                symbols=self.symbols,
                asset_class=self.asset_class,
                max_results=5,
            )

            if not relevant_builds:
                return ""

            builds_text = self.history_manager.format_for_prompt(relevant_builds)
            features_text = self.feature_tracker.format_for_prompt(
                asset_class=self.asset_class, top_n=15
            )

            sections = [
                "\n## Past Strategy Build History",
                "The following are structured results from previous strategy builds. "
                "Use these to inform your design — learn from what worked and avoid what failed.\n",
                builds_text,
            ]

            if features_text:
                sections.extend([
                    "## Feature Effectiveness Across Past Builds",
                    "Features ranked by usage frequency, with associated returns and thresholds:\n",
                    features_text,
                ])

            return "\n".join(sections)

        except Exception as e:
            print(f"  Warning: Could not load build history context: {e}")
            return ""

    def _build_system_prompt(self) -> str:
        """Build the system prompt with available indicators and output format."""
        features_list = "\n".join(f"  - {f}" for f in AVAILABLE_FEATURES)

        return f"""You are an expert quantitative trading strategy designer. Your role is to design
highly profitable algorithmic trading strategies using technical indicators and ML-optimized parameters.

## Available Technical Indicators Reference
{self.indicators_reference}

## Available Feature Names for ML Training
The ML pipeline calculates the following features from price data. You MUST select from these
exact feature names when specifying priority_features and entry/exit rules:

{features_list}

## Output Format
You MUST respond with a JSON block wrapped in ```json ... ``` tags containing your strategy design.
The JSON must follow this exact schema:

```json
{{
  "strategy_description": "Brief description of the strategy approach",
  "strategy_rationale": "Detailed rationale for why this strategy should work for these symbols",
  "forward_returns_days_options": [5, 10, 15],
  "profit_threshold_options": [2.0, 3.0, 5.0],
  "recommended_n_iter_search": 50,
  "priority_features": ["RSI_14", "MACD_HIST", "ADX", "BB_PCT_B"],
  "feature_rationale": {{
    "RSI_14": "Why this feature matters for this strategy",
    "MACD_HIST": "..."
  }},
  "entry_rules": [
    {{
      "feature": "RSI_14",
      "operator": "<=",
      "threshold": 35.0,
      "description": "Enter when RSI indicates oversold conditions"
    }}
  ],
  "entry_score_threshold": 3,
  "exit_rules": [
    {{
      "feature": "RSI_14",
      "operator": ">=",
      "threshold": 70.0,
      "description": "Exit when RSI indicates overbought conditions"
    }}
  ],
  "exit_score_threshold": 3,
  "recommended_stop_loss": 0.05,
  "recommended_trailing_stop": 0.03,
  "recommended_max_positions": 3,
  "recommended_position_size": 0.10,
  "lookback_window": 200,
  "ml_training_config": {{
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
  }}
}}
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
- Use your tools to research the current market conditions and characteristics of the given symbols
- Base your strategy design on real data and proven technical analysis principles
- The ML pipeline will find the optimal weights - you provide the structure and rules
- Think deeply about what indicators work best for the specific symbols and strategy type
{self._load_creation_guide()}
{self._build_history_context()}
"""

    def _build_initial_prompt(self) -> str:
        """Build the initial user message with the required prompt format."""
        symbols_str = ", ".join(self.symbols)
        return (
            f"You are a quantitative trading strategy expert which is designing a highly profitable "
            f"strategy called {self.strategy_name} which will trade the symbols {symbols_str}. "
            f"The kind of strategy we are trying to build is {self.description}. "
            f"The goal of the strategy is to achieve a minimum profit of {self.target}% annually "
            f"trading on the timeframe of {self.timeframe}. Design a strategy to achieve the target. "
            f"You have access to Massive.com MCP servers to get any financial data you need for any stock. "
            f"You also have access to AlphaVantage with the api key PLACEHOLDER_ROTATE_ME as well for more "
            f"financial market data and technical indicators.\n\n"
            f"Please research the symbols using the available tools, then design a complete strategy "
            f"and return your design as a JSON block."
        )

    def _build_refinement_prompt(self, backtest_results: Dict,
                                  iteration: int,
                                  previous_design: LLMStrategyDesign,
                                  iteration_history: Optional[List[Dict]] = None) -> str:
        """Build refinement prompt with full history to avoid duplicating efforts."""
        # Format previous entry/exit rules
        prev_entry_rules = json.dumps(previous_design.entry_rules, indent=2)
        prev_exit_rules = json.dumps(previous_design.exit_rules, indent=2)

        # Format backtest results
        total_return = backtest_results.get('total_return', 0)
        capture_rate = backtest_results.get('capture_rate', 0)
        win_rate = backtest_results.get('win_rate', 0)
        max_drawdown = backtest_results.get('max_drawdown', 0)
        num_trades = backtest_results.get('total_trades', 0)

        # Exit reason breakdown
        exit_reasons = backtest_results.get('exit_reasons', {})
        exit_breakdown = "\n".join(f"  - {reason}: {count}" for reason, count in exit_reasons.items())
        if not exit_breakdown:
            exit_breakdown = "  (no exit reason breakdown available)"

        # Per-symbol performance
        symbol_perf = backtest_results.get('symbol_performance', {})
        symbol_breakdown = "\n".join(
            f"  - {sym}: return={data.get('return', 0):.2f}%, "
            f"trades={data.get('trades', 0)}, "
            f"win_rate={data.get('win_rate', 0):.1f}%"
            for sym, data in symbol_perf.items()
        )
        if not symbol_breakdown:
            symbol_breakdown = "  (no per-symbol breakdown available)"

        # Build full iteration history section
        history_section = ""
        if self.design_history:
            history_lines = ["## Complete Iteration History (DO NOT repeat these approaches)"]
            for hist in self.design_history:
                h_iter = hist['iteration']
                h_results = hist['results']
                h_design = hist['design']
                history_lines.append(f"\n### Iteration {h_iter}")
                history_lines.append(f"- Capture Rate: {h_results.get('capture_rate', 0):.1f}%")
                history_lines.append(f"- Total Return: {h_results.get('total_return', 0):.2f}%")
                history_lines.append(f"- Win Rate: {h_results.get('win_rate', 0):.1f}%")
                history_lines.append(f"- Max Drawdown: {h_results.get('max_drawdown', 0):.2f}%")
                history_lines.append(f"- Total Trades: {h_results.get('total_trades', 0)}")
                history_lines.append(f"- Strategy Description: {h_design.get('strategy_description', 'N/A')[:200]}")
                history_lines.append(f"- Priority Features Used: {h_design.get('priority_features', [])}")
                history_lines.append(f"- Entry Rules: {json.dumps(h_design.get('entry_rules', []))[:500]}")
                history_lines.append(f"- Exit Rules: {json.dumps(h_design.get('exit_rules', []))[:500]}")
                history_lines.append(f"- Entry Score Threshold: {h_design.get('entry_score_threshold', 'N/A')}")
                history_lines.append(f"- Exit Score Threshold: {h_design.get('exit_score_threshold', 'N/A')}")
                history_lines.append(f"- ML Forward Days: {h_design.get('forward_returns_days_options', [])}")
                history_lines.append(f"- ML Profit Thresholds: {h_design.get('profit_threshold_options', [])}")
                history_lines.append(f"- Stop Loss: {h_design.get('recommended_stop_loss', 0)*100:.1f}%")
                history_lines.append(f"- Trailing Stop: {h_design.get('recommended_trailing_stop', 0)*100:.1f}%")
                history_lines.append(f"- Max Positions: {h_design.get('recommended_max_positions', 'N/A')}")
            history_section = "\n".join(history_lines)

        # Build ML training metrics from iteration_history
        ml_metrics_section = ""
        if iteration_history:
            ml_lines = ["\n## ML Training Metrics Per Iteration"]
            for ir in iteration_history:
                ml_lines.append(
                    f"- Iter {ir['iteration']}: {ir['training_years']}yrs data, "
                    f"{ir['hp_iterations']} HP iters, "
                    f"F1={ir.get('model_f1', 0):.4f}, "
                    f"Capture={ir['capture_rate']:.1f}%, "
                    f"Return={ir['total_return']:.2f}%"
                )
            ml_metrics_section = "\n".join(ml_lines)

        return f"""You are refining a trading strategy called {self.strategy_name} for symbols {', '.join(self.symbols)}.
The strategy description is: {self.description}
Target: {self.target}% annual return on timeframe {self.timeframe}.

{history_section}

## Latest Iteration ({iteration}) Results
- Total Return: {total_return:.2f}%
- Capture Rate: {capture_rate:.1f}% (target: {self.target}%)
- Win Rate: {win_rate:.1f}%
- Max Drawdown: {max_drawdown:.2f}%
- Total Trades: {num_trades}

### Exit Reason Breakdown
{exit_breakdown}

### Per-Symbol Performance
{symbol_breakdown}
{ml_metrics_section}

## Latest Strategy Design (Iteration {iteration})
- Strategy Rationale: {previous_design.strategy_rationale[:300] if previous_design.strategy_rationale else 'N/A'}
- Priority Features: {previous_design.priority_features}
- Forward Returns Days: {previous_design.forward_returns_days_options}
- Profit Thresholds: {previous_design.profit_threshold_options}
- Entry Rules: {prev_entry_rules}
- Entry Score Threshold: {previous_design.entry_score_threshold}
- Exit Rules: {prev_exit_rules}
- Exit Score Threshold: {previous_design.exit_score_threshold}
- Stop Loss: {previous_design.recommended_stop_loss*100:.1f}%
- Trailing Stop: {previous_design.recommended_trailing_stop*100:.1f}%
- Max Positions: {previous_design.recommended_max_positions}
- Position Size: {previous_design.recommended_position_size*100:.1f}%

{self._build_history_context()}

## CRITICAL INSTRUCTIONS
1. You have the COMPLETE history of all previous designs and their results above.
2. DO NOT repeat any approach that has already been tried. Each iteration MUST try something meaningfully different.
3. Analyze what specifically failed in each previous attempt and why.
4. If features were tried before and didn't work well, use DIFFERENT features.
5. If thresholds were too tight or too loose, adjust them significantly (not just +-5%).
6. Consider fundamentally different strategy approaches if incremental changes aren't working.

## Analysis Questions (answer these in your thinking before designing)
1. What pattern do you see across all iterations? Is the strategy improving or stuck?
2. Which features contributed most to winning trades vs losing trades?
3. Are the entry rules too loose (too many bad entries) or too tight (missing opportunities)?
4. Are the exit rules triggering too early (cutting winners) or too late (letting losers run)?
5. What HASN'T been tried yet that could work better?
6. Should the ML training parameters (forward_returns_days, profit_thresholds) be changed?

Use the tools to research current market conditions if needed, then return a COMPLETELY REVISED
strategy design as a JSON block. Make meaningful changes - don't just tweak numbers slightly.
"""

    def design_strategy(self) -> LLMStrategyDesign:
        """Initial strategy design call to Claude with tool use loop."""
        print(f"\n{'='*60}")
        print("LLM STRATEGY DESIGN (Claude API)")
        print(f"{'='*60}")
        print(f"  Model: {self.MODEL}")
        print(f"  Strategy: {self.strategy_name}")
        print(f"  Symbols: {', '.join(self.symbols)}")
        print(f"  Description: {self.description}")
        print(f"  Target: {self.target}% annually")
        print(f"  Timeframe: {self.timeframe}")
        print(f"  Thinking budget: {self.THINKING_BUDGET} tokens")

        messages = [{"role": "user", "content": self._build_initial_prompt()}]
        return self._run_claude_conversation(messages)

    def refine_strategy(self, backtest_results: Dict,
                        iteration: int,
                        previous_design: LLMStrategyDesign,
                        iteration_history: Optional[List[Dict]] = None) -> LLMStrategyDesign:
        """Refine strategy based on backtest results and full iteration history."""
        print(f"\n{'='*60}")
        print(f"LLM STRATEGY REFINEMENT (Iteration {iteration + 1})")
        print(f"{'='*60}")
        print(f"  Previous capture rate: {backtest_results.get('capture_rate', 0):.1f}%")
        print(f"  Target: {self.target}%")
        print(f"  Previous designs in history: {len(self.design_history)}")

        # Store the previous design in history before getting a new one
        self.design_history.append({
            'iteration': iteration,
            'design': {
                'strategy_description': previous_design.strategy_description,
                'priority_features': previous_design.priority_features,
                'entry_rules': previous_design.entry_rules,
                'exit_rules': previous_design.exit_rules,
                'entry_score_threshold': previous_design.entry_score_threshold,
                'exit_score_threshold': previous_design.exit_score_threshold,
                'forward_returns_days_options': previous_design.forward_returns_days_options,
                'profit_threshold_options': previous_design.profit_threshold_options,
                'recommended_stop_loss': previous_design.recommended_stop_loss,
                'recommended_trailing_stop': previous_design.recommended_trailing_stop,
                'recommended_max_positions': previous_design.recommended_max_positions,
            },
            'results': {
                'capture_rate': backtest_results.get('capture_rate', 0),
                'total_return': backtest_results.get('total_return', 0),
                'win_rate': backtest_results.get('win_rate', 0),
                'max_drawdown': backtest_results.get('max_drawdown', 0),
                'total_trades': backtest_results.get('total_trades', 0),
            }
        })

        refinement_prompt = self._build_refinement_prompt(
            backtest_results, iteration, previous_design, iteration_history
        )
        messages = [{"role": "user", "content": refinement_prompt}]
        return self._run_claude_conversation(messages)

    @staticmethod
    def _serialize_content_blocks(content) -> List[Dict]:
        """Convert response content blocks to plain dicts, filtering out thinking blocks.

        The Anthropic SDK returns content as pydantic model objects. When these are
        appended back into messages for multi-turn tool-use conversations, older
        pydantic versions (< 2.6) fail to serialize thinking blocks. Converting to
        plain dicts avoids this issue.
        """
        serialized = []
        for block in content:
            if block.type == "thinking":
                continue  # Thinking blocks should not be sent back
            elif block.type == "tool_use":
                serialized.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
            elif block.type == "text":
                serialized.append({
                    "type": "text",
                    "text": block.text,
                })
            else:
                # Unknown block type — skip to be safe
                continue
        return serialized

    def _run_claude_conversation(self, messages: List[Dict]) -> LLMStrategyDesign:
        """Run a Claude conversation with tool use loop."""
        tools = AlphaVantageTools.get_tool_definitions()
        tool_call_count = 0

        while True:
            print(f"  Calling Claude API...", flush=True)

            try:
                response = self.client.messages.create(
                    model=self.MODEL,
                    max_tokens=self.MAX_TOKENS,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": self.THINKING_BUDGET
                    },
                    system=self._build_system_prompt(),
                    tools=tools,
                    messages=messages,
                )
            except anthropic.APIError as e:
                print(f"  Claude API error: {e}")
                print(f"  Returning default design.")
                return LLMStrategyDesign(
                    strategy_description=f"Default design (API error: {e})",
                    raw_response=str(e)
                )

            # Serialize content blocks to plain dicts (avoids pydantic serialization issues)
            assistant_content = self._serialize_content_blocks(response.content)

            # Check stop reason
            if response.stop_reason == "tool_use":
                # Find tool_use blocks
                tool_uses = [b for b in response.content if b.type == "tool_use"]

                if tool_call_count >= self.MAX_TOOL_CALLS:
                    print(f"  Max tool calls ({self.MAX_TOOL_CALLS}) reached, requesting final answer...")
                    messages.append({"role": "assistant", "content": assistant_content})

                    # Must provide tool_result for every tool_use block, otherwise API errors
                    cancelled_results = []
                    for tool_use in tool_uses:
                        cancelled_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": "Tool call limit reached. Please provide your final strategy design JSON now based on the data you've already gathered.",
                            "is_error": True,
                        })
                    messages.append({"role": "user", "content": cancelled_results})
                    continue

                # Add assistant response to messages (as plain dicts, no thinking blocks)
                messages.append({"role": "assistant", "content": assistant_content})

                # Execute each tool and add results
                tool_results = []
                for tool_use in tool_uses:
                    tool_call_count += 1
                    if tool_call_count <= self.MAX_TOOL_CALLS:
                        print(f"  Tool call [{tool_call_count}]: {tool_use.name}({json.dumps(tool_use.input)})")
                        result = AlphaVantageTools.execute_tool(tool_use.name, tool_use.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": result
                        })
                    else:
                        # Over limit — return error result instead of executing
                        print(f"  Tool call [{tool_call_count}]: {tool_use.name} (SKIPPED - over limit)")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": "Tool call limit reached. Please provide your final strategy design now.",
                            "is_error": True,
                        })

                messages.append({"role": "user", "content": tool_results})

            elif response.stop_reason == "end_turn":
                # Extract text response
                text_blocks = [b for b in response.content if hasattr(b, 'text')]
                final_text = "\n".join(b.text for b in text_blocks)

                print(f"  Claude responded ({len(final_text)} chars)")
                print(f"  Tool calls used: {tool_call_count}")

                design = self._parse_claude_response(final_text)
                return design

            else:
                print(f"  Unexpected stop reason: {response.stop_reason}")
                # Try to extract whatever we got
                text_blocks = [b for b in response.content if hasattr(b, 'text')]
                if text_blocks:
                    final_text = "\n".join(b.text for b in text_blocks)
                    return self._parse_claude_response(final_text)

                return LLMStrategyDesign(
                    strategy_description=f"Design failed (stop_reason: {response.stop_reason})",
                    raw_response=str(response.content)
                )

    def _parse_claude_response(self, response_text: str) -> LLMStrategyDesign:
        """Parse Claude's structured JSON response into LLMStrategyDesign."""
        design = LLMStrategyDesign(raw_response=response_text)

        # Extract JSON block from response
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if not json_match:
            # Try without code fence
            json_match = re.search(r'\{[^{}]*"strategy_description".*\}', response_text, re.DOTALL)

        if not json_match:
            print("  Warning: Could not find JSON block in Claude's response.")
            print("  Using default design parameters.")
            design.strategy_description = "Failed to parse Claude response"
            return design

        json_str = json_match.group(1) if json_match.lastindex else json_match.group(0)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"  Warning: JSON parse error: {e}")
            print(f"  Attempting to fix common JSON issues...")
            # Try to fix trailing commas
            fixed = re.sub(r',\s*}', '}', json_str)
            fixed = re.sub(r',\s*]', ']', fixed)
            try:
                data = json.loads(fixed)
            except json.JSONDecodeError:
                print(f"  Could not parse JSON. Using default design.")
                design.strategy_description = "JSON parse failed"
                return design

        # Map parsed data to LLMStrategyDesign fields
        design.strategy_description = data.get("strategy_description", "")
        design.strategy_rationale = data.get("strategy_rationale", "")

        # ML config
        if "forward_returns_days_options" in data:
            design.forward_returns_days_options = data["forward_returns_days_options"]
        if "profit_threshold_options" in data:
            design.profit_threshold_options = data["profit_threshold_options"]
        if "recommended_n_iter_search" in data:
            design.recommended_n_iter_search = data["recommended_n_iter_search"]

        # Features - validate against available features
        raw_features = data.get("priority_features", [])
        design.priority_features = [f for f in raw_features if f in AVAILABLE_FEATURES]
        skipped = [f for f in raw_features if f not in AVAILABLE_FEATURES]
        if skipped:
            print(f"  Warning: Skipped unknown features from LLM: {skipped}")
        design.feature_rationale = data.get("feature_rationale", {})

        # Entry rules - validate feature names
        for rule in data.get("entry_rules", []):
            if rule.get("feature") in AVAILABLE_FEATURES:
                design.entry_rules.append(rule)
            else:
                print(f"  Warning: Skipped entry rule with unknown feature: {rule.get('feature')}")
        design.entry_score_threshold = data.get("entry_score_threshold", 3)

        # Exit rules - validate feature names
        for rule in data.get("exit_rules", []):
            if rule.get("feature") in AVAILABLE_FEATURES:
                design.exit_rules.append(rule)
            else:
                print(f"  Warning: Skipped exit rule with unknown feature: {rule.get('feature')}")
        design.exit_score_threshold = data.get("exit_score_threshold", 3)

        # Risk parameters — normalize percentages to decimals if Claude returned them as %
        # e.g., Claude might return 5.0 (meaning 5%) instead of 0.05
        raw_sl = data.get("recommended_stop_loss", 0.05)
        raw_ts = data.get("recommended_trailing_stop", 0.03)
        raw_ps = data.get("recommended_position_size", 0.10)

        # If value > 1, it's a percentage — convert to decimal
        design.recommended_stop_loss = raw_sl / 100.0 if raw_sl > 1 else raw_sl
        design.recommended_trailing_stop = raw_ts / 100.0 if raw_ts > 1 else raw_ts
        design.recommended_position_size = raw_ps / 100.0 if raw_ps > 1 else raw_ps
        design.recommended_max_positions = data.get("recommended_max_positions", 3)

        # Clamp to realistic ranges
        design.recommended_stop_loss = max(0.02, min(0.15, design.recommended_stop_loss))
        design.recommended_trailing_stop = max(0.01, min(0.10, design.recommended_trailing_stop))
        design.recommended_position_size = max(0.05, min(0.50, design.recommended_position_size))
        design.recommended_max_positions = max(1, min(5, design.recommended_max_positions))

        # Lookback window
        design.lookback_window = max(50, min(500, int(data.get("lookback_window", 200))))

        if raw_sl > 1 or raw_ts > 1 or raw_ps > 1:
            print(f"  Note: Normalized risk params from percentages to decimals "
                  f"(SL: {raw_sl}→{design.recommended_stop_loss}, "
                  f"TS: {raw_ts}→{design.recommended_trailing_stop}, "
                  f"PS: {raw_ps}→{design.recommended_position_size})")

        # ML training configuration
        ml_config = data.get("ml_training_config", {})
        if ml_config:
            valid_search_models = {"LightGBM", "RandomForest", "GradientBoosting", "XGBoost", "MLP", "LSTM"}
            csm = ml_config.get("config_search_model", "LightGBM")
            design.config_search_model = csm if csm in valid_search_models else "LightGBM"
            design.train_neural_networks = ml_config.get("train_neural_networks", True)
            design.train_lstm = ml_config.get("train_lstm", True)
            design.cv_folds = ml_config.get("cv_folds", 3)
            design.nn_epochs = ml_config.get("nn_epochs", 100)
            design.nn_batch_size = ml_config.get("nn_batch_size", 64)
            design.nn_learning_rate = ml_config.get("nn_learning_rate", 0.001)
            if "nn_hidden_layers" in ml_config:
                design.nn_hidden_layers = ml_config["nn_hidden_layers"]
            design.lstm_hidden_size = ml_config.get("lstm_hidden_size", 128)
            design.lstm_num_layers = ml_config.get("lstm_num_layers", 2)
            design.lstm_dropout = ml_config.get("lstm_dropout", 0.2)
            design.lstm_sequence_length = ml_config.get("lstm_sequence_length", 100)
            design.lstm_epochs = ml_config.get("lstm_epochs", 250)
            design.lstm_batch_size = ml_config.get("lstm_batch_size", 32)

        # Summary
        print(f"\n  LLM Strategy Design Summary:")
        print(f"    Description: {design.strategy_description[:100]}...")
        print(f"    Priority Features: {len(design.priority_features)} selected")
        print(f"    Entry Rules: {len(design.entry_rules)} (threshold: {design.entry_score_threshold})")
        print(f"    Exit Rules: {len(design.exit_rules)} (threshold: {design.exit_score_threshold})")
        print(f"    Forward Days: {design.forward_returns_days_options}")
        print(f"    Profit Thresholds: {design.profit_threshold_options}")
        print(f"    Stop Loss: {design.recommended_stop_loss*100:.1f}%")
        print(f"    Trailing Stop: {design.recommended_trailing_stop*100:.1f}%")
        print(f"    Max Positions: {design.recommended_max_positions}")
        print(f"    Lookback Window: {design.lookback_window} bars")
        print(f"    ML Config: MLP={'ON' if design.train_neural_networks else 'OFF'}, "
              f"LSTM={'ON' if design.train_lstm else 'OFF'}, "
              f"CV folds={design.cv_folds}")
        if design.train_neural_networks:
            print(f"    MLP: layers={design.nn_hidden_layers}, epochs={design.nn_epochs}, "
                  f"lr={design.nn_learning_rate}")
        if design.train_lstm:
            print(f"    LSTM: hidden={design.lstm_hidden_size}, layers={design.lstm_num_layers}, "
                  f"epochs={design.lstm_epochs}, seq_len={design.lstm_sequence_length}")

        return design

    def generate_strategy_readme(self, design: LLMStrategyDesign,
                                  backtest_results: Dict,
                                  model_metrics: Dict,
                                  config_path: Path,
                                  iteration_results: List[Dict]) -> str:
        """Generate a README.md for the completed strategy."""
        print(f"\n  Generating strategy README via Claude...")

        # Load config.json content
        config_content = ""
        try:
            if config_path.exists():
                config_content = config_path.read_text()
        except Exception:
            config_content = "{}"

        entry_rules_str = json.dumps(design.entry_rules, indent=2)
        exit_rules_str = json.dumps(design.exit_rules, indent=2)

        iter_summary = "\n".join(
            f"- Iteration {r['iteration']}: Capture={r['capture_rate']:.1f}%, "
            f"Return={r['total_return']:.2f}%, F1={r.get('model_f1', 0):.4f}"
            for r in iteration_results
        )

        prompt = f"""Write a comprehensive README.md for a trading strategy. Include these sections:
1. Strategy name, description, and rationale
2. How the strategy works (entry/exit logic explained in plain English)
3. Technical indicators used and why
4. ML models used to discover optimal settings
5. Key parameters and their values
6. Backtest performance results
7. Risk management settings
8. How to run the strategy

Strategy: {self.strategy_name}
Symbols: {', '.join(self.symbols)}
Description: {design.strategy_description}
Rationale: {design.strategy_rationale}
Timeframe: {self.timeframe}

Entry Rules:
{entry_rules_str}
Entry Score Threshold: {design.entry_score_threshold}

Exit Rules:
{exit_rules_str}
Exit Score Threshold: {design.exit_score_threshold}

Priority Features: {design.priority_features}

ML Training Config:
- Neural Networks (MLP): {'Enabled' if design.train_neural_networks else 'Disabled'}
  - Architectures: {design.nn_hidden_layers}
  - Epochs: {design.nn_epochs}, Batch Size: {design.nn_batch_size}, LR: {design.nn_learning_rate}
- LSTM: {'Enabled' if design.train_lstm else 'Disabled'}
  - Hidden: {design.lstm_hidden_size}, Layers: {design.lstm_num_layers}
  - Epochs: {design.lstm_epochs}, Seq Length: {design.lstm_sequence_length}
- Cross-Validation Folds: {design.cv_folds}
- Forward Return Days: {design.forward_returns_days_options}
- Profit Thresholds: {design.profit_threshold_options}

Best Model: {model_metrics.get('model_name', 'Unknown')}
Model F1: {model_metrics.get('f1', 0):.4f}
Precision: {model_metrics.get('precision', 0):.4f}
Recall: {model_metrics.get('recall', 0):.4f}

Backtest Results:
- Total Return: {backtest_results.get('total_return', 0):.2f}%
- Capture Rate: {backtest_results.get('capture_rate', 0):.1f}%
- Win Rate: {backtest_results.get('win_rate', 0):.1f}%
- Max Drawdown: {backtest_results.get('max_drawdown', 0):.2f}%

Risk Management:
- Stop Loss: {design.recommended_stop_loss*100:.1f}%
- Trailing Stop: {design.recommended_trailing_stop*100:.1f}%
- Max Positions: {design.recommended_max_positions}
- Position Size: {design.recommended_position_size*100:.1f}%

Iteration History:
{iter_summary}

config.json:
{config_content[:2000]}

Return ONLY the markdown content, no code fences wrapping it."""

        try:
            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
            )
            text_blocks = [b for b in response.content if hasattr(b, 'text')]
            readme = "\n".join(b.text for b in text_blocks)
            print(f"  README generated ({len(readme)} chars)")
            return readme
        except Exception as e:
            print(f"  Warning: README generation failed: {e}")
            return f"# {self.strategy_name}\n\nAuto-generated strategy for {', '.join(self.symbols)}.\n"

    # Marker used to identify the in-progress section in the guide file
    _PROGRESS_MARKER = "<!-- IN_PROGRESS_BUILD -->"

    def save_iteration_progress(self, design: LLMStrategyDesign,
                                model_metrics: Dict,
                                iteration: int,
                                iteration_results: List[Dict]) -> None:
        """Save iteration progress to STRATEGY_CREATION_GUIDE.md without an LLM call.

        Appends/replaces an 'In-Progress Build Data' section at the end of the
        guide file so that if the process crashes mid-build, lessons from
        completed iterations are preserved and available for the next run.
        """
        guide_path = Path(__file__).parent / "STRATEGY_CREATION_GUIDE.md"

        # Read existing guide content (without any previous in-progress section)
        existing_guide = ""
        try:
            if guide_path.exists():
                existing_guide = guide_path.read_text()
        except Exception:
            pass

        # Strip any previous in-progress section
        marker = self._PROGRESS_MARKER
        if marker in existing_guide:
            existing_guide = existing_guide[:existing_guide.index(marker)].rstrip()

        # Build structured iteration progress
        iter_lines = []
        for r in iteration_results:
            status = "TARGET MET" if r['total_return'] >= self.target else "below target"
            iter_lines.append(
                f"| {r['iteration']} | {r['training_years']}yrs | "
                f"{r['hp_iterations']} | {r.get('model_f1', 0):.4f} | "
                f"{r['total_return']:.2f}% | {r['capture_rate']:.1f}% | "
                f"{r.get('max_drawdown', 0):.2f}% | {r.get('win_rate', 0):.1f}% | {status} |"
            )
        iter_table = "\n".join(iter_lines)

        entry_rules_summary = "; ".join(
            f"{rule.get('feature', 'unknown')} {rule.get('operator', '?')} {rule.get('threshold', '?')}"
            for rule in (design.entry_rules or [])[:5]
        )
        exit_rules_summary = "; ".join(
            f"{rule.get('feature', 'unknown')} {rule.get('operator', '?')} {rule.get('threshold', '?')}"
            for rule in (design.exit_rules or [])[:5]
        )

        progress_section = f"""

{marker}
## In-Progress Build Data (auto-saved after iteration {iteration})

> This section is auto-generated during strategy building. It will be replaced
> by synthesized lessons when the build completes successfully.

**Strategy:** {self.strategy_name}
**Symbols:** {', '.join(self.symbols)}
**Description:** {design.strategy_description}
**Target:** {self.target}% return | **Timeframe:** {self.timeframe}

### Iteration Results

| Iter | Data | HP Iters | F1 | Return | Capture | MaxDD | WinRate | Status |
|------|------|----------|-----|--------|---------|-------|---------|--------|
{iter_table}

### Current Design Configuration

- **Best Model:** {model_metrics.get('model_name', 'Unknown')} (F1: {model_metrics.get('f1', 0):.4f})
- **MLP:** {'ON' if design.train_neural_networks else 'OFF'} | **LSTM:** {'ON' if design.train_lstm else 'OFF'}
- **NN Epochs:** {design.nn_epochs} | **LSTM Epochs:** {design.lstm_epochs}
- **Forward Days:** {design.forward_returns_days_options}
- **Profit Thresholds:** {design.profit_threshold_options}
- **Stop Loss:** {design.recommended_stop_loss*100:.1f}% | **Trailing Stop:** {design.recommended_trailing_stop*100:.1f}%
- **Max Positions:** {design.recommended_max_positions} | **Entry Threshold:** {design.entry_score_threshold}
- **Priority Features:** {', '.join(design.priority_features[:10])}
- **Entry Rules (top 5):** {entry_rules_summary}
- **Exit Rules (top 5):** {exit_rules_summary}

### Design Rationale
{design.strategy_rationale}
"""

        # Write the guide with progress section appended
        try:
            full_content = existing_guide + progress_section
            guide_path.write_text(full_content)
            print(f"  📝 Creation guide updated with iteration {iteration} progress")
        except Exception as e:
            print(f"  Warning: Could not save iteration progress: {e}")

    def update_creation_guide(self, design: LLMStrategyDesign,
                               backtest_results: Dict,
                               model_metrics: Dict,
                               iteration_results: List[Dict]) -> None:
        """Update STRATEGY_CREATION_GUIDE.md with lessons from this build."""
        guide_path = Path(__file__).parent / "STRATEGY_CREATION_GUIDE.md"
        print(f"\n  Updating strategy creation guide...")

        existing_guide = ""
        try:
            if guide_path.exists():
                existing_guide = guide_path.read_text()
        except Exception:
            pass

        # Strip any in-progress build data section before passing to LLM
        # (that data is already provided separately as structured parameters)
        marker = self._PROGRESS_MARKER
        if marker in existing_guide:
            existing_guide = existing_guide[:existing_guide.index(marker)].rstrip()

        entry_rules_str = json.dumps(design.entry_rules, indent=2)
        exit_rules_str = json.dumps(design.exit_rules, indent=2)

        iter_summary = "\n".join(
            f"- Iteration {r['iteration']}: Capture={r['capture_rate']:.1f}%, "
            f"Return={r['total_return']:.2f}%, F1={r.get('model_f1', 0):.4f}"
            for r in iteration_results
        )

        prompt = f"""You are maintaining a Strategy Creation Guide that helps an AI design winning trading strategies.
This guide is referenced BEFORE starting any new strategy build, so it must contain actionable lessons.

{"## Existing Guide Content:" + chr(10) + existing_guide if existing_guide else "The guide is empty - this is the first entry."}

## New Strategy Results to Incorporate

Strategy: {self.strategy_name}
Symbols: {', '.join(self.symbols)}
Description: {design.strategy_description}
Target: {self.target}% annually
Timeframe: {self.timeframe}
Final Capture Rate: {backtest_results.get('capture_rate', 0):.1f}%
Total Return: {backtest_results.get('total_return', 0):.2f}%
Win Rate: {backtest_results.get('win_rate', 0):.1f}%
Max Drawdown: {backtest_results.get('max_drawdown', 0):.2f}%
Best Model: {model_metrics.get('model_name', 'Unknown')} (F1: {model_metrics.get('f1', 0):.4f})

Priority Features: {design.priority_features}
Entry Rules: {entry_rules_str}
Exit Rules: {exit_rules_str}

ML Config:
- MLP: {'ON' if design.train_neural_networks else 'OFF'}, LSTM: {'ON' if design.train_lstm else 'OFF'}
- NN Layers: {design.nn_hidden_layers}, Epochs: {design.nn_epochs}
- LSTM Hidden: {design.lstm_hidden_size}, Layers: {design.lstm_num_layers}, Epochs: {design.lstm_epochs}
- Forward Days: {design.forward_returns_days_options}
- Profit Thresholds: {design.profit_threshold_options}

Risk: Stop={design.recommended_stop_loss*100:.1f}%, Trail={design.recommended_trailing_stop*100:.1f}%, MaxPos={design.recommended_max_positions}

Iteration History:
{iter_summary}

## Instructions
Write the COMPLETE updated guide (replacing the existing one). Structure it as:

1. **General Principles** - What makes strategies work across different asset classes
2. **Optimal ML Settings** - Which models, epochs, architectures tend to work best
3. **Feature Selection Patterns** - Which indicators work well for which types of strategies
4. **Risk Management Lessons** - What stop loss / trailing stop ranges work
5. **Common Pitfalls** - What to avoid
6. **Strategy-Specific Notes** - Indexed by asset type (gold, semiconductors, etc.)

Keep it concise and actionable. Each lesson should be something the AI can directly apply.
Return ONLY the markdown content."""

        try:
            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
            )
            text_blocks = [b for b in response.content if hasattr(b, 'text')]
            guide_content = "\n".join(b.text for b in text_blocks)

            guide_path.write_text(guide_content)
            print(f"  Creation guide updated ({len(guide_content)} chars)")
        except Exception as e:
            print(f"  Warning: Guide update failed: {e}")
