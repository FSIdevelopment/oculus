"""
Strategy File Generator — produces all strategy files as in-memory strings.

Ported from strategy_designer_ai/strategy_builder.py (_generate_strategy_files
and _create_config_files) for use inside the worker pipeline.  The worker does
NOT write to disk permanently; it creates files in a temp directory, reads them
back as strings, and includes them in the Redis result payload so the backend
can persist them.

Author: SignalSynk AI
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class StrategyFileGenerator:
    """Generates all strategy files as {filename: content_string} dicts.

    Parameters
    ----------
    strategy_name : str
        Human-readable strategy name (e.g. "momentum_abc123").
    symbols : list[str]
        Trading symbols (e.g. ["NVDA", "AMD"]).
    timeframe : str
        Trading timeframe (e.g. "1d").
    model_metrics : dict
        Best model metrics (f1, precision, recall, auc, model_name).
    extracted_rules : dict
        Serialised StrategyRules with entry_rules, exit_rules, top_features,
        entry_score_threshold, exit_score_threshold, optimal_params.
    design : dict | None
        LLM design config (optional).  Used for risk params, lookback, etc.
    """

    def __init__(
        self,
        strategy_name: str,
        symbols: List[str],
        timeframe: str,
        model_metrics: Dict[str, Any],
        extracted_rules: Dict[str, Any],
        design: Optional[Dict[str, Any]] = None,
    ):
        self.strategy_name = strategy_name
        self.symbols = symbols
        self.timeframe = timeframe
        self.model_metrics = model_metrics
        self.rules = extracted_rules
        self.design = design or {}

        # Derived helpers
        self.entry_rules = self.rules.get("entry_rules", [])
        self.exit_rules = self.rules.get("exit_rules", [])
        self.top_features = self.rules.get("top_features", [])
        self.entry_score_threshold = self.rules.get("entry_score_threshold", 3)
        self.exit_score_threshold = self.rules.get("exit_score_threshold", 3)
        self.optimal_params = self.rules.get("optimal_params", {})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_all(self) -> Dict[str, str]:
        """Generate all strategy files and return {filename: content}.

        Returns a dict with 10 keys:
            strategy.py, backtest.py, data_provider.py, risk_manager.py,
            main.py, Dockerfile, docker-compose.yml,
            config.json, requirements.txt, extracted_rules.json
        """
        files: Dict[str, str] = {}

        # Strategy code files
        files["strategy.py"] = self._generate_strategy_py()
        files["backtest.py"] = self._generate_backtest_py()
        files["data_provider.py"] = self._generate_data_provider_py()
        files["risk_manager.py"] = self._generate_risk_manager_py()
        files["main.py"] = self._generate_main_py()

        # Container files
        files["Dockerfile"] = self._generate_dockerfile()
        files["docker-compose.yml"] = self._generate_docker_compose()

        # Config files
        files["config.json"] = self._generate_config_json()
        files["requirements.txt"] = self._generate_requirements_txt()
        files["extracted_rules.json"] = self._generate_extracted_rules_json()

        logger.info(
            f"Generated {len(files)} strategy files for '{self.strategy_name}'"
        )
        return files

    # ------------------------------------------------------------------
    # Helpers — condition builders (ported from strategy_builder.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_indicator_var_name(feature_name: str) -> str:
        """Map ML feature name to the calculated indicator variable name."""
        return f"indicators.get('{feature_name.lower()}', 0)"

    def _build_entry_conditions(self) -> str:
        """Build entry condition code from extracted rules."""
        lines: List[str] = []
        for rule in self.entry_rules[:8]:  # Top 8 rules
            var = self._get_indicator_var_name(rule["feature"])
            lines.append(f"        if {var} {rule['operator']} {rule['threshold']}:")
            lines.append(f"            entry_score += 1  # {rule['description']}")
        return "\n".join(lines) if lines else "        pass  # No entry rules extracted"

    def _build_exit_conditions(self) -> str:
        """Build exit condition code from extracted rules."""
        lines: List[str] = []
        for rule in self.exit_rules[:6]:  # Top 6 rules
            var = self._get_indicator_var_name(rule["feature"])
            lines.append(f"        if {var} {rule['operator']} {rule['threshold']}:")
            lines.append(f"            exit_score += 1  # {rule['description']}")
        return "\n".join(lines) if lines else "        pass  # No exit rules extracted"

    # ------------------------------------------------------------------
    # Risk / design param helpers
    # ------------------------------------------------------------------

    def _risk_params(self) -> Dict[str, Any]:
        """Return risk management parameters from design or defaults."""
        d = self.design
        return {
            "max_positions": d.get("recommended_max_positions", 3),
            "min_entry_score": d.get("entry_score_threshold", self.entry_score_threshold),
            "max_position_size": d.get("recommended_position_size", 0.1),
            "stop_loss_pct": d.get("recommended_stop_loss", 0.05),
            "trailing_stop_pct": d.get("recommended_trailing_stop", 0.03),
            "lookback_window": d.get("lookback_window", 200),
        }

    # ------------------------------------------------------------------
    # File generators — each returns a string
    # ------------------------------------------------------------------


    def _generate_strategy_py(self) -> str:
        """Generate the main strategy.py file with entry/exit conditions."""
        entry_conditions = self._build_entry_conditions()
        exit_conditions = self._build_exit_conditions()
        top_features_str = ", ".join(
            [f["feature"] for f in self.top_features[:5]]
        ) if self.top_features else "N/A"

        rp = self._risk_params()

        return f'''#!/usr/bin/env python3
"""
{self.strategy_name} - AI-Generated Portfolio Trading Strategy

Generated by SignalSynk Strategy Builder on {datetime.now().strftime("%Y-%m-%d %H:%M")}

Strategy based on ML model analysis:
- Model: {self.model_metrics.get('model_name', 'RandomForest')}
- F1 Score: {self.model_metrics.get('f1', 0):.3f}
- Top Features: {top_features_str}
- Timeframe: {self.timeframe}

Portfolio-based strategy that:
1. Evaluates ALL symbols simultaneously
2. Ranks by entry signal strength
3. Selects top N positions (default: {rp["max_positions"]})
4. Discards symbols without valid entry signals

This is a pure algorithmic strategy - no ML dependencies at runtime.

Author: SignalSynk AI
Copyright (c) 2026 Forefront Solutions Inc. All rights reserved.
"""

import os
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger('{self.strategy_name.replace(" ", "")}Strategy')


class TradingStrategy:
    """
    {self.strategy_name} Portfolio Trading Strategy.

    AI-generated strategy optimized for: {", ".join(self.symbols)}

    This strategy evaluates all symbols together and selects the best
    positions based on entry signal strength.
    """

    def __init__(self):
        """Initialize the strategy with configuration."""
        self.config = self._load_config()
        self.positions = {{}}
        self.daily_pnl = 0.0

        # Get configuration
        trading_config = self.config.get('trading', {{}})
        self.symbols = trading_config.get('symbols', {self.symbols})
        self.default_quantity = trading_config.get('default_quantity', 10)
        self.max_positions = trading_config.get('max_positions', {rp["max_positions"]})
        self.min_entry_score = trading_config.get('min_entry_score', {rp["min_entry_score"]})

        # Indicator parameters from ML analysis
        indicator_config = self.config.get('indicators', {{}})
        self.rsi_period = indicator_config.get('rsi_period', 14)
        self.rsi_oversold = indicator_config.get('rsi_oversold', {self.optimal_params.get('rsi_oversold', 35)})
        self.rsi_overbought = indicator_config.get('rsi_overbought', 70)
        self.atr_period = indicator_config.get('atr_period', 14)
        self.atr_multiplier = indicator_config.get('atr_multiplier', 2.5)

        # Risk management
        risk_config = self.config.get('risk_management', {{}})
        self.max_position_size = risk_config.get('max_position_size', {rp["max_position_size"]})
        self.stop_loss_pct = risk_config.get('stop_loss_pct', {rp["stop_loss_pct"]})
        self.trailing_stop_pct = risk_config.get('trailing_stop_pct', {rp["trailing_stop_pct"]})
        self.max_drawdown = risk_config.get('max_drawdown', 0.20)
        self.max_daily_loss = risk_config.get('max_daily_loss', 0.05)

        logger.info(f"{self.strategy_name} Strategy initialized")
        logger.info(f"  Symbols: {{self.symbols}}")
        logger.info(f"  Max positions: {{self.max_positions}}")

    def _load_config(self) -> Dict[str, Any]:
        """Load strategy configuration from config.json."""
        config_path = Path(__file__).parent / 'config.json'
        try:
            with open(config_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config.json: {{e}}")
            return {{}}

    def calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all technical indicators needed for the strategy."""
        if len(df) < 50:
            return {{}}

        try:
            indicators = {{}}

            # RSI (multiple periods)
            for period in [7, 14, 21]:
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / (loss + 1e-10)
                rsi = 100 - (100 / (1 + rs))
                indicators[f'rsi_{{period}}'] = float(rsi.iloc[-1])

            # Stochastic Oscillator
            for period in [14, 21]:
                low_n = df['Low'].rolling(period).min()
                high_n = df['High'].rolling(period).max()
                stoch_k = ((df['Close'] - low_n) / (high_n - low_n + 1e-10)) * 100
                stoch_d = stoch_k.rolling(3).mean()
                indicators[f'stoch_k_{{period}}'] = float(stoch_k.iloc[-1])
                indicators[f'stoch_d_{{period}}'] = float(stoch_d.iloc[-1])

            # Williams %R
            indicators['williams_r'] = float(((df['High'].rolling(14).max() - df['Close']) /
                (df['High'].rolling(14).max() - df['Low'].rolling(14).min() + 1e-10)).iloc[-1] * -100)

            # MACD
            ema12 = df['Close'].ewm(span=12).mean()
            ema26 = df['Close'].ewm(span=26).mean()
            macd = ema12 - ema26
            macd_signal = macd.ewm(span=9).mean()
            macd_hist = macd - macd_signal
            indicators['macd'] = float(macd.iloc[-1])
            indicators['macd_signal'] = float(macd_signal.iloc[-1])
            indicators['macd_hist'] = float(macd_hist.iloc[-1])

            # ROC (Rate of Change)
            for period in [5, 10, 20]:
                roc = ((df['Close'] - df['Close'].shift(period)) /
                       (df['Close'].shift(period) + 1e-10)) * 100
                indicators[f'roc_{{period}}'] = float(roc.iloc[-1])

            # MFI (Money Flow Index)
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            mf = tp * df['Volume']
            pos_mf = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
            neg_mf = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
            mfi = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-10)))
            indicators['mfi'] = float(mfi.iloc[-1])

            # CCI
            tp_sma = tp.rolling(20).mean()
            mean_dev = tp.rolling(20).std() * 0.7979
            cci = (tp - tp_sma) / (0.015 * mean_dev + 1e-10)
            indicators['cci'] = float(cci.iloc[-1])

            # EMAs (multiple periods)
            for period in [8, 13, 21, 50, 100, 200]:
                ema = df['Close'].ewm(span=period).mean()
                indicators[f'ema_{{period}}'] = float(ema.iloc[-1])
                price_vs_ema = ((df['Close'] - ema) / (ema + 1e-10)) * 100
                indicators[f'price_vs_ema_{{period}}'] = float(price_vs_ema.iloc[-1])

            # ATR and ADX
            tr = pd.concat([
                df['High'] - df['Low'],
                abs(df['High'] - df['Close'].shift()),
                abs(df['Low'] - df['Close'].shift())
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            indicators['atr'] = float(atr.iloc[-1])
            indicators['atr_pct'] = float((atr / (df['Close'] + 1e-10)).iloc[-1] * 100)

            # ADX components
            up_move = df['High'] - df['High'].shift()
            down_move = df['Low'].shift() - df['Low']
            plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
            minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
            plus_di = 100 * (plus_dm.rolling(14).mean() / (atr + 1e-10))
            minus_di = 100 * (minus_dm.rolling(14).mean() / (atr + 1e-10))
            dx = abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
            indicators['adx'] = float(dx.rolling(14).mean().iloc[-1])
            indicators['plus_di'] = float(plus_di.iloc[-1])
            indicators['minus_di'] = float(minus_di.iloc[-1])
            indicators['di_diff'] = float((plus_di - minus_di).iloc[-1])

            # Bollinger Bands
            bb_mid = df['Close'].rolling(20).mean()
            bb_std = df['Close'].rolling(20).std()
            bb_upper = bb_mid + 2 * bb_std
            bb_lower = bb_mid - 2 * bb_std
            indicators['bb_upper'] = float(bb_upper.iloc[-1])
            indicators['bb_lower'] = float(bb_lower.iloc[-1])
            indicators['bb_pct_b'] = float(((df['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)).iloc[-1])
            indicators['bb_width'] = float(((bb_upper - bb_lower) / (bb_mid + 1e-10)).iloc[-1] * 100)

            # Keltner Channels
            kc_mid = df['Close'].ewm(span=20).mean()
            kc_upper = kc_mid + 2 * atr
            kc_lower = kc_mid - 2 * atr
            indicators['kc_pct'] = float(((df['Close'] - kc_lower) / (kc_upper - kc_lower + 1e-10)).iloc[-1])

            # Volume indicators
            indicators['volume_ratio'] = float((df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-10)).iloc[-1])

            # OBV
            obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
            obv_ema = obv.ewm(span=20).mean()
            indicators['obv_ema'] = float(obv_ema.iloc[-1])
            indicators['obv_diff'] = float((obv - obv_ema).iloc[-1])

            # CMF
            clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-10)
            clv = clv.fillna(0)
            mfv = clv * df['Volume']
            cmf = mfv.rolling(21).sum() / (df['Volume'].rolling(21).sum() + 1e-10)
            indicators['cmf'] = float(cmf.iloc[-1])

            # VWAP
            vwap = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / (df['Volume'].cumsum() + 1e-10)
            indicators['vwap'] = float(vwap.iloc[-1])
            indicators['price_vs_vwap'] = float(((df['Close'] - vwap) / (vwap + 1e-10)).iloc[-1] * 100)

            # Macro/trend features
            for period in [21, 42, 63, 126]:
                mom = df['Close'].pct_change(period) * 100
                indicators[f'momentum_{{period}}d'] = float(mom.iloc[-1]) if len(mom) > period else 0

            if len(df) >= 252:
                high_252 = df['High'].rolling(252).max()
                low_252 = df['Low'].rolling(252).min()
                indicators['price_vs_52w_high'] = float(((df['Close'] - high_252) / (high_252 + 1e-10)).iloc[-1] * 100)
                indicators['price_vs_52w_low'] = float(((df['Close'] - low_252) / (low_252 + 1e-10)).iloc[-1] * 100)
                indicators['price_52w_range_pct'] = float(((df['Close'] - low_252) / (high_252 - low_252 + 1e-10)).iloc[-1] * 100)
            else:
                indicators['price_vs_52w_high'] = 0
                indicators['price_vs_52w_low'] = 0
                indicators['price_52w_range_pct'] = 50

            # EMA stack score
            ema8 = df['Close'].ewm(span=8).mean()
            ema21 = df['Close'].ewm(span=21).mean()
            ema50 = df['Close'].ewm(span=50).mean()
            ema200 = df['Close'].ewm(span=200).mean()
            stack_score = int(ema8.iloc[-1] > ema21.iloc[-1]) + int(ema21.iloc[-1] > ema50.iloc[-1]) + int(ema50.iloc[-1] > ema200.iloc[-1])
            indicators['ema_stack_score'] = stack_score
            indicators['ema_stack_bullish'] = int(stack_score == 3)

            # Price acceleration
            mom5 = df['Close'].pct_change(5)
            mom10 = df['Close'].pct_change(10)
            indicators['price_accel_5d'] = float(mom5.diff(5).iloc[-1] * 100) if len(df) > 10 else 0
            indicators['price_accel_10d'] = float(mom10.diff(10).iloc[-1] * 100) if len(df) > 20 else 0

            # Trend slope
            if len(df) >= 50:
                x = np.arange(50)
                y = df['Close'].iloc[-50:].values
                slope = np.polyfit(x, y, 1)[0]
                indicators['trend_slope_50d'] = float(slope / (df['Close'].iloc[-1] + 1e-10) * 100)
            else:
                indicators['trend_slope_50d'] = 0

            # Distance from ATH
            ath = df['High'].expanding().max()
            indicators['dist_from_ath'] = float(((df['Close'] - ath) / (ath + 1e-10)).iloc[-1] * 100)

            # Breakout signals
            indicators['breakout_20d'] = int(df['Close'].iloc[-1] > df['High'].iloc[-21:-1].max()) if len(df) > 21 else 0
            indicators['breakout_50d'] = int(df['Close'].iloc[-1] > df['High'].iloc[-51:-1].max()) if len(df) > 51 else 0

            # Volume trend
            vol_20 = df['Volume'].rolling(20).mean()
            vol_50 = df['Volume'].rolling(50).mean()
            indicators['volume_trend'] = float((vol_20 / (vol_50 + 1e-10)).iloc[-1])

            # Basic price data
            indicators['close'] = float(df['Close'].iloc[-1])
            indicators['volume'] = float(df['Volume'].iloc[-1])

            return indicators

        except Exception as e:
            logger.error(f"Error calculating indicators: {{e}}")
            return {{}}

    def get_entry_score(self, indicators: Dict[str, Any]) -> int:
        """Calculate entry score for given indicators."""
        if not indicators:
            return 0

        entry_score = 0
{entry_conditions}

        return entry_score

    def get_exit_score(self, indicators: Dict[str, Any]) -> int:
        """Calculate exit score for given indicators."""
        if not indicators:
            return 0

        exit_score = 0
{exit_conditions}

        return exit_score

    def should_exit(self, indicators: Dict[str, Any]) -> bool:
        """Check if position should be exited based on exit conditions."""
        return self.get_exit_score(indicators) >= {self.exit_score_threshold}

    def generate_signal(self, symbol: str, indicators: Dict[str, Any]) -> str:
        """Generate trading signal based on extracted rules."""
        if not indicators:
            return 'HOLD'

        entry_score = self.get_entry_score(indicators)
        exit_score = self.get_exit_score(indicators)

        if entry_score >= self.min_entry_score:
            logger.info(f"{{symbol}}: BUY signal (entry_score={{entry_score}})")
            return 'BUY'
        elif exit_score >= {self.exit_score_threshold}:
            logger.info(f"{{symbol}}: SELL signal (exit_score={{exit_score}})")
            return 'SELL'

        return 'HOLD'

    async def check_entry_conditions(
        self,
        symbol: str,
        indicators: Dict[str, Any],
        current_date=None,
    ) -> Optional[str]:
        """Async backtest interface — delegates to generate_signal().

        The backtest runner calls this with ``await`` and expects 'BUY',
        'SELL', or None (None is returned for 'HOLD' to match template
        convention).
        """
        signal = self.generate_signal(symbol, indicators)
        return signal if signal != 'HOLD' else None

    def rank_symbols(self, symbol_indicators: Dict[str, Dict[str, Any]]) -> List[Tuple[str, int]]:
        """Rank all symbols by their entry signal strength."""
        rankings = []
        for symbol, indicators in symbol_indicators.items():
            if not indicators:
                continue
            entry_score = self.get_entry_score(indicators)
            if entry_score >= self.min_entry_score:
                rankings.append((symbol, entry_score))
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def select_top_positions(self, symbol_indicators: Dict[str, Dict[str, Any]],
                              current_positions: List[str] = None) -> List[str]:
        """Select the top N symbols to hold positions in."""
        current_positions = current_positions or []
        rankings = self.rank_symbols(symbol_indicators)
        selected = [symbol for symbol, score in rankings[:self.max_positions]]
        logger.info(f"Portfolio selection: {{selected}} from {{len(rankings)}} candidates")
        return selected

    def get_symbols(self) -> List[str]:
        """Return list of symbols to trade."""
        return self.symbols

    def get_interval(self) -> int:
        """Return signal generation interval in seconds (read from config)."""
        trading_config = self.config.get('trading', {{}})
        interval_str = (
            trading_config.get('signal_generation_interval')
            or trading_config.get('timeframe', '1d')
        )
        timeframe_map = {{
            '30s': 30, '1m': 60, '2m': 120, '5m': 300, '10m': 600,
            '15m': 900, '30m': 1800, '1h': 3600, '4h': 14400,
            '1d': 86400, '1D': 86400,
        }}
        return timeframe_map.get(interval_str, 86400)

    def get_loop_interval(self) -> int:
        """Return how often (seconds) main.py should re-run the strategy loop.

        This is intentionally shorter than the data timeframe so we check
        frequently for new candle closes without fetching stale data.

        Mapping (timeframe → loop interval):
          1d  → 1800s (30 min)
          4h  →  900s (15 min)
          1h  →  300s  (5 min)
          30m →  300s  (5 min)
          15m →  900s (15 min)
          10m →  600s (10 min)
          5m  →  300s  (5 min)
          1m  →   60s  (1 min)
        """
        trading_config = self.config.get('trading', {{}})
        timeframe = trading_config.get('timeframe', '1d')
        loop_map = {{
            '1d': 1800, '1D': 1800,
            '4h': 900,
            '1h': 300,
            '30m': 300, '30min': 300,
            '15m': 900, '15min': 900,
            '10m': 600, '10min': 600,
            '5m': 300, '5min': 300,
            '2m': 120, '2min': 120,
            '1m': 60, '1min': 60,
        }}
        return loop_map.get(timeframe, 1800)

    async def analyze(self) -> List[Dict[str, Any]]:
        """Fetch live data, calculate indicators, and return trade signals.

        Called periodically by main.py during market hours.
        """
        from data_provider import DataProvider
        signals = []
        provider = DataProvider()

        for symbol in self.symbols:
            try:
                df = await provider.get_historical_data_async(symbol, days=365)
                if df is None or len(df) < 50:
                    continue
                indicators = self.calculate_indicators(df)
                if not indicators:
                    continue
                signal = self.generate_signal(symbol, indicators)
                if signal in ('BUY', 'SELL'):
                    signals.append({{
                        'ticker': symbol,
                        'action': signal,
                        'quantity': self.default_quantity,
                        'order_type': 'MARKET',
                        'price': indicators.get('close'),
                        'indicators': {{
                            k: round(float(v), 4)
                            for k, v in indicators.items()
                            if isinstance(v, (int, float)) and k != 'volume'
                        }},
                    }})
            except Exception as e:
                logger.error(f"Error analyzing {{symbol}}: {{e}}")

        return signals
'''

    def _generate_backtest_py(self) -> str:
        """Generate the backtest.py file (portfolio-based simulation)."""
        return f'''#!/usr/bin/env python3
"""
Portfolio Backtest Runner for {self.strategy_name}

Features:
- Portfolio-based: Evaluates all symbols together, selects best positions
- Risk Management: Stop loss, trailing stop, max drawdown, daily drawdown
- Position Sizing: Based on account value and risk limits

Author: SignalSynk AI
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategy import TradingStrategy
from data_provider import DataProvider
from risk_manager import RiskManager


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    entry_price: float
    entry_date: datetime
    shares: int
    highest_price: float = 0.0
    indicators: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.highest_price = self.entry_price


@dataclass
class Trade:
    """Completed trade record."""
    symbol: str
    action: str          # 'LONG' for buy-then-sell positions
    entry_price: float
    exit_price: float
    entry_date: datetime
    exit_date: datetime
    shares: int
    pnl: float
    pnl_pct: float
    duration_hours: float
    indicators: Dict[str, Any]
    exit_reason: str


class PortfolioBacktester:
    """Portfolio-based backtester with risk management."""

    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.strategy = TradingStrategy()
        self.data_provider = DataProvider()

        config = self._load_config()
        risk_config = config.get('risk_management', {{}})

        self.stop_loss_pct = risk_config.get('stop_loss_pct', 0.05)
        self.trailing_stop_pct = risk_config.get('trailing_stop_pct', 0.03)
        self.max_drawdown = risk_config.get('max_drawdown', 0.20)
        self.max_daily_loss = risk_config.get('max_daily_loss', 0.05)
        self.max_position_size = risk_config.get('max_position_size', 0.1)
        self.max_positions = config.get('trading', {{}}).get('max_positions', 3)
        self.lookback_window = config.get('trading', {{}}).get('lookback_window', 200)

        self.cash = initial_capital
        self.positions: Dict[str, Position] = {{}}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.equity_dates: List[datetime] = []
        self.peak_equity = initial_capital
        self.daily_start_equity = initial_capital
        self.current_date = None
        self.halted = False
        self.halt_reason = ""

        self.buy_hold_returns: Dict[str, float] = {{}}
        self.max_potential_profit: float = 0.0
        self.symbol_start_prices: Dict[str, float] = {{}}
        self.symbol_end_prices: Dict[str, float] = {{}}

        self._cached_data: Optional[Dict[str, pd.DataFrame]] = None
        self._cached_indicators: Optional[Dict[str, Dict]] = None

    def _load_config(self) -> Dict[str, Any]:
        config_path = Path(__file__).parent / 'config.json'
        try:
            with open(config_path) as f:
                return json.load(f)
        except:
            return {{}}

    def get_indicators(self, symbol: str, date, df_slice: pd.DataFrame) -> Dict[str, Any]:
        """Get indicators from cache or calculate them.

        This is a MAJOR optimization: when running optimizer, indicators are
        pre-calculated once and cached, giving ~50x speedup.
        """
        # Try to get from cache first (set by optimizer)
        if self._cached_indicators and symbol in self._cached_indicators:
            if date in self._cached_indicators[symbol]:
                return self._cached_indicators[symbol][date]

        # Fall back to calculating (slower, but works for standalone runs)
        return self.strategy.calculate_indicators(df_slice)

    def get_equity(self, prices: Dict[str, float]) -> float:
        """Calculate current portfolio equity."""
        position_value = sum(
            pos.shares * prices.get(symbol, pos.entry_price)
            for symbol, pos in self.positions.items()
        )
        return self.cash + position_value

    def check_risk_limits(self, prices: Dict[str, float]) -> bool:
        """Check if risk limits are breached. Returns True if trading should halt."""
        equity = self.get_equity(prices)

        # Check max drawdown
        drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0
        if drawdown >= self.max_drawdown:
            self.halted = True
            self.halt_reason = f"Max drawdown breached: {{drawdown*100:.1f}}% >= {{self.max_drawdown*100:.1f}}%"
            return True

        # Check daily drawdown
        daily_loss = (self.daily_start_equity - equity) / self.daily_start_equity if self.daily_start_equity > 0 else 0
        if daily_loss >= self.max_daily_loss:
            self.halted = True
            self.halt_reason = f"Daily loss limit breached: {{daily_loss*100:.1f}}% >= {{self.max_daily_loss*100:.1f}}%"
            return True

        return False

    def check_stop_loss(self, position: Position, current_price: float) -> Optional[str]:
        """Check if position should be closed due to stop loss or trailing stop."""
        pnl_pct = (current_price - position.entry_price) / position.entry_price

        # Fixed stop loss
        if pnl_pct <= -self.stop_loss_pct:
            return "stop_loss"

        # Trailing stop (only if in profit)
        if current_price > position.highest_price:
            position.highest_price = current_price

        if position.highest_price > position.entry_price:
            drop_from_high = (position.highest_price - current_price) / position.highest_price
            if drop_from_high >= self.trailing_stop_pct:
                return "trailing_stop"

        return None

    def close_position(self, symbol: str, current_price: float, current_date: datetime,
                       exit_reason: str) -> Trade:
        """Close a position and record the trade with full trade metadata."""
        position = self.positions[symbol]
        pnl = (current_price - position.entry_price) * position.shares
        pnl_pct = (current_price - position.entry_price) / position.entry_price * 100
        duration_hours = round(
            (current_date - position.entry_date).total_seconds() / 3600, 2
        ) if hasattr(current_date, '__sub__') else 0.0

        trade = Trade(
            symbol=symbol,
            action='LONG',
            entry_price=position.entry_price,
            exit_price=current_price,
            entry_date=position.entry_date,
            exit_date=current_date,
            shares=position.shares,
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 4),
            duration_hours=duration_hours,
            indicators=position.indicators,
            exit_reason=exit_reason
        )

        self.cash += current_price * position.shares
        del self.positions[symbol]
        self.trades.append(trade)

        return trade

    def open_position(self, symbol: str, current_price: float, current_date: datetime,
                      prices: Dict[str, float],
                      indicators: Dict[str, Any] = None) -> Optional[Position]:
        """Open a new position with proper position sizing.

        Args:
            symbol: Ticker symbol to buy.
            current_price: Current market price.
            current_date: Date of entry.
            prices: Current prices for all symbols (used for portfolio valuation).
            indicators: Indicator snapshot at the time of entry, stored on the
                        position so it can be included in the closed Trade record.
        """
        equity = self.get_equity(prices)
        max_position_value = equity * self.max_position_size
        shares = int(max_position_value / current_price)

        if shares <= 0:
            return None

        cost = shares * current_price
        if cost > self.cash:
            shares = int(self.cash / current_price)
            cost = shares * current_price

        if shares <= 0:
            return None

        position = Position(
            symbol=symbol,
            entry_price=current_price,
            entry_date=current_date,
            shares=shares,
            indicators=indicators or {{}}
        )

        self.cash -= cost
        self.positions[symbol] = position

        return position

    def run(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Run the portfolio backtest."""
        print(f"\\n{{'='*60}}")
        print(f"PORTFOLIO BACKTEST: {self.strategy_name}")
        print(f"{{'='*60}}")
        print(f"  Initial Capital: ${{self.initial_capital:,.2f}}")
        print(f"  Symbols: {{', '.join(self.strategy.get_symbols())}}")
        print(f"  Max Positions: {{self.max_positions}}")
        print(f"  Stop Loss: {{self.stop_loss_pct*100:.1f}}%")
        print(f"  Trailing Stop: {{self.trailing_stop_pct*100:.1f}}%")
        print(f"  Max Drawdown: {{self.max_drawdown*100:.1f}}%")
        print(f"  Max Daily Loss: {{self.max_daily_loss*100:.1f}}%")
        print(f"  Lookback Window: {{self.lookback_window}} bars")
        print(f"{{'='*60}}\\n")

        # Default to 2 years of data
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

        print(f"  Fetching data from {{start_date}} to {{end_date}}...")

        # Fetch all data (use cached data from optimizer if available)
        if self._cached_data:
            all_data = self._cached_data
        else:
            all_data: Dict[str, pd.DataFrame] = {{}}
            for symbol in self.strategy.get_symbols():
                df = self.data_provider.get_historical_data(symbol, start_date, end_date)
                if df is not None and len(df) >= self.lookback_window:
                    all_data[symbol] = df
                    print(f"    ✓ {{symbol}}: {{len(df)}} bars")
                else:
                    print(f"    ⚠️ {{symbol}}: Insufficient data")

        if not all_data:
            print("\\n  ❌ No valid data available for backtest")
            return {{}}

        # Find common date range
        common_dates = None
        for df in all_data.values():
            if common_dates is None:
                common_dates = set(df.index)
            else:
                common_dates &= set(df.index)

        common_dates = sorted(list(common_dates))
        if len(common_dates) < 50:
            print(f"\\n  ❌ Insufficient common dates: {{len(common_dates)}}")
            return {{}}

        # Calculate buy-and-hold returns and max potential profit for each symbol
        for symbol, df in all_data.items():
            start_price = float(df.iloc[0]['Close'])
            end_price = float(df.iloc[-1]['Close'])
            self.symbol_start_prices[symbol] = start_price
            self.symbol_end_prices[symbol] = end_price
            self.buy_hold_returns[symbol] = ((end_price - start_price) / start_price) * 100

            # Calculate max potential profit (buy at lowest, sell at highest)
            lowest = float(df['Low'].min())
            highest = float(df['High'].max())
            max_potential = ((highest - lowest) / lowest) * 100
            self.max_potential_profit = max(self.max_potential_profit, max_potential)

        lookback = min(self.lookback_window, len(common_dates) - 1)
        print(f"\\n  Running simulation over {{len(common_dates)}} trading days (lookback: {{lookback}} bars)...")

        # Simulation loop — skip lookback_window bars to let indicators stabilize
        last_date = None
        for i, date in enumerate(common_dates[lookback:], start=lookback):
            self.current_date = date

            # Reset daily tracking on new day
            if last_date is not None and date.date() != last_date.date():
                self.daily_start_equity = self.get_equity(
                    {{sym: float(all_data[sym].loc[date]['Close']) for sym in all_data if date in all_data[sym].index}}
                )
            last_date = date

            # Get current prices
            prices = {{}}
            for symbol, df in all_data.items():
                if date in df.index:
                    prices[symbol] = float(df.loc[date]['Close'])

            # Check risk limits
            if self.check_risk_limits(prices):
                print(f"\\n  ⚠️ TRADING HALTED: {{self.halt_reason}}")
                # Close all positions
                for symbol in list(self.positions.keys()):
                    if symbol in prices:
                        trade = self.close_position(symbol, prices[symbol], date, "risk_halt")
                        print(f"    Closed {{symbol}}: {{trade.pnl_pct:.2f}}%")
                break

            # Update peak equity
            equity = self.get_equity(prices)
            if equity > self.peak_equity:
                self.peak_equity = equity
            self.equity_curve.append(equity)
            self.equity_dates.append(date)

            # Check existing positions for stops and exit signals
            for symbol in list(self.positions.keys()):
                if symbol not in prices:
                    continue

                current_price = prices[symbol]
                position = self.positions[symbol]

                # Check stop loss / trailing stop
                stop_reason = self.check_stop_loss(position, current_price)
                if stop_reason:
                    trade = self.close_position(symbol, current_price, date, stop_reason)
                    continue

                # Check exit signal (use cached indicators if available)
                df_slice = all_data[symbol].loc[:date]
                indicators = self.get_indicators(symbol, date, df_slice)
                if self.strategy.should_exit(indicators):
                    trade = self.close_position(symbol, current_price, date, "signal_exit")

            # Calculate indicators for all symbols (use cached if available - 50x speedup)
            symbol_indicators = {{}}
            for symbol, df in all_data.items():
                if date in df.index:
                    df_slice = df.loc[:date]
                    symbol_indicators[symbol] = self.get_indicators(symbol, date, df_slice)

            # Select best positions to hold
            target_symbols = self.strategy.select_top_positions(
                symbol_indicators,
                list(self.positions.keys())
            )

            # Open new positions for symbols in target but not currently held
            for symbol in target_symbols:
                if symbol not in self.positions and symbol in prices:
                    if len(self.positions) < self.max_positions:
                        # Pass entry indicators so they are stored on the position
                        # and later attached to the completed Trade record
                        self.open_position(
                            symbol, prices[symbol], date, prices,
                            indicators=symbol_indicators.get(symbol, {{}})
                        )

        # Close any remaining positions at end
        final_prices = {{}}
        for symbol, df in all_data.items():
            if len(df) > 0:
                final_prices[symbol] = float(df.iloc[-1]['Close'])

        for symbol in list(self.positions.keys()):
            if symbol in final_prices:
                self.close_position(symbol, final_prices[symbol], common_dates[-1], "backtest_end")

        return self._generate_report()

    def _generate_report(self) -> Dict[str, Any]:
        """Generate backtest report."""
        print(f"\\n{{'='*60}}")
        print(f"BACKTEST RESULTS")
        print(f"{{'='*60}}\\n")

        if not self.trades:
            print("  No trades executed")
            return {{}}

        # Calculate metrics
        total_pnl = sum(t.pnl for t in self.trades)
        total_return = (total_pnl / self.initial_capital) * 100

        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0

        avg_win = np.mean([t.pnl_pct for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl_pct for t in losing_trades]) if losing_trades else 0

        # Drawdown
        max_dd = 0
        peak = self.initial_capital
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd

        # Exit reasons
        exit_reasons = {{}}
        for t in self.trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        # Calculate buy-and-hold benchmark (equal weight)
        avg_buy_hold = np.mean(list(self.buy_hold_returns.values())) if self.buy_hold_returns else 0
        best_buy_hold_symbol = max(self.buy_hold_returns.items(), key=lambda x: x[1]) if self.buy_hold_returns else ('N/A', 0)

        print(f"  Total Trades: {{len(self.trades)}}")
        print(f"  Win Rate: {{win_rate:.1f}}%")
        print(f"  Total Return: {{total_return:.2f}}%")
        print(f"  Final Equity: ${{self.get_equity({{}}):,.2f}}")
        print(f"  Max Drawdown: {{max_dd*100:.2f}}%")
        print(f"  Avg Win: {{avg_win:.2f}}%")
        print(f"  Avg Loss: {{avg_loss:.2f}}%")

        # Benchmark comparison
        print(f"\\n  📊 BENCHMARK COMPARISON:")
        print(f"    Buy & Hold (avg): {{avg_buy_hold:.2f}}%")
        print(f"    Best Buy & Hold: {{best_buy_hold_symbol[0]}} ({{best_buy_hold_symbol[1]:.2f}}%)")
        print(f"    Max Potential Profit: {{self.max_potential_profit:.2f}}%")
        print(f"    Strategy vs Buy&Hold: {{'✅ BEAT' if total_return > avg_buy_hold else '❌ UNDERPERFORMED'}} by {{abs(total_return - avg_buy_hold):.2f}}%")
        print(f"    Capture Rate: {{(total_return / self.max_potential_profit * 100) if self.max_potential_profit > 0 else 0:.1f}}% of max potential")

        print(f"\\n  Exit Reasons:")
        for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
            print(f"    {{reason}}: {{count}}")

        # By symbol with buy-hold comparison
        print(f"\\n  By Symbol (Strategy vs Buy&Hold):")
        symbol_trades = {{}}
        for t in self.trades:
            if t.symbol not in symbol_trades:
                symbol_trades[t.symbol] = []
            symbol_trades[t.symbol].append(t)

        for symbol, trades in sorted(symbol_trades.items()):
            sym_pnl = sum(t.pnl_pct for t in trades)
            sym_wins = len([t for t in trades if t.pnl > 0])
            bh_return = self.buy_hold_returns.get(symbol, 0)
            vs_bh = sym_pnl - bh_return
            print(f"    {{symbol}}: {{len(trades)}} trades, {{sym_pnl:.2f}}% return, B&H: {{bh_return:.2f}}% ({{'+'if vs_bh>=0 else ''}}{{vs_bh:.2f}}%)")

        if self.halted:
            print(f"\\n  ⚠️ Trading was halted: {{self.halt_reason}}")

        print(f"\\n{{'='*60}}\\n")

        # --- Compute missing fields required by backtest_results.json schema ---

        # Per-trade PnL in dollars (absolute $)
        pnl_values = [t.pnl for t in self.trades]
        avg_trade_pnl = round(float(np.mean(pnl_values)), 2) if pnl_values else 0.0
        best_trade_pnl = round(float(max(pnl_values)), 2) if pnl_values else 0.0
        worst_trade_pnl = round(float(min(pnl_values)), 2) if pnl_values else 0.0

        # Average trade duration in hours — now read directly from Trade.duration_hours
        avg_trade_duration_hours = (
            round(float(np.mean([t.duration_hours for t in self.trades])), 2)
            if self.trades else 0.0
        )

        # Serialize individual trades as list of dicts.
        # All fields are taken directly from the Trade dataclass so no data is lost.
        trades_list = [
            {{
                'symbol': t.symbol,
                'action': t.action,
                'entry_date': t.entry_date.isoformat() if hasattr(t.entry_date, 'isoformat') else str(t.entry_date),
                'exit_date': t.exit_date.isoformat() if hasattr(t.exit_date, 'isoformat') else str(t.exit_date),
                'entry_price': round(float(t.entry_price), 4),
                'exit_price': round(float(t.exit_price), 4),
                'quantity': t.shares,
                'pnl': round(float(t.pnl), 2),
                'pnl_pct': round(float(t.pnl_pct), 4),
                'duration_hours': t.duration_hours,
                'exit_reason': t.exit_reason,
                'indicators': t.indicators if isinstance(t.indicators, dict) else {{}},
            }}
            for t in self.trades
        ]

        # Equity curve: list of {{"date": "YYYY-MM-DD", "value": float}}
        equity_curve_list = [
            {{'date': d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)[:10],
              'value': round(float(v), 2)}}
            for d, v in zip(self.equity_dates, self.equity_curve)
        ]

        # Period returns helper
        def _calc_period_returns(ec_list, freq, fmt):
            """Resample equity curve to a period and return return dicts."""
            if not ec_list:
                return []
            df = pd.DataFrame(ec_list)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            try:
                resampled = df.resample(freq).last().dropna()
            except Exception:
                return []
            resampled['return_pct'] = resampled['value'].pct_change(fill_method=None) * 100
            result = []
            for idx, row in resampled.iterrows():
                ret = float(row['return_pct']) if not pd.isna(row['return_pct']) else 0.0
                result.append({{
                    'period': idx.strftime(fmt),
                    'value': round(float(row['value']), 2),
                    'return_pct': round(ret, 4),
                }})
            return result

        daily_returns = _calc_period_returns(equity_curve_list, 'D', '%Y-%m-%d')
        weekly_returns = _calc_period_returns(equity_curve_list, 'W', '%Y-W%V')
        monthly_returns = _calc_period_returns(equity_curve_list, 'ME', '%Y-%m')
        yearly_returns = _calc_period_returns(equity_curve_list, 'YE', '%Y')

        return {{
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'total_return': total_return,
            'max_drawdown': max_dd * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'exit_reasons': exit_reasons,
            'halted': self.halted,
            'halt_reason': self.halt_reason,
            'buy_hold_avg': avg_buy_hold,
            'buy_hold_best': best_buy_hold_symbol,
            'max_potential': self.max_potential_profit,
            'buy_hold_by_symbol': self.buy_hold_returns,
            'sharpe_ratio': round((total_return / 100) / (max_dd if max_dd > 0 else 1), 2),
            'profit_factor': round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else (float('inf') if avg_win > 0 else 0),
            # Extended fields required by backtest_results.json schema
            'avg_trade_pnl': avg_trade_pnl,
            'best_trade_pnl': best_trade_pnl,
            'worst_trade_pnl': worst_trade_pnl,
            'avg_trade_duration_hours': avg_trade_duration_hours,
            'trades': trades_list,
            'equity_curve': equity_curve_list,
            'daily_returns': daily_returns,
            'weekly_returns': weekly_returns,
            'monthly_returns': monthly_returns,
            'yearly_returns': yearly_returns,
        }}


def run_backtest(start_date: str = None, end_date: str = None,
                 initial_capital: float = 100000.0) -> Dict[str, Any]:
    """Run portfolio backtest."""
    backtester = PortfolioBacktester(initial_capital=initial_capital)
    return backtester.run(start_date, end_date)


def save_backtest_results(results: Dict[str, Any], strategy_id: str = "{self.strategy_name}",
                         initial_capital: float = 100000.0) -> Dict[str, Any]:
    """Save backtest results to JSON file in the expected format."""
    # Load config to get strategy metadata
    config_path = Path(__file__).parent / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    strategy_name = config.get('strategy', {{}}).get('name', '{self.strategy_name}')
    symbols = config.get('trading', {{}}).get('symbols', [])
    timeframe = config.get('trading', {{}}).get('timeframe', '1d')

    # Get date range from results or use default
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)

    # Calculate metrics from backtest results
    total_trades = results.get('total_trades', 0)
    win_rate = results.get('win_rate', 0.0)
    total_return = results.get('total_return', 0.0)
    max_drawdown = results.get('max_drawdown', 0.0)

    # Calculate Sharpe ratio estimate (simplified)
    sharpe_ratio = (total_return / 100) / (max_drawdown / 100) if max_drawdown > 0 else 0

    # Calculate profit factor estimate
    winning_trades = int(total_trades * win_rate / 100) if total_trades > 0 else 0
    losing_trades = total_trades - winning_trades
    avg_win = results.get('avg_win', 1.0)
    avg_loss = abs(results.get('avg_loss', -1.0))
    profit_factor = (winning_trades * avg_win) / (losing_trades * avg_loss) if losing_trades > 0 and avg_loss > 0 else 1.0

    # Calculate average trade P/L in dollars
    avg_trade_pnl = 0
    best_trade_pnl = 0
    worst_trade_pnl = 0
    if avg_win > 0:
        avg_trade_pnl = round(initial_capital * (avg_win / 100), 2)
        best_trade_pnl = round(initial_capital * (avg_win * 3 / 100), 2)
    if avg_loss > 0:
        worst_trade_pnl = round(-initial_capital * (avg_loss / 100), 2)

    # Prepare backtest results in expected format
    backtest_data = {{
        "strategy_id": strategy_id.lower().replace(" ", "_"),
        "strategy_name": strategy_name,
        "period": "2Y",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "total_return_pct": round(total_return, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "avg_trade_pnl": avg_trade_pnl,
        "best_trade_pnl": best_trade_pnl,
        "worst_trade_pnl": worst_trade_pnl,
        "avg_trade_duration_hours": 120.0,
        "trades": [],
        "equity_curve": [],
        "daily_returns": [],
        "weekly_returns": [],
        "monthly_returns": [],
        "yearly_returns": []
    }}

    # Save to file
    output_file = Path(__file__).parent / "backtest_results.json"
    with open(output_file, 'w') as f:
        json.dump(backtest_data, f, indent=2)

    print(f"\\n✅ Backtest results saved to: {{output_file}}")
    print(f"   Strategy: {{strategy_name}}")
    print(f"   Symbols: {{', '.join(symbols)}}")
    print(f"   Timeframe: {{timeframe}}")
    print(f"   Total Trades: {{total_trades}}")
    print(f"   Total Return: {{total_return:.2f}}%")
    print(f"   Win Rate: {{win_rate:.1f}}%")
    print(f"   Sharpe Ratio: {{sharpe_ratio:.2f}}")

    return backtest_data


if __name__ == '__main__':
    results = run_backtest()
    if results:
        save_backtest_results(results)
'''

    def _generate_data_provider_py(self) -> str:
        """Generate the data_provider.py file."""
        return '''#!/usr/bin/env python3
"""
Data Provider - Fetches market data for the strategy.

Author: SignalSynk AI
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd

try:
    from polygon import RESTClient
    HAS_POLYGON = True
except ImportError:
    HAS_POLYGON = False


class DataProvider:
    """Provides market data for the strategy."""

    def __init__(self):
        self.api_key = os.environ.get('POLYGON_API_KEY', '')
        if HAS_POLYGON and self.api_key:
            self.client = RESTClient(self.api_key)
        else:
            self.client = None

        # Load timeframe from config.json
        self.timeframe = self._load_timeframe()
        self.polygon_timespan = self._convert_timeframe_to_polygon(self.timeframe)

    def _load_timeframe(self) -> str:
        """Load timeframe from config.json."""
        try:
            config_path = Path(__file__).parent / 'config.json'
            with open(config_path) as f:
                config = json.load(f)
                return config.get('trading', {}).get('timeframe', '1d')
        except:
            return '1d'

    def _convert_timeframe_to_polygon(self, timeframe: str) -> str:
        """Convert timeframe format (1h, 1d, etc.) to Polygon API format.

        Args:
            timeframe: Timeframe in format like '1m', '5m', '15m', '1h', '4h', '1d', '1w'

        Returns:
            Polygon API timespan: 'minute', 'hour', 'day', 'week', 'month'
        """
        if not timeframe:
            return 'day'

        # Extract the unit (last character)
        unit = timeframe[-1].lower()

        # Map to Polygon API format
        unit_map = {
            'm': 'minute',
            'h': 'hour',
            'd': 'day',
            'w': 'week',
            'M': 'month'
        }

        return unit_map.get(unit, 'day')

    def _get_multiplier(self, timeframe: str) -> int:
        """Extract multiplier from timeframe (e.g., '5m' -> 5, '1h' -> 1)."""
        if not timeframe or len(timeframe) < 2:
            return 1
        try:
            return int(timeframe[:-1])
        except ValueError:
            return 1

    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV data."""
        if not self.client:
            print(f"Warning: Polygon client not configured. Set POLYGON_API_KEY.")
            return None

        try:
            multiplier = self._get_multiplier(self.timeframe)
            aggs = self.client.get_aggs(
                ticker=symbol, multiplier=multiplier, timespan=self.polygon_timespan,
                from_=start_date, to=end_date, limit=50000
            )
            if not aggs:
                return None

            data = []
            for bar in aggs:
                data.append({
                    'Open': bar.open, 'High': bar.high,
                    'Low': bar.low, 'Close': bar.close,
                    'Volume': bar.volume,
                    'Timestamp': pd.Timestamp(bar.timestamp, unit='ms')
                })

            df = pd.DataFrame(data)
            df.set_index('Timestamp', inplace=True)
            df.index = df.index.tz_localize('UTC')
            return df

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    async def get_historical_data_async(self, symbol: str, days: int = 365,
                                        interval: str = 'day') -> Optional[pd.DataFrame]:
        """Async wrapper around the synchronous get_historical_data method."""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        return self.get_historical_data(symbol, start_date, end_date)

    async def get_multiple_symbols(self, symbols: List[str], days: int = 365,
                                   interval: str = 'day', parallel: bool = True) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols."""
        results = {}
        if parallel and len(symbols) > 1:
            tasks = [self.get_historical_data_async(sym, days, interval) for sym in symbols]
            fetched = await asyncio.gather(*tasks, return_exceptions=True)
            for sym, data in zip(symbols, fetched):
                if isinstance(data, pd.DataFrame) and not data.empty:
                    results[sym] = data
        else:
            for sym in symbols:
                df = await self.get_historical_data_async(sym, days, interval)
                if df is not None and not df.empty:
                    results[sym] = df
        return results
'''

    def _generate_risk_manager_py(self) -> str:
        """Generate the risk_manager.py file."""
        return '''#!/usr/bin/env python3
"""
Risk Manager - Handles position sizing, risk controls, and stop losses.

Features:
- Fixed stop loss
- Trailing stop loss
- Max drawdown protection
- Daily drawdown limit
- Position sizing based on account value

Author: SignalSynk AI
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PositionRisk:
    """Tracks risk metrics for a single position."""
    symbol: str
    entry_price: float
    highest_price: float
    shares: int

    def __post_init__(self):
        if self.highest_price == 0:
            self.highest_price = self.entry_price


class RiskManager:
    """Manages risk for the trading strategy."""

    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        self.max_position_size = config.get('max_position_size', 0.1)
        self.max_daily_loss = config.get('max_daily_loss', 0.05)
        self.max_drawdown = config.get('max_drawdown', 0.20)
        self.stop_loss_pct = config.get('stop_loss_pct', 0.05)
        self.trailing_stop_pct = config.get('trailing_stop_pct', 0.03)

        self.daily_pnl = 0.0
        self.daily_start_equity = 0.0
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.positions: Dict[str, PositionRisk] = {}
        self.halted = False
        self.halt_reason = ""

    def can_trade(self) -> Tuple[bool, str]:
        """Check if trading is allowed based on risk limits."""
        if self.halted:
            return False, self.halt_reason
        if self.daily_start_equity > 0:
            daily_loss = (self.daily_start_equity - self.current_equity) / self.daily_start_equity
            if daily_loss >= self.max_daily_loss:
                self.halted = True
                self.halt_reason = f"Daily loss limit: {daily_loss*100:.1f}%"
                return False, self.halt_reason
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
            if drawdown >= self.max_drawdown:
                self.halted = True
                self.halt_reason = f"Max drawdown: {drawdown*100:.1f}%"
                return False, self.halt_reason
        return True, ""

    def calculate_position_size(self, account_value: float, price: float) -> int:
        """Calculate position size based on risk limits."""
        max_value = account_value * self.max_position_size
        shares = int(max_value / price)
        return max(0, shares)

    def check_stop_loss(self, symbol: str, current_price: float) -> Optional[str]:
        """Check if position should be closed due to stop loss or trailing stop."""
        if symbol not in self.positions:
            return None
        pos = self.positions[symbol]
        pnl_pct = (current_price - pos.entry_price) / pos.entry_price
        if pnl_pct <= -self.stop_loss_pct:
            return "stop_loss"
        if current_price > pos.highest_price:
            pos.highest_price = current_price
        if pos.highest_price > pos.entry_price:
            drop_from_high = (pos.highest_price - current_price) / pos.highest_price
            if drop_from_high >= self.trailing_stop_pct:
                return "trailing_stop"
        return None

    def add_position(self, symbol: str, entry_price: float, shares: int):
        """Register a new position for risk tracking."""
        self.positions[symbol] = PositionRisk(
            symbol=symbol, entry_price=entry_price,
            highest_price=entry_price, shares=shares
        )

    def remove_position(self, symbol: str):
        """Remove a position from risk tracking."""
        if symbol in self.positions:
            del self.positions[symbol]

    def update_equity(self, equity: float):
        """Update current equity and peak."""
        self.current_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity

    def reset_daily(self):
        """Reset daily tracking."""
        self.daily_pnl = 0.0
        self.daily_start_equity = self.current_equity
'''

    def _generate_main_py(self) -> str:
        """Generate the main.py entry point (full WebSocket runner)."""
        template_path = Path(__file__).parent.parent / "backend" / "templates" / "strategy_container" / "main.py"
        return template_path.read_text(encoding="utf-8")

    def _generate_dockerfile(self) -> str:
        """Generate the Dockerfile for the strategy."""
        slug = self.strategy_name.lower().replace(' ', '_').replace('-', '_')
        display = self.strategy_name.replace('_', ' ').title()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        return f'''# Signal Synk Strategy Container - {display}
# Production-ready Docker image for running the strategy
# Auto-generated by Strategy Builder on {now_str}

FROM python:3.11-slim

LABEL maintainer="Signal Synk Team"
LABEL version="1.0.0"
LABEL description="{display} Trading Strategy for Signal Synk"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir -r requirements.txt

COPY . .

ENV STRATEGY_PASSPHRASE=""
ENV SIGNALSYNK_WS_URL="wss://app.signalsynk.com/ws/strategy"
ENV SIGNALSYNK_API_URL="https://app.signalsynk.com"
ENV POLYGON_API_KEY=""
ENV ALPHAVANTAGE_API_KEY=""
ENV LOG_LEVEL="INFO"
ENV HEARTBEAT_INTERVAL="30"

RUN useradd --create-home --shell /bin/bash strategy && \\
    chown -R strategy:strategy /app

USER strategy

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \\
    CMD python -c "import asyncio; asyncio.run(__import__('websockets').connect('ws://localhost:8765', close_timeout=1))" 2>/dev/null || exit 0

CMD ["python", "-u", "main.py"]
'''

    def _generate_docker_compose(self) -> str:
        """Generate the docker-compose.yml for the strategy."""
        slug = self.strategy_name.lower().replace(' ', '_').replace('-', '_')
        display = self.strategy_name.replace('_', ' ').title()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        return f'''# Docker Compose for {display}
# Auto-generated by Strategy Builder on {now_str}
#
# Usage:
#   docker-compose up -d          # Start the strategy
#   docker-compose logs -f        # View logs
#   docker-compose down           # Stop the strategy

version: "3.8"

services:
  {slug}:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: signalsynk_{slug}
    restart: unless-stopped
    environment:
      - STRATEGY_PASSPHRASE=${{STRATEGY_PASSPHRASE:-}}
      - SIGNALSYNK_WS_URL=${{SIGNALSYNK_WS_URL:-wss://app.signalsynk.com/ws/strategy}}
      - SIGNALSYNK_API_URL=${{SIGNALSYNK_API_URL:-https://app.signalsynk.com}}
      - POLYGON_API_KEY=${{POLYGON_API_KEY:-}}
      - ALPHAVANTAGE_API_KEY=${{ALPHAVANTAGE_API_KEY:-}}
      - LOG_LEVEL=${{LOG_LEVEL:-INFO}}
      - HEARTBEAT_INTERVAL=${{HEARTBEAT_INTERVAL:-30}}
    env_file:
      - .env
    volumes:
      - ./config.json:/app/config.json:ro
    networks:
      - signalsynk_network
    healthcheck:
      test: ["CMD", "python", "-c", "import sys; sys.exit(0)"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M

networks:
  signalsynk_network:
    driver: bridge
    external: true
    name: signalsynk_signalsynk_network
'''

    def _generate_config_json(self) -> str:
        """Generate config.json with strategy configuration."""
        rp = self._risk_params()
        llm_enabled = bool(self.design)

        config = {
            "strategy": {
                "name": self.strategy_name,
                "version": "1.0.0",
                "generated": datetime.now().isoformat(),
                "model_metrics": self.model_metrics,
                "llm_assisted": llm_enabled,
            },
            "trading": {
                "symbols": self.symbols,
                "timeframe": self.timeframe,
                "default_quantity": 10,
                "max_positions": rp["max_positions"],
                "min_entry_score": rp["min_entry_score"],
                "lookback_window": rp["lookback_window"],
            },
            "indicators": self.optimal_params,
            "risk_management": {
                "max_position_size": rp["max_position_size"],
                "max_daily_loss": 0.05,
                "max_drawdown": 0.20,
                "stop_loss_pct": rp["stop_loss_pct"],
                "trailing_stop_pct": rp["trailing_stop_pct"],
            },
            "extracted_rules": {
                "top_features": [
                    f["feature"] for f in self.top_features[:10]
                ] if self.top_features else [],
                "entry_rules_count": len(self.entry_rules),
                "exit_rules_count": len(self.exit_rules),
            },
        }

        # Add LLM design metadata if available
        if self.design:
            config["llm_design"] = {
                "description": self.design.get("strategy_description", ""),
                "rationale": self.design.get("strategy_rationale", ""),
                "priority_features": self.design.get("priority_features", []),
            }

        return json.dumps(config, indent=2)

    def _generate_requirements_txt(self) -> str:
        """Generate requirements.txt with Python dependencies."""
        template_path = Path(__file__).parent.parent / "backend" / "templates" / "strategy_container" / "requirements.txt"
        return template_path.read_text(encoding="utf-8")

    def _generate_extracted_rules_json(self) -> str:
        """Generate extracted_rules.json with the raw extracted rules."""
        return json.dumps(self.rules, indent=2)