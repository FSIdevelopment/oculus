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

    def __post_init__(self):
        self.highest_price = self.entry_price


@dataclass
class Trade:
    """Completed trade record."""
    symbol: str
    entry_price: float
    exit_price: float
    entry_date: datetime
    exit_date: datetime
    shares: int
    pnl: float
    pnl_pct: float
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


def run_backtest():
    """Run the portfolio backtest."""
    backtester = PortfolioBacktester()
    print(f"PortfolioBacktester initialized with ${{backtester.initial_capital:,.0f}}")
    print("Backtest runner ready — call backtester methods to execute.")
    return None


if __name__ == '__main__':
    results = run_backtest()
'''

    def _generate_data_provider_py(self) -> str:
        """Generate the data_provider.py file."""
        return '''#!/usr/bin/env python3
"""
Data Provider - Fetches market data for the strategy.

Author: SignalSynk AI
"""

import os
import asyncio
from datetime import datetime, timedelta
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
        self.api_key = os.environ.get('POLYGON_MASSIVE_API_KEY', '')
        if HAS_POLYGON and self.api_key:
            self.client = RESTClient(self.api_key)
        else:
            self.client = None

    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV data."""
        if not self.client:
            print(f"Warning: Polygon client not configured. Set POLYGON_MASSIVE_API_KEY.")
            return None

        try:
            aggs = self.client.get_aggs(
                ticker=symbol, multiplier=1, timespan='day',
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
        """Generate the main.py entry point."""
        return f'''#!/usr/bin/env python3
"""
Main entry point for {self.strategy_name}

Author: SignalSynk AI
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from strategy import TradingStrategy
from data_provider import DataProvider
from risk_manager import RiskManager


def main():
    """Main entry point."""
    print(f"\\n{{'='*60}}")
    print(f"{self.strategy_name}")
    print(f"{{'='*60}}\\n")

    strategy = TradingStrategy()
    data_provider = DataProvider()
    risk_manager = RiskManager()

    print("Strategy initialized successfully!")
    print(f"  Symbols: {{strategy.get_symbols()}}")
    print("\\nRun backtest.py to test the strategy.")


if __name__ == '__main__':
    main()
'''

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
ENV POLYGON_MASSIVE_API_KEY=""
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
      - POLYGON_MASSIVE_API_KEY=${{POLYGON_MASSIVE_API_KEY:-}}
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
        return """# Auto-generated requirements
pandas>=2.0.0
numpy>=1.24.0
polygon-api-client>=1.12.0
"""

    def _generate_extracted_rules_json(self) -> str:
        """Generate extracted_rules.json with the raw extracted rules."""
        return json.dumps(self.rules, indent=2)