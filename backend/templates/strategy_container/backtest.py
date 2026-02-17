#!/usr/bin/env python3
"""
Signal Synk Strategy Backtest Template

This file runs a historical backtest of the strategy and outputs results to a JSON file.
The backtest simulates trading over a historical period and calculates performance metrics.

Environment Variables:
- BACKTEST_MODE: Set to "true" to run in backtest mode
- BACKTEST_PERIOD: Time period to test (1M, 3M, 6M, 1Y, 2Y, 5Y)
- STRATEGY_ID: The strategy ID from the database
- ALPHAVANTAGE_API_KEY: API key for fetching historical data

Output:
- Creates backtest_results.json in the strategy directory

Author: Signal Synk Team
Copyright © 2026 Forefront Solutions Inc. All rights reserved.
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np

# Import the strategy class
from strategy import TradingStrategy
from risk_manager import RiskManager

# Configure logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('Backtest')

# Configuration from environment
BACKTEST_PERIOD = os.getenv('BACKTEST_PERIOD', '1Y')
BACKTEST_TIMEFRAME = os.getenv('BACKTEST_TIMEFRAME', '1D')
STRATEGY_ID = os.getenv('STRATEGY_ID', 'unknown')
RESULTS_FILE = Path(__file__).parent / 'backtest_results.json'


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (datetime,)):
            return obj.isoformat()
        if pd.isna(obj):
            return None
        return super().default(obj)


@dataclass
class Trade:
    """Represents a single trade in the backtest"""
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
    """Complete backtest results structure - stored in database"""
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


def get_period_days(period: str) -> int:
    """Convert period string to number of days"""
    period_map = {
        '1M': 30,
        '3M': 90,
        '6M': 180,
        '1Y': 365,
        '2Y': 730,
        '5Y': 1825
    }
    return period_map.get(period, 180)


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate annualized Sharpe ratio"""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    excess_returns = returns - (risk_free_rate / 252)
    return float(np.sqrt(252) * excess_returns.mean() / returns.std())


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calculate maximum drawdown percentage"""
    if len(equity_curve) < 2:
        return 0.0
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve - peak) / peak
    return float(abs(drawdown.min()) * 100)


class BacktestRunner:
    """
    Runs the strategy backtest - COMPLETELY ISOLATED from live trading.

    This class:
    - Only reads historical data (no real trades)
    - Does NOT send any signals to brokers
    - Simulates trades internally for performance calculation
    """

    def __init__(self, strategy: TradingStrategy, period: str):
        self.strategy = strategy
        self.period = period
        self.initial_capital = 100000.0
        self.capital = self.initial_capital
        self.positions: Dict[str, Dict] = {}  # symbol -> position info
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict[str, float]] = []
        self.daily_returns_list: List[float] = []
        self.ohlc_data: List[Dict[str, Any]] = []  # For candlestick charts

        # Position sizing from strategy config
        trading_config = getattr(strategy, 'config', {}).get('trading', {})
        self.max_positions = trading_config.get('max_positions', 1)
        self.position_size_pct = trading_config.get('position_size_pct', 100)  # Default 100% of allocation
        self.pyramiding = trading_config.get('pyramiding', 1)  # For strategies with pyramiding

        # Initialize Risk Manager with config
        risk_config = getattr(strategy, 'config', {}).get('risk_management', {})
        self.risk_manager = RiskManager(risk_config)

        # Initialize data provider (isolated, read-only)
        from data_provider import DataProvider
        self.data_provider = DataProvider()

    async def fetch_historical_data(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Fetch historical data using DataProvider (AlphaVantage/Yahoo)"""
        try:
            # Use the configured timeframe for data fetching
            df = await self.data_provider.get_historical_data(symbol, days, interval=BACKTEST_TIMEFRAME)
            if df is None or df.empty:
                logger.warning(f"No data available for {symbol}")
                return None

            logger.info(f"Fetched {len(df)} bars for {symbol} ({days} days, {BACKTEST_TIMEFRAME} timeframe)")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            return None

    def calculate_position_size(self, price: float, current_prices: Dict[str, float]) -> int:
        """
        Calculate position size based on available capital and strategy config.

        Position sizing logic:
        1. Calculate total portfolio value (cash + positions)
        2. Determine allocation per position based on max_positions
        3. Apply position_size_pct to the allocation
        4. Calculate number of shares that can be bought

        Examples:
        - $100k portfolio, max_positions=1, position_size_pct=100 -> use all $100k
        - $100k portfolio, max_positions=6, position_size_pct=100 -> use $16,666 per position
        - $100k portfolio, max_positions=1, position_size_pct=50 -> use $50k
        """
        # Calculate current portfolio value
        portfolio_value = self.calculate_portfolio_value(current_prices)

        # Determine allocation per position slot
        # Use max of max_positions or pyramiding (for pyramid strategies)
        max_slots = max(self.max_positions, self.pyramiding)
        allocation_per_slot = portfolio_value / max_slots

        # Apply position size percentage
        position_allocation = allocation_per_slot * (self.position_size_pct / 100.0)

        # Use available capital if less than allocation (can't spend more than we have)
        position_allocation = min(position_allocation, self.capital)

        # Calculate shares (round down to whole shares)
        if price <= 0:
            return 0
        quantity = int(position_allocation / price)

        return max(0, quantity)

    def execute_trade(self, symbol: str, action: str, price: float,
                      quantity: int, date: datetime, indicators: Dict,
                      current_prices: Dict[str, float] = None,
                      exit_reason: str = 'SIGNAL') -> Optional[Trade]:
        """Execute a simulated trade with dynamic position sizing"""
        if action == 'BUY':
            if symbol in self.positions:
                return None  # Already in position

            # Calculate position size based on portfolio and config
            if current_prices is not None:
                quantity = self.calculate_position_size(price, current_prices)

            if quantity <= 0:
                return None

            cost = price * quantity
            if cost > self.capital:
                quantity = int(self.capital / price)
                if quantity <= 0:
                    return None
                cost = price * quantity

            self.capital -= cost
            self.positions[symbol] = {
                'entry_price': price,
                'entry_date': date,
                'quantity': quantity,
                'indicators': indicators
            }

            # Register position with risk manager for tracking
            self.risk_manager.register_position(symbol, price, date, quantity)

            logger.debug(f"BUY {quantity} {symbol} @ ${price:.2f}")
            return None  # Trade not complete until exit

        elif action == 'SELL':
            if symbol not in self.positions:
                return None  # No position to sell

            position = self.positions.pop(symbol)
            proceeds = price * position['quantity']
            self.capital += proceeds

            pnl = proceeds - (position['entry_price'] * position['quantity'])
            pnl_pct = (price - position['entry_price']) / position['entry_price'] * 100
            duration = (date - position['entry_date']).total_seconds() / 3600

            # Unregister position from risk manager
            self.risk_manager.unregister_position(symbol)

            trade = Trade(
                entry_date=position['entry_date'].isoformat(),
                exit_date=date.isoformat(),
                symbol=symbol,
                action='LONG',
                entry_price=position['entry_price'],
                exit_price=price,
                quantity=position['quantity'],
                pnl=round(pnl, 2),
                pnl_pct=round(pnl_pct, 2),
                duration_hours=round(duration, 2),
                indicators=position['indicators'],
                exit_reason=exit_reason
            )
            self.trades.append(trade)
            logger.debug(f"SELL {position['quantity']} {symbol} @ ${price:.2f} (P/L: ${pnl:.2f}) [{exit_reason}]")
            return trade

        return None

    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        total = self.capital
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                total += current_prices[symbol] * position['quantity']
        return total

    def _check_intraday_stops(
        self,
        symbol: str,
        low_price: float,
        high_price: float,
        close_price: float
    ) -> tuple:
        """
        Check if stop loss or trailing stop was triggered intraday.

        In live trading, stops trigger the moment the price touches the level.
        For backtesting with daily bars, we simulate this by:
        1. Using the HIGH to update trailing stop peaks
        2. Checking if the LOW touched stop levels
        3. Calculating realistic exit price at the stop level

        Args:
            symbol: Stock symbol
            low_price: Low of the day
            high_price: High of the day
            close_price: Close of the day

        Returns:
            (exit_price, exit_reason) or (None, None) if no stop triggered
        """
        if symbol not in self.positions:
            return None, None

        position = self.positions[symbol]
        entry_price = position['entry_price']

        # Get risk manager's position tracking
        pos_risk = self.risk_manager.position_risks.get(symbol)
        if not pos_risk:
            return None, None

        # Calculate stop levels
        stop_loss_price = entry_price * (1 - self.risk_manager.stop_loss_pct / 100)
        take_profit_price = entry_price * (1 + self.risk_manager.take_profit_pct / 100)

        # Trailing stop: X% below the highest price seen
        highest_price = pos_risk.highest_price
        trailing_stop_price = highest_price * (1 - self.risk_manager.trailing_stop_pct / 100)

        # Check if LOW touched stop loss (worst case first)
        if low_price <= stop_loss_price:
            self.risk_manager.stop_loss_exits += 1
            logger.info(f"🛑 {symbol}: STOP LOSS triggered intraday (low ${low_price:.2f} <= stop ${stop_loss_price:.2f})")
            return stop_loss_price, 'STOP_LOSS'

        # Check if HIGH touched take profit
        if high_price >= take_profit_price:
            self.risk_manager.take_profit_exits += 1
            logger.info(f"🎯 {symbol}: TAKE PROFIT triggered intraday (high ${high_price:.2f} >= target ${take_profit_price:.2f})")
            return take_profit_price, 'TAKE_PROFIT'

        # Check if LOW touched trailing stop (only if position was profitable at some point)
        if highest_price > entry_price and low_price <= trailing_stop_price:
            self.risk_manager.trailing_stop_exits += 1
            gain_at_exit = ((trailing_stop_price - entry_price) / entry_price) * 100
            logger.info(f"📉 {symbol}: TRAILING STOP triggered intraday (low ${low_price:.2f} <= trail ${trailing_stop_price:.2f}, peak ${highest_price:.2f}, exit +{gain_at_exit:.1f}%)")
            return trailing_stop_price, 'TRAILING_STOP'

        # No stops triggered - update position with close price for next iteration
        self.risk_manager.update_position_price(symbol, close_price)
        return None, None

    async def run_backtest(self) -> BacktestResults:
        """Run the complete backtest"""
        start_time = datetime.now()
        days = get_period_days(self.period)

        logger.info(f"Starting backtest for {self.period} ({days} days)")
        logger.info(f"Strategy: {self.strategy.config.get('strategy_name', 'Unknown')}")

        # Get symbols - support both single ticker (TICKER) and multi-symbol (symbols) strategies
        symbols = getattr(self.strategy, 'symbols', None) or [getattr(self.strategy, 'TICKER', 'SPY')]
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Initial capital: ${self.initial_capital:,.2f}")

        # Fetch data for all symbols in parallel (much faster for multi-symbol strategies)
        if len(symbols) > 1:
            logger.info(f"Fetching data for {len(symbols)} symbols in parallel...")
            symbol_data = await self.data_provider.get_multiple_symbols(
                symbols, days, interval=BACKTEST_TIMEFRAME, parallel=True
            )
            # Store OHLC data for visualization
            for symbol, df in symbol_data.items():
                self._store_ohlc_data(symbol, df)
        else:
            # Single symbol - use direct fetch
            symbol_data: Dict[str, pd.DataFrame] = {}
            for symbol in symbols:
                df = await self.fetch_historical_data(symbol, days)
                if df is not None and not df.empty:
                    symbol_data[symbol] = df
                    self._store_ohlc_data(symbol, df)

        if not symbol_data:
            raise ValueError("No historical data available for any symbols")

        # Find common date range
        all_dates = set()
        for df in symbol_data.values():
            all_dates.update(df.index.tolist())
        dates = sorted(all_dates)

        start_date = dates[0] if dates else datetime.now(timezone.utc)
        end_date = dates[-1] if dates else datetime.now(timezone.utc)

        # Simulate day by day
        prev_value = self.initial_capital
        for i, date in enumerate(dates):
            current_prices = {}
            current_highs = {}
            current_lows = {}

            # First pass: collect OHLC data for this date
            for symbol, df in symbol_data.items():
                if date in df.index:
                    current_prices[symbol] = df.loc[date, 'Close']
                    current_highs[symbol] = df.loc[date, 'High']
                    current_lows[symbol] = df.loc[date, 'Low']

            # Start new day for risk manager
            portfolio_value = self.calculate_portfolio_value(current_prices)
            self.risk_manager.start_new_day(date, portfolio_value)

            # Update position prices and check risk conditions BEFORE strategy signals
            # IMPORTANT: For realistic stop simulation, we need to check if the LOW
            # of the day touched our stop levels (since stops trigger intraday)
            for symbol in list(self.positions.keys()):
                if symbol in current_prices:
                    close_price = current_prices[symbol]
                    high_price = current_highs.get(symbol, close_price)
                    low_price = current_lows.get(symbol, close_price)

                    # First update with HIGH to capture peak for trailing stop
                    self.risk_manager.update_position_price(symbol, high_price)

                    # Check if LOW triggered any stops (intraday simulation)
                    exit_price, risk_exit = self._check_intraday_stops(
                        symbol, low_price, high_price, close_price
                    )

                    if risk_exit:
                        # Calculate P&L before executing the trade
                        entry_price = self.positions[symbol]['entry_price']
                        pnl_pct = ((exit_price - entry_price) / entry_price) * 100

                        # Use calculated exit price (stop level, not low of day)
                        exit_prices = current_prices.copy()
                        exit_prices[symbol] = exit_price
                        self.execute_trade(
                            symbol, 'SELL', exit_price,
                            0, date, {},
                            current_prices=exit_prices,
                            exit_reason=risk_exit
                        )

                        # Add cooldown for any losing exit (stop loss OR trailing stop with loss)
                        # A trailing stop that exits at a loss means the symbol is not performing
                        if hasattr(self.strategy, 'add_cooldown'):
                            if risk_exit == 'STOP_LOSS':
                                self.strategy.add_cooldown(symbol, date)
                            elif risk_exit == 'TRAILING_STOP' and pnl_pct < 0:
                                logger.info(f"❄️ {symbol}: Adding to cooldown (trailing stop exit at loss: {pnl_pct:.1f}%)")
                                self.strategy.add_cooldown(symbol, date)

            # Check portfolio-level risk limits
            portfolio_value = self.calculate_portfolio_value(current_prices)
            self.risk_manager.check_daily_loss_limit(portfolio_value)
            self.risk_manager.check_drawdown_limit(portfolio_value)

            # Second pass: check signals and execute trades (if not paused)
            for symbol, df in symbol_data.items():
                if date in df.index:
                    # Calculate indicators
                    df_slice = df[df.index <= date]
                    if len(df_slice) >= 20:
                        indicators = await self.strategy.calculate_indicators(df_slice)
                        if indicators:
                            # Pass current date for cooldown checking (if strategy supports it)
                            action = await self.strategy.check_entry_conditions(
                                symbol, indicators, current_date=date
                            )

                            # Position tracking logic:
                            # - BUY signal: Only buy if NOT already holding AND risk allows
                            # - SELL signal: Only sell if currently holding
                            # - No pyramiding (buying more of existing position)
                            # - No short selling (selling what we don't own)

                            is_holding = symbol in self.positions

                            if action == 'BUY' and not is_holding:
                                # Check if risk manager allows new positions
                                if self.risk_manager.can_open_new_position():
                                    # Entry signal - open new position
                                    self.execute_trade(
                                        symbol, 'BUY', current_prices[symbol],
                                        0,  # quantity calculated dynamically
                                        date, indicators,
                                        current_prices=current_prices
                                    )
                                else:
                                    logger.debug(f"Skipping BUY {symbol}: {self.risk_manager.pause_reason}")
                            elif action == 'SELL' and is_holding:
                                # Exit signal - close existing position
                                self.execute_trade(
                                    symbol, 'SELL', current_prices[symbol],
                                    0,  # quantity from position
                                    date, indicators,
                                    current_prices=current_prices,
                                    exit_reason='SIGNAL'
                                )

            # Record equity curve
            portfolio_value = self.calculate_portfolio_value(current_prices)
            self.equity_curve.append({
                'date': date.isoformat(),
                'value': round(portfolio_value, 2)
            })

            # Calculate daily return
            if prev_value > 0:
                daily_return = (portfolio_value - prev_value) / prev_value
                self.daily_returns_list.append(daily_return)
            prev_value = portfolio_value

        # Close any remaining positions at end
        final_prices = {s: symbol_data[s]['Close'].iloc[-1] for s in symbol_data}
        for symbol in list(self.positions.keys()):
            if symbol in symbol_data:
                last_price = symbol_data[symbol]['Close'].iloc[-1]
                self.execute_trade(symbol, 'SELL', last_price, 0, end_date, {},
                                  current_prices=final_prices, exit_reason='END_OF_BACKTEST')

        # Calculate results
        return self._calculate_results(start_date, end_date, start_time)

    def _calculate_results(self, start_date, end_date, start_time) -> BacktestResults:
        """Calculate final backtest results with all period breakdowns"""
        duration = (datetime.now() - start_time).total_seconds()

        # Calculate metrics
        trade_pnls = [t.pnl for t in self.trades]
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]

        final_value = self.equity_curve[-1]['value'] if self.equity_curve else self.initial_capital
        total_return_pct = ((final_value - self.initial_capital) / self.initial_capital) * 100

        equity_series = pd.Series([e['value'] for e in self.equity_curve])
        returns_series = pd.Series(self.daily_returns_list) if self.daily_returns_list else pd.Series([0])

        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Calculate all period returns
        daily_returns = self._calculate_period_returns('D')
        weekly_returns = self._calculate_period_returns('W')
        monthly_returns = self._calculate_period_returns('ME')
        yearly_returns = self._calculate_period_returns('YE')

        # Count exit reasons
        exit_reason_counts = {}
        for trade in self.trades:
            reason = getattr(trade, 'exit_reason', 'SIGNAL')
            exit_reason_counts[reason] = exit_reason_counts.get(reason, 0) + 1

        return BacktestResults(
            strategy_id=STRATEGY_ID,
            strategy_name=self.strategy.config.get('strategy_name', 'Unknown'),
            period=self.period,
            start_date=start_date.isoformat() if hasattr(start_date, 'isoformat') else str(start_date),
            end_date=end_date.isoformat() if hasattr(end_date, 'isoformat') else str(end_date),
            total_return_pct=round(total_return_pct, 2),
            max_drawdown_pct=round(calculate_max_drawdown(equity_series), 2),
            sharpe_ratio=round(calculate_sharpe_ratio(returns_series), 2),
            win_rate_pct=round(len(winning_trades) / len(self.trades) * 100, 2) if self.trades else 0,
            profit_factor=round(profit_factor, 2),
            total_trades=len(self.trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_trade_pnl=round(sum(trade_pnls) / len(trade_pnls), 2) if trade_pnls else 0,
            best_trade_pnl=round(max(trade_pnls), 2) if trade_pnls else 0,
            worst_trade_pnl=round(min(trade_pnls), 2) if trade_pnls else 0,
            avg_trade_duration_hours=round(sum(t.duration_hours for t in self.trades) / len(self.trades), 2) if self.trades else 0,
            trades=[asdict(t) for t in self.trades],
            equity_curve=self.equity_curve,
            daily_returns=daily_returns,
            weekly_returns=weekly_returns,
            monthly_returns=monthly_returns,
            yearly_returns=yearly_returns,
            ohlc_data=self.ohlc_data,
            completed_at=datetime.now(timezone.utc).isoformat(),
            duration_seconds=round(duration, 2),
            risk_stats={
                **self.risk_manager.get_risk_status(),
                'exit_reason_counts': exit_reason_counts
            }
        )

    def _calculate_period_returns(self, freq: str) -> List[Dict[str, Any]]:
        """
        Calculate returns for a specific period frequency.

        Args:
            freq: 'D' (daily), 'W' (weekly), 'ME' (monthly), 'YE' (yearly)

        Returns:
            List of period returns with date, value, and return percentage
        """
        if not self.equity_curve:
            return []

        df = pd.DataFrame(self.equity_curve)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        resampled = df.resample(freq).last()
        resampled['return'] = resampled['value'].pct_change(fill_method=None) * 100

        # Format string based on frequency
        format_map = {
            'D': '%Y-%m-%d',
            'W': '%Y-W%V',
            'ME': '%Y-%m',
            'YE': '%Y'
        }
        date_format = format_map.get(freq, '%Y-%m-%d')

        result = []
        for date, row in resampled.iterrows():
            if pd.notna(row['value']):
                result.append({
                    'period': date.strftime(date_format),
                    'date': date.isoformat(),
                    'value': round(row['value'], 2),
                    'return_pct': round(row['return'], 2) if pd.notna(row['return']) else 0
                })
        return result

    def _store_ohlc_data(self, symbol: str, df: pd.DataFrame):
        """Store OHLC data for candlestick chart visualization"""
        for idx, row in df.iterrows():
            self.ohlc_data.append({
                'date': idx.isoformat(),
                'symbol': symbol,
                'open': round(float(row['Open']), 2),
                'high': round(float(row['High']), 2),
                'low': round(float(row['Low']), 2),
                'close': round(float(row['Close']), 2),
                'volume': int(row['Volume']) if 'Volume' in row else 0
            })




def update_database(results_dict: dict) -> bool:
    """
    Update the database with backtest results.

    This allows the backtest container to directly update the strategies table
    without requiring SSH or file fetching from remote hosts.
    """
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        logger.warning("DATABASE_URL not set - skipping database update")
        return False

    try:
        import psycopg2
        from psycopg2.extras import Json

        # Connect to database
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()

        # Update strategy with backtest results
        cur.execute("""
            UPDATE strategies
            SET backtest_results = %s,
                backtest_status = 'COMPLETED',
                backtest_completed_at = NOW()
            WHERE id = %s
        """, (Json(results_dict), int(STRATEGY_ID)))

        conn.commit()
        cur.close()
        conn.close()

        logger.info(f"✓ Database updated successfully for strategy {STRATEGY_ID}")
        return True

    except Exception as e:
        logger.error(f"Failed to update database: {e}", exc_info=True)
        return False


async def notify_completion(results: BacktestResults, success: bool = True):
    """
    Notify completion of backtest.

    Results are saved to the mounted volume (RESULTS_FILE) which the platform
    reads when checking backtest status. This is more reliable than WebSocket
    for Docker-to-Docker or Docker-to-host communication.
    """
    # Results are already saved to RESULTS_FILE by the main function.
    # This function exists for optional additional notification mechanisms.

    # Log completion status
    if success:
        logger.info(f"Backtest completed successfully: {results.total_return_pct}% return, {results.total_trades} trades")
    else:
        logger.warning("Backtest completed with errors")


async def main():
    """
    Main entry point for backtest.

    This runs completely isolated from live trading:
    - No real trades are executed
    - No signals are sent to brokers
    - Only historical data is used
    """
    logger.info("=" * 60)
    logger.info("Signal Synk Strategy Backtest (ISOLATED MODE)")
    logger.info("=" * 60)
    logger.info("NOTE: This is a simulation. No real trades will be executed.")

    try:
        # Initialize strategy
        strategy = TradingStrategy()

        # Call initialize if it exists (some strategies don't have it)
        if hasattr(strategy, 'initialize') and callable(strategy.initialize):
            await strategy.initialize()

        # Create and run backtest
        runner = BacktestRunner(strategy, BACKTEST_PERIOD)
        results = await runner.run_backtest()

        # Convert results to dict for storage
        results_dict = asdict(results)

        # Save results to file (for backup and local access)
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results_dict, f, indent=2, cls=NumpyJSONEncoder)

        logger.info("=" * 60)
        logger.info("BACKTEST COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Period: {results.period}")
        logger.info(f"Total Return: {results.total_return_pct}%")
        logger.info(f"Max Drawdown: {results.max_drawdown_pct}%")
        logger.info(f"Sharpe Ratio: {results.sharpe_ratio}")
        logger.info(f"Win Rate: {results.win_rate_pct}%")
        logger.info(f"Total Trades: {results.total_trades}")
        logger.info(f"Results saved to: {RESULTS_FILE}")

        # Update database directly (production - for remote Docker hosts)
        update_database(results_dict)

        # Notify platform via WebSocket
        await notify_completion(results, success=True)

        return results

    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        # Save error to results file
        error_result = {
            "error": str(e),
            "strategy_id": STRATEGY_ID,
            "period": BACKTEST_PERIOD,
            "completed_at": datetime.now(timezone.utc).isoformat()
        }
        with open(RESULTS_FILE, 'w') as f:
            json.dump(error_result, f, indent=2)
        raise


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())