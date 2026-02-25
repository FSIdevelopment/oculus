"""
Signal Synk Trading Strategy Template

Implement your trading logic here. This class is called by main.py
to analyze market conditions and generate trade signals.

Data is fetched from AlphaVantage Premium API.

To create your own strategy:
1. Copy this template folder to a new folder (e.g., 'my_strategy')
2. Modify config.json with your settings
3. Implement calculate_indicators() with your indicator logic
4. Implement check_entry_conditions() with your entry/exit logic
5. The analyze() method ties everything together

Author: Signal Synk Team
Copyright © 2026 Forefront Solutions Inc. All rights reserved.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pathlib import Path

import pandas as pd
import numpy as np
import requests

logger = logging.getLogger('Strategy')

# AlphaVantage API configuration (key passed via environment variable)
ALPHAVANTAGE_API_KEY = os.getenv('ALPHAVANTAGE_API_KEY', '')
ALPHAVANTAGE_BASE_URL = 'https://www.alphavantage.co/query'


class TradingStrategy:
    """
    Template trading strategy.

    Implement your trading logic by customizing:
    - calculate_indicators(): Add your technical indicator calculations
    - check_entry_conditions(): Define when to BUY or SELL
    - analyze(): Main loop that generates signals
    """

    def __init__(self):
        """Initialize the strategy with configuration."""
        self.config = self._load_config()
        self.positions: Dict[str, Dict] = {}  # symbol -> position info
        self.last_signals: Dict[str, str] = {}  # symbol -> last signal direction
        self.daily_pnl = 0.0
        self.signals_today = 0

        # Extract key config values
        self.symbols = self.config.get('trading', {}).get('symbols', ['AAPL'])
        self.default_quantity = self.config.get('trading', {}).get('default_quantity', 10)

        logger.info(f"Strategy initialized: {self.config.get('strategy_name', 'Unknown')}")
        logger.info(f"  Symbols: {self.symbols}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load strategy configuration from config.json."""
        config_path = Path(__file__).parent / 'config.json'
        try:
            with open(config_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config.json: {e}")
            return {}
    
    def get_interval(self) -> int:
        """
        Return the signal generation interval in seconds.

        This controls how often the strategy's analyze() method is called
        to check for entry/exit conditions (the strategy's "heartbeat").

        Note: This is separate from 'timeframe' which controls the data interval.
        For example, you might use 5m data but check for signals every 1m.

        Default: 1 minute (60 seconds)
        """
        trading_config = self.config.get('trading', {})

        # Use signal_generation_interval if set, otherwise fall back to timeframe
        interval_str = trading_config.get('signal_generation_interval') or trading_config.get('timeframe', '1m')

        # Parse timeframe string to seconds
        timeframe_map = {
            '30s': 30,
            '1m': 60,
            '2m': 120,
            '5m': 300,
            '10m': 600,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '4h': 14400,
            '1D': 86400
        }
        return timeframe_map.get(interval_str, 60)

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
        trading_config = self.config.get('trading', {})
        timeframe = trading_config.get('timeframe', '1d')
        loop_map = {
            '1d': 1800, '1D': 1800,
            '4h': 900,
            '1h': 300,
            '30m': 300, '30min': 300,
            '15m': 900, '15min': 900,
            '10m': 600, '10min': 600,
            '5m': 300, '5min': 300,
            '2m': 120, '2min': 120,
            '1m': 60, '1min': 60,
        }
        return loop_map.get(timeframe, 1800)

    async def get_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetch market data for a symbol using AlphaVantage Premium API.

        Returns a pandas DataFrame with OHLCV data, or None on error.
        """
        try:
            if not ALPHAVANTAGE_API_KEY:
                logger.error("ALPHAVANTAGE_API_KEY environment variable not set!")
                return None

            # Get data source config
            data_config = self.config.get('data_source', {})
            interval = data_config.get('interval', '5min')
            outputsize = data_config.get('outputsize', 'compact')

            # Build API request for intraday data
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': symbol,
                'interval': interval,
                'outputsize': outputsize,
                'apikey': ALPHAVANTAGE_API_KEY,
                'datatype': 'json'
            }

            logger.debug(f"Fetching {symbol} data from AlphaVantage ({interval})")
            response = requests.get(ALPHAVANTAGE_BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Check for API errors
            if 'Error Message' in data:
                logger.error(f"AlphaVantage error for {symbol}: {data['Error Message']}")
                return None

            if 'Note' in data:
                logger.warning(f"AlphaVantage rate limit: {data['Note']}")
                return None

            # Parse the time series data
            time_series_key = f'Time Series ({interval})'
            if time_series_key not in data:
                # Try daily data as fallback
                logger.debug(f"No intraday data for {symbol}, trying daily...")
                return await self._get_daily_data(symbol)

            time_series = data[time_series_key]

            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            # Rename columns to standard format
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df.astype({
                'Open': float,
                'High': float,
                'Low': float,
                'Close': float,
                'Volume': int
            })

            if df.empty:
                logger.warning(f"No data available for {symbol}")
                return None

            logger.debug(f"Fetched {len(df)} bars for {symbol} from AlphaVantage")
            return df

        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching data for {symbol}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching data for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    async def _get_daily_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fallback to daily data from AlphaVantage."""
        try:
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'outputsize': 'compact',
                'apikey': ALPHAVANTAGE_API_KEY,
                'datatype': 'json'
            }

            response = requests.get(ALPHAVANTAGE_BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if 'Time Series (Daily)' not in data:
                logger.warning(f"No daily data available for {symbol}")
                return None

            time_series = data['Time Series (Daily)']

            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            # Rename columns
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df.astype({
                'Open': float,
                'High': float,
                'Low': float,
                'Close': float,
                'Volume': int
            })

            logger.debug(f"Fetched {len(df)} daily bars for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching daily data for {symbol}: {e}")
            return None

    async def get_realtime_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time quote from AlphaVantage (Premium feature)."""
        try:
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': ALPHAVANTAGE_API_KEY,
                'datatype': 'json'
            }

            response = requests.get(ALPHAVANTAGE_BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if 'Global Quote' not in data or not data['Global Quote']:
                return None

            quote = data['Global Quote']
            return {
                'symbol': quote.get('01. symbol', symbol),
                'price': float(quote.get('05. price', 0)),
                'open': float(quote.get('02. open', 0)),
                'high': float(quote.get('03. high', 0)),
                'low': float(quote.get('04. low', 0)),
                'volume': int(quote.get('06. volume', 0)),
                'previous_close': float(quote.get('08. previous close', 0)),
                'change': float(quote.get('09. change', 0)),
                'change_percent': quote.get('10. change percent', '0%')
            }

        except Exception as e:
            logger.debug(f"Could not get real-time quote for {symbol}: {e}")
            return None
    
    def calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return prices.rolling(window=period).mean()

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    async def calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators from price data.

        Override this with your indicator calculations.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dictionary of indicator values
        """
        if df is None or len(df) < 50:  # Need enough data for indicators
            return {}

        close = df['Close']

        # Example: Calculate common indicators
        config = self.config.get('indicators', {})
        sma_fast_period = config.get('sma_fast', 20)
        sma_slow_period = config.get('sma_slow', 50)
        rsi_period = config.get('rsi_period', 14)

        sma_fast = self.calculate_sma(close, sma_fast_period)
        sma_slow = self.calculate_sma(close, sma_slow_period)
        rsi = self.calculate_rsi(close, rsi_period)

        return {
            'current_price': close.iloc[-1],
            'sma_fast': sma_fast.iloc[-1],
            'sma_slow': sma_slow.iloc[-1],
            'rsi': rsi.iloc[-1],
            'trend': 'bullish' if sma_fast.iloc[-1] > sma_slow.iloc[-1] else 'bearish'
        }

    async def check_entry_conditions(
        self,
        symbol: str,
        indicators: Dict,
        current_date=None
    ) -> Optional[str]:
        """
        Check if entry conditions are met.

        Override this with your entry/exit logic.

        Args:
            symbol: The ticker symbol
            indicators: Dictionary of calculated indicators

        Returns:
            'BUY', 'SELL', or None
        """
        if not indicators:
            return None

        config = self.config.get('indicators', {})
        rsi_oversold = config.get('rsi_oversold', 30)
        rsi_overbought = config.get('rsi_overbought', 70)

        last_signal = self.last_signals.get(symbol)
        rsi = indicators.get('rsi', 50)

        # Example: RSI-based strategy
        # Buy when RSI is oversold (below 30)
        if rsi < rsi_oversold and last_signal != 'BUY':
            logger.info(f"🟢 {symbol}: RSI oversold ({rsi:.1f})")
            return 'BUY'

        # Sell when RSI is overbought (above 70)
        if rsi > rsi_overbought and last_signal != 'SELL':
            logger.info(f"🔴 {symbol}: RSI overbought ({rsi:.1f})")
            return 'SELL'

        return None

    async def analyze(self) -> List[Dict[str, Any]]:
        """
        Main analysis method - called periodically by main.py.

        Returns a list of trade signals to execute.
        Each signal should have: ticker, action, quantity, order_type
        """
        signals = []

        for symbol in self.symbols:
            try:
                # Get market data
                df = await self.get_market_data(symbol)
                if df is None:
                    continue

                # Calculate indicators
                indicators = await self.calculate_indicators(df)
                if not indicators:
                    continue

                # Log current state
                logger.debug(
                    f"{symbol}: Price=${indicators.get('current_price', 0):.2f}, "
                    f"RSI={indicators.get('rsi', 0):.1f}, "
                    f"Trend={indicators.get('trend', 'unknown')}"
                )

                # Check for signals
                action = await self.check_entry_conditions(symbol, indicators)

                if action:
                    signal = {
                        'ticker': symbol,
                        'action': action,
                        'quantity': self.default_quantity,
                        'order_type': 'MARKET',
                        'price': indicators.get('current_price'),
                        'indicators': {
                            'rsi': round(indicators.get('rsi', 0), 2),
                            'sma_fast': round(indicators.get('sma_fast', 0), 2),
                            'sma_slow': round(indicators.get('sma_slow', 0), 2),
                            'trend': indicators.get('trend', 'unknown')
                        }
                    }
                    signals.append(signal)

                    # Track last signal to avoid duplicates
                    self.last_signals[symbol] = action
                    self.signals_today += 1

                    logger.info(
                        f"📊 SIGNAL: {action} {self.default_quantity} {symbol} "
                        f"@ ${indicators.get('current_price', 0):.2f}"
                    )

            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")

        return signals

    def get_status(self) -> Dict[str, Any]:
        """Get current strategy status for monitoring."""
        return {
            'strategy_name': self.config.get('strategy_name', 'Unknown'),
            'symbols': self.symbols,
            'signals_today': self.signals_today,
            'last_signals': self.last_signals,
            'positions': self.positions,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

