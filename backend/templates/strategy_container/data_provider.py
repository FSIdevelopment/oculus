#!/usr/bin/env python3
"""
Data Provider for Strategy Backtests

Provides historical market data from multiple sources:
1. Polygon.io (primary) - Professional-grade market data
2. AlphaVantage - For US stocks and forex
3. Alpaca - Commission-free trading platform data
4. Yahoo Finance (fallback) - Free backup data source

This module is ISOLATED and does NOT execute any real trades.
It only fetches historical data for backtesting purposes.

Author: Signal Synk Team
Copyright © 2026 Forefront Solutions Inc. All rights reserved.
"""

import os
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List
import pandas as pd
import numpy as np

logger = logging.getLogger('DataProvider')

# API Keys and Configuration
ALPHAVANTAGE_API_KEY = os.getenv('ALPHAVANTAGE_API_KEY', 'PHCRS4IARZHCWH2P')
ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY', '')
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
ALPACA_API_SECRET = os.getenv('ALPACA_API_SECRET', '')


class DataProvider:
    """
    Multi-source data provider for backtesting.
    Completely isolated from live trading - READ ONLY data access.
    """

    def __init__(self):
        self.cache: Dict[str, pd.DataFrame] = {}
        self._rate_limit_delay = 12.5  # AlphaVantage: 5 calls/min = 12 sec between calls
        self._last_api_call = 0

    async def get_historical_data(
        self,
        symbol: str,
        days: int = 365,
        interval: str = 'daily'
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data for a symbol.

        Args:
            symbol: Stock ticker (e.g., 'AAPL', 'MSFT')
            days: Number of days of historical data
            interval: 'daily', 'weekly', or 'monthly'

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
        """
        cache_key = f"{symbol}_{interval}_{days}"

        if cache_key in self.cache:
            logger.debug(f"Using cached data for {symbol}")
            return self.cache[cache_key]

        # Try providers in order: Polygon -> AlphaVantage -> Alpaca -> Yahoo
        df = None

        # 1. Try Polygon first (if API key is configured)
        if POLYGON_API_KEY:
            df = await self._fetch_polygon(symbol, days, interval)
            if df is not None and not df.empty:
                logger.info(f"Using Polygon data for {symbol}")

        # 2. Try AlphaVantage (if Polygon failed and API key is configured)
        if (df is None or df.empty) and ALPHAVANTAGE_API_KEY:
            logger.warning(f"Polygon failed for {symbol}, trying AlphaVantage")
            df = await self._fetch_alphavantage(symbol, interval)
            if df is not None and not df.empty:
                logger.info(f"Using AlphaVantage data for {symbol}")

        # 3. Try Alpaca (if previous providers failed and API key is configured)
        if (df is None or df.empty) and ALPACA_API_KEY and ALPACA_API_SECRET:
            logger.warning(f"AlphaVantage failed for {symbol}, trying Alpaca")
            df = await self._fetch_alpaca(symbol, days, interval)
            if df is not None and not df.empty:
                logger.info(f"Using Alpaca data for {symbol}")

        # 4. Fallback to Yahoo Finance (always available, no API key needed)
        if df is None or df.empty:
            logger.warning(f"All paid providers failed for {symbol}, trying Yahoo Finance")
            df = await self._fetch_yahoo(symbol, days)
            if df is not None and not df.empty:
                logger.info(f"Using Yahoo Finance data for {symbol}")

        if df is not None and not df.empty:
            # Filter to requested period
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            df = df[df.index >= cutoff_date]
            self.cache[cache_key] = df
            logger.info(f"Fetched {len(df)} bars for {symbol}")

        return df

    async def _fetch_alphavantage(
        self,
        symbol: str,
        interval: str = 'daily'
    ) -> Optional[pd.DataFrame]:
        """Fetch data from AlphaVantage API"""
        import aiohttp

        # Rate limiting
        now = asyncio.get_event_loop().time()
        time_since_last = now - self._last_api_call
        if time_since_last < self._rate_limit_delay:
            await asyncio.sleep(self._rate_limit_delay - time_since_last)

        # Map interval to AlphaVantage function and params
        intraday_intervals = {
            '1min': '1min', '5min': '5min', '15min': '15min',
            '30min': '30min', '60min': '60min',
            '1m': '1min', '5m': '5min', '15m': '15min',
            '30m': '30min', '1h': '60min', '4h': '60min'  # 4h uses hourly data
        }

        if interval in intraday_intervals:
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': symbol,
                'interval': intraday_intervals[interval],
                'apikey': ALPHAVANTAGE_API_KEY,
                'outputsize': 'full',
                'datatype': 'json',
                'extended_hours': 'false'
            }
        else:
            function_map = {
                'daily': 'TIME_SERIES_DAILY_ADJUSTED',
                '1D': 'TIME_SERIES_DAILY_ADJUSTED',
                'weekly': 'TIME_SERIES_WEEKLY_ADJUSTED',
                '1W': 'TIME_SERIES_WEEKLY_ADJUSTED',
                'monthly': 'TIME_SERIES_MONTHLY_ADJUSTED'
            }
            params = {
                'function': function_map.get(interval, 'TIME_SERIES_DAILY_ADJUSTED'),
                'symbol': symbol,
                'apikey': ALPHAVANTAGE_API_KEY,
                'outputsize': 'full',
                'datatype': 'json'
            }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(ALPHAVANTAGE_BASE_URL, params=params) as resp:
                    self._last_api_call = asyncio.get_event_loop().time()

                    if resp.status != 200:
                        logger.error(f"AlphaVantage API error: {resp.status}")
                        return None

                    data = await resp.json()

                    # Check for API errors
                    if 'Error Message' in data:
                        logger.error(f"AlphaVantage error: {data['Error Message']}")
                        return None

                    if 'Note' in data:  # Rate limit warning
                        logger.warning(f"AlphaVantage rate limit: {data['Note']}")
                        return None

                    # Parse the time series data
                    return self._parse_alphavantage_response(data, interval)

        except Exception as e:
            logger.error(f"AlphaVantage request failed: {e}")
            return None

    def _parse_alphavantage_response(
        self,
        data: dict,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """Parse AlphaVantage JSON response to DataFrame"""
        # Find the time series key
        ts_keys = [k for k in data.keys() if 'Time Series' in k]
        if not ts_keys:
            logger.error("No time series data in AlphaVantage response")
            return None

        ts_data = data[ts_keys[0]]

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(ts_data, orient='index')
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()

        # Rename columns (AlphaVantage uses numbered prefixes)
        column_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'open' in col_lower:
                column_map[col] = 'Open'
            elif 'high' in col_lower:
                column_map[col] = 'High'
            elif 'low' in col_lower:
                column_map[col] = 'Low'
            elif 'close' in col_lower and 'adjusted' not in col_lower:
                column_map[col] = 'Close'
            elif 'adjusted' in col_lower and 'close' in col_lower:
                column_map[col] = 'Adj Close'
            elif 'volume' in col_lower:
                column_map[col] = 'Volume'

        df = df.rename(columns=column_map)

        # Convert to numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Use adjusted close as Close if available
        if 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']

        return df[['Open', 'High', 'Low', 'Close', 'Volume']]

    async def _fetch_polygon(
        self,
        symbol: str,
        days: int = 365,
        interval: str = 'daily'
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Polygon.io API"""
        if not POLYGON_API_KEY:
            logger.debug("Polygon API key not configured")
            return None

        try:
            from polygon import RESTClient
            from datetime import date

            # Map interval to Polygon timespan
            timespan_map = {
                'daily': 'day',
                '1D': 'day',
                'weekly': 'week',
                '1W': 'week',
                'monthly': 'month',
                '1M': 'month',
                '1min': 'minute',
                '5min': 'minute',
                '15min': 'minute',
                '30min': 'minute',
                '60min': 'hour',
                '1m': 'minute',
                '5m': 'minute',
                '15m': 'minute',
                '30m': 'minute',
                '1h': 'hour',
                '4h': 'hour'
            }

            # Map interval to multiplier
            multiplier_map = {
                '1min': 1, '5min': 5, '15min': 15, '30min': 30, '60min': 60,
                '1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 1, '4h': 4,
                'daily': 1, '1D': 1, 'weekly': 1, '1W': 1, 'monthly': 1, '1M': 1
            }

            timespan = timespan_map.get(interval, 'day')
            multiplier = multiplier_map.get(interval, 1)

            # Calculate date range
            end_date = date.today()
            start_date = end_date - timedelta(days=days + 30)  # Extra buffer

            # Initialize Polygon client
            client = RESTClient(api_key=POLYGON_API_KEY)

            # Fetch aggregates (bars)
            aggs = []
            for agg in client.get_aggs(
                ticker=symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_=start_date.isoformat(),
                to=end_date.isoformat(),
                limit=50000
            ):
                aggs.append(agg)

            if not aggs:
                logger.warning(f"No Polygon data for {symbol}")
                return None

            # Convert to DataFrame
            data = []
            for agg in aggs:
                data.append({
                    'timestamp': pd.Timestamp(agg.timestamp, unit='ms', tz='UTC'),
                    'Open': agg.open,
                    'High': agg.high,
                    'Low': agg.low,
                    'Close': agg.close,
                    'Volume': agg.volume
                })

            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()

            logger.info(f"Fetched {len(df)} bars from Polygon for {symbol}")
            return df

        except ImportError:
            logger.error("polygon-api-client not installed. Run: pip install polygon-api-client")
            return None
        except Exception as e:
            logger.error(f"Polygon request failed for {symbol}: {e}")
            return None

    async def _fetch_alpaca(
        self,
        symbol: str,
        days: int = 365,
        interval: str = 'daily'
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Alpaca API"""
        if not ALPACA_API_KEY or not ALPACA_API_SECRET:
            logger.debug("Alpaca API credentials not configured")
            return None

        try:
            from alpaca.data import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame

            # Map interval to Alpaca TimeFrame
            timeframe_map = {
                '1min': TimeFrame.Minute,
                '5min': TimeFrame(5, 'Min'),
                '15min': TimeFrame(15, 'Min'),
                '30min': TimeFrame(30, 'Min'),
                '60min': TimeFrame.Hour,
                '1m': TimeFrame.Minute,
                '5m': TimeFrame(5, 'Min'),
                '15m': TimeFrame(15, 'Min'),
                '30m': TimeFrame(30, 'Min'),
                '1h': TimeFrame.Hour,
                '4h': TimeFrame(4, 'Hour'),
                'daily': TimeFrame.Day,
                '1D': TimeFrame.Day,
                'weekly': TimeFrame.Week,
                '1W': TimeFrame.Week,
                'monthly': TimeFrame.Month,
                '1M': TimeFrame.Month
            }

            timeframe = timeframe_map.get(interval, TimeFrame.Day)

            # Calculate date range
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days + 30)  # Extra buffer

            # Initialize Alpaca client (no auth needed for market data)
            client = StockHistoricalDataClient(
                api_key=ALPACA_API_KEY,
                secret_key=ALPACA_API_SECRET
            )

            # Create request
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start_date,
                end=end_date
            )

            # Fetch bars
            bars = client.get_stock_bars(request)

            if not bars or symbol not in bars.data:
                logger.warning(f"No Alpaca data for {symbol}")
                return None

            # Convert to DataFrame
            df = bars.df

            if df.empty:
                logger.warning(f"Empty Alpaca data for {symbol}")
                return None

            # Reset index to get timestamp as column
            df = df.reset_index()

            # Rename columns to standard format
            column_map = {
                'timestamp': 'timestamp',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            df = df.rename(columns=column_map)

            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()

            # Ensure timezone-aware
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')

            # Select only needed columns
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

            logger.info(f"Fetched {len(df)} bars from Alpaca for {symbol}")
            return df

        except ImportError:
            logger.error("alpaca-py not installed. Run: pip install alpaca-py")
            return None
        except Exception as e:
            logger.error(f"Alpaca request failed for {symbol}: {e}")
            return None

    async def _fetch_yahoo(
        self,
        symbol: str,
        days: int = 365
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance as fallback"""
        try:
            import yfinance as yf

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 30)  # Extra buffer

            # Fetch data (runs synchronously, wrapped for async)
            ticker = yf.Ticker(symbol)
            df = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ticker.history(start=start_date, end=end_date, auto_adjust=True)
            )

            if df is None or df.empty:
                logger.warning(f"No Yahoo Finance data for {symbol}")
                return None

            # Ensure timezone-aware index
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')

            # Standardize column names
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })

            # Select only needed columns
            columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            available = [c for c in columns if c in df.columns]

            return df[available]

        except ImportError:
            logger.error("yfinance not installed. Run: pip install yfinance")
            return None
        except Exception as e:
            logger.error(f"Yahoo Finance request failed: {e}")
            return None

    async def get_multiple_symbols(
        self,
        symbols: List[str],
        days: int = 365,
        interval: str = 'daily',
        parallel: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.

        Args:
            symbols: List of stock tickers
            days: Number of days of historical data
            interval: Data interval (daily, 5m, etc.)
            parallel: If True, fetch all symbols concurrently (faster but may hit rate limits)
        """
        results = {}

        if parallel and len(symbols) > 1:
            # Fetch all symbols in parallel using asyncio.gather
            logger.info(f"Fetching {len(symbols)} symbols in parallel...")

            async def fetch_with_symbol(sym: str):
                """Wrapper to return (symbol, dataframe) tuple"""
                df = await self.get_historical_data(sym, days, interval)
                return sym, df

            # Create tasks for all symbols
            tasks = [fetch_with_symbol(symbol) for symbol in symbols]

            # Execute all tasks concurrently
            fetched = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for result in fetched:
                if isinstance(result, Exception):
                    logger.error(f"Failed to fetch symbol: {result}")
                    continue
                symbol, df = result
                if df is not None and not df.empty:
                    results[symbol] = df

            logger.info(f"Parallel fetch complete: {len(results)}/{len(symbols)} symbols successful")
        else:
            # Sequential fetching (original behavior)
            for symbol in symbols:
                df = await self.get_historical_data(symbol, days, interval)
                if df is not None and not df.empty:
                    results[symbol] = df

        return results

    def clear_cache(self):
        """Clear the data cache"""
        self.cache.clear()
        logger.info("Data cache cleared")

