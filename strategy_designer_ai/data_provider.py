#!/usr/bin/env python3
"""
Data Provider for Strategy Backtests

Provides historical market data from multiple sources:
1. AlphaVantage (primary) - For US stocks and forex
2. Yahoo Finance (fallback) - Free backup data source

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
ALPHAVANTAGE_API_KEY = os.getenv('ALPHAVANTAGE_API_KEY', 'PLACEHOLDER_ROTATE_ME')
ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY', '')
POLYGON_BASE_URL = "https://api.polygon.io"


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
            interval: 'daily', 'weekly', 'monthly', '1h', 'hourly'

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
        """
        cache_key = f"{symbol}_{interval}_{days}"

        if cache_key in self.cache:
            logger.debug(f"Using cached data for {symbol}")
            return self.cache[cache_key]

        # Use Polygon as PRIMARY data source for all timeframes
        if POLYGON_API_KEY:
            logger.info(f"Using Polygon (primary) for {symbol} [{interval}]")
            df = await self._fetch_polygon(symbol, interval, days)
            if df is not None and not df.empty:
                self.cache[cache_key] = df
                logger.info(f"Fetched {len(df)} bars for {symbol}")
                return df
            logger.warning(f"Polygon returned no data for {symbol}, falling back to free APIs")

        # Fallback: AlphaVantage for daily data
        df = await self._fetch_alphavantage(symbol, interval)

        if df is None or df.empty:
            # Fallback: Yahoo Finance
            logger.warning(f"AlphaVantage failed for {symbol}, trying Yahoo Finance")
            df = await self._fetch_yahoo(symbol, days)

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
        interval: str = '1h',
        days: int = 90
    ) -> Optional[pd.DataFrame]:
        """Fetch intraday data from Polygon.io (premium data source for hourly/minute data)"""
        if not POLYGON_API_KEY:
            logger.warning("Polygon API key not configured")
            return None

        try:
            import aiohttp

            # Map interval to Polygon timespan and multiplier
            timespan_map = {
                '1m': ('minute', 1),
                '5m': ('minute', 5),
                '15m': ('minute', 15),
                '30m': ('minute', 30),
                '1h': ('hour', 1),
                '1hour': ('hour', 1),
                'hourly': ('hour', 1),
                '4h': ('hour', 4),
                '1d': ('day', 1),
                '1day': ('day', 1),
                'daily': ('day', 1),
                '1w': ('week', 1),
                'weekly': ('week', 1),
            }

            timespan, multiplier = timespan_map.get(interval.lower(), ('hour', 1))

            # Calculate date range
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)

            url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"

            params = {'apiKey': POLYGON_API_KEY, 'adjusted': 'true', 'sort': 'asc', 'limit': 50000}

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        logger.error(f"Polygon API error for {symbol}: HTTP {response.status}")
                        return None

                    data = await response.json()

                    if data.get('status') != 'OK' or not data.get('results'):
                        logger.warning(f"No Polygon data for {symbol}")
                        return None

                    # Convert to DataFrame
                    df = pd.DataFrame(data['results'])
                    df['timestamp'] = pd.to_datetime(df['t'], unit='ms', utc=True)
                    df = df.set_index('timestamp')

                    # Rename columns to match expected format
                    df = df.rename(columns={
                        'o': 'Open',
                        'h': 'High',
                        'l': 'Low',
                        'c': 'Close',
                        'v': 'Volume'
                    })

                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    logger.info(f"Polygon: Fetched {len(df)} bars for {symbol}")
                    return df

        except Exception as e:
            logger.error(f"Polygon fetch error for {symbol}: {e}")
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

