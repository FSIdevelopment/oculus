#!/usr/bin/env python3
"""
AI Strategy Designer for Semiconductor Momentum Strategy - Enhanced Version

This AI model analyzes historical data to design the optimal trading strategy:
1. Calculates theoretical maximum returns (perfect timing)
2. Computes ALL 64 technical indicators from comprehensive knowledge base
3. Identifies key technical patterns at optimal entry/exit points
4. Uses pattern analysis to find the best indicators and thresholds
5. Generates an optimized strategy configuration targeting 300%+ returns

Author: SignalSynk AI
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# Add semiconductor_momentum directory for imports (it has DataProvider with Polygon API)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'strategies', 'semiconductor_momentum'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_provider import DataProvider
from technical_indicators_index import INDICATORS

@dataclass
class TradeOpportunity:
    """Represents an ideal trade opportunity."""
    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: datetime
    exit_price: float
    return_pct: float
    holding_days: int

@dataclass
class StockAnalysis:
    """Macro analysis of a stock over the period."""
    symbol: str
    start_price: float
    end_price: float
    high_price: float
    low_price: float
    buy_hold_return: float
    max_possible_return: float
    optimal_trades: List[TradeOpportunity]


class ComprehensiveIndicatorCalculator:
    """Calculate ALL technical indicators from knowledge base."""

    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all 64 technical indicators."""
        df = df.copy()

        # =====================================================================
        # MOMENTUM INDICATORS
        # =====================================================================

        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Stochastic Oscillator
        low_14 = df['Low'].rolling(14).min()
        high_14 = df['High'].rolling(14).max()
        df['STOCH_K'] = ((df['Close'] - low_14) / (high_14 - low_14)) * 100
        df['STOCH_D'] = df['STOCH_K'].rolling(3).mean()

        # Stochastic Momentum Index (SMI)
        mid_range = (high_14 + low_14) / 2
        diff = df['Close'] - mid_range
        range_hl = high_14 - low_14
        df['SMI'] = (diff.ewm(span=3).mean().ewm(span=3).mean() /
                     (range_hl.ewm(span=3).mean().ewm(span=3).mean() / 2)) * 100

        # MACD
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

        # Momentum
        df['MOMENTUM'] = df['Close'] - df['Close'].shift(10)
        df['MOMENTUM_PCT'] = df['Close'].pct_change(10) * 100

        # Rate of Change (ROC)
        df['ROC'] = ((df['Close'] - df['Close'].shift(12)) / df['Close'].shift(12)) * 100

        # Chande Momentum Oscillator (CMO)
        up = delta.where(delta > 0, 0).rolling(14).sum()
        down = (-delta.where(delta < 0, 0)).rolling(14).sum()
        df['CMO'] = ((up - down) / (up + down)) * 100

        # Force Index
        df['FORCE_INDEX'] = (df['Close'] - df['Close'].shift(1)) * df['Volume']
        df['FORCE_INDEX_13'] = df['FORCE_INDEX'].ewm(span=13).mean()

        # Elder Ray (Bull/Bear Power)
        ema13 = df['Close'].ewm(span=13).mean()
        df['BULL_POWER'] = df['High'] - ema13
        df['BEAR_POWER'] = df['Low'] - ema13

        # Repulse Indicator
        df['REPULSE'] = (((df['High'] - df['Open']) * 3 + (df['Close'] - df['Low']) * 3 -
                         (df['Open'] - df['Low']) * 3 - (df['High'] - df['Close']) * 3) /
                        (df['High'] - df['Low'])).ewm(span=5).mean()

        # =====================================================================
        # TREND INDICATORS
        # =====================================================================

        # Multiple EMAs
        for period in [8, 13, 21, 50, 100, 200]:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()

        # Multiple SMAs
        for period in [20, 50, 100, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(period).mean()

        # DEMA (Double Exponential Moving Average)
        ema_20 = df['Close'].ewm(span=20).mean()
        df['DEMA_20'] = 2 * ema_20 - ema_20.ewm(span=20).mean()

        # TEMA (Triple Exponential Moving Average)
        ema1 = df['Close'].ewm(span=20).mean()
        ema2 = ema1.ewm(span=20).mean()
        ema3 = ema2.ewm(span=20).mean()
        df['TEMA_20'] = 3 * ema1 - 3 * ema2 + ema3

        # ADX and DMI
        tr = pd.concat([
            df['High'] - df['Low'],
            abs(df['High'] - df['Close'].shift()),
            abs(df['Low'] - df['Close'].shift())
        ], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()

        up_move = df['High'] - df['High'].shift()
        down_move = df['Low'].shift() - df['Low']
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

        df['PLUS_DI'] = 100 * (plus_dm.rolling(14).mean() / df['ATR'])
        df['MINUS_DI'] = 100 * (minus_dm.rolling(14).mean() / df['ATR'])
        dx = abs(df['PLUS_DI'] - df['MINUS_DI']) / (df['PLUS_DI'] + df['MINUS_DI']) * 100
        df['ADX'] = dx.rolling(14).mean()

        # Aroon - OPTIMIZED: vectorized calculation using rolling max/min indices
        # Calculate days since highest high and lowest low in the period
        period = 25
        high_roll = df['High'].rolling(period)
        low_roll = df['Low'].rolling(period)
        # Use idxmax/idxmin approach with position calculation
        df['AROON_UP'] = high_roll.apply(lambda x: (x.argmax() / (len(x) - 1)) * 100 if len(x) > 1 else 50, raw=True)
        df['AROON_DOWN'] = low_roll.apply(lambda x: (x.argmin() / (len(x) - 1)) * 100 if len(x) > 1 else 50, raw=True)
        df['AROON_OSC'] = df['AROON_UP'] - df['AROON_DOWN']

        # Parabolic SAR (simplified)
        df['PSAR'] = df['Close'].ewm(span=2, adjust=False).mean()  # Simplified

        # Supertrend (simplified using ATR)
        hl2 = (df['High'] + df['Low']) / 2
        df['SUPERTREND_UPPER'] = hl2 + (3 * df['ATR'])
        df['SUPERTREND_LOWER'] = hl2 - (3 * df['ATR'])

        # Ichimoku Cloud
        df['TENKAN'] = (df['High'].rolling(9).max() + df['Low'].rolling(9).min()) / 2
        df['KIJUN'] = (df['High'].rolling(26).max() + df['Low'].rolling(26).min()) / 2
        df['SENKOU_A'] = ((df['TENKAN'] + df['KIJUN']) / 2).shift(26)
        df['SENKOU_B'] = ((df['High'].rolling(52).max() + df['Low'].rolling(52).min()) / 2).shift(26)
        df['CHIKOU'] = df['Close'].shift(-26)

        # Keltner Channels
        df['KELTNER_MID'] = df['Close'].ewm(span=20).mean()
        df['KELTNER_UPPER'] = df['KELTNER_MID'] + 2 * df['ATR']
        df['KELTNER_LOWER'] = df['KELTNER_MID'] - 2 * df['ATR']

        # Donchian Channels
        df['DONCHIAN_UPPER'] = df['High'].rolling(20).max()
        df['DONCHIAN_LOWER'] = df['Low'].rolling(20).min()
        df['DONCHIAN_MID'] = (df['DONCHIAN_UPPER'] + df['DONCHIAN_LOWER']) / 2

        # Linear Regression - OPTIMIZED: using raw=True for faster numpy operations
        def linreg_value(x):
            if len(x) < 2:
                return x[-1] if len(x) > 0 else 0
            coeffs = np.polyfit(range(len(x)), x, 1)
            return coeffs[0] * (len(x) - 1) + coeffs[1]

        def linreg_slope(x):
            if len(x) < 2:
                return 0
            return np.polyfit(range(len(x)), x, 1)[0]

        df['LINREG'] = df['Close'].rolling(14).apply(linreg_value, raw=True)
        df['LINREG_SLOPE'] = df['Close'].rolling(14).apply(linreg_slope, raw=True)

        # Pivot Points (using previous day)
        df['PIVOT'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
        df['PIVOT_R1'] = 2 * df['PIVOT'] - df['Low'].shift(1)
        df['PIVOT_S1'] = 2 * df['PIVOT'] - df['High'].shift(1)

        # =====================================================================
        # VOLATILITY INDICATORS
        # =====================================================================

        # Bollinger Bands
        df['BB_MID'] = df['Close'].rolling(20).mean()
        df['BB_STD'] = df['Close'].rolling(20).std()
        df['BB_UPPER'] = df['BB_MID'] + 2 * df['BB_STD']
        df['BB_LOWER'] = df['BB_MID'] - 2 * df['BB_STD']
        df['BB_WIDTH'] = (df['BB_UPPER'] - df['BB_LOWER']) / df['BB_MID'] * 100
        df['BB_PCT_B'] = (df['Close'] - df['BB_LOWER']) / (df['BB_UPPER'] - df['BB_LOWER'])

        # ATR Percentage
        df['ATR_PCT'] = df['ATR'] / df['Close'] * 100

        # Historic Volatility
        log_returns = np.log(df['Close'] / df['Close'].shift(1))
        df['HIST_VOL'] = log_returns.rolling(20).std() * np.sqrt(252) * 100

        # Chaikin Volatility
        hl_ema = (df['High'] - df['Low']).ewm(span=10).mean()
        df['CHAIKIN_VOL'] = ((hl_ema - hl_ema.shift(10)) / hl_ema.shift(10)) * 100

        # Mass Index
        hl = df['High'] - df['Low']
        ema1_hl = hl.ewm(span=9).mean()
        ema2_hl = ema1_hl.ewm(span=9).mean()
        df['MASS_INDEX'] = (ema1_hl / ema2_hl).rolling(25).sum()

        # =====================================================================
        # VOLUME INDICATORS
        # =====================================================================

        # On Balance Volume (OBV)
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['OBV'] = obv
        df['OBV_EMA'] = pd.Series(obv).ewm(span=20).mean().values

        # Accumulation/Distribution
        clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        clv = clv.fillna(0)
        df['AD'] = (clv * df['Volume']).cumsum()

        # Chaikin Money Flow
        mfv = clv * df['Volume']
        df['CMF'] = mfv.rolling(21).sum() / df['Volume'].rolling(21).sum()

        # Chaikin Oscillator
        df['CHAIKIN_OSC'] = df['AD'].ewm(span=3).mean() - df['AD'].ewm(span=10).mean()

        # Money Flow Index
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        mf = tp * df['Volume']
        pos_mf = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
        neg_mf = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
        df['MFI'] = 100 - (100 / (1 + pos_mf / neg_mf))

        # Ease of Movement
        distance = ((df['High'] + df['Low']) / 2) - ((df['High'].shift(1) + df['Low'].shift(1)) / 2)
        box_ratio = (df['Volume'] / 1e6) / (df['High'] - df['Low'])
        df['EMV'] = (distance / box_ratio).rolling(14).mean()

        # Price Volume Trend
        pvt = [0]
        for i in range(1, len(df)):
            pct_change = (df['Close'].iloc[i] - df['Close'].iloc[i-1]) / df['Close'].iloc[i-1]
            pvt.append(pvt[-1] + df['Volume'].iloc[i] * pct_change)
        df['PVT'] = pvt

        # Volume Ratio
        df['VOLUME_RATIO'] = df['Volume'] / df['Volume'].rolling(20).mean()

        # =====================================================================
        # OSCILLATOR INDICATORS
        # =====================================================================

        # CCI (Commodity Channel Index) - OPTIMIZED: using std approximation instead of slow rolling apply
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        tp_sma = tp.rolling(20).mean()
        # Mean Absolute Deviation ≈ 0.7979 * std for normal distribution (much faster than rolling apply)
        mean_dev = tp.rolling(20).std() * 0.7979
        df['CCI'] = (tp - tp_sma) / (0.015 * mean_dev + 1e-10)

        # Williams %R
        df['WILLIAMS_R'] = ((high_14 - df['Close']) / (high_14 - low_14)) * -100

        # Detrended Price Oscillator
        shift = 11  # For 20-period
        df['DPO'] = df['Close'] - df['Close'].rolling(20).mean().shift(shift)

        # TRIX
        ema1_trix = df['Close'].ewm(span=15).mean()
        ema2_trix = ema1_trix.ewm(span=15).mean()
        ema3_trix = ema2_trix.ewm(span=15).mean()
        df['TRIX'] = (ema3_trix - ema3_trix.shift(1)) / ema3_trix.shift(1) * 100
        df['TRIX_SIGNAL'] = df['TRIX'].ewm(span=9).mean()

        # Awesome Oscillator
        df['AO'] = (df['High'] + df['Low']).rolling(5).mean() / 2 - (df['High'] + df['Low']).rolling(34).mean() / 2

        # =====================================================================
        # DERIVED/COMPOSITE SIGNALS
        # =====================================================================

        # Price vs EMAs
        df['PRICE_VS_EMA50'] = (df['Close'] - df['EMA_50']) / df['EMA_50'] * 100
        df['PRICE_VS_EMA200'] = (df['Close'] - df['EMA_200']) / df['EMA_200'] * 100

        # EMA Crossovers
        df['EMA8_VS_EMA21'] = (df['EMA_8'] - df['EMA_21']) / df['EMA_21'] * 100
        df['EMA21_VS_EMA50'] = (df['EMA_21'] - df['EMA_50']) / df['EMA_50'] * 100

        # Trend Strength (composite)
        df['TREND_STRENGTH'] = (df['ADX'] + abs(df['AROON_OSC']) / 2) / 2

        # Volume Confirmation
        df['VOLUME_CONFIRM'] = (df['CMF'] > 0) & (df['OBV'] > df['OBV_EMA'])

        return df
    

class AIStrategyDesigner:
    """AI model that designs optimal trading strategies."""
    
    def __init__(self, symbols: List[str], period_years: int = 2):
        self.symbols = symbols
        self.period_years = period_years
        self.data_provider = DataProvider()
        self.stock_data: Dict[str, pd.DataFrame] = {}
        self.analyses: Dict[str, StockAnalysis] = {}
        
    async def fetch_all_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for all symbols."""
        print(f"\n{'='*60}")
        print("STEP 1: FETCHING HISTORICAL DATA")
        print(f"{'='*60}")

        days = self.period_years * 365
        self.stock_data = await self.data_provider.get_multiple_symbols(
            self.symbols, days=days
        )

        for symbol, df in self.stock_data.items():
            if df is not None and len(df) > 0:
                print(f"  {symbol}: {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")

        return self.stock_data
    
    def analyze_macro_view(self) -> Dict[str, StockAnalysis]:
        """Analyze each stock's macro performance - what SHOULD the returns be."""
        print(f"\n{'='*60}")
        print("STEP 2: MACRO ANALYSIS - IDEAL RETURNS")
        print(f"{'='*60}")
        
        for symbol, df in self.stock_data.items():
            if df is None or len(df) < 50:
                continue
                
            start_price = df['Close'].iloc[0]
            end_price = df['Close'].iloc[-1]
            high_price = df['High'].max()
            low_price = df['Low'].min()
            buy_hold_return = ((end_price - start_price) / start_price) * 100
            
            # Find optimal trades (simplified swing trading analysis)
            optimal_trades = self._find_optimal_trades(symbol, df)
            max_return = self._calculate_max_return(optimal_trades, 100000)
            
            analysis = StockAnalysis(
                symbol=symbol,
                start_price=start_price,
                end_price=end_price,
                high_price=high_price,
                low_price=low_price,
                buy_hold_return=buy_hold_return,
                max_possible_return=max_return,
                optimal_trades=optimal_trades
            )
            self.analyses[symbol] = analysis
            
            print(f"\n  {symbol}:")
            print(f"    Start: ${start_price:.2f} → End: ${end_price:.2f}")
            print(f"    High: ${high_price:.2f}, Low: ${low_price:.2f}")
            print(f"    Buy & Hold Return: {buy_hold_return:.1f}%")
            print(f"    Max Possible Return (swing trading): {max_return:.1f}%")
            print(f"    Optimal Trades Found: {len(optimal_trades)}")
            
        return self.analyses
    
    def _find_optimal_trades(self, symbol: str, df: pd.DataFrame) -> List[TradeOpportunity]:
        """Find optimal swing trades using local minima/maxima."""
        trades = []
        prices = df['Close'].values
        dates = df.index.tolist()
        
        # Find local minima (buy points) and maxima (sell points)
        window = 10  # Look for turning points over 10 days
        
        i = window
        in_trade = False
        entry_idx = 0
        
        while i < len(prices) - window:
            # Check for local minimum (potential buy)
            if not in_trade:
                is_local_min = all(prices[i] <= prices[i-j] for j in range(1, window+1))
                is_local_min = is_local_min and all(prices[i] <= prices[i+j] for j in range(1, min(window+1, len(prices)-i)))
                
                if is_local_min:
                    entry_idx = i
                    in_trade = True
                    i += 1
                    continue
            
            # Check for local maximum (potential sell)  
            if in_trade:
                is_local_max = all(prices[i] >= prices[i-j] for j in range(1, window+1))
                is_local_max = is_local_max and all(prices[i] >= prices[i+j] for j in range(1, min(window+1, len(prices)-i)))
                
                if is_local_max:
                    entry_price = prices[entry_idx]
                    exit_price = prices[i]
                    return_pct = ((exit_price - entry_price) / entry_price) * 100
                    
                    # Only count profitable trades > 5%
                    if return_pct > 5:
                        trade = TradeOpportunity(
                            symbol=symbol,
                            entry_date=dates[entry_idx],
                            entry_price=entry_price,
                            exit_date=dates[i],
                            exit_price=exit_price,
                            return_pct=return_pct,
                            holding_days=(dates[i] - dates[entry_idx]).days
                        )
                        trades.append(trade)
                    in_trade = False
            i += 1

        return trades

    def _calculate_max_return(self, trades: List[TradeOpportunity], initial_capital: float) -> float:
        """Calculate maximum possible return from optimal trades."""
        if not trades:
            return 0.0
        capital = initial_capital
        for trade in trades:
            capital *= (1 + trade.return_pct / 100)
        return ((capital - initial_capital) / initial_capital) * 100

    def analyze_technical_patterns(self) -> Dict[str, Any]:
        """Analyze technical indicators at optimal entry/exit points."""
        print(f"\n{'='*60}")
        print("STEP 3: TECHNICAL PATTERN ANALYSIS")
        print(f"{'='*60}")

        all_entry_patterns = []
        all_exit_patterns = []

        for symbol, analysis in self.analyses.items():
            df = self.stock_data.get(symbol)
            if df is None or len(analysis.optimal_trades) == 0:
                continue

            # Calculate technical indicators for the full dataset
            df = self._add_technical_indicators(df)

            # Extract patterns at entry/exit points
            for trade in analysis.optimal_trades:
                entry_idx = df.index.get_loc(trade.entry_date) if trade.entry_date in df.index else None
                exit_idx = df.index.get_loc(trade.exit_date) if trade.exit_date in df.index else None

                if entry_idx is not None:
                    entry_pattern = self._extract_pattern(df, entry_idx, 'entry')
                    entry_pattern['symbol'] = symbol
                    entry_pattern['return_pct'] = trade.return_pct
                    all_entry_patterns.append(entry_pattern)

                if exit_idx is not None:
                    exit_pattern = self._extract_pattern(df, exit_idx, 'exit')
                    exit_pattern['symbol'] = symbol
                    all_exit_patterns.append(exit_pattern)

        # Analyze common patterns
        entry_analysis = self._analyze_pattern_statistics(all_entry_patterns, 'ENTRY')
        exit_analysis = self._analyze_pattern_statistics(all_exit_patterns, 'EXIT')

        return {'entry': entry_analysis, 'exit': exit_analysis,
                'entry_patterns': all_entry_patterns, 'exit_patterns': all_exit_patterns}

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ALL 64+ technical indicators using comprehensive calculator."""
        return ComprehensiveIndicatorCalculator.calculate_all(df)

    def _extract_pattern(self, df: pd.DataFrame, idx: int, pattern_type: str, lookback: int = 5) -> Dict[str, Any]:
        """Extract technical indicator values AND their changes (patterns) at a specific index."""
        row = df.iloc[idx]

        # Extract all available indicators
        pattern = {'type': pattern_type}

        # Key indicators to track
        key_indicators = [
            'RSI', 'STOCH_K', 'STOCH_D', 'SMI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'MOMENTUM', 'MOMENTUM_PCT', 'ROC', 'CMO', 'FORCE_INDEX_13',
            'BULL_POWER', 'BEAR_POWER', 'REPULSE',
            'ADX', 'PLUS_DI', 'MINUS_DI', 'AROON_UP', 'AROON_DOWN', 'AROON_OSC',
            'BB_PCT_B', 'BB_WIDTH', 'ATR_PCT', 'HIST_VOL', 'CHAIKIN_VOL', 'MASS_INDEX',
            'CMF', 'MFI', 'EMV', 'VOLUME_RATIO', 'CCI', 'WILLIAMS_R', 'DPO', 'TRIX', 'AO',
            'PRICE_VS_EMA50', 'PRICE_VS_EMA200', 'EMA8_VS_EMA21', 'EMA21_VS_EMA50',
            'TREND_STRENGTH', 'TENKAN', 'KIJUN'
        ]

        for indicator in key_indicators:
            if indicator in row.index:
                val = row[indicator]
                pattern[indicator.lower()] = val if pd.notna(val) else np.nan

                # CRITICAL: Also capture the CHANGE in indicators (momentum of indicator)
                if idx >= lookback and indicator in df.columns:
                    prev_val = df[indicator].iloc[idx - lookback]
                    if pd.notna(val) and pd.notna(prev_val) and prev_val != 0:
                        pattern[f'{indicator.lower()}_change_{lookback}d'] = val - prev_val
                        pattern[f'{indicator.lower()}_pct_change_{lookback}d'] = ((val - prev_val) / abs(prev_val)) * 100 if prev_val != 0 else 0

        # Add REVERSAL detection patterns
        if idx >= 3:
            # Price action patterns
            closes = df['Close'].iloc[idx-3:idx+1].values
            pattern['price_making_higher_lows'] = closes[-1] > closes[-2] and closes[-2] < closes[-3]
            pattern['price_making_lower_highs'] = closes[-1] < closes[-2] and closes[-2] > closes[-3]

            # RSI divergence detection
            if 'RSI' in df.columns:
                rsi_vals = df['RSI'].iloc[idx-3:idx+1].values
                if all(pd.notna(rsi_vals)):
                    # Bullish divergence: price lower low, RSI higher low
                    pattern['rsi_bullish_divergence'] = (closes[-1] < closes[-3] and rsi_vals[-1] > rsi_vals[-3])
                    # Bearish divergence: price higher high, RSI lower high
                    pattern['rsi_bearish_divergence'] = (closes[-1] > closes[-3] and rsi_vals[-1] < rsi_vals[-3])

            # MACD crossover detection
            if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
                macd = df['MACD'].iloc[idx-1:idx+1].values
                signal = df['MACD_Signal'].iloc[idx-1:idx+1].values
                if len(macd) == 2 and all(pd.notna(macd)) and all(pd.notna(signal)):
                    pattern['macd_bullish_cross'] = macd[-2] < signal[-2] and macd[-1] > signal[-1]
                    pattern['macd_bearish_cross'] = macd[-2] > signal[-2] and macd[-1] < signal[-1]

        return pattern

    def _analyze_pattern_statistics(self, patterns: List[Dict], pattern_type: str) -> Dict[str, Any]:
        """Analyze statistics of ALL indicator patterns AND reversal signals."""
        if not patterns:
            return {}

        df = pd.DataFrame(patterns)

        # All key indicators to analyze (levels)
        key_indicators = [
            'rsi', 'stoch_k', 'stoch_d', 'smi', 'macd', 'macd_histogram',
            'momentum_pct', 'roc', 'cmo', 'force_index_13',
            'bull_power', 'bear_power', 'repulse',
            'adx', 'plus_di', 'minus_di', 'aroon_osc',
            'bb_pct_b', 'bb_width', 'atr_pct', 'hist_vol', 'chaikin_vol', 'mass_index',
            'cmf', 'mfi', 'emv', 'volume_ratio', 'cci', 'williams_r', 'dpo', 'trix', 'ao',
            'price_vs_ema50', 'price_vs_ema200', 'ema8_vs_ema21', 'trend_strength'
        ]

        # Also analyze CHANGES (momentum of indicators)
        change_indicators = [f'{ind}_change_5d' for ind in key_indicators] + \
                           [f'{ind}_pct_change_5d' for ind in key_indicators]

        stats = {}
        print(f"\n  {pattern_type} Pattern Analysis (Top Indicators):")

        for col in key_indicators + change_indicators:
            if col in df.columns:
                valid = df[col].dropna()
                if len(valid) > 0:
                    stats[col] = {
                        'mean': valid.mean(),
                        'median': valid.median(),
                        'std': valid.std(),
                        'q25': valid.quantile(0.25),
                        'q75': valid.quantile(0.75),
                        'min': valid.min(),
                        'max': valid.max()
                    }

        # Print top level insights
        top_indicators = ['rsi', 'stoch_k', 'macd_histogram', 'adx', 'cmf', 'mfi',
                          'bb_pct_b', 'price_vs_ema50', 'volume_ratio', 'cci', 'williams_r']
        for col in top_indicators:
            if col in stats:
                s = stats[col]
                print(f"    {col:18s}: median={s['median']:7.2f}, q25={s['q25']:7.2f}, q75={s['q75']:7.2f}")

        # Analyze reversal signal frequency
        reversal_cols = ['rsi_bullish_divergence', 'rsi_bearish_divergence',
                        'macd_bullish_cross', 'macd_bearish_cross',
                        'price_making_higher_lows', 'price_making_lower_highs']

        print(f"\n  {pattern_type} Reversal Signals:")
        for col in reversal_cols:
            if col in df.columns:
                true_count = df[col].sum() if df[col].dtype == bool else 0
                total = len(df[col].dropna())
                if total > 0:
                    pct = (true_count / total) * 100
                    stats[col] = {'frequency': pct, 'count': true_count, 'total': total}
                    print(f"    {col:30s}: {pct:5.1f}% ({true_count}/{total})")

        # Analyze RSI change patterns (key for reversal detection)
        if 'rsi_change_5d' in stats:
            s = stats['rsi_change_5d']
            print(f"\n  RSI Change (5d) at {pattern_type}: median={s['median']:.2f}, q25={s['q25']:.2f}, q75={s['q75']:.2f}")

        return stats

    def generate_strategy_config(self, pattern_analysis: Dict) -> Dict[str, Any]:
        """Generate a TRUE SWING TRADING strategy that captures max possible returns."""
        print(f"\n{'='*60}")
        print("STEP 4: GENERATING AI SWING TRADING STRATEGY")
        print(f"{'='*60}")

        entry_stats = pattern_analysis.get('entry', {})
        exit_stats = pattern_analysis.get('exit', {})
        entry_patterns = pattern_analysis.get('entry_patterns', [])
        exit_patterns = pattern_analysis.get('exit_patterns', [])

        # Calculate performance metrics
        total_buy_hold = sum(a.buy_hold_return for a in self.analyses.values())
        total_max_return = sum(a.max_possible_return for a in self.analyses.values())
        all_trades = [t for a in self.analyses.values() for t in a.optimal_trades]
        avg_trade_return = np.mean([t.return_pct for t in all_trades]) if all_trades else 15
        avg_holding_days = np.mean([t.holding_days for t in all_trades]) if all_trades else 10
        total_trades = len(all_trades)

        # Per-stock averages (for targeting)
        avg_max_return_per_stock = total_max_return / len(self.symbols)
        trades_per_stock_per_year = (total_trades / len(self.symbols)) / self.period_years

        print(f"\n  ═══════════════════════════════════════════════════════════")
        print(f"  PERFORMANCE TARGET ANALYSIS")
        print(f"  ═══════════════════════════════════════════════════════════")
        print(f"    📊 Total Buy & Hold:         {total_buy_hold:.1f}%")
        print(f"    🎯 Max Possible (swing):     {total_max_return:.1f}%")
        print(f"    📈 Avg Max Per Stock:        {avg_max_return_per_stock:.1f}%")
        print(f"    💰 Avg Trade Return:         {avg_trade_return:.1f}%")
        print(f"    ⏱️  Avg Holding Days:         {avg_holding_days:.0f}")
        print(f"    🔄 Trades/Stock/Year:        {trades_per_stock_per_year:.1f}")
        print(f"    📋 Total Optimal Trades:     {total_trades}")

        # =====================================================================
        # SWING TRADING LOGIC: BUY THE DIP, SELL THE RIP
        # =====================================================================

        # Entry: Look for REVERSAL signals, not just oversold levels
        # Key insight: At optimal entry points, indicators are oversold AND turning up
        rsi_entry = entry_stats.get('rsi', {}).get('median', 33)
        rsi_entry_q25 = entry_stats.get('rsi', {}).get('q25', 28)  # Use Q25 for stronger signals
        stoch_entry = entry_stats.get('stoch_k', {}).get('median', 7)
        bb_entry = entry_stats.get('bb_pct_b', {}).get('median', 0.06)
        cci_entry = entry_stats.get('cci', {}).get('median', -141)
        mfi_entry = entry_stats.get('mfi', {}).get('median', 37)
        williams_entry = entry_stats.get('williams_r', {}).get('median', -93)
        adx_entry = entry_stats.get('adx', {}).get('median', 28)

        # Exit: Wait for EXHAUSTION signals
        rsi_exit = exit_stats.get('rsi', {}).get('median', 71)
        rsi_exit_q75 = exit_stats.get('rsi', {}).get('q75', 78)  # Use Q75 for later exits
        stoch_exit = exit_stats.get('stoch_k', {}).get('median', 93)
        bb_exit = exit_stats.get('bb_pct_b', {}).get('median', 0.96)
        cci_exit = exit_stats.get('cci', {}).get('median', 141)
        mfi_exit = exit_stats.get('mfi', {}).get('median', 65)
        williams_exit = exit_stats.get('williams_r', {}).get('median', -7)

        # Analyze reversal pattern frequency
        bullish_div_freq = entry_stats.get('rsi_bullish_divergence', {}).get('frequency', 0)
        macd_cross_freq = entry_stats.get('macd_bullish_cross', {}).get('frequency', 0)
        higher_lows_freq = entry_stats.get('price_making_higher_lows', {}).get('frequency', 0)

        print(f"\n  ═══════════════════════════════════════════════════════════")
        print(f"  SWING ENTRY: REVERSAL DETECTION")
        print(f"  ═══════════════════════════════════════════════════════════")
        print(f"    RSI Bullish Divergence freq: {bullish_div_freq:.1f}%")
        print(f"    MACD Bullish Cross freq:     {macd_cross_freq:.1f}%")
        print(f"    Price Higher Lows freq:      {higher_lows_freq:.1f}%")
        print(f"    Entry RSI threshold:         < {rsi_entry:.0f}")
        print(f"    Entry Stochastic threshold:  < {stoch_entry:.0f}")
        print(f"    Entry BB %B threshold:       < {bb_entry:.2f}")

        print(f"\n  ═══════════════════════════════════════════════════════════")
        print(f"  SWING EXIT: MOMENTUM EXHAUSTION")
        print(f"  ═══════════════════════════════════════════════════════════")
        print(f"    Exit RSI threshold:          > {rsi_exit:.0f}")
        print(f"    Exit Stochastic threshold:   > {stoch_exit:.0f}")
        print(f"    Exit BB %B threshold:        > {bb_exit:.2f}")

        # Design the SWING TRADING config
        # Key: Wide stops, let winners run, dynamic trailing
        config = {
            "strategy_name": "AI Swing Trading - Semiconductor Momentum",
            "version": "5.0.0",
            "strategy_type": "swing_trading",
            "description": f"Swing trading strategy targeting {avg_trade_return:.0f}%+ per trade, {avg_max_return_per_stock:.0f}%+ per stock",
            "ai_analysis": {
                "total_buy_hold_pct": round(total_buy_hold, 1),
                "max_possible_return_pct": round(total_max_return, 1),
                "avg_max_per_stock_pct": round(avg_max_return_per_stock, 1),
                "avg_trade_return_pct": round(avg_trade_return, 1),
                "avg_holding_days": round(avg_holding_days, 0),
                "trades_per_stock_per_year": round(trades_per_stock_per_year, 1),
                "optimal_trades_found": total_trades,
                "target_capture_pct": 50  # Aim to capture 50% of max possible
            },
            "trading": {
                "symbols": self.symbols,
                "timeframe": "1d",
                "max_positions": 3,  # Concentrated bets for max returns
                "position_size_pct": 100.0,  # Full allocation across positions
                "dynamic_sizing": {
                    "enabled": True,
                    "min_signal_score": 3,
                    "max_signal_score": 10,
                    "min_allocation_pct": 25,
                    "max_allocation_pct": 100
                },
                "min_trade_interval_days": 1,
                "default_quantity": 10
            },
            "entry_conditions": {
                "mode": "swing_reversal",  # NEW: Detect reversals
                "min_score": 3,  # 3+ conditions for entry
                "max_score": 10,
                "conditions": {
                    # Level conditions (oversold)
                    "rsi_oversold": rsi_entry,
                    "stoch_oversold": max(10, stoch_entry),
                    "williams_oversold": williams_entry,
                    "bb_pct_low": bb_entry,
                    "mfi_oversold": mfi_entry,
                    "cci_oversold": cci_entry,
                    # Trend conditions
                    "adx_trending": adx_entry,
                    "price_below_ema50": True,  # Looking for pullbacks
                    # Reversal conditions (NEW)
                    "rsi_turning_up": True,  # RSI starting to rise from oversold
                    "macd_bullish_cross": True,  # MACD crossing above signal
                    "higher_lows": True  # Price making higher lows
                },
                "reversal_detection": {
                    "enabled": True,
                    "rsi_divergence_weight": 2,  # Double weight for divergence
                    "macd_cross_weight": 2,
                    "higher_lows_weight": 1,
                    "min_reversal_signals": 1  # Need at least 1 reversal signal
                }
            },
            "exit_conditions": {
                "mode": "momentum_exhaustion",  # NEW: Wait for exhaustion
                "min_score": 4,  # Need strong exhaustion signals
                "conditions": {
                    # Overbought levels
                    "rsi_overbought": rsi_exit,
                    "stoch_overbought": stoch_exit,
                    "williams_overbought": williams_exit,
                    "bb_pct_high": bb_exit,
                    "mfi_overbought": mfi_exit,
                    "cci_overbought": cci_exit,
                    # Exhaustion signals (NEW)
                    "rsi_turning_down": True,
                    "macd_bearish_cross": True,
                    "lower_highs": True
                },
                "partial_exits": {
                    "enabled": True,
                    "levels": [
                        {"at_pct": 15, "sell_pct": 25},  # Sell 25% at +15%
                        {"at_pct": 25, "sell_pct": 25},  # Sell 25% at +25%
                        {"at_pct": 40, "sell_pct": 25}   # Sell 25% at +40%, hold rest
                    ]
                }
            },
            "indicators": {
                "rsi_period": 14,
                "stoch_k_period": 14,
                "stoch_d_period": 3,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "ema_fast": 8,
                "ema_medium": 21,
                "ema_slow": 50,
                "ema_trend": 200,
                "adx_period": 14,
                "adx_threshold": adx_entry,
                "mfi_period": 14,
                "cmf_period": 21,
                "volume_ma_period": 20,
                "bb_period": 20,
                "bb_std": 2.0,
                "atr_period": 14,
                "cci_period": 20,
                "williams_period": 14,
                "min_holding_days": max(5, int(avg_holding_days / 3))  # Hold longer
            },
            "risk_management": {
                # WIDE stops for swing trading - let trades develop
                "stop_loss_pct": 15.0,  # Wide initial stop
                "take_profit_pct": avg_trade_return * 1.5,  # Target 1.5x avg
                "trailing_stop_pct": 12.0,  # Wide trailing
                "trailing_stop_activation_pct": 10.0,  # Only trail after +10%
                "atr_based_stops": {
                    "enabled": True,
                    "atr_multiplier": 3.0  # 3x ATR stops
                },
                "max_daily_loss_pct": 5.0,
                "max_drawdown_pct": 20.0,  # User's 20% limit
                "stop_loss_cooldown_days": 5,
                "sector_rotation": {
                    "enabled": True,
                    "min_pct_above_ema20": 20,
                    "min_sector_rsi": 25,
                    "min_sector_momentum": -20,
                    "confirmation_days": 2
                }
            },
            "filters": {
                "trend_filter": False,  # Swing trade in both directions
                "volume_filter": True,
                "volatility_filter": False,  # Trade through volatility
                "adx_filter": True
            }
        }

        print(f"\n  ═══════════════════════════════════════════════════════════")
        print(f"  SWING TRADING STRATEGY CONFIGURATION")
        print(f"  ═══════════════════════════════════════════════════════════")
        print(f"    📊 Strategy Type:      SWING TRADING (Buy Dips, Sell Rips)")
        print(f"    🎯 Entry Mode:         Reversal Detection (min {config['entry_conditions']['min_score']} signals)")
        print(f"    📤 Exit Mode:          Momentum Exhaustion (min {config['exit_conditions']['min_score']} signals)")
        print(f"    📊 Max Positions:      {config['trading']['max_positions']}")
        print(f"    💰 Position Size:      {config['trading']['position_size_pct']}% (dynamic)")
        print(f"    🛑 Stop Loss:          {config['risk_management']['stop_loss_pct']}% (wide for swings)")
        print(f"    🎯 Take Profit Target: {config['risk_management']['take_profit_pct']:.1f}%")
        print(f"    📈 Trailing Stop:      {config['risk_management']['trailing_stop_pct']}% (after +{config['risk_management']['trailing_stop_activation_pct']}%)")
        print(f"    ⏱️  Min Hold Days:      {config['indicators']['min_holding_days']}")
        if config['exit_conditions']['partial_exits']['enabled']:
            print(f"    💵 Partial Exits:      Yes (scale out at +15%, +25%, +40%)")

        return config

    async def run_analysis(self) -> Dict[str, Any]:
        """Run the full AI analysis pipeline."""
        print("\n" + "="*60)
        print("AI STRATEGY DESIGNER - SEMICONDUCTOR MOMENTUM")
        print("="*60)
        print(f"Analyzing {len(self.symbols)} symbols over {self.period_years} years")

        # Step 1: Fetch data
        await self.fetch_all_data()

        # Step 2: Macro analysis
        self.analyze_macro_view()

        # Step 3: Technical pattern analysis
        pattern_analysis = self.analyze_technical_patterns()

        # Step 4: Generate optimized config
        config = self.generate_strategy_config(pattern_analysis)

        # Save config
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\n✅ New config saved to config.json")

        return {
            'analyses': {s: {'buy_hold': a.buy_hold_return, 'max_possible': a.max_possible_return,
                            'trades': len(a.optimal_trades)} for s, a in self.analyses.items()},
            'config': config
        }


async def main():
    """Main entry point."""
    symbols = ['NVDA', 'AMD', 'AVGO', 'TSM', 'QCOM', 'MU', 'MRVL', 'AMAT', 'LRCX', 'KLAC', 'ASML']

    designer = AIStrategyDesigner(symbols, period_years=2)
    result = await designer.run_analysis()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for symbol, data in result['analyses'].items():
        print(f"  {symbol}: Buy&Hold={data['buy_hold']:.1f}%, MaxPossible={data['max_possible']:.1f}%, Trades={data['trades']}")

    return result


if __name__ == "__main__":
    asyncio.run(main())

