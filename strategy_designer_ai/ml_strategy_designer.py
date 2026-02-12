#!/usr/bin/env python3
"""
ML Strategy Designer - Machine Learning Based Trading Strategy Generator

This module trains ML models to predict profitable entry/exit points:
1. Creates labeled training data from historical price movements
2. Engineers features from 64+ technical indicators
3. Trains ensemble models (Random Forest, XGBoost, LightGBM)
4. Generates trading signals based on model predictions
5. Outputs optimized strategy configuration

Author: SignalSynk AI
"""

import asyncio
import json
import os
import sys
import pickle
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score
import joblib

warnings.filterwarnings('ignore')

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                'strategies', 'semiconductor_momentum'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_provider import DataProvider

# Try to import advanced ML libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not installed. Using sklearn alternatives.")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠️  LightGBM not installed. Using sklearn alternatives.")


@dataclass
class MLConfig:
    """Configuration for ML training."""
    # Label generation
    forward_returns_days: int = 10  # Look ahead N days for returns
    profit_threshold: float = 5.0   # % gain to label as profitable entry
    loss_threshold: float = -3.0    # % loss to label as bad entry
    
    # Training
    test_size: float = 0.2
    n_splits: int = 5  # For time series cross-validation
    random_state: int = 42
    
    # Model hyperparameters
    n_estimators: int = 200
    max_depth: int = 10
    min_samples_leaf: int = 20
    
    # Feature engineering
    lookback_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    

class FeatureEngineer:
    """Creates ML features from price data and technical indicators."""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.feature_names: List[str] = []
        
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all 64+ technical indicators."""
        df = df.copy()
        
        # =====================================================================
        # MOMENTUM INDICATORS
        # =====================================================================
        
        # RSI (multiple periods)
        for period in [7, 14, 21]:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # Stochastic Oscillator
        for period in [14, 21]:
            low_n = df['Low'].rolling(period).min()
            high_n = df['High'].rolling(period).max()
            df[f'STOCH_K_{period}'] = ((df['Close'] - low_n) / (high_n - low_n + 1e-10)) * 100
            df[f'STOCH_D_{period}'] = df[f'STOCH_K_{period}'].rolling(3).mean()
        
        # Williams %R
        df['WILLIAMS_R'] = ((df['High'].rolling(14).max() - df['Close']) / 
                           (df['High'].rolling(14).max() - df['Low'].rolling(14).min() + 1e-10)) * -100
        
        # MACD
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_SIGNAL'] = df['MACD'].ewm(span=9).mean()
        df['MACD_HIST'] = df['MACD'] - df['MACD_SIGNAL']
        
        # ROC (Rate of Change)
        for period in [5, 10, 20]:
            df[f'ROC_{period}'] = ((df['Close'] - df['Close'].shift(period)) / 
                                   (df['Close'].shift(period) + 1e-10)) * 100
        
        # MFI (Money Flow Index)
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        mf = tp * df['Volume']
        pos_mf = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
        neg_mf = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
        df['MFI'] = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-10)))
        
        # CCI (Commodity Channel Index) - OPTIMIZED: using std approximation instead of slow rolling apply
        tp_sma = tp.rolling(20).mean()
        # Mean Absolute Deviation ≈ 0.7979 * std for normal distribution (much faster than rolling apply)
        mean_dev = tp.rolling(20).std() * 0.7979
        df['CCI'] = (tp - tp_sma) / (0.015 * mean_dev + 1e-10)

        # =====================================================================
        # TREND INDICATORS
        # =====================================================================

        # EMAs (multiple periods)
        for period in [8, 13, 21, 50, 100, 200]:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
            df[f'PRICE_VS_EMA_{period}'] = ((df['Close'] - df[f'EMA_{period}']) /
                                            (df[f'EMA_{period}'] + 1e-10)) * 100

        # ADX (Average Directional Index)
        tr = pd.concat([
            df['High'] - df['Low'],
            abs(df['High'] - df['Close'].shift()),
            abs(df['Low'] - df['Close'].shift())
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        up_move = df['High'] - df['High'].shift()
        down_move = df['Low'].shift() - df['Low']
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

        plus_di = 100 * (plus_dm.rolling(14).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(14).mean() / (atr + 1e-10))
        dx = abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
        df['ADX'] = dx.rolling(14).mean()
        df['PLUS_DI'] = plus_di
        df['MINUS_DI'] = minus_di
        df['DI_DIFF'] = plus_di - minus_di

        # =====================================================================
        # VOLATILITY INDICATORS
        # =====================================================================

        # ATR
        df['ATR'] = atr
        df['ATR_PCT'] = (atr / (df['Close'] + 1e-10)) * 100

        # Bollinger Bands
        for period in [20]:
            bb_mid = df['Close'].rolling(period).mean()
            bb_std = df['Close'].rolling(period).std()
            df['BB_UPPER'] = bb_mid + 2 * bb_std
            df['BB_LOWER'] = bb_mid - 2 * bb_std
            df['BB_PCT_B'] = (df['Close'] - df['BB_LOWER']) / (df['BB_UPPER'] - df['BB_LOWER'] + 1e-10)
            df['BB_WIDTH'] = (df['BB_UPPER'] - df['BB_LOWER']) / (bb_mid + 1e-10) * 100

        # Keltner Channels
        kc_mid = df['Close'].ewm(span=20).mean()
        df['KC_UPPER'] = kc_mid + 2 * atr
        df['KC_LOWER'] = kc_mid - 2 * atr
        df['KC_PCT'] = (df['Close'] - df['KC_LOWER']) / (df['KC_UPPER'] - df['KC_LOWER'] + 1e-10)

        # =====================================================================
        # VOLUME INDICATORS
        # =====================================================================

        # Volume Ratio
        df['VOLUME_RATIO'] = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-10)

        # OBV (On Balance Volume)
        obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['OBV'] = obv
        df['OBV_EMA'] = obv.ewm(span=20).mean()
        df['OBV_DIFF'] = obv - df['OBV_EMA']

        # CMF (Chaikin Money Flow)
        clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-10)
        clv = clv.fillna(0)
        mfv = clv * df['Volume']
        df['CMF'] = mfv.rolling(21).sum() / (df['Volume'].rolling(21).sum() + 1e-10)

        # VWAP
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / (df['Volume'].cumsum() + 1e-10)
        df['PRICE_VS_VWAP'] = ((df['Close'] - df['VWAP']) / (df['VWAP'] + 1e-10)) * 100

        return df

    def add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pattern recognition features (divergences, crossovers, etc.)."""
        df = df.copy()

        # =====================================================================
        # MOMENTUM / CHANGE FEATURES
        # =====================================================================

        # Indicator changes (momentum of indicators)
        for col in ['RSI_14', 'STOCH_K_14', 'MACD_HIST', 'CCI', 'MFI', 'ADX']:
            if col in df.columns:
                df[f'{col}_CHG_1'] = df[col].diff(1)
                df[f'{col}_CHG_3'] = df[col].diff(3)
                df[f'{col}_CHG_5'] = df[col].diff(5)

        # Price momentum
        for period in [1, 3, 5, 10, 20]:
            df[f'PRICE_CHG_{period}'] = df['Close'].pct_change(period) * 100

        # =====================================================================
        # CROSSOVER FEATURES
        # =====================================================================

        # MACD Crossovers
        if 'MACD' in df.columns and 'MACD_SIGNAL' in df.columns:
            df['MACD_CROSS_UP'] = ((df['MACD'].shift(1) < df['MACD_SIGNAL'].shift(1)) &
                                   (df['MACD'] > df['MACD_SIGNAL'])).astype(int)
            df['MACD_CROSS_DOWN'] = ((df['MACD'].shift(1) > df['MACD_SIGNAL'].shift(1)) &
                                     (df['MACD'] < df['MACD_SIGNAL'])).astype(int)

        # Stochastic Crossovers
        if 'STOCH_K_14' in df.columns and 'STOCH_D_14' in df.columns:
            df['STOCH_CROSS_UP'] = ((df['STOCH_K_14'].shift(1) < df['STOCH_D_14'].shift(1)) &
                                    (df['STOCH_K_14'] > df['STOCH_D_14']) &
                                    (df['STOCH_K_14'] < 30)).astype(int)
            df['STOCH_CROSS_DOWN'] = ((df['STOCH_K_14'].shift(1) > df['STOCH_D_14'].shift(1)) &
                                      (df['STOCH_K_14'] < df['STOCH_D_14']) &
                                      (df['STOCH_K_14'] > 70)).astype(int)

        # EMA Crossovers
        if 'EMA_8' in df.columns and 'EMA_21' in df.columns:
            df['EMA_CROSS_UP'] = ((df['EMA_8'].shift(1) < df['EMA_21'].shift(1)) &
                                  (df['EMA_8'] > df['EMA_21'])).astype(int)
            df['EMA_CROSS_DOWN'] = ((df['EMA_8'].shift(1) > df['EMA_21'].shift(1)) &
                                    (df['EMA_8'] < df['EMA_21'])).astype(int)

        # =====================================================================
        # DIVERGENCE FEATURES (simplified)
        # =====================================================================

        # RSI Divergence: Price making new low but RSI making higher low
        if 'RSI_14' in df.columns:
            price_5d_min = df['Close'].rolling(5).min()
            rsi_5d_min = df['RSI_14'].rolling(5).min()

            # Bullish divergence: price new low, RSI higher low
            df['RSI_BULL_DIV'] = ((df['Close'] <= price_5d_min) &
                                  (df['RSI_14'] > rsi_5d_min.shift(5)) &
                                  (df['RSI_14'] < 40)).astype(int)

            # Bearish divergence: price new high, RSI lower high
            price_5d_max = df['Close'].rolling(5).max()
            rsi_5d_max = df['RSI_14'].rolling(5).max()
            df['RSI_BEAR_DIV'] = ((df['Close'] >= price_5d_max) &
                                  (df['RSI_14'] < rsi_5d_max.shift(5)) &
                                  (df['RSI_14'] > 60)).astype(int)

        # =====================================================================
        # REVERSAL FEATURES
        # =====================================================================

        # RSI turning up/down
        if 'RSI_14' in df.columns:
            df['RSI_TURNING_UP'] = ((df['RSI_14'] > df['RSI_14'].shift(1)) &
                                    (df['RSI_14'].shift(1) < df['RSI_14'].shift(2))).astype(int)
            df['RSI_TURNING_DOWN'] = ((df['RSI_14'] < df['RSI_14'].shift(1)) &
                                      (df['RSI_14'].shift(1) > df['RSI_14'].shift(2))).astype(int)

        # Price making higher lows / lower highs
        df['HIGHER_LOW'] = ((df['Low'] > df['Low'].shift(1)) &
                            (df['Low'].shift(1) < df['Low'].shift(2))).astype(int)
        df['LOWER_HIGH'] = ((df['High'] < df['High'].shift(1)) &
                            (df['High'].shift(1) > df['High'].shift(2))).astype(int)

        # =====================================================================
        # OVERSOLD / OVERBOUGHT ZONES
        # =====================================================================

        if 'RSI_14' in df.columns:
            df['RSI_OVERSOLD'] = (df['RSI_14'] < 30).astype(int)
            df['RSI_OVERBOUGHT'] = (df['RSI_14'] > 70).astype(int)
            df['RSI_EXTREME_OVERSOLD'] = (df['RSI_14'] < 20).astype(int)
            df['RSI_EXTREME_OVERBOUGHT'] = (df['RSI_14'] > 80).astype(int)

        if 'STOCH_K_14' in df.columns:
            df['STOCH_OVERSOLD'] = (df['STOCH_K_14'] < 20).astype(int)
            df['STOCH_OVERBOUGHT'] = (df['STOCH_K_14'] > 80).astype(int)

        if 'BB_PCT_B' in df.columns:
            df['BB_OVERSOLD'] = (df['BB_PCT_B'] < 0.1).astype(int)
            df['BB_OVERBOUGHT'] = (df['BB_PCT_B'] > 0.9).astype(int)

        # =====================================================================
        # MACRO/TREND FEATURES - Long-term trend indicators for growth stocks
        # =====================================================================

        # Long-term price momentum (weekly/monthly returns)
        for period in [21, 42, 63, 126, 252]:  # ~1mo, 2mo, 3mo, 6mo, 1yr
            df[f'MOMENTUM_{period}D'] = df['Close'].pct_change(period) * 100

        # Price relative to 52-week high/low (position in range)
        high_252 = df['High'].rolling(252).max()
        low_252 = df['Low'].rolling(252).min()
        df['PRICE_VS_52W_HIGH'] = ((df['Close'] - high_252) / (high_252 + 1e-10)) * 100
        df['PRICE_VS_52W_LOW'] = ((df['Close'] - low_252) / (low_252 + 1e-10)) * 100
        df['PRICE_52W_RANGE_PCT'] = ((df['Close'] - low_252) / (high_252 - low_252 + 1e-10)) * 100

        # Higher highs / higher lows trend (uptrend strength)
        df['HIGHER_HIGH_20D'] = (df['High'] > df['High'].rolling(20).max().shift(1)).astype(int)
        df['HIGHER_LOW_20D'] = (df['Low'] > df['Low'].rolling(20).min().shift(1)).astype(int)
        df['UPTREND_SCORE'] = df['HIGHER_HIGH_20D'].rolling(20).sum() + df['HIGHER_LOW_20D'].rolling(20).sum()

        # EMA stack (bullish when short > medium > long)
        if all(f'EMA_{p}' in df.columns for p in [8, 21, 50, 200]):
            df['EMA_STACK_BULLISH'] = (
                (df['EMA_8'] > df['EMA_21']) &
                (df['EMA_21'] > df['EMA_50']) &
                (df['EMA_50'] > df['EMA_200'])
            ).astype(int)
            df['EMA_STACK_SCORE'] = (
                (df['EMA_8'] > df['EMA_21']).astype(int) +
                (df['EMA_21'] > df['EMA_50']).astype(int) +
                (df['EMA_50'] > df['EMA_200']).astype(int)
            )

        # Price acceleration (momentum of momentum)
        df['PRICE_ACCEL_5D'] = df['Close'].pct_change(5).diff(5) * 100
        df['PRICE_ACCEL_10D'] = df['Close'].pct_change(10).diff(10) * 100
        df['PRICE_ACCEL_20D'] = df['Close'].pct_change(20).diff(20) * 100

        # Trend strength over multiple timeframes
        for period in [20, 50, 100]:
            slope = df['Close'].rolling(period).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
                raw=False
            )
            df[f'TREND_SLOPE_{period}D'] = slope / (df['Close'] + 1e-10) * 100  # Normalized slope

        # Consecutive up/down days
        df['UP_DAY'] = (df['Close'] > df['Close'].shift(1)).astype(int)
        df['DOWN_DAY'] = (df['Close'] < df['Close'].shift(1)).astype(int)
        df['CONSEC_UP_DAYS'] = df['UP_DAY'].groupby((df['UP_DAY'] != df['UP_DAY'].shift()).cumsum()).cumsum()
        df['CONSEC_DOWN_DAYS'] = df['DOWN_DAY'].groupby((df['DOWN_DAY'] != df['DOWN_DAY'].shift()).cumsum()).cumsum()

        # Distance from all-time high (within available data)
        ath = df['High'].expanding().max()
        df['DIST_FROM_ATH'] = ((df['Close'] - ath) / (ath + 1e-10)) * 100

        # Volume trend (increasing volume supports trend)
        df['VOLUME_TREND_20D'] = df['Volume'].rolling(20).mean() / (df['Volume'].rolling(50).mean() + 1e-10)
        df['VOLUME_EXPANDING'] = (df['Volume'].rolling(5).mean() > df['Volume'].rolling(20).mean()).astype(int)

        # Breakout features
        df['BREAKOUT_20D_HIGH'] = (df['Close'] > df['High'].rolling(20).max().shift(1)).astype(int)
        df['BREAKOUT_50D_HIGH'] = (df['Close'] > df['High'].rolling(50).max().shift(1)).astype(int)
        df['BREAKDOWN_20D_LOW'] = (df['Close'] < df['Low'].rolling(20).min().shift(1)).astype(int)

        # Gap features (gaps up often signal momentum)
        df['GAP_UP'] = ((df['Open'] - df['Close'].shift(1)) / (df['Close'].shift(1) + 1e-10)) * 100
        df['GAP_UP_STRONG'] = (df['GAP_UP'] > 2).astype(int)  # >2% gap up

        # Consolidation detection (low volatility before potential breakout)
        df['BB_SQUEEZE'] = (df['BB_WIDTH'] < df['BB_WIDTH'].rolling(50).quantile(0.2)).astype(int)

        return df

    def create_labels(self, df: pd.DataFrame, config: MLConfig) -> pd.DataFrame:
        """Create target labels based on future returns."""
        df = df.copy()

        # Calculate forward returns for the requested days
        # Always include the config's forward_returns_days
        days_to_calculate = set([5, 10, 15, 20, config.forward_returns_days])
        for days in days_to_calculate:
            col_name = f'FORWARD_RETURN_{days}D'
            if col_name not in df.columns:
                df[col_name] = (
                    (df['Close'].shift(-days) - df['Close']) / df['Close'] * 100
                )

        # Primary label: Is this a good entry point?
        # 1 = Yes (price goes up by profit_threshold in forward_returns_days)
        # 0 = No (price doesn't go up enough or goes down)
        primary_return = df[f'FORWARD_RETURN_{config.forward_returns_days}D']
        df['LABEL_ENTRY'] = (primary_return >= config.profit_threshold).astype(int)

        # Secondary label: Is this a good exit point?
        # 1 = Yes (price goes down after this point)
        df['LABEL_EXIT'] = (primary_return <= config.loss_threshold).astype(int)

        # Multi-class label for more nuanced predictions
        # 0 = Bad entry (price goes down)
        # 1 = Neutral (price stays flat)
        # 2 = Good entry (price goes up significantly)
        df['LABEL_MULTI'] = 1  # Default neutral
        df.loc[primary_return >= config.profit_threshold, 'LABEL_MULTI'] = 2
        df.loc[primary_return <= config.loss_threshold, 'LABEL_MULTI'] = 0

        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Prepare feature matrix from dataframe.

        CRITICAL: Must exclude ALL forward-looking columns to prevent data leakage!
        A 1.0 F1 score is a sign of data leakage - the model is seeing the future.
        """
        # Select numeric columns that aren't labels or targets
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'OBV',
                       'LABEL_ENTRY', 'LABEL_EXIT', 'LABEL_MULTI', 'SYMBOL']

        # CRITICAL: Exclude ALL forward return columns (prevents data leakage!)
        # These use shift(-N) which looks into the future
        forward_cols = [col for col in df.columns if col.startswith('FORWARD_RETURN')]
        exclude_cols.extend(forward_cols)

        feature_cols = [col for col in df.columns
                       if col not in exclude_cols
                       and df[col].dtype in ['float64', 'int64', 'float32', 'int32']]

        self.feature_names = feature_cols
        return df[feature_cols].values, feature_cols


class MLStrategyDesigner:
    """Machine Learning based strategy designer."""

    def __init__(self, symbols: List[str], period_years: int = 2, config: MLConfig = None):
        self.symbols = symbols
        self.period_years = period_years
        self.config = config or MLConfig()
        self.data_provider = DataProvider()
        self.feature_engineer = FeatureEngineer(self.config)

        # Data storage
        self.stock_data: Dict[str, pd.DataFrame] = {}
        self.processed_data: Dict[str, pd.DataFrame] = {}
        self.combined_data: Optional[pd.DataFrame] = None

        # Models
        self.entry_model = None
        self.exit_model = None
        self.scaler = None
        self.feature_names: List[str] = []

        # Results
        self.training_metrics: Dict[str, Any] = {}
        self.feature_importance: Dict[str, float] = {}

    async def fetch_all_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for all symbols."""
        print(f"\n{'='*60}")
        print("STEP 1: FETCHING HISTORICAL DATA")
        print(f"{'='*60}")

        days = self.period_years * 365 + 100  # Extra for indicator warmup

        for symbol in self.symbols:
            df = await self.data_provider.get_historical_data(symbol, days, interval='day')
            if df is not None and len(df) > 0:
                self.stock_data[symbol] = df
                print(f"  ✓ {symbol}: {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")
            else:
                print(f"  ✗ {symbol}: No data available")

        return self.stock_data

    def process_all_data(self) -> pd.DataFrame:
        """Process all stock data: calculate indicators, features, and labels."""
        print(f"\n{'='*60}")
        print("STEP 2: FEATURE ENGINEERING")
        print(f"{'='*60}")

        all_data = []

        for symbol, df in self.stock_data.items():
            print(f"  Processing {symbol}...")

            # Calculate indicators
            df = self.feature_engineer.calculate_all_indicators(df)

            # Add pattern features
            df = self.feature_engineer.add_pattern_features(df)

            # Create labels
            df = self.feature_engineer.create_labels(df, self.config)

            # Add symbol column
            df['SYMBOL'] = symbol

            # Drop NaN rows
            df = df.dropna()

            self.processed_data[symbol] = df
            all_data.append(df)

            print(f"    → {len(df)} samples with {len(df.columns)} features")

        # Combine all data
        self.combined_data = pd.concat(all_data, ignore_index=True)
        print(f"\n  Total samples: {len(self.combined_data)}")

        # Class distribution
        entry_dist = self.combined_data['LABEL_ENTRY'].value_counts()
        print(f"  Entry labels: 1={entry_dist.get(1, 0)} ({entry_dist.get(1, 0)/len(self.combined_data)*100:.1f}%), "
              f"0={entry_dist.get(0, 0)} ({entry_dist.get(0, 0)/len(self.combined_data)*100:.1f}%)")

        return self.combined_data

    def train_models(self) -> Dict[str, Any]:
        """Train ML models for entry/exit prediction."""
        print(f"\n{'='*60}")
        print("STEP 3: TRAINING ML MODELS")
        print(f"{'='*60}")

        if self.combined_data is None:
            raise ValueError("Must process data first!")

        # Prepare features
        X, feature_names = self.feature_engineer.prepare_features(self.combined_data)
        self.feature_names = feature_names
        y_entry = self.combined_data['LABEL_ENTRY'].values

        print(f"  Features: {len(feature_names)}")
        print(f"  Samples: {len(X)}")

        # Scale features
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Handle NaN/Inf
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Time-based train/test split (use last 20% as test)
        split_idx = int(len(X_scaled) * (1 - self.config.test_size))
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y_entry[:split_idx], y_entry[split_idx:]

        print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")

        # =====================================================================
        # Train Entry Model
        # =====================================================================
        print(f"\n  Training Entry Prediction Model...")

        models_to_try = {}

        # Random Forest
        models_to_try['RandomForest'] = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            class_weight='balanced',
            random_state=self.config.random_state,
            n_jobs=-1
        )

        # Gradient Boosting
        models_to_try['GradientBoosting'] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state
        )

        # XGBoost (if available)
        if HAS_XGBOOST:
            models_to_try['XGBoost'] = xgb.XGBClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_child_weight=self.config.min_samples_leaf,
                scale_pos_weight=len(y_train[y_train==0]) / (len(y_train[y_train==1]) + 1),
                random_state=self.config.random_state,
                n_jobs=-1,
                verbosity=0
            )

        # LightGBM (if available)
        if HAS_LIGHTGBM:
            models_to_try['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_child_samples=self.config.min_samples_leaf,
                class_weight='balanced',
                random_state=self.config.random_state,
                n_jobs=-1,
                verbose=-1
            )

        # Train and evaluate each model
        best_model = None
        best_score = 0
        best_model_name = ""

        for name, model in models_to_try.items():
            print(f"    Training {name}...")
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            try:
                auc = roc_auc_score(y_test, y_proba)
            except:
                auc = 0.5

            print(f"      Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, "
                  f"Recall: {recall:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")

            # Use F1 score for model selection (balances precision and recall)
            if f1 > best_score:
                best_score = f1
                best_model = model
                best_model_name = name
                self.training_metrics[name] = {
                    'accuracy': accuracy, 'precision': precision,
                    'recall': recall, 'f1': f1, 'auc': auc
                }

        self.entry_model = best_model
        print(f"\n  ✓ Best Model: {best_model_name} (F1={best_score:.3f})")

        # =====================================================================
        # Feature Importance
        # =====================================================================
        print(f"\n  Top 20 Most Important Features:")

        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[::-1]

            for i in range(min(20, len(indices))):
                idx = indices[i]
                self.feature_importance[feature_names[idx]] = float(importances[idx])
                print(f"    {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

        return self.training_metrics

    def generate_strategy_config(self) -> Dict[str, Any]:
        """Generate trading strategy configuration from trained model."""
        print(f"\n{'='*60}")
        print("STEP 4: GENERATING STRATEGY CONFIGURATION")
        print(f"{'='*60}")

        if self.entry_model is None:
            raise ValueError("Must train model first!")

        # Get predictions on all data
        X, _ = self.feature_engineer.prepare_features(self.combined_data)
        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Get prediction probabilities
        proba = self.entry_model.predict_proba(X_scaled)[:, 1]

        # Find high-confidence entry points
        high_conf_mask = proba >= 0.6
        high_conf_data = self.combined_data[high_conf_mask]

        print(f"  High-confidence entry points: {len(high_conf_data)} ({len(high_conf_data)/len(self.combined_data)*100:.1f}%)")

        # Extract thresholds from high-confidence entries
        config = {
            "strategy_type": "ml_momentum",
            "model_metrics": self.training_metrics,
            "feature_importance": dict(list(self.feature_importance.items())[:15]),
            "thresholds": {},
            "entry_conditions": [],
            "exit_conditions": []
        }

        # Analyze key indicators at high-confidence entries
        key_indicators = ['RSI_14', 'STOCH_K_14', 'MACD_HIST', 'CCI', 'MFI',
                         'BB_PCT_B', 'ADX', 'VOLUME_RATIO', 'PRICE_VS_EMA_50']

        print(f"\n  Indicator Thresholds at High-Confidence Entries:")
        for ind in key_indicators:
            if ind in high_conf_data.columns and not high_conf_data[ind].isna().all():
                values = high_conf_data[ind].dropna()
                if len(values) > 0:
                    median = values.median()
                    q25 = values.quantile(0.25)
                    q75 = values.quantile(0.75)
                    config["thresholds"][ind] = {
                        "median": float(median),
                        "q25": float(q25),
                        "q75": float(q75)
                    }
                    print(f"    {ind}: median={median:.2f}, Q25={q25:.2f}, Q75={q75:.2f}")

        # Generate entry conditions based on top features
        top_features = list(self.feature_importance.keys())[:10]
        for feat in top_features:
            if feat in high_conf_data.columns:
                values = high_conf_data[feat].dropna()
                if len(values) > 0:
                    config["entry_conditions"].append({
                        "indicator": feat,
                        "operator": "<=",
                        "value": float(values.quantile(0.75))
                    })

        # Create semiconductor-specific config
        semiconductor_config = self._create_semiconductor_config(config, high_conf_data)

        return semiconductor_config

    def _create_semiconductor_config(self, ml_config: Dict, high_conf_data: pd.DataFrame) -> Dict[str, Any]:
        """Create semiconductor strategy config from ML insights."""

        # Get median values for key indicators
        def safe_median(col):
            if col in high_conf_data.columns:
                vals = high_conf_data[col].dropna()
                return float(vals.median()) if len(vals) > 0 else None
            return None

        strategy_config = {
            "name": "ML Semiconductor Momentum Strategy",
            "symbols": self.symbols,
            "model_generated": True,
            "generation_date": datetime.now().isoformat(),
            "training_metrics": ml_config.get("model_metrics", {}),

            # Entry thresholds derived from ML
            "entry": {
                "rsi_max": min(safe_median('RSI_14') or 40, 45),
                "stochastic_max": min(safe_median('STOCH_K_14') or 25, 30),
                "mfi_max": min(safe_median('MFI') or 40, 45),
                "bb_pct_b_max": min(safe_median('BB_PCT_B') or 0.2, 0.25),
                "cci_max": safe_median('CCI') or -100,
                "volume_ratio_min": max(safe_median('VOLUME_RATIO') or 1.0, 1.0),
                "adx_min": max(safe_median('ADX') or 20, 18),
                "price_vs_ema50_max": safe_median('PRICE_VS_EMA_50') or -3.0,
            },

            # Exit thresholds (inverse of entry)
            "exit": {
                "rsi_min": 70,
                "stochastic_min": 80,
                "mfi_min": 75,
                "bb_pct_b_min": 0.9,
            },

            # Risk management
            "risk": {
                "max_positions": 3,
                "capital_per_position": 0.33,
                "stop_loss": 0.15,
                "take_profit": 1.0,
                "max_daily_loss": 0.05,
                "max_drawdown": 0.20,
                "trailing_stop": 0.10
            },

            # Position sizing
            "sizing": {
                "dynamic_sizing_enabled": True,
                "min_position_size": 0.25,
                "max_position_size": 1.0,
                "signal_strength_scaling": True
            },

            # Sector rotation
            "sector_rotation": {
                "enabled": True,
                "sector_weakness_threshold": -0.015,
                "evaluation_period": 5
            },

            # Top predictive features from ML
            "ml_features": {
                "top_features": list(ml_config.get("feature_importance", {}).keys())[:15],
                "feature_thresholds": ml_config.get("thresholds", {})
            }
        }

        return strategy_config

    def save_model(self, filepath: str = None):
        """Save trained model and scaler to disk."""
        if filepath is None:
            filepath = os.path.join(os.path.dirname(__file__), 'ml_model.pkl')

        model_data = {
            'entry_model': self.entry_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics,
            'config': self.config
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\n✓ Model saved to {filepath}")

    def load_model(self, filepath: str = None):
        """Load trained model and scaler from disk."""
        if filepath is None:
            filepath = os.path.join(os.path.dirname(__file__), 'ml_model.pkl')

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.entry_model = model_data['entry_model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.training_metrics = model_data['training_metrics']

        print(f"✓ Model loaded from {filepath}")

    async def run_analysis(self) -> Dict[str, Any]:
        """Run the complete ML analysis pipeline."""
        print(f"\n{'#'*60}")
        print("ML STRATEGY DESIGNER - SEMICONDUCTOR MOMENTUM")
        print(f"{'#'*60}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Period: {self.period_years} years")
        print(f"Forward returns window: {self.config.forward_returns_days} days")
        print(f"Profit threshold: {self.config.profit_threshold}%")

        # Step 1: Fetch data
        await self.fetch_all_data()

        # Step 2: Feature engineering
        self.process_all_data()

        # Step 3: Train models
        self.train_models()

        # Step 4: Generate strategy config
        strategy_config = self.generate_strategy_config()

        # Step 5: Save model
        self.save_model()

        # Save strategy config
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'strategies', 'semiconductor_momentum', 'ml_config.json'
        )
        with open(config_path, 'w') as f:
            json.dump(strategy_config, f, indent=2, default=str)

        print(f"\n✓ Strategy config saved to {config_path}")

        # Print summary
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"  Best Model: {list(self.training_metrics.keys())[0] if self.training_metrics else 'N/A'}")
        if self.training_metrics:
            best_metrics = list(self.training_metrics.values())[0]
            print(f"  F1 Score: {best_metrics.get('f1', 0):.3f}")
            print(f"  Precision: {best_metrics.get('precision', 0):.3f}")
            print(f"  Recall: {best_metrics.get('recall', 0):.3f}")
        print(f"  Top Features: {list(self.feature_importance.keys())[:5]}")

        return strategy_config


# Semiconductor symbols
SEMICONDUCTOR_SYMBOLS = [
    "NVDA", "AMD", "AVGO", "TSM", "QCOM",
    "MU", "MRVL", "AMAT", "LRCX", "KLAC", "ASML"
]


async def main():
    """Main entry point for ML Strategy Designer."""
    print("\n" + "="*70)
    print("  ML STRATEGY DESIGNER - Training on Semiconductor Stocks")
    print("="*70)

    # Configure ML training
    config = MLConfig(
        forward_returns_days=10,   # Look ahead 10 days
        profit_threshold=5.0,       # 5% gain = good entry
        loss_threshold=-3.0,        # 3% loss = bad entry
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20
    )

    # Create designer
    designer = MLStrategyDesigner(
        symbols=SEMICONDUCTOR_SYMBOLS,
        period_years=2,
        config=config
    )

    # Run analysis
    strategy_config = await designer.run_analysis()

    print("\n" + "="*70)
    print("  TRAINING COMPLETE - Ready for backtesting")
    print("="*70)

    return strategy_config


if __name__ == "__main__":
    asyncio.run(main())
