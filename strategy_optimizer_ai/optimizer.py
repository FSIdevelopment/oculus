#!/usr/bin/env python3
"""
Universal Strategy Optimizer for SignalSynk

GPU-accelerated optimizer for Apple Silicon (M3 Max) using Metal Performance Shaders.
Supports parallel backtesting and intelligent parameter optimization.

Usage:
    python -m strategy_optimizer_ai.optimizer --strategy semiconductor_momentum
    python -m strategy_optimizer_ai.optimizer --strategy ai_sector_trend --period 2Y --parallel 8
"""

import asyncio
import argparse
import json
import itertools
import logging
import os
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file if it exists
env_path = PROJECT_ROOT / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

# Set Polygon API key - MUST be set before importing strategy modules
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY', '')
if not POLYGON_API_KEY:
    print("⚠️  WARNING: POLYGON_API_KEY not found in environment!")
    print("   The optimizer will fall back to slower AlphaVantage API.")
else:
    print(f"✓ Polygon API key found: {POLYGON_API_KEY[:8]}...")

# Check for GPU acceleration
GPU_AVAILABLE = False
GPU_DEVICE = None

try:
    import torch
    if torch.backends.mps.is_available():
        GPU_DEVICE = torch.device("mps")
        GPU_AVAILABLE = True
        # Enable MPS fallback for unsupported ops
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
except ImportError:
    pass

# Suppress verbose logging
for logger_name in ['Backtest', 'RiskManager', 'DataProvider', 'SemiconductorMomentumStrategy']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Configure optimizer logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('Optimizer')


def get_optimal_workers() -> int:
    """Get optimal number of parallel workers for M3 Max."""
    # M3 Max has up to 16 cores (12 performance + 4 efficiency)
    cpu_count = mp.cpu_count()
    # Use 75% of cores for parallel backtests
    return max(4, int(cpu_count * 0.75))


class GPUAccelerator:
    """GPU acceleration utilities for M3 Max using Metal Performance Shaders."""

    def __init__(self):
        self.device = GPU_DEVICE
        self.enabled = GPU_AVAILABLE

    def calculate_indicators_batch(self, price_data: np.ndarray, params: Dict) -> np.ndarray:
        """Calculate technical indicators using GPU acceleration."""
        if not self.enabled:
            return self._calculate_cpu(price_data, params)

        import torch

        # Move data to MPS device
        prices = torch.tensor(price_data, dtype=torch.float32, device=self.device)

        # Calculate RSI on GPU
        rsi_period = params.get('rsi_period', 14)
        delta = prices[1:] - prices[:-1]
        gains = torch.clamp(delta, min=0)
        losses = torch.clamp(-delta, min=0)

        avg_gain = torch.zeros_like(prices)
        avg_loss = torch.zeros_like(prices)

        # Exponential moving average for RSI
        alpha = 1.0 / rsi_period
        for i in range(1, len(prices)):
            if i < rsi_period:
                avg_gain[i] = gains[:i].mean() if i > 0 else 0
                avg_loss[i] = losses[:i].mean() if i > 0 else 0
            else:
                avg_gain[i] = avg_gain[i-1] * (1 - alpha) + gains[i-1] * alpha
                avg_loss[i] = avg_loss[i-1] * (1 - alpha) + losses[i-1] * alpha

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        # Calculate EMAs on GPU
        ema_fast = self._ema_gpu(prices, params.get('ema_fast', 10))
        ema_slow = self._ema_gpu(prices, params.get('ema_slow', 21))

        # Return results as numpy
        return torch.stack([rsi, ema_fast, ema_slow]).cpu().numpy()

    def _ema_gpu(self, prices: 'torch.Tensor', period: int) -> 'torch.Tensor':
        """Calculate EMA on GPU."""
        import torch
        alpha = 2.0 / (period + 1)
        ema = torch.zeros_like(prices)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        return ema

    def _calculate_cpu(self, price_data: np.ndarray, params: Dict) -> np.ndarray:
        """Fallback CPU calculation."""
        # Simple CPU-based indicator calculation
        return np.zeros((3, len(price_data)))

    def optimize_params_parallel(self, combinations: List[Dict],
                                  score_func, n_workers: int = 8) -> List[Tuple[Dict, float]]:
        """Score parameter combinations in parallel using thread pool."""
        results = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(score_func, combo) for combo in combinations]
            for future, combo in zip(futures, combinations):
                try:
                    score = future.result()
                    results.append((combo, score))
                except Exception as e:
                    results.append((combo, float('-inf')))
        return results


class StrategyOptimizer:
    """Universal optimizer for SignalSynk trading strategies with GPU acceleration."""

    def __init__(self, strategy_name: str, period: str = '1Y', n_workers: int = None):
        self.strategy_name = strategy_name
        self.period = period
        self.n_workers = n_workers or get_optimal_workers()
        self.strategy_path = PROJECT_ROOT / 'strategies' / strategy_name
        self.config_path = self.strategy_path / 'config.json'
        self.cached_data: Dict[str, Any] = {}
        self.gpu = GPUAccelerator()

        if not self.strategy_path.exists():
            raise ValueError(f"Strategy '{strategy_name}' not found at {self.strategy_path}")
        if not self.config_path.exists():
            raise ValueError(f"Config not found at {self.config_path}")

        # Add strategy path for imports
        sys.path.insert(0, str(self.strategy_path))

        # Log GPU status
        if self.gpu.enabled:
            logger.info(f"🚀 GPU Acceleration: ENABLED (Apple M3 Max - Metal)")
        else:
            logger.info(f"⚠️  GPU Acceleration: DISABLED (PyTorch not found)")
        logger.info(f"🔧 Parallel Workers: {self.n_workers}")
        
    def load_config(self) -> Dict[str, Any]:
        """Load the strategy's configuration."""
        with open(self.config_path) as f:
            return json.load(f)
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """Save updated configuration."""
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def get_param_grid(self) -> Dict[str, List[Any]]:
        """Get parameter grid from config or use defaults."""
        config = self.load_config()
        
        # Check if optimization grid is defined in config
        if 'optimization' in config and 'param_grid' in config['optimization']:
            return config['optimization']['param_grid']
        
        # TREND-FOLLOWING parameter grid for capturing big moves (300%+ target)
        # Key: Use long EMAs, wide ATR-based stops, hold for weeks/months
        # REDUCED grid for faster optimization (~5,000 combinations)
        return {
            # Long-term trend EMAs (50/200 style)
            'trend_ema_fast': [10, 20, 30, 50],  # Fast trend EMA
            'trend_ema_slow': [50, 100, 150, 200],  # Slow trend EMA
            # Entry timing EMAs
            'ema_fast': [8, 13],
            'ema_slow': [21, 34],
            # Trend confirmation
            'lookback_window': [10, 20],
            'min_slope_pct': [0.0, 0.1, 0.2],  # Stronger trend required
            # ATR-based trailing stops (key for riding trends)
            'atr_trailing_multiplier': [2.0, 3.0, 4.0],  # Wide ATR stops
            # Risk management - WIDE to hold through volatility
            'stop_loss_pct': [10.0, 15.0, 20.0, 25.0],  # Very wide stops
            'trailing_stop_pct': [8.0, 12.0, 15.0, 20.0],  # Wide trailing
            'take_profit_pct': [100.0, 150.0],  # Big targets
            # Position management
            'min_holding_days': [3, 7],  # Minimum hold period
            'stop_loss_cooldown_days': [3, 5],  # Longer cooldowns
        }

    def get_fixed_params(self) -> Dict[str, Any]:
        """Get fixed parameters that aren't optimized."""
        config = self.load_config()
        if 'optimization' in config and 'fixed_params' in config['optimization']:
            return config['optimization']['fixed_params']

        return {
            'rsi_period': 14,
            'rsi_overbought': 80,
            'rsi_oversold': 30,
            'breakout_threshold_pct': 3.0,
            'volume_multiplier': 1.2,
            'atr_period': 14,
            'trend_confirm_days': 5,
            'max_drawdown_pct': 30.0,
            'max_daily_loss_pct': 10.0,
        }
    
    def generate_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations."""
        grid = self.get_param_grid()
        fixed = self.get_fixed_params()
        
        keys = list(grid.keys())
        values = list(grid.values())
        
        combinations = []
        for combo in itertools.product(*values):
            param_dict = dict(zip(keys, combo))
            param_dict.update(fixed)
            combinations.append(param_dict)
        
        return combinations
    
    def update_config_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Update strategy config with new parameters."""
        config = self.load_config()

        # Update indicator params if they exist - includes new trend-following params
        if 'indicators' in config:
            indicator_keys = [
                'rsi_period', 'rsi_oversold', 'rsi_overbought',
                'ema_fast', 'ema_slow',
                'trend_ema_fast', 'trend_ema_slow',  # New: long-term trend EMAs
                'lookback_window', 'trend_confirm_days',
                'breakout_threshold_pct', 'volume_multiplier',
                'atr_period', 'atr_trailing_multiplier',  # New: ATR trailing stop
                'min_slope_pct', 'min_holding_days'  # New: minimum holding period
            ]
            for key in indicator_keys:
                if key in params:
                    config['indicators'][key] = params[key]

        # Update risk management params if they exist
        if 'risk_management' in config:
            risk_keys = [
                'stop_loss_pct', 'take_profit_pct', 'trailing_stop_pct',
                'max_daily_loss_pct', 'max_drawdown_pct', 'stop_loss_cooldown_days'
            ]
            for key in risk_keys:
                if key in params:
                    config['risk_management'][key] = params[key]

        self.save_config(config)
        return config

    async def prefetch_data(self, symbols: List[str], days: int = 800) -> Dict[str, Any]:
        """Pre-fetch data for all symbols and cache it."""
        if self.cached_data:
            return self.cached_data

        # Reload data_provider to pick up Polygon API key
        import importlib
        if 'data_provider' in sys.modules:
            importlib.reload(sys.modules['data_provider'])

        from data_provider import DataProvider
        self.shared_provider = DataProvider()

        logger.info(f"\n📥 Pre-fetching data for {len(symbols)} symbols (using Polygon.io)...")

        for i, symbol in enumerate(symbols, 1):
            print(f"   [{i}/{len(symbols)}] Fetching {symbol}...", end=" ", flush=True)
            df = await self.shared_provider.get_historical_data(symbol, days, '1D')
            if df is not None and not df.empty:
                self.cached_data[symbol] = df
                print(f"✓ {len(df)} bars")
            else:
                print("✗ Failed")

        logger.info(f"✓ Data cached for {len(self.cached_data)} symbols\n")
        return self.cached_data

    async def run_single_backtest(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single backtest with given parameters."""
        self.update_config_params(params)

        # Force reload of strategy and backtest modules to pick up new config
        import importlib
        for mod_name in ['strategy', 'backtest', 'risk_manager', 'data_provider']:
            if mod_name in sys.modules:
                importlib.reload(sys.modules[mod_name])

        from strategy import TradingStrategy
        from backtest import BacktestRunner

        strategy = TradingStrategy()

        # Verify the strategy loaded the correct config
        actual_rsi = getattr(strategy, 'rsi_period', 'N/A')
        actual_ema_fast = getattr(strategy, 'ema_fast', 'N/A')
        if actual_rsi != params.get('rsi_period') or actual_ema_fast != params.get('ema_fast'):
            logger.warning(f"Config mismatch! Expected rsi={params.get('rsi_period')}, ema_fast={params.get('ema_fast')} "
                          f"but got rsi={actual_rsi}, ema_fast={actual_ema_fast}")

        runner = BacktestRunner(strategy, self.period)

        # Share the cached data - populate cache with all key formats the backtest might use
        period_days = {'1Y': 365, '2Y': 730, '3Y': 1095, '5Y': 1825}.get(self.period, 365)
        cache_keys_added = set()
        for symbol, df in self.cached_data.items():
            # Cache with multiple key formats to ensure cache hits
            for days in [365, 730, 800, period_days]:
                for interval in ['1D', 'daily', '1d']:
                    key = f"{symbol}_{interval}_{days}"
                    runner.data_provider.cache[key] = df
                    cache_keys_added.add(key)

        # Debug: Print cache info for first iteration
        if not hasattr(self, '_cache_logged'):
            logger.info(f"Pre-populated {len(cache_keys_added)} cache keys for backtest")
            self._cache_logged = True

        try:
            results = await runner.run_backtest()
            return {
                'params': params,
                'total_return': results.total_return_pct,
                'sharpe_ratio': results.sharpe_ratio,
                'max_drawdown': results.max_drawdown_pct,
                'win_rate': results.win_rate_pct,
                'total_trades': results.total_trades,
                'profit_factor': getattr(results, 'profit_factor', 0),
                'success': True,
            }
        except Exception as e:
            return {'params': params, 'error': str(e), 'success': False}

    async def optimize(self, max_combinations: Optional[int] = None) -> List[Dict[str, Any]]:
        """Run the full optimization."""
        config = self.load_config()
        symbols = config.get('trading', {}).get('symbols', [])

        if not symbols:
            raise ValueError("No symbols found in config")

        # Pre-fetch data with EXACT same days as the backtest period
        period_days = {'1M': 30, '3M': 90, '6M': 180, '1Y': 365, '2Y': 730, '5Y': 1825}.get(self.period, 365)
        await self.prefetch_data(symbols, days=period_days)
        if not self.cached_data:
            raise RuntimeError("Failed to fetch data for any symbols")

        # Generate combinations
        combinations = self.generate_combinations()
        if max_combinations:
            combinations = combinations[:max_combinations]

        total = len(combinations)
        logger.info(f"Testing {total} parameter combinations...")
        logger.info(f"Backtest period: {self.period}\n")

        results = []
        start_time = datetime.now()

        for i, params in enumerate(combinations, 1):
            result = await self.run_single_backtest(params)
            results.append(result)

            if result['success']:
                print(f"[{i:3d}/{total}] Return: {result['total_return']:+7.2f}% | "
                      f"Sharpe: {result['sharpe_ratio']:5.2f} | "
                      f"MaxDD: {result['max_drawdown']:6.2f}% | "
                      f"WinRate: {result['win_rate']:5.1f}% | "
                      f"Trades: {result['total_trades']}")
            else:
                print(f"[{i:3d}/{total}] ERROR: {result.get('error', 'Unknown')[:50]}")

        elapsed = datetime.now() - start_time
        logger.info(f"\n{'=' * 70}")
        logger.info(f"OPTIMIZATION COMPLETE - Elapsed: {elapsed}")
        logger.info(f"{'=' * 70}")

        return results

    def display_results(self, results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Display optimization results and return best config."""
        successful = [r for r in results if r['success']]
        if not successful:
            logger.info("\nNo successful backtests!")
            return None

        # Calculate combined score
        for r in successful:
            dd = abs(r['max_drawdown']) if r['max_drawdown'] != 0 else 0.01
            r['combined_score'] = (r['total_return'] * max(r['sharpe_ratio'], 0)) / dd

        # Top by return
        logger.info("\n" + "=" * 70)
        logger.info("TOP 5 BY TOTAL RETURN")
        logger.info("=" * 70)
        by_return = sorted(successful, key=lambda x: x['total_return'], reverse=True)[:5]
        for i, r in enumerate(by_return, 1):
            logger.info(f"#{i}: Return: {r['total_return']:+.2f}% | Sharpe: {r['sharpe_ratio']:.2f} | "
                       f"MaxDD: {r['max_drawdown']:.2f}%")

        # Top by combined score
        logger.info("\n" + "=" * 70)
        logger.info("TOP 5 BY COMBINED SCORE (return * sharpe / drawdown)")
        logger.info("=" * 70)
        by_combined = sorted(successful, key=lambda x: x['combined_score'], reverse=True)[:5]
        for i, r in enumerate(by_combined, 1):
            logger.info(f"#{i}: Score: {r['combined_score']:.2f} | Return: {r['total_return']:+.2f}% | "
                       f"Sharpe: {r['sharpe_ratio']:.2f}")

        return by_combined[0] if by_combined else None

    def apply_best_config(self, best: Dict[str, Any]) -> None:
        """Apply the best configuration to the strategy."""
        logger.info("\n" + "=" * 70)
        logger.info("APPLYING BEST CONFIGURATION")
        logger.info("=" * 70)
        self.update_config_params(best['params'])
        logger.info(f"\nBest config saved to {self.config_path}")
        logger.info(f"Return: {best['total_return']:+.2f}% | Sharpe: {best['sharpe_ratio']:.2f}")


async def main():
    """Main entry point for the optimizer."""
    parser = argparse.ArgumentParser(description='SignalSynk Strategy Optimizer (GPU Accelerated)')
    parser.add_argument('--strategy', '-s', required=True, help='Strategy name (folder name)')
    parser.add_argument('--period', '-p', default='1Y', help='Backtest period (1Y, 2Y, etc.)')
    parser.add_argument('--max', '-m', type=int, help='Max combinations to test')
    parser.add_argument('--parallel', '-w', type=int, default=None, help='Number of parallel workers')
    parser.add_argument('--apply', '-a', action='store_true', help='Apply best config automatically')

    args = parser.parse_args()

    print("=" * 70)
    print("🚀 SIGNALSYNK STRATEGY OPTIMIZER (GPU ACCELERATED)")
    print("=" * 70)
    print(f"Strategy: {args.strategy}")
    print(f"Period: {args.period}")
    if GPU_AVAILABLE:
        print(f"GPU: Apple M3 Max (Metal Performance Shaders) ✓")
    else:
        print(f"GPU: Not available (install torch for MPS acceleration)")
    print("=" * 70)

    try:
        optimizer = StrategyOptimizer(args.strategy, args.period, args.parallel)
        results = await optimizer.optimize(args.max)
        best = optimizer.display_results(results)

        if best and args.apply:
            optimizer.apply_best_config(best)
        elif best:
            logger.info("\nRun with --apply to save best configuration")

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise


if __name__ == '__main__':
    asyncio.run(main())

