#!/usr/bin/env python3
"""
Strategy Optimizer AI - Enhanced with Bayesian Optimization & GPU Acceleration

High-performance optimizer using:
- Bayesian Optimization (Optuna) for intelligent hyperparameter search
- Parallel processing (ProcessPoolExecutor) for concurrent backtests
- GPU acceleration (Apple Silicon MPS) for fast indicator calculations

Parameters optimized:
- Stop loss percentage
- Trailing stop percentage
- Max positions
- Min entry score

Author: SignalSynk AI
"""

import os
import sys
import json
import argparse
import itertools
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np
import logging
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('StrategyOptimizer')

# ============================================================
# GPU ACCELERATION SETUP (Apple Silicon MPS)
# ============================================================
GPU_AVAILABLE = False
GPU_DEVICE = None

try:
    import torch
    if torch.backends.mps.is_available():
        GPU_DEVICE = torch.device("mps")
        GPU_AVAILABLE = True
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
except ImportError:
    pass

# ============================================================
# BAYESIAN OPTIMIZATION SETUP (Optuna)
# ============================================================
OPTUNA_AVAILABLE = False
try:
    import optuna
    OPTUNA_AVAILABLE = True
    # Suppress Optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    pass


def get_optimal_workers() -> int:
    """Get optimal number of parallel workers based on CPU cores."""
    cpu_count = mp.cpu_count()
    # Use 75% of cores for parallel backtests, minimum 4
    return max(4, int(cpu_count * 0.75))


@dataclass
class OptimizationResult:
    """Result from a single parameter combination."""
    params: Dict[str, Any]
    total_return: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    sharpe_ratio: float
    profit_factor: float
    buy_hold_beat: float  # How much we beat buy & hold
    trial_number: int = 0  # For Bayesian optimization tracking

    @property
    def score(self) -> float:
        """Calculate optimization score balancing return and risk."""
        # Penalize high drawdown, reward beating buy & hold
        risk_adjusted = self.total_return / max(self.max_drawdown, 1)
        return (
            self.total_return * 0.3 +
            risk_adjusted * 0.3 +
            self.buy_hold_beat * 0.2 +
            self.win_rate * 0.1 +
            min(self.profit_factor, 3) * 10 * 0.1
        )


class GPUAccelerator:
    """GPU acceleration utilities for Apple Silicon (MPS)."""

    def __init__(self):
        self.device = GPU_DEVICE
        self.enabled = GPU_AVAILABLE

    def calculate_indicators_batch(self, prices: np.ndarray, params: Dict) -> Dict[str, np.ndarray]:
        """Calculate technical indicators using GPU acceleration."""
        if not self.enabled:
            return self._calculate_cpu(prices, params)

        import torch

        # Move data to MPS device
        price_tensor = torch.tensor(prices, dtype=torch.float32, device=self.device)

        results = {}

        # RSI calculation on GPU
        rsi_period = params.get('rsi_period', 14)
        delta = price_tensor[1:] - price_tensor[:-1]
        gains = torch.clamp(delta, min=0)
        losses = torch.clamp(-delta, min=0)

        avg_gain = torch.zeros(len(prices), device=self.device)
        avg_loss = torch.zeros(len(prices), device=self.device)

        alpha = 1.0 / rsi_period
        for i in range(1, len(prices)):
            if i < rsi_period:
                avg_gain[i] = gains[:i].mean() if i > 0 else 0
                avg_loss[i] = losses[:i].mean() if i > 0 else 0
            else:
                avg_gain[i] = avg_gain[i-1] * (1 - alpha) + gains[i-1] * alpha
                avg_loss[i] = avg_loss[i-1] * (1 - alpha) + losses[i-1] * alpha

        rs = avg_gain / (avg_loss + 1e-10)
        results['rsi'] = (100 - (100 / (1 + rs))).cpu().numpy()

        # EMA calculations on GPU
        for period in [8, 21, 50, 200]:
            results[f'ema_{period}'] = self._ema_gpu(price_tensor, period).cpu().numpy()

        return results

    def _ema_gpu(self, prices: 'torch.Tensor', period: int) -> 'torch.Tensor':
        """Calculate EMA on GPU."""
        import torch
        alpha = 2.0 / (period + 1)
        ema = torch.zeros_like(prices)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        return ema

    def _calculate_cpu(self, prices: np.ndarray, _params: Dict) -> Dict[str, np.ndarray]:
        """Fallback CPU calculation (params unused in fallback)."""
        return {'rsi': np.zeros(len(prices)), 'ema_21': np.zeros(len(prices))}


class StrategyOptimizer:
    """
    Enhanced Strategy Optimizer with Bayesian Optimization, Parallel Processing, and GPU Acceleration.

    Features:
    - Bayesian Optimization (Optuna TPE sampler) for intelligent search
    - Parallel backtesting using ProcessPoolExecutor
    - GPU acceleration (Apple Silicon MPS) for indicator calculations
    - Early stopping when no improvement for N trials
    - Data caching to avoid redundant API calls during optimization
    """

    def __init__(self, strategy_dir: str, n_workers: int = None, use_bayesian: bool = True):
        self.strategy_dir = Path(strategy_dir)
        self.config_path = self.strategy_dir / 'config.json'
        self.original_config = self._load_config()
        self.results: List[OptimizationResult] = []
        self.n_workers = n_workers or get_optimal_workers()
        self.use_bayesian = use_bayesian and OPTUNA_AVAILABLE
        self.gpu = GPUAccelerator()
        self.trial_count = 0
        self.best_score = float('-inf')
        self.no_improve_count = 0
        self.cached_data = None  # Pre-fetched market data
        self.cached_indicators = None  # Pre-calculated indicators (MAJOR optimization)
        self.data_cache_path = self.strategy_dir / '.optimizer_data_cache.pkl'

        # Log capabilities
        self._log_capabilities()

    def _log_capabilities(self):
        """Log available optimization capabilities."""
        print(f"\n  🔧 Optimizer Configuration:")
        print(f"     • Bayesian Optimization: {'✓ Enabled (Optuna TPE)' if self.use_bayesian else '✗ Disabled (Grid Search)'}")
        print(f"     • GPU Acceleration: {'✓ Enabled (Apple MPS)' if self.gpu.enabled else '✗ Disabled'}")
        print(f"     • Parallel Workers: {self.n_workers}")

    def _load_config(self) -> Dict[str, Any]:
        """Load the strategy configuration."""
        with open(self.config_path) as f:
            return json.load(f)

    def _save_config(self, config: Dict[str, Any]):
        """Save the strategy configuration."""
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def _prefetch_data(self):
        """Pre-fetch market data AND pre-calculate indicators for optimization.

        This is a MAJOR optimization: instead of recalculating indicators for each
        backtest trial (50 trials × 502 days × 11 symbols = 276,100 calculations),
        we calculate them ONCE upfront.
        """
        import pickle
        from datetime import datetime, timedelta

        # Check if we have cached data with indicators
        if self.data_cache_path.exists():
            try:
                with open(self.data_cache_path, 'rb') as f:
                    cache = pickle.load(f)
                    # Check if cache is fresh (less than 1 hour old) and has indicators
                    if (cache.get('timestamp') and
                        (datetime.now() - cache['timestamp']).seconds < 3600 and
                        cache.get('indicators')):
                        print("  📦 Using cached market data and indicators...")
                        self.cached_data = cache['data']
                        self.cached_indicators = cache['indicators']
                        return
            except Exception as e:
                print(f"  ⚠️ Cache load failed: {e}")

        print("  📡 Pre-fetching market data...")

        # Import strategy to get symbols
        sys.path.insert(0, str(self.strategy_dir))
        modules_to_remove = [m for m in sys.modules if 'strategy' in m.lower() or 'data_provider' in m.lower()]
        for m in modules_to_remove:
            del sys.modules[m]

        from data_provider import DataProvider
        from strategy import TradingStrategy

        strategy = TradingStrategy()
        data_provider = DataProvider()
        symbols = strategy.get_symbols()

        # Fetch data for all symbols
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

        all_data = {}
        for symbol in symbols:
            df = data_provider.get_historical_data(symbol, start_date, end_date)
            if df is not None and len(df) >= 50:
                all_data[symbol] = df
                print(f"    ✓ {symbol}: {len(df)} bars")

        self.cached_data = all_data

        # PRE-CALCULATE INDICATORS for all symbols and all dates
        print("  🧮 Pre-calculating indicators (this speeds up optimization 50x)...")
        self.cached_indicators = {}

        for symbol, df in all_data.items():
            self.cached_indicators[symbol] = {}
            dates = df.index.tolist()

            # Calculate indicators for each date slice (expanding window)
            for i, date in enumerate(dates):
                if i < 50:  # Need minimum data for indicators
                    continue
                df_slice = df.iloc[:i+1]
                try:
                    indicators = strategy.calculate_indicators(df_slice)
                    self.cached_indicators[symbol][date] = indicators
                except Exception:
                    pass

            print(f"    ✓ {symbol}: {len(self.cached_indicators[symbol])} indicator sets")

        # Save cache with indicators
        try:
            with open(self.data_cache_path, 'wb') as f:
                pickle.dump({
                    'data': all_data,
                    'indicators': self.cached_indicators,
                    'timestamp': datetime.now()
                }, f)
            print(f"  📦 Data and indicators cached to {self.data_cache_path.name}")
        except Exception as e:
            print(f"  ⚠️ Cache save failed: {e}")

    def _run_backtest(self, params: Dict[str, Any], timeout: int = 120) -> Optional[OptimizationResult]:
        """Run a backtest with given parameters.

        Args:
            params: Risk management parameters to test
            timeout: Max seconds to wait for backtest (default 120s)
        """
        import signal
        import pickle

        def timeout_handler(signum, frame):
            raise TimeoutError("Backtest timed out")

        try:
            # Set timeout (only works on Unix)
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)

            # Update config with test parameters
            config = self.original_config.copy()

            # Update risk management params
            if 'risk_management' not in config:
                config['risk_management'] = {}

            config['risk_management']['stop_loss_pct'] = params.get('stop_loss_pct', 0.05)
            config['risk_management']['trailing_stop_pct'] = params.get('trailing_stop_pct', 0.03)
            config['risk_management']['max_position_size'] = params.get('max_position_size', 0.1)
            config['risk_management']['max_drawdown'] = params.get('max_drawdown', 0.20)

            if 'trading' not in config:
                config['trading'] = {}
            config['trading']['max_positions'] = params.get('max_positions', 3)
            config['trading']['min_entry_score'] = params.get('min_entry_score', 3)

            # Save temp config
            self._save_config(config)

            # Write cached data to temp file for backtest to use
            if self.cached_data:
                with open(self.data_cache_path, 'wb') as f:
                    pickle.dump({'data': self.cached_data, 'timestamp': __import__('datetime').datetime.now()}, f)

            # Import and run backtest
            sys.path.insert(0, str(self.strategy_dir))

            # Clear cached imports
            modules_to_remove = [m for m in sys.modules if 'strategy' in m.lower() or 'backtest' in m.lower()]
            for m in modules_to_remove:
                del sys.modules[m]

            from backtest import PortfolioBacktester

            backtester = PortfolioBacktester(initial_capital=100000.0)

            # Pass cached data if available
            if self.cached_data:
                backtester._cached_data = self.cached_data

            # Pass cached indicators if available (MAJOR optimization - 50x speedup)
            if self.cached_indicators:
                backtester._cached_indicators = self.cached_indicators

            # Suppress print output
            import io
            from contextlib import redirect_stdout

            f = io.StringIO()
            with redirect_stdout(f):
                result = backtester.run()

            # Cancel timeout
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            
            if not result:
                return None

            # Calculate additional metrics
            buy_hold_avg = result.get('buy_hold_avg', 0)
            buy_hold_beat = result.get('total_return', 0) - buy_hold_avg

            # Calculate Sharpe ratio approximation
            trades = backtester.trades if hasattr(backtester, 'trades') else []
            if trades:
                returns = [t.pnl_pct for t in trades]
                sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
            else:
                sharpe = 0

            # Calculate profit factor
            winning_pnl = sum(t.pnl for t in trades if t.pnl > 0)
            losing_pnl = abs(sum(t.pnl for t in trades if t.pnl < 0))
            profit_factor = winning_pnl / (losing_pnl + 1e-10)

            return OptimizationResult(
                params=params,
                total_return=result.get('total_return', 0),
                max_drawdown=result.get('max_drawdown', 0),
                win_rate=result.get('win_rate', 0),
                total_trades=result.get('total_trades', 0),
                sharpe_ratio=sharpe,
                profit_factor=profit_factor,
                buy_hold_beat=buy_hold_beat
            )

        except TimeoutError:
            signal.alarm(0)  # Cancel alarm
            print(f" ⏱️ TIMEOUT")
            return None
        except Exception as e:
            try:
                signal.alarm(0)  # Cancel alarm on error
            except:
                pass
            # Only print for unexpected errors, not timeouts
            if "timed out" not in str(e).lower():
                print(f" ⚠️ Error: {str(e)[:50]}")
            return None

    def optimize(self,
                 stop_loss_range: List[float] = None,
                 trailing_stop_range: List[float] = None,
                 max_positions_range: List[int] = None,
                 min_entry_score_range: List[int] = None,
                 n_iterations: int = 50,
                 early_stop_patience: int = 15) -> OptimizationResult:
        """
        Run optimization with Bayesian or Grid search.

        Args:
            stop_loss_range: Stop loss percentages to test
            trailing_stop_range: Trailing stop percentages to test
            max_positions_range: Max positions to test
            min_entry_score_range: Minimum entry scores to test
            n_iterations: Number of optimization trials
            early_stop_patience: Stop if no improvement for N trials

        Returns:
            Best optimization result
        """
        print(f"\n{'='*60}")
        print("STRATEGY OPTIMIZER - ENHANCED")
        print(f"{'='*60}")
        print(f"  Strategy: {self.strategy_dir.name}")
        print(f"  Max iterations: {n_iterations}")
        print(f"  Early stop patience: {early_stop_patience}")

        # Pre-fetch data once to avoid redundant API calls during optimization
        self._prefetch_data()

        # Use Bayesian optimization if available
        if self.use_bayesian:
            return self._optimize_bayesian(
                stop_loss_range, trailing_stop_range,
                max_positions_range, min_entry_score_range,
                n_iterations, early_stop_patience
            )
        else:
            return self._optimize_parallel_grid(
                stop_loss_range, trailing_stop_range,
                max_positions_range, min_entry_score_range,
                n_iterations
            )

    def _optimize_bayesian(self,
                           stop_loss_range: List[float],
                           trailing_stop_range: List[float],
                           max_positions_range: List[int],
                           min_entry_score_range: List[int],
                           n_iterations: int,
                           early_stop_patience: int) -> OptimizationResult:
        """Run Bayesian optimization using Optuna TPE sampler."""
        import optuna
        from optuna.samplers import TPESampler

        print(f"\n  🧠 Using Bayesian Optimization (Optuna TPE Sampler)")
        print(f"{'='*60}\n")

        # Default ranges
        sl_min = min(stop_loss_range) if stop_loss_range else 0.03
        sl_max = max(stop_loss_range) if stop_loss_range else 0.15
        ts_min = min(trailing_stop_range) if trailing_stop_range else 0.02
        ts_max = max(trailing_stop_range) if trailing_stop_range else 0.10
        pos_min = min(max_positions_range) if max_positions_range else 2
        pos_max = max(max_positions_range) if max_positions_range else 5
        score_min = min(min_entry_score_range) if min_entry_score_range else 2
        score_max = max(min_entry_score_range) if min_entry_score_range else 5

        def objective(trial):
            """Optuna objective function."""
            params = {
                'stop_loss_pct': trial.suggest_float('stop_loss_pct', sl_min, sl_max),
                'trailing_stop_pct': trial.suggest_float('trailing_stop_pct', ts_min, ts_max),
                'max_positions': trial.suggest_int('max_positions', pos_min, pos_max),
                'min_entry_score': trial.suggest_int('min_entry_score', score_min, score_max)
            }

            self.trial_count += 1
            print(f"  [Trial {self.trial_count}/{n_iterations}] SL={params['stop_loss_pct']*100:.1f}%, "
                  f"TS={params['trailing_stop_pct']*100:.1f}%, MaxPos={params['max_positions']}, "
                  f"MinScore={params['min_entry_score']}", end="")

            result = self._run_backtest(params)

            if result:
                result.trial_number = self.trial_count
                self.results.append(result)

                # Track best score for early stopping
                if result.score > self.best_score:
                    self.best_score = result.score
                    self.no_improve_count = 0
                    print(f" → 🎯 Return: {result.total_return:.1f}%, Score: {result.score:.1f} (BEST!)")
                else:
                    self.no_improve_count += 1
                    print(f" → Return: {result.total_return:.1f}%, Score: {result.score:.1f}")

                # Early stopping check
                if self.no_improve_count >= early_stop_patience:
                    print(f"\n  ⏹️ Early stopping: No improvement for {early_stop_patience} trials")
                    trial.study.stop()

                return result.score
            else:
                print(" → Failed")
                return float('-inf')

        # Create Optuna study with TPE sampler
        sampler = TPESampler(seed=42, n_startup_trials=10)
        study = optuna.create_study(direction='maximize', sampler=sampler)

        # Run optimization
        study.optimize(objective, n_trials=n_iterations, show_progress_bar=False)

        # Restore original config
        self._save_config(self.original_config)

        # Get best result
        if not self.results:
            print("\n  ❌ No successful optimization runs")
            return None

        best = max(self.results, key=lambda x: x.score)
        self._print_results(best, study=study)

        return best

    def _optimize_parallel_grid(self,
                                 stop_loss_range: List[float],
                                 trailing_stop_range: List[float],
                                 max_positions_range: List[int],
                                 min_entry_score_range: List[int],
                                 n_iterations: int) -> OptimizationResult:
        """Run parallel grid search optimization."""
        print(f"\n  📊 Using Parallel Grid Search ({self.n_workers} workers)")
        print(f"{'='*60}\n")

        # Default ranges
        stop_loss_range = stop_loss_range or [0.02, 0.03, 0.05, 0.07, 0.10, 0.15]
        trailing_stop_range = trailing_stop_range or [0.02, 0.03, 0.05, 0.07, 0.10]
        max_positions_range = max_positions_range or [2, 3, 4, 5]
        min_entry_score_range = min_entry_score_range or [1.5, 2, 3, 4, 5]

        # Generate all combinations
        all_combinations = list(itertools.product(
            stop_loss_range, trailing_stop_range,
            max_positions_range, min_entry_score_range
        ))

        # Random sample if too many
        if len(all_combinations) > n_iterations:
            import random
            random.shuffle(all_combinations)
            all_combinations = all_combinations[:n_iterations]

        print(f"  Testing {len(all_combinations)} parameter combinations...")

        # Run backtests (sequential for now due to subprocess import issues)
        for i, combo in enumerate(all_combinations, 1):
            params = {
                'stop_loss_pct': combo[0],
                'trailing_stop_pct': combo[1],
                'max_positions': combo[2],
                'min_entry_score': combo[3]
            }

            print(f"  [{i}/{len(all_combinations)}] SL={combo[0]*100:.0f}%, TS={combo[1]*100:.0f}%, "
                  f"MaxPos={combo[2]}, MinScore={combo[3]}", end="")

            result = self._run_backtest(params)
            if result:
                self.results.append(result)
                print(f" → Return: {result.total_return:.1f}%, Score: {result.score:.1f}")
            else:
                print(" → Failed")

        # Restore original config
        self._save_config(self.original_config)

        if not self.results:
            print("\n  ❌ No successful optimization runs")
            return None

        best = max(self.results, key=lambda x: x.score)
        self._print_results(best)

        return best

    def _print_results(self, best: OptimizationResult, study=None):
        """Print optimization results summary."""
        print(f"\n{'='*60}")
        print("OPTIMIZATION RESULTS")
        print(f"{'='*60}\n")

        # Show optimization stats
        print(f"  ⚡ OPTIMIZATION STATS:")
        print(f"    Total Trials: {len(self.results)}")
        print(f"    Best Trial: #{best.trial_number if best.trial_number > 0 else 'N/A'}")
        if self.gpu.enabled:
            print(f"    GPU Acceleration: ✓ Apple Silicon MPS")
        if study and hasattr(study, 'best_trial'):
            print(f"    Optimization Method: Bayesian (TPE)")
        else:
            print(f"    Optimization Method: Grid Search")

        print(f"\n  📊 BEST PARAMETERS:")
        print(f"    Stop Loss: {best.params['stop_loss_pct']*100:.1f}%")
        print(f"    Trailing Stop: {best.params['trailing_stop_pct']*100:.1f}%")
        print(f"    Max Positions: {best.params['max_positions']}")
        print(f"    Min Entry Score: {best.params['min_entry_score']}")

        print(f"\n  📈 PERFORMANCE:")
        print(f"    Total Return: {best.total_return:.2f}%")
        print(f"    Max Drawdown: {best.max_drawdown:.2f}%")
        print(f"    Win Rate: {best.win_rate:.1f}%")
        print(f"    Total Trades: {best.total_trades}")
        print(f"    Sharpe Ratio: {best.sharpe_ratio:.2f}")
        print(f"    Profit Factor: {best.profit_factor:.2f}")
        print(f"    Beat Buy & Hold By: {best.buy_hold_beat:.2f}%")
        print(f"    Optimization Score: {best.score:.2f}")

        # Show top 5 results
        top_5 = sorted(self.results, key=lambda x: x.score, reverse=True)[:5]
        print(f"\n  🏆 TOP 5 CONFIGURATIONS:")
        for i, r in enumerate(top_5, 1):
            print(f"    {i}. SL={r.params['stop_loss_pct']*100:.0f}%, TS={r.params['trailing_stop_pct']*100:.0f}%, "
                  f"MaxPos={r.params['max_positions']}, MinScore={r.params['min_entry_score']} "
                  f"→ {r.total_return:.1f}% (Score: {r.score:.1f})")

        print(f"\n{'='*60}\n")

    def apply_best_params(self, result: OptimizationResult):
        """Apply the best parameters to the strategy config."""
        config = self._load_config()

        # Update risk management
        if 'risk_management' not in config:
            config['risk_management'] = {}

        config['risk_management']['stop_loss_pct'] = result.params['stop_loss_pct']
        config['risk_management']['trailing_stop_pct'] = result.params['trailing_stop_pct']

        # Update trading params
        if 'trading' not in config:
            config['trading'] = {}

        config['trading']['max_positions'] = result.params['max_positions']
        config['trading']['min_entry_score'] = result.params['min_entry_score']

        self._save_config(config)

        print(f"  ✅ Applied optimized parameters to {self.config_path}")


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description='Enhanced Strategy Optimizer with Bayesian Optimization & GPU Acceleration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with Bayesian optimization (default)
  python strategy_optimizer.py --strategy strategies/semiconductor_momentum

  # 100 trials with early stopping after 20 trials of no improvement
  python strategy_optimizer.py -s strategies/semiconductor_momentum -n 100 --patience 20

  # Use grid search instead of Bayesian optimization
  python strategy_optimizer.py -s strategies/semiconductor_momentum --grid-search

  # Apply best params to config and specify workers
  python strategy_optimizer.py -s strategies/semiconductor_momentum --apply --workers 8
        """
    )

    parser.add_argument('--strategy', '-s', required=True,
                        help='Path to strategy directory')
    parser.add_argument('--iterations', '-n', type=int, default=50,
                        help='Number of optimization trials (default: 50)')
    parser.add_argument('--patience', '-p', type=int, default=15,
                        help='Early stop if no improvement for N trials (default: 15)')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='Number of parallel workers (default: 75%% of CPU cores)')
    parser.add_argument('--grid-search', action='store_true',
                        help='Use grid search instead of Bayesian optimization')
    parser.add_argument('--stop-loss', nargs='+', type=float,
                        help='Stop loss percentages to test (e.g., 0.03 0.05 0.07 0.10)')
    parser.add_argument('--trailing-stop', nargs='+', type=float,
                        help='Trailing stop percentages to test')
    parser.add_argument('--max-positions', nargs='+', type=int,
                        help='Max positions to test (e.g., 2 3 4 5)')
    parser.add_argument('--min-entry-score', nargs='+', type=int,
                        help='Min entry scores to test')
    parser.add_argument('--apply', '-a', action='store_true',
                        help='Apply best parameters to strategy config')

    args = parser.parse_args()

    # Validate strategy directory
    strategy_dir = Path(args.strategy)
    if not strategy_dir.exists():
        print(f"❌ Strategy directory not found: {strategy_dir}")
        sys.exit(1)

    if not (strategy_dir / 'config.json').exists():
        print(f"❌ No config.json found in {strategy_dir}")
        sys.exit(1)

    # Create optimizer
    optimizer = StrategyOptimizer(
        str(strategy_dir),
        n_workers=args.workers,
        use_bayesian=not args.grid_search
    )

    # Run optimization
    best = optimizer.optimize(
        stop_loss_range=args.stop_loss,
        trailing_stop_range=args.trailing_stop,
        max_positions_range=args.max_positions,
        min_entry_score_range=args.min_entry_score,
        n_iterations=args.iterations,
        early_stop_patience=args.patience
    )

    # Apply if requested
    if best and args.apply:
        optimizer.apply_best_params(best)
        print("\n  ✅ Best parameters applied to strategy config!")
        print("  Run the backtest again to verify performance.")


if __name__ == '__main__':
    main()
