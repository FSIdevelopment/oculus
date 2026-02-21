"""
Job processor - deserializes training jobs and invokes the ML pipeline.
"""

import asyncio
import importlib.util
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import redis

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from strategy_designer_ai.ml_enhanced_trainer import EnhancedMLTrainer, EnhancedMLConfig
from strategy_designer_ai.rule_extractor import RuleExtractor
from strategy_designer_ai.strategy_optimizer import StrategyOptimizer
from worker.config import config
from worker.progress_reporter import ProgressReporter
from worker.strategy_file_generator import StrategyFileGenerator

logger = logging.getLogger(__name__)


class JobProcessor:
    """Processes training jobs using the existing ML pipeline."""

    def __init__(self, redis_client: redis.Redis):
        """Initialize job processor.

        Args:
            redis_client: Redis client instance
        """
        self.redis = redis_client
        self._setup_environment()

    def _setup_environment(self):
        """Setup environment for ML pipeline."""
        # Set Polygon API key if available
        if config.polygon_api_key:
            os.environ['POLYGON_API_KEY'] = config.polygon_api_key

        # Enable GPU acceleration if available
        if config.gpu_enabled:
            try:
                import torch
                if torch.backends.mps.is_available():
                    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                    logger.info("GPU acceleration (MPS) enabled")
            except ImportError:
                logger.warning("PyTorch not available, GPU acceleration disabled")

    def _check_stop_signal(self, build_id: str) -> bool:
        """Check if build should be stopped via Redis stop key."""
        try:
            stop_key = f"oculus:build:stop:{build_id}"
            stop_flag = self.redis.get(stop_key)
            return bool(stop_flag)
        except Exception as e:
            logger.warning(f"Failed to check stop signal: {e}")
            return False

    async def process_job(self, job_data: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        """Process a training job.

        Args:
            job_data: Job configuration with fields:
                - build_id: Build identifier
                - user_id: User identifier
                - symbols: List of stock symbols
                - design: LLM design guidance (optional)
                - timeframe: Trading timeframe (e.g., "1d")
            job_id: Unique job identifier from Redis queue

        Returns:
            Result dictionary with status, metrics, rules, and backtest results
        """
        # Extract build_id first so the reporter publishes to the channel that
        # the backend WebSocket subscribes to: oculus:training:progress:build:{uuid}
        # (the full job_id contains an extra ":iter:{uuid}" suffix that would
        # cause a channel mismatch and silence all progress updates in the UI).
        build_id = job_data.get('build_id', 'unknown')
        iteration_number = job_data.get('iteration_number')
        reporter = ProgressReporter(
            job_id, self.redis,
            channel_id=f"build:{build_id}",
            iteration=iteration_number,
        )

        try:
            # Extract job parameters from backend format
            symbols = job_data.get('symbols', [])
            timeframe = job_data.get('timeframe', '1d')
            design = job_data.get('design', {})
            user_id = job_data.get('user_id', 'unknown')
            iteration_uuid = job_data.get('iteration_uuid')  # For BuildIteration DB persistence

            # Extract basic parameters (backward compatible with old job format)
            years = design.get('years', 0.25) if design else 0.25
            strategy_type = design.get('strategy_type', 'momentum') if design else 'momentum'

            # Enforce minimum data periods based on timeframe
            # Daily/weekly: at least 1 year; intraday (<1d): at least 6 months
            if timeframe in ['1d', '1day', 'daily', '1w', 'weekly']:
                years = max(7.0, 7.0)
            else:
                # Intraday (1m, 5m, 15m, 30m, 1h, 4h)
                years = max(years, 0.5)
            logger.info(f"[{job_id}] Timeframe {timeframe}: using {years} years of data")

            # Compute timeframe-appropriate label configuration.
            #
            # Intraday timeframes MUST use smaller forward periods and lower profit
            # thresholds.  The LLM designs in "calendar-day" terms (e.g. 5–15 days,
            # 4–10 % profit) which work fine for daily bars but cause all training
            # labels to collapse into a single class for hourly/sub-hourly data
            # (e.g. NVDA gains 4 % in 5 trading days nearly every time during a
            # bull run → all labels = 1, no negative class → training fails).
            _is_sub_hourly = timeframe in ['1m', '5m', '15m', '30m']
            _is_hourly = timeframe in ['1h', '2h', '4h']
            _is_intraday = _is_sub_hourly or _is_hourly

            if _is_sub_hourly:
                # Sub-hourly: very short look-ahead, very small move thresholds
                _default_forward_days = [1, 2, 3, 5]
                _default_profit_thresholds = [0.3, 0.5, 0.75, 1.0, 1.5]
            elif _is_hourly:
                # Hourly / multi-hour: short look-ahead, small move thresholds
                _default_forward_days = [1, 2, 3, 5, 7]
                _default_profit_thresholds = [0.5, 1.0, 1.5, 2.0, 3.0]
            else:
                # Daily / weekly: honour LLM design values
                _default_forward_days = design.get(
                    'forward_returns_days_options', [3, 5, 7, 10, 15, 20]
                ) if design else [3, 5, 7, 10, 15, 20]
                _default_profit_thresholds = design.get(
                    'profit_threshold_options', [1.5, 2.0, 3.0, 5.0]
                ) if design else [1.5, 2.0, 3.0, 5.0]

            # Assign final values; log override for intraday so it's visible in traces.
            forward_returns_days_options = _default_forward_days
            profit_threshold_options = _default_profit_thresholds
            if _is_intraday:
                logger.info(
                    f"[{job_id}] Intraday timeframe ({timeframe}): overriding LLM label "
                    f"config → forward_days={forward_returns_days_options}, "
                    f"profit_thresholds={profit_threshold_options}"
                )

            # Build ML config dynamically from LLM design
            # Top-level fields come directly from the design dict
            # Nested ML training fields come from the 'ml_training_config' sub-object
            ml_config = design.get('ml_training_config', {}) if design else {}

            ml_training_config = EnhancedMLConfig(
                # Label generation — timeframe-aware values computed above
                forward_returns_days_options=forward_returns_days_options,
                profit_threshold_options=profit_threshold_options,

                # Hyperparameter search — top-level or fall back to old 'hp_iterations' key
                n_iter_search=design.get(
                    'recommended_n_iter_search',
                    design.get('hp_iterations', 50) if design else 50
                ),
                cv_folds=ml_config.get('cv_folds', 3),

                # Model selection — from nested ml_training_config
                config_search_model=ml_config.get('config_search_model', 'LightGBM'),
                train_neural_networks=ml_config.get('train_neural_networks', True),
                train_lstm=ml_config.get('train_lstm', True),

                # Neural network architecture
                nn_hidden_layers=[
                    tuple(layer) for layer in ml_config.get(
                        'nn_hidden_layers', [(128, 64), (256, 128, 64), (512, 256, 128)]
                    )
                ],
                nn_epochs=ml_config.get('nn_epochs', 100),
                nn_batch_size=ml_config.get('nn_batch_size', 64),
                nn_learning_rate=ml_config.get('nn_learning_rate', 0.001),

                # LSTM architecture
                lstm_hidden_size=ml_config.get('lstm_hidden_size', 128),
                lstm_num_layers=ml_config.get('lstm_num_layers', 2),
                lstm_dropout=ml_config.get('lstm_dropout', 0.2),
                lstm_sequence_length=ml_config.get('lstm_sequence_length', 100),
                lstm_epochs=ml_config.get('lstm_epochs', 250),
                lstm_batch_size=ml_config.get('lstm_batch_size', 32),
            )

            if not symbols:
                raise ValueError("No symbols provided in job")

            # Log the active ML configuration for visibility
            logger.info(f"[{job_id}] Processing job: {symbols}, {years}y, "
                        f"{ml_training_config.n_iter_search} HP iters")
            logger.info(f"[{job_id}] Build: {build_id}, User: {user_id}")
            logger.info(f"[{job_id}] ML Config from design:")
            logger.info(f"[{job_id}]   Forward days: {ml_training_config.forward_returns_days_options}")
            logger.info(f"[{job_id}]   Profit thresholds: {ml_training_config.profit_threshold_options}")
            logger.info(f"[{job_id}]   HP search iters: {ml_training_config.n_iter_search}")
            logger.info(f"[{job_id}]   Models: MLP={'ON' if ml_training_config.train_neural_networks else 'OFF'}, "
                        f"LSTM={'ON' if ml_training_config.train_lstm else 'OFF'}")
            logger.info(f"[{job_id}]   Config search model: {ml_training_config.config_search_model}")
            reporter.report_phase("job_received", "complete")

            # Check for stop signal before starting training
            if self._check_stop_signal(build_id):
                logger.info(f"[{job_id}] Build stopped before training")
                return {
                    "status": "stopped",
                    "message": "Build stopped by user before training",
                    "design_config": design
                }

            # Step 1: Train ML models using the design-driven config
            reporter.report_phase(
                "training", "in_progress",
                message="Starting model training pipeline — this will take a few minutes...",
            )
            trainer = EnhancedMLTrainer(
                symbols=symbols,
                period_years=years,
                config=ml_training_config,
                timeframe=timeframe,
            )

            # Build a callback that pipes granular ML pipeline sub-stages back
            # to the UI Design Thinking section via the progress reporter.
            def _training_progress(message: str) -> None:
                reporter.report_phase("training", "in_progress", message=message)

            # Skip disk writes — worker sends results via Redis, and hardcoded
            # file paths (model .pkl, strategy config .json) don't exist here.
            training_results = await trainer.run_full_pipeline(
                skip_disk_writes=True,
                progress_callback=_training_progress,
            )
            reporter.report_phase("training", "complete")

            # Extract model metrics
            best_model_name = training_results.get('best_model', 'unknown')
            metrics = training_results.get('metrics', {})

            logger.info(f"[{job_id}] Training complete: {best_model_name}, F1={metrics.get('f1', 0):.4f}")
            reporter.report_metrics(metrics)

            # Step 2: Extract trading rules
            reporter.report_phase(
                "rule_extraction", "in_progress",
                message="Analysing feature importance and extracting trading rules from trained models...",
            )

            # Create rule extractor with training data from trainer
            rule_extractor = RuleExtractor(
                model=trainer.best_model,
                scaler=trainer.scaler,
                feature_names=trainer.feature_names,
                X_train=trainer.X_train,
                y_train=trainer.y_train,
                X_test=trainer.X_test,
                y_test=trainer.y_test
            )

            strategy_rules = rule_extractor.extract_all_rules(
                strategy_name=f"{strategy_type}_{job_id}",
                symbols=symbols,
                timeframe=timeframe,
                model_metrics=metrics
            )
            reporter.report_phase("rule_extraction", "complete")

            # Step 3: Generate strategy files
            # Wrapped in try/except — file generation failure should NOT crash the build
            strategy_files = {}
            try:
                reporter.report_phase(
                    "file_generation", "in_progress",
                    message="Generating strategy files from extracted rules...",
                )

                # Serialise extracted rules to a dict for the file generator
                rules_dict = rule_extractor.to_dict(strategy_rules)

                # Build model metrics dict matching the format expected by the generator
                file_gen_metrics = {
                    "model_name": best_model_name,
                    "f1": metrics.get("f1", 0),
                    "precision": metrics.get("precision", 0),
                    "recall": metrics.get("recall", 0),
                    "auc": metrics.get("auc", 0),
                }

                generator = StrategyFileGenerator(
                    strategy_name=f"{strategy_type}_{job_id}",
                    symbols=symbols,
                    timeframe=timeframe,
                    model_metrics=file_gen_metrics,
                    extracted_rules=rules_dict,
                    design=design,
                )

                strategy_files = generator.generate_all()
                logger.info(
                    f"[{job_id}] Generated {len(strategy_files)} strategy files: "
                    f"{list(strategy_files.keys())}"
                )
                reporter.report_phase("file_generation", "complete")

            except Exception as file_gen_err:
                # Log full traceback so the root cause is visible in worker logs.
                logger.exception(
                    f"[{job_id}] Strategy file generation failed (non-fatal): "
                    f"{file_gen_err}"
                )
                reporter.report_phase(
                    "file_generation", "error",
                    details={"error": str(file_gen_err)},
                )

            # Write generated strategy files to a temp directory so the optimizer
            # and backtester can import and execute them as real Python modules.
            strategy_temp_dir = None
            optimization_results = {}
            backtest_results = {
                "status": "pending",
                "note": "Backtesting requires generated strategy files",
            }

            try:
                if strategy_files:
                    strategy_temp_dir = tempfile.mkdtemp(prefix="oculus_strategy_")
                    for filename, content in strategy_files.items():
                        file_path = Path(strategy_temp_dir) / filename
                        file_path.write_text(content)
                    logger.info(
                        f"[{job_id}] Wrote {len(strategy_files)} files to "
                        f"{strategy_temp_dir}"
                    )

                    # Step 4: Optimize strategy parameters
                    try:
                        reporter.report_phase(
                            "optimization", "in_progress",
                            message="Optimizing strategy parameters (stop-loss, position sizing, entry thresholds)...",
                        )

                        optimizer = StrategyOptimizer(strategy_temp_dir)

                        # Build LLM-guided optimization ranges (mirrors strategy_builder.py)
                        sl = design.get("recommended_stop_loss", 0.05)
                        ts = design.get("recommended_trailing_stop", 0.03)
                        mp = design.get("recommended_max_positions", 3)
                        est = design.get("entry_score_threshold", 3)

                        # Clamp helpers for realistic bounds
                        def clamp_sl(v):
                            return max(0.02, min(0.15, round(v, 3)))

                        def clamp_ts(v):
                            return max(0.01, min(0.10, round(v, 3)))

                        stop_loss_range = sorted(set([
                            clamp_sl(sl * 0.5), clamp_sl(sl * 0.75),
                            clamp_sl(sl), clamp_sl(sl * 1.25), clamp_sl(sl * 1.5),
                        ]))
                        trailing_stop_range = sorted(set([
                            clamp_ts(ts * 0.5), clamp_ts(ts * 0.75),
                            clamp_ts(ts), clamp_ts(ts * 1.25), clamp_ts(ts * 1.5),
                        ]))
                        max_positions_range = sorted(set([
                            max(1, mp - 1), mp, min(5, mp + 1),
                        ]))
                        min_entry_score_range = [
                            max(1, est - 1), est, est + 1, est + 2,
                        ]

                        logger.info(
                            f"[{job_id}] Optimization ranges — "
                            f"SL: {stop_loss_range}, TS: {trailing_stop_range}, "
                            f"MaxPos: {max_positions_range}, MinScore: {min_entry_score_range}"
                        )

                        best = optimizer.optimize(
                            stop_loss_range=stop_loss_range,
                            trailing_stop_range=trailing_stop_range,
                            max_positions_range=max_positions_range,
                            min_entry_score_range=min_entry_score_range,
                            n_iterations=50,
                        )

                        if best:
                            optimizer.apply_best_params(best)
                            optimization_results = {
                                "status": "success",
                                "best_params": best.params,
                                "best_return": best.total_return,
                                "best_drawdown": best.max_drawdown,
                                "best_score": best.score,
                                "total_trials": len(optimizer.results),
                            }
                            logger.info(
                                f"[{job_id}] Optimization complete — "
                                f"Return: {best.total_return:.2f}%, "
                                f"Drawdown: {best.max_drawdown:.2f}%"
                            )
                        else:
                            optimization_results = {
                                "status": "no_results",
                                "note": "Optimizer did not find valid results",
                            }
                            logger.warning(
                                f"[{job_id}] Optimization found no valid results"
                            )

                        reporter.report_phase("optimization", "complete")

                    except Exception as opt_err:
                        logger.warning(
                            f"[{job_id}] Optimization failed (non-fatal): {opt_err}"
                        )
                        optimization_results = {
                            "status": "error",
                            "error": str(opt_err),
                        }
                        reporter.report_phase(
                            "optimization", "error",
                            details={"error": str(opt_err)},
                        )

                    # Step 5: Run backtest with real PortfolioBacktester
                    try:
                        reporter.report_phase(
                            "backtesting", "in_progress",
                            message="Running backtest with optimized parameters on historical data...",
                        )

                        # Clear cached modules so we import from the temp dir
                        modules_to_remove = [
                            m for m in sys.modules
                            if "backtest" in m or "strategy" in m.lower()
                            or "data_provider" in m.lower()
                            or "risk_manager" in m.lower()
                        ]
                        for m in modules_to_remove:
                            del sys.modules[m]

                        # Import PortfolioBacktester from the generated backtest.py
                        backtest_path = Path(strategy_temp_dir) / "backtest.py"
                        spec = importlib.util.spec_from_file_location(
                            "backtest", str(backtest_path)
                        )
                        backtest_mod = importlib.util.module_from_spec(spec)
                        sys.modules["backtest"] = backtest_mod

                        # Ensure the temp dir is on sys.path so intra-strategy
                        # imports (strategy, data_provider, etc.) resolve correctly
                        if strategy_temp_dir not in sys.path:
                            sys.path.insert(0, strategy_temp_dir)

                        spec.loader.exec_module(backtest_mod)
                        PortfolioBacktester = backtest_mod.PortfolioBacktester

                        backtester = PortfolioBacktester(initial_capital=100000.0)
                        bt_results = backtester.run()

                        # Remove temp dir from sys.path
                        if strategy_temp_dir in sys.path:
                            sys.path.remove(strategy_temp_dir)

                        if bt_results:
                            backtest_results = {
                                "status": "success",
                                "total_return": bt_results.get("total_return", 0),
                                "max_drawdown": bt_results.get("max_drawdown", 0),
                                "win_rate": bt_results.get("win_rate", 0),
                                "total_trades": bt_results.get("total_trades", 0),
                                "sharpe_ratio": bt_results.get("sharpe_ratio", 0),
                                "profit_factor": bt_results.get("profit_factor", 0),
                                "avg_trade_return": bt_results.get("avg_trade_return", 0),
                                "avg_win": bt_results.get("avg_win", 0),
                                "avg_loss": bt_results.get("avg_loss", 0),
                                "buy_hold_avg": bt_results.get("buy_hold_avg", 0),
                                "max_potential": bt_results.get("max_potential", 0),
                                # Extended fields required by backtest_results.json schema
                                "avg_trade_pnl": bt_results.get("avg_trade_pnl", 0.0),
                                "best_trade_pnl": bt_results.get("best_trade_pnl", 0.0),
                                "worst_trade_pnl": bt_results.get("worst_trade_pnl", 0.0),
                                "avg_trade_duration_hours": bt_results.get("avg_trade_duration_hours", 0.0),
                                "trades": bt_results.get("trades", []),
                                "equity_curve": bt_results.get("equity_curve", []),
                                "daily_returns": bt_results.get("daily_returns", []),
                                "weekly_returns": bt_results.get("weekly_returns", []),
                                "monthly_returns": bt_results.get("monthly_returns", []),
                                "yearly_returns": bt_results.get("yearly_returns", []),
                            }
                            # Calculate capture rate
                            max_potential = bt_results.get("max_potential", 1)
                            total_return = bt_results.get("total_return", 0)
                            if max_potential > 0:
                                backtest_results["capture_rate"] = (
                                    total_return / max_potential
                                ) * 100
                            else:
                                backtest_results["capture_rate"] = 0

                            logger.info(
                                f"[{job_id}] Backtest complete — "
                                f"Return: {total_return:.2f}%, "
                                f"Trades: {bt_results.get('total_trades', 0)}"
                            )
                        else:
                            backtest_results = {
                                "status": "no_results",
                                "note": "Backtester returned no results",
                            }

                        reporter.report_phase("backtesting", "complete")

                    except Exception as bt_err:
                        logger.warning(
                            f"[{job_id}] Backtest failed (non-fatal): {bt_err}"
                        )
                        backtest_results = {
                            "status": "error",
                            "error": str(bt_err),
                        }
                        reporter.report_phase(
                            "backtesting", "error",
                            details={"error": str(bt_err)},
                        )

                    # Re-read files from temp dir — config.json may have been
                    # updated by the optimizer's apply_best_params()
                    for filename in list(strategy_files.keys()):
                        file_path = Path(strategy_temp_dir) / filename
                        if file_path.exists():
                            strategy_files[filename] = file_path.read_text()

                else:
                    # No strategy files generated — skip optimization & backtest
                    reporter.report_phase("optimization", "complete")
                    reporter.report_phase("backtesting", "complete")
                    logger.info(
                        f"[{job_id}] Skipping optimization/backtest — "
                        f"no strategy files generated"
                    )

            finally:
                # Clean up temp directory
                if strategy_temp_dir and Path(strategy_temp_dir).exists():
                    try:
                        shutil.rmtree(strategy_temp_dir)
                        logger.info(
                            f"[{job_id}] Cleaned up temp dir: {strategy_temp_dir}"
                        )
                    except Exception as cleanup_err:
                        logger.warning(
                            f"[{job_id}] Temp dir cleanup failed: {cleanup_err}"
                        )

            # Collect comprehensive training data from trainer for BuildIteration persistence
            # Label configuration results (optimal forward_days + profit_threshold from search)
            optimal_label_config = None
            if trainer.best_config:
                optimal_label_config = {
                    "forward_days": trainer.best_config.get("forward_days"),
                    "profit_threshold": trainer.best_config.get("profit_threshold"),
                    "f1_score": trainer.best_config.get("f1", 0),
                }

            # Feature data (names + importances from best model)
            features_data = {
                "feature_names": trainer.feature_names if trainer.feature_names else [],
                "count": len(trainer.feature_names) if trainer.feature_names else 0,
            }
            if trainer.best_model and hasattr(trainer.best_model, 'feature_importances_'):
                importances = trainer.best_model.feature_importances_
                feature_importance = {}
                for i, name in enumerate(trainer.feature_names):
                    feature_importance[name] = float(importances[i])
                # Sort by importance descending, keep top 20
                sorted_features = sorted(
                    feature_importance.items(), key=lambda x: x[1], reverse=True
                )[:20]
                features_data["top_features"] = [
                    {"name": n, "importance": v} for n, v in sorted_features
                ]

            # All model evaluations (metrics for every model that was trained)
            model_evaluations = {}
            if isinstance(trainer.all_results, dict):
                for name, data in trainer.all_results.items():
                    if isinstance(data, dict):
                        model_evaluations[name] = data.get('metrics', {})

            # Best model info
            best_model_info = {
                "name": best_model_name,
                "metrics": metrics,
                "type": type(trainer.best_model).__name__ if trainer.best_model else "unknown",
            }

            # Build result matching backend expectations
            # Existing fields are preserved for backward compatibility;
            # new fields are additive — old consumers simply ignore them.
            result = {
                "build_id": build_id,
                "iteration_uuid": iteration_uuid,  # For BuildIteration DB persistence
                "status": "success",
                "phase": "complete",
                "timeframe": timeframe,  # Trading timeframe used for training/backtesting

                # --- Existing fields (backward compatible) ---
                "model_metrics": {
                    "model_name": best_model_name,
                    "f1": metrics.get('f1', 0),
                    "precision": metrics.get('precision', 0),
                    "recall": metrics.get('recall', 0),
                    "auc": metrics.get('auc', 0),
                },
                "extracted_rules": {
                    "entry_rules": [
                        {
                            "feature": r.feature,
                            "operator": r.operator,
                            "threshold": r.threshold,
                            "importance": r.importance,
                            "description": r.description
                        }
                        for r in strategy_rules.entry_rules
                    ],
                    "exit_rules": [
                        {
                            "feature": r.feature,
                            "operator": r.operator,
                            "threshold": r.threshold,
                            "importance": r.importance,
                            "description": r.description
                        }
                        for r in strategy_rules.exit_rules
                    ],
                    "top_features": strategy_rules.top_features,
                },
                "backtest_results": backtest_results,
                "optimization_results": optimization_results,
                # Include the LLM's design config so the backend can display what was chosen
                "design_config": {
                    "forward_returns_days_options": ml_training_config.forward_returns_days_options,
                    "profit_threshold_options": ml_training_config.profit_threshold_options,
                    "n_iter_search": ml_training_config.n_iter_search,
                    "cv_folds": ml_training_config.cv_folds,
                    "config_search_model": ml_training_config.config_search_model,
                    "train_neural_networks": ml_training_config.train_neural_networks,
                    "train_lstm": ml_training_config.train_lstm,
                },

                # --- NEW comprehensive fields for BuildIteration persistence ---
                # Optimal label configuration found during config search
                "optimal_label_config": optimal_label_config,

                # Feature names and top importances from the best model
                "features": features_data,

                # Hyperparameter tuning results for tree-based / sklearn models
                "hyperparameter_results": {
                    name: {
                        "best_f1_cv": data.get('metrics', {}).get('f1', 0),
                        "params": data.get('params', {}),
                    }
                    for name, data in (trainer.all_results or {}).items()
                    if isinstance(data, dict) and not name.startswith(('MLP_', 'LSTM_'))
                } if isinstance(trainer.all_results, dict) else {},

                # Neural network (MLP) training results
                "nn_training_results": {
                    name: data.get('metrics', {})
                    for name, data in (trainer.all_results or {}).items()
                    if isinstance(data, dict) and name.startswith('MLP_')
                } if isinstance(trainer.all_results, dict) else {},

                # LSTM training results
                "lstm_training_results": {
                    name: data.get('metrics', {})
                    for name, data in (trainer.all_results or {}).items()
                    if isinstance(data, dict) and name.startswith('LSTM_')
                } if isinstance(trainer.all_results, dict) else {},

                # All model evaluations (every model's metrics)
                "model_evaluations": model_evaluations,

                # Best model summary
                "best_model": best_model_info,

                # --- Generated strategy files (filename → content string) ---
                # Backend will persist these to the strategy directory on disk.
                # Empty dict if file generation failed (non-fatal).
                "strategy_files": strategy_files,
            }

            # Diagnostic: confirm strategy_files are present in the outgoing result.
            sf = result.get("strategy_files") or {}
            if sf:
                total_bytes = sum(len(v) for v in sf.values())
                logger.info(
                    f"[{job_id}] Returning {len(sf)} strategy files "
                    f"({total_bytes:,} bytes total): {list(sf.keys())}"
                )
            else:
                logger.warning(
                    f"[{job_id}] strategy_files is EMPTY in result — "
                    f"worker file generation may have failed silently. "
                    f"Backend will fall back to LLM regeneration."
                )

            logger.info(f"[{job_id}] Job complete")
            return result

        except Exception as e:
            logger.exception(f"[{job_id}] Job failed: {e}")
            reporter.report_error("processing", str(e))
            return {
                "build_id": build_id,
                "iteration_uuid": job_data.get('iteration_uuid'),  # For BuildIteration DB persistence
                "status": "error",
                "phase": "failed",
                "error": str(e)
            }

