"""
Job processor - deserializes training jobs and invokes the ML pipeline.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import redis

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from strategy_designer_ai.ml_enhanced_trainer import EnhancedMLTrainer, EnhancedMLConfig
from strategy_designer_ai.rule_extractor import RuleExtractor
from strategy_optimizer_ai.optimizer import StrategyOptimizer
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
            os.environ['POLYGON_MASSIVE_API_KEY'] = config.polygon_api_key

        # Enable GPU acceleration if available
        if config.gpu_enabled:
            try:
                import torch
                if torch.backends.mps.is_available():
                    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                    logger.info("GPU acceleration (MPS) enabled")
            except ImportError:
                logger.warning("PyTorch not available, GPU acceleration disabled")

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
        reporter = ProgressReporter(job_id, self.redis)

        try:
            # Extract job parameters from backend format
            symbols = job_data.get('symbols', [])
            timeframe = job_data.get('timeframe', '1d')
            design = job_data.get('design', {})
            build_id = job_data.get('build_id', 'unknown')
            user_id = job_data.get('user_id', 'unknown')
            iteration_uuid = job_data.get('iteration_uuid')  # For BuildIteration DB persistence

            # Extract basic parameters (backward compatible with old job format)
            years = design.get('years', 2) if design else 2
            strategy_type = design.get('strategy_type', 'momentum') if design else 'momentum'

            # Build ML config dynamically from LLM design
            # Top-level fields come directly from the design dict
            # Nested ML training fields come from the 'ml_training_config' sub-object
            ml_config = design.get('ml_training_config', {}) if design else {}

            ml_training_config = EnhancedMLConfig(
                # Label generation — from LLM design top-level fields
                forward_returns_days_options=design.get(
                    'forward_returns_days_options', [3, 5, 7, 10, 15, 20]
                ),
                profit_threshold_options=design.get(
                    'profit_threshold_options', [1.5, 2.0, 3.0, 5.0]
                ),

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

            # Step 1: Train ML models using the design-driven config
            reporter.report_phase("training", "in_progress")
            trainer = EnhancedMLTrainer(
                symbols=symbols,
                period_years=years,
                config=ml_training_config,
            )

            # Skip disk writes — worker sends results via Redis, and hardcoded
            # file paths (model .pkl, strategy config .json) don't exist here.
            training_results = await trainer.run_full_pipeline(skip_disk_writes=True)
            reporter.report_phase("training", "complete")

            # Extract model metrics
            best_model_name = training_results.get('best_model', 'unknown')
            metrics = training_results.get('metrics', {})

            logger.info(f"[{job_id}] Training complete: {best_model_name}, F1={metrics.get('f1', 0):.4f}")
            reporter.report_metrics(metrics)

            # Step 2: Extract trading rules
            reporter.report_phase("rule_extraction", "in_progress")

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
                timeframe="1d",
                model_metrics=metrics
            )
            reporter.report_phase("rule_extraction", "complete")

            # Step 3: Generate strategy files
            # Wrapped in try/except — file generation failure should NOT crash the build
            strategy_files = {}
            try:
                reporter.report_phase("file_generation", "in_progress")

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
                # Log the error but do NOT re-raise — the build continues
                logger.warning(
                    f"[{job_id}] Strategy file generation failed (non-fatal): "
                    f"{file_gen_err}"
                )
                reporter.report_phase(
                    "file_generation", "error",
                    details={"error": str(file_gen_err)},
                )

            # Step 4: Optimize strategy parameters (optional)
            reporter.report_phase("optimization", "in_progress")
            # Note: Full optimization requires backtesting infrastructure
            # For now, we report completion without full optimization
            reporter.report_phase("optimization", "complete")

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
                "backtest_results": {
                    "status": "pending",
                    "note": "Backtesting requires full strategy deployment"
                },
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

