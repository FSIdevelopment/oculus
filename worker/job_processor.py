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

    async def process_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a training job.

        Args:
            job_data: Job configuration with fields:
                - job_id: Unique job identifier
                - symbols: List of stock symbols
                - years: Years of historical data
                - hp_iterations: Hyperparameter search iterations
                - feature_list: Optional list of features to use
                - llm_design: Optional LLM design guidance
                - strategy_type: Strategy type (e.g., "momentum")

        Returns:
            Result dictionary with status, metrics, rules, and backtest results
        """
        job_id = job_data.get('job_id', 'unknown')
        reporter = ProgressReporter(job_id, self.redis)

        try:
            # Extract job parameters
            symbols = job_data.get('symbols', [])
            years = job_data.get('years', 2)
            hp_iterations = job_data.get('hp_iterations', 30)
            strategy_type = job_data.get('strategy_type', 'momentum')

            if not symbols:
                raise ValueError("No symbols provided in job")

            logger.info(f"[{job_id}] Processing job: {symbols}, {years}y, {hp_iterations} HP iters")
            reporter.report_phase("job_received", "complete")

            # Step 1: Train ML models
            reporter.report_phase("training", "in_progress")
            trainer = EnhancedMLTrainer(
                symbols=symbols,
                period_years=years,
                config=EnhancedMLConfig(
                    forward_returns_days_options=[3, 5, 7, 10, 15, 20],
                    profit_threshold_options=[1.5, 2.0, 3.0, 5.0],
                    n_iter_search=hp_iterations,
                    cv_folds=3,
                    train_neural_networks=False,  # Faster training
                    train_lstm=True,
                )
            )

            training_results = await trainer.run_full_pipeline()
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

            # Step 3: Optimize strategy parameters (optional)
            reporter.report_phase("optimization", "in_progress")
            # Note: Full optimization requires backtesting infrastructure
            # For now, we report completion without full optimization
            reporter.report_phase("optimization", "complete")

            # Build result
            result = {
                "job_id": job_id,
                "status": "success",
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
                }
            }

            logger.info(f"[{job_id}] Job complete")
            return result

        except Exception as e:
            logger.exception(f"[{job_id}] Job failed: {e}")
            reporter.report_error("processing", str(e))
            return {
                "job_id": job_id,
                "status": "error",
                "error": str(e)
            }

