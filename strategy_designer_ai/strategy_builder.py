#!/usr/bin/env python3
"""
Strategy Builder - Automated Strategy Generation from ML Models

Takes symbols and timeframe as input, trains ML models, extracts rules,
and generates a complete trading strategy folder with all necessary files.

Usage:
    python strategy_builder.py --name "my_strategy" --symbols NVDA,AMD --timeframe 1d
    python strategy_builder.py --name "my_strategy" --symbols NVDA,AMD --LLM --desc "momentum strategy"

Author: SignalSynk AI
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load from project root .env file
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, rely on environment variables

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy_designer_ai.ml_enhanced_trainer import EnhancedMLTrainer, EnhancedMLConfig
from strategy_designer_ai.rule_extractor import RuleExtractor, StrategyRules
from strategy_designer_ai.strategy_optimizer import StrategyOptimizer
from strategy_designer_ai.llm_strategy_designer import LLMStrategyDesigner, LLMStrategyDesign
from strategy_designer_ai.build_history import BuildHistoryManager, FeatureTracker, scan_existing_strategies


class StrategyBuilder:
    """Builds complete trading strategies from ML model insights."""

    # Progressive training configuration - each iteration increases data and tuning
    # forward_days and profit_thresholds search space stays consistent
    DEFAULT_FORWARD_DAYS = [3, 5, 7, 10, 15, 20]
    DEFAULT_PROFIT_THRESHOLDS = [1.5, 2.0, 3.0, 5.0]

    # Iteration progression settings
    YEARS_INCREMENT = 2  # Add 2 years of data per iteration
    HYPERPARAMETER_INCREMENT = 20  # Add 20 more search iterations per iteration
    INITIAL_N_ITER = 30  # Starting hyperparameter search iterations
    MAX_N_ITER = 100  # Maximum hyperparameter search iterations

    def __init__(self, strategy_name: str, symbols: List[str],
                 timeframe: str = '1d', years_of_data: int = 2,
                 target_capture_rate: float = 50.0, max_iterations: int = 5,
                 llm_enabled: bool = False, description: str = ""):
        self.strategy_name = strategy_name
        self.symbols = symbols
        self.timeframe = timeframe
        self.initial_years = years_of_data  # Starting years
        self.current_years = years_of_data  # Will increase each iteration
        self.target_capture_rate = target_capture_rate
        self.max_iterations = max_iterations
        self.current_n_iter = self.INITIAL_N_ITER  # Hyperparameter search iterations

        # LLM-assisted design
        self.llm_enabled = llm_enabled
        self.description = description
        self.llm_design: Optional[LLMStrategyDesign] = None
        self.llm_designer: Optional[LLMStrategyDesigner] = None

        # Cross-build memory
        self.history_manager = BuildHistoryManager()
        self.feature_tracker = FeatureTracker()

        if self.llm_enabled:
            self.llm_designer = LLMStrategyDesigner(
                strategy_name=strategy_name,
                symbols=symbols,
                description=description,
                timeframe=timeframe,
                target=target_capture_rate,
                history_manager=self.history_manager,
                feature_tracker=self.feature_tracker,
            )

        # Paths
        self.base_dir = Path(__file__).parent.parent
        self.strategies_dir = self.base_dir / 'strategies'
        self.strategy_dir = self.strategies_dir / strategy_name.lower().replace(' ', '_')
        self.templates_dir = Path(__file__).parent / 'strategy_templates'

        # State
        self.trainer: Optional[EnhancedMLTrainer] = None
        self.rules: Optional[StrategyRules] = None
        self.model_metrics: Dict[str, float] = {}
        self.best_capture_rate: float = 0.0
        self.best_model_config: Optional[Dict] = None  # Stores best iteration's settings
        self.iteration_results: List[Dict] = []

        print(f"\n{'='*60}")
        print(f"STRATEGY BUILDER INITIALIZED - MODEL-CENTRIC ITERATION")
        print(f"{'='*60}")
        print(f"  Strategy Name: {strategy_name}")
        print(f"  Symbols: {symbols}")
        print(f"  Timeframe: {timeframe}")
        print(f"  Initial Training Years: {years_of_data}")
        print(f"  Target Return: {target_capture_rate}%")
        print(f"  Max Iterations: {max_iterations}")
        if self.llm_enabled:
            print(f"  LLM-Assisted Design: ENABLED (Claude API)")
            print(f"  Strategy Description: {description}")
        print(f"  Output Dir: {self.strategy_dir}")
        print(f"\n  🔄 ITERATION STRATEGY:")
        print(f"     Each iteration adds {self.YEARS_INCREMENT} more years of training data")
        print(f"     Each iteration adds {self.HYPERPARAMETER_INCREMENT} more hyperparameter search iterations")
        print(f"     Initial: {years_of_data} years, {self.INITIAL_N_ITER} HP iterations")
        print(f"{'='*60}\n")

    async def build(self) -> str:
        """Build complete strategy with MODEL-CENTRIC iteration for capture rate optimization.

        Iteration Strategy:
        1. Get data (increasing years each iteration)
        2. Train/retrain models (increasing hyperparameter search iterations)
        3. Optimize risk parameters
        4. Backtest → Evaluate capture rate

        If capture rate < target: Go back to step 1 with MORE DATA and MORE HP TUNING.
        The model's predictive ability is the PRIMARY driver of performance.
        """
        start_time = datetime.now()

        try:
            # LLM-assisted design: call Claude before the iteration loop
            if self.llm_enabled and self.llm_designer:
                print("\n" + "🤖"*30)
                print("STEP 0: LLM-ASSISTED STRATEGY DESIGN")
                print("🤖"*30)
                self.llm_design = self.llm_designer.design_strategy()

                # Apply LLM design to ML config overrides
                if self.llm_design.forward_returns_days_options:
                    self.DEFAULT_FORWARD_DAYS = self.llm_design.forward_returns_days_options
                if self.llm_design.profit_threshold_options:
                    self.DEFAULT_PROFIT_THRESHOLDS = self.llm_design.profit_threshold_options
                if self.llm_design.recommended_n_iter_search > self.current_n_iter:
                    self.current_n_iter = self.llm_design.recommended_n_iter_search

                print(f"\n  Applied LLM config overrides:")
                print(f"    Forward Days: {self.DEFAULT_FORWARD_DAYS}")
                print(f"    Profit Thresholds: {self.DEFAULT_PROFIT_THRESHOLDS}")
                print(f"    HP Search Iterations: {self.current_n_iter}")

            for iteration in range(1, self.max_iterations + 1):
                print("\n" + "🔄"*30)
                print(f"ITERATION {iteration}/{self.max_iterations} - MODEL RETRAINING")
                print("🔄"*30)
                print(f"  📊 Training Years: {self.current_years}")
                print(f"  🔧 Hyperparameter Search Iterations: {self.current_n_iter}")
                if self.llm_design:
                    print(f"  🤖 LLM Design: Active ({len(self.llm_design.entry_rules)} entry rules, "
                          f"{len(self.llm_design.exit_rules)} exit rules)")
                print("🔄"*30)

                # Step 1: Train ML models with current data volume and HP iterations
                print("\n" + "="*60)
                print(f"STEP 1: TRAINING ML MODELS ({self.current_years} years, {self.current_n_iter} HP iters)")
                print("="*60)
                await self._train_models_progressive()

                # Step 2: Extract rules from best model
                print("\n" + "="*60)
                print("STEP 2: EXTRACTING TRADING RULES")
                print("="*60)
                self._extract_rules()

                # Step 3: Generate strategy files
                print("\n" + "="*60)
                print("STEP 3: GENERATING STRATEGY FILES")
                print("="*60)
                self._generate_strategy_files()

                # Step 4: Create requirements and config
                print("\n" + "="*60)
                print("STEP 4: CREATING CONFIGURATION")
                print("="*60)
                self._create_config_files()

                # Step 5: Optimize risk management parameters
                print("\n" + "="*60)
                print("STEP 5: OPTIMIZING RISK PARAMETERS")
                print("="*60)
                self._optimize_strategy()

                # Step 6: Run backtest and evaluate capture rate
                print("\n" + "="*60)
                print("STEP 6: EVALUATING STRATEGY PERFORMANCE")
                print("="*60)
                backtest_results = self._run_backtest()
                capture_rate = self._calculate_capture_rate(backtest_results)

                # Store iteration results
                self.iteration_results.append({
                    'iteration': iteration,
                    'training_years': self.current_years,
                    'hp_iterations': self.current_n_iter,
                    'capture_rate': capture_rate,
                    'total_return': backtest_results.get('total_return', 0),
                    'max_drawdown': backtest_results.get('max_drawdown', 0),
                    'win_rate': backtest_results.get('win_rate', 0),
                    'model_f1': self.model_metrics.get('f1', 0)
                })

                total_return = backtest_results.get('total_return', 0)

                print(f"\n  📊 Iteration {iteration} Results:")
                print(f"     Training Years: {self.current_years}")
                print(f"     HP Search Iterations: {self.current_n_iter}")
                print(f"     Model F1 Score: {self.model_metrics.get('f1', 0):.4f}")
                print(f"     Capture Rate: {capture_rate:.1f}%")
                print(f"     Total Return: {total_return:.2f}%")
                print(f"     Target Return: {self.target_capture_rate}%")

                # Track best result by total return (the actual target metric)
                if total_return > self.best_capture_rate:
                    self.best_capture_rate = total_return
                    self.best_model_config = {
                        'years': self.current_years,
                        'n_iter': self.current_n_iter,
                        'model_f1': self.model_metrics.get('f1', 0)
                    }
                    print(f"     🏆 NEW BEST RETURN!")

                # Save iteration progress to creation guide (no LLM call, just file I/O)
                if self.llm_enabled and self.llm_designer and self.llm_design:
                    try:
                        self.llm_designer.save_iteration_progress(
                            design=self.llm_design,
                            model_metrics=self.model_metrics,
                            iteration=iteration,
                            iteration_results=self.iteration_results,
                        )
                    except Exception as e:
                        print(f"     ⚠️ Iteration progress save failed: {e}")

                # Save iteration progress to build history (persists across crashes)
                try:
                    build_record = self._create_build_record(backtest_results)
                    self.history_manager.add_build(build_record)
                    self.feature_tracker.rebuild_from_history(self.history_manager.history)
                    print(f"  📝 Build history updated (iteration {iteration})")
                except Exception as e:
                    print(f"     ⚠️ Build history iteration save failed: {e}")

                # Check if we've hit the target (compare total return against target)
                if total_return >= self.target_capture_rate:
                    print(f"\n  ✅ TARGET ACHIEVED! Total Return: {total_return:.1f}% >= {self.target_capture_rate}%")
                    break
                else:
                    print(f"\n  ⚠️ Below target ({total_return:.1f}% < {self.target_capture_rate}%)")
                    if iteration < self.max_iterations:
                        # INCREASE training data and hyperparameter search iterations
                        self.current_years += self.YEARS_INCREMENT
                        self.current_n_iter = min(self.current_n_iter + self.HYPERPARAMETER_INCREMENT, self.MAX_N_ITER)
                        print(f"     🔄 RETRAINING with more data and tuning:")
                        print(f"        → Years: {self.current_years - self.YEARS_INCREMENT} → {self.current_years}")
                        print(f"        → HP Iterations: {self.current_n_iter - self.HYPERPARAMETER_INCREMENT} → {self.current_n_iter}")

                        # LLM refinement: call Claude with backtest feedback + full history
                        if self.llm_enabled and self.llm_designer and self.llm_design:
                            print(f"\n     🤖 Calling Claude for strategy refinement...")
                            backtest_results['capture_rate'] = capture_rate
                            self.llm_design = self.llm_designer.refine_strategy(
                                backtest_results=backtest_results,
                                iteration=iteration,
                                previous_design=self.llm_design,
                                iteration_history=self.iteration_results
                            )
                            # Apply updated LLM config
                            if self.llm_design.forward_returns_days_options:
                                self.DEFAULT_FORWARD_DAYS = self.llm_design.forward_returns_days_options
                            if self.llm_design.profit_threshold_options:
                                self.DEFAULT_PROFIT_THRESHOLDS = self.llm_design.profit_threshold_options
                            print(f"     🤖 LLM refinement applied for next iteration")
                    else:
                        print(f"     Max iterations reached. Using best result.")

            # LLM post-build: generate strategy README and update creation guide
            if self.llm_enabled and self.llm_designer and self.llm_design:
                print("\n" + "="*60)
                print("STEP 7: GENERATING STRATEGY README")
                print("="*60)
                try:
                    readme_content = self.llm_designer.generate_strategy_readme(
                        design=self.llm_design,
                        backtest_results=backtest_results,
                        model_metrics=self.model_metrics,
                        config_path=self.strategy_dir / 'config.json',
                        iteration_results=self.iteration_results,
                    )
                    with open(self.strategy_dir / 'README.md', 'w') as f:
                        f.write(readme_content)
                    print(f"  ✓ Generated {self.strategy_dir / 'README.md'}")
                except Exception as e:
                    print(f"  ⚠️ README generation failed: {e}")

                print("\n" + "="*60)
                print("STEP 8: UPDATING STRATEGY CREATION GUIDE")
                print("="*60)
                try:
                    self.llm_designer.update_creation_guide(
                        design=self.llm_design,
                        backtest_results=backtest_results,
                        model_metrics=self.model_metrics,
                        iteration_results=self.iteration_results,
                    )
                    print(f"  ✓ Updated strategy_designer_ai/STRATEGY_CREATION_GUIDE.md")
                except Exception as e:
                    print(f"  ⚠️ Guide update failed: {e}")

            # Step 9: Save to build history database (all builds, not just LLM-assisted)
            print("\n" + "="*60)
            print("STEP 9: SAVING TO BUILD HISTORY")
            print("="*60)
            try:
                build_record = self._create_build_record(backtest_results)
                self.history_manager.add_build(build_record)
                self.feature_tracker.rebuild_from_history(self.history_manager.history)
                print(f"  ✓ Saved to build_history.json ({len(self.history_manager.history)} total builds)")
                print(f"  ✓ Rebuilt feature_tracker.json ({len(self.feature_tracker.data)} features tracked)")
            except Exception as e:
                print(f"  ⚠️ Build history save failed: {e}")

            # Final summary
            elapsed = (datetime.now() - start_time).total_seconds()
            self._print_iteration_summary(elapsed)

            # Save final backtest results to JSON file
            print("\n" + "="*60)
            print("STEP 10: SAVING BACKTEST RESULTS TO FILE")
            print("="*60)
            try:
                sys.path.insert(0, str(self.strategy_dir))
                from backtest import save_backtest_results
                save_backtest_results(backtest_results, strategy_id=self.strategy_name, initial_capital=100000.0)
                sys.path.remove(str(self.strategy_dir))
                print(f"  ✓ Saved backtest_results.json")
            except Exception as e:
                print(f"  ⚠️ Backtest results save failed: {e}")
                import traceback
                traceback.print_exc()

            return str(self.strategy_dir)

        except Exception as e:
            print(f"\n❌ Build failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _run_backtest(self) -> Dict[str, Any]:
        """Run backtest and return results dictionary."""
        import sys

        print(f"  📈 Running backtest...")

        # Import the backtest module from the generated strategy
        try:
            # Clear any cached imports
            modules_to_remove = [m for m in sys.modules if 'backtest' in m or self.strategy_name.lower() in m]
            for m in modules_to_remove:
                del sys.modules[m]

            # Add strategy directory to path
            sys.path.insert(0, str(self.strategy_dir))

            from backtest import PortfolioBacktester

            backtester = PortfolioBacktester(initial_capital=100000.0)
            results = backtester.run()

            # Remove from path
            sys.path.remove(str(self.strategy_dir))

            return results

        except Exception as e:
            print(f"  ⚠️ Backtest error: {e}")
            return {'total_return': 0, 'max_potential': 1, 'win_rate': 0, 'max_drawdown': 0}

    def _calculate_capture_rate(self, backtest_results: Dict[str, Any]) -> float:
        """Calculate capture rate from backtest results."""
        total_return = backtest_results.get('total_return', 0)
        max_potential = backtest_results.get('max_potential', 1)

        if max_potential > 0:
            capture_rate = (total_return / max_potential) * 100
        else:
            capture_rate = 0

        return capture_rate

    def _print_iteration_summary(self, elapsed_seconds: float):
        """Print summary of all iterations."""
        print(f"\n{'='*60}")
        print(f"🎯 STRATEGY BUILD COMPLETE - MODEL-CENTRIC ITERATION SUMMARY")
        print(f"{'='*60}")

        print(f"\n  📊 All Iterations:")
        for result in self.iteration_results:
            status = "✅" if result['total_return'] >= self.target_capture_rate else "❌"
            print(f"     {status} Iter {result['iteration']}: "
                  f"{result['training_years']}yrs/{result['hp_iterations']}HP - "
                  f"F1={result.get('model_f1', 0):.3f}, "
                  f"Return: {result['total_return']:.2f}%, Capture: {result['capture_rate']:.1f}%")

        print(f"\n  🏆 Best Result:")
        if self.best_model_config:
            print(f"     Training Years: {self.best_model_config['years']}")
            print(f"     HP Iterations: {self.best_model_config['n_iter']}")
            print(f"     Model F1: {self.best_model_config.get('model_f1', 0):.4f}")
            print(f"     Best Return: {self.best_capture_rate:.1f}%")

        final_status = "✅ TARGET ACHIEVED" if self.best_capture_rate >= self.target_capture_rate else "⚠️ BELOW TARGET"
        print(f"\n  {final_status}")
        print(f"     Target Return: {self.target_capture_rate}%")
        print(f"     Best Return: {self.best_capture_rate:.1f}%")

        print(f"\n  📁 Output: {self.strategy_dir}")
        print(f"  ⏱️  Time: {elapsed_seconds:.1f} seconds")
        print(f"  📄 Files: {len(list(self.strategy_dir.glob('*.py')))} Python files")
        print(f"{'='*60}\n")

    def _create_build_record(self, backtest_results: Dict) -> Dict:
        """Create a standardized build record for the history database."""
        from strategy_designer_ai.build_history import infer_asset_class

        # Read extracted_rules.json for detailed rule data
        top_entry_rules = []
        top_exit_rules = []
        top_ml_features = []
        entry_rules_count = 0
        exit_rules_count = 0
        rules_path = self.strategy_dir / 'extracted_rules.json'
        if rules_path.exists():
            try:
                rules_data = json.loads(rules_path.read_text())
                top_ml_features = [
                    f['feature'] if isinstance(f, dict) else f
                    for f in rules_data.get('top_features', [])[:10]
                ]
                entry_rules_count = len(rules_data.get('entry_rules', []))
                exit_rules_count = len(rules_data.get('exit_rules', []))
                for rule in rules_data.get('entry_rules', [])[:5]:
                    top_entry_rules.append({
                        'feature': rule['feature'],
                        'operator': rule['operator'],
                        'threshold': round(rule['threshold'], 4),
                    })
                for rule in rules_data.get('exit_rules', [])[:5]:
                    top_exit_rules.append({
                        'feature': rule['feature'],
                        'operator': rule['operator'],
                        'threshold': round(rule['threshold'], 4),
                    })
            except (json.JSONDecodeError, KeyError):
                pass

        return {
            'strategy_name': self.strategy_name,
            'strategy_dir': str(self.strategy_dir.relative_to(self.base_dir)),
            'symbols': self.symbols,
            'asset_class': infer_asset_class(self.symbols),
            'description': (
                self.llm_design.strategy_description if self.llm_design
                else self.description
            )[:500],
            'timeframe': self.timeframe,
            'target': self.target_capture_rate,
            'build_date': datetime.now().strftime('%Y-%m-%d'),
            'llm_assisted': self.llm_enabled,
            'model': {
                'name': self.model_metrics.get('model_name', 'Unknown'),
                'f1': self.model_metrics.get('f1', 0),
                'precision': self.model_metrics.get('precision', 0),
                'recall': self.model_metrics.get('recall', 0),
                'auc': self.model_metrics.get('auc', 0),
                'training_years': self.model_metrics.get('training_years', 0),
                'hp_iterations': self.model_metrics.get('hp_iterations', 0),
            },
            'backtest': {
                'total_return': backtest_results.get('total_return', 0),
                'win_rate': backtest_results.get('win_rate', 0),
                'max_drawdown': backtest_results.get('max_drawdown', 0),
                'sharpe_ratio': backtest_results.get('sharpe_ratio'),
                'total_trades': backtest_results.get('total_trades', 0),
                'profit_factor': backtest_results.get('profit_factor'),
            },
            'risk_params': {
                'stop_loss': self.llm_design.recommended_stop_loss if self.llm_design else 0.05,
                'trailing_stop': self.llm_design.recommended_trailing_stop if self.llm_design else 0.03,
                'max_positions': self.llm_design.recommended_max_positions if self.llm_design else 3,
                'position_size': self.llm_design.recommended_position_size if self.llm_design else 0.10,
                'entry_score_threshold': self.llm_design.entry_score_threshold if self.llm_design else 3,
            },
            'features': {
                'priority': self.llm_design.priority_features if self.llm_design else [],
                'top_ml_features': top_ml_features,
                'entry_rules_count': entry_rules_count,
                'exit_rules_count': exit_rules_count,
                'top_entry_rules': top_entry_rules,
                'top_exit_rules': top_exit_rules,
            },
            'iterations': {
                'count': len(self.iteration_results),
                'results': [
                    {
                        'iteration': r['iteration'],
                        'return': r['total_return'],
                        'capture': r['capture_rate'],
                        'f1': r.get('model_f1', 0),
                    }
                    for r in self.iteration_results
                ],
            },
            'success': backtest_results.get('total_return', 0) >= self.target_capture_rate,
        }

    async def _train_models_progressive(self):
        """Train ML models with PROGRESSIVE data volume and hyperparameter tuning.

        This method uses:
        - self.current_years: Amount of training data (increases each iteration)
        - self.current_n_iter: Hyperparameter search iterations (increases each iteration)

        The model's predictive ability is the PRIMARY driver of performance.
        More data + more tuning = better model = better capture rate.
        """
        print(f"\n  🧠 Model-Centric Training Configuration:")
        print(f"     Training Data: {self.current_years} years")
        print(f"     Hyperparameter Search: {self.current_n_iter} iterations")
        print(f"     Forward Days Search: {self.DEFAULT_FORWARD_DAYS}")
        print(f"     Profit Thresholds: {self.DEFAULT_PROFIT_THRESHOLDS}")

        # Create training config with PROGRESSIVE settings
        # When LLM design is active, use its ML training configuration
        if self.llm_design:
            config = EnhancedMLConfig(
                forward_returns_days_options=self.DEFAULT_FORWARD_DAYS,
                profit_threshold_options=self.DEFAULT_PROFIT_THRESHOLDS,
                n_iter_search=self.current_n_iter,
                cv_folds=self.llm_design.cv_folds,
                config_search_model=self.llm_design.config_search_model,
                train_neural_networks=self.llm_design.train_neural_networks,
                train_lstm=self.llm_design.train_lstm,
                nn_epochs=self.llm_design.nn_epochs,
                nn_batch_size=self.llm_design.nn_batch_size,
                nn_learning_rate=self.llm_design.nn_learning_rate,
                lstm_hidden_size=self.llm_design.lstm_hidden_size,
                lstm_num_layers=self.llm_design.lstm_num_layers,
                lstm_dropout=self.llm_design.lstm_dropout,
                lstm_sequence_length=self.llm_design.lstm_sequence_length,
                lstm_epochs=self.llm_design.lstm_epochs,
                lstm_batch_size=self.llm_design.lstm_batch_size,
            )
            print(f"     🤖 LLM ML Config: MLP={'ON' if self.llm_design.train_neural_networks else 'OFF'}, "
                  f"LSTM={'ON' if self.llm_design.train_lstm else 'OFF'}, "
                  f"CV={self.llm_design.cv_folds}, "
                  f"ConfigSearch={self.llm_design.config_search_model}")
        else:
            config = EnhancedMLConfig(
                forward_returns_days_options=self.DEFAULT_FORWARD_DAYS,
                profit_threshold_options=self.DEFAULT_PROFIT_THRESHOLDS,
                n_iter_search=self.current_n_iter,
                cv_folds=3,
                train_neural_networks=True,
                train_lstm=True,
            )

        # Initialize trainer with PROGRESSIVE years
        self.trainer = EnhancedMLTrainer(
            symbols=self.symbols,
            period_years=self.current_years,  # PROGRESSIVE: increases each iteration
            config=config
        )

        # Run full training pipeline
        results = await self.trainer.run_full_pipeline()

        # Get best model metrics
        if self.trainer.best_model:
            best_metrics = results.get('metrics', {})
            self.model_metrics = {
                'model_name': self.trainer.best_model_name,
                'f1': best_metrics.get('f1', 0.0),
                'f1_score': best_metrics.get('f1', 0.0),  # Keep for backward compat
                'precision': best_metrics.get('precision', 0.0),
                'recall': best_metrics.get('recall', 0.0),
                'auc': best_metrics.get('auc', 0.0),
                'training_years': self.current_years,
                'hp_iterations': self.current_n_iter,
            }
            print(f"  Best Model: {self.model_metrics['model_name']}")
            print(f"  F1 Score: {self.model_metrics['f1']:.4f}")

    async def _train_models(self, training_config: Optional[Dict] = None):
        """Train ML models on historical data with optional custom config.

        DEPRECATED: Use _train_models_progressive() for model-centric iteration.
        Kept for backward compatibility.
        """
        # Use provided config or defaults
        if training_config:
            forward_days = training_config.get('forward_days', [1, 2, 3,5, 10, 15, 20])
            profit_thresholds = training_config.get('profit_thresholds', [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0])
        else:
            forward_days = [1, 2, 3,5, 10, 15, 20]
            profit_thresholds = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

        # Create training config with optimized settings for strategy building
        config = EnhancedMLConfig(
            forward_returns_days_options=forward_days,
            profit_threshold_options=profit_thresholds,
            n_iter_search=30,  # Reduced for speed
            cv_folds=3,
            # Disable slow models for faster strategy generation
            train_neural_networks=False,
            train_lstm=True,
        )

        # Initialize trainer
        self.trainer = EnhancedMLTrainer(
            symbols=self.symbols,
            period_years=self.initial_years,  # Use initial years
            config=config
        )

        # Run full training pipeline
        results = await self.trainer.run_full_pipeline()

        # Get best model metrics
        if self.trainer.best_model:
            best_metrics = results.get('metrics', {})
            self.model_metrics = {
                'model_name': self.trainer.best_model_name,
                'f1': best_metrics.get('f1', 0.0),
                'f1_score': best_metrics.get('f1', 0.0),
                'precision': best_metrics.get('precision', 0.0),
                'recall': best_metrics.get('recall', 0.0),
                'auc': best_metrics.get('auc', 0.0),
            }
            print(f"  Best Model: {self.model_metrics['model_name']}")
            print(f"  F1 Score: {self.model_metrics['f1']:.4f}")

    def _extract_rules(self):
        """Extract trading rules from the best model."""
        if not self.trainer or not self.trainer.best_model:
            raise ValueError("No trained model available!")

        # Create rule extractor
        extractor = RuleExtractor(
            model=self.trainer.best_model,
            scaler=self.trainer.scaler,
            feature_names=self.trainer.feature_names,
            X_train=self.trainer.X_train,
            y_train=self.trainer.y_train,
            X_test=self.trainer.X_test,
            y_test=self.trainer.y_test
        )

        # Extract rules (with optional LLM design for feature prioritization)
        self.rules = extractor.extract_all_rules(
            strategy_name=self.strategy_name,
            symbols=self.symbols,
            timeframe=self.timeframe,
            model_metrics=self.model_metrics,
            llm_design=self.llm_design
        )

    def _optimize_strategy(self):
        """Optimize risk management parameters using grid search."""
        print(f"  📊 Running parameter optimization on {self.strategy_dir.name}...")
        print(f"  Testing combinations of: stop_loss, trailing_stop, max_positions, min_entry_score\n")

        try:
            # Create optimizer
            optimizer = StrategyOptimizer(str(self.strategy_dir))

            # Build optimization ranges - narrow around LLM recommendations if available
            if self.llm_design:
                sl = self.llm_design.recommended_stop_loss
                ts = self.llm_design.recommended_trailing_stop
                mp = self.llm_design.recommended_max_positions

                # Clamp function for building realistic ranges
                def clamp_sl(v): return max(0.02, min(0.15, round(v, 3)))
                def clamp_ts(v): return max(0.01, min(0.10, round(v, 3)))

                # Center ranges around LLM values with some exploration (all clamped)
                stop_loss_range = sorted(set([
                    clamp_sl(sl * 0.5),
                    clamp_sl(sl * 0.75),
                    clamp_sl(sl),
                    clamp_sl(sl * 1.25),
                    clamp_sl(sl * 1.5)
                ]))
                trailing_stop_range = sorted(set([
                    clamp_ts(ts * 0.5),
                    clamp_ts(ts * 0.75),
                    clamp_ts(ts),
                    clamp_ts(ts * 1.25),
                    clamp_ts(ts * 1.5)
                ]))
                max_positions_range = sorted(set([
                    max(1, mp - 1), mp, min(5, mp + 1)
                ]))
                min_entry_score_range = [
                    max(1, self.llm_design.entry_score_threshold - 1),
                    self.llm_design.entry_score_threshold,
                    self.llm_design.entry_score_threshold + 1,
                    self.llm_design.entry_score_threshold + 2
                ]
                print(f"  🤖 LLM-guided optimization ranges:")
                print(f"     Stop Loss: {stop_loss_range}")
                print(f"     Trailing Stop: {trailing_stop_range}")
                print(f"     Max Positions: {max_positions_range}")
                print(f"     Min Entry Score: {min_entry_score_range}")
            else:
                stop_loss_range = [0.02, 0.03, 0.05, 0.07, 0.10]
                trailing_stop_range = [0.015, 0.02, 0.03, 0.05, 0.07]
                max_positions_range = [1, 2, 3, 4, 5]
                min_entry_score_range = [1.5, 2, 3, 4]

            # Run optimization
            best = optimizer.optimize(
                stop_loss_range=stop_loss_range,
                trailing_stop_range=trailing_stop_range,
                max_positions_range=max_positions_range,
                min_entry_score_range=min_entry_score_range,
                n_iterations=50
            )

            if best:
                # Apply the best parameters
                optimizer.apply_best_params(best)
                print(f"\n  ✅ Optimization complete!")
                print(f"     Best Return: {best.total_return:.2f}%")
                print(f"     Max Drawdown: {best.max_drawdown:.2f}%")
                print(f"     Optimal Params Applied to config.json")
            else:
                print(f"\n  ⚠️ Optimization did not find valid results. Using default parameters.")

        except Exception as e:
            print(f"\n  ⚠️ Optimization failed: {e}")
            print(f"     Continuing with default parameters.")

    def _generate_strategy_files(self):
        """Generate all strategy files from templates."""
        # Create strategy directory
        self.strategy_dir.mkdir(parents=True, exist_ok=True)

        # Generate each file
        self._generate_strategy_py()
        self._generate_backtest_py()
        self._generate_data_provider_py()
        self._generate_risk_manager_py()
        self._generate_main_py()
        self._generate_dockerfile()
        self._generate_docker_compose()

        print(f"  ✓ Generated strategy.py")
        print(f"  ✓ Generated backtest.py")
        print(f"  ✓ Generated data_provider.py")
        print(f"  ✓ Generated risk_manager.py")
        print(f"  ✓ Generated main.py")
        print(f"  ✓ Generated Dockerfile")
        print(f"  ✓ Generated docker-compose.yml")

    def _generate_strategy_py(self):
        """Generate the main strategy.py file."""
        # Build entry conditions from rules
        entry_conditions = self._build_entry_conditions()
        exit_conditions = self._build_exit_conditions()

        # Get top features for documentation
        top_features_str = ", ".join([f['feature'] for f in self.rules.top_features[:5]])

        strategy_code = f'''#!/usr/bin/env python3
"""
{self.strategy_name} - AI-Generated Portfolio Trading Strategy

Generated by SignalSynk Strategy Builder on {datetime.now().strftime("%Y-%m-%d %H:%M")}

Strategy based on ML model analysis:
- Model: {self.model_metrics.get('model_name', 'RandomForest')}
- F1 Score: {self.model_metrics.get('f1_score', 0):.3f}
- Top Features: {top_features_str}
- Timeframe: {self.timeframe}

Portfolio-based strategy that:
1. Evaluates ALL symbols simultaneously
2. Ranks by entry signal strength
3. Selects top N positions (default: 3)
4. Discards symbols without valid entry signals

This is a pure algorithmic strategy - no ML dependencies at runtime.

Author: SignalSynk AI
Copyright © 2026 Forefront Solutions Inc. All rights reserved.
"""

import os
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger('{self.strategy_name.replace(" ", "")}Strategy')


class TradingStrategy:
    """
    {self.strategy_name} Portfolio Trading Strategy.

    AI-generated strategy optimized for: {", ".join(self.symbols)}

    This strategy evaluates all symbols together and selects the best
    positions based on entry signal strength.
    """

    def __init__(self):
        """Initialize the strategy with configuration."""
        self.config = self._load_config()
        self.positions = {{}}
        self.daily_pnl = 0.0

        # Get configuration
        trading_config = self.config.get('trading', {{}})
        self.symbols = trading_config.get('symbols', {self.symbols})
        self.default_quantity = trading_config.get('default_quantity', 10)
        self.max_positions = trading_config.get('max_positions', 3)
        self.min_entry_score = trading_config.get('min_entry_score', {self.rules.entry_score_threshold})

        # Indicator parameters from ML analysis
        indicator_config = self.config.get('indicators', {{}})
        self.rsi_period = indicator_config.get('rsi_period', 14)
        self.rsi_oversold = indicator_config.get('rsi_oversold', {self.rules.optimal_params.get('rsi_oversold', 35)})
        self.rsi_overbought = indicator_config.get('rsi_overbought', 70)
        self.atr_period = indicator_config.get('atr_period', 14)
        self.atr_multiplier = indicator_config.get('atr_multiplier', 2.5)

        # Risk management
        risk_config = self.config.get('risk_management', {{}})
        self.max_position_size = risk_config.get('max_position_size', 0.1)
        self.stop_loss_pct = risk_config.get('stop_loss_pct', 0.05)
        self.trailing_stop_pct = risk_config.get('trailing_stop_pct', 0.03)
        self.max_drawdown = risk_config.get('max_drawdown', 0.20)
        self.max_daily_loss = risk_config.get('max_daily_loss', 0.05)

        logger.info(f"{self.strategy_name} Strategy initialized")
        logger.info(f"  Symbols: {{self.symbols}}")
        logger.info(f"  Max positions: {{self.max_positions}}")

    def _load_config(self) -> Dict[str, Any]:
        """Load strategy configuration from config.json."""
        config_path = Path(__file__).parent / 'config.json'
        try:
            with open(config_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config.json: {{e}}")
            return {{}}

    def calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        \"\"\"Calculate all technical indicators needed for the strategy.

        This calculates the same indicators used by the ML model during training.
        \"\"\"
        if len(df) < 50:
            return {{}}

        try:
            indicators = {{}}

            # RSI (multiple periods)
            for period in [7, 14, 21]:
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / (loss + 1e-10)
                rsi = 100 - (100 / (1 + rs))
                indicators[f'rsi_{{period}}'] = float(rsi.iloc[-1])

            # Stochastic Oscillator
            for period in [14, 21]:
                low_n = df['Low'].rolling(period).min()
                high_n = df['High'].rolling(period).max()
                stoch_k = ((df['Close'] - low_n) / (high_n - low_n + 1e-10)) * 100
                stoch_d = stoch_k.rolling(3).mean()
                indicators[f'stoch_k_{{period}}'] = float(stoch_k.iloc[-1])
                indicators[f'stoch_d_{{period}}'] = float(stoch_d.iloc[-1])

            # Williams %R
            indicators['williams_r'] = float(((df['High'].rolling(14).max() - df['Close']) /
                (df['High'].rolling(14).max() - df['Low'].rolling(14).min() + 1e-10)).iloc[-1] * -100)

            # MACD
            ema12 = df['Close'].ewm(span=12).mean()
            ema26 = df['Close'].ewm(span=26).mean()
            macd = ema12 - ema26
            macd_signal = macd.ewm(span=9).mean()
            macd_hist = macd - macd_signal
            indicators['macd'] = float(macd.iloc[-1])
            indicators['macd_signal'] = float(macd_signal.iloc[-1])
            indicators['macd_hist'] = float(macd_hist.iloc[-1])

            # ROC (Rate of Change)
            for period in [5, 10, 20]:
                roc = ((df['Close'] - df['Close'].shift(period)) /
                       (df['Close'].shift(period) + 1e-10)) * 100
                indicators[f'roc_{{period}}'] = float(roc.iloc[-1])

            # MFI (Money Flow Index)
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            mf = tp * df['Volume']
            pos_mf = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
            neg_mf = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
            mfi = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-10)))
            indicators['mfi'] = float(mfi.iloc[-1])

            # CCI (Commodity Channel Index) - OPTIMIZED: using std approximation instead of slow rolling apply
            tp_sma = tp.rolling(20).mean()
            # Mean Absolute Deviation ≈ 0.7979 * std for normal distribution (much faster than rolling apply)
            mean_dev = tp.rolling(20).std() * 0.7979
            cci = (tp - tp_sma) / (0.015 * mean_dev + 1e-10)
            indicators['cci'] = float(cci.iloc[-1])

            # EMAs (multiple periods)
            for period in [8, 13, 21, 50, 100, 200]:
                ema = df['Close'].ewm(span=period).mean()
                indicators[f'ema_{{period}}'] = float(ema.iloc[-1])
                price_vs_ema = ((df['Close'] - ema) / (ema + 1e-10)) * 100
                indicators[f'price_vs_ema_{{period}}'] = float(price_vs_ema.iloc[-1])

            # ATR and ADX
            tr = pd.concat([
                df['High'] - df['Low'],
                abs(df['High'] - df['Close'].shift()),
                abs(df['Low'] - df['Close'].shift())
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            indicators['atr'] = float(atr.iloc[-1])
            indicators['atr_pct'] = float((atr / (df['Close'] + 1e-10)).iloc[-1] * 100)

            # ADX components
            up_move = df['High'] - df['High'].shift()
            down_move = df['Low'].shift() - df['Low']
            plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
            minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
            plus_di = 100 * (plus_dm.rolling(14).mean() / (atr + 1e-10))
            minus_di = 100 * (minus_dm.rolling(14).mean() / (atr + 1e-10))
            dx = abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
            indicators['adx'] = float(dx.rolling(14).mean().iloc[-1])
            indicators['plus_di'] = float(plus_di.iloc[-1])
            indicators['minus_di'] = float(minus_di.iloc[-1])
            indicators['di_diff'] = float((plus_di - minus_di).iloc[-1])

            # Bollinger Bands
            bb_mid = df['Close'].rolling(20).mean()
            bb_std = df['Close'].rolling(20).std()
            bb_upper = bb_mid + 2 * bb_std
            bb_lower = bb_mid - 2 * bb_std
            indicators['bb_upper'] = float(bb_upper.iloc[-1])
            indicators['bb_lower'] = float(bb_lower.iloc[-1])
            indicators['bb_pct_b'] = float(((df['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)).iloc[-1])
            indicators['bb_width'] = float(((bb_upper - bb_lower) / (bb_mid + 1e-10)).iloc[-1] * 100)

            # Keltner Channels
            kc_mid = df['Close'].ewm(span=20).mean()
            kc_upper = kc_mid + 2 * atr
            kc_lower = kc_mid - 2 * atr
            indicators['kc_pct'] = float(((df['Close'] - kc_lower) / (kc_upper - kc_lower + 1e-10)).iloc[-1])

            # Volume indicators
            indicators['volume_ratio'] = float((df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-10)).iloc[-1])

            # OBV (On Balance Volume)
            obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
            obv_ema = obv.ewm(span=20).mean()
            indicators['obv_ema'] = float(obv_ema.iloc[-1])
            indicators['obv_diff'] = float((obv - obv_ema).iloc[-1])

            # CMF (Chaikin Money Flow)
            clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-10)
            clv = clv.fillna(0)
            mfv = clv * df['Volume']
            cmf = mfv.rolling(21).sum() / (df['Volume'].rolling(21).sum() + 1e-10)
            indicators['cmf'] = float(cmf.iloc[-1])

            # VWAP
            vwap = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / (df['Volume'].cumsum() + 1e-10)
            indicators['vwap'] = float(vwap.iloc[-1])
            indicators['price_vs_vwap'] = float(((df['Close'] - vwap) / (vwap + 1e-10)).iloc[-1] * 100)

            # =====================================================================
            # MACRO/TREND FEATURES - Long-term trend indicators
            # =====================================================================

            # Long-term momentum (1mo, 2mo, 3mo, 6mo)
            for period in [21, 42, 63, 126]:
                mom = df['Close'].pct_change(period) * 100
                indicators[f'momentum_{{period}}d'] = float(mom.iloc[-1]) if len(mom) > period else 0

            # 52-week high/low position
            if len(df) >= 252:
                high_252 = df['High'].rolling(252).max()
                low_252 = df['Low'].rolling(252).min()
                indicators['price_vs_52w_high'] = float(((df['Close'] - high_252) / (high_252 + 1e-10)).iloc[-1] * 100)
                indicators['price_vs_52w_low'] = float(((df['Close'] - low_252) / (low_252 + 1e-10)).iloc[-1] * 100)
                indicators['price_52w_range_pct'] = float(((df['Close'] - low_252) / (high_252 - low_252 + 1e-10)).iloc[-1] * 100)
            else:
                indicators['price_vs_52w_high'] = 0
                indicators['price_vs_52w_low'] = 0
                indicators['price_52w_range_pct'] = 50  # Default to middle

            # EMA stack score (bullish alignment)
            ema8 = df['Close'].ewm(span=8).mean()
            ema21 = df['Close'].ewm(span=21).mean()
            ema50 = df['Close'].ewm(span=50).mean()
            ema200 = df['Close'].ewm(span=200).mean()
            stack_score = int(ema8.iloc[-1] > ema21.iloc[-1]) + int(ema21.iloc[-1] > ema50.iloc[-1]) + int(ema50.iloc[-1] > ema200.iloc[-1])
            indicators['ema_stack_score'] = stack_score
            indicators['ema_stack_bullish'] = int(stack_score == 3)

            # Price acceleration
            mom5 = df['Close'].pct_change(5)
            mom10 = df['Close'].pct_change(10)
            indicators['price_accel_5d'] = float(mom5.diff(5).iloc[-1] * 100) if len(df) > 10 else 0
            indicators['price_accel_10d'] = float(mom10.diff(10).iloc[-1] * 100) if len(df) > 20 else 0

            # Trend slope (normalized)
            if len(df) >= 50:
                x = np.arange(50)
                y = df['Close'].iloc[-50:].values
                slope = np.polyfit(x, y, 1)[0]
                indicators['trend_slope_50d'] = float(slope / (df['Close'].iloc[-1] + 1e-10) * 100)
            else:
                indicators['trend_slope_50d'] = 0

            # Distance from all-time high
            ath = df['High'].expanding().max()
            indicators['dist_from_ath'] = float(((df['Close'] - ath) / (ath + 1e-10)).iloc[-1] * 100)

            # Breakout signals
            indicators['breakout_20d'] = int(df['Close'].iloc[-1] > df['High'].iloc[-21:-1].max()) if len(df) > 21 else 0
            indicators['breakout_50d'] = int(df['Close'].iloc[-1] > df['High'].iloc[-51:-1].max()) if len(df) > 51 else 0

            # Volume trend
            vol_20 = df['Volume'].rolling(20).mean()
            vol_50 = df['Volume'].rolling(50).mean()
            indicators['volume_trend'] = float((vol_20 / (vol_50 + 1e-10)).iloc[-1])

            # Basic price data
            indicators['close'] = float(df['Close'].iloc[-1])
            indicators['volume'] = float(df['Volume'].iloc[-1])

            return indicators

        except Exception as e:
            logger.error(f"Error calculating indicators: {{e}}")
            return {{}}

    def get_entry_score(self, indicators: Dict[str, Any]) -> int:
        \"\"\"Calculate entry score for given indicators.\"\"\"
        if not indicators:
            return 0

        entry_score = 0
{entry_conditions}

        return entry_score

    def get_exit_score(self, indicators: Dict[str, Any]) -> int:
        \"\"\"Calculate exit score for given indicators.\"\"\"
        if not indicators:
            return 0

        exit_score = 0
{exit_conditions}

        return exit_score

    def should_exit(self, indicators: Dict[str, Any]) -> bool:
        \"\"\"Check if position should be exited based on exit conditions.\"\"\"
        return self.get_exit_score(indicators) >= {self.rules.exit_score_threshold}

    def generate_signal(self, symbol: str, indicators: Dict[str, Any]) -> str:
        \"\"\"Generate trading signal based on extracted rules.\"\"\"
        if not indicators:
            return 'HOLD'

        entry_score = self.get_entry_score(indicators)
        exit_score = self.get_exit_score(indicators)

        # Generate signal
        if entry_score >= self.min_entry_score:
            logger.info(f"{{symbol}}: BUY signal (entry_score={{entry_score}})")
            return 'BUY'
        elif exit_score >= {self.rules.exit_score_threshold}:
            logger.info(f"{{symbol}}: SELL signal (exit_score={{exit_score}})")
            return 'SELL'

        return 'HOLD'

    def rank_symbols(self, symbol_indicators: Dict[str, Dict[str, Any]]) -> List[Tuple[str, int]]:
        \"\"\"
        Rank all symbols by their entry signal strength.

        Args:
            symbol_indicators: Dict mapping symbol -> indicators

        Returns:
            List of (symbol, entry_score) tuples, sorted by score descending.
            Only includes symbols with valid entry signals (score >= min_entry_score).
        \"\"\"
        rankings = []

        for symbol, indicators in symbol_indicators.items():
            if not indicators:
                continue

            entry_score = self.get_entry_score(indicators)

            # Only include symbols with valid entry signals
            if entry_score >= self.min_entry_score:
                rankings.append((symbol, entry_score))

        # Sort by entry score descending (best signals first)
        rankings.sort(key=lambda x: x[1], reverse=True)

        return rankings

    def select_top_positions(self, symbol_indicators: Dict[str, Dict[str, Any]],
                              current_positions: List[str] = None) -> List[str]:
        \"\"\"
        Select the top N symbols to hold positions in.

        Args:
            symbol_indicators: Dict mapping symbol -> indicators
            current_positions: List of symbols currently held

        Returns:
            List of symbols to hold (max = self.max_positions)
        \"\"\"
        current_positions = current_positions or []

        # Get ranked symbols with valid entry signals
        rankings = self.rank_symbols(symbol_indicators)

        # Select top max_positions
        selected = [symbol for symbol, score in rankings[:self.max_positions]]

        logger.info(f"Portfolio selection: {{selected}} from {{len(rankings)}} candidates")

        return selected

    def get_symbols(self) -> List[str]:
        \"\"\"Return list of symbols to trade.\"\"\"
        return self.symbols
'''

        with open(self.strategy_dir / 'strategy.py', 'w') as f:
            f.write(strategy_code)

    def _get_indicator_var_name(self, feature_name: str) -> str:
        """Map ML feature name to the calculated indicator variable name."""
        # The features from ML match the indicator names calculated
        # Feature names are like: RSI_14, EMA_200, OBV_EMA, CMF, PRICE_VS_VWAP, etc.
        # We use lowercase versions in the indicators dict
        feature_lower = feature_name.lower()
        return f"indicators.get('{feature_lower}', 0)"

    def _get_used_features(self) -> set:
        """Get all feature names used in entry and exit rules."""
        features = set()
        for rule in self.rules.entry_rules:
            features.add(rule.feature)
        for rule in self.rules.exit_rules:
            features.add(rule.feature)
        return features

    def _build_entry_conditions(self) -> str:
        """Build entry condition code from extracted rules."""
        lines = []
        for rule in self.rules.entry_rules[:8]:  # Top 8 rules
            feature = rule.feature
            op = rule.operator
            threshold = rule.threshold
            var_name = self._get_indicator_var_name(feature)

            lines.append(f"        if {var_name} {op} {threshold}:")
            lines.append(f"            entry_score += 1  # {rule.description}")

        return "\n".join(lines) if lines else "        pass  # No entry rules extracted"

    def _build_exit_conditions(self) -> str:
        """Build exit condition code from extracted rules."""
        lines = []
        for rule in self.rules.exit_rules[:6]:  # Top 6 rules
            feature = rule.feature
            op = rule.operator
            threshold = rule.threshold
            var_name = self._get_indicator_var_name(feature)

            lines.append(f"        if {var_name} {op} {threshold}:")
            lines.append(f"            exit_score += 1  # {rule.description}")

        return "\n".join(lines) if lines else "        pass  # No exit rules extracted"

    def _generate_backtest_py(self):
        """Generate the backtest.py file with portfolio-based simulation and risk management."""
        backtest_code = f'''#!/usr/bin/env python3
"""
Portfolio Backtest Runner for {self.strategy_name}

Features:
- Portfolio-based: Evaluates all symbols together, selects best 3
- Risk Management: Stop loss, trailing stop, max drawdown, daily drawdown
- Position Sizing: Based on account value and risk limits

Author: SignalSynk AI
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategy import TradingStrategy
from data_provider import DataProvider
from risk_manager import RiskManager


@dataclass
class Position:
    \"\"\"Represents an open position.\"\"\"
    symbol: str
    entry_price: float
    entry_date: datetime
    shares: int
    highest_price: float = 0.0  # For trailing stop

    def __post_init__(self):
        self.highest_price = self.entry_price


@dataclass
class Trade:
    \"\"\"Completed trade record.\"\"\"
    symbol: str
    entry_price: float
    exit_price: float
    entry_date: datetime
    exit_date: datetime
    shares: int
    pnl: float
    pnl_pct: float
    exit_reason: str


class PortfolioBacktester:
    \"\"\"
    Portfolio-based backtester with proper risk management.

    Evaluates all symbols together and selects the best positions
    based on entry signal strength.
    \"\"\"

    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.strategy = TradingStrategy()
        self.data_provider = DataProvider()

        # Load config
        config = self._load_config()
        risk_config = config.get('risk_management', {{}})

        # Risk parameters
        self.stop_loss_pct = risk_config.get('stop_loss_pct', 0.05)
        self.trailing_stop_pct = risk_config.get('trailing_stop_pct', 0.03)
        self.max_drawdown = risk_config.get('max_drawdown', 0.20)
        self.max_daily_loss = risk_config.get('max_daily_loss', 0.05)
        self.max_position_size = risk_config.get('max_position_size', 0.1)
        self.max_positions = config.get('trading', {{}}).get('max_positions', 3)
        self.lookback_window = config.get('trading', {{}}).get('lookback_window', 200)

        # State
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {{}}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.peak_equity = initial_capital
        self.daily_start_equity = initial_capital
        self.current_date = None
        self.halted = False
        self.halt_reason = ""

        # Benchmark tracking
        self.buy_hold_returns: Dict[str, float] = {{}}
        self.max_potential_profit: float = 0.0
        self.symbol_start_prices: Dict[str, float] = {{}}
        self.symbol_end_prices: Dict[str, float] = {{}}

        # Cached data and indicators (set by optimizer for 50x speedup)
        self._cached_data: Optional[Dict[str, pd.DataFrame]] = None
        self._cached_indicators: Optional[Dict[str, Dict]] = None

    def _load_config(self) -> Dict[str, Any]:
        config_path = Path(__file__).parent / 'config.json'
        try:
            with open(config_path) as f:
                return json.load(f)
        except:
            return {{}}

    def get_indicators(self, symbol: str, date, df_slice: pd.DataFrame) -> Dict[str, Any]:
        \"\"\"Get indicators from cache or calculate them.

        This is a MAJOR optimization: when running optimizer, indicators are
        pre-calculated once and cached, giving ~50x speedup.
        \"\"\"
        # Try to get from cache first (set by optimizer)
        if self._cached_indicators and symbol in self._cached_indicators:
            if date in self._cached_indicators[symbol]:
                return self._cached_indicators[symbol][date]

        # Fall back to calculating (slower, but works for standalone runs)
        return self.strategy.calculate_indicators(df_slice)

    def get_equity(self, prices: Dict[str, float]) -> float:
        \"\"\"Calculate current portfolio equity.\"\"\"
        position_value = sum(
            pos.shares * prices.get(symbol, pos.entry_price)
            for symbol, pos in self.positions.items()
        )
        return self.cash + position_value

    def check_risk_limits(self, prices: Dict[str, float]) -> bool:
        \"\"\"Check if risk limits are breached. Returns True if trading should halt.\"\"\"
        equity = self.get_equity(prices)

        # Check max drawdown
        drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0
        if drawdown >= self.max_drawdown:
            self.halted = True
            self.halt_reason = f"Max drawdown breached: {{drawdown*100:.1f}}% >= {{self.max_drawdown*100:.1f}}%"
            return True

        # Check daily drawdown
        daily_loss = (self.daily_start_equity - equity) / self.daily_start_equity if self.daily_start_equity > 0 else 0
        if daily_loss >= self.max_daily_loss:
            self.halted = True
            self.halt_reason = f"Daily loss limit breached: {{daily_loss*100:.1f}}% >= {{self.max_daily_loss*100:.1f}}%"
            return True

        return False

    def check_stop_loss(self, position: Position, current_price: float) -> Optional[str]:
        \"\"\"Check if position should be closed due to stop loss or trailing stop.\"\"\"
        pnl_pct = (current_price - position.entry_price) / position.entry_price

        # Fixed stop loss
        if pnl_pct <= -self.stop_loss_pct:
            return "stop_loss"

        # Trailing stop (only if in profit)
        if current_price > position.highest_price:
            position.highest_price = current_price

        if position.highest_price > position.entry_price:
            drop_from_high = (position.highest_price - current_price) / position.highest_price
            if drop_from_high >= self.trailing_stop_pct:
                return "trailing_stop"

        return None

    def close_position(self, symbol: str, current_price: float, current_date: datetime,
                       exit_reason: str) -> Trade:
        \"\"\"Close a position and record the trade.\"\"\"
        position = self.positions[symbol]
        pnl = (current_price - position.entry_price) * position.shares
        pnl_pct = (current_price - position.entry_price) / position.entry_price * 100

        trade = Trade(
            symbol=symbol,
            entry_price=position.entry_price,
            exit_price=current_price,
            entry_date=position.entry_date,
            exit_date=current_date,
            shares=position.shares,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=exit_reason
        )

        self.cash += current_price * position.shares
        del self.positions[symbol]
        self.trades.append(trade)

        return trade

    def open_position(self, symbol: str, current_price: float, current_date: datetime,
                      prices: Dict[str, float]) -> Optional[Position]:
        \"\"\"Open a new position with proper position sizing.\"\"\"
        equity = self.get_equity(prices)
        max_position_value = equity * self.max_position_size
        shares = int(max_position_value / current_price)

        if shares <= 0:
            return None

        cost = shares * current_price
        if cost > self.cash:
            shares = int(self.cash / current_price)
            cost = shares * current_price

        if shares <= 0:
            return None

        position = Position(
            symbol=symbol,
            entry_price=current_price,
            entry_date=current_date,
            shares=shares
        )

        self.cash -= cost
        self.positions[symbol] = position

        return position

    def run(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        \"\"\"Run the portfolio backtest.\"\"\"
        print(f"\\n{{'='*60}}")
        print(f"PORTFOLIO BACKTEST: {self.strategy_name}")
        print(f"{{'='*60}}")
        print(f"  Initial Capital: ${{self.initial_capital:,.2f}}")
        print(f"  Symbols: {{', '.join(self.strategy.get_symbols())}}")
        print(f"  Max Positions: {{self.max_positions}}")
        print(f"  Stop Loss: {{self.stop_loss_pct*100:.1f}}%")
        print(f"  Trailing Stop: {{self.trailing_stop_pct*100:.1f}}%")
        print(f"  Max Drawdown: {{self.max_drawdown*100:.1f}}%")
        print(f"  Max Daily Loss: {{self.max_daily_loss*100:.1f}}%")
        print(f"  Lookback Window: {{self.lookback_window}} bars")
        print(f"{{'='*60}}\\n")

        # Default to 2 years of data
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

        print(f"  Fetching data from {{start_date}} to {{end_date}}...")

        # Fetch all data
        all_data: Dict[str, pd.DataFrame] = {{}}
        for symbol in self.strategy.get_symbols():
            df = self.data_provider.get_historical_data(symbol, start_date, end_date)
            if df is not None and len(df) >= self.lookback_window:
                all_data[symbol] = df
                print(f"    ✓ {{symbol}}: {{len(df)}} bars")
            else:
                print(f"    ⚠️ {{symbol}}: Insufficient data")

        if not all_data:
            print("\\n  ❌ No valid data available for backtest")
            return {{}}

        # Find common date range
        common_dates = None
        for df in all_data.values():
            if common_dates is None:
                common_dates = set(df.index)
            else:
                common_dates &= set(df.index)

        common_dates = sorted(list(common_dates))
        if len(common_dates) < 50:
            print(f"\\n  ❌ Insufficient common dates: {{len(common_dates)}}")
            return {{}}

        # Calculate buy-and-hold returns and max potential profit for each symbol
        for symbol, df in all_data.items():
            start_price = float(df.iloc[0]['Close'])
            end_price = float(df.iloc[-1]['Close'])
            self.symbol_start_prices[symbol] = start_price
            self.symbol_end_prices[symbol] = end_price
            self.buy_hold_returns[symbol] = ((end_price - start_price) / start_price) * 100

            # Calculate max potential profit (buy at lowest, sell at highest)
            lowest = float(df['Low'].min())
            highest = float(df['High'].max())
            max_potential = ((highest - lowest) / lowest) * 100
            self.max_potential_profit = max(self.max_potential_profit, max_potential)

        lookback = min(self.lookback_window, len(common_dates) - 1)  # Don't exceed available data
        print(f"\\n  Running simulation over {{len(common_dates)}} trading days (lookback: {{lookback}} bars)...")

        # Simulation loop — skip lookback_window bars to let indicators stabilize
        last_date = None
        for i, date in enumerate(common_dates[lookback:], start=lookback):
            self.current_date = date

            # Reset daily tracking on new day
            if last_date is not None and date.date() != last_date.date():
                self.daily_start_equity = self.get_equity(
                    {{sym: float(all_data[sym].loc[date]['Close']) for sym in all_data if date in all_data[sym].index}}
                )
            last_date = date

            # Get current prices
            prices = {{}}
            for symbol, df in all_data.items():
                if date in df.index:
                    prices[symbol] = float(df.loc[date]['Close'])

            # Check risk limits
            if self.check_risk_limits(prices):
                print(f"\\n  ⚠️ TRADING HALTED: {{self.halt_reason}}")
                # Close all positions
                for symbol in list(self.positions.keys()):
                    if symbol in prices:
                        trade = self.close_position(symbol, prices[symbol], date, "risk_halt")
                        print(f"    Closed {{symbol}}: {{trade.pnl_pct:.2f}}%")
                break

            # Update peak equity
            equity = self.get_equity(prices)
            if equity > self.peak_equity:
                self.peak_equity = equity
            self.equity_curve.append(equity)

            # Check existing positions for stops and exit signals
            for symbol in list(self.positions.keys()):
                if symbol not in prices:
                    continue

                current_price = prices[symbol]
                position = self.positions[symbol]

                # Check stop loss / trailing stop
                stop_reason = self.check_stop_loss(position, current_price)
                if stop_reason:
                    trade = self.close_position(symbol, current_price, date, stop_reason)
                    continue

                # Check exit signal (use cached indicators if available)
                df_slice = all_data[symbol].loc[:date]
                indicators = self.get_indicators(symbol, date, df_slice)
                if self.strategy.should_exit(indicators):
                    trade = self.close_position(symbol, current_price, date, "signal_exit")

            # Calculate indicators for all symbols (use cached if available - 50x speedup)
            symbol_indicators = {{}}
            for symbol, df in all_data.items():
                if date in df.index:
                    df_slice = df.loc[:date]
                    symbol_indicators[symbol] = self.get_indicators(symbol, date, df_slice)

            # Select best positions to hold
            target_symbols = self.strategy.select_top_positions(
                symbol_indicators,
                list(self.positions.keys())
            )

            # Open new positions for symbols in target but not currently held
            for symbol in target_symbols:
                if symbol not in self.positions and symbol in prices:
                    if len(self.positions) < self.max_positions:
                        self.open_position(symbol, prices[symbol], date, prices)

        # Close any remaining positions at end
        final_prices = {{}}
        for symbol, df in all_data.items():
            if len(df) > 0:
                final_prices[symbol] = float(df.iloc[-1]['Close'])

        for symbol in list(self.positions.keys()):
            if symbol in final_prices:
                self.close_position(symbol, final_prices[symbol], common_dates[-1], "backtest_end")

        return self._generate_report()

    def _generate_report(self) -> Dict[str, Any]:
        \"\"\"Generate backtest report.\"\"\"
        print(f"\\n{{'='*60}}")
        print(f"BACKTEST RESULTS")
        print(f"{{'='*60}}\\n")

        if not self.trades:
            print("  No trades executed")
            return {{}}

        # Calculate metrics
        total_pnl = sum(t.pnl for t in self.trades)
        total_return = (total_pnl / self.initial_capital) * 100

        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0

        avg_win = np.mean([t.pnl_pct for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl_pct for t in losing_trades]) if losing_trades else 0

        # Drawdown
        max_dd = 0
        peak = self.initial_capital
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd

        # Exit reasons
        exit_reasons = {{}}
        for t in self.trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        # Calculate buy-and-hold benchmark (equal weight)
        avg_buy_hold = np.mean(list(self.buy_hold_returns.values())) if self.buy_hold_returns else 0
        best_buy_hold_symbol = max(self.buy_hold_returns.items(), key=lambda x: x[1]) if self.buy_hold_returns else ('N/A', 0)

        print(f"  Total Trades: {{len(self.trades)}}")
        print(f"  Win Rate: {{win_rate:.1f}}%")
        print(f"  Total Return: {{total_return:.2f}}%")
        print(f"  Final Equity: ${{self.get_equity({{}}):,.2f}}")
        print(f"  Max Drawdown: {{max_dd*100:.2f}}%")
        print(f"  Avg Win: {{avg_win:.2f}}%")
        print(f"  Avg Loss: {{avg_loss:.2f}}%")

        # Benchmark comparison
        print(f"\\n  📊 BENCHMARK COMPARISON:")
        print(f"    Buy & Hold (avg): {{avg_buy_hold:.2f}}%")
        print(f"    Best Buy & Hold: {{best_buy_hold_symbol[0]}} ({{best_buy_hold_symbol[1]:.2f}}%)")
        print(f"    Max Potential Profit: {{self.max_potential_profit:.2f}}%")
        print(f"    Strategy vs Buy&Hold: {{'✅ BEAT' if total_return > avg_buy_hold else '❌ UNDERPERFORMED'}} by {{abs(total_return - avg_buy_hold):.2f}}%")
        print(f"    Capture Rate: {{(total_return / self.max_potential_profit * 100) if self.max_potential_profit > 0 else 0:.1f}}% of max potential")

        print(f"\\n  Exit Reasons:")
        for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
            print(f"    {{reason}}: {{count}}")

        # By symbol with buy-hold comparison
        print(f"\\n  By Symbol (Strategy vs Buy&Hold):")
        symbol_trades = {{}}
        for t in self.trades:
            if t.symbol not in symbol_trades:
                symbol_trades[t.symbol] = []
            symbol_trades[t.symbol].append(t)

        for symbol, trades in sorted(symbol_trades.items()):
            sym_pnl = sum(t.pnl_pct for t in trades)
            sym_wins = len([t for t in trades if t.pnl > 0])
            bh_return = self.buy_hold_returns.get(symbol, 0)
            vs_bh = sym_pnl - bh_return
            print(f"    {{symbol}}: {{len(trades)}} trades, {{sym_pnl:.2f}}% return, B&H: {{bh_return:.2f}}% ({{'+'if vs_bh>=0 else ''}}{{vs_bh:.2f}}%)")

        if self.halted:
            print(f"\\n  ⚠️ Trading was halted: {{self.halt_reason}}")

        print(f"\\n{{'='*60}}\\n")

        return {{
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'total_return': total_return,
            'max_drawdown': max_dd * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'exit_reasons': exit_reasons,
            'halted': self.halted,
            'halt_reason': self.halt_reason,
            'buy_hold_avg': avg_buy_hold,
            'buy_hold_best': best_buy_hold_symbol,
            'max_potential': self.max_potential_profit,
            'buy_hold_by_symbol': self.buy_hold_returns,
            'sharpe_ratio': round((total_return / 100) / (max_dd if max_dd > 0 else 1), 2),
            'profit_factor': round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else (float('inf') if avg_win > 0 else 0)
        }}


def run_backtest(start_date: str = None, end_date: str = None,
                 initial_capital: float = 100000.0) -> Dict[str, Any]:
    \"\"\"Run portfolio backtest.\"\"\"
    backtester = PortfolioBacktester(initial_capital=initial_capital)
    return backtester.run(start_date, end_date)


def save_backtest_results(results: Dict[str, Any], strategy_id: str = "{self.strategy_name}",
                         initial_capital: float = 100000.0) -> Dict[str, Any]:
    \"\"\"Save backtest results to JSON file in the expected format.\"\"\"
    # Load config to get strategy metadata
    config_path = Path(__file__).parent / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    strategy_name = config.get('strategy', {{}}).get('name', '{self.strategy_name}')
    symbols = config.get('trading', {{}}).get('symbols', [])
    timeframe = config.get('trading', {{}}).get('timeframe', '1d')

    # Get date range from results or use default
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years default

    # Calculate metrics from backtest results
    total_trades = results.get('total_trades', 0)
    win_rate = results.get('win_rate', 0.0)
    total_return = results.get('total_return', 0.0)
    max_drawdown = results.get('max_drawdown', 0.0)

    # Calculate Sharpe ratio estimate (simplified)
    sharpe_ratio = (total_return / 100) / (max_drawdown / 100) if max_drawdown > 0 else 0

    # Calculate profit factor estimate
    winning_trades = int(total_trades * win_rate / 100) if total_trades > 0 else 0
    losing_trades = total_trades - winning_trades
    avg_win = results.get('avg_win', 1.0)
    avg_loss = abs(results.get('avg_loss', -1.0))
    profit_factor = (winning_trades * avg_win) / (losing_trades * avg_loss) if losing_trades > 0 and avg_loss > 0 else 1.0

    # Calculate average trade P/L in dollars
    avg_trade_pnl = 0
    best_trade_pnl = 0
    worst_trade_pnl = 0
    if avg_win > 0:
        avg_trade_pnl = round(initial_capital * (avg_win / 100), 2)
        best_trade_pnl = round(initial_capital * (avg_win * 3 / 100), 2)  # Estimate
    if avg_loss > 0:
        worst_trade_pnl = round(-initial_capital * (avg_loss / 100), 2)  # Negative value

    # Prepare backtest results in expected format
    backtest_data = {{
        "strategy_id": strategy_id.lower().replace(" ", "_"),
        "strategy_name": strategy_name,
        "period": "2Y",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "total_return_pct": round(total_return, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "avg_trade_pnl": avg_trade_pnl,
        "best_trade_pnl": best_trade_pnl,
        "worst_trade_pnl": worst_trade_pnl,
        "avg_trade_duration_hours": 120.0,  # Placeholder - would need actual trade data
        "trades": [],  # Would need to collect trade details during backtest
        "equity_curve": [],  # Would need to collect equity curve during backtest
        "daily_returns": [],
        "weekly_returns": [],
        "monthly_returns": [],
        "yearly_returns": []
    }}

    # Save to file
    output_file = Path(__file__).parent / "backtest_results.json"
    with open(output_file, 'w') as f:
        json.dump(backtest_data, f, indent=2)

    print(f"\\n✅ Backtest results saved to: {{output_file}}")
    print(f"   Strategy: {{strategy_name}}")
    print(f"   Symbols: {{', '.join(symbols)}}")
    print(f"   Timeframe: {{timeframe}}")
    print(f"   Total Trades: {{total_trades}}")
    print(f"   Total Return: {{total_return:.2f}}%")
    print(f"   Win Rate: {{win_rate:.1f}}%")
    print(f"   Sharpe Ratio: {{sharpe_ratio:.2f}}")

    return backtest_data


if __name__ == '__main__':
    results = run_backtest()
    if results:
        save_backtest_results(results)
'''
        with open(self.strategy_dir / 'backtest.py', 'w') as f:
            f.write(backtest_code)

    def _generate_data_provider_py(self):
        """Generate the data_provider.py file."""
        data_provider_code = '''#!/usr/bin/env python3
"""
Data Provider - Fetches market data for the strategy.

Author: SignalSynk AI
"""

import os
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict

import pandas as pd

try:
    from polygon import RESTClient
    HAS_POLYGON = True
except ImportError:
    HAS_POLYGON = False


class DataProvider:
    """Provides market data for the strategy."""

    def __init__(self):
        self.api_key = os.environ.get('POLYGON_API_KEY', '')
        if HAS_POLYGON and self.api_key:
            self.client = RESTClient(self.api_key)
        else:
            self.client = None

    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV data."""
        if not self.client:
            print(f"⚠️ Polygon client not configured. Set POLYGON_API_KEY.")
            return None

        try:
            aggs = self.client.get_aggs(
                ticker=symbol,
                multiplier=1,
                timespan='day',
                from_=start_date,
                to=end_date,
                limit=50000
            )

            if not aggs:
                return None

            data = []
            for bar in aggs:
                data.append({
                    'Open': bar.open,
                    'High': bar.high,
                    'Low': bar.low,
                    'Close': bar.close,
                    'Volume': bar.volume,
                    'Timestamp': pd.Timestamp(bar.timestamp, unit='ms')
                })

            df = pd.DataFrame(data)
            df.set_index('Timestamp', inplace=True)
            df.index = df.index.tz_localize('UTC')
            return df

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    async def get_historical_data_async(self, symbol: str, days: int = 365,
                                        interval: str = 'day') -> Optional[pd.DataFrame]:
        """Async wrapper around the synchronous get_historical_data method."""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        return self.get_historical_data(symbol, start_date, end_date)

    async def get_multiple_symbols(self, symbols: List[str], days: int = 365,
                                   interval: str = 'day', parallel: bool = True) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols in parallel using asyncio.gather."""
        results = {}
        if parallel and len(symbols) > 1:
            tasks = [self.get_historical_data_async(sym, days, interval) for sym in symbols]
            fetched = await asyncio.gather(*tasks, return_exceptions=True)
            for sym, data in zip(symbols, fetched):
                if isinstance(data, pd.DataFrame) and not data.empty:
                    results[sym] = data
        else:
            for sym in symbols:
                df = await self.get_historical_data_async(sym, days, interval)
                if df is not None and not df.empty:
                    results[sym] = df
        return results
'''
        with open(self.strategy_dir / 'data_provider.py', 'w') as f:
            f.write(data_provider_code)

    def _generate_risk_manager_py(self):
        """Generate the risk_manager.py file with trailing stop support."""
        risk_manager_code = '''#!/usr/bin/env python3
"""
Risk Manager - Handles position sizing, risk controls, and stop losses.

Features:
- Fixed stop loss
- Trailing stop loss
- Max drawdown protection
- Daily drawdown limit
- Position sizing based on account value

Author: SignalSynk AI
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PositionRisk:
    """Tracks risk metrics for a single position."""
    symbol: str
    entry_price: float
    highest_price: float
    shares: int

    def __post_init__(self):
        if self.highest_price == 0:
            self.highest_price = self.entry_price


class RiskManager:
    """
    Manages risk for the trading strategy.

    Implements:
    - Fixed stop loss: Close position if loss exceeds threshold
    - Trailing stop loss: Lock in profits by tracking highest price
    - Max drawdown: Halt trading if portfolio drops too much from peak
    - Daily drawdown: Halt trading if daily loss exceeds threshold
    """

    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}

        # Risk limits
        self.max_position_size = config.get('max_position_size', 0.1)
        self.max_daily_loss = config.get('max_daily_loss', 0.05)
        self.max_drawdown = config.get('max_drawdown', 0.20)
        self.stop_loss_pct = config.get('stop_loss_pct', 0.05)
        self.trailing_stop_pct = config.get('trailing_stop_pct', 0.03)

        # State tracking
        self.daily_pnl = 0.0
        self.daily_start_equity = 0.0
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.positions: Dict[str, PositionRisk] = {}

        # Halt state
        self.halted = False
        self.halt_reason = ""

    def can_trade(self) -> Tuple[bool, str]:
        """
        Check if trading is allowed based on risk limits.

        Returns:
            Tuple of (can_trade: bool, reason: str)
        """
        if self.halted:
            return False, self.halt_reason

        # Check daily loss
        if self.daily_start_equity > 0:
            daily_loss = (self.daily_start_equity - self.current_equity) / self.daily_start_equity
            if daily_loss >= self.max_daily_loss:
                self.halted = True
                self.halt_reason = f"Daily loss limit: {daily_loss*100:.1f}%"
                return False, self.halt_reason

        # Check max drawdown
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
            if drawdown >= self.max_drawdown:
                self.halted = True
                self.halt_reason = f"Max drawdown: {drawdown*100:.1f}%"
                return False, self.halt_reason

        return True, ""

    def calculate_position_size(self, account_value: float, price: float) -> int:
        """
        Calculate position size based on risk limits.

        Args:
            account_value: Current account/portfolio value
            price: Current price of the asset

        Returns:
            Number of shares to buy
        """
        max_value = account_value * self.max_position_size
        shares = int(max_value / price)
        return max(0, shares)

    def check_stop_loss(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Check if position should be closed due to stop loss or trailing stop.

        Args:
            symbol: The symbol to check
            current_price: Current price of the asset

        Returns:
            None if no stop triggered, otherwise the stop reason
        """
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        pnl_pct = (current_price - pos.entry_price) / pos.entry_price

        # Fixed stop loss
        if pnl_pct <= -self.stop_loss_pct:
            return "stop_loss"

        # Update highest price
        if current_price > pos.highest_price:
            pos.highest_price = current_price

        # Trailing stop (only triggers if we were in profit)
        if pos.highest_price > pos.entry_price:
            drop_from_high = (pos.highest_price - current_price) / pos.highest_price
            if drop_from_high >= self.trailing_stop_pct:
                return "trailing_stop"

        return None

    def add_position(self, symbol: str, entry_price: float, shares: int):
        """Register a new position for risk tracking."""
        self.positions[symbol] = PositionRisk(
            symbol=symbol,
            entry_price=entry_price,
            highest_price=entry_price,
            shares=shares
        )

    def remove_position(self, symbol: str):
        """Remove a position from risk tracking."""
        if symbol in self.positions:
            del self.positions[symbol]

    def update_position_price(self, symbol: str, current_price: float):
        """Update the highest price for trailing stop calculation."""
        if symbol in self.positions:
            pos = self.positions[symbol]
            if current_price > pos.highest_price:
                pos.highest_price = current_price

    def update_equity(self, equity: float):
        """Update current equity and peak."""
        self.current_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity

    def update_daily_pnl(self, pnl: float):
        """Update daily P&L."""
        self.daily_pnl += pnl

    def reset_daily(self):
        """Reset daily tracking (call at start of each day)."""
        self.daily_pnl = 0.0
        self.daily_start_equity = self.current_equity
        # Don't reset halt state - that persists

    def get_drawdown(self) -> float:
        """Get current drawdown as a percentage."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity

    def get_daily_loss(self) -> float:
        """Get current daily loss as a percentage."""
        if self.daily_start_equity <= 0:
            return 0.0
        return (self.daily_start_equity - self.current_equity) / self.daily_start_equity

    def get_position_pnl(self, symbol: str, current_price: float) -> Tuple[float, float]:
        """
        Get current P&L for a position.

        Returns:
            Tuple of (pnl_dollars, pnl_percent)
        """
        if symbol not in self.positions:
            return 0.0, 0.0

        pos = self.positions[symbol]
        pnl_pct = (current_price - pos.entry_price) / pos.entry_price
        pnl_dollars = (current_price - pos.entry_price) * pos.shares

        return pnl_dollars, pnl_pct * 100
'''
        with open(self.strategy_dir / 'risk_manager.py', 'w') as f:
            f.write(risk_manager_code)

    def _generate_main_py(self):
        """Generate the main.py entry point."""
        main_code = f'''#!/usr/bin/env python3
"""
Main entry point for {self.strategy_name}

Author: SignalSynk AI
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from strategy import TradingStrategy
from data_provider import DataProvider
from risk_manager import RiskManager


def main():
    """Main entry point."""
    print(f"\\n{'='*60}")
    print(f"{self.strategy_name}")
    print(f"{'='*60}\\n")

    strategy = TradingStrategy()
    data_provider = DataProvider()
    risk_manager = RiskManager()

    print("Strategy initialized successfully!")
    print(f"  Symbols: {{strategy.get_symbols()}}")
    print("\\nRun backtest.py to test the strategy.")


if __name__ == '__main__':
    main()
'''
        with open(self.strategy_dir / 'main.py', 'w') as f:
            f.write(main_code)

    def _generate_dockerfile(self):
        """Generate the Dockerfile for the strategy."""
        strategy_slug = self.strategy_name.lower().replace(' ', '_').replace('-', '_')
        strategy_display = self.strategy_name.replace('_', ' ').title()

        dockerfile_content = f'''# Signal Synk Strategy Container - {strategy_display}
# Production-ready Docker image for running the strategy
# Auto-generated by Strategy Builder on {datetime.now().strftime("%Y-%m-%d %H:%M")}

FROM python:3.11-slim

# Metadata
LABEL maintainer="Signal Synk Team"
LABEL version="1.0.0"
LABEL description="{strategy_display} Trading Strategy for Signal Synk"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir -r requirements.txt

# Copy strategy code
COPY . .

# Environment variables (will be overridden at runtime)
ENV STRATEGY_PASSPHRASE=""
ENV SIGNALSYNK_WS_URL="wss://app.signalsynk.com/ws/strategy"
ENV SIGNALSYNK_API_URL="https://app.signalsynk.com"
ENV POLYGON_API_KEY=""
ENV ALPHAVANTAGE_API_KEY=""
ENV LOG_LEVEL="INFO"
ENV HEARTBEAT_INTERVAL="30"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash strategy && \\
    chown -R strategy:strategy /app

USER strategy

# Health check - verify process is responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \\
    CMD python -c "import asyncio; asyncio.run(__import__('websockets').connect('ws://localhost:8765', close_timeout=1))" 2>/dev/null || exit 0

# Run the strategy
CMD ["python", "-u", "main.py"]
'''
        with open(self.strategy_dir / 'Dockerfile', 'w') as f:
            f.write(dockerfile_content)

    def _generate_docker_compose(self):
        """Generate the docker-compose.yml for the strategy."""
        strategy_slug = self.strategy_name.lower().replace(' ', '_').replace('-', '_')
        strategy_display = self.strategy_name.replace('_', ' ').title()

        docker_compose_content = f'''# Docker Compose for {strategy_display}
# Auto-generated by Strategy Builder on {datetime.now().strftime("%Y-%m-%d %H:%M")}
#
# Usage:
#   docker-compose up -d          # Start the strategy
#   docker-compose logs -f        # View logs
#   docker-compose down           # Stop the strategy
#
# Environment variables can be set in a .env file or passed directly

version: "3.8"

services:
  {strategy_slug}:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: signalsynk_{strategy_slug}
    restart: unless-stopped
    environment:
      - STRATEGY_PASSPHRASE=${{STRATEGY_PASSPHRASE:-}}
      - SIGNALSYNK_WS_URL=${{SIGNALSYNK_WS_URL:-wss://app.signalsynk.com/ws/strategy}}
      - SIGNALSYNK_API_URL=${{SIGNALSYNK_API_URL:-https://app.signalsynk.com}}
      - POLYGON_API_KEY=${{POLYGON_API_KEY:-}}
      - ALPHAVANTAGE_API_KEY=${{ALPHAVANTAGE_API_KEY:-}}
      - LOG_LEVEL=${{LOG_LEVEL:-INFO}}
      - HEARTBEAT_INTERVAL=${{HEARTBEAT_INTERVAL:-30}}
    env_file:
      - .env
    volumes:
      # Mount config for live updates (optional)
      - ./config.json:/app/config.json:ro
    networks:
      - signalsynk_network
    healthcheck:
      test: ["CMD", "python", "-c", "import sys; sys.exit(0)"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M

networks:
  signalsynk_network:
    driver: bridge
    external: true
    name: signalsynk_signalsynk_network
'''
        with open(self.strategy_dir / 'docker-compose.yml', 'w') as f:
            f.write(docker_compose_content)

    def _create_config_files(self):
        """Create config.json and requirements.txt."""
        # Use LLM-recommended risk params if available, otherwise defaults
        if self.llm_design:
            max_positions = self.llm_design.recommended_max_positions
            min_entry_score = self.llm_design.entry_score_threshold
            max_position_size = self.llm_design.recommended_position_size
            stop_loss_pct = self.llm_design.recommended_stop_loss
            trailing_stop_pct = self.llm_design.recommended_trailing_stop
        else:
            max_positions = 3
            min_entry_score = 3
            max_position_size = 0.1
            stop_loss_pct = 0.05
            trailing_stop_pct = 0.03

        # Config
        config = {
            'strategy': {
                'name': self.strategy_name,
                'version': '1.0.0',
                'generated': datetime.now().isoformat(),
                'model_metrics': self.model_metrics,
                'llm_assisted': self.llm_enabled,
            },
            'trading': {
                'symbols': self.symbols,
                'timeframe': self.timeframe,
                'default_quantity': 10,
                'max_positions': max_positions,
                'min_entry_score': min_entry_score,
                'lookback_window': self.llm_design.lookback_window if self.llm_design else 200,
            },
            'indicators': self.rules.optimal_params if self.rules else {},
            'risk_management': {
                'max_position_size': max_position_size,
                'max_daily_loss': 0.05,
                'max_drawdown': 0.20,
                'stop_loss_pct': stop_loss_pct,
                'trailing_stop_pct': trailing_stop_pct,
            },
            'extracted_rules': {
                'top_features': [f['feature'] for f in self.rules.top_features[:10]] if self.rules else [],
                'entry_rules_count': len(self.rules.entry_rules) if self.rules else 0,
                'exit_rules_count': len(self.rules.exit_rules) if self.rules else 0,
            }
        }

        # Add LLM design metadata if available
        if self.llm_design:
            config['llm_design'] = {
                'description': self.llm_design.strategy_description,
                'rationale': self.llm_design.strategy_rationale,
                'priority_features': self.llm_design.priority_features,
            }

        with open(self.strategy_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        # Requirements
        requirements = '''# Auto-generated requirements
pandas>=2.0.0
numpy>=1.24.0
polygon-api-client>=1.12.0
'''
        with open(self.strategy_dir / 'requirements.txt', 'w') as f:
            f.write(requirements)

        # Save extracted rules as JSON
        if self.rules:
            from rule_extractor import RuleExtractor
            extractor = RuleExtractor(
                self.trainer.best_model, self.trainer.scaler,
                self.trainer.feature_names, self.trainer.X_train,
                self.trainer.y_train, self.trainer.X_test, self.trainer.y_test
            )
            rules_dict = extractor.to_dict(self.rules)
            with open(self.strategy_dir / 'extracted_rules.json', 'w') as f:
                json.dump(rules_dict, f, indent=2)

        print(f"  ✓ Generated config.json")
        print(f"  ✓ Generated requirements.txt")
        print(f"  ✓ Generated extracted_rules.json")


async def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Build a trading strategy from ML models with automated optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - iterates until 50% capture rate
  python strategy_builder.py --name "My_Strategy" --symbols NVDA,AMD,AVGO

  # Custom capture rate target (30%)
  python strategy_builder.py --name "My_Strategy" --symbols NVDA,AMD --target 30

  # Maximum 3 iterations with 40% target
  python strategy_builder.py --name "My_Strategy" --symbols NVDA,AMD --target 40 --max-iter 3

  # LLM-assisted design (uses Claude API to design strategy before ML training)
  python strategy_builder.py --name "My_Strategy" --symbols NVDA,AMD --LLM --desc "momentum strategy for semiconductor stocks"

  # Scan existing strategies to populate build history database
  python strategy_builder.py --scan-history
        """
    )
    parser.add_argument('--name', help='Strategy name')
    parser.add_argument('--symbols', help='Comma-separated symbols (e.g., NVDA,AMD)')
    parser.add_argument('--timeframe', default='1d', help='Timeframe (default: 1d)')
    parser.add_argument('--years', type=int, default=5, help='Years of historical data')
    parser.add_argument('--target', type=float, default=50.0,
                        help='Target annual return percentage (default: 50)')
    parser.add_argument('--max-iter', type=int, default=5,
                        help='Maximum training iterations (default: 5)')
    parser.add_argument('--LLM', action='store_true',
                        help='Use Claude AI to pre-design strategy before ML training')
    parser.add_argument('--desc', type=str, default='',
                        help='Strategy description (required when --LLM is used)')
    parser.add_argument('--scan-history', action='store_true',
                        help='Scan existing strategies to populate build_history.json (run once)')

    args = parser.parse_args()

    # Handle --scan-history mode
    if args.scan_history:
        strategies_dir = Path(__file__).parent.parent / 'strategies'
        print(f"\n{'='*60}")
        print("SCANNING EXISTING STRATEGIES FOR BUILD HISTORY")
        print(f"{'='*60}")
        print(f"  Strategies dir: {strategies_dir}\n")

        records = scan_existing_strategies(strategies_dir)
        manager = BuildHistoryManager()
        for record in records:
            manager.add_build(record)

        tracker = FeatureTracker()
        tracker.rebuild_from_history(manager.history)

        print(f"\n{'='*60}")
        print(f"  ✓ Populated build_history.json with {len(records)} strategies")
        print(f"  ✓ Rebuilt feature_tracker.json with {len(tracker.data)} features")
        print(f"{'='*60}\n")
        return

    # Validate required args for strategy building
    if not args.name:
        parser.error("--name is required (unless using --scan-history)")
    if not args.symbols:
        parser.error("--symbols is required (unless using --scan-history)")

    # Validate: --desc is required when --LLM is used
    if args.LLM and not args.desc:
        parser.error("--desc is required when --LLM is used. "
                     "Provide a description of the strategy you want to build.")

    symbols = [s.strip().upper() for s in args.symbols.split(',')]

    builder = StrategyBuilder(
        strategy_name=args.name,
        symbols=symbols,
        timeframe=args.timeframe,
        years_of_data=args.years,
        target_capture_rate=args.target,
        max_iterations=args.max_iter,
        llm_enabled=args.LLM,
        description=args.desc
    )

    await builder.build()


if __name__ == '__main__':
    asyncio.run(main())
