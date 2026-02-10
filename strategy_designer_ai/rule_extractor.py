#!/usr/bin/env python3
"""
Rule Extractor - Extracts trading rules from trained ML models

Analyzes feature importances and decision boundaries to generate
pure algorithmic trading rules without needing ML at runtime.

Author: SignalSynk AI
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ExtractedRule:
    """Represents an extracted trading rule."""
    feature: str
    operator: str  # '<', '>', '<=', '>=', '=='
    threshold: float
    importance: float
    description: str


@dataclass
class StrategyRules:
    """Complete set of extracted strategy rules."""
    name: str
    symbols: List[str]
    timeframe: str
    
    # Key features and their importances
    top_features: List[Dict[str, Any]] = field(default_factory=list)
    
    # Entry rules
    entry_rules: List[ExtractedRule] = field(default_factory=list)
    entry_score_threshold: int = 3
    
    # Exit rules
    exit_rules: List[ExtractedRule] = field(default_factory=list)
    exit_score_threshold: int = 3
    
    # Optimal parameters discovered
    optimal_params: Dict[str, Any] = field(default_factory=dict)
    
    # Model performance metrics
    model_metrics: Dict[str, float] = field(default_factory=dict)


class RuleExtractor:
    """Extracts trading rules from trained ML models."""
    
    def __init__(self, model, scaler, feature_names: List[str], 
                 X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
    def _manual_permutation_importance(self, n_repeats: int = 5) -> np.ndarray:
        """Compute permutation importance manually for non-sklearn models (e.g. PyTorch LSTM).

        Works with any model that has predict() or predict_proba() methods.
        """
        from sklearn.metrics import f1_score as _f1_score

        # Get baseline predictions
        if hasattr(self.model, 'predict'):
            baseline_pred = self.model.predict(self.X_test)
        else:
            baseline_pred = (self.model.predict_proba(self.X_test)[:, 1] > 0.5).astype(int)
        baseline_score = _f1_score(self.y_test, baseline_pred, zero_division=0)

        n_features = self.X_test.shape[1]
        importances = np.zeros(n_features)

        for feat_idx in range(n_features):
            scores = []
            for _ in range(n_repeats):
                X_permuted = self.X_test.copy()
                np.random.shuffle(X_permuted[:, feat_idx])
                if hasattr(self.model, 'predict'):
                    perm_pred = self.model.predict(X_permuted)
                else:
                    perm_pred = (self.model.predict_proba(X_permuted)[:, 1] > 0.5).astype(int)
                perm_score = _f1_score(self.y_test, perm_pred, zero_division=0)
                scores.append(baseline_score - perm_score)
            importances[feat_idx] = np.mean(scores)

        # Clamp negatives to zero
        importances = np.maximum(importances, 0)
        return importances

    def extract_feature_importances(self, top_n: int = 15) -> List[Dict[str, Any]]:
        """Extract top N most important features from the model."""
        importances = []

        # Get feature importances based on model type
        if hasattr(self.model, 'feature_importances_'):
            raw_importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            raw_importances = np.abs(self.model.coef_).flatten()
        else:
            # Check if this is a PyTorch model (LSTM/neural net) which needs special handling
            is_pytorch = False
            try:
                import torch
                is_pytorch = isinstance(self.model, torch.nn.Module)
            except ImportError:
                pass

            if is_pytorch:
                # PyTorch models (especially LSTM) expect 3D tensor input and can't do
                # standard per-feature permutation on flat 2D numpy arrays.
                # Use uniform importances — LLM priority features will be boosted in merge step.
                print("    PyTorch model detected — using uniform importances "
                      "(LLM priority features will be boosted)")
                raw_importances = np.ones(len(self.feature_names)) / len(self.feature_names)
            else:
                # Try sklearn permutation importance, then manual fallback
                try:
                    from sklearn.inspection import permutation_importance
                    result = permutation_importance(self.model, self.X_test, self.y_test,
                                                   n_repeats=10, random_state=42, n_jobs=-1)
                    raw_importances = result.importances_mean
                except Exception:
                    print("    Using manual permutation importance (non-sklearn model)...")
                    raw_importances = self._manual_permutation_importance(n_repeats=5)
        
        # Create list of (feature, importance) tuples
        for i, (name, imp) in enumerate(zip(self.feature_names, raw_importances)):
            importances.append({
                'feature': name,
                'importance': float(imp),
                'rank': 0  # Will be set after sorting
            })
        
        # Sort by importance and assign ranks
        importances.sort(key=lambda x: x['importance'], reverse=True)
        for i, item in enumerate(importances):
            item['rank'] = i + 1
        
        return importances[:top_n]

    def extract_decision_thresholds(self, top_features: List[Dict], 
                                    max_depth: int = 4) -> Dict[str, List[float]]:
        """Extract optimal thresholds for top features using decision tree analysis."""
        thresholds = {}
        
        # Train a shallow decision tree to find key split points
        dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        dt.fit(self.X_train, self.y_train)
        
        # Extract thresholds from the tree
        tree = dt.tree_
        for feature_info in top_features:
            feature_name = feature_info['feature']
            if feature_name in self.feature_names:
                feature_idx = self.feature_names.index(feature_name)
                
                # Find all thresholds used for this feature in the tree
                feature_thresholds = []
                for node_id in range(tree.node_count):
                    if tree.feature[node_id] == feature_idx:
                        feature_thresholds.append(float(tree.threshold[node_id]))
                
                if feature_thresholds:
                    thresholds[feature_name] = sorted(set(feature_thresholds))
        
        return thresholds

    def _inverse_transform_value(self, value: float, feature_idx: int) -> float:
        """Inverse transform a single scaled value back to original scale."""
        if self.scaler is None:
            return value

        try:
            # RobustScaler uses: X_scaled = (X - center_) / scale_
            # So: X_original = X_scaled * scale_ + center_
            center = self.scaler.center_[feature_idx]
            scale = self.scaler.scale_[feature_idx]
            return value * scale + center
        except (AttributeError, IndexError):
            return value

    def analyze_feature_distributions(self, top_features: List[Dict]) -> Dict[str, Dict]:
        """Analyze feature distributions for buy vs non-buy signals.

        Returns distributions in ORIGINAL (unscaled) feature space for use
        in the generated strategy which calculates raw indicator values.
        """
        distributions = {}

        for feature_info in top_features:
            feature_name = feature_info['feature']
            if feature_name not in self.feature_names:
                continue

            feature_idx = self.feature_names.index(feature_name)
            feature_values = self.X_train[:, feature_idx]

            # Split by label
            buy_values = feature_values[self.y_train == 1]
            no_buy_values = feature_values[self.y_train == 0]

            # Calculate percentiles in scaled space
            buy_25th_scaled = float(np.percentile(buy_values, 25))
            buy_75th_scaled = float(np.percentile(buy_values, 75))
            buy_mean_scaled = float(np.mean(buy_values))
            no_buy_25th_scaled = float(np.percentile(no_buy_values, 25))
            no_buy_75th_scaled = float(np.percentile(no_buy_values, 75))
            no_buy_mean_scaled = float(np.mean(no_buy_values))

            # Inverse transform to get real-world values
            distributions[feature_name] = {
                'buy_mean': self._inverse_transform_value(buy_mean_scaled, feature_idx),
                'buy_std': float(np.std(buy_values)),  # Keep scaled for comparison purposes
                'buy_median': self._inverse_transform_value(float(np.median(buy_values)), feature_idx),
                'buy_25th': self._inverse_transform_value(buy_25th_scaled, feature_idx),
                'buy_75th': self._inverse_transform_value(buy_75th_scaled, feature_idx),
                'no_buy_mean': self._inverse_transform_value(no_buy_mean_scaled, feature_idx),
                'no_buy_std': float(np.std(no_buy_values)),  # Keep scaled
                'no_buy_median': self._inverse_transform_value(float(np.median(no_buy_values)), feature_idx),
                'no_buy_25th': self._inverse_transform_value(no_buy_25th_scaled, feature_idx),
                'no_buy_75th': self._inverse_transform_value(no_buy_75th_scaled, feature_idx),
                'direction': 'higher' if np.mean(buy_values) > np.mean(no_buy_values) else 'lower'
            }

        return distributions

    def generate_entry_rules(self, top_features: List[Dict],
                             thresholds: Dict[str, List[float]],
                             distributions: Dict[str, Dict]) -> List[ExtractedRule]:
        """Generate entry rules from feature analysis."""
        rules = []

        for feature_info in top_features[:10]:  # Top 10 features for entry
            feature_name = feature_info['feature']
            importance = feature_info['importance']

            if feature_name not in distributions:
                continue

            dist = distributions[feature_name]
            direction = dist['direction']

            # Determine optimal threshold based on distribution analysis
            if direction == 'higher':
                # For features where higher = buy signal
                threshold = dist['buy_25th']  # Use 25th percentile of buy signals
                operator = '>='
                desc = f"Enter when {feature_name} >= {threshold:.4f} (bullish signal)"
            else:
                # For features where lower = buy signal
                threshold = dist['buy_75th']  # Use 75th percentile of buy signals
                operator = '<='
                desc = f"Enter when {feature_name} <= {threshold:.4f} (oversold signal)"

            rules.append(ExtractedRule(
                feature=feature_name,
                operator=operator,
                threshold=round(threshold, 4),
                importance=importance,
                description=desc
            ))

        return rules

    def generate_exit_rules(self, top_features: List[Dict],
                            distributions: Dict[str, Dict]) -> List[ExtractedRule]:
        """Generate exit rules (inverse of entry conditions)."""
        rules = []

        for feature_info in top_features[:8]:  # Top 8 features for exit
            feature_name = feature_info['feature']
            importance = feature_info['importance']

            if feature_name not in distributions:
                continue

            dist = distributions[feature_name]
            direction = dist['direction']

            # Exit rules are inverse of entry
            if direction == 'higher':
                # Exit when feature drops (was bullish, now losing momentum)
                threshold = dist['no_buy_75th'] if 'no_buy_75th' in dist else dist['no_buy_mean']
                operator = '<='
                desc = f"Exit when {feature_name} <= {threshold:.4f} (momentum fading)"
            else:
                # Exit when feature rises (was oversold, now overbought)
                threshold = dist['no_buy_25th'] if 'no_buy_25th' in dist else dist['no_buy_mean']
                operator = '>='
                desc = f"Exit when {feature_name} >= {threshold:.4f} (overbought)"

            rules.append(ExtractedRule(
                feature=feature_name,
                operator=operator,
                threshold=round(threshold, 4),
                importance=importance,
                description=desc
            ))

        return rules

    def extract_all_rules(self, strategy_name: str, symbols: List[str],
                          timeframe: str, model_metrics: Dict[str, float] = None,
                          llm_design=None) -> StrategyRules:
        """Main method to extract all rules from the trained model.

        Args:
            llm_design: Optional LLMStrategyDesign from Claude API. When provided,
                        LLM-prioritized features get boosted importance and LLM rules
                        are merged with ML-extracted rules.
        """
        print(f"\n{'='*60}")
        print(f"EXTRACTING TRADING RULES")
        if llm_design:
            print(f"  (with LLM design guidance: {len(llm_design.priority_features)} priority features)")
        print(f"{'='*60}")

        # Step 1: Extract feature importances
        print("  Analyzing feature importances...", flush=True)
        top_features = self.extract_feature_importances(top_n=15)

        # Boost LLM priority features if provided
        if llm_design and llm_design.priority_features:
            print(f"  Boosting LLM priority features: {llm_design.priority_features[:5]}...")
            llm_feature_set = set(llm_design.priority_features)
            for feature_info in top_features:
                if feature_info['feature'] in llm_feature_set:
                    feature_info['importance'] *= 1.5
            # Re-sort and re-rank after boosting
            top_features.sort(key=lambda x: x['importance'], reverse=True)
            for i, item in enumerate(top_features):
                item['rank'] = i + 1

        print(f"    Top 5 features: {[f['feature'] for f in top_features[:5]]}")

        # Step 2: Extract decision thresholds
        print("  Extracting decision thresholds...", flush=True)
        thresholds = self.extract_decision_thresholds(top_features)

        # Step 3: Analyze feature distributions
        print("  Analyzing feature distributions...", flush=True)
        distributions = self.analyze_feature_distributions(top_features)

        # Step 4: Generate entry rules
        print("  Generating entry rules...", flush=True)
        entry_rules = self.generate_entry_rules(top_features, thresholds, distributions)

        # Merge LLM entry rules for features not already covered by ML
        if llm_design and llm_design.entry_rules:
            ml_entry_features = {r.feature for r in entry_rules}
            for llm_rule in llm_design.entry_rules:
                if llm_rule.get('feature') not in ml_entry_features:
                    entry_rules.append(ExtractedRule(
                        feature=llm_rule['feature'],
                        operator=llm_rule.get('operator', '>='),
                        threshold=llm_rule.get('threshold', 0.0),
                        importance=0.5,  # Default importance for LLM-only rules
                        description=llm_rule.get('description', f"LLM-designed rule for {llm_rule['feature']}")
                    ))
            print(f"    Merged with LLM rules: {len(entry_rules)} total entry rules")
        else:
            print(f"    Generated {len(entry_rules)} entry rules")

        # Step 5: Generate exit rules
        print("  Generating exit rules...", flush=True)
        exit_rules = self.generate_exit_rules(top_features, distributions)

        # Merge LLM exit rules for features not already covered by ML
        if llm_design and llm_design.exit_rules:
            ml_exit_features = {r.feature for r in exit_rules}
            for llm_rule in llm_design.exit_rules:
                if llm_rule.get('feature') not in ml_exit_features:
                    exit_rules.append(ExtractedRule(
                        feature=llm_rule['feature'],
                        operator=llm_rule.get('operator', '>='),
                        threshold=llm_rule.get('threshold', 0.0),
                        importance=0.5,
                        description=llm_rule.get('description', f"LLM-designed rule for {llm_rule['feature']}")
                    ))
            print(f"    Merged with LLM rules: {len(exit_rules)} total exit rules")
        else:
            print(f"    Generated {len(exit_rules)} exit rules")

        # Step 6: Extract optimal parameters
        optimal_params = self._extract_optimal_params(distributions)

        # Use LLM score thresholds if provided
        entry_score_threshold = llm_design.entry_score_threshold if llm_design else 3
        exit_score_threshold = llm_design.exit_score_threshold if llm_design else 3

        # Create strategy rules object
        strategy_rules = StrategyRules(
            name=strategy_name,
            symbols=symbols,
            timeframe=timeframe,
            top_features=top_features,
            entry_rules=entry_rules,
            entry_score_threshold=entry_score_threshold,
            exit_rules=exit_rules,
            exit_score_threshold=exit_score_threshold,
            optimal_params=optimal_params,
            model_metrics=model_metrics or {}
        )

        print(f"\n✓ Rule extraction complete!")
        return strategy_rules

    def _extract_optimal_params(self, distributions: Dict[str, Dict]) -> Dict[str, Any]:
        """Extract optimal indicator parameters from distributions."""
        params = {}

        # Map feature names to indicator parameters
        param_mapping = {
            'RSI': 'rsi_oversold',
            'STOCH_K': 'stoch_oversold',
            'MFI': 'mfi_oversold',
            'CCI': 'cci_oversold',
            'ATR_PCT': 'atr_threshold',
            'ADX': 'adx_threshold',
        }

        for feature_name, dist in distributions.items():
            for key, param_name in param_mapping.items():
                if key in feature_name.upper():
                    if dist['direction'] == 'lower':
                        params[param_name] = round(dist['buy_75th'], 2)
                    else:
                        params[param_name] = round(dist['buy_25th'], 2)
                    break

        return params

    def to_dict(self, strategy_rules: StrategyRules) -> Dict[str, Any]:
        """Convert StrategyRules to a dictionary for JSON serialization."""
        return {
            'name': strategy_rules.name,
            'symbols': strategy_rules.symbols,
            'timeframe': strategy_rules.timeframe,
            'top_features': strategy_rules.top_features,
            'entry_rules': [
                {
                    'feature': r.feature,
                    'operator': r.operator,
                    'threshold': r.threshold,
                    'importance': r.importance,
                    'description': r.description
                } for r in strategy_rules.entry_rules
            ],
            'exit_rules': [
                {
                    'feature': r.feature,
                    'operator': r.operator,
                    'threshold': r.threshold,
                    'importance': r.importance,
                    'description': r.description
                } for r in strategy_rules.exit_rules
            ],
            'entry_score_threshold': strategy_rules.entry_score_threshold,
            'exit_score_threshold': strategy_rules.exit_score_threshold,
            'optimal_params': strategy_rules.optimal_params,
            'model_metrics': strategy_rules.model_metrics
        }

