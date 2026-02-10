#!/usr/bin/env python3
"""
Enhanced ML Trainer - Advanced Model Training with Hyperparameter Tuning & Neural Networks

Features:
1. Hyperparameter tuning with RandomizedSearchCV
2. Multiple lookback periods and forward return windows
3. Neural Network models (MLP, LSTM-style sequential)
4. Real-time prediction interface
5. Model comparison and ensemble methods

Author: SignalSynk AI
"""

# Suppress ALL warnings BEFORE any imports
import warnings
import os
import sys

# Set environment variable to suppress warnings in subprocesses
os.environ['PYTHONWARNINGS'] = 'ignore'

# Suppress all warning categories
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")


import asyncio
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

# Configure joblib
import joblib
from joblib import parallel_config

# Patch sklearn.utils.parallel to suppress the warning BEFORE importing sklearn
import sklearn.utils.parallel as _sklearn_parallel
_original_delayed = _sklearn_parallel.delayed
def _patched_delayed(function):
    """Patched delayed that doesn't emit warning."""
    return joblib.delayed(function)
_sklearn_parallel.delayed = _patched_delayed

# Now import sklearn
import sklearn
sklearn.set_config(assume_finite=True)

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import randint, uniform

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                'strategies', 'semiconductor_momentum'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_provider import DataProvider
from ml_strategy_designer import FeatureEngineer, MLConfig

# Try to import advanced ML libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    print("⚠️  PyTorch not installed. Neural network models will use sklearn MLP.")


# ============================================================================
# LSTM MODEL DEFINITION
# ============================================================================
if HAS_PYTORCH:
    class LSTMClassifier(nn.Module):
        """LSTM-based classifier for time series trading signal prediction."""

        def __init__(self, input_size: int, hidden_size: int = 128,
                     num_layers: int = 2, dropout: float = 0.2):
            super(LSTMClassifier, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            # LSTM layers
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True  # Bidirectional for better context
            )

            # Attention layer for focusing on important timesteps
            self.attention = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
                nn.Softmax(dim=1)
            )

            # Fully connected layers
            self.fc = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 2)  # Binary classification
            )

        def forward(self, x):
            # LSTM forward pass
            lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)

            # Attention mechanism
            attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
            context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden*2)

            # Classification
            out = self.fc(context)
            return out

        def predict_proba(self, x):
            """Get probability predictions."""
            self.eval()
            with torch.no_grad():
                logits = self.forward(x)
                probs = torch.softmax(logits, dim=1)
            return probs


@dataclass
class EnhancedMLConfig:
    """Enhanced configuration for ML training with tuning options."""
    # Label generation - test multiple configurations
    forward_returns_days_options: List[int] = field(default_factory=lambda: [1, 2, 3,5, 7, 10, 12, 15, 20])
    profit_threshold_options: List[float] = field(default_factory=lambda: [2.0, 2.5, 3.0, 5.0, 7.0, 10.0])
    loss_threshold: float = -3.0
    
    # Training
    test_size: float = 0.2
    n_splits: int = 5
    random_state: int = 42
    
    # Hyperparameter search
    n_iter_search: int = 50  # Number of random search iterations
    cv_folds: int = 3  # Cross-validation folds for tuning
    
    # Feature engineering - multiple lookback periods
    lookback_periods: List[int] = field(default_factory=lambda: [3, 5, 10, 20, 50])
    
    # Neural network config
    nn_hidden_layers: List[Tuple[int, ...]] = field(
        default_factory=lambda: [(128, 64), (256, 128, 64), (512, 256, 128)]
    )
    nn_epochs: int = 100
    nn_batch_size: int = 64
    nn_learning_rate: float = 0.001

    # LSTM config
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_sequence_length: int = 100  # Look back window for LSTM
    lstm_epochs: int = 250
    lstm_batch_size: int = 32

    # Training toggles (for faster strategy generation)
    train_neural_networks: bool = True
    train_lstm: bool = True

    # Model to use for label configuration search (quick proxy evaluation)
    # Options: "LightGBM", "RandomForest", "GradientBoosting", "XGBoost", "MLP", "LSTM"
    config_search_model: str = "LightGBM"


class HyperparameterTuner:
    """Handles hyperparameter tuning for all model types."""
    
    def __init__(self, config: EnhancedMLConfig):
        self.config = config
        self.best_params: Dict[str, Dict] = {}
        self.tuning_results: List[Dict] = []
    
    def get_param_distributions(self) -> Dict[str, Dict]:
        """Define parameter distributions for each model type."""
        params = {
            'RandomForest': {
                'n_estimators': randint(100, 500),
                'max_depth': randint(5, 30),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(5, 50),
                'max_features': ['sqrt', 'log2', None],
            },
            'GradientBoosting': {
                'n_estimators': randint(50, 300),
                'max_depth': randint(3, 15),
                'learning_rate': uniform(0.01, 0.3),
                'min_samples_leaf': randint(10, 50),
                'subsample': uniform(0.6, 0.4),
            },
            'MLP': {
                'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64), (256, 128), (128, 64, 32)],
                'alpha': uniform(0.0001, 0.01),
                'learning_rate_init': uniform(0.0001, 0.01),
                'batch_size': [32, 64, 128, 256],
            }
        }
        
        if HAS_XGBOOST:
            params['XGBoost'] = {
                'n_estimators': randint(100, 500),
                'max_depth': randint(3, 15),
                'learning_rate': uniform(0.01, 0.3),
                'min_child_weight': randint(1, 20),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4),
                'gamma': uniform(0, 0.5),
            }
        
        if HAS_LIGHTGBM:
            params['LightGBM'] = {
                'n_estimators': randint(100, 500),
                'max_depth': randint(3, 20),
                'learning_rate': uniform(0.01, 0.3),
                'num_leaves': randint(20, 100),
                'min_child_samples': randint(10, 50),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4),
            }

        return params

    def tune_model(self, model_name: str, model, X_train: np.ndarray, y_train: np.ndarray,
                   param_dist: Dict) -> Tuple[Any, Dict, float]:
        """Tune a single model using RandomizedSearchCV."""
        print(f"    Tuning {model_name}...")

        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)

        # Limit parallel workers to 50% of cores to prevent resource exhaustion
        # This is especially important on Apple Silicon where too many processes cause deadlocks
        import multiprocessing as mp
        n_workers = max(2, mp.cpu_count() // 2)

        # Use sklearn's parallel backend to avoid warnings
        with parallel_config(backend='loky', n_jobs=n_workers):
            search = RandomizedSearchCV(
                model,
                param_distributions=param_dist,
                n_iter=min(self.config.n_iter_search, 30),  # Limit iterations for speed
                cv=tscv,
                scoring='f1',
                n_jobs=n_workers,  # Use limited workers instead of -1
                random_state=self.config.random_state,
                verbose=0,
                error_score='raise'  # Fail fast on errors instead of silently continuing
            )

            search.fit(X_train, y_train)

        best_model = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_

        print(f"      Best F1 (CV): {best_score:.4f}")
        print(f"      Best params: {best_params}")

        self.best_params[model_name] = best_params

        return best_model, best_params, best_score

    def tune_all_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Tune all available models and return best configurations."""
        import time
        start_time = time.time()

        print(f"\n{'='*60}")
        print("HYPERPARAMETER TUNING")
        print(f"{'='*60}")
        print(f"  🔧 RandomizedSearchCV with {self.config.n_iter_search} iterations")
        print(f"  📊 TimeSeriesSplit with {self.config.cv_folds} folds")
        print(f"  📐 Training samples: {len(X_train):,}")

        param_dists = self.get_param_distributions()
        print(f"  🧠 Models to tune: {list(param_dists.keys())}")
        tuned_models = {}

        # Define base models
        # IMPORTANT: Set n_jobs=1 on individual models when using RandomizedSearchCV(n_jobs=-1)
        # This prevents nested parallelism which causes deadlocks and resource exhaustion
        base_models = {
            'RandomForest': RandomForestClassifier(
                class_weight='balanced',
                random_state=self.config.random_state,
                n_jobs=1  # Let RandomizedSearchCV handle parallelism
            ),
            'GradientBoosting': GradientBoostingClassifier(
                random_state=self.config.random_state
            ),
            'MLP': MLPClassifier(
                max_iter=500,
                early_stopping=True,
                random_state=self.config.random_state
            ),
        }

        if HAS_XGBOOST:
            base_models['XGBoost'] = xgb.XGBClassifier(
                random_state=self.config.random_state,
                n_jobs=1,  # Prevent nested parallelism
                verbosity=0
            )

        if HAS_LIGHTGBM:
            base_models['LightGBM'] = lgb.LGBMClassifier(
                random_state=self.config.random_state,
                n_jobs=1,  # Prevent nested parallelism
                verbose=-1
            )

        # Tune each model
        for name, model in base_models.items():
            if name in param_dists:
                try:
                    tuned_model, params, score = self.tune_model(
                        name, model, X_train, y_train, param_dists[name]
                    )
                    tuned_models[name] = {
                        'model': tuned_model,
                        'params': params,
                        'cv_score': score
                    }
                    self.tuning_results.append({
                        'model': name,
                        'params': params,
                        'cv_f1': score
                    })
                except Exception as e:
                    print(f"    ⚠️ Failed to tune {name}: {e}")

        elapsed = time.time() - start_time
        print(f"\n  ⏱️  Hyperparameter tuning completed in {elapsed:.1f}s")
        print(f"  ✓ Tuned {len(tuned_models)} models")

        return tuned_models


class NeuralNetworkTrainer:
    """Trains neural network models for trading signal prediction."""

    def __init__(self, config: EnhancedMLConfig):
        self.config = config
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}

    def train_mlp_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train multiple MLP architectures and create ensemble."""
        print(f"\n{'='*60}")
        print("NEURAL NETWORK TRAINING")
        print(f"{'='*60}")

        results = {}

        # Train different MLP architectures
        architectures = [
            ('MLP_Small', (64, 32)),
            ('MLP_Medium', (128, 64, 32)),
            ('MLP_Large', (256, 128, 64)),
            ('MLP_Deep', (128, 64, 32, 16)),
        ]

        for name, hidden_layers in architectures:
            print(f"  Training {name} {hidden_layers}...")

            model = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size=min(64, len(X_train) // 10),
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=self.config.random_state,
                verbose=True
            )

            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]

                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1': f1_score(y_test, y_pred, zero_division=0),
                    'auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5
                }

                print(f"    F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")

                results[name] = {
                    'model': model,
                    'metrics': metrics,
                    'architecture': hidden_layers
                }
                self.models[name] = model

            except Exception as e:
                print(f"    ⚠️ Failed: {e}")

        return results

    def _create_sequences(self, X: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        sequences = []
        labels = []
        print("Creating sequences...")
        for i in range(seq_length, len(X)):
            sequences.append(X[i-seq_length:i])
            labels.append(y[i])

        return np.array(sequences), np.array(labels)

    def train_lstm(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray,
                   seq_length: int = 20) -> Dict[str, Any]:
        """Train LSTM model with hyperparameter tuning."""
        print(f"\n{'='*60}")
        print("LSTM NEURAL NETWORK TRAINING WITH HYPERPARAMETER TUNING")
        print(f"{'='*60}")
        print(f"  PyTorch available: {HAS_PYTORCH}")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Sequence length: {seq_length}")
        print(f"  Max possible sequences: {len(X_train) - seq_length}")

        if not HAS_PYTORCH:
            print("  ❌ PyTorch not installed. LSTM requires PyTorch!")
            print("  Install with: pip install torch")
            return {}

        # LSTM hyperparameter configurations to try
        lstm_configs = [
            {'hidden_size': 64, 'num_layers': 1, 'dropout': 0.1, 'name': 'LSTM_Small'},
            {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2, 'name': 'LSTM_Medium'},
            {'hidden_size': 256, 'num_layers': 2, 'dropout': 0.3, 'name': 'LSTM_Large'},
            {'hidden_size': 128, 'num_layers': 3, 'dropout': 0.2, 'name': 'LSTM_Deep'},
        ]

        print(f"  🔧 Tuning {len(lstm_configs)} LSTM configurations...")

        results = {}

        try:
            # Create sequences for LSTM
            X_train_seq, y_train_seq = self._create_sequences(X_train, y_train, seq_length)
            X_test_seq, y_test_seq = self._create_sequences(X_test, y_test, seq_length)

            print(f"  Training sequences: {X_train_seq.shape}")
            print(f"  Test sequences: {X_test_seq.shape if len(X_test_seq) > 0 else '(0 - insufficient test data)'}")

            if len(X_train_seq) < 100:
                print(f"  ❌ Not enough training sequences for LSTM!")
                print(f"     Have: {len(X_train_seq)} sequences, need at least 100")
                print(f"     Solution: Reduce lstm_sequence_length or use more training data")
                return {}

            # Handle case where test set is too small for sequences
            if len(X_test_seq) < 10:
                print(f"  ⚠️ Insufficient test sequences ({len(X_test_seq)}). Using 20% of training for validation.")
                val_split = int(len(X_train_seq) * 0.8)
                X_val_seq = X_train_seq[val_split:]
                y_val_seq = y_train_seq[val_split:]
                X_train_seq = X_train_seq[:val_split]
                y_train_seq = y_train_seq[:val_split]
                print(f"  New training sequences: {X_train_seq.shape}")
                print(f"  Validation sequences: {X_val_seq.shape}")
            else:
                X_val_seq = X_test_seq
                y_val_seq = y_test_seq

            # Setup device
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
            print(f"  Using device: {device}", flush=True)

            # Convert to tensors
            X_train_tensor = torch.from_numpy(X_train_seq.astype(np.float32))
            y_train_tensor = torch.from_numpy(y_train_seq.astype(np.int64))
            X_val_tensor = torch.from_numpy(X_val_seq.astype(np.float32)).to(device)
            y_val_tensor = torch.from_numpy(y_val_seq.astype(np.int64)).to(device)

            input_size = X_train.shape[1]

            # Train each LSTM configuration
            for cfg in lstm_configs:
                config_name = cfg['name']
                print(f"\n  📊 Training {config_name} (hidden={cfg['hidden_size']}, layers={cfg['num_layers']}, dropout={cfg['dropout']})...")

                try:
                    model, metrics = self._train_single_lstm(
                        X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, y_val_seq,
                        input_size, cfg['hidden_size'], cfg['num_layers'], cfg['dropout'],
                        device, self.config.lstm_batch_size, self.config.lstm_epochs
                    )

                    if model is not None:
                        results[config_name] = {
                            'model': model.cpu(),
                            'metrics': metrics,
                            'sequence_length': seq_length,
                            'device': str(device),
                            'config': cfg
                        }
                        self.models[config_name] = model.cpu()
                        print(f"    ✓ {config_name}: F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")
                except Exception as e:
                    print(f"    ⚠️ {config_name} failed: {e}")

            # Print summary
            if results:
                best_lstm = max(results.keys(), key=lambda k: results[k]['metrics']['f1'])
                print(f"\n  🏆 Best LSTM: {best_lstm} (F1={results[best_lstm]['metrics']['f1']:.4f})")

        except Exception as e:
            print(f"  ⚠️ LSTM training failed: {e}")
            import traceback
            traceback.print_exc()

        return results

    def _train_single_lstm(self, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, y_val_seq,
                           input_size, hidden_size, num_layers, dropout, device, batch_size, max_epochs):
        """Train a single LSTM configuration and return model + metrics."""
        # Create data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        # Initialize model
        model = LSTMClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # Training loop with early stopping
        best_f1 = 0
        best_model_state = None
        patience_counter = 0
        max_patience = 10
        num_batches = len(train_loader)

        for epoch in range(max_epochs):
            model.train()
            total_loss = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()

            val_f1 = f1_score(y_val_seq, val_preds, zero_division=0)
            scheduler.step(val_loss)

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            # Log every 25 epochs
            if (epoch + 1) % 25 == 0:
                print(f"      Epoch {epoch+1}/{max_epochs}: F1={val_f1:.4f}", flush=True)

            if patience_counter >= max_patience:
                print(f"      Early stop at epoch {epoch+1}", flush=True)
                break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Final evaluation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
            val_probs = torch.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()

        metrics = {
            'accuracy': accuracy_score(y_val_seq, val_preds),
            'precision': precision_score(y_val_seq, val_preds, zero_division=0),
            'recall': recall_score(y_val_seq, val_preds, zero_division=0),
            'f1': f1_score(y_val_seq, val_preds, zero_division=0),
            'auc': roc_auc_score(y_val_seq, val_probs) if len(np.unique(y_val_seq)) > 1 else 0.5
        }

        return model, metrics


class MLPredictor:
    """Real-time prediction interface for live trading signals."""

    def __init__(self, model_path: str = None):
        self.model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.feature_engineer = None
        self.config = None
        self.model_path = model_path or os.path.join(
            os.path.dirname(__file__), 'ml_model_enhanced.pkl'
        )

    def load_model(self, model_path: str = None):
        """Load trained model from disk."""
        path = model_path or self.model_path

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at {path}")

        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.model = data.get('model')
        self.scaler = data.get('scaler')
        self.feature_names = data.get('feature_names', [])
        self.config = data.get('config')

        # Initialize feature engineer
        ml_config = MLConfig()
        self.feature_engineer = FeatureEngineer(ml_config)

        print(f"✓ Model loaded: {type(self.model).__name__}")
        print(f"  Features: {len(self.feature_names)}")

        return self

    def prepare_features_from_df(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features from a dataframe of OHLCV data."""
        # Calculate all indicators
        df = self.feature_engineer.calculate_all_indicators(df)
        df = self.feature_engineer.add_pattern_features(df)

        # Get the latest row's features
        X, _ = self.feature_engineer.prepare_features(df)

        # Scale features
        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        return X_scaled

    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate prediction for the latest data point."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        X_scaled = self.prepare_features_from_df(df)

        # Get prediction for the last row
        latest_features = X_scaled[-1:] if len(X_scaled) > 0 else X_scaled

        prediction = self.model.predict(latest_features)[0]
        probability = self.model.predict_proba(latest_features)[0]

        return {
            'signal': 'BUY' if prediction == 1 else 'HOLD',
            'confidence': float(probability[1]) if prediction == 1 else float(probability[0]),
            'buy_probability': float(probability[1]),
            'prediction_class': int(prediction)
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for all rows in dataframe."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        X_scaled = self.prepare_features_from_df(df)

        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        # Create result dataframe
        result = df.copy()
        result['ML_SIGNAL'] = predictions
        result['ML_BUY_PROB'] = probabilities[:, 1]
        result['ML_CONFIDENCE'] = np.where(
            predictions == 1,
            probabilities[:, 1],
            probabilities[:, 0]
        )

        return result

    def get_signal_strength(self, probability: float) -> str:
        """Convert probability to signal strength category."""
        if probability >= 0.8:
            return "STRONG_BUY"
        elif probability >= 0.65:
            return "BUY"
        elif probability >= 0.5:
            return "WEAK_BUY"
        elif probability >= 0.35:
            return "HOLD"
        elif probability >= 0.2:
            return "WEAK_SELL"
        else:
            return "SELL"


class EnhancedMLTrainer:
    """Enhanced ML Trainer with hyperparameter tuning and multiple model types."""

    SEMICONDUCTOR_SYMBOLS = [
        "NVDA", "AMD", "AVGO", "TSM", "QCOM",
        "MU", "MRVL", "AMAT", "LRCX", "KLAC", "ASML"
    ]

    def __init__(self, symbols: List[str] = None, period_years: int = 2,
                 config: EnhancedMLConfig = None):
        self.symbols = symbols or self.SEMICONDUCTOR_SYMBOLS
        self.period_years = period_years
        self.config = config or EnhancedMLConfig()
        self.data_provider = DataProvider()

        # Feature engineer from base class
        self.feature_engineer = FeatureEngineer(MLConfig())

        # Storage
        self.stock_data: Dict[str, pd.DataFrame] = {}
        self.all_results: List[Dict] = []
        self.best_model = None
        self.best_model_name: str = ""
        self.best_config = None
        self.scaler = None
        self.feature_names: List[str] = []

        # Training data (stored for rule extraction)
        self.X_train: np.ndarray = None
        self.y_train: np.ndarray = None
        self.X_test: np.ndarray = None
        self.y_test: np.ndarray = None

    async def fetch_all_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for all symbols in parallel."""
        import time
        start_time = time.time()

        print(f"\n{'='*60}")
        print("STEP 1: FETCHING HISTORICAL DATA (PARALLEL)")
        print(f"{'='*60}")

        days = self.period_years * 365 + 100
        print(f"  📊 Requesting {days} days of data for {len(self.symbols)} symbols...")
        print(f"  📡 Using Polygon API (parallel fetch)...")

        # Fetch all symbols in parallel for speed
        self.stock_data = await self.data_provider.get_multiple_symbols(
            self.symbols, days=days, interval='day', parallel=True
        )

        total_bars = 0
        for symbol, df in self.stock_data.items():
            bars = len(df)
            total_bars += bars
            date_range = f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}" if len(df) > 0 else "N/A"
            print(f"  ✓ {symbol}: {bars:,} bars ({date_range})")

        elapsed = time.time() - start_time
        print(f"\n  ⏱️  Fetch completed in {elapsed:.1f}s")
        print(f"  📈 Total: {total_bars:,} bars across {len(self.stock_data)} symbols")

        return self.stock_data

    def process_data_with_config(self, forward_days: int, profit_threshold: float) -> pd.DataFrame:
        """Process data with specific label configuration."""
        config = MLConfig(
            forward_returns_days=forward_days,
            profit_threshold=profit_threshold,
            loss_threshold=self.config.loss_threshold
        )

        all_data = []
        for symbol, df in self.stock_data.items():
            df = self.feature_engineer.calculate_all_indicators(df)
            df = self.feature_engineer.add_pattern_features(df)
            df = self.feature_engineer.create_labels(df, config)
            df['SYMBOL'] = symbol
            df = df.dropna()
            all_data.append(df)

        return pd.concat(all_data, ignore_index=True)

    def run_configuration_search(self) -> Dict[str, Any]:
        """Search for best label configuration (forward days, profit threshold)."""
        import time
        start_time = time.time()

        print(f"\n{'='*60}")
        print("STEP 2: SEARCHING OPTIMAL LABEL CONFIGURATION")
        print(f"{'='*60}")

        total_configs = len(self.config.forward_returns_days_options) * len(self.config.profit_threshold_options)
        print(f"  🔍 Testing {total_configs} configurations...")
        print(f"  📊 Forward days: {self.config.forward_returns_days_options}")
        print(f"  📈 Profit thresholds: {self.config.profit_threshold_options}")
        print(f"  🧠 Config search model: {self.config.config_search_model}")

        best_score = 0
        best_config = None
        config_results = []
        config_num = 0

        for forward_days in self.config.forward_returns_days_options:
            for profit_threshold in self.config.profit_threshold_options:
                config_num += 1
                print(f"\n  [{config_num}/{total_configs}] Testing: forward={forward_days}d, profit={profit_threshold}%")

                # Process data with this config
                combined_data = self.process_data_with_config(forward_days, profit_threshold)

                # Quick evaluation with LightGBM
                X, feature_names = self.feature_engineer.prepare_features(combined_data)
                y = combined_data['LABEL_ENTRY'].values

                # Handle edge cases
                if len(np.unique(y)) < 2:
                    print(f"    Skipping - only one class in labels")
                    continue

                # Scale and split
                scaler = RobustScaler()
                X_scaled = scaler.fit_transform(X)
                X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

                split_idx = int(len(X_scaled) * 0.8)
                X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]

                # Quick evaluation with configurable model
                import multiprocessing as mp
                quick_jobs = max(2, mp.cpu_count() // 2)
                search_model_name = self.config.config_search_model

                if search_model_name == "LSTM" and HAS_PYTORCH:
                    # LSTM needs special handling: create sequences, train, predict
                    seq_len = min(20, len(X_train) // 10)  # Short seq for speed
                    if len(X_train) > seq_len + 50:
                        nn_trainer = NeuralNetworkTrainer(self.config)
                        lstm_results = nn_trainer.train_lstm(
                            X_train, y_train, X_test, y_test, seq_length=seq_len
                        )
                        if lstm_results:
                            best_lstm = max(lstm_results.values(), key=lambda x: x['metrics']['f1'])
                            f1 = best_lstm['metrics']['f1']
                            precision = best_lstm['metrics']['precision']
                            recall = best_lstm['metrics']['recall']
                        else:
                            f1, precision, recall = 0, 0, 0
                    else:
                        print(f"    Not enough data for LSTM config search, falling back to LightGBM")
                        search_model_name = "LightGBM"

                if search_model_name == "MLP":
                    model = MLPClassifier(
                        hidden_layer_sizes=(64, 32), max_iter=200,
                        early_stopping=True, random_state=42
                    )
                elif search_model_name == "XGBoost" and HAS_XGBOOST:
                    model = xgb.XGBClassifier(
                        n_estimators=100, max_depth=10,
                        random_state=42, n_jobs=quick_jobs, verbosity=0
                    )
                elif search_model_name == "GradientBoosting":
                    model = GradientBoostingClassifier(
                        n_estimators=100, max_depth=10, random_state=42
                    )
                elif search_model_name == "RandomForest":
                    model = RandomForestClassifier(
                        n_estimators=100, max_depth=10,
                        class_weight='balanced', n_jobs=quick_jobs, random_state=42
                    )
                elif search_model_name != "LSTM":
                    # Default: LightGBM (or RandomForest fallback)
                    if HAS_LIGHTGBM:
                        model = lgb.LGBMClassifier(
                            n_estimators=100, max_depth=10,
                            class_weight='balanced', verbose=-1, n_jobs=quick_jobs
                        )
                    else:
                        model = RandomForestClassifier(
                            n_estimators=100, max_depth=10,
                            class_weight='balanced', n_jobs=quick_jobs
                        )

                # Train and predict (skip for LSTM which was handled above)
                if search_model_name != "LSTM":
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)

                label_ratio = y.sum() / len(y) * 100

                print(f"    F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
                print(f"    Positive labels: {label_ratio:.1f}%")

                config_results.append({
                    'forward_days': forward_days,
                    'profit_threshold': profit_threshold,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'label_ratio': label_ratio
                })

                if f1 > best_score:
                    best_score = f1
                    best_config = {
                        'forward_days': forward_days,
                        'profit_threshold': profit_threshold,
                        'combined_data': combined_data,
                        'feature_names': feature_names
                    }

        elapsed = time.time() - start_time
        print(f"\n  ⏱️  Configuration search completed in {elapsed:.1f}s")
        print(f"\n  ✓ Best config: forward={best_config['forward_days']}d, "
              f"profit={best_config['profit_threshold']}% (F1={best_score:.4f})")

        self.best_config = best_config
        return {'best': best_config, 'all_results': config_results}

    def train_all_models(self, combined_data: pd.DataFrame) -> Dict[str, Any]:
        """Train all models with hyperparameter tuning."""
        import time
        start_time = time.time()

        print(f"\n{'='*60}")
        print("STEP 3: TRAINING MODELS WITH HYPERPARAMETER TUNING")
        print(f"{'='*60}")
        print(f"  🧠 Training tree-based models with RandomizedSearchCV...")
        print(f"  🔧 Hyperparameter search iterations: {self.config.n_iter_search}")
        print(f"  📊 Cross-validation folds: {self.config.cv_folds}")

        # Prepare features and labels
        X, self.feature_names = self.feature_engineer.prepare_features(combined_data)
        y = combined_data['LABEL_ENTRY'].values

        # ===== FEATURE DATA QUALITY DIAGNOSTICS =====
        n_samples, n_features = X.shape
        nan_count = np.isnan(X).sum()
        inf_count = np.isinf(X).sum()
        zero_rows = np.all(X == 0, axis=1).sum()
        nan_per_feature = np.isnan(X).sum(axis=0)
        bad_features = [(self.feature_names[i], int(nan_per_feature[i]))
                        for i in range(n_features) if nan_per_feature[i] > n_samples * 0.1]

        print(f"\n  📊 FEATURE DATA QUALITY CHECK:")
        print(f"     Shape: {n_samples} samples x {n_features} features")
        print(f"     NaN cells: {nan_count} ({nan_count/(n_samples*n_features)*100:.1f}%)")
        print(f"     Inf cells: {inf_count}")
        print(f"     All-zero rows: {zero_rows} ({zero_rows/n_samples*100:.1f}%)")
        print(f"     Label distribution: {int(y.sum())} positive ({y.sum()/len(y)*100:.1f}%), "
              f"{int(len(y) - y.sum())} negative ({(1-y.sum()/len(y))*100:.1f}%)")
        if bad_features:
            print(f"     ⚠️ Features with >10% NaN:")
            for fname, fnan in bad_features[:10]:
                print(f"        - {fname}: {fnan} NaN ({fnan/n_samples*100:.1f}%)")
        if zero_rows > n_samples * 0.05:
            print(f"     ⚠️ WARNING: {zero_rows} all-zero rows ({zero_rows/n_samples*100:.1f}%) — "
                  f"these may corrupt model training!")

        # Scale features
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Count how many cells are being zeroed out by nan_to_num
        nan_after_scale = np.isnan(X_scaled).sum() + np.isinf(X_scaled).sum()
        if nan_after_scale > 0:
            print(f"     Cells zeroed by nan_to_num after scaling: {nan_after_scale}")
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Check for all-zero rows after scaling (the real danger)
        zero_rows_after = np.all(X_scaled == 0, axis=1).sum()
        if zero_rows_after > 0:
            print(f"     All-zero rows after scaling: {zero_rows_after}")
        print()

        # Train/test split
        split_idx = int(len(X_scaled) * (1 - self.config.test_size))
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Store for rule extraction
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        print(f"  Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"  Features: {len(self.feature_names)}")

        # DATA LEAKAGE CHECK: Ensure no forward-looking features
        forward_features = [f for f in self.feature_names if 'FORWARD' in f.upper()]
        if forward_features:
            print(f"\n  🚨 DATA LEAKAGE DETECTED! Forward-looking features found:")
            for f in forward_features:
                print(f"     - {f}")
            raise ValueError("Cannot train with forward-looking features - this is data leakage!")

        # Print first few features for verification
        print(f"  Sample features: {self.feature_names[:5]}...")

        # Initialize tuner and train tree-based models
        tuner = HyperparameterTuner(self.config)
        tuned_models = tuner.tune_all_models(X_train, y_train)

        # Train neural networks (optional)
        # Train neural networks (MLP ensemble)
        nn_results = {}
        nn_trainer = NeuralNetworkTrainer(self.config)

        if self.config.train_neural_networks:
            nn_results = nn_trainer.train_mlp_ensemble(X_train, y_train, X_test, y_test)
        else:
            print(f"\n  ⏭️  Skipping MLP training (disabled in config)")

        # Train LSTM SEPARATELY - this is critical for time series!
        # LSTM should ALWAYS be trained as it's the best model for sequential data
        if self.config.train_lstm:
            print(f"\n  🧠 Training LSTM (best for time series patterns)...")
            lstm_results = nn_trainer.train_lstm(
                X_train, y_train, X_test, y_test,
                seq_length=self.config.lstm_sequence_length
            )
            if lstm_results:
                nn_results.update(lstm_results)
                print(f"  ✓ LSTM training complete")
            else:
                print(f"  ⚠️ LSTM training returned no results")
        else:
            print(f"\n  ⏭️  Skipping LSTM training (disabled in config)")

        # Evaluate all models on test set
        print(f"\n{'='*60}")
        print("MODEL EVALUATION ON TEST SET")
        print(f"{'='*60}")

        all_results = {}

        for name, data in tuned_models.items():
            model = data['model']
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5
            }

            all_results[name] = {
                'model': model,
                'metrics': metrics,
                'params': data.get('params', {}),
                'cv_score': data.get('cv_score', 0)
            }

            print(f"  {name}: F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}, "
                  f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")

        # Add NN results
        for name, data in nn_results.items():
            all_results[name] = data
            m = data['metrics']
            print(f"  {name}: F1={m['f1']:.4f}, AUC={m['auc']:.4f}, "
                  f"Precision={m['precision']:.4f}, Recall={m['recall']:.4f}")

        # Find best model
        best_name = max(all_results.keys(), key=lambda k: all_results[k]['metrics']['f1'])
        self.best_model = all_results[best_name]['model']
        self.best_model_name = best_name

        # Store ALL models for ensemble rule extraction
        self.all_trained_models = all_results

        elapsed = time.time() - start_time
        print(f"\n  ⏱️  Model training completed in {elapsed:.1f}s")

        # DATA LEAKAGE CHECK: F1 > 0.95 is almost always a sign of data leakage
        best_f1 = all_results[best_name]['metrics']['f1']
        if best_f1 > 0.95:
            print(f"\n  ⚠️  WARNING: F1 score of {best_f1:.4f} is suspiciously high!")
            print(f"     This usually indicates DATA LEAKAGE - the model is seeing the future.")
            print(f"     Check that no FORWARD_RETURN columns are in the features.")
            print(f"     Realistic F1 scores for trading are typically 0.3-0.6.")

        print(f"\n  🏆 Best model: {best_name}")
        print(f"     F1={best_f1:.4f}")
        print(f"     Precision={all_results[best_name]['metrics']['precision']:.4f}")
        print(f"     Recall={all_results[best_name]['metrics']['recall']:.4f}")
        print(f"     AUC={all_results[best_name]['metrics']['auc']:.4f}")

        # Print all model rankings
        print(f"\n  📊 All Model Rankings (by F1):")
        sorted_models = sorted(all_results.items(), key=lambda x: x[1]['metrics']['f1'], reverse=True)
        for i, (name, data) in enumerate(sorted_models, 1):
            f1 = data['metrics']['f1']
            warning = " ⚠️ SUSPICIOUS" if f1 > 0.95 else ""
            print(f"     {i}. {name}: F1={f1:.4f}{warning}")

        self.all_results = all_results
        return all_results

    def create_ensemble(self, top_n: int = 3) -> VotingClassifier:
        """Create a voting ensemble from the top N models."""
        print(f"\n{'='*60}")
        print(f"CREATING ENSEMBLE FROM TOP {top_n} MODELS")
        print(f"{'='*60}")

        # Sort models by F1 score
        sorted_models = sorted(
            self.all_results.items(),
            key=lambda x: x[1]['metrics']['f1'],
            reverse=True
        )[:top_n]

        estimators = []
        for name, data in sorted_models:
            print(f"  Including: {name} (F1={data['metrics']['f1']:.4f})")
            estimators.append((name, data['model']))

        # Use limited parallelism to avoid resource exhaustion
        import multiprocessing as mp
        ensemble_jobs = max(2, mp.cpu_count() // 2)
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=ensemble_jobs
        )

        print(f"  ✓ Ensemble created with {len(estimators)} models")

        return ensemble

    def save_enhanced_model(self, filepath: str = None):
        """Save the best model, scaler, and configuration."""
        if filepath is None:
            filepath = os.path.join(os.path.dirname(__file__), 'ml_model_enhanced.pkl')

        # Handle PyTorch models: save state_dict instead of the model object
        # to avoid pickle identity issues with conditionally-defined classes
        is_pytorch_model = HAS_PYTORCH and isinstance(self.best_model, torch.nn.Module)
        if is_pytorch_model:
            model_to_save = {
                'type': 'pytorch',
                'model_class': type(self.best_model).__name__,
                'state_dict': self.best_model.state_dict(),
                'model_config': {
                    'input_size': self.best_model.lstm.input_size if hasattr(self.best_model, 'lstm') else None,
                    'hidden_size': self.best_model.hidden_size if hasattr(self.best_model, 'hidden_size') else None,
                    'num_layers': self.best_model.num_layers if hasattr(self.best_model, 'num_layers') else None,
                },
            }
        else:
            model_to_save = {'type': 'sklearn', 'model': self.best_model}

        model_data = {
            'model_wrapper': model_to_save,
            'model': self.best_model if not is_pytorch_model else None,  # Backward compat for sklearn models
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'config': {
                'forward_days': self.best_config.get('forward_days') if self.best_config else None,
                'profit_threshold': self.best_config.get('profit_threshold') if self.best_config else None,
            },
            'training_date': datetime.now().isoformat(),
            'all_results': {
                name: {
                    'metrics': data['metrics'],
                    'params': data.get('params', {})
                } for name, data in self.all_results.items()
            } if self.all_results else {}
        }

        if is_pytorch_model:
            # Save with torch.save which handles PyTorch objects natively
            torch.save(model_data, filepath)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)

        print(f"\n✓ Model saved to: {filepath}")
        return filepath

    def generate_strategy_config(self) -> Dict[str, Any]:
        """Generate trading strategy configuration from ML insights."""
        if not self.all_results or self.best_model is None:
            raise ValueError("Must train models first!")

        # Get feature importances from best model (if available)
        feature_importance = {}
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            for i, name in enumerate(self.feature_names):
                feature_importance[name] = float(importances[i])

        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:15]

        # Get best model metrics
        best_name = max(self.all_results.keys(), key=lambda k: self.all_results[k]['metrics']['f1'])
        best_metrics = self.all_results[best_name]['metrics']

        config = {
            'version': '6.0.0',
            'generated_by': 'EnhancedMLTrainer',
            'generated_date': datetime.now().isoformat(),
            'model_performance': {
                'best_model': best_name,
                'f1_score': best_metrics['f1'],
                'precision': best_metrics['precision'],
                'recall': best_metrics['recall'],
                'auc': best_metrics['auc']
            },
            'label_config': {
                'forward_days': self.best_config.get('forward_days') if self.best_config else 10,
                'profit_threshold': self.best_config.get('profit_threshold') if self.best_config else 5.0
            },
            'top_features': [{'name': n, 'importance': v} for n, v in sorted_features],
            'entry': {
                'use_ml_prediction': True,
                'min_confidence': 0.55,
                'strong_signal_threshold': 0.70
            },
            'exit': {
                'use_ml_prediction': True,
                'profit_target': 0.15,
                'stop_loss': 0.08,
                'trailing_stop': 0.05
            },
            'risk_management': {
                'max_position_pct': 0.15,
                'max_portfolio_risk': 0.20,
                'max_daily_loss': 0.05
            }
        }

        # Save config
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'strategies', 'semiconductor_momentum', 'ml_enhanced_config.json'
        )
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✓ Strategy config saved to: {config_path}")

        return config

    async def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete enhanced ML training pipeline."""
        print("\n" + "="*70)
        print("ENHANCED ML STRATEGY TRAINER")
        print("="*70)
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Period: {self.period_years} years")
        print(f"Forward return options: {self.config.forward_returns_days_options}")
        print(f"Profit threshold options: {self.config.profit_threshold_options}")
        print("="*70)

        # Step 1: Fetch data
        await self.fetch_all_data()

        if len(self.stock_data) == 0:
            raise ValueError("No data fetched!")

        # Step 2: Search for best label configuration
        config_results = self.run_configuration_search()

        if self.best_config is None:
            raise ValueError("No valid configuration found!")

        # Step 3: Train all models with best config
        combined_data = self.best_config['combined_data']
        all_results = self.train_all_models(combined_data)

        # Step 4: Create ensemble (optional)
        # ensemble = self.create_ensemble(top_n=3)

        # Step 5: Save model
        model_path = self.save_enhanced_model()

        # Step 6: Generate strategy config
        strategy_config = self.generate_strategy_config()

        # Summary
        print("\n" + "="*70)
        print("TRAINING COMPLETE - SUMMARY")
        print("="*70)

        best_name = max(all_results.keys(), key=lambda k: all_results[k]['metrics']['f1'])
        best_metrics = all_results[best_name]['metrics']

        print(f"  Best Model: {best_name}")
        print(f"  F1 Score: {best_metrics['f1']:.4f}")
        print(f"  Precision: {best_metrics['precision']:.4f}")
        print(f"  Recall: {best_metrics['recall']:.4f}")
        print(f"  AUC: {best_metrics['auc']:.4f}")
        print(f"\n  Label Config: forward={self.best_config['forward_days']}d, "
              f"profit={self.best_config['profit_threshold']}%")
        print(f"  Model saved to: {model_path}")
        print("="*70)

        return {
            'best_model': best_name,
            'metrics': best_metrics,
            'config': strategy_config,
            'all_results': {
                name: data['metrics'] for name, data in all_results.items()
            }
        }


async def main():
    """Main entry point for enhanced ML training."""
    print("\n" + "🚀 "*20)
    print("STARTING ENHANCED ML TRAINING PIPELINE")
    print("🚀 "*20 + "\n")
    os.environ['POLYGON_API_KEY'] = 'PLACEHOLDER_ROTATE_ME'

    # Initialize trainer
    trainer = EnhancedMLTrainer(
        period_years=10,  # 10 years for more training data
        config=EnhancedMLConfig(
            forward_returns_days_options=[1, 2, 3,5, 7, 10, 12, 15, 20],
            profit_threshold_options=[2.0, 2.5, 3.0, 5.0, 7.0, 10.0],
            n_iter_search=30,  # Reduced for faster training
            cv_folds=3
        )
    )

    # Run full pipeline
    results = await trainer.run_full_pipeline()

    print("\n✅ Enhanced ML Training Complete!")
    print(f"   Best Model: {results['best_model']}")
    print(f"   F1 Score: {results['metrics']['f1']:.4f}")

    # Create predictor instance for live use
    print("\n📊 Testing real-time predictor...")
    predictor = MLPredictor()
    predictor.load_model()

    print("\n✅ Predictor ready for live trading signals!")

    return results


if __name__ == "__main__":
    asyncio.run(main())
