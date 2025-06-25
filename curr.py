# type: ignore
"""
Enhanced Manual Trading Signal System - Production Ready with All Improvements
==============================================================================
INTEGRATED VERSION with:
- Improved ML ensemble models with validation
- Robust error handling system
- Adaptive signal thresholds
- All original features preserved

Version: 2.0 - Production Ready with Enhanced Reliability
"""

import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import akshare as ak
import json
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from colorama import init, Fore, Back, Style
import pandas_ta as ta
from scipy import stats
import warnings
import threading
import time
import os
from dataclasses import dataclass, asdict
from collections import defaultdict
import sqlite3
from contextlib import contextmanager
import hashlib
import traceback
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import requests
import socket
import functools
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import class_weight

warnings.filterwarnings('ignore')

# Initialize colorama for colored output
init(autoreset=True)

# ============================================================================
# ERROR HANDLING SYSTEM
# ============================================================================

class TradingSystemError(Exception):
    """Base exception for trading system"""
    pass

class DataError(TradingSystemError):
    """Data-related errors"""
    pass

class ExecutionError(TradingSystemError):
    """Trade execution errors"""
    pass

class CriticalError(TradingSystemError):
    """Critical errors requiring immediate attention"""
    pass

def robust_retry(error_type: str = 'general', 
                max_retries: int = 3,
                critical: bool = False):
    """Decorator for automatic retry with error handling"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if hasattr(self, 'error_handler'):
                return self.error_handler.safe_execute(
                    func, self, *args,
                    error_type=error_type,
                    max_retries=max_retries,
                    critical=critical,
                    **kwargs
                )
            else:
                # Fallback if no error handler
                return func(self, *args, **kwargs)
        return wrapper
    return decorator

class ErrorHandler:
    """Comprehensive error handling and recovery system"""
    
    def __init__(self, logger: logging.Logger, failsafe_manager=None):
        self.logger = logger
        self.failsafe = failsafe_manager
        self.error_counts = {}
        self.error_history = []
        self.recovery_strategies = {
            'data_fetch': self._recover_data_fetch,
            'calculation': self._recover_calculation,
            'database': self._recover_database,
            'network': self._recover_network,
            'execution': self._recover_execution
        }
    
    def safe_execute(self, func: Callable, *args, 
                    error_type: str = 'general',
                    max_retries: int = 3,
                    retry_delay: float = 1.0,
                    default_return: Any = None,
                    critical: bool = False,
                    **kwargs) -> Any:
        """
        Safely execute a function with retry logic and error handling
        """
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                last_error = e
                error_key = f"{func.__name__}_{error_type}"
                
                # Log error with context
                self.logger.error(
                    f"Error in {func.__name__} (attempt {attempt + 1}/{max_retries}): "
                    f"{type(e).__name__}: {str(e)}"
                )
                
                # Record error
                self._record_error(error_key, e)
                
                # Check if we should stop retrying
                if self._should_stop_retrying(error_key, e):
                    break
                
                # Apply recovery strategy if available
                if error_type in self.recovery_strategies:
                    recovered = self.recovery_strategies[error_type](e, func, args, kwargs)
                    if recovered is not None:
                        return recovered
                
                # Wait before retry (exponential backoff)
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    self.logger.info(f"Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
        
        # All retries failed
        self._handle_failure(func.__name__, last_error, critical)
        
        return default_return
    
    def _record_error(self, error_key: str, error: Exception):
        """Record error for pattern analysis"""
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        self.error_history.append({
            'timestamp': datetime.datetime.now(),
            'key': error_key,
            'type': type(error).__name__,
            'message': str(error),
            'traceback': traceback.format_exc()
        })
        
        # Keep only recent history
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-500:]
    
    def _should_stop_retrying(self, error_key: str, error: Exception) -> bool:
        """Determine if we should stop retrying based on error pattern"""
        
        # Don't retry certain errors
        non_retryable = [
            KeyboardInterrupt,
            SystemExit,
            MemoryError,
            KeyError,  # Usually indicates logic error
            ValueError,  # Usually indicates bad input
            AttributeError  # Usually indicates code error
        ]
        
        if type(error) in non_retryable:
            return True
        
        # Stop if this error is happening too frequently
        if self.error_counts.get(error_key, 0) > 10:
            self.logger.warning(f"Error {error_key} occurring too frequently, stopping retries")
            return True
        
        return False
    
    def _handle_failure(self, func_name: str, error: Exception, critical: bool):
        """Handle complete failure after all retries"""
        
        self.logger.error(f"All retries failed for {func_name}: {error}")
        
        if critical and self.failsafe:
            self.failsafe.activate_emergency_stop(
                f"Critical error in {func_name}: {str(error)}"
            )
    
    # Recovery strategies
    def _recover_data_fetch(self, error: Exception, func, args, kwargs) -> Optional[Any]:
        """Try alternative data sources or use cached data"""
        if "get_market_data" in func.__name__:
            self.logger.info("Attempting to use cached data...")
            # Return None to trigger cache usage in main code
            return None
        return None
    
    def _recover_calculation(self, error: Exception, func, args, kwargs) -> Optional[Any]:
        """Use simplified calculations or default values"""
        if isinstance(error, (ValueError, ZeroDivisionError)):
            self.logger.info("Using default calculation values...")
            # Return safe default based on function
            if "calculate_signals" in func.__name__ or "generate_enhanced_signals" in func.__name__:
                return {'signal': 'ERROR', 'strength': 0, 'reasons': ['Calculation error'], 'data_quality': 0}
        return None
    
    def _recover_database(self, error: Exception, func, args, kwargs) -> Optional[Any]:
        """Try database reconnection or use backup"""
        self.logger.info("Attempting database recovery...")
        return None
    
    def _recover_network(self, error: Exception, func, args, kwargs) -> Optional[Any]:
        """Handle network errors with fallback"""
        self.logger.info("Network error detected, using offline mode...")
        return None
    
    def _recover_execution(self, error: Exception, func, args, kwargs) -> Optional[Any]:
        """Handle trade execution errors"""
        self.logger.error("Trade execution failed - manual intervention may be required")
        return None
    
    def get_error_summary(self) -> Dict:
        """Get summary of recent errors"""
        summary = {
            'total_errors': sum(self.error_counts.values()),
            'error_types': {},
            'most_common': None,
            'recent_errors': []
        }
        
        # Group by error type
        for error in self.error_history[-100:]:  # Last 100 errors
            error_type = error['type']
            summary['error_types'][error_type] = summary['error_types'].get(error_type, 0) + 1
        
        # Find most common
        if self.error_counts:
            summary['most_common'] = max(self.error_counts.items(), key=lambda x: x[1])
        
        # Recent errors
        summary['recent_errors'] = [
            {
                'time': e['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'key': e['key'],
                'message': e['message']
            }
            for e in self.error_history[-5:]
        ]
        
        return summary

# ============================================================================
# DATA STRUCTURES (Original)
# ============================================================================

@dataclass
class TradingAlert:
    """Trading alert data structure"""
    timestamp: datetime.datetime
    symbol: str
    alert_type: str
    message: str
    urgency: str
    action_required: bool

@dataclass
class DataQualityCheck:
    """Data quality check results"""
    symbol: str
    timestamp: datetime.datetime
    is_valid: bool
    issues: List[str]
    quality_score: float
    data_points: int

# ============================================================================
# DATABASE MANAGER (Original)
# ============================================================================

class DatabaseManager:
    """Robust database management for trading data"""
    
    def __init__(self, db_path: str = 'trading_system.db'):
        self.db_path = db_path
        self.backup_path = f"{db_path}.backup"
        self._init_database()
        
    def _init_database(self):
        """Initialize database with proper schema"""
        with self.get_connection() as conn:
            # Positions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    quantity INTEGER NOT NULL,
                    buy_price REAL NOT NULL,
                    buy_date DATE NOT NULL,
                    cost_basis REAL NOT NULL,
                    commission REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Trade history table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    commission REAL NOT NULL,
                    stamp_duty REAL DEFAULT 0,
                    pnl REAL DEFAULT 0,
                    return_pct REAL DEFAULT 0,
                    trade_date DATE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Watchlist table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS watchlist (
                    symbol TEXT PRIMARY KEY,
                    added_date DATE NOT NULL,
                    priority INTEGER DEFAULT 0,
                    notes TEXT
                )
            """)
            
            # Alerts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    symbol TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    urgency TEXT NOT NULL,
                    action_required BOOLEAN NOT NULL,
                    acknowledged BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Performance snapshots table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_date DATE UNIQUE NOT NULL,
                    portfolio_value REAL NOT NULL,
                    n_positions INTEGER NOT NULL,
                    total_pnl REAL NOT NULL,
                    daily_return REAL,
                    volatility REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    var_95 REAL,
                    market_regime TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Data quality log table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_quality_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    check_timestamp TIMESTAMP NOT NULL,
                    is_valid BOOLEAN NOT NULL,
                    quality_score REAL NOT NULL,
                    issues TEXT,
                    data_points INTEGER
                )
            """)
            
            # System state table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Paper trading tables
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_positions (
                    symbol TEXT PRIMARY KEY,
                    quantity INTEGER NOT NULL,
                    avg_price REAL NOT NULL,
                    cost_basis REAL NOT NULL,
                    unrealized_pnl REAL DEFAULT 0,
                    last_price REAL,
                    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    order_price REAL NOT NULL,
                    fill_price REAL NOT NULL,
                    commission REAL NOT NULL,
                    stamp_duty REAL DEFAULT 0,
                    slippage REAL NOT NULL,
                    pnl REAL DEFAULT 0,
                    return_pct REAL DEFAULT 0,
                    cash_after REAL NOT NULL,
                    portfolio_value REAL NOT NULL,
                    signal_strength REAL,
                    signal_reason TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_performance (
                    date DATE PRIMARY KEY,
                    portfolio_value REAL NOT NULL,
                    cash REAL NOT NULL,
                    positions_value REAL NOT NULL,
                    daily_return REAL,
                    total_return REAL,
                    max_drawdown REAL,
                    n_trades INTEGER DEFAULT 0,
                    n_positions INTEGER DEFAULT 0
                )
            """)
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Get database connection with proper error handling"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")  # Write-ahead logging
            conn.execute("PRAGMA synchronous=NORMAL")
            yield conn
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            raise Exception(f"Database error: {e}")
        finally:
            if conn:
                conn.close()
    
    def backup_database(self):
        """Create database backup"""
        try:
            import shutil
            shutil.copy2(self.db_path, self.backup_path)
            return True
        except Exception as e:
            logging.error(f"Backup failed: {e}")
            return False
    
    def verify_integrity(self) -> bool:
        """Verify database integrity"""
        with self.get_connection() as conn:
            result = conn.execute("PRAGMA integrity_check").fetchone()
            return result[0] == "ok"

# ============================================================================
# IMPROVED ML TRADING ENGINE
# ============================================================================

class ImprovedMLTradingEngine:
    """Enhanced ML engine with validation and ensemble methods"""
    
    def __init__(self, model_dir: str = 'ml_models'):
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.is_trained = False
        self.training_features = []
        self.ensemble_weights = {}
        self.validation_metrics = {}
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        self._load_models()
    
    def _load_models(self):
        """Load saved models or initialize new ones"""
        try:
            # Try to load existing models
            model_files = {
                'rf': 'rf_model.pkl',
                'gbm': 'gbm_model.pkl',
                'ridge': 'ridge_model.pkl'
            }
            
            all_loaded = True
            for model_name, filename in model_files.items():
                model_path = os.path.join(self.model_dir, filename)
                if os.path.exists(model_path):
                    self.models[model_name] = joblib.load(model_path)
                else:
                    all_loaded = False
                    break
            
            if all_loaded:
                # Load scaler
                scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
                if os.path.exists(scaler_path):
                    self.scalers['standard'] = joblib.load(scaler_path)
                    
                    # Load other metadata
                    metadata_path = os.path.join(self.model_dir, 'model_metadata.json')
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            self.feature_importance = metadata.get('feature_importance', {})
                            self.training_features = metadata.get('training_features', [])
                            self.ensemble_weights = metadata.get('ensemble_weights', {})
                            self.validation_metrics = metadata.get('validation_metrics', {})
                    
                    self.is_trained = True
                    logging.info("ML models loaded successfully")
                else:
                    all_loaded = False
            
            if not all_loaded:
                self._initialize_models()
                
        except Exception as e:
            logging.warning(f"Could not load ML models: {e}")
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize new ML models"""
        self.models = {
            'rf': RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                min_samples_split=100,
                min_samples_leaf=50,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'gbm': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                min_samples_split=100,
                min_samples_leaf=50,
                subsample=0.8,
                random_state=42
            ),
            'ridge': Ridge(alpha=1.0, random_state=42)
        }
        self.scalers['standard'] = StandardScaler()
        self.is_trained = False
        self.training_features = []
        self.ensemble_weights = {'rf': 0.4, 'gbm': 0.4, 'ridge': 0.2}
        logging.info("Initialized new ML models")
    
    def prepare_features(self, technical_indicators: Dict) -> pd.DataFrame:
        """Prepare features for ML prediction - must match training features"""
        if self.training_features:
            # Use saved training features
            features = {}
            for feature in self.training_features:
                if feature in technical_indicators:
                    features[feature] = technical_indicators[feature]
                else:
                    # Map technical indicators to training features
                    features[feature] = self._map_feature(feature, technical_indicators)
            
            return pd.DataFrame([features])
        else:
            # Fallback to default features if no training features saved
            return self._prepare_default_features(technical_indicators)
    
    def _map_feature(self, feature: str, technical_indicators: Dict) -> float:
        """Map technical indicators to expected features"""
        if feature == 'returns_1d' and 'momentum_5d' in technical_indicators:
            return technical_indicators.get('momentum_5d', 0) / 500
        elif feature == 'returns_5d' and 'momentum_5d' in technical_indicators:
            return technical_indicators.get('momentum_5d', 0) / 100
        elif feature == 'returns_20d' and 'momentum_20d' in technical_indicators:
            return technical_indicators.get('momentum_20d', 0) / 100
        elif feature == 'volume_ratio' and 'obv_momentum' in technical_indicators:
            return 1 + technical_indicators.get('obv_momentum', 0)
        elif 'rsi' in feature:
            return technical_indicators.get(feature, 50.0)
        elif 'volatility' in feature:
            return technical_indicators.get('volatility_20d', 0.2)
        elif feature in ['high_low_ratio', 'bb_width']:
            return technical_indicators.get('bb_width', 0.02)
        elif feature in ['close_to_high', 'bb_position']:
            return technical_indicators.get('bb_position', 0.5)
        elif feature == 'price_efficiency':
            return 0.5
        elif feature == 'trend_strength':
            return technical_indicators.get('momentum_20d', 0) / 100
        elif feature == 'log_price':
            return 5.0
        elif feature == 'symbol_hash':
            return 0.5
        elif feature == 'volume_trend':
            return 0.0
        else:
            return 0.0
    
    def _prepare_default_features(self, technical_indicators: Dict) -> pd.DataFrame:
        """Prepare default features when no training features are available"""
        # Define a minimal feature set that should work
        features = {
            'rsi_14': technical_indicators.get('rsi_14', 50.0),
            'rsi_7': technical_indicators.get('rsi_7', 50.0),
            'momentum_5d': technical_indicators.get('momentum_5d', 0.0),
            'momentum_20d': technical_indicators.get('momentum_20d', 0.0),
            'volatility_20d': technical_indicators.get('volatility_20d', 0.2),
            'bb_position': technical_indicators.get('bb_position', 0.5),
            'macd': technical_indicators.get('macd', 0.0),
            'macd_signal': technical_indicators.get('macd_signal', 0.0),
            'distance_from_high_20d': technical_indicators.get('distance_from_high_20d', 0.0),
            'distance_from_low_20d': technical_indicators.get('distance_from_low_20d', 0.0)
        }
        
        return pd.DataFrame([features])
    
    def predict(self, technical_indicators: Dict) -> Dict:
        """Make ML prediction with classification models"""
        try:
            # Prepare features
            features_df = self.prepare_features(technical_indicators)
            
            if not self.is_trained:
                return {
                    'ml_signal': 0.0,
                    'ml_confidence': 0.0,
                    'prediction_quality': 'untrained',
                    'feature_coverage': 0.0
                }
            
            # Scale features
            features_scaled = self.scalers['standard'].transform(features_df)
            
            # Get predictions from all models
            predictions = {}
            probabilities = {}
            
            for name, model in self.models.items():
                try:
                    # Get prediction
                    pred = model.predict(features_scaled)[0]
                    predictions[name] = pred
                    
                    # Get probabilities if available
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(features_scaled)[0]
                        # Find the index for class 1 (buy signal)
                        if 1 in model.classes_:
                            buy_idx = list(model.classes_).index(1)
                            probabilities[name] = prob[buy_idx]
                        else:
                            probabilities[name] = 0.5
                    else:
                        # For models without probability (shouldn't happen with our choices)
                        probabilities[name] = 0.5 if pred == 0 else (0.7 if pred == 1 else 0.3)
                        
                except Exception as e:
                    self.logger.error(f"Prediction error for {name}: {e}")
                    predictions[name] = 0
                    probabilities[name] = 0.5
            
            # Calculate ensemble prediction based on probabilities
            avg_buy_prob = sum(prob * self.ensemble_weights.get(name, 1/len(self.models))
                            for name, prob in probabilities.items())
            
            # Convert probability to signal
            if avg_buy_prob > 0.80:  # 65% confidence for buy
                ml_signal = 0.04  # Positive signal
            elif avg_buy_prob > 0.65:
                ml_signal = 0.02
            elif avg_buy_prob < 0.35:  # 35% = 65% confidence for sell
                ml_signal = -0.02  # Negative signal
            elif avg_buy_prob < 0.2:
                ml_signal = -0.04
            else:
                ml_signal = 0  # No clear signal
            
            # Confidence is how far from 50% we are
            ml_confidence = abs(avg_buy_prob - 0.5) * 2
            
            # Feature coverage
            feature_coverage = len([f for f in features_df.columns if features_df[f].iloc[0] != 0]) / len(features_df.columns)
            
            return {
                'ml_signal': float(ml_signal),
                'ml_confidence': float(ml_confidence),
                'prediction_quality': 'trained',
                'feature_coverage': feature_coverage,
                'buy_probability': avg_buy_prob,
                'model_agreement': np.std(list(probabilities.values())),
                'individual_predictions': predictions
            }
            
        except Exception as e:
            logging.error(f"ML prediction error: {e}")
            return {
                'ml_signal': 0.0,
                'ml_confidence': 0.0,
                'prediction_quality': 'error',
                'feature_coverage': 0.0
            }
    
    def train_ensemble_models(self, X_train, y_train, X_val, y_val):
        """Train multiple models and create ensemble"""
        
        ensemble_predictions = {}
        model_scores = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Validate
            val_pred = model.predict(X_val)
            val_score = r2_score(y_val, val_pred)
            val_mse = mean_squared_error(y_val, val_pred)
            
            # Store predictions for ensemble
            ensemble_predictions[name] = val_pred
            model_scores[name] = val_score
            
            print(f"  {name} - R²: {val_score:.3f}, MSE: {val_mse:.6f}")
            
            # Cross-validation for robustness check
            if name != 'ridge':  # Ridge is fast enough for CV
                cv_scores = self._time_series_cv(model, X_train, y_train)
                print(f"  {name} - CV Mean R²: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        # Calculate ensemble weights based on validation performance
        scores_array = np.array(list(model_scores.values()))
        scores_array = np.maximum(scores_array, 0)  # Ensure non-negative
        
        if scores_array.sum() > 0:
            weights = np.exp(scores_array * 2) / np.exp(scores_array * 2).sum()
        else:
            weights = np.ones(len(self.models)) / len(self.models)
        
        self.ensemble_weights = dict(zip(model_scores.keys(), weights))
        print(f"\nEnsemble weights: {self.ensemble_weights}")
        
        # Test ensemble performance
        ensemble_pred = sum(
            ensemble_predictions[name] * weight
            for name, weight in self.ensemble_weights.items()
        )
        ensemble_score = r2_score(y_val, ensemble_pred)
        print(f"Ensemble R²: {ensemble_score:.3f}")
        
        # Calculate feature importance (from RF model)
        if 'rf' in self.models:
            importance = self.models['rf'].feature_importances_
            self.feature_importance = dict(zip(self.training_features, importance))
        
        return ensemble_score > 0.01  # Minimum acceptable performance
    
    def _time_series_cv(self, model, X, y, n_splits=5):
        """Time series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
            
            # Clone model to avoid refitting
            model_clone = model.__class__(**model.get_params())
            model_clone.fit(X_train_cv, y_train_cv)
            
            pred = model_clone.predict(X_val_cv)
            score = r2_score(y_val_cv, pred)
            scores.append(score)
        
        return np.array(scores)
    
    def validate_on_recent_data(self, symbol_data_dict: Dict) -> bool:
        """Validate model on recent out-of-sample data"""
        print("\nValidating on recent data...")
        
        all_predictions = []
        all_actuals = []
        
        for symbol, df in symbol_data_dict.items():
            if len(df) < 100:
                continue
            
            # Use last 20% for validation
            split_idx = int(len(df) * 0.8)
            val_df = df.iloc[split_idx:]
            
            for i in range(len(val_df) - 5):
                try:
                    # Extract features for validation
                    window_data = val_df.iloc[:i+1]
                    if len(window_data) < 20:
                        continue
                    
                    # Calculate technical indicators (simplified)
                    close = window_data['close']
                    features = {
                        'rsi_14': 50.0,  # Placeholder
                        'momentum_5d': (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0,
                        'momentum_20d': (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) >= 21 else 0,
                        'volatility_20d': close.pct_change().tail(20).std() * np.sqrt(252) if len(close) >= 20 else 0.2,
                        'bb_position': 0.5,  # Placeholder
                        'macd': 0.0,  # Placeholder
                        'macd_signal': 0.0,  # Placeholder
                        'distance_from_high_20d': 0.0,  # Placeholder
                        'distance_from_low_20d': 0.0  # Placeholder
                    }
                    
                    # Get prediction
                    pred_result = self.predict(features)
                    
                    # Get actual return
                    actual = (val_df.iloc[i+5]['close'] / val_df.iloc[i]['close'] - 1) if i+5 < len(val_df) else 0
                    
                    all_predictions.append(pred_result['ml_signal'])
                    all_actuals.append(actual)
                except:
                    continue
        
        if len(all_predictions) > 0:
            # Calculate validation metrics
            predictions_array = np.array(all_predictions)
            actuals_array = np.array(all_actuals)
            
            # Directional accuracy
            direction_correct = ((predictions_array > 0) == (actuals_array > 0)).mean()
            
            # Correlation
            if len(predictions_array) > 1:
                correlation = np.corrcoef(predictions_array, actuals_array)[0, 1]
            else:
                correlation = 0
            
            # Profitable predictions accuracy
            profitable_predictions = predictions_array[predictions_array > 0.005]  # 0.5% threshold
            if len(profitable_predictions) > 0:
                profitable_accuracy = (actuals_array[predictions_array > 0.005] > 0).mean()
            else:
                profitable_accuracy = 0
            
            self.validation_metrics = {
                'direction_accuracy': direction_correct,
                'correlation': correlation,
                'profitable_accuracy': profitable_accuracy,
                'n_samples': len(all_predictions)
            }
            
            print(f"\nValidation Results:")
            print(f"  Direction accuracy: {direction_correct:.1%}")
            print(f"  Correlation: {correlation:.3f}")
            print(f"  Profitable signals accuracy: {profitable_accuracy:.1%}")
            print(f"  Samples validated: {len(all_predictions)}")
            
            return direction_correct > 0.52  # Better than random
        
        return False
    def _prepare_ml_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """ROBUST: Prepare ML features with comprehensive error handling"""
        try:
            # Validate inputs first
            if df is None or df.empty:
                self.logger.error(f"Invalid dataframe for {symbol}")
                return None
                
            if len(df) < 70:  # Need at least 70 rows for features
                self.logger.error(f"Insufficient data for {symbol}: {len(df)} rows")
                return None
            
            # Ensure required columns exist
            required_columns = ['close', 'high', 'low', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.error(f"Missing columns for {symbol}: {missing_columns}")
                return None
            
            # Get clean data series with error handling
            try:
                close = df['close'].fillna(method='ffill').fillna(method='bfill')
                high = df['high'].fillna(method='ffill').fillna(method='bfill')
                low = df['low'].fillna(method='ffill').fillna(method='bfill')
                volume = df['volume'].fillna(0)
                
                # Validate data ranges
                if close.isna().all() or (close <= 0).all():
                    self.logger.error(f"Invalid close prices for {symbol}")
                    return None
                    
            except Exception as e:
                self.logger.error(f"Error extracting price data for {symbol}: {e}")
                return None
            
            features_list = []
            debug_counters = {'buy': 0, 'sell': 0, 'neutral': 0, 'total': 0}
            
            # Process each day with robust error handling
            for i in range(60, len(df) - 5):  # Need 5 days future for target
                try:
                    feature_dict = {}
                    
                    # Basic price features with safety checks
                    current_close = close.iloc[i] if i < len(close) else close.iloc[-1]
                    if current_close <= 0:
                        continue
                    
                    # Returns with safety checks
                    try:
                        feature_dict['returns_1d'] = (close.iloc[i] / close.iloc[i-1] - 1) if i > 0 and close.iloc[i-1] > 0 else 0
                        feature_dict['returns_5d'] = (close.iloc[i] / close.iloc[i-5] - 1) if i >= 5 and close.iloc[i-5] > 0 else 0
                        feature_dict['returns_20d'] = (close.iloc[i] / close.iloc[i-20] - 1) if i >= 20 and close.iloc[i-20] > 0 else 0
                    except (IndexError, ZeroDivisionError):
                        feature_dict['returns_1d'] = 0
                        feature_dict['returns_5d'] = 0
                        feature_dict['returns_20d'] = 0
                    
                    # Volatility features with safety checks
                    try:
                        returns_20 = close.iloc[max(0, i-20):i].pct_change().dropna()
                        feature_dict['volatility_20d'] = returns_20.std() * np.sqrt(252) if len(returns_20) > 1 else 0.2
                        
                        returns_5 = close.iloc[max(0, i-5):i].pct_change().dropna()
                        feature_dict['volatility_5d'] = returns_5.std() * np.sqrt(252) if len(returns_5) > 1 else 0.2
                    except:
                        feature_dict['volatility_20d'] = 0.2
                        feature_dict['volatility_5d'] = 0.2
                    
                    # Technical indicators with comprehensive error handling
                    window_df = df.iloc[max(0, i-60):i+1]
                    window_close = window_df['close'].fillna(method='ffill').fillna(method='bfill')
                    
                    if len(window_close) < 10:
                        continue
                    
                    # RSI with multiple fallbacks
                    try:
                        rsi_14 = ta.rsi(window_close, length=14)
                        feature_dict['rsi_14'] = rsi_14.iloc[-1] if rsi_14 is not None and len(rsi_14) > 0 and not pd.isna(rsi_14.iloc[-1]) else 50
                    except:
                        feature_dict['rsi_14'] = 50
                    
                    try:
                        rsi_7 = ta.rsi(window_close, length=7)
                        feature_dict['rsi_7'] = rsi_7.iloc[-1] if rsi_7 is not None and len(rsi_7) > 0 and not pd.isna(rsi_7.iloc[-1]) else 50
                    except:
                        feature_dict['rsi_7'] = 50
                    
                    try:
                        rsi_28 = ta.rsi(window_close, length=28)
                        feature_dict['rsi_28'] = rsi_28.iloc[-1] if rsi_28 is not None and len(rsi_28) > 0 and not pd.isna(rsi_28.iloc[-1]) else 50
                    except:
                        feature_dict['rsi_28'] = 50
                    
                    # MACD with comprehensive error handling
                    feature_dict['macd'] = 0
                    feature_dict['macd_signal'] = 0
                    feature_dict['macd_histogram'] = 0
                    
                    if len(window_close) >= 26:
                        try:
                            macd_result = ta.macd(window_close, fast=12, slow=26, signal=9)
                            if macd_result is not None and not macd_result.empty and len(macd_result.columns) >= 3:
                                if not pd.isna(macd_result.iloc[-1, 0]):
                                    feature_dict['macd'] = float(macd_result.iloc[-1, 0])
                                if not pd.isna(macd_result.iloc[-1, 1]):
                                    feature_dict['macd_signal'] = float(macd_result.iloc[-1, 1])
                                if not pd.isna(macd_result.iloc[-1, 2]):
                                    feature_dict['macd_histogram'] = float(macd_result.iloc[-1, 2])
                        except Exception as e:
                            pass  # Keep defaults
                    
                    # Bollinger Bands with error handling
                    feature_dict['bb_position'] = 0.5
                    feature_dict['bb_width'] = 0.02
                    
                    if len(window_close) >= 20:
                        try:
                            bbands = ta.bbands(window_close, length=20, std=2)
                            if bbands is not None and not bbands.empty and len(bbands.columns) >= 3:
                                lower = bbands.iloc[-1, 0]
                                middle = bbands.iloc[-1, 1]
                                upper = bbands.iloc[-1, 2]
                                current = window_close.iloc[-1]
                                
                                if not any(pd.isna([lower, middle, upper, current])) and upper > lower and middle > 0:
                                    feature_dict['bb_position'] = float((current - lower) / (upper - lower))
                                    feature_dict['bb_width'] = float((upper - lower) / middle)
                        except:
                            pass  # Keep defaults
                    
                    # Volume features with safety checks
                    try:
                        volume_window = volume.iloc[max(0, i-20):i]
                        avg_volume = volume_window.mean()
                        current_volume = volume.iloc[i] if i < len(volume) else avg_volume
                        
                        feature_dict['volume_ratio'] = float(current_volume / avg_volume) if avg_volume > 0 else 1.0
                        
                        # Volume trend
                        if i >= 20:
                            recent_vol = volume.iloc[max(0, i-5):i].mean()
                            older_vol = volume.iloc[max(0, i-20):i-5].mean()
                            feature_dict['volume_trend'] = float((recent_vol / older_vol - 1)) if older_vol > 0 else 0
                        else:
                            feature_dict['volume_trend'] = 0
                    except:
                        feature_dict['volume_ratio'] = 1.0
                        feature_dict['volume_trend'] = 0
                    
                    # Price patterns with safety checks
                    try:
                        current_high = high.iloc[i] if i < len(high) else current_close
                        current_low = low.iloc[i] if i < len(low) else current_close
                        
                        if current_high > current_low and current_close > 0:
                            feature_dict['high_low_ratio'] = float((current_high - current_low) / current_close)
                            feature_dict['close_to_high'] = float((current_close - current_low) / (current_high - current_low))
                        else:
                            feature_dict['high_low_ratio'] = 0
                            feature_dict['close_to_high'] = 0.5
                    except:
                        feature_dict['high_low_ratio'] = 0
                        feature_dict['close_to_high'] = 0.5
                    
                    # Support/Resistance with safety checks
                    try:
                        if i >= 20:
                            high_20 = high.iloc[max(0, i-20):i].max()
                            low_20 = low.iloc[max(0, i-20):i].min()
                            
                            if high_20 > 0 and low_20 > 0:
                                feature_dict['distance_from_high_20d'] = float((current_close / high_20 - 1))
                                feature_dict['distance_from_low_20d'] = float((current_close / low_20 - 1))
                            else:
                                feature_dict['distance_from_high_20d'] = 0
                                feature_dict['distance_from_low_20d'] = 0
                        else:
                            feature_dict['distance_from_high_20d'] = 0
                            feature_dict['distance_from_low_20d'] = 0
                    except:
                        feature_dict['distance_from_high_20d'] = 0
                        feature_dict['distance_from_low_20d'] = 0
                    
                    # Market microstructure with safety checks
                    try:
                        prev_close = close.iloc[i-1] if i > 0 else current_close
                        current_high = high.iloc[i] if i < len(high) else current_close
                        current_low = low.iloc[i] if i < len(low) else current_close
                        
                        if current_high > current_low:
                            feature_dict['price_efficiency'] = float(1 - abs(current_close - prev_close) / (current_high - current_low))
                        else:
                            feature_dict['price_efficiency'] = 0.5
                    except:
                        feature_dict['price_efficiency'] = 0.5
                    
                    # Trend strength with safety checks
                    try:
                        sma_20 = close.iloc[max(0, i-20):i].mean() if i >= 20 else current_close
                        sma_50 = close.iloc[max(0, i-50):i].mean() if i >= 50 else sma_20
                        
                        if sma_50 > 0:
                            feature_dict['trend_strength'] = float((sma_20 - sma_50) / sma_50)
                        else:
                            feature_dict['trend_strength'] = 0
                    except:
                        feature_dict['trend_strength'] = 0
                    
                    # Symbol-specific features
                    try:
                        feature_dict['log_price'] = float(np.log(current_close)) if current_close > 0 else 0
                        feature_dict['symbol_hash'] = float(hash(symbol) % 100 / 100)
                    except:
                        feature_dict['log_price'] = 0
                        feature_dict['symbol_hash'] = 0.5
                    
                    # FIXED: Calculate target with robust error handling
                    try:
                        if i + 5 < len(close):
                            future_prices = close.iloc[i+1:i+6]  # Next 5 days
                            
                            if len(future_prices) >= 5 and current_close > 0:
                                max_return = float(future_prices.max() / current_close - 1)
                                min_return = float(future_prices.min() / current_close - 1)
                                
                                debug_counters['total'] += 1
                                
                                # 3% threshold for A-shares
                                if max_return > 0.03:
                                    feature_dict['target'] = 1  # BUY
                                    debug_counters['buy'] += 1
                                elif min_return < -0.03:
                                    feature_dict['target'] = -1  # SELL
                                    debug_counters['sell'] += 1
                                else:
                                    feature_dict['target'] = 0  # HOLD
                                    debug_counters['neutral'] += 1
                                
                                # Validate all features are numeric
                                for key, value in feature_dict.items():
                                    if not isinstance(value, (int, float)) or pd.isna(value):
                                        if key == 'target':
                                            feature_dict[key] = 0
                                        else:
                                            feature_dict[key] = 0.0
                                
                                features_list.append(feature_dict)
                            
                    except Exception as e:
                        self.logger.warning(f"Error calculating target for {symbol} at index {i}: {e}")
                        continue
                        
                except Exception as e:
                    self.logger.warning(f"Error processing features for {symbol} at index {i}: {e}")
                    continue
            
            # Debug output
            if debug_counters['total'] > 0:
                print(f"  {symbol} - Buy: {debug_counters['buy']}, Sell: {debug_counters['sell']}, Neutral: {debug_counters['neutral']}")
            
            if len(features_list) < 10:
                self.logger.error(f"Insufficient valid features for {symbol}: {len(features_list)}")
                return None
            
            # Create DataFrame with additional validation
            try:
                features_df = pd.DataFrame(features_list)
                
                # Ensure no infinite or NaN values
                features_df = features_df.replace([np.inf, -np.inf], 0)
                features_df = features_df.fillna(0)
                
                # Validate target column
                if 'target' not in features_df.columns:
                    self.logger.error(f"Target column missing for {symbol}")
                    return None
                
                print(f"  {symbol} - Generated {len(features_df)} valid feature rows")
                return features_df
                
            except Exception as e:
                self.logger.error(f"Error creating DataFrame for {symbol}: {e}")
                return None
            
        except Exception as e:
            self.logger.error(f"Critical error in feature preparation for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _remove_outliers(self, df: pd.DataFrame, n_std: float = 3) -> pd.DataFrame:
        """Remove outliers from training data"""
        print(f"Before outlier removal: {len(df)} samples")
        
        # For classification, don't filter the target column
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != 'target']
        
        for col in numeric_columns:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    mask = abs(df[col] - mean) <= n_std * std
                    df = df[mask]
        
        print(f"After outlier removal: {len(df)} samples remain")
        return df
    def _train_with_validation(self, data: pd.DataFrame) -> bool:
        """Train classification models with proper handling"""
        try:
            print(f"\n_train_with_validation received {len(data)} samples")
            
            # Check if 'target' column exists
            if 'target' not in data.columns:
                self.logger.error("No 'target' column in training data!")
                return False
            
            # Print initial distribution
            print(f"Initial target distribution:")
            print(data['target'].value_counts().sort_index())
            
            # CRITICAL: Remove neutral samples for binary classification
            print("\nRemoving neutral samples...")
            training_data = data[data['target'] != 0].copy()
            
            # Verify the filtering worked
            print(f"After removing neutral: {len(training_data)} samples (was {len(data)})")
            print(f"Binary class distribution: {training_data['target'].value_counts().sort_index().to_dict()}")
            
            # Double-check no neutrals remain
            if 0 in training_data['target'].unique():
                self.logger.error("Neutral samples still present after filtering!")
                return False
            
            if len(training_data) < 100:
                self.logger.error(f"Insufficient training samples: {len(training_data)}")
                return False
            
            # Separate features and targets
            feature_cols = [col for col in training_data.columns if col not in ['target', 'symbol_hash']]
            
            print(f"\nNumber of features: {len(feature_cols)}")
            if len(feature_cols) == 0:
                self.logger.error("No features found!")
                return False
            
            X = training_data[feature_cols]
            y = training_data['target']  # Now only -1 or 1
            
            # Fill any NaN values
            X = X.fillna(0)
            
            # Calculate class weights for imbalanced data
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y)
            class_weights = compute_class_weight('balanced', classes=classes, y=y)
            class_weight_dict = dict(zip(classes, class_weights))
            
            print(f"\nClass weights: {class_weight_dict}")
            
            # Import classifiers (make sure these imports are at the top of your file)
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
            
            # Initialize CLASSIFICATION models (not regression!)
            self.models = {
                'rf': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=6,
                    min_samples_split=50,  # Reduced from 100
                    min_samples_leaf=20,   # Reduced from 50
                    max_features='sqrt',
                    class_weight=class_weight_dict,
                    random_state=42,
                    n_jobs=-1
                ),
                'gbm': GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=4,
                    min_samples_split=50,  # Reduced from 100
                    min_samples_leaf=20,   # Reduced from 50
                    subsample=0.8,
                    random_state=42
                ),
                'logistic': LogisticRegression(
                    class_weight=class_weight_dict,
                    random_state=42,
                    max_iter=1000,
                    C=1.0  # Regularization parameter
                )
            }
            
            # Split data chronologically (80/20)
            split_point = int(len(X) * 0.8)
            X_train, X_val = X.iloc[:split_point], X.iloc[split_point:]
            y_train, y_val = y.iloc[:split_point], y.iloc[split_point:]
            
            print(f"\nTraining set: {len(X_train)} samples")
            print(f"Validation set: {len(X_val)} samples")
            print(f"Training distribution: {y_train.value_counts().to_dict()}")
            print(f"Validation distribution: {y_val.value_counts().to_dict()}")
            
            # Save training features
            self.training_features = list(feature_cols)
            
            # Initialize and fit scaler
            self.scalers['standard'] = StandardScaler()
            X_train_scaled = self.scalers['standard'].fit_transform(X_train)
            X_val_scaled = self.scalers['standard'].transform(X_val)
            
            # Train and evaluate each model
            ensemble_predictions = {}
            model_scores = {}
            
            print("\n" + "="*50)
            print("TRAINING MODELS")
            print("="*50)
            
            for name, model in self.models.items():
                print(f"\nTraining {name}...")
                
                try:
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Make predictions
                    train_pred = model.predict(X_train_scaled)
                    val_pred = model.predict(X_val_scaled)
                    
                    # Calculate metrics
                    train_acc = accuracy_score(y_train, train_pred)
                    val_acc = accuracy_score(y_val, val_pred)
                    val_balanced_acc = balanced_accuracy_score(y_val, val_pred)
                    
                    # Store predictions and scores
                    ensemble_predictions[name] = val_pred
                    model_scores[name] = val_balanced_acc
                    
                    print(f"  {name} - Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}, Balanced Acc: {val_balanced_acc:.3f}")
                    
                    # Print classification report for best model
                    if val_balanced_acc > 0.5:
                        print(f"\n  Classification Report for {name}:")
                        print(classification_report(y_val, val_pred, target_names=['Sell', 'Buy']))
                        
                except Exception as e:
                    print(f"  Error training {name}: {e}")
                    model_scores[name] = 0
                    ensemble_predictions[name] = np.zeros(len(y_val))
            
            # Calculate ensemble weights based on performance
            total_score = sum(model_scores.values())
            if total_score > 0:
                self.ensemble_weights = {name: score/total_score for name, score in model_scores.items()}
            else:
                # Equal weights if all models failed
                self.ensemble_weights = {name: 1/len(self.models) for name in self.models}
            
            print(f"\nEnsemble weights: {self.ensemble_weights}")
            
            # Evaluate ensemble performance
            ensemble_pred = np.zeros(len(y_val))
            for name, pred in ensemble_predictions.items():
                ensemble_pred += pred * self.ensemble_weights[name]
            
            # Convert ensemble predictions to binary
            ensemble_pred_binary = np.where(ensemble_pred > 0, 1, -1)
            ensemble_balanced_acc = balanced_accuracy_score(y_val, ensemble_pred_binary)
            
            print(f"\nEnsemble Balanced Accuracy: {ensemble_balanced_acc:.3f}")
            
            # Check if performance is acceptable
            best_score = max(model_scores.values())
            print(f"\nBest individual model score: {best_score:.3f}")
            
            # Success criteria: better than random (0.5) for binary classification
            if best_score > 0.52 or ensemble_balanced_acc > 0.52:
                self._save_models()
                self.is_trained = True
                print(f"\n{Fore.GREEN}Model training successful!{Style.RESET_ALL}")
                
                # Save validation metrics
                self.validation_metrics = {
                    'best_model_score': best_score,
                    'ensemble_score': ensemble_balanced_acc,
                    'model_scores': model_scores,
                    'n_train': len(X_train),
                    'n_val': len(X_val)
                }
                
                return True
            else:
                print(f"\n{Fore.RED}Model performance too low. Best score: {best_score:.3f}{Style.RESET_ALL}")
                return False
                
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _save_models(self):
        """Save trained models and metadata"""
        try:
            # Save individual models
            for name, model in self.models.items():
                joblib.dump(model, os.path.join(self.model_dir, f'{name}_model.pkl'))
            
            # Save scaler
            joblib.dump(self.scalers['standard'], os.path.join(self.model_dir, 'scaler.pkl'))
            
            # Save metadata
            metadata = {
                'feature_importance': self.feature_importance,
                'training_features': self.training_features,
                'ensemble_weights': self.ensemble_weights,
                'validation_metrics': self.validation_metrics
            }
            
            with open(os.path.join(self.model_dir, 'model_metadata.json'), 'w') as f:
                json.dump(metadata, f)
                
        except Exception as e:
            logging.error(f"Model save error: {e}")

# ============================================================================
# ADAPTIVE SIGNAL THRESHOLDS
# ============================================================================

class AdaptiveSignalThresholds:
    """Dynamic signal threshold optimization based on market conditions and performance"""
    
    def __init__(self, db_manager):
        self.db = db_manager
        self.base_thresholds = {
            'buy_threshold': 2.0,        # REDUCED from 3.5
            'strong_buy_threshold': 3.0,  # REDUCED from 5.0
            'sell_threshold': -2.0,      # REDUCED from -3.5
            'strong_sell_threshold': -3.0, # REDUCED from -5.0
            'min_signals': 2             # REDUCED from 3
        }
        
        self.adaptive_thresholds = self.base_thresholds.copy()
        self.market_regime_adjustments = {
            'steady_bull': {'buy': 0.9, 'sell': 1.1},  # Lower buy thresholds
            'volatile_bull': {'buy': 1.1, 'sell': 0.9},  # Higher buy thresholds
            'range_bound': {'buy': 0.8, 'sell': 0.8},  # Lower both thresholds
            'bear': {'buy': 1.2, 'sell': 0.8},  # Higher buy, lower sell
            'crisis': {'buy': 1.5, 'sell': 0.7},  # Much higher buy, lower sell
            'unknown': {'buy': 1.0, 'sell': 1.0}  # No adjustment
        }
        
        self.performance_window = 30  # Days to analyze for performance
        self.optimization_history = []
    
    def optimize_thresholds(self, market_regime: str, recent_performance: Dict) -> Dict:
        """
        Dynamically adjust signal thresholds based on:
        1. Market regime
        2. Recent signal performance
        3. Win rate and profitability
        """
        
        # Start with base thresholds
        optimized = self.base_thresholds.copy()
        
        # 1. Apply market regime adjustments
        regime_adj = self.market_regime_adjustments.get(market_regime, {'buy': 1.0, 'sell': 1.0})
        
        optimized['buy_threshold'] *= regime_adj['buy']
        optimized['strong_buy_threshold'] *= regime_adj['buy']
        optimized['sell_threshold'] *= regime_adj['sell']
        optimized['strong_sell_threshold'] *= regime_adj['sell']
        
        # 2. Analyze recent signal performance
        signal_analysis = self._analyze_signal_performance()
        
        if signal_analysis['sample_size'] > 20:
            # Adjust based on win rate
            if signal_analysis['win_rate'] < 0.4:
                # Poor win rate - be more selective
                optimized['buy_threshold'] *= 1.15
                optimized['strong_buy_threshold'] *= 1.15
                optimized['min_signals'] = min(4, optimized['min_signals'] + 1)
                
            elif signal_analysis['win_rate'] > 0.6:
                # Good win rate - can be slightly less selective
                optimized['buy_threshold'] *= 0.95
                optimized['strong_buy_threshold'] *= 0.95
                optimized['min_signals'] = max(2, optimized['min_signals'] - 1)
            
            # Adjust based on average return
            if signal_analysis['avg_return'] < -0.02:
                # Losing money on average - tighten thresholds
                optimized['buy_threshold'] *= 1.2
                optimized['sell_threshold'] *= 0.8
                
            elif signal_analysis['avg_return'] > 0.03:
                # Good returns - current thresholds working well
                pass
        
        # 3. Volatility adjustment
        if 'volatility' in recent_performance:
            vol = recent_performance['volatility']
            if vol > 0.4:  # High volatility
                # Require stronger signals in volatile markets
                vol_multiplier = 1 + (vol - 0.4)
                optimized['buy_threshold'] *= vol_multiplier
                optimized['strong_buy_threshold'] *= vol_multiplier
        
        # 4. Signal frequency adjustment
        if signal_analysis['signal_frequency'] < 0.05:  # Less than 5% of days
            # Too few signals - relax thresholds slightly
            optimized['buy_threshold'] *= 0.9
            optimized['sell_threshold'] *= 0.9
            
        elif signal_analysis['signal_frequency'] > 0.3:  # More than 30% of days
            # Too many signals - tighten thresholds
            optimized['buy_threshold'] *= 1.1
            optimized['sell_threshold'] *= 1.1
        
        # 5. Ensure reasonable bounds
        optimized['buy_threshold'] = np.clip(optimized['buy_threshold'], 1.5, 6.0)
        optimized['strong_buy_threshold'] = np.clip(optimized['strong_buy_threshold'], 2.5, 8.0)
        optimized['sell_threshold'] = np.clip(optimized['sell_threshold'], -6.0, -1.5)
        optimized['strong_sell_threshold'] = np.clip(optimized['strong_sell_threshold'], -8.0, -2.5)
        optimized['min_signals'] = int(np.clip(optimized['min_signals'], 1, 5))
        
        # Store optimization
        self.adaptive_thresholds = optimized
        self._record_optimization(market_regime, signal_analysis, optimized)
        
        return optimized
    
    def _analyze_signal_performance(self) -> Dict:
        """Analyze performance of recent signals"""
        try:
            with self.db.get_connection() as conn:
                # Get recent trades based on signals
                recent_trades = conn.execute("""
                    SELECT * FROM trade_history
                    WHERE trade_date > date('now', '-30 days')
                    ORDER BY trade_date DESC
                """).fetchall()
                
                if not recent_trades:
                    return {
                        'sample_size': 0,
                        'win_rate': 0.5,
                        'avg_return': 0,
                        'signal_frequency': 0.1,
                        'sharpe': 0
                    }
                
                # Calculate metrics
                df = pd.DataFrame(recent_trades)
                
                # Win rate (for completed trades)
                completed = df[df['action'] == 'SELL']
                win_rate = (completed['return_pct'] > 0).mean() if len(completed) > 0 else 0.5
                
                # Average return
                avg_return = completed['return_pct'].mean() / 100 if len(completed) > 0 else 0
                
                # Signal frequency (approximate)
                days_analyzed = 30
                signal_frequency = len(df[df['action'] == 'BUY']) / days_analyzed
                
                # Sharpe ratio of returns
                if len(completed) > 5:
                    returns = completed['return_pct'] / 100
                    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                else:
                    sharpe = 0
                
                return {
                    'sample_size': len(completed),
                    'win_rate': win_rate,
                    'avg_return': avg_return,
                    'signal_frequency': signal_frequency,
                    'sharpe': sharpe,
                    'total_trades': len(df)
                }
                
        except Exception as e:
            logging.error(f"Error analyzing signal performance: {e}")
            return {
                'sample_size': 0,
                'win_rate': 0.5,
                'avg_return': 0,
                'signal_frequency': 0.1,
                'sharpe': 0
            }
    
    def _record_optimization(self, regime: str, analysis: Dict, thresholds: Dict):
        """Record threshold optimization for analysis"""
        self.optimization_history.append({
            'timestamp': pd.Timestamp.now(),
            'regime': regime,
            'analysis': analysis,
            'thresholds': thresholds
        })
        
        # Keep only recent history
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-50:]
    
    def get_threshold_recommendations(self) -> Dict:
        """Get recommendations for threshold adjustments"""
        analysis = self._analyze_signal_performance()
        
        recommendations = {
            'current_thresholds': self.adaptive_thresholds,
            'performance_analysis': analysis,
            'suggestions': []
        }
        
        # Generate suggestions based on analysis
        if analysis['sample_size'] > 10:
            if analysis['win_rate'] < 0.35:
                recommendations['suggestions'].append(
                    "Win rate is low - consider increasing buy thresholds by 20%"
                )
            elif analysis['win_rate'] > 0.65 and analysis['signal_frequency'] < 0.1:
                recommendations['suggestions'].append(
                    "High win rate but few signals - consider decreasing thresholds by 10%"
                )
            
            if analysis['avg_return'] < -0.01:
                recommendations['suggestions'].append(
                    "Negative average returns - review signal factors and increase min_signals requirement"
                )
            
            if analysis['sharpe'] < 0.5:
                recommendations['suggestions'].append(
                    "Low risk-adjusted returns - focus on higher confidence signals only"
                )
        else:
            recommendations['suggestions'].append(
                "Insufficient trade history for optimization - continue monitoring"
            )
        
        return recommendations

# ============================================================================
# DATA VALIDATOR (Original)
# ============================================================================

class DataValidator:
    """Comprehensive data validation system"""
    
    def __init__(self):
        self.validation_rules = {
            'price_range': (0.01, 10000),  # Valid price range
            'volume_min': 1000,  # Minimum volume
            'max_price_change': 0.11,  # 11% limit (10% + buffer)
            'min_data_points': 20,  # Minimum required data points
            'max_gap_days': 15,  # Maximum allowed gap in data (increased for holidays)
            'outlier_threshold': 4  # Standard deviations for outlier detection
        }
    
    def validate_market_data(self, df: pd.DataFrame, symbol: str) -> DataQualityCheck:
        """Comprehensive data validation"""
        issues = []
        
        if df is None or df.empty:
            return DataQualityCheck(
                symbol=symbol,
                timestamp=datetime.datetime.now(),
                is_valid=False,
                issues=["No data available"],
                quality_score=0.0,
                data_points=0
            )
        
        # 1. Check data completeness
        data_points = len(df)
        if data_points < self.validation_rules['min_data_points']:
            issues.append(f"Insufficient data points: {data_points}")
        
        # 2. Check for gaps in dates
        if 'date' in df.index.names:
            dates = df.index
        else:
            dates = pd.to_datetime(df.index)
        
        date_diff = dates.to_series().diff()
        max_gap = date_diff.max()
        if max_gap > pd.Timedelta(days=self.validation_rules['max_gap_days']):
            issues.append(f"Data gap detected: {max_gap.days} days")
        
        # 3. Validate price data
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                # Check price range
                prices = df[col]
                min_price = prices.min()
                max_price = prices.max()
                
                if min_price < self.validation_rules['price_range'][0]:
                    issues.append(f"{col} price too low: {min_price}")
                if max_price > self.validation_rules['price_range'][1]:
                    issues.append(f"{col} price too high: {max_price}")
                
                # Check for zeros or NaN
                if prices.isna().any():
                    issues.append(f"{col} contains NaN values")
                if (prices == 0).any():
                    issues.append(f"{col} contains zero values")
        
        # 4. OHLC consistency check
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            ohlc_issues = (
                (df['high'] < df['low']).any() or
                (df['high'] < df['open']).any() or
                (df['high'] < df['close']).any() or
                (df['low'] > df['open']).any() or
                (df['low'] > df['close']).any()
            )
            if ohlc_issues:
                issues.append("OHLC consistency violation")
        
        # 5. Volume validation
        if 'volume' in df.columns:
            if (df['volume'] < self.validation_rules['volume_min']).all():
                issues.append("Volume too low across all periods")
            if df['volume'].isna().any():
                issues.append("Volume contains NaN values")
        
        # 6. Price change validation
        if 'close' in df.columns and len(df) > 1:
            returns = df['close'].pct_change()
            max_return = returns.abs().max()
            
            if max_return > self.validation_rules['max_price_change']:
                issues.append(f"Excessive price change detected: {max_return:.2%}")
        
        # 7. Outlier detection
        if 'close' in df.columns and len(df) > 20:
            returns = df['close'].pct_change().dropna()
            mean_return = returns.mean()
            std_return = returns.std()
            
            outliers = (returns - mean_return).abs() > (self.validation_rules['outlier_threshold'] * std_return)
            if outliers.any():
                issues.append(f"Outliers detected: {outliers.sum()} data points")
        
        # 8. Check for suspended trading
        if 'volume' in df.columns and len(df) > 5:
            zero_volume_pct = (df['volume'] == 0).sum() / len(df)
            if zero_volume_pct > 0.1:  # More than 10% zero volume
                issues.append(f"Possible trading suspension: {zero_volume_pct:.1%} zero volume")
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(df, issues)
        
        # Determine if data is valid
        critical_issues = [
            "No data available",
            "OHLC consistency violation",
            "Insufficient data points"
        ]
        
        is_valid = not any(issue in str(issues) for issue in critical_issues)
        
        return DataQualityCheck(
            symbol=symbol,
            timestamp=datetime.datetime.now(),
            is_valid=is_valid,
            issues=issues,
            quality_score=quality_score,
            data_points=data_points
        )
    
    def _calculate_quality_score(self, df: pd.DataFrame, issues: List[str]) -> float:
        """Calculate data quality score (0-1)"""
        if not isinstance(df, pd.DataFrame) or df.empty:
            return 0.0
        
        score = 1.0
        
        # Deduct for each issue
        issue_penalties = {
            'Insufficient data points': 0.3,
            'Data gap detected': 0.2,
            'NaN values': 0.15,
            'zero values': 0.15,
            'OHLC consistency': 0.25,
            'Volume too low': 0.1,
            'Excessive price change': 0.15,
            'Outliers detected': 0.1,
            'trading suspension': 0.2
        }
        
        for issue in issues:
            for key, penalty in issue_penalties.items():
                if key in issue:
                    score -= penalty
        
        # Bonus for good data
        if len(df) > 60:
            score += 0.1
        if 'volume' in df.columns and df['volume'].mean() > 1000000:
            score += 0.05
        
        return max(0.0, min(1.0, score))

# ============================================================================
# FAILSAFE MANAGER (Original)
# ============================================================================

class FailsafeManager:
    """Comprehensive failsafe and emergency control system"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.emergency_stop = False
        self.circuit_breakers = {
            'max_daily_trades': 20,
            'max_position_size': 0.1,  # 10% of portfolio
            'max_daily_loss': 0.05,  # 5% daily loss limit
            'max_positions': 20,
            'min_cash_ratio': 0.1,  # Keep 10% in cash
            'max_correlation': 0.9,  # Position correlation limit
            'min_data_quality': 0.7  # Minimum data quality score (RAISED FROM 0.6)
        }
        self.daily_stats = {
            'trades_today': 0,
            'loss_today': 0.0,
            'last_reset': datetime.date.today()
        }
        
    def check_all_systems(self) -> Tuple[bool, List[str]]:
        """Run all system checks"""
        issues = []
        
        # Check emergency stop
        if self.emergency_stop:
            issues.append("EMERGENCY STOP ACTIVE")
            return False, issues
        
        # Check database integrity
        if not self.db.verify_integrity():
            issues.append("Database integrity check failed")
        
        # Check daily limits
        self._reset_daily_stats_if_needed()
        
        if self.daily_stats['trades_today'] >= self.circuit_breakers['max_daily_trades']:
            issues.append(f"Daily trade limit reached: {self.daily_stats['trades_today']}")
        
        if self.daily_stats['loss_today'] >= self.circuit_breakers['max_daily_loss']:
            issues.append(f"Daily loss limit reached: {self.daily_stats['loss_today']:.2%}")
        
        # Check system resources
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            if cpu_percent > 90:
                issues.append(f"High CPU usage: {cpu_percent}%")
            if memory_percent > 90:
                issues.append(f"High memory usage: {memory_percent}%")
        except:
            pass  # psutil not available
        
        # All clear if no issues
        return len(issues) == 0, issues
    
    def validate_trade(self, symbol: str, action: str, quantity: int, 
                      price: float, portfolio_value: float) -> Tuple[bool, str]:
        """Validate trade against all safety rules"""
        
        # Check emergency stop
        if self.emergency_stop:
            return False, "Emergency stop is active"
        
        # Check position size
        position_value = quantity * price
        position_pct = position_value / portfolio_value
        
        if position_pct > self.circuit_breakers['max_position_size']:
            return False, f"Position too large: {position_pct:.1%} of portfolio"
        
        # Check daily trade limit
        if self.daily_stats['trades_today'] >= self.circuit_breakers['max_daily_trades']:
            return False, "Daily trade limit reached"
        
        # Additional checks for SELL orders
        if action == 'BUY':
            # Check cash ratio
            with self.db.get_connection() as conn:
                positions_count = conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
                if positions_count >= self.circuit_breakers['max_positions']:
                    return False, f"Maximum positions limit reached: {positions_count}"
        
        return True, "Trade validated"
    
    def record_trade(self, pnl: float = 0.0):
        """Record trade for monitoring"""
        self.daily_stats['trades_today'] += 1
        if pnl < 0:
            self.daily_stats['loss_today'] += abs(pnl)
        
        # Save to database
        with self.db.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO system_state (key, value)
                VALUES (?, ?)
            """, ('daily_stats', json.dumps(self.daily_stats, default=str)))
            conn.commit()
    
    def _reset_daily_stats_if_needed(self):
        """Reset daily statistics if new day"""
        if self.daily_stats['last_reset'] < datetime.date.today():
            self.daily_stats = {
                'trades_today': 0,
                'loss_today': 0.0,
                'last_reset': datetime.date.today()
            }
    
    def activate_emergency_stop(self, reason: str):
        """Activate emergency stop"""
        self.emergency_stop = True
        logging.critical(f"EMERGENCY STOP ACTIVATED: {reason}")
        
        # Save state
        with self.db.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO system_state (key, value)
                VALUES ('emergency_stop', ?)
            """, (json.dumps({'active': True, 'reason': reason, 'timestamp': str(datetime.datetime.now())}),))
            conn.commit()
    
    def deactivate_emergency_stop(self):
        """Deactivate emergency stop"""
        self.emergency_stop = False
        logging.info("Emergency stop deactivated")
        
        with self.db.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO system_state (key, value)
                VALUES ('emergency_stop', ?)
            """, (json.dumps({'active': False, 'timestamp': str(datetime.datetime.now())}),))
            conn.commit()

# ============================================================================
# PRE-MARKET CHECKER (Original)
# ============================================================================

class PreMarketChecker:
    """Pre-market checklist system for A-share trading"""
    
    def __init__(self, system):
        self.system = system
        self.logger = logging.getLogger(__name__)
        
    def pre_market_checklist(self) -> Tuple[bool, Dict]:
        """Run comprehensive pre-market checks before trading"""
        print(f"\n{Fore.CYAN}=== PRE-MARKET CHECKLIST ==={Style.RESET_ALL}")
        print(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        checks = {
            'database_integrity': self._check_database(),
            'data_freshness': self._check_data_freshness(),
            'system_resources': self._check_system_resources(),
            'internet_connection': self._test_connection(),
            'trading_calendar': self._is_trading_day(),
            'market_hours': self._check_market_hours(),
            'emergency_stop': self._check_emergency_stop(),
            'risk_limits': self._check_risk_limits(),
            'data_providers': self._check_data_providers(),
            'ml_model_status': self._check_ml_status()
        }
        
        # Display results
        all_passed = True
        for check_name, (passed, message) in checks.items():
            status = f"{Fore.GREEN}✓ PASS{Style.RESET_ALL}" if passed else f"{Fore.RED}✗ FAIL{Style.RESET_ALL}"
            print(f"  {check_name}: {status} - {message}")
            if not passed:
                all_passed = False
        
        # Generate summary
        if all_passed:
            print(f"\n{Fore.GREEN}All checks passed! System ready for trading.{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.RED}Some checks failed! Review issues before trading.{Style.RESET_ALL}")
        
        return all_passed, checks
    
    def _check_database(self) -> Tuple[bool, str]:
        """Check database integrity and backup status"""
        try:
            # Check integrity
            if not self.system.db.verify_integrity():
                return False, "Database integrity check failed"
            
            # Check backup freshness
            if os.path.exists(self.system.db.backup_path):
                backup_time = os.path.getmtime(self.system.db.backup_path)
                hours_old = (time.time() - backup_time) / 3600
                if hours_old > 24:
                    return False, f"Backup is {hours_old:.1f} hours old (>24h)"
            else:
                return False, "No backup found"
            
            return True, "Database OK, backup current"
        except Exception as e:
            return False, f"Database check error: {e}"
    
    def _check_data_freshness(self) -> Tuple[bool, str]:
        """Check if market data is fresh"""
        try:
            # Try multiple symbols and longer periods
            test_symbols = ['000001', '000002', '600519']
            
            for symbol in test_symbols:
                df = self.system.get_market_data_cached(symbol, days=30)  # Use longer period
                if df is not None and len(df) > 5:  # Lower threshold
                    latest_date = df.index[-1]
                    today = datetime.date.today()
                    
                    # More lenient for future dates
                    days_diff = (today - latest_date.date()).days
                    if days_diff > 10:  # More lenient threshold
                        continue
                    
                    return True, f"Latest data from {latest_date.date()} for {symbol}"
            
            return False, "Cannot fetch sufficient market data from any test symbol"
        except Exception as e:
            return False, f"Data freshness check error: {e}"
    
    def _check_system_resources(self) -> Tuple[bool, str]:
        """Check CPU, memory, and disk space"""
        try:
            import psutil
            
            # CPU check
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 80:
                return False, f"High CPU usage: {cpu_percent}%"
            
            # Memory check
            memory = psutil.virtual_memory()
            if memory.percent > 85:
                return False, f"High memory usage: {memory.percent}%"
            
            # Disk space check
            disk = psutil.disk_usage('/')
            if disk.percent > 90:
                return False, f"Low disk space: {100-disk.percent:.1f}% free"
            
            return True, f"CPU: {cpu_percent}%, Memory: {memory.percent}%, Disk: {disk.percent}%"
        except ImportError:
            return True, "Resource check skipped (psutil not installed)"
        except Exception as e:
            return False, f"Resource check error: {e}"
    
    def _test_connection(self) -> Tuple[bool, str]:
        """Test internet connectivity to critical services"""
        try:
            # Test connection to data provider - CORRECTED URL
            socket.setdefaulttimeout(5)
            socket.create_connection(("akshare.akfamily.xyz", 443)).close()  # Using HTTPS port
            
            return True, "Internet connection OK"
        except Exception as e:
            return False, f"Connection error: {e}"
        finally:
            socket.setdefaulttimeout(None)
    
    def _is_trading_day(self) -> Tuple[bool, str]:
        """Check if today is a trading day"""
        today = datetime.date.today()
        weekday = today.weekday()
        
        # Basic weekday check
        if weekday >= 5:  # Saturday = 5, Sunday = 6
            return False, f"Weekend - {today.strftime('%A')}"
        
        # TODO: Add holiday calendar check
        # For now, just check weekdays
        return True, f"Trading day - {today.strftime('%Y-%m-%d %A')}"
    
    def _check_market_hours(self) -> Tuple[bool, str]:
        """Check if within or near market hours"""
        now = datetime.datetime.now()
        current_time = now.time()
        
        # A-share market hours
        morning_open = datetime.time(9, 30)
        morning_close = datetime.time(11, 30)
        afternoon_open = datetime.time(13, 0)
        afternoon_close = datetime.time(15, 0)
        
        # Pre-market window (9:00-9:30)
        pre_market_start = datetime.time(9, 0)
        
        if current_time < pre_market_start:
            return True, f"Pre-market preparation time"
        elif pre_market_start <= current_time < morning_open:
            return True, f"Pre-market session (opens at 9:30)"
        elif morning_open <= current_time <= morning_close:
            return True, f"Morning session active"
        elif morning_close < current_time < afternoon_open:
            return True, f"Lunch break (reopens at 13:00)"
        elif afternoon_open <= current_time <= afternoon_close:
            return True, f"Afternoon session active"
        else:
            return True, f"After hours (market closed)"
    
    def _check_emergency_stop(self) -> Tuple[bool, str]:
        """Check emergency stop status"""
        if self.system.failsafe.emergency_stop:
            return False, "EMERGENCY STOP IS ACTIVE"
        return True, "Emergency stop inactive"
    
    def _check_risk_limits(self) -> Tuple[bool, str]:
        """Check if within risk limits"""
        # Check daily trading limits
        if self.system.failsafe.daily_stats['trades_today'] >= self.system.failsafe.circuit_breakers['max_daily_trades']:
            return False, f"Daily trade limit reached ({self.system.failsafe.daily_stats['trades_today']})"
        
        # Check loss limits
        if self.system.failsafe.daily_stats['loss_today'] >= self.system.failsafe.circuit_breakers['max_daily_loss']:
            return False, f"Daily loss limit reached ({self.system.failsafe.daily_stats['loss_today']:.2%})"
        
        return True, "Within risk limits"
    
    def _check_data_providers(self) -> Tuple[bool, str]:
        """Check data provider availability"""
        try:
            # Quick test fetch
            import akshare as ak
            test_df = ak.stock_zh_a_spot_em()
            if test_df is None or test_df.empty:
                return False, "Data provider returned empty data"
            return True, "Data providers responding"
        except Exception as e:
            return False, f"Data provider error: {str(e)[:50]}"
    
    def _check_ml_status(self) -> Tuple[bool, str]:
        """Check ML model status"""
        if not self.system.ml_engine.is_trained:
            return False, "ML model not trained"
        
        # Check model age (if implemented)
        return True, "ML model ready"

# ============================================================================
# REALISTIC EXECUTION MODEL (Original)
# ============================================================================

class RealisticExecutionModel:
    """Realistic execution model with slippage and market impact for A-shares"""
    
    def __init__(self, system):
        self.system = system
        self.logger = logging.getLogger(__name__)
        
        # A-share specific parameters
        self.price_tick = 0.01  # Minimum price movement
        self.lot_size = 100     # Minimum trading unit
        
        # Commission structure for A-shares
        self.commission_rate = 0.0008  # 0.08% (negotiable, using higher retail rate)
        self.stamp_duty = 0.001        # 0.1% on sells only
        self.min_commission = 5.0      # Minimum 5 CNY per trade
        
    def estimate_slippage(self, symbol: str, action: str, quantity: int) -> float:
        """
        Estimate slippage based on order size, liquidity, and market conditions
        
        Factors considered:
        - Order size relative to average volume
        - Current volatility
        - Time of day
        - Market regime
        """
        try:
            # Get recent market data
            df = self.system.get_market_data_cached(symbol, days=20)
            if df is None:
                return 0.002  # Default 0.2% if no data
            
            # Calculate average daily volume
            avg_volume = df['volume'].tail(10).mean()
            if avg_volume == 0:
                return 0.005  # High slippage for illiquid stocks
            
            # Order size impact
            order_impact = quantity / avg_volume
            
            # Volatility impact
            returns = df['close'].pct_change()
            volatility = returns.tail(10).std()
            
            # Time of day impact
            now = datetime.datetime.now()
            time_factor = self._get_time_impact(now)
            
            # Market regime impact
            regime_factor = 1.0
            if self.system.market_regime:
                regime = self.system.market_regime.get('regime', 'unknown')
                regime_factors = {
                    'crisis': 2.0,
                    'volatile_bull': 1.5,
                    'bear': 1.3,
                    'range_bound': 1.0,
                    'steady_bull': 0.8
                }
                regime_factor = regime_factors.get(regime, 1.0)
            
            # Calculate base slippage
            base_slippage = 0.001  # 0.1% base
            
            # Order size component (linear + square root for large orders)
            size_impact = order_impact * 0.1 + np.sqrt(order_impact) * 0.02
            
            # Volatility component
            vol_impact = volatility * 10  # Roughly converts daily vol to slippage
            
            # Direction component (buying usually has more slippage in A-shares)
            direction_factor = 1.2 if action == 'BUY' else 1.0
            
            # Combine all factors
            total_slippage = (base_slippage + size_impact + vol_impact) * time_factor * regime_factor * direction_factor
            
            # Cap slippage at reasonable levels
            max_slippage = 0.01  # 1% maximum
            total_slippage = min(total_slippage, max_slippage)
            
            self.logger.info(f"Slippage estimate for {symbol}: {total_slippage:.3%} "
                           f"(impact: {order_impact:.1%}, vol: {volatility:.1%})")
            
            return total_slippage
            
        except Exception as e:
            self.logger.error(f"Error estimating slippage: {e}")
            return 0.003  # Default 0.3% on error
    
    def _get_time_impact(self, current_time: datetime.datetime) -> float:
        """Get time-based impact factor for A-share market"""
        hour = current_time.hour
        minute = current_time.minute
        
        # Opening auction (9:15-9:25) and first 30 min: high impact
        if (hour == 9 and minute < 30) or (hour == 9 and minute >= 30) or (hour == 10 and minute < 0):
            return 1.5
        
        # Closing 30 minutes: high impact
        elif hour == 14 and minute >= 30:
            return 1.5
        
        # Lunch reopening: medium impact
        elif hour == 13 and minute < 30:
            return 1.2
        
        # Normal trading hours: normal impact
        else:
            return 1.0
    
    def calculate_transaction_costs(self, symbol: str, action: str, quantity: int, price: float) -> Dict:
        """Calculate all transaction costs for A-share trading"""
        order_value = quantity * price
        
        # Commission (both buy and sell)
        commission = max(order_value * self.commission_rate, self.min_commission)
        
        # Stamp duty (sell only)
        stamp_duty = order_value * self.stamp_duty if action == 'SELL' else 0
        
        # Slippage estimate
        slippage_rate = self.estimate_slippage(symbol, action, quantity)
        slippage_cost = order_value * slippage_rate
        
        # Total costs
        total_costs = commission + stamp_duty + slippage_cost
        
        return {
            'commission': commission,
            'stamp_duty': stamp_duty,
            'slippage': slippage_cost,
            'slippage_rate': slippage_rate,
            'total_costs': total_costs,
            'cost_rate': total_costs / order_value,
            'estimated_fill_price': price * (1 + slippage_rate) if action == 'BUY' else price * (1 - slippage_rate)
        }
    
    def validate_order(self, symbol: str, action: str, quantity: int, price: float) -> Tuple[bool, str]:
        """Validate order against A-share trading rules"""
        # Check lot size (must be multiple of 100)
        if quantity % self.lot_size != 0:
            return False, f"Quantity must be multiple of {self.lot_size}"
        
        # Check minimum order size
        if quantity < self.lot_size:
            return False, f"Minimum order size is {self.lot_size} shares"
        
        # Check price tick compliance
        if price % self.price_tick > 0.0001:  # Small tolerance for float precision
            return False, f"Price must be multiple of {self.price_tick}"
        
        # Check daily price limits (±10% for main board, ±20% for STAR/ChiNext with registration)
        df = self.system.get_market_data_cached(symbol, days=2)
        if df is not None and len(df) >= 2:
            prev_close = df['close'].iloc[-2]
            
            # Determine board type (simplified - you may need more sophisticated logic)
            if symbol.startswith('688') or symbol.startswith('300'):
                # STAR Market or ChiNext with registration system
                price_limit = 0.20
            else:
                # Main board
                price_limit = 0.10
            
            price_change = (price - prev_close) / prev_close
            if abs(price_change) > price_limit:
                return False, f"Price exceeds daily limit ({price_change:.1%} vs ±{price_limit:.0%})"
        
        return True, "Order validated"

# ============================================================================
# PAPER TRADING MODE (Original)
# ============================================================================

class PaperTradingMode:
    """Paper trading mode for testing strategies without real money"""
    
    def __init__(self, real_system, initial_capital: float = 1000000):
        self.real_system = real_system
        self.initial_capital = initial_capital
        self.paper_cash = initial_capital
        self.paper_positions = {}
        self.paper_trades = []
        self.paper_orders = []
        self.start_date = datetime.date.today()
        self.execution_model = RealisticExecutionModel(real_system)
        
        # Performance tracking
        self.daily_values = []
        self.peak_value = initial_capital
        self.max_drawdown = 0
        
        # Initialize paper trading database tables
        self._init_paper_db()
    
    def _init_paper_db(self):
        """Initialize paper trading tables in database"""
        # Tables are already created in DatabaseManager
        pass
    
    def execute_trade(self, symbol: str, action: str, quantity: int, order_price: float, 
                     signal_strength: float = 0, signal_reason: str = "") -> Dict:
        """Execute paper trade with realistic simulation"""
        
        # Validate order
        valid, reason = self.execution_model.validate_order(symbol, action, quantity, order_price)
        if not valid:
            return {'success': False, 'reason': reason}
        
        # Calculate transaction costs
        costs = self.execution_model.calculate_transaction_costs(symbol, action, quantity, order_price)
        fill_price = costs['estimated_fill_price']
        
        # Check if we have enough cash for buy orders
        if action == 'BUY':
            required_cash = quantity * fill_price + costs['commission']
            if required_cash > self.paper_cash:
                return {'success': False, 'reason': f'Insufficient cash: need {required_cash:.2f}, have {self.paper_cash:.2f}'}
        
        # Check if we have position for sell orders
        elif action == 'SELL':
            if symbol not in self.paper_positions or self.paper_positions[symbol]['quantity'] < quantity:
                return {'success': False, 'reason': 'Insufficient position to sell'}
        
        # Execute trade
        timestamp = datetime.datetime.now()
        trade_pnl = 0
        return_pct = 0
        
        if action == 'BUY':
            # Update cash
            total_cost = quantity * fill_price + costs['commission']
            self.paper_cash -= total_cost
            
            # Update or create position
            if symbol in self.paper_positions:
                pos = self.paper_positions[symbol]
                new_quantity = pos['quantity'] + quantity
                new_cost_basis = pos['cost_basis'] + total_cost
                new_avg_price = new_cost_basis / new_quantity
                
                self.paper_positions[symbol] = {
                    'quantity': new_quantity,
                    'avg_price': new_avg_price,
                    'cost_basis': new_cost_basis,
                    'last_price': fill_price,
                    'last_update': timestamp
                }
            else:
                self.paper_positions[symbol] = {
                    'quantity': quantity,
                    'avg_price': fill_price,
                    'cost_basis': total_cost,
                    'last_price': fill_price,
                    'last_update': timestamp
                }
        
        else:  # SELL
            pos = self.paper_positions[symbol]
            
            # Calculate proceeds and P&L
            gross_proceeds = quantity * fill_price
            net_proceeds = gross_proceeds - costs['commission'] - costs['stamp_duty']
            self.paper_cash += net_proceeds
            
            # Calculate P&L
            cost_basis = pos['avg_price'] * quantity
            trade_pnl = net_proceeds - cost_basis
            return_pct = (trade_pnl / cost_basis) * 100
            
            # Update position
            remaining_quantity = pos['quantity'] - quantity
            if remaining_quantity == 0:
                del self.paper_positions[symbol]
            else:
                pos['quantity'] = remaining_quantity
                pos['cost_basis'] = pos['avg_price'] * remaining_quantity
        
        # Record trade
        portfolio_value = self._calculate_portfolio_value()
        
        trade_record = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'order_price': order_price,
            'fill_price': fill_price,
            'commission': costs['commission'],
            'stamp_duty': costs['stamp_duty'],
            'slippage': costs['slippage'],
            'slippage_rate': costs['slippage_rate'],
            'pnl': trade_pnl,
            'return_pct': return_pct,
            'cash_after': self.paper_cash,
            'portfolio_value': portfolio_value,
            'signal_strength': signal_strength,
            'signal_reason': signal_reason
        }
        
        self.paper_trades.append(trade_record)
        self._save_trade_to_db(trade_record)
        
        # Log trade
        action_color = Fore.GREEN if action == 'BUY' else Fore.RED
        print(f"\n{Fore.CYAN}[PAPER TRADE]{Style.RESET_ALL} {action_color}{action}{Style.RESET_ALL} {symbol}")
        print(f"  Quantity: {quantity:,}")
        print(f"  Order price: {order_price:.2f}")
        print(f"  Fill price: {fill_price:.2f} (slippage: {costs['slippage_rate']:.2%})")
        print(f"  Commission: {costs['commission']:.2f}")
        if action == 'SELL':
            print(f"  Stamp duty: {costs['stamp_duty']:.2f}")
            pnl_color = Fore.GREEN if trade_pnl > 0 else Fore.RED
            print(f"  P&L: {pnl_color}{trade_pnl:+.2f} ({return_pct:+.1f}%){Style.RESET_ALL}")
        print(f"  Cash: {self.paper_cash:,.2f}")
        print(f"  Portfolio value: {portfolio_value:,.2f}")
        
        return {
            'success': True,
            'trade': trade_record
        }
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        positions_value = 0
        
        for symbol, pos in self.paper_positions.items():
            # Get current price
            df = self.real_system.get_market_data_cached(symbol)
            if df is not None and len(df) > 0:
                current_price = df['close'].iloc[-1]
                pos['last_price'] = current_price
                positions_value += pos['quantity'] * current_price
            else:
                # Use last known price
                positions_value += pos['quantity'] * pos.get('last_price', pos['avg_price'])
        
        return self.paper_cash + positions_value
    
    def update_daily_performance(self):
        """Update daily performance metrics"""
        today = datetime.date.today()
        portfolio_value = self._calculate_portfolio_value()
        
        # Calculate positions value
        positions_value = portfolio_value - self.paper_cash
        
        # Calculate returns
        total_return = (portfolio_value / self.initial_capital - 1) * 100
        
        # Calculate daily return
        daily_return = 0
        if self.daily_values:
            last_value = self.daily_values[-1]['portfolio_value']
            daily_return = (portfolio_value / last_value - 1) * 100
        
        # Update max drawdown
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        
        drawdown = (portfolio_value - self.peak_value) / self.peak_value * 100
        self.max_drawdown = min(self.max_drawdown, drawdown)
        
        # Count today's trades
        today_trades = len([t for t in self.paper_trades 
                          if t['timestamp'].date() == today])
        
        # Save to tracking
        performance = {
            'date': today,
            'portfolio_value': portfolio_value,
            'cash': self.paper_cash,
            'positions_value': positions_value,
            'daily_return': daily_return,
            'total_return': total_return,
            'max_drawdown': self.max_drawdown,
            'n_trades': today_trades,
            'n_positions': len(self.paper_positions)
        }
        
        self.daily_values.append(performance)
        self._save_performance_to_db(performance)
        
        return performance
    
    def generate_report(self) -> Dict:
        """Generate comprehensive paper trading report"""
        if not self.paper_trades:
            return {'error': 'No trades to analyze'}
        
        # Update current performance
        current_performance = self.update_daily_performance()
        
        # Analyze trades
        trades_df = pd.DataFrame(self.paper_trades)
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        
        # Calculate metrics
        total_trades = len(self.paper_trades)
        buy_trades_count = len(trades_df[trades_df['action'] == 'BUY'])
        sell_trades_count = len(sell_trades)
        
        # Win rate and P&L stats
        if len(sell_trades) > 0:
            winning_trades = sell_trades[sell_trades['pnl'] > 0]
            win_rate = len(winning_trades) / len(sell_trades) * 100
            
            avg_win = winning_trades['return_pct'].mean() if len(winning_trades) > 0 else 0
            avg_loss = sell_trades[sell_trades['pnl'] < 0]['return_pct'].mean() if len(sell_trades[sell_trades['pnl'] < 0]) > 0 else 0
            
            total_pnl = sell_trades['pnl'].sum()
            best_trade = sell_trades.loc[sell_trades['return_pct'].idxmax()]
            worst_trade = sell_trades.loc[sell_trades['return_pct'].idxmin()]
        else:
            win_rate = avg_win = avg_loss = total_pnl = 0
            best_trade = worst_trade = None
        
        # Calculate Sharpe ratio
        if len(self.daily_values) > 1:
            returns = pd.Series([d['daily_return'] for d in self.daily_values[1:]])
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe = 0
        
        # Transaction cost analysis
        total_commission = trades_df['commission'].sum()
        total_stamp_duty = trades_df['stamp_duty'].sum()
        total_slippage = trades_df['slippage'].sum()
        total_costs = total_commission + total_stamp_duty + total_slippage
        
        report = {
            'summary': {
                'start_date': self.start_date,
                'end_date': datetime.date.today(),
                'days_active': (datetime.date.today() - self.start_date).days,
                'initial_capital': self.initial_capital,
                'final_value': current_performance['portfolio_value'],
                'total_return': current_performance['total_return'],
                'max_drawdown': self.max_drawdown,
                'sharpe_ratio': sharpe
            },
            'trading_stats': {
                'total_trades': total_trades,
                'buy_trades': buy_trades_count,
                'sell_trades': sell_trades_count,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'total_pnl': total_pnl,
                'best_trade': {
                    'symbol': best_trade['symbol'],
                    'return': best_trade['return_pct'],
                    'pnl': best_trade['pnl']
                } if best_trade is not None else None,
                'worst_trade': {
                    'symbol': worst_trade['symbol'],
                    'return': worst_trade['return_pct'],
                    'pnl': worst_trade['pnl']
                } if worst_trade is not None else None
            },
            'cost_analysis': {
                'total_commission': total_commission,
                'total_stamp_duty': total_stamp_duty,
                'total_slippage': total_slippage,
                'total_costs': total_costs,
                'costs_pct_of_volume': total_costs / trades_df['quantity'].dot(trades_df['fill_price']) * 100 if len(trades_df) > 0 else 0
            },
            'current_state': {
                'cash': self.paper_cash,
                'positions': len(self.paper_positions),
                'positions_list': [
                    {
                        'symbol': symbol,
                        'quantity': pos['quantity'],
                        'avg_price': pos['avg_price'],
                        'last_price': pos.get('last_price', pos['avg_price']),
                        'unrealized_pnl': pos['quantity'] * (pos.get('last_price', pos['avg_price']) - pos['avg_price']),
                        'unrealized_pct': (pos.get('last_price', pos['avg_price']) / pos['avg_price'] - 1) * 100
                    }
                    for symbol, pos in self.paper_positions.items()
                ]
            }
        }
        
        return report
    
    def display_report(self):
        """Display formatted paper trading report"""
        report = self.generate_report()
        
        if 'error' in report:
            print(f"{Fore.RED}{report['error']}{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}PAPER TRADING PERFORMANCE REPORT{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        
        # Summary
        summary = report['summary']
        print(f"\n{Fore.YELLOW}Summary:{Style.RESET_ALL}")
        print(f"  Period: {summary['start_date']} to {summary['end_date']} ({summary['days_active']} days)")
        print(f"  Initial capital: {summary['initial_capital']:,.0f}")
        print(f"  Final value: {summary['final_value']:,.0f}")
        
        return_color = Fore.GREEN if summary['total_return'] > 0 else Fore.RED
        print(f"  Total return: {return_color}{summary['total_return']:+.2f}%{Style.RESET_ALL}")
        
        dd_color = Fore.YELLOW if summary['max_drawdown'] > -10 else Fore.RED
        print(f"  Max drawdown: {dd_color}{summary['max_drawdown']:.2f}%{Style.RESET_ALL}")
        print(f"  Sharpe ratio: {summary['sharpe_ratio']:.2f}")
        
        # Trading stats
        stats = report['trading_stats']
        print(f"\n{Fore.YELLOW}Trading Statistics:{Style.RESET_ALL}")
        print(f"  Total trades: {stats['total_trades']} ({stats['buy_trades']} buys, {stats['sell_trades']} sells)")
        
        if stats['sell_trades'] > 0:
            wr_color = Fore.GREEN if stats['win_rate'] > 50 else Fore.YELLOW if stats['win_rate'] > 40 else Fore.RED
            print(f"  Win rate: {wr_color}{stats['win_rate']:.1f}%{Style.RESET_ALL}")
            print(f"  Avg win: +{stats['avg_win']:.2f}%")
            print(f"  Avg loss: {stats['avg_loss']:.2f}%")
            
            pnl_color = Fore.GREEN if stats['total_pnl'] > 0 else Fore.RED
            print(f"  Total P&L: {pnl_color}{stats['total_pnl']:+,.2f}{Style.RESET_ALL}")
            
            if stats['best_trade']:
                print(f"  Best trade: {stats['best_trade']['symbol']} (+{stats['best_trade']['return']:.1f}%)")
            if stats['worst_trade']:
                print(f"  Worst trade: {stats['worst_trade']['symbol']} ({stats['worst_trade']['return']:.1f}%)")
        
        # Cost analysis
        costs = report['cost_analysis']
        print(f"\n{Fore.YELLOW}Transaction Costs:{Style.RESET_ALL}")
        print(f"  Commission: {costs['total_commission']:,.2f}")
        print(f"  Stamp duty: {costs['total_stamp_duty']:,.2f}")
        print(f"  Slippage: {costs['total_slippage']:,.2f}")
        print(f"  Total costs: {costs['total_costs']:,.2f} ({costs['costs_pct_of_volume']:.2%} of volume)")
        
        # Current positions
        current = report['current_state']
        print(f"\n{Fore.YELLOW}Current State:{Style.RESET_ALL}")
        print(f"  Cash: {current['cash']:,.2f}")
        print(f"  Positions: {current['positions']}")
        
        if current['positions_list']:
            print(f"\n  {Fore.CYAN}Open Positions:{Style.RESET_ALL}")
            for pos in current['positions_list']:
                pnl_color = Fore.GREEN if pos['unrealized_pnl'] > 0 else Fore.RED
                print(f"    {pos['symbol']}: {pos['quantity']} @ {pos['avg_price']:.2f}")
                print(f"      Current: {pos['last_price']:.2f} ({pnl_color}{pos['unrealized_pct']:+.1f}%{Style.RESET_ALL})")
    
    def _save_trade_to_db(self, trade: Dict):
        """Save paper trade to database"""
        with self.real_system.db.get_connection() as conn:
            conn.execute("""
                INSERT INTO paper_trades 
                (timestamp, symbol, action, quantity, order_price, fill_price,
                 commission, stamp_duty, slippage, pnl, return_pct,
                 cash_after, portfolio_value, signal_strength, signal_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade['timestamp'], trade['symbol'], trade['action'],
                trade['quantity'], trade['order_price'], trade['fill_price'],
                trade['commission'], trade['stamp_duty'], trade['slippage'],
                trade['pnl'], trade['return_pct'], trade['cash_after'],
                trade['portfolio_value'], trade['signal_strength'], trade['signal_reason']
            ))
            conn.commit()
    
    def _save_performance_to_db(self, performance: Dict):
        """Save daily performance to database"""
        with self.real_system.db.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO paper_performance
                (date, portfolio_value, cash, positions_value, daily_return,
                 total_return, max_drawdown, n_trades, n_positions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                performance['date'], performance['portfolio_value'],
                performance['cash'], performance['positions_value'],
                performance['daily_return'], performance['total_return'],
                performance['max_drawdown'], performance['n_trades'],
                performance['n_positions']
            ))
            conn.commit()

# ============================================================================
# ENHANCED MANUAL TRADING SYSTEM (Main Class with Improvements)
# ============================================================================

class EnhancedManualTradingSystem:
    """Production-ready trading system with all fixes and improvements implemented"""
    
    def __init__(self):
        self.today = datetime.datetime.now().date()
        
        # Initialize database
        self.db = DatabaseManager()
        
        # Initialize improved ML engine
        self.ml_engine = ImprovedMLTradingEngine()
        
        # Initialize components
        self.data_validator = DataValidator()
        self.failsafe = FailsafeManager(self.db)
        
        # Initialize error handler
        self.logger = self._setup_logging()
        self.error_handler = ErrorHandler(self.logger, self.failsafe)
        
        # Initialize adaptive thresholds
        self.threshold_optimizer = AdaptiveSignalThresholds(self.db)
        
        # Initialize new components
        self.pre_market_checker = PreMarketChecker(self)
        self.execution_model = RealisticExecutionModel(self)
        self.paper_trading = PaperTradingMode(self)
        
        # Cache for data efficiency
        self.data_cache = {}
        self.cache_expiry = 3600  # 1 hour
        
        # Risk parameters - UPDATED WITH HIGHER DATA QUALITY REQUIREMENT
        self.risk_params = {
            'max_position_pct': 0.05,  # 5% max per position
            'stop_loss': 0.05,  # 5% stop loss
            'take_profit': 0.15,  # 15% take profit
            'max_portfolio_risk': 0.02,  # 2% max portfolio risk (VaR)
            'volatility_lookback': 20,  # Days for volatility calc
            'confidence_level': 0.95,  # For VaR calculation
            'min_data_quality': 0.7  # Minimum acceptable data quality (RAISED FROM 0.6)
        }
        
        # Alert system
        self.alerts = []
        self.alert_thread = None
        self.monitoring_active = False
        
        # Market data
        self.market_regime = None
        self.sector_data = {}
        
        # Load positions and watchlist from database
        self.positions = self._load_positions_from_db()
        self.watchlist = self._load_watchlist_from_db()
        
        # Verify system integrity on startup
        self._verify_system_integrity()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # File handler for all logs
        file_handler = logging.FileHandler('trading_system.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # File handler for errors only
        error_handler = logging.FileHandler('trading_errors.log')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(log_format))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(log_format))
        
        # Setup logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(error_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _verify_system_integrity(self):
        """Verify system integrity on startup"""
        self.logger.info("Starting system integrity verification...")
        
        # Check database
        if not self.db.verify_integrity():
            self.logger.error("Database integrity check failed!")
            self.failsafe.activate_emergency_stop("Database integrity check failed")
        
        # Check failsafe status
        system_ok, issues = self.failsafe.check_all_systems()
        if not system_ok:
            self.logger.warning(f"System issues detected: {issues}")
        
        # Backup database
        if self.db.backup_database():
            self.logger.info("Database backup created successfully")
        
        self.logger.info("System integrity verification completed")
    
    def _load_positions_from_db(self) -> Dict:
        """Load positions from database"""
        positions = {}
        try:
            with self.db.get_connection() as conn:
                rows = conn.execute("SELECT * FROM positions").fetchall()
                for row in rows:
                    positions[row['symbol']] = {
                        'quantity': row['quantity'],
                        'buy_price': row['buy_price'],
                        'buy_date': datetime.datetime.strptime(row['buy_date'], '%Y-%m-%d').date(),
                        'cost_basis': row['cost_basis'],
                        'commission': row['commission']
                    }
        except Exception as e:
            self.logger.error(f"Error loading positions: {e}")
        
        return positions
    
    def _load_watchlist_from_db(self) -> List[str]:
        """Load watchlist from database"""
        watchlist = []
        try:
            with self.db.get_connection() as conn:
                rows = conn.execute("SELECT symbol FROM watchlist ORDER BY priority DESC").fetchall()
                watchlist = [row['symbol'] for row in rows]
        except Exception as e:
            self.logger.error(f"Error loading watchlist: {e}")
        
        return watchlist
    
    @robust_retry(error_type='data_fetch', max_retries=3)
    def get_market_data_cached(self, symbol: str, days: int = 60) -> Optional[pd.DataFrame]:
        """Get market data with validation and caching - WITH ERROR HANDLING"""
        cache_key = f"{symbol}_{days}"
        
        # Check cache
        if cache_key in self.data_cache:
            cached_data, cache_time = self.data_cache[cache_key]
            if datetime.datetime.now().timestamp() - cache_time < self.cache_expiry:
                return cached_data
        
        # Fetch new data
        try:
            end_date = datetime.datetime.now().strftime('%Y%m%d')
            start_date = (datetime.datetime.now() - timedelta(days=days*2)).strftime('%Y%m%d')
            
            # Debug logging
            self.logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
            
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
            
            if df is None or len(df) < 10:
                self.logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # Rename columns
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'turnover',
                '振幅': 'amplitude',
                '涨跌幅': 'pct_change',
                '涨跌额': 'change',
                '换手率': 'turnover_rate'
            })
            
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            
            # Ensure numeric types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Validate data
            validation_result = self.data_validator.validate_market_data(df, symbol)
            
            # Log validation results
            if validation_result.issues:
                self.logger.warning(f"Data quality issues for {symbol}: {validation_result.issues}")
            
            # Store validation result in database
            with self.db.get_connection() as conn:
                conn.execute("""
                    INSERT INTO data_quality_log 
                    (symbol, check_timestamp, is_valid, quality_score, issues, data_points)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (symbol, validation_result.timestamp, validation_result.is_valid,
                      validation_result.quality_score, json.dumps(validation_result.issues),
                      validation_result.data_points))
                conn.commit()
            
            # Only cache and return if data quality is acceptable
            if validation_result.quality_score >= self.risk_params['min_data_quality']:
                # Cache the data
                self.data_cache[cache_key] = (df, datetime.datetime.now().timestamp())
                return df
            else:
                self.logger.error(f"Data quality too low for {symbol}: {validation_result.quality_score:.2f}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            self.logger.debug(traceback.format_exc())
            return None
    
    @robust_retry(error_type='calculation', max_retries=2)
    def calculate_comprehensive_factors(self, df: pd.DataFrame) -> Dict:
        """MINIMAL FIX: Add basic error handling to existing code without complete rewrite"""
        factors = {}
        
        if len(df) < 20:
            self.logger.warning(f"Insufficient data for factor calculation: {len(df)} rows")
            return factors
        
        try:
            # Basic data cleaning - minimal change
            close = df['close'].fillna(method='ffill').fillna(method='bfill')
            high = df['high'].fillna(method='ffill').fillna(method='bfill')
            low = df['low'].fillna(method='bfill').fillna(method='bfill')
            volume = df['volume'].fillna(0)
            
            # Validate basic data quality - minimal addition
            if close.isna().all() or (close <= 0).all():
                self.logger.error("Invalid close price data")
                return factors
            
            # 1. MOMENTUM CALCULATIONS (existing code + minimal safety)
            try:
                # 5-day momentum with bounds checking - MINIMAL CHANGE
                if len(close) >= 6:
                    current = close.iloc[-1]
                    past_5d = close.iloc[-6]
                    if past_5d > 0:
                        momentum_5d = (current / past_5d - 1) * 100
                        factors['momentum_5d'] = np.clip(momentum_5d, -50, 50)  # Just add bounds
                    else:
                        factors['momentum_5d'] = 0
                else:
                    factors['momentum_5d'] = 0
                
                # 20-day momentum with bounds checking - MINIMAL CHANGE
                if len(close) >= 21:
                    current = close.iloc[-1]
                    past_20d = close.iloc[-21]
                    if past_20d > 0:
                        momentum_20d = (current / past_20d - 1) * 100
                        factors['momentum_20d'] = np.clip(momentum_20d, -100, 100)  # Just add bounds
                    else:
                        factors['momentum_20d'] = 0
                else:
                    factors['momentum_20d'] = 0
                    
            except Exception as e:
                self.logger.error(f"Momentum calculation error: {e}")
                factors['momentum_5d'] = 0
                factors['momentum_20d'] = 0
            
            # 2. RSI CALCULATIONS (keep using pandas_ta - MINIMAL CHANGE)
            try:
                if len(close) >= 14:
                    rsi_14 = ta.rsi(close, length=14)
                    factors['rsi_14'] = rsi_14.iloc[-1] if rsi_14 is not None and len(rsi_14) > 0 and not pd.isna(rsi_14.iloc[-1]) else 50
                else:
                    factors['rsi_14'] = 50
                    
                if len(close) >= 7:
                    rsi_7 = ta.rsi(close, length=7)
                    factors['rsi_7'] = rsi_7.iloc[-1] if rsi_7 is not None and len(rsi_7) > 0 and not pd.isna(rsi_7.iloc[-1]) else 50
                else:
                    factors['rsi_7'] = 50
                    
                if len(close) >= 28:
                    rsi_28 = ta.rsi(close, length=28)
                    factors['rsi_28'] = rsi_28.iloc[-1] if rsi_28 is not None and len(rsi_28) > 0 and not pd.isna(rsi_28.iloc[-1]) else 50
                else:
                    factors['rsi_28'] = 50
                    
            except Exception as e:
                self.logger.error(f"RSI calculation error: {e}")
                factors['rsi_14'] = 50
                factors['rsi_7'] = 50
                factors['rsi_28'] = 50
            
            # 3. MACD CALCULATIONS (keep using pandas_ta - MINIMAL CHANGE)
            try:
                factors['macd'] = 0
                factors['macd_signal'] = 0
                factors['macd_histogram'] = 0
                
                if len(close) >= 26:
                    macd_result = ta.macd(close, fast=12, slow=26, signal=9)
                    if macd_result is not None and not macd_result.empty and len(macd_result.columns) >= 3:
                        if not pd.isna(macd_result.iloc[-1, 0]):
                            factors['macd'] = float(macd_result.iloc[-1, 0])
                        if not pd.isna(macd_result.iloc[-1, 1]):
                            factors['macd_signal'] = float(macd_result.iloc[-1, 1])
                        if not pd.isna(macd_result.iloc[-1, 2]):
                            factors['macd_histogram'] = float(macd_result.iloc[-1, 2])
                            
            except Exception as e:
                self.logger.error(f"MACD calculation error: {e}")
                factors['macd'] = 0
                factors['macd_signal'] = 0
                factors['macd_histogram'] = 0
            
            # 4. BOLLINGER BANDS (keep using pandas_ta - MINIMAL CHANGE)
            try:
                factors['bb_position'] = 0.5
                factors['bb_width'] = 0.02
                
                if len(close) >= 20:
                    bbands = ta.bbands(close, length=20, std=2)
                    if bbands is not None and not bbands.empty and len(bbands.columns) >= 3:
                        lower = bbands.iloc[-1, 0]
                        middle = bbands.iloc[-1, 1]
                        upper = bbands.iloc[-1, 2]
                        current = close.iloc[-1]
                        
                        if not any(pd.isna([lower, middle, upper, current])) and upper > lower and middle > 0:
                            factors['bb_position'] = float((current - lower) / (upper - lower))
                            factors['bb_width'] = float((upper - lower) / middle)
                            
            except Exception as e:
                self.logger.error(f"Bollinger Bands calculation error: {e}")
                factors['bb_position'] = 0.5
                factors['bb_width'] = 0.02
            
            # 5. VOLUME INDICATORS (keep existing logic - MINIMAL CHANGE)
            try:
                if len(volume) >= 20:
                    # OBV (On-Balance Volume) - keep using pandas_ta
                    obv = ta.obv(close, volume)
                    if obv is not None and len(obv) >= 10:
                        obv_momentum = (obv.iloc[-1] / obv.iloc[-11] - 1) if obv.iloc[-11] != 0 else 0
                        factors['obv_momentum'] = float(np.clip(obv_momentum, -0.5, 0.5))  # Add bounds
                    else:
                        factors['obv_momentum'] = 0
                    
                    # Volume ratio
                    avg_volume = volume.iloc[-20:].mean()
                    current_volume = volume.iloc[-1]
                    factors['volume_ratio'] = float(current_volume / avg_volume) if avg_volume > 0 else 1.0
                    
                    # Volume trend
                    recent_vol = volume.iloc[-5:].mean()
                    older_vol = volume.iloc[-20:-5].mean()
                    factors['volume_trend'] = float((recent_vol / older_vol - 1)) if older_vol > 0 else 0
                else:
                    factors['obv_momentum'] = 0
                    factors['volume_ratio'] = 1.0
                    factors['volume_trend'] = 0
                    
            except Exception as e:
                self.logger.error(f"Volume indicators calculation error: {e}")
                factors['obv_momentum'] = 0
                factors['volume_ratio'] = 1.0
                factors['volume_trend'] = 0
            
            # 6. VOLATILITY CALCULATIONS (existing code + safety)
            try:
                if len(close) >= 20:
                    returns = close.pct_change().dropna()
                    if len(returns) >= 20:
                        vol_20d = returns.tail(20).std() * np.sqrt(252)
                        factors['volatility_20d'] = float(vol_20d) if not pd.isna(vol_20d) else 0.2
                    else:
                        factors['volatility_20d'] = 0.2
                else:
                    factors['volatility_20d'] = 0.2
                    
            except Exception as e:
                self.logger.error(f"Volatility calculation error: {e}")
                factors['volatility_20d'] = 0.2
            
            # 7. TREND STRENGTH INDICATORS (keep using pandas_ta where possible)
            try:
                # ADX and DMI
                if len(df) >= 14:
                    adx_result = ta.adx(high, low, close, length=14)
                    if adx_result is not None and not adx_result.empty:
                        if 'ADX_14' in adx_result.columns:
                            factors['adx'] = float(adx_result['ADX_14'].iloc[-1]) if not pd.isna(adx_result['ADX_14'].iloc[-1]) else 25
                        if 'DMP_14' in adx_result.columns:
                            factors['dmp'] = float(adx_result['DMP_14'].iloc[-1]) if not pd.isna(adx_result['DMP_14'].iloc[-1]) else 25
                        if 'DMN_14' in adx_result.columns:
                            factors['dmn'] = float(adx_result['DMN_14'].iloc[-1]) if not pd.isna(adx_result['DMN_14'].iloc[-1]) else 25
                    else:
                        factors['adx'] = 25
                        factors['dmp'] = 25
                        factors['dmn'] = 25
                else:
                    factors['adx'] = 25
                    factors['dmp'] = 25
                    factors['dmn'] = 25
                    
            except Exception as e:
                self.logger.error(f"ADX calculation error: {e}")
                factors['adx'] = 25
                factors['dmp'] = 25
                factors['dmn'] = 25
            
            # 8. MONEY FLOW INDEX (keep using pandas_ta)
            try:
                if len(df) >= 14:
                    mfi = ta.mfi(high, low, close, volume, length=14)
                    factors['mfi_14'] = float(mfi.iloc[-1]) if mfi is not None and not pd.isna(mfi.iloc[-1]) else 50
                else:
                    factors['mfi_14'] = 50
                    
            except Exception as e:
                self.logger.error(f"MFI calculation error: {e}")
                factors['mfi_14'] = 50
            
            # 9. PRICE PATTERNS (existing code + safety)
            try:
                current_high = high.iloc[-1]
                current_low = low.iloc[-1]
                current_close = close.iloc[-1]
                
                if current_high > current_low and current_close > 0:
                    factors['high_low_ratio'] = float((current_high - current_low) / current_close)
                    factors['close_to_high'] = float((current_close - current_low) / (current_high - current_low))
                else:
                    factors['high_low_ratio'] = 0
                    factors['close_to_high'] = 0.5
                    
                # Support/Resistance levels
                if len(high) >= 20 and len(low) >= 20:
                    high_20 = high.iloc[-20:].max()
                    low_20 = low.iloc[-20:].min()
                    
                    if high_20 > 0 and low_20 > 0:
                        factors['distance_from_high_20d'] = float((current_close / high_20 - 1))
                        factors['distance_from_low_20d'] = float((current_close / low_20 - 1))
                    else:
                        factors['distance_from_high_20d'] = 0
                        factors['distance_from_low_20d'] = 0
                else:
                    factors['distance_from_high_20d'] = 0
                    factors['distance_from_low_20d'] = 0
                    
            except Exception as e:
                self.logger.error(f"Price patterns calculation error: {e}")
                factors['high_low_ratio'] = 0
                factors['close_to_high'] = 0.5
                factors['distance_from_high_20d'] = 0
                factors['distance_from_low_20d'] = 0
            
            # MINIMAL ADDITION: Final validation to prevent NaN/infinite values
            for key, value in factors.items():
                if pd.isna(value) or not np.isfinite(value):
                    if 'rsi' in key or 'mfi' in key or 'adx' in key or 'dmp' in key or 'dmn' in key:
                        factors[key] = 50.0  # Default for percentage indicators
                    elif 'bb_position' in key or 'close_to_high' in key:
                        factors[key] = 0.5   # Default for ratio indicators
                    elif 'bb_width' in key:
                        factors[key] = 0.02  # Default for width
                    elif 'volatility' in key:
                        factors[key] = 0.2   # Default volatility
                    elif 'volume_ratio' in key:
                        factors[key] = 1.0   # Default volume ratio
                    else:
                        factors[key] = 0.0   # Default for other indicators
            
            self.logger.info(f"Calculated {len(factors)} technical factors successfully")
            return factors
            
        except Exception as e:
            self.logger.error(f"Critical error in factor calculation: {e}")
            import traceback
            traceback.print_exc()
            return {}

    
    def get_ml_prediction(self, symbol: str) -> Dict:
        """Get ML prediction for a symbol - WITH IMPROVED ML ENGINE"""
        try:
            # Get data
            df = self.get_market_data_cached(symbol, days=90)
            if df is None or len(df) < 60:
                return {
                    'ml_signal': 0,
                    'ml_confidence': 0,
                    'prediction_quality': 'insufficient_data',
                    'feature_coverage': 0
                }
            
            # Calculate technical indicators
            factors = self.calculate_comprehensive_factors(df)
            
            if not factors:
                return {
                    'ml_signal': 0,
                    'ml_confidence': 0,
                    'prediction_quality': 'no_factors',
                    'feature_coverage': 0
                }
            
            # Get ML prediction using improved engine
            ml_result = self.ml_engine.predict(factors)
            
            return ml_result
            
        except Exception as e:
            self.logger.error(f"ML prediction failed for {symbol}: {e}")
            return {
                'ml_signal': 0,
                'ml_confidence': 0,
                'prediction_quality': 'error',
                'feature_coverage': 0
            }
    
    @robust_retry(error_type='calculation', max_retries=2)
    def generate_enhanced_signals(self, symbol: str, aggressive: bool = False) -> Dict:
        """FIXED: Improved signal generation with relaxed ML requirements and better thresholds"""
        
        # Check failsafe first
        system_ok, issues = self.failsafe.check_all_systems()
        if not system_ok:
            return {
                'signal': 'SYSTEM_ERROR',
                'strength': 0,
                'reasons': issues,
                'factors': {},
                'data_quality': 0
            }
        
        df = self.get_market_data_cached(symbol, days=90)
        
        if df is None or len(df) < 30:
            return {'signal': 'ERROR', 'strength': 0, 'reasons': ['Insufficient data'], 'data_quality': 0}
        
        # Get data quality score
        validation_result = self.data_validator.validate_market_data(df, symbol)
        
        # Calculate comprehensive factors
        factors = self.calculate_comprehensive_factors(df)
        
        # Get ML prediction
        ml_pred = self.get_ml_prediction(symbol)
        
        # Initialize detailed scoring
        buy_signals = []
        sell_signals = []
        
        # 1. FIXED: ML Signal with relaxed requirements
        if ml_pred['ml_confidence'] > 0.15 and ml_pred['prediction_quality'] == 'trained':  # REDUCED from 0.4 to 0.15
            ml_weight = 1.0  # REDUCED from 1.5 to prevent over-weighting
            # FIXED: Use ML signal with lower thresholds
            if ml_pred['ml_signal'] > 0.005:  # REDUCED from 0.02
                buy_signals.append({
                    'factor': 'ML',
                    'score': ml_weight * max(ml_pred['ml_confidence'], 0.25),  # MINIMUM 0.25 weight
                    'reason': f"ML bullish ({ml_pred['ml_signal']:.3f}, conf: {ml_pred['ml_confidence']:.1%})"
                })
            elif ml_pred['ml_signal'] < -0.005:  # REDUCED from -0.02
                sell_signals.append({
                    'factor': 'ML',
                    'score': ml_weight * max(ml_pred['ml_confidence'], 0.25),  # MINIMUM 0.25 weight
                    'reason': f"ML bearish ({ml_pred['ml_signal']:.3f}, conf: {ml_pred['ml_confidence']:.1%})"
                })
        
        # 2. Trend Analysis (keep existing)
        if 'adx' in factors and factors['adx'] > 25:
            if 'dmp' in factors and 'dmn' in factors:
                trend_weight = 2.5
                if factors['dmp'] > factors['dmn'] * 1.15:  # REDUCED from 1.2
                    buy_signals.append({
                        'factor': 'Trend',
                        'score': trend_weight,
                        'reason': f"Strong uptrend (ADX: {factors['adx']:.1f}, +DI: {factors['dmp']:.1f})"
                    })
                elif factors['dmn'] > factors['dmp'] * 1.15:  # REDUCED from 1.2
                    sell_signals.append({
                        'factor': 'Trend',
                        'score': trend_weight,
                        'reason': f"Strong downtrend (ADX: {factors['adx']:.1f}, -DI: {factors['dmn']:.1f})"
                    })
        
        # 3. Momentum with confirmation (RELAXED requirements)
        if 'momentum_20d' in factors and 'momentum_5d' in factors:
            # FIXED: Relax momentum requirements
            if factors['momentum_20d'] > 3 and factors['momentum_5d'] > -1:  # REDUCED from 5 and 0
                momentum_score = 1.5 * (1 + min(factors['momentum_20d'] / 20, 1))
                buy_signals.append({
                    'factor': 'Momentum',
                    'score': momentum_score,
                    'reason': f"Positive momentum (20d: {factors['momentum_20d']:.1f}%, 5d: {factors['momentum_5d']:.1f}%)"
                })
            elif factors['momentum_20d'] < -3 and factors['momentum_5d'] < 1:  # REDUCED from -5 and 0
                momentum_score = 1.5 * (1 + min(abs(factors['momentum_20d']) / 20, 1))
                sell_signals.append({
                    'factor': 'Momentum',
                    'score': momentum_score,
                    'reason': f"Negative momentum (20d: {factors['momentum_20d']:.1f}%, 5d: {factors['momentum_5d']:.1f}%)"
                })
        
        # 4. RSI with multiple timeframes (RELAXED thresholds)
        rsi_confirmed = False
        if 'rsi_14' in factors and 'rsi_7' in factors:
            # FIXED: Relax RSI thresholds
            if factors['rsi_14'] < 40 and factors['rsi_7'] < 45:  # INCREASED from 35 and 40
                buy_signals.append({
                    'factor': 'RSI',
                    'score': 2.0,
                    'reason': f"Oversold (RSI14: {factors['rsi_14']:.1f}, RSI7: {factors['rsi_7']:.1f})"
                })
                rsi_confirmed = True
            elif factors['rsi_14'] > 60 and factors['rsi_7'] > 55:  # REDUCED from 65 and 60
                sell_signals.append({
                    'factor': 'RSI',
                    'score': 2.0,
                    'reason': f"Overbought (RSI14: {factors['rsi_14']:.1f}, RSI7: {factors['rsi_7']:.1f})"
                })
                rsi_confirmed = True
        
        # 5. MACD with strength check (keep existing)
        if all(k in factors for k in ['macd', 'macd_signal', 'macd_histogram']):
            hist_strength = abs(factors['macd_histogram'])
            if factors['macd'] > factors['macd_signal'] and factors['macd_histogram'] > 0:
                macd_score = 1.0 + min(hist_strength * 10, 0.5)
                buy_signals.append({
                    'factor': 'MACD',
                    'score': macd_score,
                    'reason': f"MACD bullish crossover (histogram: {factors['macd_histogram']:.3f})"
                })
            elif factors['macd'] < factors['macd_signal'] and factors['macd_histogram'] < 0:
                macd_score = 1.0 + min(hist_strength * 10, 0.5)
                sell_signals.append({
                    'factor': 'MACD',
                    'score': macd_score,
                    'reason': f"MACD bearish crossover (histogram: {factors['macd_histogram']:.3f})"
                })
        
        # 6. Bollinger Bands with volatility consideration (RELAXED thresholds)
        if 'bb_position' in factors and 'bb_width' in factors:
            if factors['bb_position'] < 0.25 and factors['bb_width'] > 0.015:  # RELAXED from 0.2 and 0.02
                bb_score = 1.5 if rsi_confirmed else 1.0
                buy_signals.append({
                    'factor': 'BB',
                    'score': bb_score,
                    'reason': f"Near lower BB ({factors['bb_position']:.2f}, width: {factors['bb_width']:.2f})"
                })
            elif factors['bb_position'] > 0.75 and factors['bb_width'] > 0.015:  # RELAXED from 0.8 and 0.02
                bb_score = 1.5 if rsi_confirmed else 1.0
                sell_signals.append({
                    'factor': 'BB',
                    'score': bb_score,
                    'reason': f"Near upper BB ({factors['bb_position']:.2f}, width: {factors['bb_width']:.2f})"
                })
        
        # 7. Volume confirmation (RELAXED requirements)
        if 'obv_momentum' in factors and abs(factors['obv_momentum']) > 0.03:  # REDUCED from 0.05
            if factors['obv_momentum'] > 0.05 and factors.get('momentum_5d', 0) > -2:  # RELAXED requirements
                buy_signals.append({
                    'factor': 'Volume',
                    'score': 1.0,
                    'reason': f"Volume accumulation ({factors['obv_momentum']:.1%})"
                })
            elif factors['obv_momentum'] < -0.05 and factors.get('momentum_5d', 0) < 2:  # RELAXED requirements
                sell_signals.append({
                    'factor': 'Volume',
                    'score': 1.0,
                    'reason': f"Volume distribution ({factors['obv_momentum']:.1%})"
                })
        
        # 8. Money Flow Index (RELAXED thresholds)
        if 'mfi_14' in factors:
            if factors['mfi_14'] < 30:  # INCREASED from 25
                buy_signals.append({
                    'factor': 'MFI',
                    'score': 1.5,
                    'reason': f"MFI oversold ({factors['mfi_14']:.1f})"
                })
            elif factors['mfi_14'] > 70:  # REDUCED from 75
                sell_signals.append({
                    'factor': 'MFI',
                    'score': 1.5,
                    'reason': f"MFI overbought ({factors['mfi_14']:.1f})"
                })
        
        # Calculate total scores
        buy_score = sum(s['score'] for s in buy_signals)
        sell_score = sum(s['score'] for s in sell_signals)
        
        # Get top reasons
        all_signals = sorted(buy_signals + sell_signals, key=lambda x: x['score'], reverse=True)
        reasons = [s['reason'] for s in all_signals[:5]]
        
        # Calculate net score
        net_score = buy_score - sell_score
        
        # FIXED: Less penalty for data quality
        if validation_result.quality_score < 0.7:  # REDUCED from 0.8
            penalty = max(0.8, validation_result.quality_score)  # LESS penalty
            net_score *= penalty
            reasons.append(f"Adjusted for data quality: {validation_result.quality_score:.1%}")
        
        # FIXED: Less volatility penalty
        volatility = factors.get('volatility_20d', 0.3)
        if volatility > 0.6:  # INCREASED from 0.5
            net_score *= 0.8  # LESS penalty (was 0.6)
            reasons.append(f"High volatility penalty ({volatility:.1%})")
        elif volatility > 0.45:  # INCREASED from 0.4
            net_score *= 0.9  # LESS penalty (was 0.8)
        
        # GET ADAPTIVE THRESHOLDS
        regime = self.market_regime.get('regime', 'unknown') if self.market_regime else 'unknown'
        performance = {'volatility': volatility}
        
        thresholds = self.threshold_optimizer.optimize_thresholds(regime, performance)
        
        # FIXED: Apply aggressive mode more aggressively
        if aggressive:
            buy_threshold = thresholds['buy_threshold'] * 0.6      # MORE aggressive (was 0.7)
            strong_buy_threshold = thresholds['strong_buy_threshold'] * 0.6  # MORE aggressive (was 0.7)
            sell_threshold = thresholds['sell_threshold'] * 0.6    # MORE aggressive (was 0.7)
            strong_sell_threshold = thresholds['strong_sell_threshold'] * 0.6 # MORE aggressive (was 0.7)
            min_signals = max(1, thresholds['min_signals'] - 1)    # REDUCED requirements
        else:
            buy_threshold = thresholds['buy_threshold']
            strong_buy_threshold = thresholds['strong_buy_threshold']
            sell_threshold = thresholds['sell_threshold']
            strong_sell_threshold = thresholds['strong_sell_threshold']
            min_signals = thresholds['min_signals']
        
        # FIXED: More lenient signal classification
        min_confirmations = min_signals
        if abs(net_score) >= max(buy_threshold, abs(sell_threshold)) * 0.7:  # 70% of threshold
            min_confirmations = max(1, min_signals - 1)
        
        # Determine signal with FIXED logic
        if net_score >= strong_buy_threshold and len(buy_signals) >= min_confirmations:
            signal = 'STRONG_BUY'
        elif net_score >= buy_threshold and len(buy_signals) >= min_confirmations:
            signal = 'BUY'
        elif net_score <= strong_sell_threshold and len(sell_signals) >= min_confirmations:
            signal = 'STRONG_SELL'
        elif net_score <= sell_threshold and len(sell_signals) >= min_confirmations:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        return {
            'signal': signal,
            'strength': abs(net_score),
            'buy_score': buy_score,
            'sell_score': sell_score,
            'net_score': net_score,
            'reasons': reasons,
            'factors': factors,
            'volatility': volatility,
            'ml_signal': ml_pred['ml_signal'],
            'ml_confidence': ml_pred['ml_confidence'],
            'ml_quality': ml_pred['prediction_quality'],
            'data_quality': validation_result.quality_score,
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'thresholds_used': thresholds
        }
    
    @robust_retry(error_type='execution', critical=True)
    def add_position(self, symbol: str, quantity: int, buy_price: float, 
                     buy_date: Optional[datetime.date] = None) -> bool:
        """Add position with validation and database storage - WITH ERROR HANDLING"""
        if buy_date is None:
            buy_date = self.today
        
        # Calculate position value
        position_value = quantity * buy_price
        commission = position_value * 0.0008
        
        # Get current portfolio value for validation
        portfolio_value = self._calculate_portfolio_value()
        
        # Validate trade
        valid, reason = self.failsafe.validate_trade(
            symbol, 'BUY', quantity, buy_price, portfolio_value
        )
        
        if not valid:
            self.logger.warning(f"Trade validation failed: {reason}")
            print(f"{Fore.RED}Trade rejected: {reason}{Style.RESET_ALL}")
            return False
        
        # Add to database
        with self.db.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO positions 
                (symbol, quantity, buy_price, buy_date, cost_basis, commission)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (symbol, quantity, buy_price, buy_date.strftime('%Y-%m-%d'),
                  position_value, commission))
            
            # Record in trade history
            conn.execute("""
                INSERT INTO trade_history 
                (symbol, action, quantity, price, commission, trade_date)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (symbol, 'BUY', quantity, buy_price, commission, buy_date.strftime('%Y-%m-%d')))
            
            conn.commit()
        
        # Update in-memory positions
        self.positions[symbol] = {
            'quantity': quantity,
            'buy_price': buy_price,
            'buy_date': buy_date,
            'cost_basis': position_value,
            'commission': commission
        }
        
        # Record trade in failsafe
        self.failsafe.record_trade()
        
        # Remove from watchlist if present
        if symbol in self.watchlist:
            self.remove_from_watchlist(symbol)
        
        self.logger.info(f"Added position: {symbol} - {quantity} shares @ {buy_price}")
        print(f"{Fore.GREEN}Position added successfully{Style.RESET_ALL}")
        
        return True
    
    @robust_retry(error_type='execution', critical=True)
    def remove_position(self, symbol: str, sell_price: float) -> bool:
        """Remove position with proper recording - WITH ERROR HANDLING"""
        if symbol not in self.positions:
            self.logger.error(f"No position found for {symbol}")
            return False
        
        pos = self.positions[symbol]
        
        # Calculate P&L
        revenue = pos['quantity'] * sell_price
        commission = revenue * 0.0008  # 0.08% commission
        stamp_duty = revenue * 0.001   # 0.1% stamp duty
        
        net_profit = revenue - pos['cost_basis'] - pos['commission'] - commission - stamp_duty
        return_pct = net_profit / pos['cost_basis'] * 100
        
        # Validate trade
        portfolio_value = self._calculate_portfolio_value()
        valid, reason = self.failsafe.validate_trade(
            symbol, 'SELL', pos['quantity'], sell_price, portfolio_value
        )
        
        if not valid:
            self.logger.warning(f"Trade validation failed: {reason}")
            print(f"{Fore.RED}Trade rejected: {reason}{Style.RESET_ALL}")
            return False
        
        # Record in database
        with self.db.get_connection() as conn:
            # Remove from positions
            conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
            
            # Record in trade history
            conn.execute("""
                INSERT INTO trade_history 
                (symbol, action, quantity, price, commission, stamp_duty, pnl, return_pct, trade_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (symbol, 'SELL', pos['quantity'], sell_price, commission, stamp_duty,
                  net_profit, return_pct, self.today.strftime('%Y-%m-%d')))
            
            conn.commit()
        
        # Record trade in failsafe
        self.failsafe.record_trade(net_profit)
        
        # Remove from in-memory positions
        del self.positions[symbol]
        
        self.logger.info(
            f"Sold {symbol}: {pos['quantity']} shares @ {sell_price}\n"
            f"  Net profit: {net_profit:.2f} ({return_pct:.2f}%)"
        )
        
        # Display results
        profit_color = Fore.GREEN if net_profit > 0 else Fore.RED
        print(f"\n{Fore.CYAN}=== TRADE EXECUTED ==={Style.RESET_ALL}")
        print(f"Symbol: {symbol}")
        print(f"Quantity: {pos['quantity']:,}")
        print(f"Buy price: {pos['buy_price']:.2f}")
        print(f"Sell price: {sell_price:.2f}")
        print(f"Revenue: {revenue:,.2f}")
        print(f"Total fees: {pos['commission'] + commission + stamp_duty:,.2f}")
        print(f"Net profit: {profit_color}{net_profit:+,.2f} ({return_pct:+.2f}%){Style.RESET_ALL}")
        
        return True
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        total_value = 0
        for symbol, pos in self.positions.items():
            df = self.get_market_data_cached(symbol)
            if df is not None:
                current_price = df['close'].iloc[-1]
                total_value += pos['quantity'] * current_price
        
        return total_value if total_value > 0 else 1000000  # Default 1M if no positions
    
    def add_to_watchlist(self, symbol: str, priority: int = 0, notes: str = "") -> bool:
        """Add symbol to watchlist with priority"""
        try:
            with self.db.get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO watchlist (symbol, added_date, priority, notes)
                    VALUES (?, ?, ?, ?)
                """, (symbol, self.today.strftime('%Y-%m-%d'), priority, notes))
                conn.commit()
            
            if symbol not in self.watchlist:
                self.watchlist.append(symbol)
            
            self.logger.info(f"Added {symbol} to watchlist")
            print(f"{Fore.GREEN}{symbol} added to watchlist{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding to watchlist: {e}")
            return False
    
    def remove_from_watchlist(self, symbol: str) -> bool:
        """Remove symbol from watchlist"""
        try:
            with self.db.get_connection() as conn:
                conn.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol,))
                conn.commit()
            
            if symbol in self.watchlist:
                self.watchlist.remove(symbol)
            
            self.logger.info(f"Removed {symbol} from watchlist")
            print(f"{Fore.YELLOW}{symbol} removed from watchlist{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing from watchlist: {e}")
            return False
    
    @robust_retry(error_type='calculation', max_retries=2)
    def detect_market_regime(self) -> Dict:
        """Detect current market regime with error handling"""
        # Get Shanghai Composite Index data
        market_df = ak.stock_zh_index_daily(symbol="sh000001")
        
        if market_df is None or len(market_df) < 60:
            return {'regime': 'unknown', 'confidence': 0, 'recommendation': 'Unable to determine regime'}
        
        # Validate market data
        validation = self.data_validator.validate_market_data(market_df, "sh000001")
        if validation.quality_score < 0.5:
            return {'regime': 'unknown', 'confidence': 0, 'recommendation': 'Poor market data quality'}
        
        # Rename columns
        market_df = market_df.rename(columns={
            'date': 'date',
            'close': 'close',
            'volume': 'volume'
        })
        
        # Calculate regime indicators
        returns = market_df['close'].pct_change()
        volatility = returns.rolling(20).std() * np.sqrt(252)
        trend_ma50 = market_df['close'].rolling(50).mean()
        trend_ma200 = market_df['close'].rolling(200).mean()
        
        current_price = market_df['close'].iloc[-1]
        current_vol = volatility.iloc[-1]
        vol_percentile = (volatility < current_vol).sum() / len(volatility)
        
        # Trend strength
        trend_strength_50 = (current_price - trend_ma50.iloc[-1]) / trend_ma50.iloc[-1] if len(trend_ma50) > 0 else 0
        trend_strength_200 = (current_price - trend_ma200.iloc[-1]) / trend_ma200.iloc[-1] if len(trend_ma200) > 200 else 0
        
        # Market breadth (simplified)
        advances = (returns > 0).rolling(20).sum().iloc[-1]
        declines = (returns < 0).rolling(20).sum().iloc[-1]
        breadth = advances / (advances + declines) if (advances + declines) > 0 else 0.5
        
        # Classify regime
        if vol_percentile > 0.8:
            if trend_strength_50 > 0.02:
                regime = 'volatile_bull'
            else:
                regime = 'crisis'
        else:
            if trend_strength_50 > 0.02 and trend_strength_200 > 0:
                regime = 'steady_bull'
            elif trend_strength_50 < -0.02 and trend_strength_200 < 0:
                regime = 'bear'
            else:
                regime = 'range_bound'
        
        confidence = 0.8 if abs(trend_strength_50) > 0.05 else 0.5
        
        # Adjust confidence based on data quality
        confidence *= validation.quality_score
        
        self.market_regime = {
            'regime': regime,
            'volatility_percentile': vol_percentile,
            'trend_strength_50': trend_strength_50,
            'trend_strength_200': trend_strength_200,
            'market_breadth': breadth,
            'confidence': confidence,
            'recommendation': self._get_regime_recommendation(regime),
            'data_quality': validation.quality_score
        }
        
        return self.market_regime
    
    def _get_regime_recommendation(self, regime: str) -> str:
        """Get trading recommendation based on regime"""
        recommendations = {
            'steady_bull': 'Increase positions, focus on momentum stocks',
            'volatile_bull': 'Reduce position sizes, tighten stops, take profits',
            'range_bound': 'Focus on mean reversion, sell at resistance, buy at support',
            'bear': 'Defensive positioning, quality stocks only, reduce exposure',
            'crisis': 'Minimal positions, preserve capital, wait for clarity',
            'unknown': 'Unable to determine regime, trade cautiously'
        }
        return recommendations.get(regime, 'Unable to determine regime')
    
    def can_sell_today(self, symbol: str) -> bool:
        """Check if position can be sold today (T+1 rule)"""
        if symbol not in self.positions:
            return False
        
        buy_date = self.positions[symbol].get('buy_date')
        if not buy_date:
            return True  # If no buy date recorded, assume can sell
        
        # Can sell if bought before today
        return buy_date < self.today
    
    def check_positions_enhanced(self, show_details: bool = True):
        """Enhanced position checking with all fixes"""
        print(f"\n{Fore.CYAN}=== ENHANCED POSITION CHECK ==={Style.RESET_ALL}")
        print(f"Date: {self.today}")
        print(f"Total positions: {len(self.positions)}")
        
        # Check system status
        system_ok, issues = self.failsafe.check_all_systems()
        if not system_ok:
            print(f"{Fore.RED}SYSTEM ISSUES: {', '.join(issues)}{Style.RESET_ALL}")
        
        # Get market regime
        regime = self.detect_market_regime()
        regime_color = Fore.GREEN if regime['regime'] in ['steady_bull'] else Fore.YELLOW if regime['regime'] in ['range_bound', 'volatile_bull'] else Fore.RED
        print(f"Market Regime: {regime_color}{regime['regime'].upper()}{Style.RESET_ALL} - {regime['recommendation']}\n")
        
        if not self.positions:
            print("No positions to display")
            return
        
        total_value = 0
        total_pnl = 0
        position_details = []
        
        for symbol, pos in self.positions.items():
            df = self.get_market_data_cached(symbol)
            
            if df is None:
                print(f"{Fore.RED}Failed to get data for {symbol}{Style.RESET_ALL}")
                continue
            
            current_price = df['close'].iloc[-1]
            current_value = pos['quantity'] * current_price
            pnl = current_value - pos['cost_basis']
            pnl_pct = pnl / pos['cost_basis'] * 100
            
            total_value += current_value
            total_pnl += pnl
            
            # Get enhanced signals
            signals = self.generate_enhanced_signals(symbol)
            
            # Check if can sell today (T+1)
            can_sell = self.can_sell_today(symbol)
            
            position_details.append({
                'symbol': symbol,
                'quantity': pos['quantity'],
                'entry_price': pos['buy_price'],
                'current_price': current_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'value': current_value,
                'can_sell': can_sell,
                'signal': signals['signal'],
                'signal_strength': signals['strength'],
                'data_quality': signals.get('data_quality', 0)
            })
            
            if show_details:
                # Display enhanced position info
                print(f"{Fore.YELLOW}{symbol}{Style.RESET_ALL}")
                print(f"  Position: {pos['quantity']:,} @ {pos['buy_price']:.2f}")
                print(f"  Current: {current_price:.2f}")
                print(f"  Value: {current_value:,.0f}")
                
                # P&L with color
                pnl_color = Fore.GREEN if pnl >= 0 else Fore.RED
                print(f"  P&L: {pnl_color}{pnl:+,.0f} ({pnl_pct:+.2f}%){Style.RESET_ALL}")
                
                # Data quality
                quality_color = Fore.GREEN if signals['data_quality'] >= 0.8 else Fore.YELLOW if signals['data_quality'] >= 0.6 else Fore.RED
                print(f"  Data quality: {quality_color}{signals['data_quality']:.1%}{Style.RESET_ALL}")
                
                # T+1 status
                if can_sell:
                    print(f"  Status: {Fore.GREEN}Can sell today{Style.RESET_ALL}")
                else:
                    print(f"  Status: {Fore.YELLOW}Cannot sell until tomorrow (T+1){Style.RESET_ALL}")
                
                # Signal
                signal_color = Fore.GREEN if 'BUY' in signals['signal'] else Fore.RED if 'SELL' in signals['signal'] else Fore.YELLOW
                print(f"  Signal: {signal_color}{signals['signal']}{Style.RESET_ALL} "
                      f"(strength: {signals['strength']:.1f})")
                
                # Adaptive thresholds used
                if 'thresholds_used' in signals:
                    print(f"  Thresholds: Buy={signals['thresholds_used']['buy_threshold']:.1f}, "
                          f"Sell={signals['thresholds_used']['sell_threshold']:.1f}")
                
                print()
        
        # Portfolio Summary
        print(f"{Fore.CYAN}=== PORTFOLIO SUMMARY ==={Style.RESET_ALL}")
        print(f"Total value: {total_value:,.0f}")
        total_pnl_color = Fore.GREEN if total_pnl >= 0 else Fore.RED
        print(f"Total P&L: {total_pnl_color}{total_pnl:+,.0f} ({total_pnl/total_value*100 if total_value > 0 else 0:+.1f}%){Style.RESET_ALL}")
    
    def analyze_sector_rotation(self) -> Dict:
        """Analyze sector rotation opportunities"""
        try:
            # Define sector representatives (using actual A-share sector indices/ETFs)
            sectors = {
                'technology': '159939',  # Info Tech ETF
                'finance': '512800',     # Banking ETF
                'consumer': '159928',    # Consumer ETF
                'healthcare': '159929',  # Healthcare ETF
                'industrial': '159945',  # Industrial ETF
                'energy': '159945',      # Energy sector
                'materials': '159944'    # Materials ETF
            }
            
            sector_analysis = {}
            
            for sector, symbol in sectors.items():
                df = self.get_market_data_cached(symbol, days=60)
                if df is not None and len(df) > 20:
                    # Validate data quality
                    validation = self.data_validator.validate_market_data(df, symbol)
                    if validation.quality_score < 0.6:
                        continue
                    
                    close = df['close']
                    
                    # Calculate metrics
                    momentum_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) > 5 else 0
                    momentum_20d = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) > 20 else 0
                    
                    # RSI
                    rsi = ta.rsi(close, length=14).iloc[-1] if len(close) >= 14 else 50
                    
                    # Relative strength vs market
                    avg_momentum = momentum_20d  # Would compare vs index in production
                    
                    sector_analysis[sector] = {
                        'symbol': symbol,
                        'momentum_5d': momentum_5d,
                        'momentum_20d': momentum_20d,
                        'rsi': rsi,
                        'relative_strength': avg_momentum,
                        'trend': 'BULLISH' if momentum_20d > 2 and rsi < 70 else 'BEARISH' if momentum_20d < -2 else 'NEUTRAL',
                        'data_quality': validation.quality_score
                    }
            
            # Rank sectors
            if sector_analysis:
                ranked_sectors = sorted(sector_analysis.items(), 
                                      key=lambda x: x[1]['momentum_20d'], 
                                      reverse=True)
                
                self.sector_data = {
                    'analysis': sector_analysis,
                    'rankings': ranked_sectors,
                    'top_sectors': [s[0] for s in ranked_sectors[:3]],
                    'bottom_sectors': [s[0] for s in ranked_sectors[-3:]],
                    'rotation_signal': self._generate_rotation_signal(sector_analysis),
                    'last_update': datetime.datetime.now()
                }
            else:
                self.sector_data = {'analysis': {}, 'rankings': [], 'top_sectors': [], 'bottom_sectors': []}
            
            return self.sector_data
            
        except Exception as e:
            self.logger.error(f"Error analyzing sectors: {e}")
            return {'analysis': {}, 'error': str(e)}
    
    def _generate_rotation_signal(self, sector_analysis: Dict) -> str:
        """Generate sector rotation signal"""
        if not sector_analysis:
            return "Insufficient data"
        
        # Count bullish/bearish sectors
        bullish = sum(1 for s in sector_analysis.values() if s['trend'] == 'BULLISH')
        bearish = sum(1 for s in sector_analysis.values() if s['trend'] == 'BEARISH')
        
        if bullish > len(sector_analysis) * 0.6:
            return "Broad market strength - Consider growth sectors"
        elif bearish > len(sector_analysis) * 0.6:
            return "Broad market weakness - Defensive sectors preferred"
        else:
            return "Mixed signals - Focus on individual stock selection"
    
    def analyze_correlations(self) -> pd.DataFrame:
        """Analyze correlations between positions"""
        if len(self.positions) < 2:
            return pd.DataFrame()
        
        try:
            # Get returns for all positions
            returns_data = {}
            symbols = list(self.positions.keys())
            
            for symbol in symbols:
                df = self.get_market_data_cached(symbol, days=60)
                if df is not None and len(df) > 30:
                    # Check data quality
                    validation = self.data_validator.validate_market_data(df, symbol)
                    if validation.quality_score >= 0.6:
                        returns_data[symbol] = df['close'].pct_change().dropna()
            
            if len(returns_data) < 2:
                return pd.DataFrame()
            
            # Align all series to same dates
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()
            
            if returns_df.empty or len(returns_df) < 20:
                return pd.DataFrame()
            
            # Calculate correlation matrix
            corr_matrix = returns_df.corr()
            
            # Find interesting correlations
            correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    symbol1 = corr_matrix.columns[i]
                    symbol2 = corr_matrix.columns[j]
                    correlation = corr_matrix.iloc[i, j]
                    
                    # Analyze the pair
                    analysis = self._analyze_pair(returns_df[symbol1], returns_df[symbol2], correlation)
                    
                    correlations.append({
                        'pair': f"{symbol1}/{symbol2}",
                        'correlation': correlation,
                        'relationship': analysis['relationship'],
                        'diversification': analysis['diversification'],
                        'risk': analysis['risk']
                    })
            
            return pd.DataFrame(correlations)
            
        except Exception as e:
            self.logger.error(f"Error analyzing correlations: {e}")
            return pd.DataFrame()
    
    def _analyze_pair(self, returns1: pd.Series, returns2: pd.Series, correlation: float) -> Dict:
        """Analyze a pair of assets"""
        if correlation > 0.8:
            relationship = "Highly correlated"
            diversification = "Poor"
            risk = "Concentration risk"
        elif correlation > 0.5:
            relationship = "Moderately correlated"
            diversification = "Limited"
            risk = "Some concentration"
        elif correlation > -0.3:
            relationship = "Low correlation"
            diversification = "Good"
            risk = "Well diversified"
        else:
            relationship = "Negative correlation"
            diversification = "Excellent"
            risk = "Natural hedge"
        
        return {
            'relationship': relationship,
            'diversification': diversification,
            'risk': risk
        }
    
    def calculate_position_size(self, symbol: str, signal_strength: float, 
                              portfolio_value: float, current_price: float,
                              volatility: float) -> int:
        """Calculate position size using Kelly Criterion and risk management"""
        
        # Base position size from risk parameters
        max_position_value = portfolio_value * self.risk_params['max_position_pct']
        
        # Get ML prediction if available
        ml_pred = self.get_ml_prediction(symbol)
        ml_confidence = ml_pred['ml_confidence']
        
        # Get data quality
        df = self.get_market_data_cached(symbol)
        if df is None:
            return 0
        
        validation = self.data_validator.validate_market_data(df, symbol)
        if validation.quality_score < self.risk_params['min_data_quality']:
            return 0
        
        # Combine signal strength with ML confidence
        combined_signal = signal_strength * 0.7 + ml_confidence * 5 * 0.3  # Weight technical 70%, ML 30%
        
        # Adjust for signal strength (0-1 scale)
        signal_multiplier = min(abs(combined_signal) / 5, 1.0)
        
        # Volatility adjustment - reduce size for higher volatility
        volatility_multiplier = np.exp(-volatility * 2)
        volatility_multiplier = np.clip(volatility_multiplier, 0.3, 1.0)
        
        # Market regime adjustment
        regime_multiplier = 1.0
        if self.market_regime:
            regime = self.market_regime.get('regime', 'unknown')
            regime_multipliers = {
                'steady_bull': 1.1,
                'volatile_bull': 0.8,
                'range_bound': 0.9,
                'bear': 0.6,
                'crisis': 0.4,
                'unknown': 0.8
            }
            regime_multiplier = regime_multipliers.get(regime, 0.8)
        
        # Portfolio heat check - reduce size if too many positions
        current_positions = len(self.positions)
        concentration_multiplier = np.exp(-current_positions / 10)
        concentration_multiplier = 1 - concentration_multiplier * 0.3  # Max 30% reduction
        
        # Data quality adjustment
        quality_multiplier = validation.quality_score
        
        # Calculate final position value
        position_value = (max_position_value * signal_multiplier * 
                         volatility_multiplier * regime_multiplier * 
                         concentration_multiplier * quality_multiplier)
        
        # Apply minimum position size
        min_position_value = portfolio_value * 0.01  # 1% minimum
        position_value = max(position_value, min_position_value)
        
        # Validate against failsafe limits
        position_pct = position_value / portfolio_value
        if position_pct > self.failsafe.circuit_breakers['max_position_size']:
            position_value = portfolio_value * self.failsafe.circuit_breakers['max_position_size']
        
        # Convert to shares (round to 100)
        shares = int(position_value / current_price / 100) * 100
        
        # Ensure minimum 100 shares
        shares = max(shares, 100)
        
        self.logger.info(
            f"Position sizing for {symbol}: {shares} shares "
            f"(signal: {signal_multiplier:.2f}, vol: {volatility_multiplier:.2f}, "
            f"regime: {regime_multiplier:.2f}, ML: {ml_confidence:.2f}, "
            f"quality: {quality_multiplier:.2f})"
        )
        
        return shares
    
    def calculate_portfolio_risk_metrics(self) -> Dict:
        """Calculate portfolio-level risk metrics"""
        if not self.positions:
            return {}
        
        try:
            # Collect returns data for all positions
            returns_data = {}
            position_values = {}
            
            for symbol, pos in self.positions.items():
                df = self.get_market_data_cached(symbol)
                if df is not None and len(df) > 20:
                    # Validate data
                    validation = self.data_validator.validate_market_data(df, symbol)
                    if validation.quality_score >= 0.6:
                        returns_data[symbol] = df['close'].pct_change().dropna()
                        current_price = df['close'].iloc[-1]
                        position_values[symbol] = pos['quantity'] * current_price
            
            if not returns_data:
                return {}
            
            # Create portfolio returns
            total_value = sum(position_values.values())
            weights = {symbol: value/total_value for symbol, value in position_values.items()}
            
            # Align returns to same dates
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()
            
            if returns_df.empty:
                return {}
            
            # Calculate portfolio returns
            portfolio_returns = pd.Series(0, index=returns_df.index)
            for symbol, weight in weights.items():
                if symbol in returns_df.columns:
                    portfolio_returns += returns_df[symbol] * weight
            
            # Risk metrics
            metrics = {}
            
            # Volatility
            metrics['portfolio_volatility'] = portfolio_returns.std() * np.sqrt(252)
            
            # VaR (Value at Risk)
            confidence = self.risk_params['confidence_level']
            metrics['var_95'] = np.percentile(portfolio_returns, (1 - confidence) * 100)
            
            # CVaR (Conditional VaR)
            var_threshold = metrics['var_95']
            metrics['cvar_95'] = portfolio_returns[portfolio_returns <= var_threshold].mean()
            
            # Maximum drawdown
            cumulative = (1 + portfolio_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            metrics['max_drawdown'] = drawdown.min()
            metrics['current_drawdown'] = drawdown.iloc[-1]
            
            # Sharpe ratio (assuming 0% risk-free rate)
            if portfolio_returns.std() > 0:
                metrics['sharpe_ratio'] = (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252))
            else:
                metrics['sharpe_ratio'] = 0
            
            # Correlation matrix for diversification check
            if len(returns_df.columns) > 1:
                correlation_matrix = returns_df.corr()
                # Average correlation (excluding diagonal)
                mask = np.ones(correlation_matrix.shape, dtype=bool)
                np.fill_diagonal(mask, False)
                metrics['avg_correlation'] = correlation_matrix.values[mask].mean()
            
            # Calculate portfolio beta (simplified - against market)
            try:
                market_df = ak.stock_zh_index_daily(symbol="sh000001")
                if market_df is not None and len(market_df) > len(portfolio_returns):
                    market_returns = market_df['close'].pct_change().tail(len(portfolio_returns))
                    if len(market_returns) == len(portfolio_returns):
                        covariance = portfolio_returns.cov(market_returns)
                        market_variance = market_returns.var()
                        metrics['portfolio_beta'] = covariance / market_variance if market_variance > 0 else 1.0
            except:
                metrics['portfolio_beta'] = 1.0
            
            # Store in database
            with self.db.get_connection() as conn:
                conn.execute("""
                    UPDATE performance_snapshots 
                    SET volatility = ?, sharpe_ratio = ?, max_drawdown = ?, var_95 = ?
                    WHERE snapshot_date = ?
                """, (metrics['portfolio_volatility'], metrics['sharpe_ratio'], 
                      metrics['max_drawdown'], metrics['var_95'],
                      self.today.strftime('%Y-%m-%d')))
                conn.commit()
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk metrics: {e}")
            return {}
    
    def get_optimal_execution_time(self, symbol: str, action: str) -> Dict:
        """Suggest optimal intraday execution timing"""
        current_time = datetime.datetime.now()
        suggestions = []
        confidence = 0.5
        
        # Analyze historical intraday patterns
        try:
            # Get recent data to analyze patterns
            df = self.get_market_data_cached(symbol, days=30)
            if df is not None and len(df) > 10:
                # Analyze typical daily patterns
                returns = df['close'].pct_change()
                volatility = returns.std()
                
                # Morning vs afternoon performance
                # This is simplified - in reality, you'd need intraday data
                morning_tendency = returns.mean()  # Simplified
                
                if action == 'BUY':
                    if current_time.hour == 9 and current_time.minute < 45:
                        suggestions.append("Wait 15-30 minutes - Opening volatility typically high")
                        confidence = 0.7
                    elif current_time.hour == 10 and current_time.minute < 30:
                        suggestions.append("Good timing - Morning dip often occurs around now")
                        confidence = 0.8
                    elif current_time.hour == 14 and current_time.minute > 30:
                        suggestions.append("Avoid - Late session often sees rally")
                        confidence = 0.6
                    else:
                        suggestions.append("Neutral timing")
                        confidence = 0.5
                        
                else:  # SELL
                    if current_time.hour == 9 and 45 <= current_time.minute <= 59:
                        suggestions.append("Good timing - Opening pop often fades")
                        confidence = 0.7
                    elif current_time.hour == 14 and current_time.minute < 45:
                        suggestions.append("Good timing - Before closing volatility")
                        confidence = 0.8
                    elif current_time.hour == 15:
                        suggestions.append("Too late - Market closing soon")
                        confidence = 0.3
                    else:
                        suggestions.append("Neutral timing")
                        confidence = 0.5
        
        except Exception as e:
            self.logger.error(f"Error analyzing execution timing: {e}")
            suggestions.append("Unable to analyze timing")
        
        return {
            'current_time': current_time.strftime('%H:%M:%S'),
            'action': action,
            'suggestions': suggestions,
            'confidence': confidence,
            'market_hours': self._get_market_status()
        }
    
    def _get_market_status(self) -> str:
        """Get current market status"""
        now = datetime.datetime.now()
        
        # China market hours: 9:30-11:30, 13:00-15:00
        if now.weekday() >= 5:  # Weekend
            return "CLOSED - Weekend"
        elif now.hour < 9 or (now.hour == 9 and now.minute < 30):
            return "PRE-MARKET"
        elif (now.hour == 9 and now.minute >= 30) or (10 <= now.hour < 11) or (now.hour == 11 and now.minute <= 30):
            return "MORNING SESSION"
        elif 11 < now.hour < 13:
            return "LUNCH BREAK"
        elif 13 <= now.hour < 15:
            return "AFTERNOON SESSION"
        else:
            return "AFTER HOURS"
    
    def backtest_strategy(self, symbol: str, start_date: str, end_date: str, 
                        initial_capital: float = 100000, aggressive: bool = False) -> Dict:
        """COMPLETE FIXED: Backtest strategy with error handling and proper execution"""
        print(f"\n{Fore.CYAN}=== BACKTESTING {symbol} ==={Style.RESET_ALL}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Initial capital: {initial_capital:,.0f}")
        if aggressive:
            print(f"Mode: {Fore.YELLOW}AGGRESSIVE (relaxed thresholds){Style.RESET_ALL}")
        
        try:
            # Get historical data
            days_needed = (datetime.datetime.strptime(end_date, '%Y-%m-%d') - 
                        datetime.datetime.strptime(start_date, '%Y-%m-%d')).days + 90
            
            df = self.get_market_data_cached(symbol, days=days_needed)
            
            if df is None or len(df) < 90:
                return {'error': 'Insufficient data for backtesting'}
            
            # Validate data quality
            validation = self.data_validator.validate_market_data(df, symbol)
            if validation.quality_score < 0.6:  # RELAXED from 0.7
                return {'error': f'Data quality too low: {validation.quality_score:.1%}'}
            
            # Check if requested date range exists in data
            print(f"Available data range: {df.index[0]} to {df.index[-1]}")
            
            # Filter to backtest period with error handling
            try:
                df_filtered = df[start_date:end_date]
            except:
                df_filtered = df
            
            if df_filtered.empty:
                return {
                    'error': f'No data available for period {start_date} to {end_date}',
                    'available_range': f"{df.index[0]} to {df.index[-1]}"
                }
            
            df = df_filtered
            print(f"Backtest data range: {df.index[0]} to {df.index[-1]} ({len(df)} days)")
            
            # Initialize backtest
            position = 0
            cash = initial_capital
            trades = []
            portfolio_values = []
            signals_history = []
            
            # CRITICAL: Store original settings and apply aggressive mode globally
            original_thresholds = self.threshold_optimizer.base_thresholds.copy()
            original_adaptive = self.threshold_optimizer.adaptive_thresholds.copy()
            
            if aggressive:
                # Apply aggressive thresholds globally during backtest
                self.threshold_optimizer.base_thresholds = {
                    'buy_threshold': original_thresholds['buy_threshold'] * 0.6,
                    'strong_buy_threshold': original_thresholds['strong_buy_threshold'] * 0.6,
                    'sell_threshold': original_thresholds['sell_threshold'] * 0.6,
                    'strong_sell_threshold': original_thresholds['strong_sell_threshold'] * 0.6,
                    'min_signals': max(1, original_thresholds['min_signals'] - 1)
                }
                self.threshold_optimizer.adaptive_thresholds = self.threshold_optimizer.base_thresholds.copy()
            
            print(f"Using thresholds: BUY={self.threshold_optimizer.base_thresholds['buy_threshold']:.1f}, "
                f"STRONG_BUY={self.threshold_optimizer.base_thresholds['strong_buy_threshold']:.1f}, "
                f"MIN_SIGNALS={self.threshold_optimizer.base_thresholds['min_signals']}")
            
            # Track execution decisions
            buy_attempts = 0
            buy_executed = 0
            sell_attempts = 0
            sell_executed = 0
            
            # Run through each day
            for i in range(60, len(df)):  # Need 60 days of history for indicators
                current_date = df.index[i]
                
                # Get data up to current date (no look-ahead bias)
                historical_data = df[df.index <= current_date].tail(90)
                
                # CRITICAL: Generate signals using CURRENT method with proper cache handling
                cache_key = f"{symbol}_90"
                old_cache = self.data_cache.get(cache_key)
                self.data_cache[cache_key] = (historical_data, datetime.datetime.now().timestamp())
                
                # Generate signals - this uses your FIXED signal generation
                signals = self.generate_enhanced_signals(symbol, aggressive=aggressive)
                
                # Restore cache
                if old_cache:
                    self.data_cache[cache_key] = old_cache
                elif cache_key in self.data_cache:
                    del self.data_cache[cache_key]
                
                current_price = historical_data['close'].iloc[-1]
                
                # Record signal with comprehensive details
                signals_history.append({
                    'date': current_date,
                    'signal': signals['signal'],
                    'strength': signals['strength'],
                    'net_score': signals.get('net_score', 0),
                    'buy_score': signals.get('buy_score', 0),
                    'sell_score': signals.get('sell_score', 0),
                    'ml_confidence': signals.get('ml_confidence', 0),
                    'data_quality': signals.get('data_quality', 0),
                    'buy_signals': signals.get('buy_signals', 0),
                    'sell_signals': signals.get('sell_signals', 0),
                    'reasons': signals.get('reasons', [])[:2],
                    'price': current_price
                })
                
                # EXECUTE BUY TRADES with VERY RELAXED criteria
                if signals['signal'] in ['BUY', 'STRONG_BUY'] and position == 0:
                    buy_attempts += 1
                    
                    # ULTRA-RELAXED buy conditions - almost always execute
                    data_quality_ok = signals.get('data_quality', 0) >= 0.5  # VERY LOW threshold
                    strength_ok = signals.get('strength', 0) >= 0.3  # VERY LOW threshold
                    net_score_ok = signals.get('net_score', 0) >= 1.0  # REASONABLE threshold
                    
                    # Debug output
                    print(f"  {current_date.strftime('%Y-%m-%d')}: {signals['signal']} signal detected")
                    print(f"    Strength: {signals.get('strength', 0):.2f} (need ≥0.3: {strength_ok})")
                    print(f"    Net score: {signals.get('net_score', 0):.2f} (need ≥1.0: {net_score_ok})")
                    print(f"    Data quality: {signals.get('data_quality', 0):.1%} (need ≥50%: {data_quality_ok})")
                    print(f"    ML confidence: {signals.get('ml_confidence', 0):.1%}")
                    print(f"    Buy signals: {signals.get('buy_signals', 0)}")
                    
                    # VERY PERMISSIVE execution criteria
                    should_execute = data_quality_ok and (strength_ok or net_score_ok)
                    
                    if should_execute:
                        # Calculate position size
                        available_cash = cash * 0.95  # Use 95% of cash
                        shares = int(available_cash / current_price / 100) * 100  # Round to 100 shares
                        
                        # Lower minimum position requirements
                        min_shares = 100
                        if shares >= min_shares:
                            position = shares
                            cost = shares * current_price * 1.0008  # Commission
                            cash -= cost
                            
                            trades.append({
                                'date': current_date,
                                'action': 'BUY',
                                'price': current_price,
                                'shares': shares,
                                'value': cost,
                                'signal': signals['signal'],
                                'strength': signals.get('strength', 0),
                                'net_score': signals.get('net_score', 0),
                                'ml_confidence': signals.get('ml_confidence', 0),
                                'reasons': ', '.join(signals.get('reasons', [])[:2])
                            })
                            
                            buy_executed += 1
                            print(f"    ✅ EXECUTED BUY: {shares:,} shares @ {current_price:.2f}")
                            print(f"       Cost: {cost:,.0f}, Cash remaining: {cash:,.0f}")
                            print(f"       Reasons: {', '.join(signals.get('reasons', [])[:2])}")
                        else:
                            print(f"    ❌ INSUFFICIENT CASH: Need {min_shares * current_price:,.0f}, have {available_cash:,.0f}")
                    else:
                        # Detailed rejection reasons
                        reasons = []
                        if not data_quality_ok:
                            reasons.append(f"data_quality={signals.get('data_quality', 0):.1%}<50%")
                        if not strength_ok:
                            reasons.append(f"strength={signals.get('strength', 0):.2f}<0.3")
                        if not net_score_ok:
                            reasons.append(f"net_score={signals.get('net_score', 0):.2f}<1.0")
                        print(f"    ❌ REJECTED: {', '.join(reasons)}")
                
                # EXECUTE SELL TRADES with RELAXED criteria
                elif signals['signal'] in ['SELL', 'STRONG_SELL'] and position > 0:
                    sell_attempts += 1
                    
                    # RELAXED sell conditions
                    data_quality_ok = signals.get('data_quality', 0) >= 0.5  # LOW threshold
                    strength_ok = signals.get('strength', 0) >= 0.3  # LOW threshold
                    net_score_ok = signals.get('net_score', 0) <= -0.5  # REASONABLE threshold
                    
                    print(f"  {current_date.strftime('%Y-%m-%d')}: {signals['signal']} signal detected")
                    print(f"    Strength: {signals.get('strength', 0):.2f} (need ≥0.3: {strength_ok})")
                    print(f"    Net score: {signals.get('net_score', 0):.2f} (need ≤-0.5: {net_score_ok})")
                    print(f"    Data quality: {signals.get('data_quality', 0):.1%} (need ≥50%: {data_quality_ok})")
                    
                    should_execute = data_quality_ok and (strength_ok or net_score_ok)
                    
                    if should_execute:
                        # Sell position
                        proceeds = position * current_price * (1 - 0.0008 - 0.001)  # Commission + stamp duty
                        cash += proceeds
                        
                        # Calculate trade P&L
                        buy_trades = [t for t in trades if t['action'] == 'BUY']
                        if buy_trades:
                            last_buy = buy_trades[-1]
                            buy_cost = last_buy['value']
                            trade_pnl = proceeds - buy_cost
                            trade_return = (proceeds / buy_cost - 1) * 100
                        else:
                            trade_pnl = 0
                            trade_return = 0
                        
                        trades.append({
                            'date': current_date,
                            'action': 'SELL',
                            'price': current_price,
                            'shares': position,
                            'value': proceeds,
                            'signal': signals['signal'],
                            'pnl': trade_pnl,
                            'return': trade_return,
                            'strength': signals.get('strength', 0),
                            'net_score': signals.get('net_score', 0),
                            'reasons': ', '.join(signals.get('reasons', [])[:2])
                        })
                        
                        sell_executed += 1
                        return_color = Fore.GREEN if trade_return > 0 else Fore.RED
                        print(f"    ✅ EXECUTED SELL: {position:,} shares @ {current_price:.2f}")
                        print(f"       Proceeds: {proceeds:,.0f}, P&L: {return_color}{trade_pnl:+,.0f} ({trade_return:+.1f}%){Style.RESET_ALL}")
                        print(f"       Reasons: {', '.join(signals.get('reasons', [])[:2])}")
                        
                        position = 0
                    else:
                        # Detailed rejection reasons
                        reasons = []
                        if not data_quality_ok:
                            reasons.append(f"data_quality={signals.get('data_quality', 0):.1%}<50%")
                        if not strength_ok:
                            reasons.append(f"strength={signals.get('strength', 0):.2f}<0.3")
                        if not net_score_ok:
                            reasons.append(f"net_score={signals.get('net_score', 0):.2f}>-0.5")
                        print(f"    ❌ SELL REJECTED: {', '.join(reasons)}")
                
                # Record portfolio value
                portfolio_value = cash + position * current_price
                portfolio_values.append({
                    'date': current_date,
                    'value': portfolio_value,
                    'cash': cash,
                    'position_value': position * current_price,
                    'position': position
                })
            
            # Restore original thresholds
            self.threshold_optimizer.base_thresholds = original_thresholds
            self.threshold_optimizer.adaptive_thresholds = original_adaptive
            
            # Print execution summary
            print(f"\n{Fore.CYAN}EXECUTION SUMMARY:{Style.RESET_ALL}")
            print(f"Buy signals: {buy_attempts} detected, {buy_executed} executed ({buy_executed/buy_attempts*100 if buy_attempts > 0 else 0:.1f}%)")
            print(f"Sell signals: {sell_attempts} detected, {sell_executed} executed ({sell_executed/sell_attempts*100 if sell_attempts > 0 else 0:.1f}%)")
            
            # FIXED: Handle case where no trading occurred (CRITICAL ERROR FIX)
            if not portfolio_values:
                print(f"\n{Fore.YELLOW}WARNING: No portfolio values recorded - no trading activity{Style.RESET_ALL}")
                return {
                    'error': 'No trading activity during period',
                    'initial_capital': initial_capital,
                    'final_value': initial_capital,
                    'total_return': 0,
                    'buy_hold_return': 0,
                    'excess_return': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'num_trades': 0,
                    'win_rate': 0,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'profit_factor': 0,
                    'best_trade': 0,
                    'worst_trade': 0,
                    'trades': trades,
                    'final_position': 'CASH',
                    'data_quality': validation.quality_score,
                    'signal_summary': {},
                    'total_signals': len(signals_history),
                    'buy_attempts': buy_attempts,
                    'buy_executed': buy_executed,
                    'sell_attempts': sell_attempts,
                    'sell_executed': sell_executed,
                    'diagnostic_info': {
                        'data_length': len(df),
                        'signals_generated': len(signals_history),
                        'date_range_actual': f"{df.index[0]} to {df.index[-1]}" if len(df) > 0 else "No data",
                        'no_signals_reason': 'Market conditions may not have triggered any signals during this period'
                    }
                }
            
            # Calculate metrics normally if portfolio_values exists
            final_value = portfolio_values[-1]['value']
            total_return = (final_value / initial_capital - 1) * 100
            
            # Calculate daily returns with safety check
            values = pd.Series([p['value'] for p in portfolio_values])
            daily_returns = values.pct_change().dropna()
            
            # Sharpe ratio
            sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 and daily_returns.std() > 0 else 0
            
            # Maximum drawdown
            if len(daily_returns) > 0:
                cumulative = (1 + daily_returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = ((cumulative - running_max) / running_max).min()
            else:
                drawdown = 0
            
            # Win rate
            completed_trades = [t for t in trades if t['action'] == 'SELL']
            winning_trades = [t for t in completed_trades if t.get('pnl', 0) > 0]
            win_rate = len(winning_trades) / len(completed_trades) if completed_trades else 0
            
            # Average win/loss
            wins = [t['return'] for t in completed_trades if t.get('pnl', 0) > 0]
            losses = [t['return'] for t in completed_trades if t.get('pnl', 0) < 0]
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            
            # Profit factor
            total_wins = sum(t['pnl'] for t in completed_trades if t.get('pnl', 0) > 0)
            total_losses = abs(sum(t['pnl'] for t in completed_trades if t.get('pnl', 0) < 0))
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
            
            # Buy and hold comparison with safety check
            if len(df) > 60:
                buy_hold_return = (df['close'].iloc[-1] / df['close'].iloc[60] - 1) * 100
            else:
                buy_hold_return = 0
            
            # Enhanced signal analysis
            signal_summary = pd.DataFrame(signals_history) if signals_history else pd.DataFrame()
            
            results = {
                'initial_capital': initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'buy_hold_return': buy_hold_return,
                'excess_return': total_return - buy_hold_return,
                'sharpe_ratio': sharpe,
                'max_drawdown': drawdown,
                'num_trades': len([t for t in trades if t['action'] == 'BUY']),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'best_trade': max(completed_trades, key=lambda x: x.get('return', 0))['return'] if completed_trades else 0,
                'worst_trade': min(completed_trades, key=lambda x: x.get('return', 0))['return'] if completed_trades else 0,
                'trades': trades,
                'final_position': 'CASH' if position == 0 else f"{position} shares",
                'data_quality': validation.quality_score,
                'signal_summary': signal_summary['signal'].value_counts().to_dict() if not signal_summary.empty else {},
                'total_signals': len(signals_history),
                'buy_attempts': buy_attempts,
                'buy_executed': buy_executed,
                'sell_attempts': sell_attempts,
                'sell_executed': sell_executed,
                'diagnostic_info': {
                    'data_length': len(df),
                    'portfolio_values_count': len(portfolio_values),
                    'signals_generated': len(signals_history),
                    'date_range_requested': f"{start_date} to {end_date}",
                    'date_range_actual': f"{df.index[0]} to {df.index[-1]}" if len(df) > 0 else "No data"
                }
            }
            
            # Display results
            self._display_backtest_results(results)
            
            return results
            
        except Exception as e:
            # Restore original thresholds in case of error
            if 'original_thresholds' in locals():
                self.threshold_optimizer.base_thresholds = original_thresholds
                self.threshold_optimizer.adaptive_thresholds = original_adaptive
            
            self.logger.error(f"Backtest error: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

    
    def _display_backtest_results(self, results: Dict):
        """Display backtest results in a formatted way"""
        print(f"\n{Fore.CYAN}=== BACKTEST RESULTS ==={Style.RESET_ALL}")
        
        # Data quality
        if 'data_quality' in results:
            quality_color = Fore.GREEN if results['data_quality'] >= 0.8 else Fore.YELLOW
            print(f"Data quality: {quality_color}{results['data_quality']:.1%}{Style.RESET_ALL}")
        
        # Returns
        print(f"\n{Fore.YELLOW}Returns:{Style.RESET_ALL}")
        return_color = Fore.GREEN if results['total_return'] > 0 else Fore.RED
        print(f"  Strategy return: {return_color}{results['total_return']:.2f}%{Style.RESET_ALL}")
        print(f"  Buy & Hold return: {results['buy_hold_return']:.2f}%")
        
        excess_color = Fore.GREEN if results['excess_return'] > 0 else Fore.RED
        print(f"  Excess return: {excess_color}{results['excess_return']:.2f}%{Style.RESET_ALL}")
        
        # Risk metrics
        print(f"\n{Fore.YELLOW}Risk Metrics:{Style.RESET_ALL}")
        print(f"  Sharpe ratio: {results['sharpe_ratio']:.2f}")
        print(f"  Max drawdown: {results['max_drawdown']:.1%}")
        
        # Trading statistics
        print(f"\n{Fore.YELLOW}Trading Statistics:{Style.RESET_ALL}")
        print(f"  Number of trades: {results['num_trades']}")
        print(f"  Win rate: {results['win_rate']:.1%}")
        print(f"  Average win: {results['avg_win']:.2f}%")
        print(f"  Average loss: {results['avg_loss']:.2f}%")
        print(f"  Profit factor: {results['profit_factor']:.2f}")
        print(f"  Best trade: {results['best_trade']:.2f}%")
        print(f"  Worst trade: {results['worst_trade']:.2f}%")
        
        # Final state
        print(f"\n{Fore.YELLOW}Final State:{Style.RESET_ALL}")
        print(f"  Final value: {results['final_value']:,.0f}")
        print(f"  Final position: {results['final_position']}")
        
        # Signal analysis
        if 'signal_summary' in results and results['signal_summary']:
            print(f"\n{Fore.YELLOW}Signal Analysis:{Style.RESET_ALL}")
            print(f"  Total days analyzed: {results.get('total_signals', 0)}")
            print("  Signal distribution:")
            for signal, count in results['signal_summary'].items():
                print(f"    {signal}: {count} days")
            
            if 'avg_scores' in results and results['avg_scores']:
                print("  Average scores by signal:")
                for signal, scores in results['avg_scores'].items():
                    if isinstance(scores, dict):
                        print(f"    {signal}: buy={scores.get('buy_score', 0):.1f}, "
                              f"sell={scores.get('sell_score', 0):.1f}, "
                              f"net={scores.get('net_score', 0):.1f}")
        
        # Recent trades
        if results['trades']:
            print(f"\n{Fore.YELLOW}Recent Trades:{Style.RESET_ALL}")
            for trade in results['trades'][-5:]:
                action_color = Fore.GREEN if trade['action'] == 'BUY' else Fore.RED
                print(f"  {trade['date'].strftime('%Y-%m-%d')}: {action_color}{trade['action']}{Style.RESET_ALL} "
                      f"{trade['shares']} @ {trade['price']:.2f}")
                if 'return' in trade:
                    return_color = Fore.GREEN if trade['return'] > 0 else Fore.RED
                    print(f"    Return: {return_color}{trade['return']:.1f}%{Style.RESET_ALL}")
    
    def start_monitoring(self):
        """Start real-time monitoring with safety checks"""
        if self.monitoring_active:
            print("Monitoring already active")
            return
        
        # Check system before starting
        system_ok, issues = self.failsafe.check_all_systems()
        if not system_ok:
            print(f"{Fore.RED}Cannot start monitoring - system issues:{Style.RESET_ALL}")
            for issue in issues:
                print(f"  • {issue}")
            return
        
        self.monitoring_active = True
        self.alert_thread = threading.Thread(target=self._monitor_loop)
        self.alert_thread.daemon = True
        self.alert_thread.start()
        
        print(f"{Fore.GREEN}Real-time monitoring started{Style.RESET_ALL}")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring_active = False
        print(f"{Fore.YELLOW}Monitoring stopped{Style.RESET_ALL}")
    
    def _monitor_loop(self):
        """Main monitoring loop with enhanced safety"""
        while self.monitoring_active:
            try:
                # Check if emergency stop is active
                if self.failsafe.emergency_stop:
                    self.logger.warning("Monitoring paused - emergency stop active")
                    time.sleep(300)  # Check every 5 minutes
                    continue
                
                # Only monitor during market hours
                market_status = self._get_market_status()
                if 'SESSION' in market_status:
                    # Check each position
                    for symbol, pos in self.positions.items():
                        self._check_position_alerts(symbol, pos)
                    
                    # Check watchlist
                    for symbol in self.watchlist[:10]:  # Limit to top 10
                        self._check_watchlist_alerts(symbol)
                
                # Sleep based on market status
                sleep_time = 60 if 'SESSION' in market_status else 300
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Monitor error: {e}")
                time.sleep(60)
    
    def _check_position_alerts(self, symbol: str, position: Dict):
        """Check for position-specific alerts with data validation"""
        try:
            df = self.get_market_data_cached(symbol)
            if df is None:
                return
            
            # Validate data quality
            validation = self.data_validator.validate_market_data(df, symbol)
            if validation.quality_score < 0.6:
                self.logger.warning(f"Poor data quality for {symbol}: {validation.quality_score:.1%}")
                return
            
            current_price = df['close'].iloc[-1]
            pnl_pct = (current_price / position['buy_price'] - 1) * 100
            
            # Stop loss alert
            if pnl_pct <= -self.risk_params['stop_loss'] * 100:
                self._create_alert(
                    symbol=symbol,
                    alert_type='STOP_LOSS',
                    message=f"Down {pnl_pct:.1f}% - Stop loss triggered",
                    urgency='high',
                    action_required=True
                )
            
            # Take profit alert
            elif pnl_pct >= self.risk_params['take_profit'] * 100:
                self._create_alert(
                    symbol=symbol,
                    alert_type='TAKE_PROFIT',
                    message=f"Up {pnl_pct:.1f}% - Take profit target reached",
                    urgency='medium',
                    action_required=True
                )
            
            # Signal change alert
            signals = self.generate_enhanced_signals(symbol)
            if signals['signal'] in ['STRONG_SELL'] and self.can_sell_today(symbol):
                self._create_alert(
                    symbol=symbol,
                    alert_type='SELL_SIGNAL',
                    message=f"{signals['signal']} - {signals['reasons'][0]}",
                    urgency='medium',
                    action_required=True
                )
            
        except Exception as e:
            self.logger.error(f"Alert check error for {symbol}: {e}")
    
    def _check_watchlist_alerts(self, symbol: str):
        """Check for watchlist alerts with data validation"""
        try:
            signals = self.generate_enhanced_signals(symbol)
            
            # Only alert on high quality signals
            if (signals['signal'] in ['STRONG_BUY'] and 
                signals['strength'] > 3 and
                signals.get('data_quality', 0) >= 0.7):
                
                self._create_alert(
                    symbol=symbol,
                    alert_type='BUY_OPPORTUNITY',
                    message=f"{signals['signal']} - {signals['reasons'][0]}",
                    urgency='medium',
                    action_required=False
                )
            
        except Exception as e:
            self.logger.error(f"Watchlist alert error for {symbol}: {e}")
    
    def _create_alert(self, symbol: str, alert_type: str, message: str, 
                     urgency: str, action_required: bool):
        """Create and store alert in database"""
        alert = TradingAlert(
            timestamp=datetime.datetime.now(),
            symbol=symbol,
            alert_type=alert_type,
            message=message,
            urgency=urgency,
            action_required=action_required
        )
        
        self.alerts.append(alert)
        
        # Store in database
        try:
            with self.db.get_connection() as conn:
                conn.execute("""
                    INSERT INTO alerts 
                    (timestamp, symbol, alert_type, message, urgency, action_required)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (alert.timestamp, alert.symbol, alert.alert_type, 
                      alert.message, alert.urgency, alert.action_required))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error storing alert: {e}")
        
        # Display alert
        urgency_color = Fore.RED if urgency == 'high' else Fore.YELLOW if urgency == 'medium' else Fore.WHITE
        print(f"\n{urgency_color}[ALERT] {alert_type} - {symbol}: {message}{Style.RESET_ALL}")
        
        # Log alert
        self.logger.info(f"Alert: {alert_type} - {symbol}: {message}")
    
    def view_alerts(self, hours: int = 24):
        """View recent alerts from database"""
        try:
            with self.db.get_connection() as conn:
                alerts = conn.execute("""
                    SELECT * FROM alerts 
                    WHERE timestamp > datetime('now', ? || ' hours')
                    ORDER BY timestamp DESC
                """, (-hours,)).fetchall()
                
                if not alerts:
                    print(f"\nNo alerts in the last {hours} hours")
                    return
                
                print(f"\n{Fore.CYAN}=== RECENT ALERTS ({hours}h) ==={Style.RESET_ALL}")
                
                for alert in alerts:
                    urgency_color = Fore.RED if alert['urgency'] == 'high' else Fore.YELLOW if alert['urgency'] == 'medium' else Fore.WHITE
                    action_str = " [ACTION REQUIRED]" if alert['action_required'] else ""
                    ack_str = " [ACKNOWLEDGED]" if alert['acknowledged'] else ""
                    
                    print(f"{alert['timestamp'][:16]} - "
                          f"{urgency_color}{alert['alert_type']}{Style.RESET_ALL} - "
                          f"{alert['symbol']}: {alert['message']}{action_str}{ack_str}")
                          
        except Exception as e:
            self.logger.error(f"Error viewing alerts: {e}")
    
    @robust_retry(error_type='data_fetch', max_retries=2)
    def get_universe_symbols(self) -> List[str]:
        """FIXED: Get universe symbols with fallback for connection errors"""
        
        # Hardcoded fallback list of liquid A-share stocks
        FALLBACK_UNIVERSE = [
            # Large cap banks and finance
            '000001',  # 平安银行
            '600036',  # 招商银行
            '600000',  # 浦发银行
            '600016',  # 民生银行
            '601318',  # 中国平安
            
            # Consumer and retail
            '600519',  # 贵州茅台
            '000858',  # 五粮液
            '002415',  # 海康威视
            '000333',  # 美的集团
            '000651',  # 格力电器
            
            # Technology
            '300059',  # 东方财富
            '300750',  # 宁德时代
            '002475',  # 立讯精密
            '300015',  # 爱尔眼科
            '002594',  # 比亚迪
            
            # Industrial and materials
            '600276',  # 恒瑞医药
            '000568',  # 泸州老窖
            '600048',  # 保利发展
            '600030',  # 中信证券
            '600050',  # 中国联通
            
            # Additional liquid stocks
            '000002',  # 万科A
            '600104',  # 上汽集团
            '601166',  # 兴业银行
            '000876',  # 新希望
            '002304',  # 洋河股份
            
            # Tech and growth
            '300014',  # 亿纬锂能
            '300760',  # 迈瑞医疗
            '688981',  # 中芯国际
            '600809',  # 山西汾酒
            '002460'   # 赣锋锂业
        ]
        
        try:
            # Try to get live data with timeout
            import socket
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(5)  # 5 second timeout
            
            try:
                df = ak.stock_zh_a_spot_em()
                socket.setdefaulttimeout(old_timeout)
                
                if df is not None and not df.empty and '成交额' in df.columns:
                    # Filter for liquid stocks
                    df['成交额'] = pd.to_numeric(df['成交额'], errors='coerce')
                    df = df.sort_values('成交额', ascending=False)
                    
                    # Get top 50 by turnover
                    symbols = df['代码'].head(50).tolist()
                    self.logger.info(f"Retrieved {len(symbols)} symbols from live data")
                    return symbols
                else:
                    raise Exception("Invalid data returned from API")
                    
            except Exception as e:
                socket.setdefaulttimeout(old_timeout)
                raise e
                
        except Exception as e:
            self.logger.warning(f"Failed to get live universe data: {e}")
            self.logger.info(f"Using fallback universe of {len(FALLBACK_UNIVERSE)} symbols")
            return FALLBACK_UNIVERSE
    
    def _get_default_universe(self) -> List[str]:
        """Get default universe of liquid stocks"""
        return ['000001', '000002', '000858', '600519', '600036',
                '000333', '002415', '600276', '300750', '000651',
                '600000', '600016', '600030', '600048', '600050',
                '000568', '002594', '300059', '002475', '300015']
    
    def analyze_performance(self):
        """Analyze historical performance from database"""
        try:
            with self.db.get_connection() as conn:
                # Get performance snapshots
                snapshots = conn.execute("""
                    SELECT * FROM performance_snapshots
                    ORDER BY snapshot_date DESC
                    LIMIT 60
                """).fetchall()
                
                if len(snapshots) < 2:
                    print("Insufficient performance history")
                    return
                
                print(f"\n{Fore.CYAN}=== PERFORMANCE ANALYSIS ==={Style.RESET_ALL}")
                
                # Convert to DataFrame for analysis
                df = pd.DataFrame(snapshots)
                df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
                df = df.set_index('snapshot_date').sort_index()
                
                # Summary statistics
                latest = df.iloc[-1]
                oldest = df.iloc[0]
                total_return = (latest['portfolio_value'] / oldest['portfolio_value'] - 1) * 100
                avg_positions = df['n_positions'].mean()
                
                print(f"\nPeriod: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
                print(f"Total return: {total_return:+.2f}%")
                print(f"Current portfolio value: {latest['portfolio_value']:,.0f}")
                print(f"Average positions: {avg_positions:.1f}")
                
                # Calculate Sharpe from daily returns
                if 'daily_return' in df.columns:
                    returns = df['daily_return'].dropna()
                    if len(returns) > 0 and returns.std() > 0:
                        sharpe = returns.mean() / returns.std() * np.sqrt(252)
                        print(f"Sharpe ratio: {sharpe:.2f}")
                
                # Best/worst days
                if 'daily_return' in df.columns:
                    best_day = df['daily_return'].idxmax()
                    worst_day = df['daily_return'].idxmin()
                    
                    print(f"\nBest day: {best_day.strftime('%Y-%m-%d')} ({df.loc[best_day, 'daily_return']:.2%})")
                    print(f"Worst day: {worst_day.strftime('%Y-%m-%d')} ({df.loc[worst_day, 'daily_return']:.2%})")
                
                # Trade history analysis
                trades = conn.execute("""
                    SELECT * FROM trade_history
                    WHERE action = 'SELL'
                    ORDER BY trade_date DESC
                """).fetchall()
                
                if trades:
                    trades_df = pd.DataFrame(trades)
                    winning_trades = trades_df[trades_df['return_pct'] > 0]
                    losing_trades = trades_df[trades_df['return_pct'] < 0]
                    
                    print(f"\n{Fore.YELLOW}TRADING STATISTICS:{Style.RESET_ALL}")
                    print(f"Total trades: {len(trades_df)}")
                    print(f"Win rate: {len(winning_trades) / len(trades_df):.1%}")
                    print(f"Average win: {winning_trades['return_pct'].mean():.2f}%")
                    print(f"Average loss: {losing_trades['return_pct'].mean():.2f}%")
                    print(f"Best trade: {trades_df['return_pct'].max():.2f}%")
                    print(f"Worst trade: {trades_df['return_pct'].min():.2f}%")
                    print(f"Total P&L: {trades_df['pnl'].sum():,.0f}")
                    
        except Exception as e:
            self.logger.error(f"Error analyzing performance: {e}")
    
    def generate_risk_dashboard(self):
        """Generate comprehensive risk dashboard with all features"""
        print(f"\n{Fore.CYAN}=== RISK MANAGEMENT DASHBOARD ==={Style.RESET_ALL}")
        print(f"Generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # System Status
        system_ok, issues = self.failsafe.check_all_systems()
        status_color = Fore.GREEN if system_ok else Fore.RED
        print(f"\n{Fore.YELLOW}System Status:{Style.RESET_ALL} {status_color}{'OK' if system_ok else 'ISSUES DETECTED'}{Style.RESET_ALL}")
        if not system_ok:
            for issue in issues:
                print(f"  {Fore.RED}• {issue}{Style.RESET_ALL}")
        
        # Circuit Breaker Status
        print(f"\n{Fore.YELLOW}Circuit Breakers:{Style.RESET_ALL}")
        print(f"  Daily trades: {self.failsafe.daily_stats['trades_today']}/{self.failsafe.circuit_breakers['max_daily_trades']}")
        print(f"  Daily loss: {self.failsafe.daily_stats['loss_today']:.2%}/{self.failsafe.circuit_breakers['max_daily_loss']:.2%}")
        print(f"  Emergency stop: {'ACTIVE' if self.failsafe.emergency_stop else 'Inactive'}")
        
        # Portfolio Risk Metrics
        risk_metrics = self.calculate_portfolio_risk_metrics()
        if risk_metrics:
            print(f"\n{Fore.YELLOW}Portfolio Risk Metrics:{Style.RESET_ALL}")
            print(f"  Volatility: {risk_metrics.get('portfolio_volatility', 0):.1%} annualized")
            print(f"  VaR (95%): {risk_metrics.get('var_95', 0):.2%}")
            print(f"  CVaR (95%): {risk_metrics.get('cvar_95', 0):.2%}")
            print(f"  Max Drawdown: {risk_metrics.get('max_drawdown', 0):.1%}")
            print(f"  Current Drawdown: {risk_metrics.get('current_drawdown', 0):.1%}")
            print(f"  Sharpe Ratio: {risk_metrics.get('sharpe_ratio', 0):.2f}")
            print(f"  Portfolio Beta: {risk_metrics.get('portfolio_beta', 1):.2f}")
            
            if 'avg_correlation' in risk_metrics:
                corr_color = Fore.GREEN if risk_metrics['avg_correlation'] < 0.5 else Fore.YELLOW if risk_metrics['avg_correlation'] < 0.7 else Fore.RED
                print(f"  Avg Correlation: {corr_color}{risk_metrics['avg_correlation']:.2f}{Style.RESET_ALL}")
        
        # Position Concentration
        if self.positions:
            print(f"\n{Fore.YELLOW}Position Analysis:{Style.RESET_ALL}")
            position_values = {}
            total_value = 0
            
            for symbol, pos in self.positions.items():
                df = self.get_market_data_cached(symbol)
                if df is not None:
                    current_price = df['close'].iloc[-1]
                    position_value = pos['quantity'] * current_price
                    position_values[symbol] = position_value
                    total_value += position_value
            
            if total_value > 0:
                # Find largest positions
                sorted_positions = sorted(position_values.items(), key=lambda x: x[1], reverse=True)
                print("  Top positions:")
                for symbol, value in sorted_positions[:3]:
                    weight = value / total_value
                    weight_color = Fore.GREEN if weight < 0.15 else Fore.YELLOW if weight < 0.25 else Fore.RED
                    print(f"    {symbol}: {weight_color}{weight:.1%}{Style.RESET_ALL} ({value:,.0f})")
        
        # Data Quality Summary
        print(f"\n{Fore.YELLOW}Data Quality Summary:{Style.RESET_ALL}")
        
        with self.db.get_connection() as conn:
            recent_checks = conn.execute("""
                SELECT symbol, AVG(quality_score) as avg_quality, COUNT(*) as checks
                FROM data_quality_log
                WHERE check_timestamp > datetime('now', '-1 day')
                GROUP BY symbol
            """).fetchall()
            
            if recent_checks:
                for check in recent_checks:
                    quality_color = Fore.GREEN if check['avg_quality'] >= 0.8 else Fore.YELLOW if check['avg_quality'] >= 0.6 else Fore.RED
                    print(f"  {check['symbol']}: {quality_color}{check['avg_quality']:.1%}{Style.RESET_ALL} ({check['checks']} checks)")
            else:
                print("  No recent data quality checks")
        
        # ML Model Status
        print(f"\n{Fore.YELLOW}ML Model Status:{Style.RESET_ALL}")
        ml_status = "Trained" if self.ml_engine.is_trained else "Untrained"
        ml_color = Fore.GREEN if self.ml_engine.is_trained else Fore.YELLOW
        print(f"  Status: {ml_color}{ml_status}{Style.RESET_ALL}")
        if self.ml_engine.feature_importance:
            print("  Top features:")
            sorted_features = sorted(self.ml_engine.feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True)[:3]
            for feat, imp in sorted_features:
                print(f"    • {feat}: {imp:.3f}")
        
        if hasattr(self.ml_engine, 'validation_metrics') and self.ml_engine.validation_metrics:
            print(f"  Model performance:")
            print(f"    Direction accuracy: {self.ml_engine.validation_metrics.get('direction_accuracy', 0):.1%}")
            print(f"    Correlation: {self.ml_engine.validation_metrics.get('correlation', 0):.3f}")
        
        # Adaptive Thresholds Status
        print(f"\n{Fore.YELLOW}Adaptive Thresholds:{Style.RESET_ALL}")
        current_thresholds = self.threshold_optimizer.adaptive_thresholds
        print(f"  Buy threshold: {current_thresholds['buy_threshold']:.1f}")
        print(f"  Sell threshold: {current_thresholds['sell_threshold']:.1f}")
        print(f"  Min signals: {current_thresholds['min_signals']}")
        
        # Error Summary
        if hasattr(self, 'error_handler'):
            error_summary = self.error_handler.get_error_summary()
            if error_summary['total_errors'] > 0:
                print(f"\n{Fore.YELLOW}Error Summary:{Style.RESET_ALL}")
                print(f"  Total errors: {error_summary['total_errors']}")
                if error_summary['most_common']:
                    print(f"  Most common: {error_summary['most_common'][0]} ({error_summary['most_common'][1]} times)")
                
                for error_type, count in error_summary['error_types'].items():
                    print(f"  {error_type}: {count}")
        
        # Database Status
        print(f"\n{Fore.YELLOW}Database Status:{Style.RESET_ALL}")
        db_ok = self.db.verify_integrity()
        db_color = Fore.GREEN if db_ok else Fore.RED
        print(f"  Integrity: {db_color}{'OK' if db_ok else 'FAILED'}{Style.RESET_ALL}")
        
        # Get file size
        if os.path.exists(self.db.db_path):
            size_mb = os.path.getsize(self.db.db_path) / (1024 * 1024)
            print(f"  Size: {size_mb:.1f} MB")
            if os.path.exists(self.db.backup_path):
                print(f"  Backup: Available")
            else:
                print(f"  Backup: {Fore.YELLOW}Not found{Style.RESET_ALL}")
    
    def generate_orders_enhanced(self, capital_per_position: float = 50000):
        """Fixed version without duplicate displays"""
        print(f"\n{Fore.CYAN}=== ENHANCED ORDER SUGGESTIONS ==={Style.RESET_ALL}")
        
        # Check system first
        system_ok, issues = self.failsafe.check_all_systems()
        if not system_ok:
            print(f"{Fore.RED}System issues preventing trading:{Style.RESET_ALL}")
            for issue in issues:
                print(f"  • {issue}")
            return
        
        # Check if emergency stop is active
        if self.failsafe.emergency_stop:
            print(f"{Fore.RED}EMERGENCY STOP IS ACTIVE - No trading allowed{Style.RESET_ALL}")
            return
        
        # Get portfolio value and market regime
        portfolio_value = self._calculate_portfolio_value()
        regime = self.detect_market_regime()
        
        # Risk metrics
        risk_metrics = self.calculate_portfolio_risk_metrics()
        portfolio_risk_ok = True
        
        if risk_metrics:
            if abs(risk_metrics.get('var_95', 0)) > self.risk_params['max_portfolio_risk']:
                print(f"{Fore.RED}Portfolio risk exceeds limit! VaR: {risk_metrics['var_95']:.2%}{Style.RESET_ALL}")
                portfolio_risk_ok = False
        
        # Generate sell orders
        sell_orders = []
        for symbol, pos in self.positions.items():
            if not self.can_sell_today(symbol):
                continue
            
            signals = self.generate_enhanced_signals(symbol)
            
            # Skip if data quality is too low
            if signals.get('data_quality', 0) < self.risk_params['min_data_quality']:
                continue
            
            df = self.get_market_data_cached(symbol)
            if df is None:
                continue
            
            current_price = df['close'].iloc[-1]
            pnl_pct = (current_price / pos['buy_price'] - 1) * 100
            
            # Get timing recommendation
            timing = self.get_optimal_execution_time(symbol, 'SELL')
            
            # Sell criteria
            should_sell = False
            urgency = "low"
            reason = ""
            
            # Stop loss
            if pnl_pct <= -self.risk_params['stop_loss'] * 100:
                should_sell = True
                urgency = "high"
                reason = f"Stop loss ({pnl_pct:.1f}%)"
            # Take profit
            elif pnl_pct >= self.risk_params['take_profit'] * 100:
                should_sell = True
                urgency = "medium"
                reason = f"Take profit ({pnl_pct:.1f}%)"
            # Sell signal
            elif signals['signal'] in ['SELL', 'STRONG_SELL'] and signals['data_quality'] >= 0.7:
                should_sell = True
                urgency = "medium"
                reason = f"{signals['signal']} (strength: {signals['strength']:.1f})"
            # Portfolio risk reduction
            elif not portfolio_risk_ok and pnl_pct > 0:
                should_sell = True
                urgency = "high"
                reason = "Risk reduction"
            # Market regime
            elif regime['regime'] in ['crisis', 'bear'] and pnl_pct > 0:
                should_sell = True
                urgency = "medium"
                reason = f"{regime['regime'].title()} market regime"
            
            if should_sell:
                sell_orders.append({
                    'symbol': symbol,
                    'quantity': pos['quantity'],
                    'current_price': current_price,
                    'pnl_pct': pnl_pct,
                    'reason': reason,
                    'urgency': urgency,
                    'signal_strength': signals['strength'],
                    'data_quality': signals.get('data_quality', 0),
                    'timing_recommendation': timing['suggestions'][0] if timing['suggestions'] else "Neutral timing"
                })
        
        # Display sell orders
        if sell_orders:
            print(f"\n{Fore.RED}SELL ORDERS:{Style.RESET_ALL}")
            for order in sorted(sell_orders, key=lambda x: x['urgency'], reverse=True):
                urgency_color = Fore.RED if order['urgency'] == 'high' else Fore.YELLOW
                print(f"  {urgency_color}[{order['urgency'].upper()}]{Style.RESET_ALL} SELL {order['symbol']}")
                print(f"    Quantity: {order['quantity']:,}")
                print(f"    Current: {order['current_price']:.2f} ({order['pnl_pct']:+.1f}%)")
                print(f"    Reason: {order['reason']}")
                print(f"    Timing: {order['timing_recommendation']}")
                print(f"    Data quality: {order['data_quality']:.1%}")
                print(f"    Estimated proceeds: {order['quantity'] * order['current_price']:,.0f}")
                print()
        
        # Generate buy orders - FIXED: Consistent variable naming
        available_capital = capital_per_position * (10 - len(self.positions) + len(sell_orders))
        
        if available_capital > 0 and portfolio_risk_ok and regime['regime'] not in ['crisis', 'bear']:
            print(f"\n{Fore.GREEN}BUY OPPORTUNITIES:{Style.RESET_ALL}")
            print(f"Available capital: {available_capital:,.0f}")
            
            # Check sector rotation
            sectors = self.analyze_sector_rotation()
            if sectors and 'top_sectors' in sectors:
                print(f"Top sectors: {', '.join(sectors['top_sectors'])}")
            
            # Scan universe for opportunities
            universe = self.get_universe_symbols()
            opportunities = []  # FIXED: Use consistent variable name throughout
            
            # Add watchlist to universe (prioritized)
            scan_list = self.watchlist + [s for s in universe[:30] if s not in self.watchlist]
            
            for symbol in scan_list[:40]:  # Limit scan for efficiency
                if symbol in self.positions:
                    continue
                
                # Use the improved signal generation
                signals = self.generate_enhanced_signals(symbol)
                
                # Skip low quality data
                if signals.get('data_quality', 0) < self.risk_params['min_data_quality']:
                    continue
                
                # Require minimum signal strength and confirmations
                if signals['signal'] in ['BUY', 'STRONG_BUY'] and signals.get('buy_signals', 0) >= 2:
                    df = self.get_market_data_cached(symbol)
                    if df is not None:
                        current_price = df['close'].iloc[-1]
                        
                        # Get additional metrics
                        factors = signals.get('factors', {})
                        
                        # Calculate position size
                        position_size = self.calculate_position_size(
                            symbol, 
                            signals['net_score'],
                            portfolio_value,
                            current_price,
                            signals['volatility']
                        )
                        
                        opportunities.append({  # FIXED: Use same variable name consistently
                            'symbol': symbol,
                            'signal': signals['signal'],
                            'strength': signals['strength'],
                            'buy_signals': signals.get('buy_signals', 0),
                            'ml_confidence': signals.get('ml_confidence', 0),
                            'price': current_price,
                            'suggested_shares': position_size,
                            'total_cost': position_size * current_price,
                            'reasons': signals['reasons'][:3],
                            'volatility': signals['volatility'],
                            'data_quality': signals.get('data_quality', 0),
                            'momentum': factors.get('momentum_5d', 0),
                            'rsi': factors.get('rsi_14', 50),
                            'from_watchlist': symbol in self.watchlist
                        })
            
            # Sort by strength and number of confirming signals
            opportunities.sort(key=lambda x: (x['strength'] + x['buy_signals']), reverse=True)
            
            # Group by volatility
            low_vol = [o for o in opportunities if o['volatility'] <= 0.35]
            med_vol = [o for o in opportunities if 0.35 < o['volatility'] <= 0.5]
            high_vol = [o for o in opportunities if o['volatility'] > 0.5]
            
            # Display opportunities by volatility category
            if low_vol:
                print(f"\n{Fore.GREEN}LOW VOLATILITY OPPORTUNITIES (≤35%):{Style.RESET_ALL}")
                self._display_opportunities(low_vol[:5], "LOW VOL")
            
            if med_vol:
                print(f"\n{Fore.YELLOW}MEDIUM VOLATILITY OPPORTUNITIES (35%-50%):{Style.RESET_ALL}")
                self._display_opportunities(med_vol[:3], "MED VOL")
            
            if high_vol:  # Show high vol regardless of mode
                print(f"\n{Fore.RED}HIGH VOLATILITY OPPORTUNITIES (>50%):{Style.RESET_ALL}")
                self._display_opportunities(high_vol[:2], "HIGH VOL")
            
            # Summary - FIXED: Now the count matches what was actually found
            print(f"\n{Fore.CYAN}OPPORTUNITY SUMMARY:{Style.RESET_ALL}")
            print(f"Total opportunities found: {len(opportunities)}")  # FIXED: This now matches
            print(f"Low volatility: {len(low_vol)}")
            print(f"Medium volatility: {len(med_vol)}")
            print(f"High volatility: {len(high_vol)}")
            
            if not opportunities:
                print(f"\n{Fore.YELLOW}No strong buy signals found. Consider:{Style.RESET_ALL}")
                print("  • Lowering signal thresholds (use aggressive mode)")
                print("  • Expanding universe scan")
                print("  • Waiting for better market conditions")
        
        elif regime['regime'] in ['crisis', 'bear']:
            print(f"\n{Fore.YELLOW}Market regime is {regime['regime']} - Limited buy opportunities{Style.RESET_ALL}")
        
        # Trading plan summary
        print(f"\n{Fore.CYAN}=== TRADING PLAN SUMMARY ==={Style.RESET_ALL}")
        print(f"Market regime: {regime['regime']} ({regime['confidence']:.0%} confidence)")
        print(f"Sell orders: {len(sell_orders)}")
        if sell_orders:
            total_sell_value = sum(o['quantity'] * o['current_price'] for o in sell_orders)
            print(f"Total sell value: {total_sell_value:,.0f}")
        
        # FIXED: Use the consistent variable name
        opportunities_count = len(opportunities) if 'opportunities' in locals() else 0
        print(f"Buy opportunities identified: {opportunities_count}")
        print(f"Portfolio risk status: {'OK' if portfolio_risk_ok else 'EXCEEDED'}")
        
        # Correlation check
        if len(self.positions) >= 2:
            correlations = self.analyze_correlations()
            if not correlations.empty:
                high_corr = correlations[correlations['correlation'] > 0.8]
                if not high_corr.empty:
                    print(f"\n{Fore.YELLOW}  High correlations detected:{Style.RESET_ALL}")
                    for _, corr in high_corr.iterrows():
                        print(f"    {corr['pair']}: {corr['correlation']:.2f}")
        
        # Show threshold recommendations
        threshold_recommendations = self.threshold_optimizer.get_threshold_recommendations()
        if threshold_recommendations['suggestions']:
            print(f"\n{Fore.YELLOW}Threshold Optimization Suggestions:{Style.RESET_ALL}")
            for suggestion in threshold_recommendations['suggestions']:
                print(f"  • {suggestion}")
    
    def _display_opportunities(self, opportunities: List[Dict], vol_category: str):
        """Helper to display opportunities without duplication"""
        for i, opp in enumerate(opportunities):
            print(f"  {i+1}. BUY {opp['symbol']} [{vol_category}]")
            print(f"     Signal: {opp['signal']} (strength: {opp['strength']:.1f}, ML: {opp['ml_confidence']:.1%})")
            print(f"     Price: {opp['price']:.2f}")
            print(f"     Suggested shares: {opp['suggested_shares']:,}")
            print(f"     Total cost: {opp['total_cost']:,.0f}")
            print(f"     Volatility: {opp['volatility']:.1%}")
            print(f"     Data quality: {opp['data_quality']:.1%}")
            print(f"     Momentum: {opp['momentum']:.1%}, RSI: {opp['rsi']:.0f}")
            
            timing = self.get_optimal_execution_time(opp['symbol'], 'BUY')
            print(f"     Timing: {timing['suggestions'][0] if timing['suggestions'] else 'Neutral timing'}")
            print(f"     Reasons: {', '.join(opp['reasons'][:2])}")
            print()
    
    def daily_report_enhanced(self):
        """Generate comprehensive daily report with all safeguards"""
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}ENHANCED DAILY TRADING REPORT - {self.today}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        
        # Run pre-market checks first
        checks_passed, check_results = self.run_pre_market_checks()
        
        if not checks_passed:
            print(f"\n{Fore.RED}⚠️  Pre-market checks failed! Review issues before trading.{Style.RESET_ALL}")
            response = input("Continue with report anyway? (yes/no): ")
            if response.lower() != 'yes':
                return
        
        # Backup database first
        if self.db.backup_database():
            print(f"{Fore.GREEN}✓ Database backup created{Style.RESET_ALL}")
        
        # System checks
        system_ok, issues = self.failsafe.check_all_systems()
        if not system_ok:
            print(f"\n{Fore.RED}⚠ SYSTEM ISSUES DETECTED:{Style.RESET_ALL}")
            for issue in issues:
                print(f"  • {issue}")
        
        # Market overview
        regime = self.detect_market_regime()
        
        print(f"\n{Fore.YELLOW}MARKET OVERVIEW:{Style.RESET_ALL}")
        print(f"Market regime: {regime['regime']} ({regime['confidence']:.0%} confidence)")
        print(f"Data quality: {regime.get('data_quality', 0):.1%}")
        print(f"Recommendation: {regime['recommendation']}")
        
        # Check positions
        self.check_positions_enhanced()
        
        # Generate orders
        self.generate_orders_enhanced()
        
        # Risk dashboard
        self.generate_risk_dashboard()
        
        # Add paper trading summary if active
        if self.paper_trading.paper_trades:
            print(f"\n{Fore.CYAN}=== PAPER TRADING SUMMARY ==={Style.RESET_ALL}")
            performance = self.paper_trading.update_daily_performance()
            print(f"Paper portfolio value: {performance['portfolio_value']:,.0f}")
            print(f"Paper total return: {performance['total_return']:+.2f}%")
            print(f"Paper trades today: {performance['n_trades']}")
        
        # Save performance snapshot
        self._save_performance_snapshot()
        
        self.logger.info(f"Daily report generated for {self.today}")
    
    def diagnose_signals(self, symbol: str, days: int = 30):
        """Diagnose why signals may not be generating trades"""
        print(f"\n{Fore.CYAN}=== SIGNAL DIAGNOSIS FOR {symbol} ==={Style.RESET_ALL}")
        
        df = self.get_market_data_cached(symbol, days=days+60)
        if df is None:
            print("No data available")
            return
        
        # Take last N days
        df_recent = df.tail(days)
        
        signal_history = []
        for i in range(len(df_recent)):
            date = df_recent.index[i]
            historical_data = df[df.index <= date].tail(90)
            
            if len(historical_data) < 60:
                continue
            
            # Temporarily set cache
            temp_cache = self.data_cache.copy()
            self.data_cache = {f"{symbol}_90": (historical_data, datetime.datetime.now().timestamp())}
            
            signals = self.generate_enhanced_signals(symbol)
            
            self.data_cache = temp_cache
            
            signal_history.append({
                'date': date,
                'signal': signals['signal'],
                'net_score': signals.get('net_score', 0),
                'buy_score': signals.get('buy_score', 0),
                'sell_score': signals.get('sell_score', 0),
                'ml_confidence': signals.get('ml_confidence', 0),
                'data_quality': signals.get('data_quality', 0),
                'reasons': signals.get('reasons', [])[:2]
            })
        
        if not signal_history:
            print("No signals generated")
            return
        
        # Convert to DataFrame for analysis
        df_signals = pd.DataFrame(signal_history)
        
        # Summary statistics
        print(f"\n{Fore.YELLOW}Signal Summary (Last {days} days):{Style.RESET_ALL}")
        signal_counts = df_signals['signal'].value_counts()
        for signal, count in signal_counts.items():
            print(f"  {signal}: {count} days ({count/len(df_signals)*100:.1f}%)")
        
        # Score analysis
        print(f"\n{Fore.YELLOW}Score Statistics:{Style.RESET_ALL}")
        print(f"  Buy scores: min={df_signals['buy_score'].min():.1f}, "
              f"avg={df_signals['buy_score'].mean():.1f}, "
              f"max={df_signals['buy_score'].max():.1f}")
        print(f"  Sell scores: min={df_signals['sell_score'].min():.1f}, "
              f"avg={df_signals['sell_score'].mean():.1f}, "
              f"max={df_signals['sell_score'].max():.1f}")
        print(f"  Net scores: min={df_signals['net_score'].min():.1f}, "
              f"avg={df_signals['net_score'].mean():.1f}, "
              f"max={df_signals['net_score'].max():.1f}")
        
        # ML and data quality
        print(f"\n{Fore.YELLOW}Quality Metrics:{Style.RESET_ALL}")
        print(f"  ML confidence: avg={df_signals['ml_confidence'].mean():.1%}")
        print(f"  Data quality: avg={df_signals['data_quality'].mean():.1%}")
        
        # Show some examples
        print(f"\n{Fore.YELLOW}Recent Signal Examples:{Style.RESET_ALL}")
        for _, row in df_signals.tail(5).iterrows():
            signal_color = Fore.GREEN if 'BUY' in row['signal'] else Fore.RED if 'SELL' in row['signal'] else Fore.YELLOW
            print(f"  {row['date'].strftime('%Y-%m-%d')}: {signal_color}{row['signal']}{Style.RESET_ALL} "
                  f"(net: {row['net_score']:.1f}, quality: {row['data_quality']:.1%})")
            if row['reasons']:
                print(f"    Reasons: {', '.join(row['reasons'])}")
        
        # Threshold analysis
        print(f"\n{Fore.YELLOW}Signal Threshold Analysis:{Style.RESET_ALL}")
        buy_threshold_3 = (df_signals['net_score'] >= 3).sum()
        buy_threshold_1_5 = (df_signals['net_score'] >= 1.5).sum()
        sell_threshold_3 = (df_signals['net_score'] <= -3).sum()
        sell_threshold_1_5 = (df_signals['net_score'] <= -1.5).sum()
        
        print(f"  Days meeting STRONG_BUY (score >= 3): {buy_threshold_3}")
        print(f"  Days meeting BUY (score >= 1.5): {buy_threshold_1_5}")
        print(f"  Days meeting STRONG_SELL (score <= -3): {sell_threshold_3}")
        print(f"  Days meeting SELL (score <= -1.5): {sell_threshold_1_5}")
        
        # Recommendations
        print(f"\n{Fore.YELLOW}Recommendations:{Style.RESET_ALL}")
        if buy_threshold_3 == 0 and sell_threshold_3 == 0:
            print("  ⚠️  No strong signals generated - thresholds may be too high")
        if df_signals['ml_confidence'].mean() < 0.1:
            print("  ⚠️  ML confidence very low - model may need training")
        if df_signals['data_quality'].mean() < 0.8:
            print("  ⚠️  Data quality issues may be preventing trades")
        if abs(df_signals['net_score'].mean()) < 0.5:
            print("  ⚠️  Signals too neutral - may need to adjust scoring weights")
    
    def _save_performance_snapshot(self):
        """Save daily performance snapshot to database"""
        try:
            portfolio_value = self._calculate_portfolio_value()
            
            # Get yesterday's value for return calculation
            with self.db.get_connection() as conn:
                yesterday = conn.execute("""
                    SELECT portfolio_value FROM performance_snapshots
                    WHERE snapshot_date < ?
                    ORDER BY snapshot_date DESC
                    LIMIT 1
                """, (self.today.strftime('%Y-%m-%d'),)).fetchone()
                
                daily_return = 0
                if yesterday:
                    daily_return = (portfolio_value / yesterday['portfolio_value'] - 1)
                
                # Calculate total P&L
                total_pnl = sum(
                    pos['quantity'] * self.get_market_data_cached(symbol)['close'].iloc[-1] - pos['cost_basis']
                    for symbol, pos in self.positions.items()
                    if self.get_market_data_cached(symbol) is not None
                )
                
                # Insert snapshot
                conn.execute("""
                    INSERT OR REPLACE INTO performance_snapshots
                    (snapshot_date, portfolio_value, n_positions, total_pnl, daily_return, market_regime)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (self.today.strftime('%Y-%m-%d'), portfolio_value, len(self.positions),
                      total_pnl, daily_return, self.market_regime.get('regime') if self.market_regime else 'unknown'))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error saving performance snapshot: {e}")
    
    def retrain_ml_model_enhanced(self, force_retrain: bool = False):
        """FIXED: Retrain ML model with network error resilience"""
        print(f"\n{Fore.CYAN}=== RETRAINING ML MODEL ==={Style.RESET_ALL}")
        
        try:
            # Collect training data from multiple symbols
            all_data = []
            symbols_processed = 0
            symbols_failed = 0
            
            # Get diverse universe of stocks with fallback
            try:
                universe = self.get_universe_symbols()
                if universe is None or len(universe) == 0:
                    raise Exception("Universe symbols unavailable")
                universe = universe[:30]  # Limit for stability
            except Exception as e:
                self.logger.warning(f"Error getting universe: {e}")
                # Use hardcoded reliable symbols
                universe = [
                    '000001', '600519', '000002', '600036', '000858',
                    '000333', '002415', '600276', '300750', '000651',
                    '600000', '600016', '600030', '600048', '600050',
                    '000568', '002594', '300059', '002475', '300015'
                ]
                print(f"Using fallback universe of {len(universe)} symbols")
            
            print(f"Processing {len(universe)} symbols for training data...")
            
            for symbol in universe:
                try:
                    print(f"Processing {symbol}...", end=' ')
                    
                    # Get data with timeout and retry
                    df = None
                    for attempt in range(2):
                        try:
                            df = self.get_market_data_cached(symbol, days=252)
                            if df is not None:
                                break
                        except Exception as e:
                            if attempt == 0:
                                print(f"retry...", end=' ')
                                time.sleep(1)
                            else:
                                raise e
                    
                    if df is None or len(df) < 100:
                        print(f"❌ Insufficient data: {len(df) if df is not None else 0} rows")
                        symbols_failed += 1
                        continue
                    
                    # Validate data quality
                    try:
                        validation = self.data_validator.validate_market_data(df, symbol)
                        if validation.quality_score < 0.6:
                            print(f"❌ Poor quality: {validation.quality_score:.1%}")
                            symbols_failed += 1
                            continue
                    except Exception as e:
                        print(f"❌ Validation failed: {str(e)[:30]}")
                        symbols_failed += 1
                        continue
                    
                    # Calculate features
                    try:
                        features_df = self.ml_engine._prepare_ml_features(df, symbol)
                        
                        if features_df is not None and len(features_df) > 20:
                            all_data.append(features_df)
                            symbols_processed += 1
                            print(f"✅ {len(features_df)} samples")
                        else:
                            print(f"❌ Feature prep failed")
                            symbols_failed += 1
                    except Exception as e:
                        print(f"❌ Error: {str(e)[:30]}")
                        symbols_failed += 1
                        continue
                        
                except Exception as e:
                    print(f"❌ {symbol}: {str(e)[:30]}")
                    symbols_failed += 1
                    continue
            
            print(f"\nSummary: {symbols_processed} successful, {symbols_failed} failed")
            
            if symbols_processed < 3:  # Reduced minimum requirement
                print(f"{Fore.RED}Insufficient symbols for training: {symbols_processed} (need at least 3){Style.RESET_ALL}")
                return False
            
            # Combine all data
            try:
                combined_data = pd.concat(all_data, ignore_index=True)
                print(f"Total training samples: {len(combined_data)}")
                
                if combined_data.empty or 'target' not in combined_data.columns:
                    print(f"{Fore.RED}Invalid combined data{Style.RESET_ALL}")
                    return False
                
                print(f"Target distribution:")
                print(combined_data['target'].value_counts().sort_index())
                
            except Exception as e:
                print(f"{Fore.RED}Error combining data: {e}{Style.RESET_ALL}")
                return False
            
            # Remove outliers safely
            try:
                combined_data = self.ml_engine._remove_outliers(combined_data)
            except Exception as e:
                print(f"Warning: Outlier removal failed: {e}")
            
            # Train model
            try:
                success = self.ml_engine._train_with_validation(combined_data)
                
                if success:
                    print(f"{Fore.GREEN}ML model retrained successfully{Style.RESET_ALL}")
                    # Try to analyze performance (but don't fail if it errors)
                    try:
                        self._analyze_model_performance()
                    except Exception as e:
                        print(f"Warning: Performance analysis failed: {e}")
                    return True
                else:
                    print(f"{Fore.RED}Model training failed - performance too low{Style.RESET_ALL}")
                    return False
                    
            except Exception as e:
                print(f"{Fore.RED}Training failed: {e}{Style.RESET_ALL}")
                return False
                
        except Exception as e:
            self.logger.error(f"Model retraining error: {e}")
            print(f"{Fore.RED}Critical error: {e}{Style.RESET_ALL}")
            return False
    

    
    def _analyze_model_performance(self):
        """FIXED: Analyze ML model performance with robust error handling"""
        print(f"\n{Fore.YELLOW}Model Analysis:{Style.RESET_ALL}")
        
        try:
            # Get universe with error handling
            try:
                universe = self.get_universe_symbols()
                if universe is None or len(universe) == 0:
                    universe = ['000001', '600519', '000002', '600036', '000858']  # Minimal fallback
                universe = universe[:20]  # Limit to 20 for performance
            except Exception as e:
                self.logger.error(f"Error getting universe for analysis: {e}")
                universe = ['000001', '600519', '000002', '600036', '000858']  # Minimal fallback
            
            # Test predictions on available universe
            predictions = {}
            successful_predictions = 0
            
            for symbol in universe:
                try:
                    pred = self.get_ml_prediction(symbol)
                    if pred and pred.get('prediction_quality') == 'trained':
                        predictions[symbol] = pred
                        successful_predictions += 1
                except Exception as e:
                    self.logger.warning(f"Prediction failed for {symbol}: {e}")
                    continue
            
            if predictions:
                ml_signals = [p['ml_signal'] for p in predictions.values()]
                ml_confidences = [p['ml_confidence'] for p in predictions.values()]
                
                print(f"  Prediction diversity:")
                print(f"    Symbols analyzed: {len(predictions)}")
                print(f"    Signal range: [{min(ml_signals):.3f}, {max(ml_signals):.3f}]")
                print(f"    Signal std dev: {np.std(ml_signals):.3f}")
                print(f"    Confidence range: [{min(ml_confidences):.1%}, {max(ml_confidences):.1%}]")
                print(f"    Unique signals: {len(set(np.round(ml_signals, 3)))}")
            else:
                print(f"  No valid predictions generated from {len(universe)} symbols")
            
            # Show top features if available
            if hasattr(self.ml_engine, 'feature_importance') and self.ml_engine.feature_importance:
                print(f"\n  Top 5 Features:")
                sorted_features = sorted(self.ml_engine.feature_importance.items(), 
                                    key=lambda x: x[1], reverse=True)[:5]
                for feat, imp in sorted_features:
                    print(f"    {feat}: {imp:.3f}")
            else:
                print(f"\n  Feature importance not available")
                
            # Show ensemble weights if available
            if hasattr(self.ml_engine, 'ensemble_weights') and self.ml_engine.ensemble_weights:
                print(f"\n  Ensemble Weights:")
                for model, weight in self.ml_engine.ensemble_weights.items():
                    print(f"    {model}: {weight:.1%}")
                    
        except Exception as e:
            self.logger.error(f"Model analysis error: {e}")
            print(f"  Model analysis failed: {str(e)[:100]}")
    
    def analyze_signal_weakness(self, n_symbols: int = 10):
        """Analyze why signals might be weak across multiple symbols"""
        print(f"\n{Fore.CYAN}=== SIGNAL WEAKNESS ANALYSIS ==={Style.RESET_ALL}")
        
        universe = self.get_universe_symbols()[:n_symbols]
        signal_stats = {
            'total_analyzed': 0,
            'strong_signals': 0,
            'weak_signals': 0,
            'hold_signals': 0,
            'ml_issues': 0,
            'low_confidence': 0,
            'insufficient_confirmations': 0,
            'errors': 0
        }
        
        weak_examples = []
        
        for symbol in universe:
            try:
                signals = self.generate_enhanced_signals(symbol)
                
                # Check if signal generation failed
                if signals['signal'] == 'ERROR':
                    signal_stats['errors'] += 1
                    continue
                    
                signal_stats['total_analyzed'] += 1
                
                if signals['signal'] in ['STRONG_BUY', 'STRONG_SELL']:
                    signal_stats['strong_signals'] += 1
                elif signals['signal'] in ['BUY', 'SELL']:
                    signal_stats['weak_signals'] += 1
                else:
                    signal_stats['hold_signals'] += 1
                
                # Analyze weaknesses - use get() to avoid KeyError
                ml_quality = signals.get('ml_quality', 'unknown')
                if ml_quality != 'trained':
                    signal_stats['ml_issues'] += 1
                    
                ml_confidence = signals.get('ml_confidence', 0)
                if ml_confidence < 0.4:
                    signal_stats['low_confidence'] += 1
                    
                if signals.get('buy_signals', 0) < 2 and signals.get('sell_signals', 0) < 2:
                    signal_stats['insufficient_confirmations'] += 1
                
                # Collect examples of weak signals
                net_score = signals.get('net_score', 0)
                if abs(net_score) < 2:
                    weak_examples.append({
                        'symbol': symbol,
                        'net_score': net_score,
                        'buy_signals': signals.get('buy_signals', 0),
                        'sell_signals': signals.get('sell_signals', 0),
                        'ml_confidence': ml_confidence,
                        'reasons': signals.get('reasons', [])[:2]
                    })
                    
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {e}")
                signal_stats['errors'] += 1
                continue
        
        # Display analysis
        if signal_stats['total_analyzed'] > 0:
            print(f"\nSignal Distribution:")
            print(f"  Strong signals: {signal_stats['strong_signals']} ({signal_stats['strong_signals']/signal_stats['total_analyzed']*100:.1f}%)")
            print(f"  Weak signals: {signal_stats['weak_signals']} ({signal_stats['weak_signals']/signal_stats['total_analyzed']*100:.1f}%)")
            print(f"  Hold signals: {signal_stats['hold_signals']} ({signal_stats['hold_signals']/signal_stats['total_analyzed']*100:.1f}%)")
            
            print(f"\nCommon Issues:")
            print(f"  ML not trained/low quality: {signal_stats['ml_issues']}")
            print(f"  Low ML confidence: {signal_stats['low_confidence']}")
            print(f"  Insufficient confirmations: {signal_stats['insufficient_confirmations']}")
            if signal_stats['errors'] > 0:
                print(f"  Errors during analysis: {signal_stats['errors']}")
            
            if weak_examples:
                print(f"\nWeak Signal Examples:")
                for ex in weak_examples[:3]:
                    print(f"  {ex['symbol']}: net={ex['net_score']:.1f}, "
                          f"buy_signals={ex['buy_signals']}, sell_signals={ex['sell_signals']}, "
                          f"ML={ex['ml_confidence']:.1%}")
            
            print(f"\nRecommendations:")
            if signal_stats['ml_issues'] > signal_stats['total_analyzed'] * 0.3:
                print("  • Retrain ML model with more diverse data")
            if signal_stats['hold_signals'] > signal_stats['total_analyzed'] * 0.7:
                print("  • Consider using aggressive mode for more signals")
                print("  • Current market conditions may be range-bound")
            if signal_stats['insufficient_confirmations'] > signal_stats['total_analyzed'] * 0.5:
                print("  • Lower confirmation requirements in signal generation")
                print("  • Add more technical indicators")
        else:
            print(f"\n{Fore.RED}No symbols could be analyzed successfully{Style.RESET_ALL}")
            if signal_stats['errors'] > 0:
                print(f"Total errors: {signal_stats['errors']}")
                print("Check data quality and internet connection")
    
    # Add new methods for enhanced functionality
    def run_pre_market_checks(self):
        """Run pre-market checks"""
        return self.pre_market_checker.pre_market_checklist()
    
    def estimate_trade_costs(self, symbol: str, action: str, quantity: int, price: float):
        """Estimate transaction costs"""
        return self.execution_model.calculate_transaction_costs(symbol, action, quantity, price)
    
    def execute_paper_trade(self, symbol: str, action: str, quantity: int, price: float, 
                           signal_strength: float = 0, signal_reason: str = ""):
        """Execute paper trade"""
        return self.paper_trading.execute_trade(symbol, action, quantity, price, 
                                              signal_strength, signal_reason)
    
    def display_ml_health(self):
        """Display ML model health metrics"""
        print(f"\n{Fore.CYAN}=== ML MODEL HEALTH ==={Style.RESET_ALL}")
        
        # Get recent predictions
        predictions = []
        for symbol in list(self.positions.keys())[:5]:  # Check up to 5 positions
            pred = self.get_ml_prediction(symbol)
            predictions.append(pred)
        
        if predictions:
            # Check prediction diversity
            ml_signals = [p['ml_signal'] for p in predictions]
            print(f"Signal diversity: {np.std(ml_signals):.3f}")
            print(f"Average confidence: {np.mean([p['ml_confidence'] for p in predictions]):.1%}")
            
        # Show validation metrics
        if hasattr(self.ml_engine, 'validation_metrics') and self.ml_engine.validation_metrics:
            metrics = self.ml_engine.validation_metrics
            print(f"\nValidation Performance:")
            print(f"  Direction accuracy: {metrics.get('direction_accuracy', 0):.1%}")
            print(f"  Correlation: {metrics.get('correlation', 0):.3f}")
            print(f"  Profitable accuracy: {metrics.get('profitable_accuracy', 0):.1%}")
        
        # Show ensemble weights
        if hasattr(self.ml_engine, 'ensemble_weights'):
            print(f"\nEnsemble Weights:")
            for model, weight in self.ml_engine.ensemble_weights.items():
                print(f"  {model}: {weight:.1%}")
    
    def track_signal_performance(self):
        """Track and display signal performance over time"""
        recommendations = self.threshold_optimizer.get_threshold_recommendations()
        
        print(f"\n{Fore.CYAN}=== SIGNAL PERFORMANCE ==={Style.RESET_ALL}")
        print(f"Current thresholds:")
        for key, value in recommendations['current_thresholds'].items():
            print(f"  {key}: {value}")
        
        analysis = recommendations['performance_analysis']
        print(f"\nPerformance (last 30 days):")
        print(f"  Sample size: {analysis['sample_size']} trades")
        print(f"  Win rate: {analysis['win_rate']:.1%}")
        print(f"  Avg return: {analysis['avg_return']:.2%}")
        print(f"  Signal frequency: {analysis['signal_frequency']:.1%} of days")
        print(f"  Sharpe: {analysis['sharpe']:.2f}")
        
        print(f"\nRecommendations:")
        for suggestion in recommendations['suggestions']:
            print(f"  • {suggestion}")
    
    def display_error_summary(self):
        """Display error summary and patterns"""
        summary = self.error_handler.get_error_summary()
        
        print(f"\n{Fore.CYAN}=== ERROR SUMMARY ==={Style.RESET_ALL}")
        print(f"Total errors: {summary['total_errors']}")
        
        if summary['most_common']:
            print(f"Most common: {summary['most_common'][0]} ({summary['most_common'][1]} times)")
        
        if summary['error_types']:
            print(f"\nError types:")
            for error_type, count in summary['error_types'].items():
                print(f"  {error_type}: {count}")
        
        print(f"\nRecent errors:")
        for error in summary['recent_errors']:
            print(f"  {error['time']} - {error['key']}: {error['message']}")


# ============================================================================
# MAIN FUNCTION AND MENU SYSTEM
# ============================================================================

def main():
    """Main function with interactive menu"""
    system = EnhancedManualTradingSystem()
    
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}ENHANCED MANUAL TRADING SYSTEM - PRODUCTION READY{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print("Professional-grade A-share trading with all enhancements")
    print(f"{Fore.GREEN}✓ Pre-market checks{Style.RESET_ALL}")
    print(f"{Fore.GREEN}✓ Realistic execution model{Style.RESET_ALL}")
    print(f"{Fore.GREEN}✓ Paper trading mode{Style.RESET_ALL}")
    print(f"{Fore.GREEN}✓ Enhanced risk management (0.8 data quality){Style.RESET_ALL}")
    print(f"{Fore.GREEN}✓ All production safeguards{Style.RESET_ALL}")
    print(f"{Fore.GREEN}✓ Improved ML with ensemble models{Style.RESET_ALL}")
    print(f"{Fore.GREEN}✓ Adaptive signal thresholds{Style.RESET_ALL}")
    print(f"{Fore.GREEN}✓ Robust error handling{Style.RESET_ALL}")
    
    while True:
        print(f"\n{Fore.YELLOW}=== MAIN MENU ==={Style.RESET_ALL}")
        print("1. Daily trading report (with pre-market checks)")
        print("2. Check positions")
        print("3. Generate trading signals")
        print("4. Add position")
        print("5. Remove position (sell)")
        print("6. Risk dashboard")
        print("7. Watchlist management")
        print("8. Backtest strategy")
        print("9. Performance analysis")
        print("10. Start/stop monitoring")
        print("11. Market analysis")
        print("12. View alerts")
        print("13. System controls")
        print("14. Database maintenance")
        print("15. Retrain ML model")
        print("16. Analyze signal weakness")
        print("17. Paper trading")
        print("18. Pre-market checklist")
        print("19. Estimate trade costs")
        print("20. ML model health")
        print("21. Signal performance tracking")
        print("22. Error summary")
        print("0. Exit")
        
        choice = input("\nEnter choice: ")
        
        if choice == '1':
            system.daily_report_enhanced()
            
        elif choice == '2':
            system.check_positions_enhanced()
            
        elif choice == '3':
            sub_menu = True
            while sub_menu:
                print(f"\n{Fore.YELLOW}Signal Generation:{Style.RESET_ALL}")
                print("1. Single symbol")
                print("2. All positions")
                print("3. Watchlist")
                print("4. Universe scan")
                print("5. Diagnose signals")
                print("0. Back")
                
                sub_choice = input("Choice: ")
                
                if sub_choice == '1':
                    symbol = input("Enter symbol: ").upper()
                    signals = system.generate_enhanced_signals(symbol)
                    
                    print(f"\n{Fore.CYAN}Signals for {symbol}:{Style.RESET_ALL}")
                    signal_color = Fore.GREEN if 'BUY' in signals['signal'] else Fore.RED if 'SELL' in signals['signal'] else Fore.YELLOW
                    print(f"Signal: {signal_color}{signals['signal']}{Style.RESET_ALL}")
                    print(f"Strength: {signals['strength']:.1f}")
                    print(f"Buy score: {signals['buy_score']:.1f}")
                    print(f"Sell score: {signals['sell_score']:.1f}")
                    print(f"Volatility: {signals['volatility']:.1%}")
                    print(f"Data quality: {signals.get('data_quality', 0):.1%}")
                    print(f"ML confidence: {signals.get('ml_confidence', 0):.1%}")
                    print(f"ML quality: {signals.get('ml_quality', 'unknown')}")
                    print(f"Buy signals: {signals.get('buy_signals', 0)}")
                    print(f"Sell signals: {signals.get('sell_signals', 0)}")
                    print(f"Reasons: {', '.join(signals['reasons'])}")
                    
                    # Show adaptive thresholds used
                    if 'thresholds_used' in signals:
                        print(f"\nThresholds used:")
                        print(f"  Buy: {signals['thresholds_used']['buy_threshold']:.1f}")
                        print(f"  Sell: {signals['thresholds_used']['sell_threshold']:.1f}")
                        print(f"  Min signals: {signals['thresholds_used']['min_signals']}")
                    
                    # Execution timing
                    timing = system.get_optimal_execution_time(symbol, 'BUY' if 'BUY' in signals['signal'] else 'SELL')
                    print(f"\nExecution timing: {timing['suggestions'][0] if timing['suggestions'] else 'Neutral'}")
                    
                elif sub_choice == '2':
                    for symbol in system.positions:
                        signals = system.generate_enhanced_signals(symbol)
                        signal_color = Fore.GREEN if 'BUY' in signals['signal'] else Fore.RED if 'SELL' in signals['signal'] else Fore.YELLOW
                        print(f"\n{symbol}: {signal_color}{signals['signal']}{Style.RESET_ALL} "
                              f"(strength: {signals['strength']:.1f}, ML: {signals.get('ml_confidence', 0):.1%}, "
                              f"quality: {signals.get('data_quality', 0):.1%}, "
                              f"confirmations: {signals.get('buy_signals', 0) + signals.get('sell_signals', 0)})")
                
                elif sub_choice == '3':
                    for symbol in system.watchlist:
                        signals = system.generate_enhanced_signals(symbol)
                        signal_color = Fore.GREEN if 'BUY' in signals['signal'] else Fore.RED if 'SELL' in signals['signal'] else Fore.YELLOW
                        print(f"\n{symbol}: {signal_color}{signals['signal']}{Style.RESET_ALL} "
                              f"(strength: {signals['strength']:.1f}, ML: {signals.get('ml_confidence', 0):.1%}, "
                              f"quality: {signals.get('data_quality', 0):.1%}, "
                              f"confirmations: {signals.get('buy_signals', 0) + signals.get('sell_signals', 0)})")
                
                elif sub_choice == '4':
                    print("Scanning universe (this may take a moment)...")
                    universe = system.get_universe_symbols()[:20]  # Top 20 for efficiency
                    strong_signals = []
                    
                    for symbol in universe:
                        if symbol not in system.positions:
                            signals = system.generate_enhanced_signals(symbol)
                            if signals['signal'] in ['BUY', 'STRONG_BUY'] and signals['strength'] > 2 and signals.get('buy_signals', 0) >= 2:
                                strong_signals.append((symbol, signals))
                    
                    strong_signals.sort(key=lambda x: x[1]['strength'], reverse=True)
                    
                    print(f"\n{Fore.GREEN}Top Buy Signals:{Style.RESET_ALL}")
                    for symbol, signals in strong_signals[:5]:
                        print(f"{symbol}: {signals['signal']} (strength: {signals['strength']:.1f}, "
                              f"confirmations: {signals.get('buy_signals', 0)}, "
                              f"ML: {signals.get('ml_confidence', 0):.1%}, "
                              f"quality: {signals.get('data_quality', 0):.1%})")
                
                elif sub_choice == '5':
                    symbol = input("Symbol to diagnose: ").upper()
                    days = int(input("Days to analyze (default 30): ") or "30")
                    system.diagnose_signals(symbol, days)
                
                elif sub_choice == '0':
                    sub_menu = False
                    
        elif choice == '4':
            symbol = input("Symbol: ").upper()
            quantity = int(input("Quantity: "))
            price = float(input("Buy price: "))
            date_str = input("Buy date (YYYY-MM-DD) or Enter for today: ")
            
            if date_str:
                buy_date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
            else:
                buy_date = None
            
            system.add_position(symbol, quantity, price, buy_date)
            
        elif choice == '5':
            if not system.positions:
                print("No positions to sell")
                continue
                
            print("\nCurrent positions:")
            for symbol in system.positions:
                can_sell = system.can_sell_today(symbol)
                status = f"{Fore.GREEN}Can sell{Style.RESET_ALL}" if can_sell else f"{Fore.YELLOW}T+1 restricted{Style.RESET_ALL}"
                print(f"  {symbol}: {status}")
            
            symbol = input("\nSymbol to sell: ").upper()
            if symbol in system.positions:
                if system.can_sell_today(symbol):
                    price = float(input("Sell price: "))
                    system.remove_position(symbol, price)
                else:
                    print(f"{Fore.YELLOW}Cannot sell {symbol} today due to T+1 rule{Style.RESET_ALL}")
            else:
                print("Symbol not in positions")
                
        elif choice == '6':
            system.generate_risk_dashboard()
            
        elif choice == '7':
            sub_menu = True
            while sub_menu:
                print(f"\n{Fore.YELLOW}Watchlist Management:{Style.RESET_ALL}")
                print(f"Current watchlist: {', '.join(system.watchlist) if system.watchlist else 'Empty'}")
                print("\n1. Add symbol")
                print("2. Remove symbol")
                print("3. View with signals")
                print("4. Clear watchlist")
                print("0. Back")
                
                sub_choice = input("Choice: ")
                
                if sub_choice == '1':
                    symbol = input("Symbol to add: ").upper()
                    priority = int(input("Priority (0-10, default 0): ") or "0")
                    notes = input("Notes (optional): ")
                    system.add_to_watchlist(symbol, priority, notes)
                elif sub_choice == '2':
                    symbol = input("Symbol to remove: ").upper()
                    system.remove_from_watchlist(symbol)
                elif sub_choice == '3':
                    # View watchlist with signals
                    for symbol in system.watchlist:
                        signals = system.generate_enhanced_signals(symbol)
                        signal_color = Fore.GREEN if 'BUY' in signals['signal'] else Fore.RED if 'SELL' in signals['signal'] else Fore.YELLOW
                        print(f"\n{symbol}: {signal_color}{signals['signal']}{Style.RESET_ALL} "
                              f"(strength: {signals['strength']:.1f}, confirmations: {signals.get('buy_signals', 0)}, "
                              f"quality: {signals.get('data_quality', 0):.1%})")
                elif sub_choice == '4':
                    confirm = input("Clear entire watchlist? (yes/no): ")
                    if confirm.lower() == 'yes':
                        system.watchlist = []
                        with system.db.get_connection() as conn:
                            conn.execute("DELETE FROM watchlist")
                            conn.commit()
                        print("Watchlist cleared")
                elif sub_choice == '0':
                    sub_menu = False
                    
        elif choice == '8':
            symbol = input("Symbol to backtest: ").upper()
            start_date = input("Start date (YYYY-MM-DD): ")
            end_date = input("End date (YYYY-MM-DD): ")
            capital = float(input("Initial capital (default 100000): ") or "100000")
            mode = input("Use aggressive mode? (yes/no, default no): ").lower()
            aggressive = mode == 'yes'
            
            system.backtest_strategy(symbol, start_date, end_date, capital, aggressive)
            
        elif choice == '9':
            system.analyze_performance()
            
        elif choice == '10':
            if system.monitoring_active:
                system.stop_monitoring()
            else:
                system.start_monitoring()
                
        elif choice == '11':
            print(f"\n{Fore.CYAN}=== MARKET ANALYSIS ==={Style.RESET_ALL}")
            
            # Market regime
            regime = system.detect_market_regime()
            print(f"\n{Fore.YELLOW}Market Regime:{Style.RESET_ALL}")
            print(f"  Current: {regime['regime']}")
            print(f"  Confidence: {regime['confidence']:.0%}")
            print(f"  Data quality: {regime.get('data_quality', 0):.1%}")
            print(f"  Volatility percentile: {regime.get('volatility_percentile', 0):.0%}")
            print(f"  Recommendation: {regime['recommendation']}")
            
            # Sector rotation
            sectors = system.analyze_sector_rotation()
            if sectors and 'rankings' in sectors:
                print(f"\n{Fore.YELLOW}Sector Performance:{Style.RESET_ALL}")
                for sector, data in sectors['rankings'][:5]:
                    momentum_color = Fore.GREEN if data['momentum_20d'] > 0 else Fore.RED
                    print(f"  {sector}: {momentum_color}{data['momentum_20d']:+.1f}%{Style.RESET_ALL} "
                          f"(RSI: {data['rsi']:.0f}, Quality: {data.get('data_quality', 0):.1%})")
                
                if sectors.get('rotation_signal'):
                    print(f"\nRotation signal: {sectors['rotation_signal']}")
            
            # Correlation analysis
            if len(system.positions) >= 2:
                correlations = system.analyze_correlations()
                if not correlations.empty:
                    print(f"\n{Fore.YELLOW}Portfolio Correlations:{Style.RESET_ALL}")
                    for _, corr in correlations.iterrows():
                        corr_color = Fore.GREEN if corr['correlation'] < 0.5 else Fore.YELLOW if corr['correlation'] < 0.7 else Fore.RED
                        print(f"  {corr['pair']}: {corr_color}{corr['correlation']:.2f}{Style.RESET_ALL} - {corr['diversification']}")
            
            # Risk metrics
            risk_metrics = system.calculate_portfolio_risk_metrics()
            if risk_metrics:
                print(f"\n{Fore.YELLOW}Portfolio Risk Summary:{Style.RESET_ALL}")
                print(f"  Volatility: {risk_metrics.get('portfolio_volatility', 0):.1%}")
                print(f"  VaR (95%): {risk_metrics.get('var_95', 0):.2%}")
                print(f"  Sharpe: {risk_metrics.get('sharpe_ratio', 0):.2f}")
                print(f"  Beta: {risk_metrics.get('portfolio_beta', 1):.2f}")
                
        elif choice == '12':
            hours = int(input("View alerts for last N hours (default 24): ") or "24")
            system.view_alerts(hours)
            
        elif choice == '13':
            sub_menu = True
            while sub_menu:
                print(f"\n{Fore.YELLOW}System Controls:{Style.RESET_ALL}")
                print(f"Emergency stop: {'ACTIVE' if system.failsafe.emergency_stop else 'Inactive'}")
                print("\n1. Activate emergency stop")
                print("2. Deactivate emergency stop")
                print("3. View circuit breaker settings")
                print("4. Check system status")
                print("5. View/modify risk parameters")
                print("0. Back")
                
                sub_choice = input("Choice: ")
                
                if sub_choice == '1':
                    reason = input("Reason for emergency stop: ")
                    system.failsafe.activate_emergency_stop(reason)
                elif sub_choice == '2':
                    confirm = input("Deactivate emergency stop? (yes/no): ")
                    if confirm.lower() == 'yes':
                        system.failsafe.deactivate_emergency_stop()
                elif sub_choice == '3':
                    print("\nCircuit Breaker Settings:")
                    for key, value in system.failsafe.circuit_breakers.items():
                        print(f"  {key}: {value}")
                elif sub_choice == '4':
                    ok, issues = system.failsafe.check_all_systems()
                    status_color = Fore.GREEN if ok else Fore.RED
                    print(f"\nSystem status: {status_color}{'OK' if ok else 'ISSUES'}{Style.RESET_ALL}")
                    if not ok:
                        for issue in issues:
                            print(f"  • {issue}")
                elif sub_choice == '5':
                    print("\nCurrent Risk Parameters:")
                    for key, value in system.risk_params.items():
                        print(f"  {key}: {value}")
                elif sub_choice == '0':
                    sub_menu = False
                    
        elif choice == '14':
            sub_menu = True
            while sub_menu:
                print(f"\n{Fore.YELLOW}Database Maintenance:{Style.RESET_ALL}")
                print("1. Backup database")
                print("2. Verify integrity")
                print("3. View recent alerts")
                print("4. View trade history")
                print("5. Clear old data")
                print("6. Export data")
                print("0. Back")
                
                sub_choice = input("Choice: ")
                
                if sub_choice == '1':
                    if system.db.backup_database():
                        print(f"{Fore.GREEN}Backup created successfully{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}Backup failed{Style.RESET_ALL}")
                elif sub_choice == '2':
                    if system.db.verify_integrity():
                        print(f"{Fore.GREEN}Database integrity OK{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}Database integrity check failed!{Style.RESET_ALL}")
                elif sub_choice == '3':
                    with system.db.get_connection() as conn:
                        alerts = conn.execute("""
                            SELECT * FROM alerts 
                            WHERE timestamp > datetime('now', '-1 day')
                            ORDER BY timestamp DESC
                            LIMIT 10
                        """).fetchall()
                        
                        if alerts:
                            print("\nRecent alerts:")
                            for alert in alerts:
                                urgency_color = Fore.RED if alert['urgency'] == 'high' else Fore.YELLOW
                                print(f"{alert['timestamp'][:16]} - {urgency_color}{alert['alert_type']}{Style.RESET_ALL} - {alert['symbol']}: {alert['message']}")
                        else:
                            print("No recent alerts")
                elif sub_choice == '4':
                    with system.db.get_connection() as conn:
                        trades = conn.execute("""
                            SELECT * FROM trade_history 
                            ORDER BY trade_date DESC
                            LIMIT 10
                        """).fetchall()
                        
                        if trades:
                            print("\nRecent trades:")
                            for trade in trades:
                                action_color = Fore.GREEN if trade['action'] == 'BUY' else Fore.RED
                                pnl_color = Fore.GREEN if trade['pnl'] > 0 else Fore.RED if trade['pnl'] < 0 else Fore.WHITE
                                print(f"{trade['trade_date']} - {action_color}{trade['action']}{Style.RESET_ALL} {trade['symbol']} "
                                      f"{trade['quantity']} @ {trade['price']:.2f}")
                                if trade['pnl'] is not None:
                                    print(f"  P&L: {pnl_color}{trade['pnl']:+.0f} ({trade['return_pct']:+.1f}%){Style.RESET_ALL}")
                        else:
                            print("No trade history")
                elif sub_choice == '5':
                    confirm = input("Clear data older than 30 days? (yes/no): ")
                    if confirm.lower() == 'yes':
                        with system.db.get_connection() as conn:
                            conn.execute("DELETE FROM data_quality_log WHERE check_timestamp < datetime('now', '-30 days')")
                            conn.execute("DELETE FROM alerts WHERE timestamp < datetime('now', '-30 days')")
                            conn.commit()
                        print("Old data cleared")
                elif sub_choice == '6':
                    # Export positions to CSV
                    import csv
                    with open('positions_export.csv', 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['Symbol', 'Quantity', 'Buy Price', 'Buy Date', 'Cost Basis'])
                        for symbol, pos in system.positions.items():
                            writer.writerow([symbol, pos['quantity'], pos['buy_price'], 
                                           pos['buy_date'], pos['cost_basis']])
                    print(f"{Fore.GREEN}Positions exported to positions_export.csv{Style.RESET_ALL}")
                elif sub_choice == '0':
                    sub_menu = False
                    
        elif choice == '15':
            print("Retraining ML model with enhanced features...")
            success = system.retrain_ml_model_enhanced()
            if success:
                print(f"{Fore.GREEN}Model retrained successfully!{Style.RESET_ALL}")
                # Test the new model diversity
                system._analyze_model_performance()
            else:
                print(f"{Fore.RED}Model training failed{Style.RESET_ALL}")
                
        elif choice == '16':
            n_symbols = int(input("Number of symbols to analyze (default 20): ") or "20")
            system.analyze_signal_weakness(n_symbols)
            
        elif choice == '17':
            sub_menu = True
            while sub_menu:
                print(f"\n{Fore.YELLOW}Paper Trading:{Style.RESET_ALL}")
                print("1. Execute paper trade")
                print("2. View paper positions")
                print("3. View paper trading report")
                print("4. Reset paper trading")
                print("0. Back")
                
                sub_choice = input("Choice: ")
                
                if sub_choice == '1':
                    symbol = input("Symbol: ").upper()
                    action = input("Action (BUY/SELL): ").upper()
                    quantity = int(input("Quantity: "))
                    price = float(input("Price: "))
                    
                    # Get current signals for context
                    signals = system.generate_enhanced_signals(symbol)
                    
                    result = system.execute_paper_trade(
                        symbol=symbol,
                        action=action,
                        quantity=quantity,
                        price=price,
                        signal_strength=signals['strength'],
                        signal_reason=signals['signal']
                    )
                    
                    if not result['success']:
                        print(f"{Fore.RED}Paper trade failed: {result['reason']}{Style.RESET_ALL}")
                
                elif sub_choice == '2':
                    if system.paper_trading.paper_positions:
                        print("\nPaper positions:")
                        for symbol, pos in system.paper_trading.paper_positions.items():
                            pnl_pct = (pos.get('last_price', pos['avg_price']) / pos['avg_price'] - 1) * 100
                            pnl_color = Fore.GREEN if pnl_pct > 0 else Fore.RED
                            print(f"  {symbol}: {pos['quantity']} @ {pos['avg_price']:.2f} "
                                  f"({pnl_color}{pnl_pct:+.1f}%{Style.RESET_ALL})")
                    else:
                        print("No paper positions")
                
                elif sub_choice == '3':
                    system.paper_trading.display_report()
                
                elif sub_choice == '4':
                    confirm = input("Reset paper trading? This will clear all paper trades (yes/no): ")
                    if confirm.lower() == 'yes':
                        system.paper_trading = PaperTradingMode(system)
                        print("Paper trading reset")
                
                elif sub_choice == '0':
                    sub_menu = False
        
        elif choice == '18':
            passed, results = system.run_pre_market_checks()
            if passed:
                print(f"\n{Fore.GREEN}All pre-market checks passed!{Style.RESET_ALL}")
            else:
                print(f"\n{Fore.RED}Some pre-market checks failed - review before trading{Style.RESET_ALL}")
        
        elif choice == '19':
            symbol = input("Symbol: ").upper()
            action = input("Action (BUY/SELL): ").upper()
            quantity = int(input("Quantity: "))
            price = float(input("Price: "))
            
            costs = system.estimate_trade_costs(symbol, action, quantity, price)
            
            print(f"\n{Fore.CYAN}=== TRANSACTION COST ESTIMATE ==={Style.RESET_ALL}")
            print(f"Order value: {quantity * price:,.2f}")
            print(f"Commission: {costs['commission']:.2f}")
            if action == 'SELL':
                print(f"Stamp duty: {costs['stamp_duty']:.2f}")
            print(f"Estimated slippage: {costs['slippage']:.2f} ({costs['slippage_rate']:.2%})")
            print(f"Total costs: {costs['total_costs']:.2f} ({costs['cost_rate']:.2%})")
            print(f"Estimated fill price: {costs['estimated_fill_price']:.2f}")
        
        elif choice == '20':
            system.display_ml_health()
        
        elif choice == '21':
            system.track_signal_performance()
        
        elif choice == '22':
            system.display_error_summary()
            
        elif choice == '0':
            if system.monitoring_active:
                system.stop_monitoring()
            print(f"\n{Fore.CYAN}Thank you for using Enhanced Manual Trading System{Style.RESET_ALL}")
            break
        
        else:
            print(f"{Fore.RED}Invalid choice{Style.RESET_ALL}")


if __name__ == "__main__":
    main()