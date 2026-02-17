
"""
ML Trading Engine for the trading platform.
Includes ensemble models, feature preparation, training, and prediction.
"""

import json
import logging
import os
import traceback

import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
from colorama import Fore, Style
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict

class ImprovedMLTradingEngine:
    """Enhanced ML engine with validation and ensemble methods"""
    
    def __init__(self, model_dir: str = 'ml_models'):
        self.model_dir = model_dir
        self.logger = logging.getLogger(__name__)
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
                'logistic': 'logistic_model.pkl'
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
        """Initialize new ML classification models.

        NOTE: These are placeholder instances.  The actual models (with proper
        class_weight) are created inside ``_train_with_validation`` once the
        training data distribution is known.  These exist only so that
        ``predict()`` can return a safe untrained result before training.
        """
        self.models = {
            'rf': RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_split=50,
                min_samples_leaf=20,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'gbm': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                min_samples_split=50,
                min_samples_leaf=20,
                subsample=0.8,
                random_state=42
            ),
            'logistic': LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0
            )
        }
        self.scalers['standard'] = StandardScaler()
        self.is_trained = False
        self.training_features = []
        self.ensemble_weights = {'rf': 0.4, 'gbm': 0.4, 'logistic': 0.2}
        self.logger.info("Initialized new ML classification models")
    
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
                except Exception:
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
                close = df['close'].ffill().bfill()
                high = df['high'].ffill().bfill()
                low = df['low'].ffill().bfill()
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
                    except Exception:
                        feature_dict['volatility_20d'] = 0.2
                        feature_dict['volatility_5d'] = 0.2
                    
                    # Technical indicators with comprehensive error handling
                    window_df = df.iloc[max(0, i-60):i+1]
                    window_close = window_df['close'].ffill().bfill()
                    
                    if len(window_close) < 10:
                        continue
                    
                    # RSI with multiple fallbacks
                    try:
                        rsi_14 = ta.rsi(window_close, length=14)
                        feature_dict['rsi_14'] = rsi_14.iloc[-1] if rsi_14 is not None and len(rsi_14) > 0 and not pd.isna(rsi_14.iloc[-1]) else 50
                    except Exception:
                        feature_dict['rsi_14'] = 50
                    
                    try:
                        rsi_7 = ta.rsi(window_close, length=7)
                        feature_dict['rsi_7'] = rsi_7.iloc[-1] if rsi_7 is not None and len(rsi_7) > 0 and not pd.isna(rsi_7.iloc[-1]) else 50
                    except Exception:
                        feature_dict['rsi_7'] = 50
                    
                    try:
                        rsi_28 = ta.rsi(window_close, length=28)
                        feature_dict['rsi_28'] = rsi_28.iloc[-1] if rsi_28 is not None and len(rsi_28) > 0 and not pd.isna(rsi_28.iloc[-1]) else 50
                    except Exception:
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
                        except Exception:
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
                    except Exception:
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
                    except Exception:
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
                    except Exception:
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
                    except Exception:
                        feature_dict['price_efficiency'] = 0.5
                    
                    # Trend strength with safety checks
                    try:
                        sma_20 = close.iloc[max(0, i-20):i].mean() if i >= 20 else current_close
                        sma_50 = close.iloc[max(0, i-50):i].mean() if i >= 50 else sma_20
                        
                        if sma_50 > 0:
                            feature_dict['trend_strength'] = float((sma_20 - sma_50) / sma_50)
                        else:
                            feature_dict['trend_strength'] = 0
                    except Exception:
                        feature_dict['trend_strength'] = 0
                    
                    # Symbol-specific features
                    try:
                        feature_dict['log_price'] = float(np.log(current_close)) if current_close > 0 else 0
                        feature_dict['symbol_hash'] = float(hash(symbol) % 100 / 100)
                    except Exception:
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
            from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report  # noqa
            
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
