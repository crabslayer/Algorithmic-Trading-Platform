
"""
Adaptive signal threshold optimization based on market conditions and performance.
"""

import logging

import numpy as np
import pandas as pd
from typing import Dict

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
