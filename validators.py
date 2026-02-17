
"""
Data validation and failsafe management for the trading platform.
"""

import datetime
from datetime import timezone, timedelta
import json
import logging

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from .models import DataQualityCheck
from .database import DatabaseManager

_BEIJING_TZ = timezone(timedelta(hours=8))

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
            'last_reset': datetime.datetime.now(_BEIJING_TZ).date()
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
        except Exception:
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
        """Reset daily statistics if new trading day (Beijing time)"""
        beijing_today = datetime.datetime.now(_BEIJING_TZ).date()
        if self.daily_stats['last_reset'] < beijing_today:
            self.daily_stats = {
                'trades_today': 0,
                'loss_today': 0.0,
                'last_reset': beijing_today
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
