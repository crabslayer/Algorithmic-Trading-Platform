
"""
Pre-market checklist system for A-share trading.
"""

import datetime
from datetime import timezone, timedelta
import logging
import os
import socket
import time

from colorama import Fore, Style

from typing import Dict, Tuple

_BEIJING_TZ = timezone(timedelta(hours=8))

class PreMarketChecker:
    """Pre-market checklist system for A-share trading"""
    
    def __init__(self, system):
        self.system = system
        self.logger = logging.getLogger(__name__)
        
    def pre_market_checklist(self) -> Tuple[bool, Dict]:
        """Run comprehensive pre-market checks before trading"""
        print(f"\n{Fore.CYAN}=== PRE-MARKET CHECKLIST ==={Style.RESET_ALL}")
        print(f"Time (Beijing): {datetime.datetime.now(_BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S')}")
        
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
        """Check if market data is fresh (with timeout protection)"""
        import concurrent.futures
        
        def _fetch_check():
            test_symbols = ['000001', '000002', '600519']
            for symbol in test_symbols:
                df = self.system.get_market_data_cached(symbol, days=30)
                if df is not None and len(df) > 5:
                    latest_date = df.index[-1]
                    today = datetime.datetime.now(_BEIJING_TZ).date()
                    days_diff = (today - latest_date.date()).days
                    if days_diff <= 10:
                        return True, f"Latest data from {latest_date.date()} for {symbol}"
            return False, "Cannot fetch sufficient market data from any test symbol"
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_fetch_check)
                return future.result(timeout=30)  # 30 second timeout
        except concurrent.futures.TimeoutError:
            return False, "Data freshness check timed out (30s) - possible network issue"
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
        """Check if today (Beijing time) is an A-share trading day"""
        # Use Beijing time — user is in US Eastern, local date may differ
        beijing_now = datetime.datetime.now(_BEIJING_TZ)
        today = beijing_now.date()
        weekday = today.weekday()
        
        # Weekend check
        if weekday >= 5:
            return False, f"Weekend (Beijing) - {today.strftime('%A %Y-%m-%d')}"
        
        # Official A-share market holidays (weekday closures only)
        # Source: CSRC / CFFEX / SZSE official announcements
        CN_HOLIDAYS = {
            # ---- 2025 ----
            # New Year's Day
            datetime.date(2025, 1, 1),
            # Spring Festival
            datetime.date(2025, 1, 28), datetime.date(2025, 1, 29),
            datetime.date(2025, 1, 30), datetime.date(2025, 1, 31),
            datetime.date(2025, 2, 3),  datetime.date(2025, 2, 4),
            # Qingming Festival
            datetime.date(2025, 4, 4),
            # Labor Day
            datetime.date(2025, 5, 1),  datetime.date(2025, 5, 2),
            datetime.date(2025, 5, 5),
            # Dragon Boat Festival
            datetime.date(2025, 6, 2),
            # National Day + Mid-Autumn Festival
            datetime.date(2025, 10, 1), datetime.date(2025, 10, 2),
            datetime.date(2025, 10, 3), datetime.date(2025, 10, 6),
            datetime.date(2025, 10, 7), datetime.date(2025, 10, 8),

            # ---- 2026 ----
            # New Year's Day
            datetime.date(2026, 1, 1),  datetime.date(2026, 1, 2),
            # Spring Festival
            datetime.date(2026, 2, 16), datetime.date(2026, 2, 17),
            datetime.date(2026, 2, 18), datetime.date(2026, 2, 19),
            datetime.date(2026, 2, 20), datetime.date(2026, 2, 23),
            # Qingming Festival
            datetime.date(2026, 4, 6),
            # Labor Day
            datetime.date(2026, 5, 1),  datetime.date(2026, 5, 4),
            datetime.date(2026, 5, 5),
            # Dragon Boat Festival
            datetime.date(2026, 6, 19),
            # Mid-Autumn Festival
            datetime.date(2026, 9, 25),
            # National Day
            datetime.date(2026, 10, 1), datetime.date(2026, 10, 2),
            datetime.date(2026, 10, 5), datetime.date(2026, 10, 6),
            datetime.date(2026, 10, 7),
        }
        
        # Holiday name lookup for friendly messages
        HOLIDAY_NAMES = {
            (1, 1): "元旦 New Year's Day", (1, 2): "元旦 New Year's Day",
            (1, 28): "春节 Spring Festival", (1, 29): "春节 Spring Festival",
            (1, 30): "春节 Spring Festival", (1, 31): "春节 Spring Festival",
            (2, 3): "春节 Spring Festival", (2, 4): "春节 Spring Festival",
            (2, 16): "春节 Spring Festival", (2, 17): "春节 Spring Festival",
            (2, 18): "春节 Spring Festival", (2, 19): "春节 Spring Festival",
            (2, 20): "春节 Spring Festival", (2, 23): "春节 Spring Festival",
            (4, 4): "清明节 Qingming Festival", (4, 6): "清明节 Qingming Festival",
            (5, 1): "劳动节 Labor Day", (5, 2): "劳动节 Labor Day",
            (5, 4): "劳动节 Labor Day", (5, 5): "劳动节 Labor Day",
            (6, 2): "端午节 Dragon Boat Festival",
            (6, 19): "端午节 Dragon Boat Festival",
            (9, 25): "中秋节 Mid-Autumn Festival",
            (10, 1): "国庆节 National Day", (10, 2): "国庆节 National Day",
            (10, 3): "国庆节 National Day", (10, 5): "国庆节 National Day",
            (10, 6): "国庆节 National Day", (10, 7): "国庆节 National Day",
            (10, 8): "国庆节 National Day",
        }
        
        if today in CN_HOLIDAYS:
            key = (today.month, today.day)
            name = HOLIDAY_NAMES.get(key, "Public Holiday")
            return False, f"Market closed — {name} ({today.strftime('%Y-%m-%d')})"
        
        # Check if holiday data covers this year
        holiday_years = {d.year for d in CN_HOLIDAYS}
        if today.year not in holiday_years:
            return True, (f"Trading day - {today.strftime('%Y-%m-%d %A')} "
                          f"(⚠ no holiday calendar for {today.year}, update needed)")
        
        return True, f"Trading day - {today.strftime('%Y-%m-%d %A')} (Beijing time)"
    
    def _check_market_hours(self) -> Tuple[bool, str]:
        """Check if within or near market hours (Beijing time)"""
        now = datetime.datetime.now(_BEIJING_TZ)
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
        """Check data provider availability (with timeout + retry)"""
        import concurrent.futures

        def _probe():
            import akshare as ak
            # Use a lighter endpoint first (individual stock history is more
            # reliable from overseas than the full spot table).
            for attempt in range(2):
                try:
                    test_df = ak.stock_zh_a_hist(
                        symbol="000001", period="daily",
                        start_date=(datetime.datetime.now(_BEIJING_TZ)
                                    - timedelta(days=10)).strftime('%Y%m%d'),
                        end_date=datetime.datetime.now(_BEIJING_TZ).strftime('%Y%m%d'),
                        adjust="qfq"
                    )
                    if test_df is not None and not test_df.empty:
                        return True, f"Data provider OK ({len(test_df)} rows from 000001)"
                except Exception:
                    time.sleep(2)  # brief pause before retry
            # Fallback: try the spot table
            try:
                test_df = ak.stock_zh_a_spot_em()
                if test_df is not None and not test_df.empty:
                    return True, "Data providers responding (spot table)"
            except Exception as e:
                return False, f"Data provider error: {str(e)[:80]}"
            return False, "Data provider returned empty data"

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_probe)
                return future.result(timeout=30)
        except concurrent.futures.TimeoutError:
            return False, "Data provider check timed out (30s) - possible overseas network issue"
        except Exception as e:
            return False, f"Data provider check error: {str(e)[:80]}"
    
    def _check_ml_status(self) -> Tuple[bool, str]:
        """Check ML model status"""
        if not self.system.ml_engine.is_trained:
            return False, "ML model not trained"
        
        # Check model age (if implemented)
        return True, "ML model ready"
