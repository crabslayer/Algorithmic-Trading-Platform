
"""
Error handling system for the trading platform.
Contains custom exceptions, retry decorator, and the ErrorHandler class.
"""

import datetime
import functools
import logging
import time
import traceback
from typing import Any, Callable, Dict, List, Optional


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
    """Decorator for automatic retry with error handling.

    The decorator's own parameters (error_type, max_retries, critical) are
    passed to safe_execute via a dedicated ``_retry_opts`` dict so they can
    never collide with keyword arguments of the wrapped function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if hasattr(self, 'error_handler'):
                return self.error_handler.safe_execute(
                    func, self, *args,
                    _retry_opts={
                        'error_type': error_type,
                        'max_retries': max_retries,
                        'critical': critical,
                    },
                    **kwargs
                )
            else:
                return func(self, *args, **kwargs)
        return wrapper
    return decorator


class ErrorHandler:
    """Comprehensive error handling and recovery system"""

    def __init__(self, logger: logging.Logger, failsafe_manager=None):
        self.logger = logger
        self.failsafe = failsafe_manager
        self.error_counts: Dict[str, int] = {}
        self.error_history: List[Dict] = []
        self.recovery_strategies = {
            'data_fetch': self._recover_data_fetch,
            'calculation': self._recover_calculation,
            'database': self._recover_database,
            'network': self._recover_network,
            'execution': self._recover_execution
        }

    def safe_execute(self, func: Callable, *args,
                     _retry_opts: Optional[Dict] = None,
                     **kwargs) -> Any:
        """Safely execute a function with retry logic and error handling.

        Retry parameters are passed via the ``_retry_opts`` dict to avoid
        colliding with keyword arguments of the wrapped function.
        """
        opts = _retry_opts or {}
        error_type: str = opts.get('error_type', 'general')
        max_retries: int = opts.get('max_retries', 3)
        retry_delay: float = opts.get('retry_delay', 1.0)
        default_return: Any = opts.get('default_return', None)
        critical: bool = opts.get('critical', False)

        last_error = None

        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                last_error = e
                error_key = f"{func.__name__}_{error_type}"

                self.logger.error(
                    f"Error in {func.__name__} (attempt {attempt + 1}/{max_retries}): "
                    f"{type(e).__name__}: {str(e)}"
                )

                self._record_error(error_key, e)

                if self._should_stop_retrying(error_key, e):
                    break

                if error_type in self.recovery_strategies:
                    recovered = self.recovery_strategies[error_type](e, func, args, kwargs)
                    if recovered is not None:
                        return recovered

                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    self.logger.info(f"Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)

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

        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-500:]

    def _should_stop_retrying(self, error_key: str, error: Exception) -> bool:
        """Determine if we should stop retrying based on error pattern"""
        # These usually indicate logic/code errors, not transient failures
        non_retryable = (
            KeyError,
            ValueError,
            AttributeError,
            TypeError,
        )

        if isinstance(error, non_retryable):
            return True

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
    def _recover_data_fetch(self, error, func, args, kwargs) -> Optional[Any]:
        """Try alternative data sources or use cached data"""
        if "get_market_data" in func.__name__:
            self.logger.info("Attempting to use cached data...")
        return None

    def _recover_calculation(self, error, func, args, kwargs) -> Optional[Any]:
        """Use simplified calculations or default values"""
        if isinstance(error, (ValueError, ZeroDivisionError)):
            self.logger.info("Using default calculation values...")
            if "calculate_signals" in func.__name__ or "generate_enhanced_signals" in func.__name__:
                return {'signal': 'ERROR', 'strength': 0, 'reasons': ['Calculation error'], 'data_quality': 0}
        return None

    def _recover_database(self, error, func, args, kwargs) -> Optional[Any]:
        """Try database reconnection or use backup"""
        self.logger.info("Attempting database recovery...")
        return None

    def _recover_network(self, error, func, args, kwargs) -> Optional[Any]:
        """Handle network errors with fallback"""
        self.logger.info("Network error detected, using offline mode...")
        return None

    def _recover_execution(self, error, func, args, kwargs) -> Optional[Any]:
        """Handle trade execution errors"""
        self.logger.error("Trade execution failed - manual intervention may be required")
        return None

    def get_error_summary(self) -> Dict:
        """Get summary of recent errors"""
        summary: Dict[str, Any] = {
            'total_errors': sum(self.error_counts.values()),
            'error_types': {},
            'most_common': None,
            'recent_errors': []
        }

        for error in self.error_history[-100:]:
            error_type = error['type']
            summary['error_types'][error_type] = summary['error_types'].get(error_type, 0) + 1

        if self.error_counts:
            summary['most_common'] = max(self.error_counts.items(), key=lambda x: x[1])

        summary['recent_errors'] = [
            {
                'time': e['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'key': e['key'],
                'message': e['message']
            }
            for e in self.error_history[-5:]
        ]

        return summary
