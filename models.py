
"""
Data structures for the trading platform.
"""

import datetime
from dataclasses import dataclass
from typing import List


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
