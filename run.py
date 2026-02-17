#!/usr/bin/env python3
"""
Run the Enhanced Manual Trading System.
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import os

# Ensure the parent directory is in sys.path so 'trading_system' package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_system import main

if __name__ == "__main__":
    main()
