
"""
Database management for the trading platform.
Handles SQLite database operations, schema initialization, backups, and integrity checks.
"""

import logging
import shutil
import sqlite3
import threading
import time
from contextlib import contextmanager


class DatabaseManager:
    """Robust database management for trading data.

    Uses a per-thread connection cache and a reentrant lock so that:
    - The main thread and the monitor thread each get their own connection.
    - Multiple ``with db.get_connection()`` blocks in the same thread reuse
      the same connection (avoiding "database is locked" errors).
    """

    def __init__(self, db_path: str = 'trading_system.db'):
        self.db_path = db_path
        self.backup_path = f"{db_path}.backup"
        self.logger = logging.getLogger(__name__)

        # Thread-local storage for connections
        self._local = threading.local()
        self._lock = threading.RLock()

        # Cached integrity result (avoid running PRAGMA on every signal call)
        self._integrity_ok: bool = True
        self._integrity_checked_at: float = 0.0
        self._INTEGRITY_CACHE_SECONDS = 300  # re-check every 5 minutes

        self._init_database()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_thread_connection(self) -> sqlite3.Connection:
        """Return (or create) a connection for the current thread."""
        conn = getattr(self._local, 'conn', None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    @contextmanager
    def get_connection(self):
        """Get database connection with proper error handling.

        Yields the thread-local connection.  The connection is NOT closed at
        the end of each block -- it persists for the lifetime of the thread.
        Rollback is performed on error.
        """
        conn = self._get_thread_connection()
        try:
            yield conn
        except sqlite3.Error as e:
            conn.rollback()
            raise Exception(f"Database error: {e}") from e

    def close_connection(self):
        """Explicitly close the current thread's connection (e.g. on shutdown)."""
        conn = getattr(self._local, 'conn', None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_database(self):
        """Initialize database with proper schema"""
        with self.get_connection() as conn:
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

            conn.execute("""
                CREATE TABLE IF NOT EXISTS watchlist (
                    symbol TEXT PRIMARY KEY,
                    added_date DATE NOT NULL,
                    priority INTEGER DEFAULT 0,
                    notes TEXT
                )
            """)

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

            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

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

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def backup_database(self) -> bool:
        """Create database backup"""
        try:
            with self._lock:
                shutil.copy2(self.db_path, self.backup_path)
            return True
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            return False

    def verify_integrity(self) -> bool:
        """Verify database integrity (cached -- re-checks every 5 min)."""
        now = time.monotonic()
        if now - self._integrity_checked_at < self._INTEGRITY_CACHE_SECONDS:
            return self._integrity_ok

        try:
            with self.get_connection() as conn:
                result = conn.execute("PRAGMA integrity_check").fetchone()
                self._integrity_ok = result[0] == "ok"
        except Exception as e:
            self.logger.error(f"Integrity check failed: {e}")
            self._integrity_ok = False

        self._integrity_checked_at = now
        return self._integrity_ok

    def force_integrity_check(self) -> bool:
        """Force an immediate integrity check (bypasses cache)."""
        self._integrity_checked_at = 0.0
        return self.verify_integrity()
