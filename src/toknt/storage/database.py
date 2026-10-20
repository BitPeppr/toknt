"""SQLite connectivity for the local analytics index.

The application database is fully owned by toknt: it may use WAL mode and
write freely. (Source databases, by contrast, are only ever opened read-only
by adapters.) Connections enable foreign keys and a busy timeout so
concurrent CLI invocations degrade gracefully instead of failing instantly.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from toknt import config
from toknt.storage.migrations import apply_migrations

#: Tables reported by ``toknt doctor`` and tests.
KNOWN_TABLES: tuple[str, ...] = (
    "sources",
    "checkpoints",
    "scan_runs",
    "sessions",
    "usage_events",
    "tool_events",
    "quality_issues",
    "pricing",
)


def connect(db_path: Path) -> sqlite3.Connection:
    """Open (creating if needed) the analytics database at ``db_path``."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def open_database(db_path: Path | None = None) -> sqlite3.Connection:
    """Open the database at ``db_path`` (default: configured location) and
    bring the schema up to date."""
    conn = connect(db_path if db_path is not None else config.db_path())
    apply_migrations(conn)
    return conn


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Row count for every known table (0 for tables missing from old schemas)."""
    counts: dict[str, int] = {}
    for table in KNOWN_TABLES:
        try:
            counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except sqlite3.OperationalError:
            counts[table] = 0
    return counts
