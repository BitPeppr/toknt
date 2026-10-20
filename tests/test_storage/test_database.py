"""Database connectivity and configuration paths."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from toknt import config
from toknt.storage import database


def test_connect_creates_missing_parent_directories(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "deeper" / "toknt.db"
    conn = database.connect(db_path)
    try:
        assert db_path.exists()
    finally:
        conn.close()


def test_connection_pragmas(tmp_path: Path) -> None:
    conn = database.connect(tmp_path / "toknt.db")
    try:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        conn.close()


def test_open_database_migrates_and_returns_empty_canonical_tables(data_dir: Path) -> None:
    conn = database.open_database()
    try:
        counts = database.table_counts(conn)
    finally:
        conn.close()
    assert set(counts) == set(database.KNOWN_TABLES)
    assert all(count == 0 for count in counts.values())
    assert config.db_path() == data_dir / config.DB_FILENAME
    assert config.db_path().exists()


def test_data_dir_env_override(data_dir: Path) -> None:
    assert config.data_dir() == data_dir


def test_open_database_accepts_explicit_path(tmp_path: Path) -> None:
    explicit = tmp_path / "custom.db"
    conn = database.open_database(explicit)
    try:
        assert conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0] > 0
    finally:
        conn.close()


def test_foreign_keys_cascade_from_sources(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO sources (id, agent, kind, location, created_at_ms, updated_at_ms) "
        "VALUES ('s1', 'opencode', 'sqlite', '/tmp/db', 1, 1)"
    )
    conn.execute(
        "INSERT INTO checkpoints (source_id, stream, offset, updated_at_ms) "
        "VALUES ('s1', 'a', 0, 1)"
    )
    conn.execute("DELETE FROM sources WHERE id = 's1'")
    assert conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 0
