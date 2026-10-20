"""Schema migrations: applied exactly once, recoverable, and STRICT."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from toknt.storage import database
from toknt.storage.migrations import MIGRATIONS, apply_migrations, schema_version


@pytest.fixture
def raw_conn(tmp_path: Path) -> sqlite3.Connection:
    connection = database.connect(tmp_path / "raw.db")
    yield connection
    connection.close()


class TestApplyingMigrations:
    def test_fresh_database_applies_every_migration_in_order(
        self, raw_conn: sqlite3.Connection
    ) -> None:
        applied = apply_migrations(raw_conn)
        assert applied == [migration.version for migration in MIGRATIONS]
        assert schema_version(raw_conn) == max(m.version for m in MIGRATIONS)

    def test_reapplying_is_a_noop(self, raw_conn: sqlite3.Connection) -> None:
        apply_migrations(raw_conn)
        assert apply_migrations(raw_conn) == []

    def test_recovers_when_the_version_row_is_missing(
        self, raw_conn: sqlite3.Connection
    ) -> None:
        # Simulates a crash between executing migration SQL and recording it:
        # the SQL is idempotent, so re-running must succeed.
        apply_migrations(raw_conn)
        raw_conn.execute("DELETE FROM schema_migrations")
        raw_conn.commit()
        applied = apply_migrations(raw_conn)
        assert applied == [migration.version for migration in MIGRATIONS]
        assert schema_version(raw_conn) == max(m.version for m in MIGRATIONS)

    def test_creates_every_canonical_table(self, raw_conn: sqlite3.Connection) -> None:
        apply_migrations(raw_conn)
        rows = raw_conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        names = {row[0] for row in rows}
        expected = set(database.KNOWN_TABLES) | {"schema_migrations", "sqlite_sequence"}
        assert expected <= names

    def test_creates_query_indexes(self, raw_conn: sqlite3.Connection) -> None:
        apply_migrations(raw_conn)
        rows = raw_conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'").fetchall()
        names = {row[0] for row in rows}
        assert {
            "idx_usage_ts",
            "idx_usage_model_ts",
            "idx_tool_name_ts",
            "idx_tool_call",
        } <= names


class TestStrictTyping:
    def test_strict_tables_reject_mismatched_types(self, raw_conn: sqlite3.Connection) -> None:
        apply_migrations(raw_conn)
        with pytest.raises(sqlite3.IntegrityError):
            raw_conn.execute(
                "INSERT INTO sources (id, agent, kind, location, created_at_ms, updated_at_ms) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("s", "opencode", "sqlite", "/tmp/x", "not-a-number", 1),
            )

    def test_not_null_columns_are_enforced(self, raw_conn: sqlite3.Connection) -> None:
        apply_migrations(raw_conn)
        with pytest.raises(sqlite3.IntegrityError):
            raw_conn.execute(
                "INSERT INTO usage_events (id, agent, source_id, ts_ms, ingested_at_ms) "
                "VALUES (?, ?, ?, ?, ?)",
                ("e1", "opencode", "opencode:db", None, 1),
            )
