"""Versioned schema migrations for the local analytics database.

The schema is never treated as disposable: every structural change becomes a
new :class:`Migration` appended to :data:`MIGRATIONS`, applied in order and
recorded in ``schema_migrations``. Migrations use ``IF NOT EXISTS`` so a crash
between executing SQL and recording the version is safely recoverable.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from toknt.util import now_ms


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    description: str
    sql: str


_SCHEMA_MIGRATIONS_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    applied_at_ms INTEGER NOT NULL
) STRICT
"""

# Milestone 1: canonical schema. STRICT tables make the type discipline
# (None = not reported, ints stay ints) a database-level guarantee.
_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    kind TEXT NOT NULL,
    location TEXT NOT NULL,
    fingerprint TEXT,
    last_scan_at_ms INTEGER,
    last_scan_status TEXT,
    last_scan_note TEXT,
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS checkpoints (
    source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    stream TEXT NOT NULL,
    offset INTEGER NOT NULL DEFAULT 0,
    size INTEGER,
    mtime_ms INTEGER,
    fingerprint TEXT,
    updated_at_ms INTEGER NOT NULL,
    PRIMARY KEY (source_id, stream)
) STRICT;

CREATE TABLE IF NOT EXISTS scan_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    mode TEXT NOT NULL,
    started_at_ms INTEGER NOT NULL,
    finished_at_ms INTEGER,
    status TEXT NOT NULL DEFAULT 'running',
    events_inserted INTEGER NOT NULL DEFAULT 0,
    events_updated INTEGER NOT NULL DEFAULT 0,
    events_skipped INTEGER NOT NULL DEFAULT 0,
    errors INTEGER NOT NULL DEFAULT 0,
    notes TEXT
) STRICT;

CREATE TABLE IF NOT EXISTS sessions (
    agent TEXT NOT NULL,
    session_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    parent_session_id TEXT,
    project TEXT,
    directory TEXT,
    title TEXT,
    agent_mode TEXT,
    model TEXT,
    provider TEXT,
    time_created_ms INTEGER,
    time_updated_ms INTEGER,
    cost_reported REAL,
    quality TEXT NOT NULL DEFAULT 'ok',
    quality_notes TEXT,
    PRIMARY KEY (agent, session_id)
) STRICT;

CREATE TABLE IF NOT EXISTS usage_events (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_event_id TEXT,
    session_id TEXT,
    parent_session_id TEXT,
    project TEXT,
    directory TEXT,
    agent_mode TEXT,
    provider TEXT,
    model TEXT,
    ts_ms INTEGER NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    reasoning_tokens INTEGER,
    cache_read_tokens INTEGER,
    cache_write_tokens INTEGER,
    provider_total_tokens INTEGER,
    token_scheme TEXT NOT NULL DEFAULT 'unknown',
    cost_reported REAL,
    cost_currency TEXT,
    finish_reason TEXT,
    quality TEXT NOT NULL DEFAULT 'ok',
    quality_notes TEXT,
    source_event_type TEXT,
    ingested_at_ms INTEGER NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage_events (ts_ms);
CREATE INDEX IF NOT EXISTS idx_usage_agent_ts ON usage_events (agent, ts_ms);
CREATE INDEX IF NOT EXISTS idx_usage_model_ts ON usage_events (model, ts_ms);
CREATE INDEX IF NOT EXISTS idx_usage_session ON usage_events (agent, session_id);
CREATE INDEX IF NOT EXISTS idx_usage_project_ts ON usage_events (project, ts_ms);

CREATE TABLE IF NOT EXISTS tool_events (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_event_id TEXT,
    session_id TEXT,
    call_id TEXT,
    tool_name TEXT NOT NULL,
    tool_category TEXT NOT NULL DEFAULT 'other',
    ts_ms INTEGER NOT NULL,
    duration_ms INTEGER,
    outcome TEXT NOT NULL DEFAULT 'unknown',
    outcome_provenance TEXT NOT NULL DEFAULT 'absent',
    outcome_confidence REAL,
    model TEXT,
    agent_mode TEXT,
    quality TEXT NOT NULL DEFAULT 'ok',
    quality_notes TEXT,
    source_event_type TEXT,
    ingested_at_ms INTEGER NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_tool_ts ON tool_events (ts_ms);
CREATE INDEX IF NOT EXISTS idx_tool_name_ts ON tool_events (tool_name, ts_ms);
CREATE INDEX IF NOT EXISTS idx_tool_agent_ts ON tool_events (agent, ts_ms);
CREATE INDEX IF NOT EXISTS idx_tool_session ON tool_events (agent, session_id);
CREATE INDEX IF NOT EXISTS idx_tool_call ON tool_events (call_id);

CREATE TABLE IF NOT EXISTS quality_issues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent TEXT NOT NULL,
    source_id TEXT NOT NULL,
    code TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'warning',
    message TEXT NOT NULL,
    occurrences INTEGER NOT NULL DEFAULT 1,
    first_seen_ms INTEGER NOT NULL,
    last_seen_ms INTEGER NOT NULL,
    sample_ref TEXT,
    UNIQUE (agent, source_id, code)
) STRICT;
"""

_SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS pricing (
    model TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT '',
    currency TEXT NOT NULL DEFAULT 'USD',
    input_per_mtok REAL,
    output_per_mtok REAL,
    cache_read_per_mtok REAL,
    cache_write_per_mtok REAL,
    effective_from_ms INTEGER NOT NULL DEFAULT 0,
    note TEXT,
    PRIMARY KEY (model, provider, effective_from_ms)
) STRICT;
"""

#: v3 re-keys quality issues by sample reference so repeated scans of one
#: defect replace its count instead of accumulating drift, and distinct
#: affected files/rows are counted separately (existing rows are carried
#: over, merging duplicates that share a key).
_SCHEMA_V3 = """
CREATE TEMP TABLE IF NOT EXISTS quality_issues_v2 AS
    SELECT agent, source_id, code, severity, message, occurrences,
           first_seen_ms, last_seen_ms, COALESCE(sample_ref, '') AS sample_ref
    FROM quality_issues;
DROP TABLE IF EXISTS quality_issues;
CREATE TABLE quality_issues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent TEXT NOT NULL,
    source_id TEXT NOT NULL,
    code TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'warning',
    message TEXT NOT NULL,
    occurrences INTEGER NOT NULL DEFAULT 1,
    first_seen_ms INTEGER NOT NULL,
    last_seen_ms INTEGER NOT NULL,
    sample_ref TEXT NOT NULL DEFAULT '',
    UNIQUE (agent, source_id, code, sample_ref)
) STRICT;
INSERT INTO quality_issues (
    agent, source_id, code, severity, message, occurrences,
    first_seen_ms, last_seen_ms, sample_ref
)
    SELECT agent, source_id, code, MAX(severity), MAX(message),
           MAX(occurrences), MIN(first_seen_ms), MAX(last_seen_ms), sample_ref
    FROM temp.quality_issues_v2
    GROUP BY agent, source_id, code, sample_ref;
DROP TABLE IF EXISTS temp.quality_issues_v2;
"""

MIGRATIONS: tuple[Migration, ...] = (
    Migration(version=1, description="initial canonical schema", sql=_SCHEMA_V1),
    Migration(version=2, description="model pricing table", sql=_SCHEMA_V2),
    Migration(
        version=3,
        description="quality issues keyed per affected record, replace-on-rescan",
        sql=_SCHEMA_V3,
    ),
)


def applied_versions(conn: sqlite3.Connection) -> set[int]:
    conn.execute(_SCHEMA_MIGRATIONS_DDL)
    return {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}


def schema_version(conn: sqlite3.Connection) -> int:
    versions = applied_versions(conn)
    return max(versions, default=0)


def apply_migrations(
    conn: sqlite3.Connection,
    migrations: tuple[Migration, ...] = MIGRATIONS,
) -> list[int]:
    """Apply pending migrations in order; returns the versions applied.

    Each migration is committed individually, so a failure leaves the
    database at the last fully applied version.
    """
    applied = applied_versions(conn)
    newly_applied: list[int] = []
    for migration in sorted(migrations, key=lambda m: m.version):
        if migration.version in applied:
            continue
        conn.executescript(migration.sql)
        conn.execute(
            "INSERT INTO schema_migrations (version, description, applied_at_ms) VALUES (?, ?, ?)",
            (migration.version, migration.description, now_ms()),
        )
        conn.commit()
        newly_applied.append(migration.version)
    return newly_applied
