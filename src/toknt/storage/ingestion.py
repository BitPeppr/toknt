"""Canonical-record persistence: sources, checkpoints, scan runs and upserts.

Every write is idempotent. Events are keyed by their stable canonical id, so
repeated scans, overlapping full rescan/imports and duplicated source records
converge on the same rows and never inflate totals.

Storage deliberately does not reinterpret adapter output: quality notes are
stored exactly as the adapter produced them (adapters are expected to attach
``UsageEvent.consistency_notes()`` observations themselves).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from toknt.models.events import Agent, SessionRecord, ToolEvent, UsageEvent
from toknt.util import now_ms

_CHUNK = 500


@dataclass(frozen=True, slots=True)
class UpsertStats:
    inserted: int = 0
    updated: int = 0

    @property
    def total(self) -> int:
        return self.inserted + self.updated


@dataclass(frozen=True, slots=True)
class Checkpoint:
    source_id: str
    stream: str
    offset: int
    size: int | None
    mtime_ms: int | None
    fingerprint: str | None
    updated_at_ms: int


_USAGE_COLUMNS: tuple[str, ...] = (
    "id",
    "agent",
    "source_id",
    "source_event_id",
    "session_id",
    "parent_session_id",
    "project",
    "directory",
    "agent_mode",
    "provider",
    "model",
    "ts_ms",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "provider_total_tokens",
    "token_scheme",
    "cost_reported",
    "cost_currency",
    "finish_reason",
    "quality",
    "quality_notes",
    "source_event_type",
    "ingested_at_ms",
)

_TOOL_COLUMNS: tuple[str, ...] = (
    "id",
    "agent",
    "source_id",
    "source_event_id",
    "session_id",
    "call_id",
    "tool_name",
    "tool_category",
    "ts_ms",
    "duration_ms",
    "outcome",
    "outcome_provenance",
    "outcome_confidence",
    "model",
    "agent_mode",
    "quality",
    "quality_notes",
    "source_event_type",
    "ingested_at_ms",
)

_SESSION_COLUMNS: tuple[str, ...] = (
    "agent",
    "session_id",
    "source_id",
    "parent_session_id",
    "project",
    "directory",
    "title",
    "agent_mode",
    "model",
    "provider",
    "time_created_ms",
    "time_updated_ms",
    "cost_reported",
    "quality",
    "quality_notes",
)


def _notes_json(notes: Sequence[str]) -> str:
    return json.dumps(list(notes))


def _usage_row(event: UsageEvent, ingested_at_ms: int) -> tuple:
    tokens = event.tokens
    return (
        event.id,
        str(event.agent),
        event.source_id,
        event.source_event_id,
        event.session_id,
        event.parent_session_id,
        event.project,
        event.directory,
        event.agent_mode,
        event.provider,
        event.model,
        event.ts_ms,
        tokens.input,
        tokens.output,
        tokens.reasoning,
        tokens.cache_read,
        tokens.cache_write,
        event.provider_total_tokens,
        str(event.token_scheme),
        event.cost_reported,
        event.cost_currency,
        event.finish_reason,
        str(event.quality),
        _notes_json(event.quality_notes),
        event.source_event_type,
        ingested_at_ms,
    )


def _tool_row(event: ToolEvent, ingested_at_ms: int) -> tuple:
    return (
        event.id,
        str(event.agent),
        event.source_id,
        event.source_event_id,
        event.session_id,
        event.call_id,
        event.tool_name,
        str(event.tool_category),
        event.ts_ms,
        event.duration_ms,
        str(event.outcome),
        str(event.outcome_provenance),
        event.outcome_confidence,
        event.model,
        event.agent_mode,
        str(event.quality),
        _notes_json(event.quality_notes),
        event.source_event_type,
        ingested_at_ms,
    )


def _session_row(record: SessionRecord) -> tuple:
    return (
        str(record.agent),
        record.session_id,
        record.source_id,
        record.parent_session_id,
        record.project,
        record.directory,
        record.title,
        record.agent_mode,
        record.model,
        record.provider,
        record.time_created_ms,
        record.time_updated_ms,
        record.cost_reported,
        str(record.quality),
        _notes_json(record.quality_notes),
    )


def _existing_keys(
    conn: sqlite3.Connection,
    table: str,
    key_columns: tuple[str, ...],
    keys: Iterable[tuple],
) -> set[tuple]:
    key_list = list(keys)
    if not key_list:
        return set()
    columns = ", ".join(key_columns)
    width = len(key_columns)
    found: set[tuple] = set()
    for start in range(0, len(key_list), _CHUNK):
        chunk = key_list[start : start + _CHUNK]
        row_placeholder = "(" + ", ".join("?" * width) + ")"
        placeholders = ", ".join(row_placeholder for _ in chunk)
        sql = f"SELECT {columns} FROM {table} WHERE ({columns}) IN (VALUES {placeholders})"
        params = [value for key in chunk for value in key]
        found.update(tuple(row) for row in conn.execute(sql, params))
    return found


def _upsert(
    conn: sqlite3.Connection,
    table: str,
    columns: tuple[str, ...],
    conflict_columns: tuple[str, ...],
    rows: Sequence[tuple],
    *,
    min_columns: Sequence[str] = (),
    max_columns: Sequence[str] = (),
) -> UpsertStats:
    if not rows:
        return UpsertStats()
    key_indexes = [columns.index(name) for name in conflict_columns]
    deduped: dict[tuple, tuple] = {}
    for row in rows:
        deduped[tuple(row[i] for i in key_indexes)] = row
    existing = _existing_keys(conn, table, conflict_columns, deduped.keys())

    min_set = set(min_columns)
    max_set = set(max_columns)
    _merged_assignments: dict[str, str] = {}
    for name in set(min_set) | set(max_set):
        _merged_assignments[name] = (
            f"{name} = "
            f"({'min' if name in min_set else 'max'}("
            f"COALESCE({table}.{name}, excluded.{name}),"
            f" COALESCE(excluded.{name}, {table}.{name})))"
        )
    assignments = ", ".join(
        _merged_assignments.get(name, f"{name} = excluded.{name}")
        for name in columns
        if name not in conflict_columns
    )
    placeholders = ", ".join("?" * len(columns))
    conflict = ", ".join(conflict_columns)
    sql = (
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT({conflict}) DO UPDATE SET {assignments}"
    )
    conn.executemany(sql, list(deduped.values()))
    conn.commit()

    inserted = sum(1 for key in deduped if key not in existing)
    return UpsertStats(inserted=inserted, updated=len(deduped) - inserted)


def upsert_usage_events(
    conn: sqlite3.Connection,
    events: Iterable[UsageEvent],
    *,
    ingested_at_ms: int | None = None,
) -> UpsertStats:
    written_at = ingested_at_ms if ingested_at_ms is not None else now_ms()
    rows = [_usage_row(event, written_at) for event in events]
    return _upsert(conn, "usage_events", _USAGE_COLUMNS, ("id",), rows)


def upsert_tool_events(
    conn: sqlite3.Connection,
    events: Iterable[ToolEvent],
    *,
    ingested_at_ms: int | None = None,
) -> UpsertStats:
    written_at = ingested_at_ms if ingested_at_ms is not None else now_ms()
    rows = [_tool_row(event, written_at) for event in events]
    return _upsert(conn, "tool_events", _TOOL_COLUMNS, ("id",), rows)


def upsert_sessions(conn: sqlite3.Connection, records: Iterable[SessionRecord]) -> UpsertStats:
    """Session cache rows.

    ``time_created_ms`` never gets later and ``time_updated_ms`` never gets
    earlier: a later scan window that contains no activity for a session must
    not erase what earlier scans learned about when it started or was last
    active.
    """
    rows = [_session_row(record) for record in records]
    return _upsert(
        conn,
        "sessions",
        _SESSION_COLUMNS,
        ("agent", "session_id"),
        rows,
        min_columns=("time_created_ms",),
        max_columns=("time_updated_ms",),
    )


def reconcile_source_events(
    conn: sqlite3.Connection,
    *,
    source_id: str,
    table: str,
    seen_ids: Iterable[str],
    key_column: str = "id",
) -> int:
    """Delete canonical rows that no longer exist in the source.

    Only call this after a *full* scan has re-read the entire source: the
    seen-id set must be authoritative, otherwise valid history would be
    deleted. Returns the number of deleted rows.
    """
    conn.execute("DROP TABLE IF EXISTS temp.toknt_seen")
    conn.execute("CREATE TEMP TABLE toknt_seen (key TEXT PRIMARY KEY)")
    ids = list(dict.fromkeys(seen_ids))
    for start in range(0, len(ids), _CHUNK):
        chunk = ids[start : start + _CHUNK]
        conn.executemany(
            "INSERT OR IGNORE INTO temp.toknt_seen (key) VALUES (?)",
            [(item,) for item in chunk],
        )
    cursor = conn.execute(
        f"DELETE FROM {table} WHERE source_id = ? "
        f"AND {key_column} NOT IN (SELECT key FROM temp.toknt_seen)",
        (source_id,),
    )
    deleted = cursor.rowcount if cursor.rowcount > 0 else 0
    conn.execute("DROP TABLE IF EXISTS temp.toknt_seen")
    conn.commit()
    return deleted


def delete_source_events(
    conn: sqlite3.Connection,
    *,
    source_id: str,
    table: str,
    reference_prefix: str,
) -> int:
    """Delete canonical rows whose ``source_event_id`` starts with a prefix.

    Adapters use this only when the source itself invalidates previously
    ingested rows (e.g. a file switches to a richer storage format and the
    derived counters of the old format no longer apply). Keep prefixes
    narrow: every matching row is deleted. Returns the number of rows removed.
    """
    escaped = (
        reference_prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    )
    cursor = conn.execute(
        f"DELETE FROM {table} WHERE source_id = ? "  # noqa: S608 - internal table names
        "AND source_event_id LIKE ? ESCAPE '\\'",
        (source_id, escaped + "%"),
    )
    deleted = cursor.rowcount if cursor.rowcount > 0 else 0
    conn.commit()
    return deleted


def ensure_source(
    conn: sqlite3.Connection,
    *,
    source_id: str,
    agent: Agent | str,
    kind: str,
    location: str,
    fingerprint: str | None = None,
    status: str = "ok",
    note: str | None = None,
) -> None:
    """Register (or refresh) a discovered source and its last-scan status."""
    now = now_ms()
    conn.execute(
        """
        INSERT INTO sources (
            id, agent, kind, location, fingerprint,
            last_scan_at_ms, last_scan_status, last_scan_note,
            created_at_ms, updated_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            agent = excluded.agent,
            kind = excluded.kind,
            location = excluded.location,
            fingerprint = COALESCE(excluded.fingerprint, sources.fingerprint),
            last_scan_at_ms = COALESCE(excluded.last_scan_at_ms, sources.last_scan_at_ms),
            last_scan_status = excluded.last_scan_status,
            last_scan_note = excluded.last_scan_note,
            updated_at_ms = excluded.updated_at_ms
        """,
        (source_id, str(agent), kind, location, fingerprint, now, status, note, now, now),
    )
    conn.commit()


def save_checkpoint(
    conn: sqlite3.Connection,
    source_id: str,
    stream: str,
    *,
    offset: int = 0,
    size: int | None = None,
    mtime_ms: int | None = None,
    fingerprint: str | None = None,
) -> None:
    """Persist how far a source stream (file, database, ...) has been processed."""
    conn.execute(
        """
        INSERT INTO checkpoints
            (source_id, stream, offset, size, mtime_ms, fingerprint, updated_at_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_id, stream) DO UPDATE SET
            offset = excluded.offset,
            size = excluded.size,
            mtime_ms = excluded.mtime_ms,
            fingerprint = excluded.fingerprint,
            updated_at_ms = excluded.updated_at_ms
        """,
        (source_id, stream, offset, size, mtime_ms, fingerprint, now_ms()),
    )
    conn.commit()


def load_checkpoint(conn: sqlite3.Connection, source_id: str, stream: str) -> Checkpoint | None:
    row = conn.execute(
        """
        SELECT source_id, stream, offset, size, mtime_ms, fingerprint, updated_at_ms
        FROM checkpoints WHERE source_id = ? AND stream = ?
        """,
        (source_id, stream),
    ).fetchone()
    if row is None:
        return None
    return Checkpoint(**dict(row))


def start_scan_run(conn: sqlite3.Connection, source_id: str, mode: str) -> int:
    cursor = conn.execute(
        "INSERT INTO scan_runs (source_id, mode, started_at_ms, status) "
        "VALUES (?, ?, ?, 'running')",
        (source_id, mode, now_ms()),
    )
    conn.commit()
    assert cursor.lastrowid is not None
    return cursor.lastrowid


def finish_scan_run(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    status: str,
    inserted: int = 0,
    updated: int = 0,
    skipped: int = 0,
    errors: int = 0,
    notes: str | None = None,
) -> None:
    conn.execute(
        """
        UPDATE scan_runs SET
            status = ?, finished_at_ms = ?, events_inserted = ?, events_updated = ?,
            events_skipped = ?, errors = ?, notes = ?
        WHERE id = ?
        """,
        (status, now_ms(), inserted, updated, skipped, errors, notes, run_id),
    )
    conn.commit()


def record_quality_issue(
    conn: sqlite3.Connection,
    *,
    agent: Agent | str,
    source_id: str,
    code: str,
    message: str,
    severity: str = "warning",
    occurrences: int = 1,
    sample_ref: str | None = None,
) -> None:
    """Aggregate a data-quality problem so ``toknt doctor`` can report it.

    Rows are keyed by (agent, source, code, ``sample_ref``) and re-observing
    the same problem replaces the count instead of accumulating it, so
    repeated scans of the same defect never inflate the numbers. Distinct
    sample refs (different files or rows) produce separate rows, which the
    summary adds up. The counts describe the source as last observed; a
    full rescan clears and re-emits them.

    ``code`` is a stable machine identifier (e.g. ``unparsable_row``);
    ``sample_ref`` should identify the affected record (file:line, row id) —
    never record content, prompts or tool arguments.
    """
    now = now_ms()
    occurrences = max(1, occurrences)
    sample_ref = sample_ref or ""
    conn.execute(
        """
        INSERT INTO quality_issues (
            agent, source_id, code, severity, message,
            occurrences, first_seen_ms, last_seen_ms, sample_ref
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(agent, source_id, code, sample_ref) DO UPDATE SET
            severity = excluded.severity,
            message = excluded.message,
            occurrences = MAX(quality_issues.occurrences, excluded.occurrences),
            last_seen_ms = excluded.last_seen_ms
        """,
        (
            str(agent),
            source_id,
            code,
            severity,
            message,
            occurrences,
            now,
            now,
            sample_ref,
        ),
    )
    conn.commit()


def clear_quality_issues(conn: sqlite3.Connection, *, source_id: str) -> int:
    """Forget a source's quality issues; a full rescan re-emits what remains."""
    cursor = conn.execute("DELETE FROM quality_issues WHERE source_id = ?", (source_id,))
    deleted = cursor.rowcount if cursor.rowcount > 0 else 0
    conn.commit()
    return deleted


def quality_issue_summary(conn: sqlite3.Connection) -> list[Mapping[str, object]]:
    rows = conn.execute(
        """
        SELECT agent, code, severity, message,
               SUM(occurrences) AS occurrences,
               MAX(last_seen_ms) AS last_seen_ms,
               MIN(sample_ref) AS sample_ref
        FROM quality_issues
        GROUP BY agent, code, severity, message
        ORDER BY occurrences DESC, agent, code
        """
    ).fetchall()
    return [dict(row) for row in rows]
