"""Incremental ingestion: idempotent upserts and bookkeeping.

The critical property: repeated ingestion of identical records must never
inflate totals (the plan's "stable aggregates after repeated ingestion").
"""

from __future__ import annotations

import sqlite3

from toknt.models.events import SessionRecord, ToolEvent, UsageEvent
from toknt.storage.ingestion import (
    clear_quality_issues,
    ensure_source,
    finish_scan_run,
    load_checkpoint,
    quality_issue_summary,
    record_quality_issue,
    save_checkpoint,
    start_scan_run,
    upsert_sessions,
    upsert_tool_events,
    upsert_usage_events,
)

SOURCE = "opencode:db"


class TestUsageEventUpserts:
    def test_first_write_inserts(
        self, conn: sqlite3.Connection, usage_events: list[UsageEvent]
    ) -> None:
        stats = upsert_usage_events(conn, usage_events)
        assert (stats.inserted, stats.updated) == (3, 0)
        assert stats.total == 3

    def test_repeated_writes_update_without_inflating(
        self, conn: sqlite3.Connection, usage_events: list[UsageEvent]
    ) -> None:
        upsert_usage_events(conn, usage_events)
        stats = upsert_usage_events(conn, usage_events)
        assert (stats.inserted, stats.updated) == (0, 3)

        row = conn.execute(
            "SELECT COUNT(*) AS n, SUM(input_tokens) AS total FROM usage_events"
        ).fetchone()
        assert row["n"] == 3
        assert row["total"] == 451 + 32377 + 120

    def test_duplicate_records_within_one_batch_collapse(
        self, conn: sqlite3.Connection, usage_events: list[UsageEvent]
    ) -> None:
        event = usage_events[0]
        stats = upsert_usage_events(conn, [event, event])
        assert stats.inserted == 1
        assert conn.execute("SELECT COUNT(*) FROM usage_events").fetchone()[0] == 1

    def test_updated_fields_overwrite_old_values(
        self, conn: sqlite3.Connection, usage_events: list[UsageEvent]
    ) -> None:
        import dataclasses

        original = usage_events[0]
        upsert_usage_events(conn, [original])
        revised = dataclasses.replace(original, cost_reported=0.42, quality="derived")
        upsert_usage_events(conn, [revised])
        row = conn.execute(
            "SELECT cost_reported, quality FROM usage_events WHERE id = ?", (original.id,)
        ).fetchone()
        assert row["cost_reported"] == 0.42
        assert row["quality"] == "derived"

    def test_quality_notes_round_trip_as_json(
        self, conn: sqlite3.Connection, usage_events: list[UsageEvent]
    ) -> None:
        partial = [e for e in usage_events if e.quality_notes]
        assert partial, "fixture must include an event with quality notes"
        upsert_usage_events(conn, partial)
        stored = conn.execute(
            "SELECT quality_notes FROM usage_events WHERE id = ?", (partial[0].id,)
        ).fetchone()[0]
        assert "missing output_tokens" in stored


class TestToolEventUpserts:
    def test_round_trip_preserves_outcomes(
        self, conn: sqlite3.Connection, tool_events: list[ToolEvent]
    ) -> None:
        stats = upsert_tool_events(conn, tool_events)
        assert stats.inserted == 3
        outcomes = {
            row["outcome"]
            for row in conn.execute("SELECT outcome FROM tool_events").fetchall()
        }
        assert outcomes == {"success", "failure", "pending"}

    def test_repeat_is_idempotent(
        self, conn: sqlite3.Connection, tool_events: list[ToolEvent]
    ) -> None:
        upsert_tool_events(conn, tool_events)
        stats = upsert_tool_events(conn, tool_events)
        assert stats.inserted == 0
        assert conn.execute("SELECT COUNT(*) FROM tool_events").fetchone()[0] == 3


class TestSessionUpserts:
    def test_insert_then_update(
        self, conn: sqlite3.Connection
    ) -> None:
        record = SessionRecord(
            agent="codex",
            session_id="s1",
            source_id="codex:sessions",
            project="/Users/dev/project-beta",
            title="first title",
        )
        first = upsert_sessions(conn, [record])
        assert (first.inserted, first.updated) == (1, 0)

        updated = upsert_sessions(conn, [SessionRecord(
            agent="codex",
            session_id="s1",
            source_id="codex:sessions",
            project="/Users/dev/project-beta",
            title="second title",
        )])
        assert (updated.inserted, updated.updated) == (0, 1)
        row = conn.execute("SELECT title FROM sessions").fetchone()
        assert row["title"] == "second title"

    def test_recency_merges_instead_of_clobbering(
        self, conn: sqlite3.Connection
    ) -> None:
        """An event-less later scan keeps what earlier scans learned.

        Batch-local ``time_updated_ms`` can be None when a scan window holds
        no events (appended text with no tool/usage records); that must not
        erase the last-activity time, and creation time is the earliest known.
        """
        def _record(**overrides: object) -> SessionRecord:
            base = dict(
                agent="codex",
                session_id="s1",
                source_id="codex:sessions",
                project="/Users/dev/project-beta",
            )
            base.update(overrides)
            return SessionRecord(**base)

        upsert_sessions(conn, [_record(time_created_ms=100, time_updated_ms=200)])
        # Later batch, no events: updated unknown, created re-derived later.
        upsert_sessions(conn, [_record(time_created_ms=150, time_updated_ms=None)])
        row = conn.execute(
            "SELECT time_created_ms, time_updated_ms FROM sessions"
        ).fetchone()
        assert (row["time_created_ms"], row["time_updated_ms"]) == (100, 200)

        # Genuinely newer activity still advances the updated time.
        upsert_sessions(conn, [_record(time_created_ms=50, time_updated_ms=300)])
        row = conn.execute(
            "SELECT time_created_ms, time_updated_ms FROM sessions"
        ).fetchone()
        assert (row["time_created_ms"], row["time_updated_ms"]) == (50, 300)


class TestSourceBookkeeping:
    def test_ensure_source_upserts_a_single_row(self, conn: sqlite3.Connection) -> None:
        ensure_source(conn, source_id=SOURCE, agent="opencode", kind="sqlite", location="/tmp/db")
        ensure_source(
            conn,
            source_id=SOURCE,
            agent="opencode",
            kind="sqlite",
            location="/tmp/db",
            status="partial",
            note="1 malformed row",
        )
        rows = conn.execute("SELECT * FROM sources").fetchall()
        assert len(rows) == 1
        assert rows[0]["last_scan_status"] == "partial"
        assert rows[0]["last_scan_note"] == "1 malformed row"

    def test_checkpoint_round_trip(self, conn: sqlite3.Connection) -> None:
        ensure_source(conn, source_id=SOURCE, agent="opencode", kind="sqlite", location="/tmp/db")
        assert load_checkpoint(conn, SOURCE, "messages") is None
        save_checkpoint(
            conn, SOURCE, "messages", offset=42, size=100, mtime_ms=1234, fingerprint="fp"
        )
        checkpoint = load_checkpoint(conn, SOURCE, "messages")
        assert checkpoint is not None
        assert checkpoint.offset == 42
        assert checkpoint.size == 100
        assert checkpoint.mtime_ms == 1234
        assert checkpoint.fingerprint == "fp"

    def test_scan_run_lifecycle(self, conn: sqlite3.Connection) -> None:
        ensure_source(conn, source_id=SOURCE, agent="opencode", kind="sqlite", location="/tmp/db")
        run_id = start_scan_run(conn, SOURCE, mode="incremental")
        finish_scan_run(conn, run_id, status="ok", inserted=3, updated=1, skipped=2, errors=0)
        row = conn.execute("SELECT * FROM scan_runs WHERE id = ?", (run_id,)).fetchone()
        assert row["status"] == "ok"
        assert row["mode"] == "incremental"
        assert row["events_inserted"] == 3
        assert row["events_updated"] == 1
        assert row["events_skipped"] == 2
        assert row["finished_at_ms"] is not None


class TestQualityIssues:
    def test_repeated_observation_replaces_count(
        self, conn: sqlite3.Connection
    ) -> None:
        """Re-observing the same defect must never inflate its count."""
        ensure_source(conn, source_id=SOURCE, agent="opencode", kind="sqlite", location="/tmp/db")
        for _ in range(3):
            record_quality_issue(
                conn,
                agent="opencode",
                source_id=SOURCE,
                code="unparsable_row",
                message="Malformed JSON row",
                occurrences=2,
                sample_ref="storage/message/s1/m3.json",
            )
        summary = quality_issue_summary(conn)
        assert len(summary) == 1
        assert summary[0]["occurrences"] == 2

    def test_distinct_sample_refs_count_separately(
        self, conn: sqlite3.Connection
    ) -> None:
        """Each affected record is its own row; the summary adds them up."""
        ensure_source(conn, source_id=SOURCE, agent="opencode", kind="sqlite", location="/tmp/db")
        for ref in ("storage/message/s1/m3.json", "storage/message/s2/m9.json"):
            record_quality_issue(
                conn,
                agent="opencode",
                source_id=SOURCE,
                code="unparsable_row",
                message="Malformed JSON row",
                sample_ref=ref,
            )
        summary = quality_issue_summary(conn)
        assert summary[0]["occurrences"] == 2
        assert summary[0]["sample_ref"] == "storage/message/s1/m3.json"

    def test_clear_quality_issues_scopes_to_source(
        self, conn: sqlite3.Connection
    ) -> None:
        ensure_source(conn, source_id=SOURCE, agent="opencode", kind="sqlite", location="/tmp/db")
        ensure_source(
            conn, source_id="codex:sessions", agent="codex", kind="jsonl_dir", location="/tmp/x"
        )
        record_quality_issue(conn, agent="opencode", source_id=SOURCE, code="a", message="A")
        record_quality_issue(conn, agent="codex", source_id="codex:sessions", code="b", message="B")
        assert clear_quality_issues(conn, source_id=SOURCE) == 1
        remaining = quality_issue_summary(conn)
        assert [row["code"] for row in remaining] == ["b"]
        # Clearing an unknown source is a no-op.
        assert clear_quality_issues(conn, source_id="missing:source") == 0

    def test_distinct_codes_are_separate_rows(self, conn: sqlite3.Connection) -> None:
        ensure_source(conn, source_id=SOURCE, agent="opencode", kind="sqlite", location="/tmp/db")
        record_quality_issue(
            conn, agent="opencode", source_id=SOURCE, code="a", message="A"
        )
        record_quality_issue(
            conn, agent="opencode", source_id=SOURCE, code="b", message="B"
        )
        assert len(quality_issue_summary(conn)) == 2


class TestForeignKeys:
    def test_scan_runs_require_a_registered_source(self, conn: sqlite3.Connection) -> None:
        import pytest

        with pytest.raises(sqlite3.IntegrityError):
            start_scan_run(conn, "unknown:source", mode="incremental")
