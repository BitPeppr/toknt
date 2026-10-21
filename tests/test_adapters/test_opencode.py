"""OpenCode adapter tests against a synthetic database and file tree.

The synthetic source covers both storage generations the adapter supports,
including overlap between them (dedupe), partial token components, unknown
tool statuses, malformed rows, subagent sessions and watermarked rescans.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

import pytest

from tests.support.opencode_db import (
    MINUTE_MS,
    add_old_message,
    assistant_message,
    build_legacy_file_storage,
    build_opencode_database,
    delete_message,
    tokens,
    update_message_tokens,
)
from toknt.adapters.opencode import OpenCodeAdapter
from toknt.discovery import DiscoveredSource
from toknt.models.events import Agent
from toknt.storage import ingestion
from toknt.util import now_ms


@pytest.fixture
def opencode_source(tmp_path: Path) -> tuple[DiscoveredSource, Path, dict]:
    """Synthetic OpenCode data dir + discovered source pointing at it."""
    now = now_ms()
    data_dir = tmp_path / "opencode"
    db_path = data_dir / "opencode.db"
    ids = build_opencode_database(db_path, now_ms=now)
    source = DiscoveredSource(
        source_id="opencode:db",
        agent=Agent.OPENCODE,
        kind="sqlite",
        location=str(data_dir),
        exists=True,
    )
    return source, db_path, ids


def scan(conn: sqlite3.Connection, source: DiscoveredSource, *, full: bool = False):
    """Register the source (as the service does) and scan it."""
    ingestion.ensure_source(
        conn,
        source_id=source.source_id,
        agent=source.agent,
        kind=source.kind,
        location=source.location,
        status="scanning",
        note=source.note,
    )
    return OpenCodeAdapter().scan(conn, source, full=full)


def count(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> int:
    return conn.execute(sql, params).fetchone()[0]


class TestUsageIngestion:
    def test_message_count_deduped_across_generations(
        self, conn: sqlite3.Connection, opencode_source: tuple
    ) -> None:
        source, _, _ = opencode_source
        report = scan(conn, source)
        assert report.usage_inserted == 6
        assert count(conn, "SELECT COUNT(*) FROM usage_events") == 6
        # msg_c exists in both session_message and message: exactly one event.
        assert (
            count(
                conn,
                "SELECT COUNT(*) FROM usage_events WHERE source_event_id = 'msg_c'",
            )
            == 1
        )
        # The streaming message without tokens must not become a usage event.
        assert (
            count(
                conn,
                "SELECT COUNT(*) FROM usage_events WHERE source_event_id = 'msg_stream'",
            )
            == 0
        )
        # The user turn must not be ingested as usage.
        assert (
            count(
                conn,
                "SELECT COUNT(*) FROM usage_events WHERE source_event_id = 'msg_user'",
            )
            == 0
        )

    def test_full_token_components_recorded(
        self, conn: sqlite3.Connection, opencode_source: tuple
    ) -> None:
        source, _, _ = opencode_source
        scan(conn, source)
        row = conn.execute(
            "SELECT input_tokens, output_tokens, reasoning_tokens, cache_read_tokens,"
            " cache_write_tokens, provider_total_tokens, token_scheme, cost_reported,"
            " cost_currency, model, provider, finish_reason, quality"
            " FROM usage_events WHERE source_event_id = 'msg_a'"
        ).fetchone()
        assert row["input_tokens"] == 100
        assert row["output_tokens"] == 50
        assert row["reasoning_tokens"] == 7
        assert row["cache_read_tokens"] == 900
        assert row["cache_write_tokens"] == 3
        assert row["provider_total_tokens"] == 1060
        assert row["token_scheme"] == "anthropic"
        assert row["cost_reported"] == 0.05
        assert row["cost_currency"] == "USD"
        assert row["model"] == "claude-3-5-sonnet"
        assert row["provider"] == "anthropic"
        assert row["finish_reason"] == "stop"
        assert row["quality"] == "ok"

    def test_partial_components_flagged_not_counted(
        self, conn: sqlite3.Connection, opencode_source: tuple
    ) -> None:
        source, _, _ = opencode_source
        scan(conn, source)
        row = conn.execute(
            "SELECT quality, quality_notes FROM usage_events WHERE source_event_id = 'msg_d'"
        ).fetchone()
        assert row["quality"] == "partial"
        assert "missing_components: reasoning" in row["quality_notes"]

    def test_subagent_session_links_parent_and_project(
        self, conn: sqlite3.Connection, opencode_source: tuple
    ) -> None:
        source, _, _ = opencode_source
        scan(conn, source)
        row = conn.execute(
            "SELECT session_id, parent_session_id, project, agent_mode FROM usage_events"
            " WHERE source_event_id = 'msg_e'"
        ).fetchone()
        assert row["session_id"] == "ses_2"
        assert row["parent_session_id"] == "ses_1"
        assert row["project"] == "/work/alpha"
        assert row["agent_mode"] == "task"

    def test_sessions_upserted_with_metadata(
        self, conn: sqlite3.Connection, opencode_source: tuple
    ) -> None:
        source, _, _ = opencode_source
        scan(conn, source)
        row = conn.execute(
            "SELECT title, parent_session_id, cost_reported FROM sessions"
            " WHERE session_id = 'ses_1'"
        ).fetchone()
        assert row["title"] == "Main session"
        assert row["parent_session_id"] is None
        assert row["cost_reported"] == 0.42
        child = conn.execute(
            "SELECT parent_session_id, title FROM sessions WHERE session_id = 'ses_2'"
        ).fetchone()
        assert child["parent_session_id"] == "ses_1"
        assert child["title"] is None


class TestForkedSessions:
    """opencode forks copy parent history under new ids — never double count."""

    def test_copied_history_rows_are_skipped(
        self, conn: sqlite3.Connection, opencode_source: tuple
    ) -> None:
        source, db_path, _ids = opencode_source
        now = now_ms()
        fork_moment = now - 20 * MINUTE_MS
        conn_src = sqlite3.connect(db_path)
        try:
            # a fork of ses_1 created 20 minutes ago
            conn_src.execute(
                "INSERT INTO session_v2 (id, project_id, fork_session_id, directory,"
                " title, time_created, time_updated, cost)"
                " VALUES ('ses_fork', 'proj_1', 'ses_1', '/work/alpha',"
                " 'Forked session', ?, ?, 0.0)",
                (fork_moment, now - 5 * MINUTE_MS),
            )
            # a row copied from the parent (created long before the fork)
            copied = assistant_message(
                created_ms=now - 90 * MINUTE_MS,
                tokens=tokens(input=1000, output=100, cache_read=5000),
            )
            # a genuine post-fork response
            fresh = assistant_message(
                created_ms=now - 10 * MINUTE_MS,
                tokens=tokens(input=200, output=40, cache_read=300),
            )
            for message_id, payload in (("msg_copied", copied), ("msg_fresh", fresh)):
                created = json.loads(payload)["time"]["created"]
                conn_src.execute(
                    "INSERT INTO session_message (id, session_id, type,"
                    " time_created, time_updated, data) VALUES (?, ?, 'assistant', ?, ?, ?)",
                    (message_id, "ses_fork", created, created, payload),
                )
            conn_src.commit()
        finally:
            conn_src.close()

        report = scan(conn, source, full=True)
        # the fixture carries one deliberately malformed row, so the scan is
        # "partial" for that reason alone; the fork skip is a note, not an error
        assert any("forked-history rows skipped" in note for note in report.notes)
        # the copied row is excluded; the fresh one counts
        assert (
            count(
                conn,
                "SELECT COUNT(*) FROM usage_events WHERE source_event_id = 'msg_copied'",
            )
            == 0
        )
        assert (
            count(
                conn,
                "SELECT COUNT(*) FROM usage_events WHERE source_event_id = 'msg_fresh'",
            )
            == 1
        )
        total = count(
            conn,
            "SELECT SUM(input_tokens + output_tokens + cache_read_tokens)"
            " FROM usage_events WHERE source_event_id = 'msg_fresh'",
        )
        assert total == 200 + 40 + 300

    def test_sessions_without_fork_are_untouched(
        self, conn: sqlite3.Connection, opencode_source: tuple
    ) -> None:
        source, _, _ids = opencode_source
        scan(conn, source, full=True)
        # the canonical fixture has no forks: every assistant row with
        # tokens becomes an event (the pre-existing behavior)
        assert count(conn, "SELECT COUNT(*) FROM usage_events") == 6


class TestToolIngestion:
    def test_all_tool_calls_with_outcomes(
        self, conn: sqlite3.Connection, opencode_source: tuple
    ) -> None:
        source, _, _ = opencode_source
        scan(conn, source)
        rows = {
            row["source_event_id"]: row
            for row in conn.execute(
                "SELECT source_event_id, tool_name, tool_category, outcome,"
                " outcome_provenance, duration_ms, quality, quality_notes, session_id"
                " FROM tool_events"
            )
        }
        assert set(rows) == {"call_1", "call_2", "call_3", "call_4", "call_5", "call_6"}

        assert rows["call_1"]["tool_name"] == "bash"
        assert rows["call_1"]["tool_category"] == "shell"
        assert rows["call_1"]["outcome"] == "success"
        assert rows["call_1"]["outcome_provenance"] == "status_field"
        assert rows["call_1"]["duration_ms"] == 1200

        assert rows["call_2"]["tool_name"] == "edit"
        assert rows["call_2"]["outcome"] == "failure"
        assert rows["call_2"]["tool_category"] == "file_write"

        assert rows["call_3"]["outcome"] == "pending"
        assert rows["call_3"]["tool_name"] == "todowrite"

        assert rows["call_4"]["outcome"] == "success"
        assert rows["call_4"]["source_event_id"] == "call_4"

        # call_5 appears both embedded and in the part table: deduped.
        assert (
            count(
                conn,
                "SELECT COUNT(*) FROM tool_events WHERE source_event_id = 'call_5'",
            )
            == 1
        )
        assert rows["call_5"]["session_id"] == "ses_2"

        assert rows["call_6"]["outcome"] == "unknown"
        assert rows["call_6"]["outcome_provenance"] == "absent"
        assert rows["call_6"]["quality"] == "ambiguous"
        assert "unrecognized tool status: quantum" in rows["call_6"]["quality_notes"]

    def test_non_tool_parts_ignored(
        self, conn: sqlite3.Connection, opencode_source: tuple
    ) -> None:
        source, _, _ = opencode_source
        scan(conn, source)
        assert (
            count(
                conn,
                "SELECT COUNT(*) FROM tool_events WHERE source_event_id LIKE 'part_3%'",
            )
            == 0
        )


class TestQualityAndErrors:
    def test_malformed_row_recorded_as_quality_issue(
        self, conn: sqlite3.Connection, opencode_source: tuple
    ) -> None:
        source, _, _ = opencode_source
        report = scan(conn, source)
        assert report.status == "partial"
        assert report.errors == 1
        assert report.skipped == 1
        issue = conn.execute(
            "SELECT code, occurrences, sample_ref FROM quality_issues WHERE code = 'unparsable_row'"
        ).fetchone()
        assert issue["occurrences"] == 1
        assert issue["sample_ref"] == "session_message:msg_malformed"

    def test_malformed_row_count_is_rescan_stable(
        self, conn: sqlite3.Connection, opencode_source: tuple
    ) -> None:
        """Overlapping incremental windows re-observe rows; counts must not grow."""
        source, _, _ = opencode_source
        scan(conn, source)
        scan(conn, source)
        scan(conn, source, full=True)
        issue = conn.execute(
            "SELECT occurrences FROM quality_issues WHERE code = 'unparsable_row'"
        ).fetchone()
        assert issue["occurrences"] == 1

    def test_source_database_is_opened_read_only(
        self, conn: sqlite3.Connection, opencode_source: tuple
    ) -> None:
        source, db_path, _ = opencode_source
        before = hashlib.sha256(db_path.read_bytes()).hexdigest()
        files_before = sorted(p.name for p in db_path.parent.iterdir())
        scan(conn, source)
        after = hashlib.sha256(db_path.read_bytes()).hexdigest()
        assert after == before
        assert sorted(p.name for p in db_path.parent.iterdir()) == files_before


class TestIncrementalScans:
    def test_an_unchanged_database_is_not_read_again(
        self, conn: sqlite3.Connection, opencode_source: tuple
    ) -> None:
        """The message table has no index on time_updated.

        Re-reading it costs a full table scan of a database that reaches
        half a gigabyte, so a scan that finds the file unchanged stops
        there — the file's size and mtime are the cheap signal.
        """
        source, db_path, _ = opencode_source
        scan(conn, source)
        second = scan(conn, source)
        assert second.rows_seen == 0
        assert any("no changes" in note for note in second.notes)
        # the same bytes with a new timestamp are read again
        os.utime(db_path, None)
        third = scan(conn, source)
        assert third.rows_seen > 0

    def test_rescan_is_idempotent(
        self, conn: sqlite3.Connection, opencode_source: tuple
    ) -> None:
        source, _, _ = opencode_source
        scan(conn, source)
        first_totals = conn.execute(
            "SELECT SUM(input_tokens), SUM(output_tokens) FROM usage_events"
        ).fetchone()
        second = scan(conn, source)
        assert second.usage_inserted == 0
        assert count(conn, "SELECT COUNT(*) FROM usage_events") == 6
        assert conn.execute(
            "SELECT SUM(input_tokens), SUM(output_tokens) FROM usage_events"
        ).fetchone()[:] == first_totals[:]

    def test_updated_message_is_seen_incrementally(
        self, conn: sqlite3.Connection, opencode_source: tuple
    ) -> None:
        source, db_path, _ = opencode_source
        scan(conn, source)
        update_message_tokens(db_path, message_id="msg_b", input_tokens=1000, now_ms=now_ms())
        report = scan(conn, source)
        assert report.usage_inserted == 0
        assert report.usage_updated >= 1
        row = conn.execute(
            "SELECT input_tokens FROM usage_events WHERE source_event_id = 'msg_b'"
        ).fetchone()
        assert row["input_tokens"] == 1000

    def test_incremental_skips_out_of_window_rows_full_rescan_gets_them(
        self, conn: sqlite3.Connection, opencode_source: tuple
    ) -> None:
        source, db_path, _ = opencode_source
        scan(conn, source)
        add_old_message(db_path, now_ms=now_ms())
        incremental = scan(conn, source)
        assert incremental.usage_inserted == 0
        assert count(conn, "SELECT COUNT(*) FROM usage_events WHERE source_event_id = 'msg_f'") == 0
        full = scan(conn, source, full=True)
        assert full.usage_inserted == 1
        assert count(conn, "SELECT COUNT(*) FROM usage_events WHERE source_event_id = 'msg_f'") == 1

    def test_full_rescan_reconciles_deleted_rows(
        self, conn: sqlite3.Connection, opencode_source: tuple
    ) -> None:
        source, db_path, _ = opencode_source
        scan(conn, source)
        delete_message(db_path, message_id="msg_b")
        report = scan(conn, source, full=True)
        assert report.reconciled == 1
        assert count(conn, "SELECT COUNT(*) FROM usage_events WHERE source_event_id = 'msg_b'") == 0
        assert count(conn, "SELECT COUNT(*) FROM usage_events") == 5


class TestWatermarkStreams:
    def test_generations_keep_independent_watermarks(
        self, conn: sqlite3.Connection, opencode_source: tuple
    ) -> None:
        """The v1 table's stale max(time_updated) must never shrink the v2
        incremental window (they once shared a stream key — the v1 overwrite
        left every later incremental reading from weeks earlier)."""
        source, db_path, _ids = opencode_source
        scan(conn, source, full=True)
        rows = {
            stream: offset
            for stream, offset in conn.execute(
                "SELECT stream, offset FROM checkpoints WHERE source_id = 'opencode:db'"
            ).fetchall()
        }
        assert "session_message" in rows and "message" in rows
        assert rows["session_message"] != rows["message"], (
            "v1 and v2 watermarks must be stored under separate streams"
        )
        # the v2 watermark tracks the current generation's newest row
        src = sqlite3.connect(db_path)
        expected = src.execute(
            "SELECT MAX(time_updated) FROM session_message"
        ).fetchone()[0]
        src.close()
        assert rows["session_message"] == expected


class TestLegacyFileStorage:
    def test_message_files_are_ingested(
        self, conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        root = tmp_path / "opencode-legacy"
        ids = build_legacy_file_storage(root, now_ms=now_ms())
        source = DiscoveredSource(
            source_id="opencode:db",
            agent=Agent.OPENCODE,
            kind="sqlite",  # the adapter falls back to file storage when the db is absent
            location=str(root),
            exists=True,
        )
        report = scan(conn, source)
        assert report.usage_inserted == 1
        row = conn.execute(
            "SELECT session_id, model, provider, input_tokens, output_tokens,"
            " reasoning_tokens, cost_reported, token_scheme, source_event_type"
            " FROM usage_events WHERE source_event_id = ?",
            (ids["message_id"],),
        ).fetchone()
        assert row["session_id"] == "ses_file"
        assert row["model"] == "claude-3-5-sonnet"
        assert row["provider"] == "anthropic"
        assert row["input_tokens"] == 11
        assert row["output_tokens"] == 4
        assert row["reasoning_tokens"] == 2
        assert row["cost_reported"] == 0.02
        assert row["token_scheme"] == "anthropic"
        assert row["source_event_type"] == "storage.message.assistant"
        # The user message file must be ignored.
        assert count(conn, "SELECT COUNT(*) FROM usage_events") == 1

    def test_rescan_of_files_is_idempotent(
        self, conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        root = tmp_path / "opencode-legacy"
        build_legacy_file_storage(root, now_ms=now_ms())
        source = DiscoveredSource(
            source_id="opencode:db",
            agent=Agent.OPENCODE,
            kind="sqlite",
            location=str(root),
            exists=True,
        )
        scan(conn, source)
        second = scan(conn, source)
        assert second.usage_inserted == 0
        assert count(conn, "SELECT COUNT(*) FROM usage_events") == 1
