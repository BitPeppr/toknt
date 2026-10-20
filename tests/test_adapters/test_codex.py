"""Codex adapter tests against synthetic rollout files.

The fixtures mirror the real shapes (verified against local
``~/.codex/sessions`` in 2026-10): current ``token_usage_record`` files with
their ``token_count`` mirror, legacy cumulative counters, turn-context model
attribution, paired tool calls, subagent threads, and malformed lines.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from tests.support.codex_files import (
    MINUTE_MS,
    append_records,
    build_codex_sessions,
    call,
    counts,
    output,
    session_meta,
    token_count,
    turn_context,
    usage_record,
    write_rollout,
)
from toknt.adapters.codex import CodexAdapter
from toknt.discovery import DiscoveredSource
from toknt.models.events import Agent
from toknt.storage import ingestion
from toknt.util import now_ms


@pytest.fixture
def codex_source(tmp_path: Path) -> tuple[DiscoveredSource, Path, dict]:
    root = tmp_path / "codex" / "sessions"
    ids = build_codex_sessions(root, now_ms=now_ms())
    source = DiscoveredSource(
        source_id="codex:sessions",
        agent=Agent.CODEX,
        kind="jsonl_dir",
        location=str(root),
        exists=True,
    )
    return source, root, ids


def scan(conn: sqlite3.Connection, source: DiscoveredSource, *, full: bool = False):
    ingestion.ensure_source(
        conn,
        source_id=source.source_id,
        agent=source.agent,
        kind=source.kind,
        location=source.location,
        status="scanning",
        note=source.note,
    )
    return CodexAdapter().scan(conn, source, full=full)


def count(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> int:
    return conn.execute(sql, params).fetchone()[0]


class TestUsageIngestion:
    def test_current_records_with_mirror_ignored(
        self, conn: sqlite3.Connection, codex_source: tuple
    ) -> None:
        source, _, _ = codex_source
        report = scan(conn, source)
        # 2 current + 1 subagent + 2 legacy usage events; the token_count
        # mirror of the current file must not be ingested.
        assert report.usage_inserted == 5
        assert (
            count(
                conn,
                "SELECT COUNT(*) FROM usage_events WHERE input_tokens = 9999",
            )
            == 0
        )
        assert (
            count(
                conn,
                "SELECT COUNT(*) FROM usage_events WHERE source_event_type = 'token_usage_record'",
            )
            == 3
        )
        assert (
            count(
                conn,
                "SELECT COUNT(*) FROM usage_events"
                " WHERE source_event_type = 'event_msg.token_count'",
            )
            == 2
        )

    def test_openai_scheme_components(
        self, conn: sqlite3.Connection, codex_source: tuple
    ) -> None:
        source, _, _ = codex_source
        scan(conn, source)
        row = conn.execute(
            "SELECT input_tokens, output_tokens, reasoning_tokens, cache_read_tokens,"
            " cache_write_tokens, provider_total_tokens, token_scheme, provider,"
            " model, quality, quality_notes"
            " FROM usage_events WHERE source_event_id = 'resp_current_1'"
        ).fetchone()
        assert row["input_tokens"] == 1000
        assert row["output_tokens"] == 100
        assert row["reasoning_tokens"] == 40
        assert row["cache_read_tokens"] == 400
        assert row["cache_write_tokens"] == 0
        assert row["provider_total_tokens"] == 1100
        assert row["token_scheme"] == "openai"
        assert row["provider"] == "openai"
        # Model attribution comes from turn_context, not thread_settings_applied.
        assert row["model"] == "gpt-6-luna"
        assert row["quality"] == "ok"
        assert row["quality_notes"] == "[]"

    def test_legacy_last_token_usage_is_per_response(
        self, conn: sqlite3.Connection, codex_source: tuple
    ) -> None:
        source, _, _ = codex_source
        scan(conn, source)
        rows = conn.execute(
            "SELECT source_event_id, input_tokens, output_tokens, cache_read_tokens,"
            " provider_total_tokens, quality, model"
            " FROM usage_events WHERE source_event_type = 'event_msg.token_count'"
            " ORDER BY ts_ms"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0]["input_tokens"] == 1000
        assert rows[0]["output_tokens"] == 100
        assert rows[0]["cache_read_tokens"] == 200
        assert rows[0]["quality"] == "ok"
        assert rows[0]["model"] == "gpt-5.5"
        # The second event uses last_token_usage (delta), not the cumulative
        # totals (2500/400).
        assert rows[1]["input_tokens"] == 1500
        assert rows[1]["output_tokens"] == 300
        assert rows[1]["cache_read_tokens"] == 500
        assert rows[1]["provider_total_tokens"] == 1800
        assert rows[1]["source_event_id"].startswith("tc:")

    def test_subagent_thread_keeps_own_session_and_parent(
        self, conn: sqlite3.Connection, codex_source: tuple
    ) -> None:
        source, _, _ = codex_source
        scan(conn, source)
        row = conn.execute(
            "SELECT session_id, parent_session_id, agent_mode,"
            " project, directory FROM usage_events WHERE source_event_id = 'resp_sub_1'"
        ).fetchone()
        assert row["session_id"] == "thread-sub"
        assert row["parent_session_id"] == "thread-root"
        assert row["agent_mode"] == "subagent:Linnaeus"
        assert row["project"] == "/work/alpha"
        assert row["directory"] == "/work/alpha"

    def test_sessions_cache_rows(
        self, conn: sqlite3.Connection, codex_source: tuple
    ) -> None:
        source, _, _ = codex_source
        scan(conn, source)
        parent = conn.execute(
            "SELECT parent_session_id, project, model, provider"
            " FROM sessions WHERE session_id = 'thread-root'"
        ).fetchone()
        assert parent["parent_session_id"] is None
        assert parent["project"] == "/work/alpha"
        assert parent["model"] == "gpt-6-luna"
        assert parent["provider"] == "openai"
        child = conn.execute(
            "SELECT parent_session_id, agent_mode FROM sessions WHERE session_id = 'thread-sub'"
        ).fetchone()
        assert child["parent_session_id"] == "thread-root"
        assert child["agent_mode"] == "subagent:Linnaeus"


class TestToolIngestion:
    def test_outcomes_by_output_text(
        self, conn: sqlite3.Connection, codex_source: tuple
    ) -> None:
        source, _, _ = codex_source
        scan(conn, source)
        rows = {
            row["source_event_id"]: row
            for row in conn.execute(
                "SELECT source_event_id, tool_name, tool_category, outcome,"
                " outcome_provenance, outcome_confidence, duration_ms, session_id"
                " FROM tool_events"
            )
        }
        assert set(rows) == {
            "call_current_a",
            "call_current_b",
            "call_current_c",
            "call_current_d",
            "call_sub_a",
            "call_legacy_a",
        }

        assert rows["call_current_a"]["tool_name"] == "exec"
        assert rows["call_current_a"]["tool_category"] == "shell"
        assert rows["call_current_a"]["outcome"] == "success"
        assert rows["call_current_a"]["outcome_provenance"] == "output_text"
        assert rows["call_current_a"]["outcome_confidence"] == 0.8
        assert rows["call_current_a"]["duration_ms"] == 2000

        # List-shaped output blocks are flattened before classification.
        assert rows["call_current_b"]["outcome"] == "failure"
        assert rows["call_current_b"]["duration_ms"] == 500

        # No output written yet: pending, never guessed.
        assert rows["call_current_c"]["outcome"] == "pending"
        assert rows["call_current_c"]["outcome_provenance"] == "absent"

        assert rows["call_current_d"]["outcome"] == "failure"
        assert rows["call_current_d"]["outcome_confidence"] == 0.7

        assert rows["call_sub_a"]["tool_name"] == "apply_patch"
        assert rows["call_sub_a"]["tool_category"] == "file_write"
        assert rows["call_sub_a"]["outcome"] == "success"
        assert rows["call_sub_a"]["session_id"] == "thread-sub"

        assert rows["call_legacy_a"]["outcome"] == "success"


class TestQualityAndErrors:
    def test_report_counts_and_issues(
        self, conn: sqlite3.Connection, codex_source: tuple
    ) -> None:
        source, _, _ = codex_source
        report = scan(conn, source)
        assert report.status == "partial"
        assert report.errors == 1  # malformed JSON line
        assert report.skipped == 2  # malformed line + null token_count info
        rows = {
            row["code"]: row
            for row in conn.execute(
                "SELECT code, occurrences, sample_ref FROM quality_issues"
            )
        }
        assert rows["unparsable_row"]["occurrences"] == 1
        assert rows["missing_token_info"]["occurrences"] == 1

    def test_orphan_output_is_reported(self, conn: sqlite3.Connection, tmp_path: Path) -> None:
        root = tmp_path / "orphan-sessions"
        write_rollout(
            root,
            "2026/10/03/rollout-orphan.jsonl",
            [
                session_meta(thread_id="thread-orphan", ts_ms=now_ms(), cwd="/work/o"),
                output(
                    ordinal=5,
                    ts_ms=now_ms(),
                    call_id="call_nowhere",
                    text="Process exited with code 0\n",
                ),
            ],
        )
        source = DiscoveredSource(
            source_id="codex:sessions",
            agent=Agent.CODEX,
            kind="jsonl_dir",
            location=str(root),
            exists=True,
        )
        report = scan(conn, source)
        assert count(conn, "SELECT COUNT(*) FROM tool_events") == 0
        assert (
            count(
                conn,
                "SELECT COUNT(*) FROM quality_issues WHERE code = 'orphan_tool_output'",
            )
            == 1
        )
        assert report.skipped == 1

    def test_source_files_unchanged(
        self, conn: sqlite3.Connection, codex_source: tuple
    ) -> None:
        source, root, ids = codex_source
        before = {
            path: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*.jsonl"))
        }
        scan(conn, source)
        after = {
            path: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*.jsonl"))
        }
        assert after == before
        assert ids["current_path"] in before


class TestIncrementalScans:
    def test_rescan_is_idempotent(
        self, conn: sqlite3.Connection, codex_source: tuple
    ) -> None:
        source, _, _ = codex_source
        scan(conn, source)
        usage_before = count(conn, "SELECT COUNT(*) FROM usage_events")
        tools_before = count(conn, "SELECT COUNT(*) FROM tool_events")
        second = scan(conn, source)
        assert second.usage_inserted == 0
        assert second.tool_inserted == 0
        assert count(conn, "SELECT COUNT(*) FROM usage_events") == usage_before
        assert count(conn, "SELECT COUNT(*) FROM tool_events") == tools_before

    def test_append_is_read_incrementally(
        self, conn: sqlite3.Connection, codex_source: tuple
    ) -> None:
        source, _, ids = codex_source
        scan(conn, source)
        t0 = now_ms()
        append_records(
            ids["current_path"],
            [
                turn_context(ts_ms=t0, turn_id="turn-3", model="gpt-6-luna", ordinal=50),
                usage_record(
                    ordinal=51,
                    ts_ms=t0 + 1000,
                    thread_id="thread-root",
                    session_id="thread-root",
                    turn_id="turn-3",
                    response_id="resp_current_3",
                    input_tokens=10,
                    output_tokens=5,
                    reasoning_tokens=1,
                    cached_tokens=0,
                    thread_total=3315,
                ),
                call(ordinal=52, ts_ms=t0 + 2000, call_id="call_current_e", name="exec"),
                output(
                    ordinal=53,
                    ts_ms=t0 + 3000,
                    call_id="call_current_e",
                    text="Exit code: 0\n",
                ),
            ],
        )
        report = scan(conn, source)
        assert report.usage_inserted == 1
        assert report.tool_inserted == 1
        assert count(conn, "SELECT COUNT(*) FROM usage_events") == 6
        assert count(conn, "SELECT COUNT(*) FROM tool_events") == 7
        assert (
            count(
                conn,
                "SELECT COUNT(*) FROM usage_events WHERE source_event_id = 'resp_current_1'",
            )
            == 1
        )

    def test_pending_call_resolves_on_later_scan(
        self, conn: sqlite3.Connection, codex_source: tuple
    ) -> None:
        source, _, ids = codex_source
        scan(conn, source)
        row = conn.execute(
            "SELECT outcome FROM tool_events WHERE source_event_id = 'call_current_c'"
        ).fetchone()
        assert row["outcome"] == "pending"

        append_records(
            ids["current_path"],
            [
                output(
                    ordinal=60,
                    ts_ms=now_ms() + 5000,
                    call_id="call_current_c",
                    text="Process exited with code 0\nOutput:\nwrote\n",
                    kind="function_call_output",
                )
            ],
        )

        report = scan(conn, source)
        assert report.usage_inserted == 0
        assert report.tool_inserted == 0
        assert report.tool_updated == 1
        row = conn.execute(
            "SELECT outcome, outcome_provenance, duration_ms FROM tool_events"
            " WHERE source_event_id = 'call_current_c'"
        ).fetchone()
        assert row["outcome"] == "success"
        assert row["outcome_provenance"] == "output_text"
        assert row["duration_ms"] is not None
        assert (
            count(
                conn,
                "SELECT COUNT(*) FROM tool_events WHERE source_event_id = 'call_current_c'",
            )
            == 1
        )

    def test_format_flip_retracts_legacy_events(
        self, conn: sqlite3.Connection, codex_source: tuple
    ) -> None:
        source, _, ids = codex_source
        scan(conn, source)
        legacy_file_events = count(
            conn,
            "SELECT COUNT(*) FROM usage_events WHERE source_event_id LIKE 'tc:%'",
        )
        assert legacy_file_events == 2

        t0 = now_ms()
        append_records(
            ids["legacy_path"],
            [
                turn_context(ts_ms=t0, turn_id="turn-l2", model="gpt-5.5", ordinal=30),
                usage_record(
                    ordinal=31,
                    ts_ms=t0 + 1000,
                    thread_id="thread-legacy",
                    session_id="thread-legacy",
                    turn_id="turn-l2",
                    response_id="resp_legacy_2",
                    input_tokens=77,
                    output_tokens=7,
                    reasoning_tokens=0,
                    cached_tokens=0,
                    thread_total=77,
                ),
            ],
        )
        report = scan(conn, source)
        assert any("re-ingested" in note for note in report.notes)
        # The legacy counters were retracted in favour of the richer records.
        assert (
            count(
                conn,
                "SELECT COUNT(*) FROM usage_events WHERE source_event_id LIKE 'tc:%'",
            )
            == 0
        )
        assert (
            count(
                conn,
                "SELECT COUNT(*) FROM usage_events WHERE source_event_id = 'resp_legacy_2'",
            )
            == 1
        )
        # The file's pre-flip current-format processing must not wipe the
        # other files' legacy events (there are none here, but be explicit).
        assert count(conn, "SELECT COUNT(*) FROM usage_events") == 4

    def test_full_rescan_reconciles_deleted_files(
        self, conn: sqlite3.Connection, codex_source: tuple
    ) -> None:
        source, _, ids = codex_source
        scan(conn, source)
        ids["sub_path"].unlink()
        report = scan(conn, source, full=True)
        assert report.reconciled == 2  # one usage event + one tool event
        assert (
            count(
                conn,
                "SELECT COUNT(*) FROM usage_events WHERE session_id = 'thread-sub'",
            )
            == 0
        )
        assert (
            count(
                conn,
                "SELECT COUNT(*) FROM tool_events WHERE session_id = 'thread-sub'",
            )
            == 0
        )


class TestDerivedCounters:
    def test_cumulative_only_events_are_marked_derived(
        self, conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        root = tmp_path / "derived-sessions"
        t0 = now_ms() - MINUTE_MS
        write_rollout(
            root,
            "2026/06/04/rollout-derived.jsonl",
            [
                session_meta(thread_id="thread-derived", ts_ms=t0, cwd="/work/d"),
                turn_context(ts_ms=t0, turn_id="turn-d", model="gpt-5.5", ordinal=7),
                token_count(
                    ordinal=14,
                    ts_ms=t0 + MINUTE_MS,
                    total=counts(input_=100, output=20, cached=0, total=120),
                    last=None,
                ),
                token_count(
                    ordinal=20,
                    ts_ms=t0 + 2 * MINUTE_MS,
                    total=counts(input_=300, output=90, cached=0, total=390),
                    last=None,
                ),
            ],
        )
        source = DiscoveredSource(
            source_id="codex:sessions",
            agent=Agent.CODEX,
            kind="jsonl_dir",
            location=str(root),
            exists=True,
        )
        scan(conn, source)
        rows = conn.execute(
            "SELECT input_tokens, output_tokens, provider_total_tokens, quality,"
            " quality_notes FROM usage_events ORDER BY ts_ms"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0]["input_tokens"] == 100
        assert rows[0]["output_tokens"] == 20
        assert rows[0]["provider_total_tokens"] == 120
        assert rows[0]["quality"] == "derived"
        assert "delta_of_cumulative_totals" in rows[0]["quality_notes"]
        # Second event: 300-100 input, 90-20 output (increments, not totals).
        assert rows[1]["input_tokens"] == 200
        assert rows[1]["output_tokens"] == 70
        assert rows[1]["provider_total_tokens"] == 270
