"""Claude Code adapter tests against synthetic project transcripts.

Claude Code is not installed on the development machine, so these fixtures
follow the documented transcript format: ``assistant`` lines with
``message.usage`` and tool-use blocks, ``user`` lines with ``tool_result``
blocks, streaming duplicate snapshots, sidechain lines and subagent
transcripts under ``<session>/subagents/``.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from tests.support.claude_files import (
    MINUTE_MS,
    append_records,
    assistant_record,
    build_claude_projects,
    usage_payload,
    user_record,
    write_transcript,
)
from toknt.adapters.claude_code import ClaudeCodeAdapter
from toknt.discovery import DiscoveredSource
from toknt.models.events import Agent
from toknt.storage import ingestion
from toknt.util import now_ms


@pytest.fixture
def claude_source(tmp_path: Path) -> tuple[DiscoveredSource, Path, dict]:
    root = tmp_path / "claude" / "projects"
    ids = build_claude_projects(root, now_ms=now_ms())
    source = DiscoveredSource(
        source_id="claude_code:projects",
        agent=Agent.CLAUDE_CODE,
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
    return ClaudeCodeAdapter().scan(conn, source, full=full)


def count(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> int:
    return conn.execute(sql, params).fetchone()[0]


class TestUsageIngestion:
    def test_streaming_snapshots_dedupe_keep_largest(
        self, conn: sqlite3.Connection, claude_source: tuple
    ) -> None:
        source, _, _ = claude_source
        scan(conn, source)
        # 5 main-file responses (the two req_main_1 snapshots collapse) plus
        # one subagent response.
        assert count(conn, "SELECT COUNT(*) FROM usage_events") == 6
        assert (
            count(
                conn,
                "SELECT COUNT(*) FROM usage_events WHERE source_event_id = 'req_main_1'",
            )
            == 1
        )
        row = conn.execute(
            "SELECT input_tokens, output_tokens, reasoning_tokens, cache_read_tokens,"
            " cache_write_tokens, token_scheme, provider, model, quality,"
            " quality_notes, cost_reported, cost_currency"
            " FROM usage_events WHERE source_event_id = 'req_main_1'"
        ).fetchone()
        # The final snapshot wins: 100/300/5000/2000, not 50/10/100/0.
        assert row["input_tokens"] == 100
        assert row["output_tokens"] == 300
        assert row["cache_read_tokens"] == 5000
        assert row["cache_write_tokens"] == 2000
        # Structural zero: thinking is inside output_tokens.
        assert row["reasoning_tokens"] == 0
        assert row["token_scheme"] == "anthropic"
        assert row["provider"] == "anthropic"
        assert row["model"] == "claude-sonnet-4-5"
        assert row["quality"] == "ok"
        assert row["quality_notes"] == "[]"
        assert row["cost_reported"] == 0.42
        assert row["cost_currency"] == "USD"

    def test_usage_key_falls_back_to_record_uuid(
        self, conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        """Without requestId/message.id, the record uuid keys the response."""
        root = tmp_path / "uuid-projects"
        t0 = now_ms()
        write_transcript(
            root,
            "-Users-dev-uuid/uuid-sess.jsonl",
            [
                assistant_record(
                    ts_ms=t0,
                    session_id="uuid-sess",
                    request_id=None,
                    message_id=None,
                    uuid="uuid-fallback-1",
                ),
                # Re-sent snapshot sharing only the uuid: dedupe still applies.
                assistant_record(
                    ts_ms=t0 + 1_000,
                    session_id="uuid-sess",
                    request_id=None,
                    message_id=None,
                    uuid="uuid-fallback-1",
                    usage=usage_payload(input_=40, output=60),
                ),
            ],
        )
        source = DiscoveredSource(
            source_id="claude_code:projects",
            agent=Agent.CLAUDE_CODE,
            kind="jsonl_dir",
            location=str(root),
            exists=True,
        )
        scan(conn, source)
        rows = conn.execute(
            "SELECT source_event_id, input_tokens, output_tokens FROM usage_events"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["source_event_id"] == "uuid-fallback-1"
        assert rows[0]["input_tokens"] == 40
        assert rows[0]["output_tokens"] == 60

    def test_sidechain_and_subagent_attribution(
        self, conn: sqlite3.Connection, claude_source: tuple
    ) -> None:
        source, _, _ = claude_source
        scan(conn, source)
        main = conn.execute(
            "SELECT agent_mode FROM usage_events WHERE source_event_id = 'req_main_1'"
        ).fetchone()
        assert main["agent_mode"] is None
        side = conn.execute(
            "SELECT agent_mode FROM usage_events WHERE source_event_id = 'req_main_5'"
        ).fetchone()
        assert side["agent_mode"] == "sidechain"
        sub = conn.execute(
            "SELECT session_id, parent_session_id, agent_mode, model"
            " FROM usage_events WHERE source_event_id = 'req_sub_1'"
        ).fetchone()
        assert sub["session_id"] == "sess-sub-1"
        assert sub["parent_session_id"] == "sess-main-1"
        assert sub["agent_mode"] == "subagent:Explore"
        assert sub["model"] == "claude-haiku-4-5"

    def test_sessions_cache_rows(
        self, conn: sqlite3.Connection, claude_source: tuple
    ) -> None:
        source, _, _ = claude_source
        scan(conn, source)
        main = conn.execute(
            "SELECT parent_session_id, project, model, provider, agent_mode"
            " FROM sessions WHERE session_id = 'sess-main-1'"
        ).fetchone()
        assert main["parent_session_id"] is None
        assert main["project"] == "/Users/dev/project-gamma"
        assert main["model"] == "claude-sonnet-4-5"
        assert main["provider"] == "anthropic"
        assert main["agent_mode"] is None
        sub = conn.execute(
            "SELECT parent_session_id, agent_mode FROM sessions WHERE session_id = 'sess-sub-1'"
        ).fetchone()
        assert sub["parent_session_id"] == "sess-main-1"
        assert sub["agent_mode"] == "subagent:Explore"

    def test_missing_usage_and_api_error_lines(
        self, conn: sqlite3.Connection, claude_source: tuple
    ) -> None:
        source, _, _ = claude_source
        report = scan(conn, source)
        # One malformed line plus one assistant line without usage.
        assert report.status == "partial"
        assert report.errors == 1
        assert report.skipped == 2
        rows = {
            row["code"]: row
            for row in conn.execute(
                "SELECT code, occurrences FROM quality_issues"
            )
        }
        assert rows["unparsable_row"]["occurrences"] == 1
        assert rows["missing_usage"]["occurrences"] == 1

    def test_source_files_unchanged(
        self, conn: sqlite3.Connection, claude_source: tuple
    ) -> None:
        source, root, _ = claude_source
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


class TestToolIngestion:
    def test_outcomes_from_result_field(
        self, conn: sqlite3.Connection, claude_source: tuple
    ) -> None:
        source, _, _ = claude_source
        scan(conn, source)
        rows = {
            row["source_event_id"]: row
            for row in conn.execute(
                "SELECT source_event_id, tool_name, tool_category, outcome,"
                " outcome_provenance, outcome_confidence, duration_ms, session_id,"
                " agent_mode FROM tool_events"
            )
        }
        assert set(rows) == {"toolu_1", "toolu_2", "toolu_3", "toolu_sub_1"}

        assert rows["toolu_1"]["tool_name"] == "Bash"
        assert rows["toolu_1"]["tool_category"] == "shell"
        assert rows["toolu_1"]["outcome"] == "success"
        assert rows["toolu_1"]["outcome_provenance"] == "result_field"
        assert rows["toolu_1"]["outcome_confidence"] == 0.9
        assert rows["toolu_1"]["duration_ms"] == 30_000

        assert rows["toolu_2"]["tool_name"] == "Edit"
        assert rows["toolu_2"]["tool_category"] == "file_write"
        assert rows["toolu_2"]["outcome"] == "failure"

        # No tool_result written yet: pending, never guessed.
        assert rows["toolu_3"]["outcome"] == "pending"
        assert rows["toolu_3"]["outcome_provenance"] == "absent"

        assert rows["toolu_sub_1"]["tool_name"] == "Grep"
        assert rows["toolu_sub_1"]["tool_category"] == "search"
        assert rows["toolu_sub_1"]["outcome"] == "success"
        assert rows["toolu_sub_1"]["session_id"] == "sess-sub-1"
        assert rows["toolu_sub_1"]["agent_mode"] == "subagent:Explore"


class TestIncrementalScans:
    def test_rescan_is_idempotent(
        self, conn: sqlite3.Connection, claude_source: tuple
    ) -> None:
        source, _, _ = claude_source
        scan(conn, source)
        usage_before = count(conn, "SELECT COUNT(*) FROM usage_events")
        tools_before = count(conn, "SELECT COUNT(*) FROM tool_events")
        second = scan(conn, source)
        assert second.usage_inserted == 0
        assert second.tool_inserted == 0
        assert count(conn, "SELECT COUNT(*) FROM usage_events") == usage_before
        assert count(conn, "SELECT COUNT(*) FROM tool_events") == tools_before

    def test_append_is_read_incrementally(
        self, conn: sqlite3.Connection, claude_source: tuple
    ) -> None:
        source, _, ids = claude_source
        scan(conn, source)
        t0 = now_ms()
        append_records(
            ids["main_path"],
            [
                assistant_record(
                    ts_ms=t0,
                    session_id=ids["sessions"]["main"],
                    request_id="req_main_8",
                    message_id="msg_main_8",
                    usage=usage_payload(input_=11, output=22, cache_read=33),
                    content=[
                        {"type": "tool_use", "id": "toolu_4", "name": "Write", "input": {}}
                    ],
                ),
                user_record(
                    ts_ms=t0 + 5_000,
                    session_id=ids["sessions"]["main"],
                    tool_results=[("toolu_4", False)],
                ),
            ],
        )
        report = scan(conn, source)
        assert report.usage_inserted == 1
        assert report.tool_inserted == 1
        assert count(conn, "SELECT COUNT(*) FROM usage_events") == 7
        assert count(conn, "SELECT COUNT(*) FROM tool_events") == 5

    def test_pending_call_resolves_on_later_scan(
        self, conn: sqlite3.Connection, claude_source: tuple
    ) -> None:
        source, _, ids = claude_source
        scan(conn, source)
        row = conn.execute(
            "SELECT outcome FROM tool_events WHERE source_event_id = 'toolu_3'"
        ).fetchone()
        assert row["outcome"] == "pending"

        append_records(
            ids["main_path"],
            [
                user_record(
                    ts_ms=now_ms(),
                    session_id=ids["sessions"]["main"],
                    tool_results=[("toolu_3", True)],
                )
            ],
        )
        report = scan(conn, source)
        assert report.tool_inserted == 0
        assert report.tool_updated == 1
        row = conn.execute(
            "SELECT outcome, outcome_provenance, duration_ms FROM tool_events"
            " WHERE source_event_id = 'toolu_3'"
        ).fetchone()
        assert row["outcome"] == "failure"
        assert row["outcome_provenance"] == "result_field"
        assert row["duration_ms"] is not None
        assert (
            count(
                conn,
                "SELECT COUNT(*) FROM tool_events WHERE source_event_id = 'toolu_3'",
            )
            == 1
        )

    def test_full_rescan_reconciles_deleted_files(
        self, conn: sqlite3.Connection, claude_source: tuple
    ) -> None:
        source, _, ids = claude_source
        scan(conn, source)
        ids["sub_path"].unlink()
        report = scan(conn, source, full=True)
        assert report.reconciled == 2  # one usage event + one tool event
        assert (
            count(
                conn,
                "SELECT COUNT(*) FROM usage_events WHERE session_id = 'sess-sub-1'",
            )
            == 0
        )
        assert (
            count(
                conn,
                "SELECT COUNT(*) FROM tool_events WHERE session_id = 'sess-sub-1'",
            )
            == 0
        )


class TestQualityAndErrors:
    def test_orphan_tool_result_is_reported(
        self, conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        root = tmp_path / "orphan-projects"
        write_transcript(
            root,
            "-Users-dev-orphan/orphan-sess.jsonl",
            [
                user_record(
                    ts_ms=now_ms(),
                    session_id="orphan-sess",
                    tool_results=[("toolu_nowhere", False)],
                )
            ],
        )
        source = DiscoveredSource(
            source_id="claude_code:projects",
            agent=Agent.CLAUDE_CODE,
            kind="jsonl_dir",
            location=str(root),
            exists=True,
        )
        report = scan(conn, source)
        assert count(conn, "SELECT COUNT(*) FROM tool_events") == 0
        assert (
            count(
                conn,
                "SELECT COUNT(*) FROM quality_issues WHERE code = 'orphan_tool_result'",
            )
            == 1
        )
        assert report.skipped == 1

    def test_partial_usage_marks_event_partial(
        self, conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        root = tmp_path / "partial-projects"
        t0 = now_ms() - MINUTE_MS
        write_transcript(
            root,
            "-Users-dev-partial/partial-sess.jsonl",
            [
                assistant_record(
                    ts_ms=t0,
                    session_id="partial-sess",
                    request_id="req_partial",
                    message_id="msg_partial",
                    usage={"input_tokens": 5, "output_tokens": 3},
                )
            ],
        )
        source = DiscoveredSource(
            source_id="claude_code:projects",
            agent=Agent.CLAUDE_CODE,
            kind="jsonl_dir",
            location=str(root),
            exists=True,
        )
        scan(conn, source)
        row = conn.execute(
            "SELECT quality, quality_notes, cache_read_tokens, cache_write_tokens"
            " FROM usage_events WHERE source_event_id = 'req_partial'"
        ).fetchone()
        assert row["quality"] == "partial"
        assert row["cache_read_tokens"] is None
        assert row["cache_write_tokens"] is None
        assert "missing_components: cache_read, cache_write" in row["quality_notes"]

    def test_issue_counts_survive_rescans(
        self, conn: sqlite3.Connection, claude_source: tuple
    ) -> None:
        """Repeated and full rescans of the same defect must not inflate counts."""
        source, _, _ = claude_source
        scan(conn, source)
        scan(conn, source)  # unchanged → nothing re-emitted
        scan(conn, source, full=True)
        scan(conn, source, full=True)  # cleared and re-emitted at the same size
        unparsable = conn.execute(
            "SELECT occurrences, sample_ref FROM quality_issues WHERE code = 'unparsable_row'"
        ).fetchall()
        missing = conn.execute(
            "SELECT occurrences FROM quality_issues WHERE code = 'missing_usage'"
        ).fetchall()
        assert len(unparsable) == 1
        assert unparsable[0]["occurrences"] == 1
        assert unparsable[0]["sample_ref"].endswith("sess-main-1.jsonl")
        assert len(missing) == 1
        assert missing[0]["occurrences"] == 1

    def test_fixed_defect_disappears_after_full_rescan(
        self, conn: sqlite3.Connection, claude_source: tuple
    ) -> None:
        source, _, ids = claude_source
        scan(conn, source)
        assert (
            count(conn, "SELECT COUNT(*) FROM quality_issues WHERE code = 'unparsable_row'")
            == 1
        )
        # Repair: rewrite the transcript without the malformed trailing line.
        ids["main_path"].write_text(
            ids["main_path"].read_text(encoding="utf-8").replace("{broken json\n", ""),
            encoding="utf-8",
        )
        scan(conn, source, full=True)
        assert (
            count(conn, "SELECT COUNT(*) FROM quality_issues WHERE code = 'unparsable_row'")
            == 0
        )
        # Still-defective records continue to be reported.
        assert count(conn, "SELECT COUNT(*) FROM quality_issues WHERE code = 'missing_usage'") == 1

