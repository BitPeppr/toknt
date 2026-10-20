"""pi-family adapter tests against synthetic session trees.

The record shapes mirror the format verified against a live pi corpus on
this machine: ``session`` headers, ``model_change`` entries and ``message``
entries whose assistant side carries models.dev usage (input/output exclude
cache read/write, which are reported separately) and ``toolCall`` blocks
answered by ``toolResult`` messages with ``isError``.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from toknt.adapters import registered_adapters
from toknt.discovery import DiscoveredSource
from toknt.models.events import Harness
from toknt.storage import ingestion

MINUTE = 60_000
T0 = 1_760_000_000_000  # a fixed epoch-ms anchor


def session_file(
    root: Path, slug: str, name: str, entries: list[dict]
) -> Path:
    directory = root / slug
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8"
    )
    return path


def header(session_id: str = "sess-1", cwd: str = "/work/demo") -> dict:
    return {
        "type": "session",
        "version": 2,
        "id": session_id,
        "timestamp": "2026-10-01T09:00:00.000Z",
        "cwd": cwd,
    }


def assistant(
    entry_id: str,
    *,
    usage: dict,
    model: str = "glm-5.3",
    ts: str = "2026-10-01T09:01:00.000Z",
    calls: list[dict] | None = None,
) -> dict:
    content: list[dict] = []
    if calls:
        content.extend(calls)
    return {
        "type": "message",
        "id": entry_id,
        "parentId": None,
        "timestamp": ts,
        "message": {
            "role": "assistant",
            "content": content,
            "api": "openai-completions",
            "provider": "aihubmix",
            "model": model,
            "usage": usage,
            "stopReason": "stop",
            "timestamp": T0 + MINUTE,
        },
    }


def tool_result(entry_id: str, call_id: str, *, is_error: bool) -> dict:
    return {
        "type": "message",
        "id": entry_id,
        "parentId": None,
        "timestamp": "2026-10-01T09:02:00.000Z",
        "message": {
            "role": "toolResult",
            "content": [],
            "toolCallId": call_id,
            "toolName": "bash",
            "isError": is_error,
            "timestamp": T0 + 2 * MINUTE,
        },
    }


USAGE = {
    "input": 100,
    "output": 20,
    "cacheRead": 30,
    "cacheWrite": 5,
    "totalTokens": 155,
    "cost": {"input": 0.001, "output": 0.002, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.003},
}


@pytest.fixture
def source(tmp_path: Path) -> DiscoveredSource:
    root = tmp_path / "sessions"
    root.mkdir()
    return DiscoveredSource(
        source_id="pi:sessions",
        agent=Harness.PI,
        kind="jsonl_dir",
        location=str(root),
        exists=True,
    )


def scan(conn: sqlite3.Connection, source: DiscoveredSource, *, full: bool = False):
    ingestion.ensure_source(
        conn,
        source_id=source.source_id,
        agent=source.agent,
        kind=source.kind,
        location=source.location,
        status="scanning",
    )
    adapter = next(a for a in registered_adapters() if a.source_id == source.source_id)
    return adapter.scan(conn, source, full=full)


def rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list:
    return conn.execute(sql, params).fetchall()


class TestUsage:
    def test_anthropic_scheme_totals_and_cost(
        self, conn: sqlite3.Connection, source: DiscoveredSource
    ) -> None:
        session_file(source.location and Path(source.location), "--work-demo--", "a.jsonl", [
            header(),
            assistant("m1", usage=USAGE),
        ])
        scan(conn, source)
        events = rows(
            conn,
            "SELECT input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,"
            " reasoning_tokens, provider_total_tokens, token_scheme, cost_reported"
            " FROM usage_events",
        )
        assert len(events) == 1
        event = events[0]
        assert event["input_tokens"] == 100
        assert event["output_tokens"] == 20
        assert event["cache_read_tokens"] == 30
        assert event["cache_write_tokens"] == 5
        assert event["reasoning_tokens"] == 0  # structural zero: inside output
        assert event["provider_total_tokens"] == 155  # 100+20+30+5
        assert event["token_scheme"] == "anthropic"
        assert event["cost_reported"] == pytest.approx(0.003)

    def test_error_messages_are_skipped(
        self, conn: sqlite3.Connection, source: DiscoveredSource
    ) -> None:
        error_entry = {
            "type": "message",
            "id": "m2",
            "timestamp": "2026-10-01T09:03:00.000Z",
            "message": {
                "role": "assistant",
                "content": [],
                "usage": {**USAGE, "input": 0, "output": 0, "totalTokens": 0},
                "stopReason": "error",
                "errorMessage": "403 forbidden",
            },
        }
        session_file(Path(source.location), "--work-demo--", "b.jsonl", [
            header(),
            assistant("m1", usage=USAGE),
            error_entry,
        ])
        report = scan(conn, source)
        assert report.usage_inserted == 1
        assert report.status == "ok"

    def test_session_record_carries_cwd_and_model(
        self, conn: sqlite3.Connection, source: DiscoveredSource
    ) -> None:
        session_file(Path(source.location), "--work-demo--", "c.jsonl", [
            header(cwd="/work/demo"),
            {"type": "model_change", "id": "mc1", "provider": "aihubmix",
             "modelId": "glm-5.3", "timestamp": "2026-10-01T09:00:30.000Z"},
            assistant("m1", usage=USAGE),
        ])
        scan(conn, source)
        session = rows(conn, "SELECT project, model, provider FROM sessions")[0]
        assert session["project"] == "/work/demo"
        assert session["model"] == "glm-5.3"
        assert session["provider"] == "aihubmix"


class TestTools:
    def test_call_and_result_pair_by_id(
        self, conn: sqlite3.Connection, source: DiscoveredSource
    ) -> None:
        session_file(Path(source.location), "--work-demo--", "d.jsonl", [
            header(),
            assistant(
                "m1",
                usage=USAGE,
                calls=[{"type": "toolCall", "id": "call-1", "name": "bash",
                        "arguments": {"command": "ls"}}],
            ),
            tool_result("m2", "call-1", is_error=False),
            assistant(
                "m3",
                usage=USAGE,
                ts="2026-10-01T09:05:00.000Z",
                calls=[{"type": "toolCall", "id": "call-2", "name": "edit",
                        "arguments": {"path": "x"}}],
            ),
            tool_result("m4", "call-2", is_error=True),
        ])
        scan(conn, source)
        tools = rows(
            conn,
            "SELECT tool_name, outcome, outcome_provenance FROM tool_events ORDER BY ts_ms",
        )
        assert [(t["tool_name"], t["outcome"]) for t in tools] == [
            ("bash", "success"),
            ("edit", "failure"),
        ]
        assert all(t["outcome_provenance"] == "result_field" for t in tools)

    def test_unanswered_call_stays_pending(
        self, conn: sqlite3.Connection, source: DiscoveredSource
    ) -> None:
        session_file(Path(source.location), "--work-demo--", "e.jsonl", [
            header(),
            assistant(
                "m1",
                usage=USAGE,
                calls=[{"type": "toolCall", "id": "call-9", "name": "grep",
                        "arguments": {}}],
            ),
        ])
        scan(conn, source)
        outcome = rows(conn, "SELECT outcome FROM tool_events")[0]["outcome"]
        assert outcome == "pending"


class TestIncremental:
    def test_append_only_reparse_reads_new_entries(
        self, conn: sqlite3.Connection, source: DiscoveredSource
    ) -> None:
        path = session_file(Path(source.location), "--work-demo--", "f.jsonl", [
            header(),
            assistant("m1", usage=USAGE),
        ])
        scan(conn, source)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(assistant(
                    "m2",
                    usage=USAGE,
                    ts="2026-10-01T09:10:00.000Z",
                )) + "\n"
            )
        report = scan(conn, source)
        assert report.usage_inserted == 1
        assert report.usage_updated == 0
        assert rows(conn, "SELECT COUNT(*) FROM usage_events")[0][0] == 2

    def test_full_rescan_is_idempotent(
        self, conn: sqlite3.Connection, source: DiscoveredSource
    ) -> None:
        session_file(Path(source.location), "--work-demo--", "g.jsonl", [
            header(),
            assistant("m1", usage=USAGE),
        ])
        scan(conn, source, full=True)
        scan(conn, source, full=True)
        assert rows(conn, "SELECT COUNT(*) FROM usage_events")[0][0] == 1
        assert rows(conn, "SELECT COUNT(*) FROM sessions")[0][0] == 1
