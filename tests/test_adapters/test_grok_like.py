"""Grok adapter tests against synthetic per-session state directories.

Fixtures mirror the format verified against a live corpus: ``usage.json``
with per-turn ``turns[]`` and per-model ``modelUsage`` maps,
``summary.json`` session metadata, and ``events.jsonl`` tool events with
explicit outcomes. ``totalTokens == inputTokens + outputTokens`` (cached
reads are a subset of input) — the openai scheme.
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

T0 = 1_760_000_000_000  # epoch ms


@pytest.fixture
def source(tmp_path: Path) -> DiscoveredSource:
    root = tmp_path / "sessions"
    root.mkdir()
    return DiscoveredSource(
        source_id="grok:sessions",
        agent=Harness.GROK,
        kind="json_dir",
        location=str(root),
        exists=True,
    )


def write_session(
    root: Path,
    cwd_slug: str,
    session_id: str,
    *,
    usage: dict | None,
    summary: dict | None = None,
    events: list[dict] | None = None,
) -> Path:
    directory = root / cwd_slug / session_id
    directory.mkdir(parents=True)
    if usage is not None:
        (directory / "usage.json").write_text(json.dumps(usage), "utf-8")
    if summary is not None:
        (directory / "summary.json").write_text(json.dumps(summary), "utf-8")
    if events is not None:
        (directory / "events.jsonl").write_text(
            "\n".join(json.dumps(event) for event in events) + "\n", "utf-8"
        )
    return directory


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


def rows(conn: sqlite3.Connection, sql: str) -> list:
    return conn.execute(sql).fetchall()


def usage_payload(total_in: int, total_out: int, turns: list[dict]) -> dict:
    return {
        "sessionId": "sess-1",
        "updatedAt": "2026-10-13T04:12:50.821080+00:00",
        "session": {
            "inputTokens": total_in,
            "outputTokens": total_out,
            "totalTokens": total_in + total_out,
        },
        "turns": turns,
    }


def turn(number: int, *, model: str, input_tokens: int, output_tokens: int) -> dict:
    return {
        "turnNumber": number,
        "endedAt": "2026-10-13T03:39:11.088378+00:00",
        "primaryModelId": model,
        "modelUsage": {
            model: {
                "inputTokens": input_tokens,
                "outputTokens": output_tokens,
                "cachedReadTokens": input_tokens // 2,
                "cacheCreationTokens": 0,
                "reasoningTokens": output_tokens // 2,
                "totalTokens": input_tokens + output_tokens,
            }
        },
    }


class TestUsage:
    def test_openai_scheme_per_turn_per_model(
        self, conn: sqlite3.Connection, source: DiscoveredSource
    ) -> None:
        write_session(
            Path(source.location), "%2Fwork%2Fdemo", "sess-1",
            usage=usage_payload(300, 60, [
                turn(1, model="grok-4.7", input_tokens=200, output_tokens=40),
                turn(2, model="grok-4.7", input_tokens=100, output_tokens=20),
            ]),
            summary={"info": {"id": "sess-1", "cwd": "/work/demo"},
                     "current_model_id": "grok-4.7"},
        )
        scan(conn, source)
        events = rows(
            conn,
            "SELECT input_tokens, output_tokens, cache_read_tokens, reasoning_tokens,"
            " token_scheme, model FROM usage_events ORDER BY ts_ms",
        )
        assert len(events) == 2
        first = events[0]
        assert first["input_tokens"] == 200
        assert first["output_tokens"] == 40
        assert first["cache_read_tokens"] == 100  # subset of input
        assert first["reasoning_tokens"] == 20    # subset of output
        assert first["token_scheme"] == "openai"
        assert first["model"] == "grok-4.7"

    def test_session_metadata_from_summary(
        self, conn: sqlite3.Connection, source: DiscoveredSource
    ) -> None:
        write_session(
            Path(source.location), "%2Fwork%2Fdemo", "sess-2",
            usage=usage_payload(10, 5, []),
            summary={"info": {"id": "sess-2", "cwd": "/work/demo"},
                     "session_summary": "Fix the flaky test",
                     "current_model_id": "grok-4.7"},
        )
        scan(conn, source)
        session = rows(conn, "SELECT project, title, model FROM sessions")[0]
        assert session["project"] == "/work/demo"
        assert session["title"] == "Fix the flaky test"
        assert session["model"] == "grok-4.7"


class TestTools:
    def test_events_log_pairs_started_and_completed(
        self, conn: sqlite3.Connection, source: DiscoveredSource
    ) -> None:
        write_session(
            Path(source.location), "%2Fwork%2Fdemo", "sess-3",
            usage=usage_payload(10, 5, []),
            events=[
                {"ts": "2026-10-13T03:39:10.000Z", "type": "tool_started",
                 "tool_name": "grep", "tool_call_id": "call-1"},
                {"ts": "2026-10-13T03:39:10.050Z", "type": "tool_completed",
                 "tool_name": "grep", "tool_call_id": "call-1",
                 "duration_ms": 50, "outcome": "success"},
                {"ts": "2026-10-13T03:39:11.000Z", "type": "tool_started",
                 "tool_name": "read_file", "tool_call_id": "call-2"},
                {"ts": "2026-10-13T03:39:11.020Z", "type": "tool_completed",
                 "tool_name": "read_file", "tool_call_id": "call-2",
                 "duration_ms": 20, "outcome": "failure"},
            ],
        )
        scan(conn, source)
        tools = rows(
            conn,
            "SELECT tool_name, outcome, duration_ms, outcome_provenance"
            " FROM tool_events ORDER BY ts_ms",
        )
        assert [tuple(t) for t in tools] == [
            ("grep", "success", 50, "status_field"),
            ("read_file", "failure", 20, "status_field"),
        ]

    def test_unfinished_call_is_pending(
        self, conn: sqlite3.Connection, source: DiscoveredSource
    ) -> None:
        write_session(
            Path(source.location), "%2Fwork%2Fdemo", "sess-4",
            usage=usage_payload(10, 5, []),
            events=[
                {"ts": "2026-10-13T03:39:10.000Z", "type": "tool_started",
                 "tool_name": "bash", "tool_call_id": "call-9"},
            ],
        )
        scan(conn, source)
        outcome = rows(conn, "SELECT outcome FROM tool_events")[0]["outcome"]
        assert outcome == "pending"
