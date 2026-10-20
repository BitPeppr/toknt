"""Gemini-family adapter tests (Gemini CLI checkpoint stores + Qwen Code).

Fixtures follow the documented session format: whole-session ``.json``
objects and replayable ``.jsonl`` files (metadata first line, then messages
and ``$set``/``$rewindTo`` control lines). Assistant messages carry Gemini
API token categories where ``total == input + output + thoughts`` and
``cached`` is a subset of input — recorded as openai-scheme usage with the
thoughts surfaced as ``reasoning`` and any mismatch flagged.
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

T0 = "2026-10-01T09:00:00.000Z"


@pytest.fixture
def source(tmp_path: Path) -> DiscoveredSource:
    root = tmp_path / "gemini" / "tmp"
    (root / "hash1" / "chats").mkdir(parents=True)
    return DiscoveredSource(
        source_id="gemini:chats",
        agent=Harness.GEMINI,
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


def rows(conn: sqlite3.Connection, sql: str) -> list:
    return conn.execute(sql).fetchall()


def gemini_message(message_id: str, *, tokens: dict, calls: list | None = None,
                   model: str = "gemini-3-pro") -> dict:
    message = {
        "id": message_id,
        "timestamp": T0,
        "type": "gemini",
        "content": "hi",
        "tokens": tokens,
        "model": model,
    }
    if calls:
        message["toolCalls"] = calls
    return message


TOKENS = {"input": 120, "output": 12, "cached": 40, "thoughts": 8, "tool": 0, "total": 140}


class TestWholeSessionJson:
    def test_tokens_use_openai_scheme_with_thoughts_flagged(
        self, conn: sqlite3.Connection, source: DiscoveredSource
    ) -> None:
        record = {
            "sessionId": "s1",
            "startTime": T0,
            "lastUpdated": T0,
            "messages": [
                {"id": "u1", "timestamp": T0, "type": "user", "content": "hi"},
                gemini_message("g1", tokens=TOKENS),
            ],
        }
        path = Path(source.location) / "hash1" / "chats" / "session-s1.json"
        path.write_text(json.dumps(record), "utf-8")
        scan(conn, source)
        event = rows(conn, "SELECT * FROM usage_events")[0]
        assert event["input_tokens"] == 120
        assert event["output_tokens"] == 12
        assert event["reasoning_tokens"] == 8
        assert event["cache_read_tokens"] == 40
        assert event["token_scheme"] == "openai"
        # reported total (140) counts thoughts additively; the openai scheme
        # deliberately does not, so the mismatch is flagged, never silently
        # folded into the total.
        assert event["provider_total_tokens"] == 140
        assert "provider_total_mismatch" in event["quality_notes"]

    def test_tool_calls_map_status_to_outcomes(
        self, conn: sqlite3.Connection, source: DiscoveredSource
    ) -> None:
        record = {
            "sessionId": "s2",
            "startTime": T0,
            "lastUpdated": T0,
            "messages": [
                gemini_message(
                    "g1",
                    tokens=TOKENS,
                    calls=[
                        {"id": "c1", "name": "write_file", "status": "success",
                         "timestamp": T0, "args": {}},
                        {"id": "c2", "name": "run_shell", "status": "error",
                         "timestamp": T0, "args": {}},
                    ],
                ),
            ],
        }
        path = Path(source.location) / "hash1" / "chats" / "session-s2.json"
        path.write_text(json.dumps(record), "utf-8")
        scan(conn, source)
        tools = rows(conn, "SELECT tool_name, outcome FROM tool_events ORDER BY tool_name")
        assert sorted((t["tool_name"], t["outcome"]) for t in tools) == [
            ("run_shell", "failure"),
            ("write_file", "success"),
        ]


class TestSessionJsonl:
    def test_metadata_line_then_messages_and_set_patches(
        self, conn: sqlite3.Connection, source: DiscoveredSource
    ) -> None:
        lines = [
            {"sessionId": "s3", "startTime": T0, "lastUpdated": T0},
            {"id": "u1", "timestamp": T0, "type": "user", "content": "hi"},
            gemini_message("g1", tokens=TOKENS),
            {"$set": {"lastUpdated": "2026-10-01T09:05:00.000Z",
                      "summary": "List src files"}},
        ]
        path = Path(source.location) / "hash1" / "chats" / "session-s3.jsonl"
        path.write_text("\n".join(json.dumps(line) for line in lines), "utf-8")
        scan(conn, source)
        assert rows(conn, "SELECT COUNT(*) FROM usage_events")[0][0] == 1
        session = rows(conn, "SELECT time_created_ms, time_updated_ms FROM sessions")[0]
        assert session["time_created_ms"] is not None
        assert session["time_updated_ms"] is not None

    def test_control_lines_are_ignored_not_indexed(
        self, conn: sqlite3.Connection, source: DiscoveredSource
    ) -> None:
        lines = [
            {"sessionId": "s4", "startTime": T0, "lastUpdated": T0},
            gemini_message("g1", tokens=TOKENS),
            {"$rewindTo": "g1"},
            {"$patch": {"id": "g1", "toolCalls": []}},
        ]
        path = Path(source.location) / "hash1" / "chats" / "session-s4.jsonl"
        path.write_text("\n".join(json.dumps(line) for line in lines), "utf-8")
        report = scan(conn, source)
        assert report.usage_inserted == 1
        # rewind drops later history from the source itself; a rewind line
        # never creates usage of its own
        assert rows(conn, "SELECT COUNT(*) FROM tool_events")[0][0] == 0


class TestQwenSharesTheParser:
    def test_qwen_sessions_are_discovered_separately(
        self, conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        root = tmp_path / "qwen" / "projects"
        (root / "p1" / "chats").mkdir(parents=True)
        record = {
            "sessionId": "q1",
            "startTime": T0,
            "lastUpdated": T0,
            "messages": [gemini_message("g1", tokens=TOKENS, model="qwen3-coder")],
        }
        (root / "p1" / "chats" / "session-q1.json").write_text(
            json.dumps(record), "utf-8"
        )
        source = DiscoveredSource(
            source_id="qwen:projects", agent=Harness.QWEN, kind="jsonl_dir",
            location=str(root), exists=True,
        )
        ingestion.ensure_source(
            conn, source_id=source.source_id, agent=source.agent, kind=source.kind,
            location=source.location, status="scanning",
        )
        adapter = next(a for a in registered_adapters() if a.source_id == "qwen:projects")
        report = adapter.scan(conn, source, full=True)
        assert report.usage_inserted == 1
        model = rows(conn, "SELECT model FROM usage_events")[0]["model"]
        assert model == "qwen3-coder"
