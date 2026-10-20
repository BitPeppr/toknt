"""Claude-Code-shaped fork adapter tests (Command Code, CodeBuddy, ZCode).

Command Code wraps messages in ``{"type":"message"}`` envelopes after a
``{"type":"session"}`` header and keeps sidecar snapshot/backup streams
beside transcripts; the fixtures pin that the sidecars never become
phantom sessions. CodeBuddy/ZCode are plain Claude transcripts in their own
project trees.
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

T0 = 1_760_000_000_000


@pytest.fixture
def commandcode_source(tmp_path: Path) -> DiscoveredSource:
    root = tmp_path / "projects"
    root.mkdir()
    return DiscoveredSource(
        source_id="commandcode:projects",
        agent=Harness.COMMANDCODE,
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


USAGE = {
    "input_tokens": 120,
    "output_tokens": 30,
    "cache_read_input_tokens": 40,
    "cache_creation_input_tokens": 0,
}


def envelope(message_id: str, *, role: str, ts: str, usage: dict | None = None,
             blocks: list | None = None) -> dict:
    message: dict = {"role": role, "content": blocks or []}
    if usage is not None:
        message["usage"] = usage
    return {
        "type": "message",
        "id": message_id,
        "timestamp": ts,
        "message": message,
    }


class TestCommandCode:
    def test_envelope_messages_yield_usage(
        self, conn: sqlite3.Connection, commandcode_source: DiscoveredSource
    ) -> None:
        lines = [
            {"type": "session", "version": 3, "id": "sess-1",
             "timestamp": "2026-10-01T09:00:00Z", "cwd": "/work/demo"},
            envelope("m1", role="user", ts="2026-10-01T09:00:05Z",
                     blocks=[{"type": "text", "text": "list files"}]),
            envelope("m2", role="assistant", ts="2026-10-01T09:00:10Z", usage=USAGE),
        ]
        path = Path(commandcode_source.location) / "work-demo" / "sess-1.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", "utf-8")
        scan(conn, commandcode_source)
        events = rows(
            conn,
            "SELECT input_tokens, cache_read_tokens, token_scheme FROM usage_events",
        )
        assert len(events) == 1
        assert events[0]["input_tokens"] == 120
        assert events[0]["cache_read_tokens"] == 40
        assert events[0]["token_scheme"] == "anthropic"
        session = rows(conn, "SELECT session_id, project FROM sessions")[0]
        assert session["session_id"] == "sess-1"
        assert session["project"] == "/work/demo"

    def test_sidecar_files_never_become_sessions(
        self, conn: sqlite3.Connection, commandcode_source: DiscoveredSource
    ) -> None:
        base = Path(commandcode_source.location) / "work-demo"
        base.mkdir(parents=True)
        real = [
            {"type": "session", "version": 3, "id": "sess-1",
             "timestamp": "2026-10-01T09:00:00Z", "cwd": "/work/demo"},
            envelope("m1", role="assistant", ts="2026-10-01T09:00:10Z", usage=USAGE),
        ]
        (base / "sess-1.jsonl").write_text(
            "\n".join(json.dumps(line) for line in real) + "\n", "utf-8"
        )
        snapshots = [
            {"type": "snapshot", "messages": []},
            {"type": "snapshot", "messages": []},
        ]
        (base / "sess-1.checkpoints.jsonl").write_text(
            "\n".join(json.dumps(line) for line in snapshots), "utf-8"
        )
        (base / "sess-1.prompts.jsonl").write_text(
            json.dumps({"prompt": "hi"}), "utf-8"
        )
        (base / "sess-1.v2.bak").write_text(
            json.dumps({"role": "user", "content": "old"}), "utf-8"
        )
        report = scan(conn, commandcode_source)
        assert report.usage_inserted == 1
        assert rows(conn, "SELECT COUNT(*) FROM sessions")[0][0] == 1
        assert rows(conn, "SELECT COUNT(*) FROM usage_events")[0][0] == 1

    def test_legacy_v2_lines_read_as_turns_without_usage(
        self, conn: sqlite3.Connection, commandcode_source: DiscoveredSource
    ) -> None:
        lines = [
            {"role": "user", "content": "hi", "timestamp": "2026-10-01T09:00:05Z",
             "sessionId": "sess-2"},
            {"role": "assistant", "content": "hello", "timestamp": "2026-10-01T09:00:10Z",
             "sessionId": "sess-2"},
        ]
        path = Path(commandcode_source.location) / "work-demo" / "sess-2.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", "utf-8")
        report = scan(conn, commandcode_source)
        assert report.usage_inserted == 0
        # the v2 session still exists so its history is not invisible
        assert rows(conn, "SELECT COUNT(*) FROM sessions")[0][0] == 1


class TestClonedProject:
    @pytest.mark.parametrize(
        ("source_id", "harness"), [("codebuddy:projects", Harness.CODEBUDDY),
                                   ("zcode:projects", Harness.ZCODE)]
    )
    def test_plain_claude_transcripts_ingest(
        self, conn: sqlite3.Connection, tmp_path: Path, source_id: str, harness: Harness
    ) -> None:
        root = tmp_path / harness.value / "projects" / "demo"
        root.mkdir(parents=True)
        lines = [
            {"type": "user", "message": {"role": "user", "content": []},
             "sessionId": "sess-9", "timestamp": "2026-10-01T09:00:00Z", "cwd": "/work"},
            {"type": "assistant",
             "message": {"role": "assistant", "content": [], "model": "kimi-k2",
                         "id": "msg_9", "usage": USAGE, "stop_reason": "end_turn"},
             "sessionId": "sess-9", "timestamp": "2026-10-01T09:00:10Z", "cwd": "/work"},
        ]
        (root / "sess-9.jsonl").write_text(
            "\n".join(json.dumps(line) for line in lines) + "\n", "utf-8"
        )
        source = DiscoveredSource(
            source_id=source_id, agent=harness, kind="jsonl_dir",
            location=str(root.parent), exists=True,
        )
        ingestion.ensure_source(
            conn, source_id=source_id, agent=harness, kind="jsonl_dir",
            location=str(root.parent), status="scanning",
        )
        adapter = next(a for a in registered_adapters() if a.source_id == source_id)
        report = adapter.scan(conn, source, full=True)
        assert report.usage_inserted == 1
        agent_value = rows(conn, "SELECT agent FROM usage_events")[0]["agent"]
        assert agent_value == harness.value
