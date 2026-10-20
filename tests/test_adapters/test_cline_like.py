"""Cline-family adapter tests: modern CLI sessions and VS Code task dirs.

The modern CLI fixtures mirror the live corpus verified on this machine:
``<id>.json`` manifests with ``metadata.usage`` and ``<id>.messages.json``
dicts whose assistant entries carry per-response ``metrics`` and
``modelInfo``, with ``tool_use``/``tool_result`` content blocks. The VS Code
fixtures follow the documented ``api_conversation_history.json`` task format
(sessions and tools only — no token counts exist in that format).
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


def write_session(
    root: Path,
    session_id: str,
    *,
    manifest: dict,
    messages: list[dict],
) -> Path:
    directory = root / session_id
    directory.mkdir(parents=True)
    (directory / f"{session_id}.json").write_text(json.dumps(manifest), "utf-8")
    (directory / f"{session_id}.messages.json").write_text(
        json.dumps({"version": 1, "sessionId": session_id, "messages": messages}),
        "utf-8",
    )
    return directory


def manifest(session_id: str, **metadata) -> dict:
    return {
        "version": 1,
        "session_id": session_id,
        "started_at": "2026-10-01T09:00:00.000Z",
        "ended_at": "2026-10-01T09:10:00.000Z",
        "cwd": "/work/demo",
        "model": "deepseek-v4",
        "provider": "cline",
        "status": "completed",
        "metadata": metadata,
    }


def assistant(message_id: str, *, metrics: dict | None, ts: int, calls: list | None = None,
              results: list | None = None) -> dict:
    content: list[dict] = []
    if calls:
        content.extend(calls)
    if results:
        content.extend(results)
    entry = {"id": message_id, "role": "assistant", "ts": ts, "content": content}
    if metrics is not None:
        entry["metrics"] = metrics
        entry["modelInfo"] = {"id": "deepseek-v4", "provider": "cline"}
    return entry


METRICS = {"inputTokens": 100, "outputTokens": 20, "cacheReadTokens": 30, "cacheWriteTokens": 0}


@pytest.fixture
def source(tmp_path: Path) -> DiscoveredSource:
    root = tmp_path / "sessions"
    root.mkdir()
    return DiscoveredSource(
        source_id="cline:sessions",
        agent=Harness.CLINE,
        kind="json_dir",
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


def one(conn: sqlite3.Connection, sql: str):
    return conn.execute(sql).fetchall()


class TestModernCli:
    def test_usage_uses_anthropic_scheme(
        self, conn: sqlite3.Connection, source: DiscoveredSource
    ) -> None:
        write_session(Path(source.location), "s1", manifest=manifest("s1"),
                      messages=[assistant("m1", metrics=METRICS, ts=T0)])
        scan(conn, source)
        event = one(conn, "SELECT * FROM usage_events")[0]
        assert event["input_tokens"] == 100
        assert event["cache_read_tokens"] == 30
        assert event["token_scheme"] == "anthropic"
        assert event["reasoning_tokens"] == 0  # billed inside output

    def test_tool_result_entries_classify_outcomes(
        self, conn: sqlite3.Connection, source: DiscoveredSource
    ) -> None:
        messages = [
            assistant(
                "m1",
                metrics=METRICS,
                ts=T0,
                calls=[{"type": "tool_use", "id": "call-1", "name": "run_commands",
                        "input": {"commands": ["ls"]}}],
            ),
            assistant(
                "m2",
                metrics=METRICS,
                ts=T0 + 60_000,
                results=[
                    {"type": "tool_result", "tool_use_id": "call-1", "name": "run_commands",
                     "content": [{"query": "ls", "result": "ok", "success": True}]},
                ],
            ),
            assistant(
                "m3",
                metrics=METRICS,
                ts=T0 + 120_000,
                calls=[{"type": "tool_use", "id": "call-2", "name": "run_commands",
                        "input": {"commands": ["boom"]}}],
            ),
            assistant(
                "m4",
                metrics=METRICS,
                ts=T0 + 180_000,
                results=[
                    {"type": "tool_result", "tool_use_id": "call-2", "name": "run_commands",
                     "content": [{"query": "boom", "error": "Command exited with code 1",
                                  "success": False}]},
                ],
            ),
        ]
        write_session(Path(source.location), "s2", manifest=manifest("s2"), messages=messages)
        scan(conn, source)
        tools = one(conn, "SELECT tool_name, outcome FROM tool_events ORDER BY ts_ms")
        assert [tuple(t) for t in tools] == [
            ("run_commands", "success"),
            ("run_commands", "failure"),
        ]

    def test_manifest_metadata_fills_session(
        self, conn: sqlite3.Connection, source: DiscoveredSource
    ) -> None:
        write_session(
            Path(source.location),
            "s3",
            manifest=manifest("s3", title="Fix bug", totalCost=0.25),
            messages=[assistant("m1", metrics=METRICS, ts=T0)],
        )
        scan(conn, source)
        session = one(conn, "SELECT title, project, cost_reported FROM sessions")[0]
        assert session["title"] == "Fix bug"
        assert session["project"] == "/work/demo"
        assert session["cost_reported"] == pytest.approx(0.25)

    def test_whole_file_rewrite_is_reparsed(
        self, conn: sqlite3.Connection, source: DiscoveredSource
    ) -> None:
        directory = write_session(
            Path(source.location), "s4", manifest=manifest("s4"),
            messages=[assistant("m1", metrics=METRICS, ts=T0)],
        )
        scan(conn, source)
        messages_path = directory / "s4.messages.json"
        payload = json.loads(messages_path.read_text())
        payload["messages"].append(
            assistant("m2", metrics=METRICS, ts=T0 + 300_000)
        )
        messages_path.write_text(json.dumps(payload))
        report = scan(conn, source)
        assert report.usage_inserted == 1
        assert one(conn, "SELECT COUNT(*) FROM usage_events")[0][0] == 2

    def test_full_rescan_never_wipes_unchanged_sessions(
        self, conn: sqlite3.Connection, source: DiscoveredSource
    ) -> None:
        """A full scan re-reads unchanged whole-file stores; its reconcile
        step must therefore still see every event (a pre-fix bug wiped
        whole-file sources on `toknt scan --full` because unchanged files
        produced an empty seen-id set)."""
        write_session(
            Path(source.location), "s5", manifest=manifest("s5"),
            messages=[assistant("m1", metrics=METRICS, ts=T0)],
        )
        scan(conn, source, full=True)
        assert one(conn, "SELECT COUNT(*) FROM usage_events")[0][0] == 1
        # second full scan: files unchanged, but the reconcile must still
        # see the event ids and keep them
        scan(conn, source, full=True)
        assert one(conn, "SELECT COUNT(*) FROM usage_events")[0][0] == 1
        assert one(conn, "SELECT COUNT(*) FROM sessions")[0][0] == 1


class TestVSCodeTasks:
    def test_sessions_and_tools_only(self, conn: sqlite3.Connection, tmp_path: Path) -> None:
        root = tmp_path / "tasks"
        task = root / "task-1"
        task.mkdir(parents=True)
        (task / "api_conversation_history.json").write_text(json.dumps([
            {"role": "user", "content": [{"type": "text", "text": "go"}], "ts": T0},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "execute_command",
                 "input": {"command": "ls"}}], "ts": T0 + 1000},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "is_error": False}], "ts": T0 + 2000},
        ]), "utf-8")
        source = DiscoveredSource(
            source_id="roo:tasks", agent=Harness.ROO, kind="json_dir",
            location=str(root), exists=True,
        )
        ingestion.ensure_source(
            conn, source_id=source.source_id, agent=source.agent, kind=source.kind,
            location=source.location, status="scanning",
        )
        adapter = next(a for a in registered_adapters() if a.source_id == "roo:tasks")
        report = adapter.scan(conn, source, full=True)
        assert report.usage_inserted == 0  # the VS Code format has no usage
        assert report.tool_inserted == 1
        assert report.sessions_written == 1
        quality = one(conn, "SELECT quality, quality_notes FROM sessions")[0]
        assert quality["quality"] == "partial"
        assert "no token counts" in quality["quality_notes"]
        tool = one(conn, "SELECT outcome FROM tool_events")[0]
        assert tuple(tool) == ("success",)
