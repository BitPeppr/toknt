"""Synthetic OpenCode database builder for adapter tests.

The schema mirrors the subset of tables and columns the adapter reads from a
real OpenCode installation (observed 2026-10): the current v2 tables
(``session_v2`` / ``session_message`` with embedded tool parts), the legacy
v1 tables (``session`` / ``message`` / ``part``) and ``project``.

Everything is deterministic and generated in a temporary directory; tests
never touch the developer's real agent data.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

MINUTE_MS = 60_000

_SCHEMA = """
CREATE TABLE project (
    id TEXT PRIMARY KEY,
    worktree TEXT
);
CREATE TABLE session (
    id TEXT PRIMARY KEY,
    project_id TEXT,
    parent_id TEXT,
    directory TEXT,
    title TEXT,
    agent TEXT,
    model TEXT,
    time_created INTEGER,
    time_updated INTEGER,
    cost REAL
);
CREATE TABLE session_v2 (
    id TEXT PRIMARY KEY,
    project_id TEXT,
    parent_id TEXT,
    fork_session_id TEXT,
    directory TEXT,
    title TEXT,
    agent TEXT,
    model TEXT,
    time_created INTEGER,
    time_updated INTEGER,
    cost REAL
);
CREATE TABLE message (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    time_created INTEGER,
    time_updated INTEGER,
    data TEXT
);
CREATE TABLE session_message (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    type TEXT,
    time_created INTEGER,
    time_updated INTEGER,
    data TEXT
);
CREATE TABLE part (
    id TEXT PRIMARY KEY,
    message_id TEXT,
    session_id TEXT,
    time_created INTEGER,
    time_updated INTEGER,
    data TEXT
);
"""


def create_schema(conn: sqlite3.Connection) -> None:
    """Create the synthetic OpenCode schema (used by other test builders)."""
    conn.executescript(_SCHEMA)


def assistant_message(
    *,
    created_ms: int,
    tokens: dict | None = None,
    model: str = "claude-3-5-sonnet",
    provider: str = "anthropic",
    agent: str = "build",
    cost: float | None = None,
    content: list | None = None,
) -> str:
    """Serialized assistant message ``data`` JSON, shaped like OpenCode's."""
    data: dict = {
        "role": "assistant",
        "time": {"created": created_ms, "completed": created_ms},
        "agent": agent,
        "model": {"id": model, "providerID": provider},
        "finish": "stop",
    }
    if tokens is not None:
        data["tokens"] = tokens
    if cost is not None:
        data["cost"] = cost
    if content is not None:
        data["content"] = content
    return json.dumps(data)


def tokens(
    *,
    input: int | None = 100,
    output: int | None = 50,
    reasoning: int | None = 7,
    cache_read: int | None = 900,
    cache_write: int | None = 3,
) -> dict:
    """Anthropic-style tokens block; ``total`` matches the additive sum."""
    present = [value for value in (input, output, reasoning, cache_read, cache_write) if value]
    return {
        "input": input,
        "output": output,
        "reasoning": reasoning,
        "cache": {"read": cache_read, "write": cache_write},
        "total": sum(present),
    }


def tool_content(
    *,
    call_id: str,
    name: str,
    status: str,
    start_ms: int,
    end_ms: int | None = None,
) -> dict:
    """An embedded tool part inside an assistant message's content."""
    state: dict = {"status": status}
    if start_ms is not None:
        state["time"] = {"start": start_ms}
        if end_ms is not None:
            state["time"]["end"] = end_ms
    if status == "error":
        state["error"] = "command failed"
    elif status == "completed":
        state["output"] = "done"
    return {"type": "tool", "id": call_id, "name": name, "executed": True, "state": state}


def build_opencode_database(path: Path, *, now_ms: int) -> dict[str, dict]:
    """Create a synthetic ``opencode.db`` covering both storage generations.

    Returns a mapping of logical names to ids/timestamps for assertions.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_SCHEMA)

        t = now_ms
        ids: dict[str, dict] = {
            "times": {
                "msg_a": t - 90 * MINUTE_MS,
                "msg_b": t - 80 * MINUTE_MS,
                "msg_c": t - 70 * MINUTE_MS,
                "msg_d": t - 60 * MINUTE_MS,
                "msg_stream": t - 55 * MINUTE_MS,
                "msg_e": t - 50 * MINUTE_MS,
                "msg_old": t - 30 * 24 * 60 * MINUTE_MS,
            },
            "sessions": {"ses_1": "ses_1", "ses_2": "ses_2"},
            "messages": {
                "msg_a": "msg_a",
                "msg_b": "msg_b",
                "msg_c": "msg_c",
                "msg_d": "msg_d",
                "msg_e": "msg_e",
                "msg_old": "msg_old",
                "msg_f": "msg_f",
            },
            "calls": {
                "call_1": "call_1",
                "call_2": "call_2",
                "call_3": "call_3",
                "call_4": "call_4",
                "call_5": "call_5",
                "call_6": "call_6",
            },
        }

        conn.execute(
            "INSERT INTO project (id, worktree) VALUES ('proj_1', '/work/alpha')"
        )
        conn.executemany(
            "INSERT INTO session_v2 (id, project_id, parent_id, directory, title, agent,"
            " model, time_created, time_updated, cost) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "ses_1",
                    "proj_1",
                    None,
                    "/work/alpha",
                    "Main session",
                    "build",
                    json.dumps({"id": "claude-3-5-sonnet", "providerID": "anthropic"}),
                    t - 120 * MINUTE_MS,
                    t - 10 * MINUTE_MS,
                    0.42,
                ),
                (
                    "ses_2",
                    "proj_1",
                    "ses_1",
                    "/work/alpha/sub",
                    None,
                    "task",
                    None,
                    t - 60 * MINUTE_MS,
                    t - 20 * MINUTE_MS,
                    0.01,
                ),
            ],
        )

        # v2 messages (current generation) ------------------------------------
        conn.executemany(
            "INSERT INTO session_message (id, session_id, type, time_created, time_updated,"
            " data) VALUES (?, ?, ?, ?, ?, ?)",
            [
                # a: full components, embedded completed tool call
                (
                    "msg_a",
                    "ses_1",
                    "assistant",
                    t - 90 * MINUTE_MS,
                    t - 89 * MINUTE_MS,
                    assistant_message(
                        created_ms=t - 90 * MINUTE_MS,
                        tokens=tokens(),
                        cost=0.05,
                        content=[
                            tool_content(
                                call_id="call_1",
                                name="bash",
                                status="completed",
                                start_ms=t - 89 * MINUTE_MS,
                                end_ms=t - 89 * MINUTE_MS + 1_200,
                            )
                        ],
                    ),
                ),
                # b: minimal components
                (
                    "msg_b",
                    "ses_1",
                    "assistant",
                    t - 80 * MINUTE_MS,
                    t - 80 * MINUTE_MS,
                    assistant_message(
                        created_ms=t - 80 * MINUTE_MS,
                        tokens=tokens(input=10, output=5, reasoning=0, cache_read=0, cache_write=0),
                    ),
                ),
                # c: also present in the legacy table (same id -> must dedupe),
                # embedded failing tool call
                (
                    "msg_c",
                    "ses_1",
                    "assistant",
                    t - 70 * MINUTE_MS,
                    t - 70 * MINUTE_MS,
                    assistant_message(
                        created_ms=t - 70 * MINUTE_MS,
                        tokens=tokens(
                            input=20, output=10, reasoning=1, cache_read=0, cache_write=0
                        ),
                        content=[
                            tool_content(
                                call_id="call_2",
                                name="edit",
                                status="error",
                                start_ms=t - 70 * MINUTE_MS,
                            )
                        ],
                    ),
                ),
                # d: missing reasoning (partial components) + unknown tool status
                (
                    "msg_d",
                    "ses_1",
                    "assistant",
                    t - 60 * MINUTE_MS,
                    t - 60 * MINUTE_MS,
                    assistant_message(
                        created_ms=t - 60 * MINUTE_MS,
                        tokens={
                            "input": 5,
                            "output": 2,
                            "cache": {"read": 0, "write": 0},
                            "total": 7,
                        },
                        content=[
                            tool_content(
                                call_id="call_6",
                                name="mystery",
                                status="quantum",
                                start_ms=t - 60 * MINUTE_MS,
                            )
                        ],
                    ),
                ),
                # streaming message: no tokens yet, pending tool call
                (
                    "msg_stream",
                    "ses_1",
                    "assistant",
                    t - 55 * MINUTE_MS,
                    t - 55 * MINUTE_MS,
                    assistant_message(
                        created_ms=t - 55 * MINUTE_MS,
                        content=[
                            tool_content(
                                call_id="call_3",
                                name="todowrite",
                                status="pending",
                                start_ms=t - 55 * MINUTE_MS,
                            )
                        ],
                    ),
                ),
                # subagent session message, embedded tool call duplicated in part table
                (
                    "msg_e",
                    "ses_2",
                    "assistant",
                    t - 50 * MINUTE_MS,
                    t - 50 * MINUTE_MS,
                    assistant_message(
                        created_ms=t - 50 * MINUTE_MS,
                        agent="task",
                        tokens=tokens(
                            input=20, output=10, reasoning=0, cache_read=0, cache_write=0
                        ),
                        content=[
                            tool_content(
                                call_id="call_5",
                                name="bash",
                                status="completed",
                                start_ms=t - 50 * MINUTE_MS,
                                end_ms=t - 50 * MINUTE_MS + 500,
                            )
                        ],
                    ),
                ),
                # malformed JSON payload
                (
                    "msg_malformed",
                    "ses_1",
                    "assistant",
                    t - 40 * MINUTE_MS,
                    t - 40 * MINUTE_MS,
                    "{not json",
                ),
                # user turn: no tokens, ignored by usage but present
                (
                    "msg_user",
                    "ses_1",
                    "user",
                    t - 95 * MINUTE_MS,
                    t - 95 * MINUTE_MS,
                    json.dumps({"text": "hi"}),
                ),
            ],
        )

        # legacy table: msg_c duplicated, msg_old legacy-only ------------------
        old_ts = t - 30 * 24 * 60 * MINUTE_MS
        conn.executemany(
            "INSERT INTO message (id, session_id, time_created, time_updated, data)"
            " VALUES (?, ?, ?, ?, ?)",
            [
                (
                    "msg_c",
                    "ses_1",
                    t - 70 * MINUTE_MS,
                    t - 70 * MINUTE_MS,
                    assistant_message(
                        created_ms=t - 70 * MINUTE_MS,
                        tokens=tokens(
                            input=20, output=10, reasoning=1, cache_read=0, cache_write=0
                        ),
                    ),
                ),
                (
                    "msg_old",
                    "ses_1",
                    old_ts,
                    old_ts,
                    assistant_message(
                        created_ms=old_ts,
                        tokens=tokens(input=1, output=1, reasoning=1, cache_read=0, cache_write=0),
                    ),
                ),
            ],
        )

        # legacy parts table ---------------------------------------------------
        conn.executemany(
            "INSERT INTO part (id, message_id, session_id, time_created, time_updated, data)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    "part_1",
                    "msg_old",
                    "ses_1",
                    old_ts,
                    old_ts,
                    json.dumps(
                        {
                            "type": "tool",
                            "tool": "bash",
                            "callID": "call_4",
                            "state": {"status": "completed"},
                        }
                    ),
                ),
                # same call as msg_e's embedded part -> dedupe by call id
                (
                    "part_2",
                    "msg_e",
                    "ses_2",
                    t - 50 * MINUTE_MS,
                    t - 50 * MINUTE_MS,
                    json.dumps(
                        {
                            "type": "tool",
                            "tool": "bash",
                            "callID": "call_5",
                            "state": {"status": "completed", "output": "done"},
                        }
                    ),
                ),
                # non-tool part -> ignored
                (
                    "part_3",
                    "msg_a",
                    "ses_1",
                    t - 90 * MINUTE_MS,
                    t - 90 * MINUTE_MS,
                    json.dumps({"type": "text", "text": "hello"}),
                ),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return ids


def add_old_message(path: Path, *, now_ms: int) -> None:
    """Insert ``msg_f`` dated far outside any incremental window."""
    ts = now_ms - 400 * 24 * 60 * MINUTE_MS
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "INSERT INTO session_message (id, session_id, type, time_created, time_updated,"
            " data) VALUES (?, ?, ?, ?, ?, ?)",
            (
                "msg_f",
                "ses_1",
                "assistant",
                ts,
                ts,
                assistant_message(
                    created_ms=ts,
                    tokens=tokens(input=2, output=2, reasoning=0, cache_read=0, cache_write=0),
                ),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def update_message_tokens(path: Path, *, message_id: str, input_tokens: int, now_ms: int) -> None:
    """Rewrite a message's tokens and bump ``time_updated`` to now."""
    conn = sqlite3.connect(path)
    try:
        row = conn.execute(
            "SELECT data FROM session_message WHERE id = ?", (message_id,)
        ).fetchone()
        data = json.loads(row[0])
        data["tokens"] = {
            "input": input_tokens,
            "output": 5,
            "reasoning": 0,
            "cache": {"read": 0, "write": 0},
            "total": input_tokens + 5,
        }
        conn.execute(
            "UPDATE session_message SET data = ?, time_updated = ? WHERE id = ?",
            (json.dumps(data), now_ms, message_id),
        )
        conn.commit()
    finally:
        conn.close()


def delete_message(path: Path, *, message_id: str) -> None:
    """Remove a message from the source database (both generations)."""
    conn = sqlite3.connect(path)
    try:
        conn.execute("DELETE FROM session_message WHERE id = ?", (message_id,))
        conn.execute("DELETE FROM message WHERE id = ?", (message_id,))
        conn.execute("DELETE FROM part WHERE message_id = ?", (message_id,))
        conn.commit()
    finally:
        conn.close()


def build_legacy_file_storage(root: Path, *, now_ms: int) -> dict[str, str]:
    """Create ``storage/message/<session>/<message>.json`` files (no database)."""
    directory = root / "storage" / "message" / "ses_file"
    directory.mkdir(parents=True, exist_ok=True)
    ts = now_ms - 45 * MINUTE_MS
    payload = {
        "id": "msg_file_1",
        "role": "assistant",
        "sessionID": "ses_file",
        "agent": "build",
        "modelID": "claude-3-5-sonnet",
        "providerID": "anthropic",
        "time": {"created": ts},
        "cost": 0.02,
        "tokens": tokens(input=11, output=4, reasoning=2, cache_read=0, cache_write=0),
    }
    path = directory / "msg_file_1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    user_payload = {"id": "msg_file_2", "role": "user", "time": {"created": ts}}
    (directory / "msg_file_2.json").write_text(json.dumps(user_payload), encoding="utf-8")
    return {"message_file": str(path), "session_id": "ses_file", "message_id": "msg_file_1"}
