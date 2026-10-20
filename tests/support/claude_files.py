"""Synthetic Claude Code project transcripts.

Mirrors the documented Claude Code JSONL layout: one transcript file per
session under ``~/.claude/projects/<project-slug>/`` with ``assistant`` lines
carrying ``message.usage`` and tool-use blocks, ``user`` lines carrying
``tool_result`` blocks, streaming duplicate snapshots, sidechain lines,
subagent transcripts under ``<session>/subagents/``, API-error lines and a
malformed trailing line.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.support.codex_files import MINUTE_MS, iso

SECOND_MS = 1_000

__all__ = [
    "MINUTE_MS",
    "SECOND_MS",
    "append_records",
    "assistant_record",
    "build_claude_projects",
    "usage_payload",
    "user_record",
    "write_transcript",
]


def usage_payload(
    *,
    input_: int,
    output: int,
    cache_read: int = 0,
    cache_write: int = 0,
) -> dict:
    return {
        "input_tokens": input_,
        "cache_creation_input_tokens": cache_write,
        "cache_read_input_tokens": cache_read,
        "output_tokens": output,
    }


def assistant_record(
    *,
    ts_ms: int,
    session_id: str,
    request_id: str | None = None,
    message_id: str | None = None,
    uuid: str | None = None,
    model: str | None = "claude-sonnet-4-5",
    usage: dict | None = None,
    omit_usage: bool = False,
    content: list | None = None,
    is_sidechain: bool = False,
    api_error: bool = False,
    cost: float | None = None,
    cwd: str = "/Users/dev/project-gamma",
    extra: dict | None = None,
) -> dict:
    message: dict = {
        "role": "assistant",
        "content": content if content is not None else [{"type": "text", "text": "…"}],
    }
    if model is not None:
        message["model"] = model
    if not omit_usage:
        message["usage"] = (
            usage if usage is not None else usage_payload(input_=10, output=5)
        )
    if message_id is not None:
        message["id"] = message_id
    record: dict = {
        "type": "assistant",
        "message": message,
        "timestamp": iso(ts_ms),
        "sessionId": session_id,
        "cwd": cwd,
        "uuid": uuid or f"a-{ts_ms}",
    }
    if request_id is not None:
        record["requestId"] = request_id
    if is_sidechain:
        record["isSidechain"] = True
    if api_error:
        record["isApiErrorMessage"] = True
    if cost is not None:
        record["costUSD"] = cost
    if extra:
        record.update(extra)
    return record


def user_record(
    *,
    ts_ms: int,
    session_id: str,
    tool_results: list[tuple[str, bool]] | None = None,
    text: str | None = None,
    is_sidechain: bool = False,
    cwd: str = "/Users/dev/project-gamma",
) -> dict:
    content: list = []
    if text is not None:
        content.append({"type": "text", "text": text})
    for tool_use_id, is_error in tool_results or []:
        content.append(
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": "…",
                "is_error": is_error,
            }
        )
    return {
        "type": "user",
        "message": {"role": "user", "content": content},
        "timestamp": iso(ts_ms),
        "sessionId": session_id,
        "cwd": cwd,
        "uuid": f"u-{ts_ms}",
        **({"isSidechain": True} if is_sidechain else {}),
    }


def tool_use(name: str, tool_use_id: str) -> dict:
    return {"type": "tool_use", "id": tool_use_id, "name": name, "input": {}}


def write_transcript(root: Path, relpath: str, records: list[dict] | str) -> Path:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(records, str):
        path.write_text(records, encoding="utf-8")
        return path
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return path


def append_records(path: Path, records: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def build_claude_projects(root: Path, *, now_ms: int) -> dict:
    """Write a main session and a subagent transcript under ``root``."""
    t = now_ms - 30 * MINUTE_MS
    project = "-Users-dev-project-gamma"
    session = "sess-main-1"

    main_records = [
        user_record(ts_ms=t, session_id=session, text="run the checks"),
        # Streaming wrote the same response twice; the later snapshot is the
        # final one and carries the reported cost.
        assistant_record(
            ts_ms=t + MINUTE_MS,
            session_id=session,
            request_id="req_main_1",
            message_id="msg_main_1",
            usage=usage_payload(input_=50, output=10, cache_read=100),
        ),
        assistant_record(
            ts_ms=t + MINUTE_MS + 5 * SECOND_MS,
            session_id=session,
            request_id="req_main_1",
            message_id="msg_main_1",
            usage=usage_payload(input_=100, output=300, cache_read=5000, cache_write=2000),
            cost=0.42,
        ),
        assistant_record(
            ts_ms=t + 2 * MINUTE_MS,
            session_id=session,
            request_id="req_main_2",
            message_id="msg_main_2",
            usage=usage_payload(input_=10, output=50, cache_read=7000),
            content=[tool_use("Bash", "toolu_1")],
        ),
        user_record(
            ts_ms=t + 2 * MINUTE_MS + 30 * SECOND_MS,
            session_id=session,
            tool_results=[("toolu_1", False)],
        ),
        assistant_record(
            ts_ms=t + 3 * MINUTE_MS,
            session_id=session,
            request_id="req_main_3",
            message_id="msg_main_3",
            usage=usage_payload(input_=20, output=40, cache_read=8000),
            content=[tool_use("Edit", "toolu_2")],
        ),
        user_record(
            ts_ms=t + 3 * MINUTE_MS + 20 * SECOND_MS,
            session_id=session,
            tool_results=[("toolu_2", True)],
        ),
        assistant_record(
            ts_ms=t + 4 * MINUTE_MS,
            session_id=session,
            request_id="req_main_4",
            message_id="msg_main_4",
            usage=usage_payload(input_=30, output=60, cache_read=9000),
            content=[tool_use("Read", "toolu_3")],
        ),
        assistant_record(
            ts_ms=t + 5 * MINUTE_MS,
            session_id=session,
            request_id="req_main_5",
            message_id="msg_main_5",
            usage=usage_payload(input_=5, output=15, cache_write=100),
            is_sidechain=True,
        ),
        # Failed API requests carry no usage and must not be flagged.
        assistant_record(
            ts_ms=t + 6 * MINUTE_MS,
            session_id=session,
            request_id="req_main_6",
            message_id="msg_main_6",
            api_error=True,
            omit_usage=True,
        ),
        # A response that should have usage but does not: flagged.
        assistant_record(
            ts_ms=t + 7 * MINUTE_MS,
            session_id=session,
            request_id="req_main_7",
            message_id="msg_main_7",
            omit_usage=True,
        ),
        {"type": "summary", "summary": "…", "leafUuid": "u-1"},
    ]
    main_path = write_transcript(root, f"{project}/{session}.jsonl", main_records)
    with main_path.open("a", encoding="utf-8") as handle:
        handle.write("{broken json\n")

    sub_records = [
        assistant_record(
            ts_ms=t + 8 * MINUTE_MS,
            session_id="sess-sub-1",
            request_id="req_sub_1",
            message_id="msg_sub_1",
            model="claude-haiku-4-5",
            usage=usage_payload(input_=7, output=9, cache_write=50),
            content=[tool_use("Grep", "toolu_sub_1")],
            extra={"agentType": "Explore", "agentId": "agent-abc"},
        ),
        user_record(
            ts_ms=t + 8 * MINUTE_MS + 10 * SECOND_MS,
            session_id="sess-sub-1",
            tool_results=[("toolu_sub_1", False)],
        ),
    ]
    sub_path = write_transcript(
        root, f"{project}/{session}/subagents/agent-explore.jsonl", sub_records
    )

    return {
        "root": root,
        "main_path": main_path,
        "sub_path": sub_path,
        "sessions": {"main": session, "sub": "sess-sub-1"},
    }
