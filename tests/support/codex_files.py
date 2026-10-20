"""Synthetic Codex rollout files covering both storage generations.

Mirrors the shapes observed in real ``~/.codex/sessions`` files (2026-10):
current-generation ``token_usage_record`` events plus their ``token_count``
mirror, legacy cumulative counters, ``turn_context`` model attribution, tool
calls paired by ``call_id``, subagent thread metadata, and malformed lines.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

MINUTE_MS = 60_000


def iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def session_meta(
    *,
    thread_id: str,
    ts_ms: int,
    cwd: str,
    thread_source: str = "user",
    parent_thread_id: str | None = None,
    nickname: str | None = None,
) -> dict:
    payload: dict = {
        "session_id": parent_thread_id or thread_id,
        "id": thread_id,
        "timestamp": iso(ts_ms),
        "cwd": cwd,
        "originator": "codex-tui",
        "cli_version": "0.160.0",
        "source": "cli",
        "thread_source": thread_source,
        "model_provider": "openai",
    }
    if parent_thread_id is not None:
        payload["parent_thread_id"] = parent_thread_id
        payload["forked_from_id"] = parent_thread_id
    if nickname is not None:
        payload["agent_nickname"] = nickname
        payload["source"] = {"subagent": {"thread_spawn": {"agent_nickname": nickname}}}
    return {"timestamp": iso(ts_ms), "ordinal": 0, "type": "session_meta", "payload": payload}


def turn_context(*, ts_ms: int, turn_id: str, model: str, ordinal: int) -> dict:
    return {
        "timestamp": iso(ts_ms),
        "ordinal": ordinal,
        "type": "turn_context",
        "payload": {"turn_id": turn_id, "cwd": "/work/x", "model": model},
    }


def usage_record(
    *,
    ordinal: int,
    ts_ms: int,
    thread_id: str,
    session_id: str,
    turn_id: str,
    response_id: str,
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int,
    cached_tokens: int,
    thread_total: int,
) -> dict:
    usage = {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_tokens,
        "cache_write_input_tokens": 0,
        "output_tokens": output_tokens,
        "reasoning_output_tokens": reasoning_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    return {
        "timestamp": iso(ts_ms),
        "ordinal": ordinal,
        "type": "token_usage_record",
        "payload": {
            "thread_id": thread_id,
            "turn_id": turn_id,
            "session_id": session_id,
            "root_turn_id": turn_id,
            "response_id": response_id,
            "usage": usage,
            "thread_token_usage": {"total_tokens": thread_total},
        },
    }


def token_count(
    *,
    ordinal: int,
    ts_ms: int,
    total: dict | None,
    last: dict | None,
) -> dict:
    return {
        "timestamp": iso(ts_ms),
        "ordinal": ordinal,
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": (
                None
                if total is None and last is None
                else {"total_token_usage": total, "last_token_usage": last}
            ),
        },
    }


def counts(
    *,
    input_: int,
    output: int,
    reasoning: int = 0,
    cached: int = 0,
    total: int | None = None,
) -> dict:
    return {
        "input_tokens": input_,
        "cached_input_tokens": cached,
        "cache_write_input_tokens": 0,
        "output_tokens": output,
        "reasoning_output_tokens": reasoning,
        "total_tokens": total if total is not None else input_ + output,
    }


def call(
    *,
    ordinal: int,
    ts_ms: int,
    call_id: str,
    name: str,
    kind: str = "custom_tool_call",
    status: str | None = "completed",
) -> dict:
    payload: dict = {"type": kind, "name": name, "call_id": call_id}
    if kind == "custom_tool_call":
        payload["input"] = "…"
        if status is not None:
            payload["status"] = status
    else:
        payload["arguments"] = "{}"
    return {
        "timestamp": iso(ts_ms),
        "ordinal": ordinal,
        "type": "response_item",
        "payload": payload,
    }


def output(
    *,
    ordinal: int,
    ts_ms: int,
    call_id: str,
    text: str | list,
    kind: str = "custom_tool_call_output",
) -> dict:
    return {
        "timestamp": iso(ts_ms),
        "ordinal": ordinal,
        "type": "response_item",
        "payload": {"type": kind, "call_id": call_id, "output": text},
    }


def write_rollout(root: Path, relpath: str, records: list[dict] | str) -> Path:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(records, str):
        path.write_text(records, encoding="utf-8")
        return path
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return path


def build_codex_sessions(root: Path, *, now_ms: int) -> dict:
    """Write three rollout files (current, subagent, legacy) under ``root``.

    Returns identifiers used by the tests.
    """
    t = now_ms - 2 * 60 * MINUTE_MS

    # ---- current-generation session (root thread) -------------------------
    current_records = [
        session_meta(thread_id="thread-root", ts_ms=t, cwd="/work/alpha"),
        turn_context(ts_ms=t + MINUTE_MS, turn_id="turn-1", model="gpt-6-luna", ordinal=7),
        usage_record(
            ordinal=15,
            ts_ms=t + 2 * MINUTE_MS,
            thread_id="thread-root",
            session_id="thread-root",
            turn_id="turn-1",
            response_id="resp_current_1",
            input_tokens=1000,
            output_tokens=100,
            reasoning_tokens=40,
            cached_tokens=400,
            thread_total=1100,
        ),
        # The mirror event carries different numbers on purpose: ingesting it
        # would be detected as double counting.
        token_count(
            ordinal=16,
            ts_ms=t + 2 * MINUTE_MS + 1000,
            total=counts(input_=9999, output=9999, total=19998),
            last=counts(input_=9999, output=9999, total=19998),
        ),
        call(
            ordinal=20,
            ts_ms=t + 2 * MINUTE_MS + 2000,
            call_id="call_current_a",
            name="exec",
        ),
        output(
            ordinal=21,
            ts_ms=t + 2 * MINUTE_MS + 4000,
            call_id="call_current_a",
            text="Exit code: 0\nWall time: 2.0 seconds\nOutput:\ndone\n",
        ),
        turn_context(ts_ms=t + 3 * MINUTE_MS, turn_id="turn-2", model="gpt-6-luna", ordinal=30),
        usage_record(
            ordinal=31,
            ts_ms=t + 4 * MINUTE_MS,
            thread_id="thread-root",
            session_id="thread-root",
            turn_id="turn-2",
            response_id="resp_current_2",
            input_tokens=2000,
            output_tokens=200,
            reasoning_tokens=50,
            cached_tokens=1500,
            thread_total=3300,
        ),
        call(
            ordinal=35,
            ts_ms=t + 5 * MINUTE_MS,
            call_id="call_current_b",
            name="exec_command",
            kind="function_call",
            status=None,
        ),
        {
            "timestamp": iso(t + 5 * MINUTE_MS + 500),
            "ordinal": 36,
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call_current_b",
                "output": [
                    {"type": "input_text", "text": "Process exited with code 1\nOutput:\nboom\n"}
                ],
            },
        },
        call(
            ordinal=40,
            ts_ms=t + 6 * MINUTE_MS,
            call_id="call_current_c",
            name="write_stdin",
            kind="function_call",
            status=None,
        ),
        # A user-visible failure that only the output text reveals.
        call(
            ordinal=41,
            ts_ms=t + 6 * MINUTE_MS + 500,
            call_id="call_current_d",
            name="exec",
        ),
        output(
            ordinal=42,
            ts_ms=t + 6 * MINUTE_MS + 900,
            call_id="call_current_d",
            text="Script failed\nTraceback…",
        ),
    ]
    current_path = write_rollout(
        root,
        "2026/10/03/rollout-2026-10-03T12-00-00-thread-root.jsonl",
        current_records,
    )

    # ---- subagent session (own thread, parent = root) ---------------------
    sub_records = [
        session_meta(
            thread_id="thread-sub",
            ts_ms=t + 8 * MINUTE_MS,
            cwd="/work/alpha",
            thread_source="subagent",
            parent_thread_id="thread-root",
            nickname="Linnaeus",
        ),
        turn_context(
            ts_ms=t + 8 * MINUTE_MS + MINUTE_MS,
            turn_id="turn-s1",
            model="gpt-6-luna",
            ordinal=6,
        ),
        usage_record(
            ordinal=12,
            ts_ms=t + 9 * MINUTE_MS,
            thread_id="thread-sub",
            session_id="thread-root",
            turn_id="turn-s1",
            response_id="resp_sub_1",
            input_tokens=500,
            output_tokens=50,
            reasoning_tokens=10,
            cached_tokens=0,
            thread_total=550,
        ),
        call(
            ordinal=14,
            ts_ms=t + 9 * MINUTE_MS + 1000,
            call_id="call_sub_a",
            name="apply_patch",
        ),
        output(
            ordinal=15,
            ts_ms=t + 9 * MINUTE_MS + 1200,
            call_id="call_sub_a",
            text="Script completed\nWall time 0.2 seconds\nOutput:\nSuccess.\n",
        ),
    ]
    sub_path = write_rollout(
        root,
        "2026/10/03/rollout-2026-10-03T12-08-00-thread-sub.jsonl",
        sub_records,
    )

    # ---- legacy-generation session ----------------------------------------
    legacy_records = [
        session_meta(thread_id="thread-legacy", ts_ms=t - 20 * MINUTE_MS, cwd="/work/beta"),
        turn_context(
            ts_ms=t - 18 * MINUTE_MS, turn_id="turn-l1", model="gpt-5.5", ordinal=7
        ),
        token_count(
            ordinal=14,
            ts_ms=t - 16 * MINUTE_MS,
            total=counts(input_=1000, output=100, cached=200),
            last=counts(input_=1000, output=100, cached=200),
        ),
        call(
            ordinal=18,
            ts_ms=t - 15 * MINUTE_MS,
            call_id="call_legacy_a",
            name="exec_command",
            kind="function_call",
            status=None,
        ),
        {
            "timestamp": iso(t - 15 * MINUTE_MS + 300),
            "ordinal": 19,
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call_legacy_a",
                "output": "Process exited with code 0\nOutput:\nok\n",
            },
        },
        token_count(
            ordinal=22,
            ts_ms=t - 10 * MINUTE_MS,
            total=counts(input_=2500, output=400, cached=700),
            last=counts(input_=1500, output=300, cached=500),
        ),
        # A counter event without usage details (observed in real files).
        token_count(ordinal=23, ts_ms=t - 9 * MINUTE_MS, total=None, last=None),
        # A malformed trailing line.
    ]
    legacy_path = write_rollout(
        root,
        "2026/06/04/rollout-2026-06-04T14-00-00-thread-legacy.jsonl",
        legacy_records,
    )
    with legacy_path.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")

    return {
        "root": root,
        "current_path": current_path,
        "sub_path": sub_path,
        "legacy_path": legacy_path,
        "threads": {
            "root": "thread-root",
            "sub": "thread-sub",
            "legacy": "thread-legacy",
        },
    }


def append_records(path: Path, records: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
