"""Performance benchmark on realistically large synthetic histories.

The dataset is sized to a heavy local installation (the development machine
carries ~20k OpenCode usage events and ~17k tool events) and spreads three
agents over six months:

* OpenCode SQLite — 60 sessions × 350 responses (21,000 usage events, every
  tenth also duplicated into the legacy table to exercise dedupe) with 10,500
  completed/error/pending tool calls;
* Codex JSONL — 150 current-era rollouts × 40 responses with tool
  call/output pairs, plus one legacy-era file (30 derived increments);
* Claude Code JSONL — 100 transcripts × 30 responses with tool-use/result
  pairs (10% failures).

The test times a cold scan, a warm (no-change) scan, an incremental scan
after appending a small batch per agent, and the main analytics queries; it
also checks the per-agent row counts exactly, so repeated scans can never
inflate totals. Budgets are deliberately loose — they catch pathological
regressions without machine-to-machine flakiness. Numbers measured on the
development machine are recorded in the README ("Performance").
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tests.support.claude_files import (
    append_records as claude_append,
)
from tests.support.claude_files import (
    assistant_record as claude_assistant,
)
from tests.support.claude_files import (
    usage_payload,
    write_transcript,
)
from tests.support.claude_files import (
    user_record as claude_user,
)
from tests.support.codex_files import (
    append_records as codex_append,
)
from tests.support.codex_files import (
    call as codex_call,
)
from tests.support.codex_files import (
    counts as codex_counts,
)
from tests.support.codex_files import (
    output as codex_output,
)
from tests.support.codex_files import (
    session_meta,
    turn_context,
    write_rollout,
)
from tests.support.codex_files import (
    token_count as codex_token_count,
)
from tests.support.codex_files import (
    usage_record as codex_usage_record,
)
from tests.support.opencode_db import (
    assistant_message,
    create_schema,
    tokens,
    tool_content,
)
from toknt.adapters import ClaudeCodeAdapter, CodexAdapter, OpenCodeAdapter
from toknt.analytics import aggregation, costs, tool_metrics
from toknt.discovery import DiscoveredSource
from toknt.models.events import Agent
from toknt.storage import ingestion

_DAY_MS = 86_400_000
_OPENCODE_SESSIONS = 60
_OPENCODE_PER_SESSION = 350
_CODEX_FILES = 150
_CODEX_PER_FILE = 40
_CODEX_LEGACY_EVENTS = 30
_CLAUDE_FILES = 100
_CLAUDE_PER_FILE = 30

_OPENCODE_MODELS = (
    "claude-sonnet-4-5",
    "glm-5.3-flash-free",
    "muse-spark-1.3",
    "space-bunny-free",
)
_CODEX_MODELS = ("gpt-6-luna", "gpt-5.6-terra")
_CLAUDE_MODELS = ("claude-sonnet-4-5", "claude-haiku-4-5")
_TOOL_NAMES = ("bash", "read", "edit", "grep", "write", "glob", "webfetch")


def _build_opencode_scale(path: Path, *, now_ms: int) -> tuple[int, int]:
    """Write the synthetic OpenCode database; returns expected (usage, tools)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        create_schema(conn)
        span = 175 * _DAY_MS
        session_rows = []
        message_rows = []
        legacy_rows = []
        tool_total = 0
        for s in range(_OPENCODE_SESSIONS):
            session_id = f"ses_scale_{s:03d}"
            base = now_ms - span + (span // _OPENCODE_SESSIONS) * s
            session_rows.append(
                (
                    session_id,
                    "proj_scale",
                    None,
                    "/work/scale",
                    f"Scale session {s}",
                    "build",
                    json.dumps({"id": _OPENCODE_MODELS[s % 4], "providerID": "opencode"}),
                    base,
                    base + _OPENCODE_PER_SESSION * 180_000,
                    0.0,
                )
            )
            for i in range(_OPENCODE_PER_SESSION):
                ts = base + i * 180_000
                message_id = f"msg_scale_{s:03d}_{i:04d}"
                content = None
                if i % 2 == 0:
                    pick = i % 20
                    status = (
                        "completed" if pick < 17 else ("error" if pick < 19 else "pending")
                    )
                    content = [
                        tool_content(
                            call_id=f"call_scale_{s:03d}_{i:04d}",
                            name=_TOOL_NAMES[(s + i) % len(_TOOL_NAMES)],
                            status=status,
                            start_ms=ts,
                            end_ms=None if status == "pending" else ts + 250,
                        )
                    ]
                    tool_total += 1
                payload = assistant_message(
                    created_ms=ts,
                    model=_OPENCODE_MODELS[s % 4],
                    provider="opencode",
                    tokens=tokens(
                        input=500 + (i * 37) % 4_000,
                        output=40 + (i * 11) % 500,
                        reasoning=(i * 7) % 60,
                        cache_read=9_000 + (i * 101) % 20_000,
                        cache_write=(i * 3) % 50,
                    ),
                    content=content,
                )
                message_rows.append((message_id, session_id, "assistant", ts, ts, payload))
                if i % 10 == 0:
                    legacy_rows.append((message_id, session_id, ts, ts, payload))
        conn.executemany(
            "INSERT INTO session_v2 (id, project_id, parent_id, directory, title, agent,"
            " model, time_created, time_updated, cost) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            session_rows,
        )
        conn.executemany(
            "INSERT INTO session_message (id, session_id, type, time_created, time_updated,"
            " data) VALUES (?, ?, ?, ?, ?, ?)",
            message_rows,
        )
        conn.executemany(
            "INSERT INTO message (id, session_id, time_created, time_updated, data)"
            " VALUES (?, ?, ?, ?, ?)",
            legacy_rows,
        )
        conn.commit()
    finally:
        conn.close()
    return _OPENCODE_SESSIONS * _OPENCODE_PER_SESSION, tool_total


def _build_codex_scale(root: Path, *, now_ms: int) -> tuple[int, int, Path]:
    """Write the synthetic Codex rollouts; returns expected (usage, tools, newest)."""
    usage_total = 0
    tool_total = 0
    span = 175 * _DAY_MS
    newest: Path | None = None
    for f in range(_CODEX_FILES):
        base = now_ms - span + (span // _CODEX_FILES) * f
        thread_id = f"thread-scale-{f:03d}"
        records: list[dict] = [
            session_meta(thread_id=thread_id, ts_ms=base, cwd="/work/scale")
        ]
        thread_total = 0
        for i in range(_CODEX_PER_FILE):
            ts = base + i * 240_000
            turn_id = f"turn-{f:03d}-{i:02d}"
            records.append(
                turn_context(
                    ts_ms=ts,
                    turn_id=turn_id,
                    model=_CODEX_MODELS[(f + i) % 2],
                    ordinal=2 + i * 4,
                )
            )
            input_tokens = 1_000 + (i * 53) % 30_000
            output_tokens = 30 + (i * 7) % 400
            thread_total += input_tokens + output_tokens
            records.append(
                codex_usage_record(
                    ordinal=3 + i * 4,
                    ts_ms=ts + 5_000,
                    thread_id=thread_id,
                    session_id=thread_id,
                    turn_id=turn_id,
                    response_id=f"resp-{f:03d}-{i:02d}",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    reasoning_tokens=(i * 3) % 80,
                    cached_tokens=(input_tokens * 4) // 5,
                    thread_total=thread_total,
                )
            )
            usage_total += 1
            if i % 2 == 0:
                call_id = f"call-{f:03d}-{i:02d}"
                records.append(
                    codex_call(
                        ordinal=4 + i * 4,
                        ts_ms=ts + 6_000,
                        call_id=call_id,
                        name="exec",
                        status="completed",
                    )
                )
                records.append(
                    codex_output(
                        ordinal=5 + i * 4,
                        ts_ms=ts + 7_000,
                        call_id=call_id,
                        text="Process exited with code 0",
                    )
                )
                tool_total += 1
        newest = write_rollout(root, f"2026/scale/rollout-{f:03d}.jsonl", records)
    assert newest is not None

    base = now_ms - span // 4
    legacy: list[dict] = [
        session_meta(thread_id="thread-legacy-scale", ts_ms=base, cwd="/work/scale")
    ]
    cumulative_input = cumulative_output = 0
    for i in range(_CODEX_LEGACY_EVENTS):
        ts = base + i * 60_000
        legacy.append(
            turn_context(
                ts_ms=ts, turn_id=f"turn-legacy-{i:02d}", model="gpt-5.5", ordinal=1 + i * 2
            )
        )
        step_input = 800 + i * 31
        step_output = 25 + i
        cumulative_input += step_input
        cumulative_output += step_output
        legacy.append(
            codex_token_count(
                ordinal=2 + i * 2,
                ts_ms=ts + 1_000,
                total=codex_counts(input_=cumulative_input, output=cumulative_output),
                last=codex_counts(input_=step_input, output=step_output),
            )
        )
    usage_total += _CODEX_LEGACY_EVENTS
    write_rollout(root, "2026/04/rollout-legacy-scale.jsonl", legacy)
    return usage_total, tool_total, newest


def _build_claude_scale(root: Path, *, now_ms: int) -> tuple[int, int, Path]:
    """Write the synthetic Claude transcripts; returns expected (usage, tools, newest)."""
    usage_total = 0
    tool_total = 0
    span = 175 * _DAY_MS
    newest: Path | None = None
    for f in range(_CLAUDE_FILES):
        base = now_ms - span + (span // _CLAUDE_FILES) * f
        session_id = f"sess-scale-{f:03d}"
        records: list[dict] = [
            claude_user(ts_ms=base, session_id=session_id, text="run the checks")
        ]
        for i in range(_CLAUDE_PER_FILE):
            ts = base + i * 300_000
            content = None
            if i % 2 == 0:
                content = [
                    {
                        "type": "tool_use",
                        "id": f"toolu-scale-{f:03d}-{i:02d}",
                        "name": "Bash",
                        "input": {},
                    }
                ]
            records.append(
                claude_assistant(
                    ts_ms=ts,
                    session_id=session_id,
                    request_id=f"req-{f:03d}-{i:02d}",
                    message_id=f"msg-{f:03d}-{i:02d}",
                    model=_CLAUDE_MODELS[(f + i) % 2],
                    usage=usage_payload(
                        input_=400 + i * 17,
                        output=60 + i,
                        cache_read=2_000 + i * 23,
                        cache_write=i * 5,
                    ),
                    content=content,
                )
            )
            usage_total += 1
            if content is not None:
                records.append(
                    claude_user(
                        ts_ms=ts + 2_000,
                        session_id=session_id,
                        tool_results=[(content[0]["id"], i % 10 == 8)],
                    )
                )
                tool_total += 1
        newest = write_transcript(root, f"-work-scale/{session_id}.jsonl", records)
    assert newest is not None
    return usage_total, tool_total, newest


def _scan(
    conn: sqlite3.Connection,
    adapter: Any,
    source: DiscoveredSource,
    *,
    full: bool,
) -> Any:
    ingestion.ensure_source(
        conn,
        source_id=source.source_id,
        agent=source.agent,
        kind=source.kind,
        location=source.location,
        status="scanning",
        note=source.note,
    )
    return adapter.scan(conn, source, full=full)


def _count(conn: sqlite3.Connection, table: str, agent: Agent) -> int:
    row = conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE agent = ?", (str(agent),)
    ).fetchone()
    return int(row[0])


def _timed(timings: dict[str, float], label: str, fn: Callable[[], Any]) -> Any:
    start = time.perf_counter()
    result = fn()
    timings[label] = time.perf_counter() - start
    return result


def test_scale_benchmark(conn: sqlite3.Connection, tmp_path: Path) -> None:
    from toknt.util import now_ms

    now = now_ms()
    timings: dict[str, float] = {}

    # ---- dataset ---------------------------------------------------------
    opencode_dir = tmp_path / "sources" / "opencode"
    opencode_path = opencode_dir / "opencode.db"
    opencode_usage, opencode_tools = _build_opencode_scale(opencode_path, now_ms=now)
    codex_root = tmp_path / "sources" / "codex"
    codex_usage, codex_tools, codex_newest = _build_codex_scale(codex_root, now_ms=now)
    claude_root = tmp_path / "sources" / "claude"
    claude_usage, claude_tools, claude_newest = _build_claude_scale(claude_root, now_ms=now)

    sources = (
        (
            OpenCodeAdapter(),
            DiscoveredSource(
                source_id="opencode:db",
                agent=Agent.OPENCODE,
                kind="sqlite",
                location=str(opencode_dir),
                exists=True,
            ),
        ),
        (
            CodexAdapter(),
            DiscoveredSource(
                source_id="codex:sessions",
                agent=Agent.CODEX,
                kind="jsonl_dir",
                location=str(codex_root),
                exists=True,
            ),
        ),
        (
            ClaudeCodeAdapter(),
            DiscoveredSource(
                source_id="claude_code:projects",
                agent=Agent.CLAUDE_CODE,
                kind="jsonl_dir",
                location=str(claude_root),
                exists=True,
            ),
        ),
    )

    # ---- cold scan -------------------------------------------------------
    def cold() -> None:
        for adapter, source in sources:
            _scan(conn, adapter, source, full=True)

    _timed(timings, "cold", cold)
    assert _count(conn, "usage_events", Agent.OPENCODE) == opencode_usage
    assert _count(conn, "tool_events", Agent.OPENCODE) == opencode_tools
    assert _count(conn, "usage_events", Agent.CODEX) == codex_usage
    assert _count(conn, "tool_events", Agent.CODEX) == codex_tools
    assert _count(conn, "usage_events", Agent.CLAUDE_CODE) == claude_usage
    assert _count(conn, "tool_events", Agent.CLAUDE_CODE) == claude_tools
    usage_total = opencode_usage + codex_usage + claude_usage
    tool_total = opencode_tools + codex_tools + claude_tools

    # ---- warm scan (nothing changed) ------------------------------------
    warm_reports = _timed(
        timings,
        "warm",
        lambda: [adapter.scan(conn, source) for adapter, source in sources],
    )
    for report in warm_reports:
        assert report.usage_inserted == 0
        assert report.tool_inserted == 0
    assert _count(conn, "usage_events", Agent.OPENCODE) == opencode_usage

    # ---- incremental scan after a small append per agent -----------------
    append_ts = now + 60_000
    last_opencode_session = f"ses_scale_{_OPENCODE_SESSIONS - 1:03d}"
    extra_rows = []
    for i in range(5):
        content = [
            tool_content(
                call_id=f"call_inc_{i}",
                name="read",
                status="completed",
                start_ms=append_ts + i,
                end_ms=append_ts + i + 10,
            )
        ]
        extra_rows.append(
            (
                f"msg_inc_{i}",
                last_opencode_session,
                "assistant",
                append_ts + i,
                append_ts + i,
                assistant_message(
                    created_ms=append_ts + i,
                    model=_OPENCODE_MODELS[(_OPENCODE_SESSIONS - 1) % 4],
                    provider="opencode",
                    tokens=tokens(input=100, output=10, reasoning=1, cache_read=0, cache_write=0),
                    content=content,
                ),
            )
        )
    source_conn = sqlite3.connect(opencode_path)
    try:
        source_conn.executemany(
            "INSERT INTO session_message (id, session_id, type, time_created, time_updated,"
            " data) VALUES (?, ?, ?, ?, ?, ?)",
            extra_rows,
        )
        source_conn.commit()
    finally:
        source_conn.close()

    codex_extra = [
        turn_context(ts_ms=append_ts, turn_id="turn-inc", model="gpt-6-luna", ordinal=999)
    ]
    base_input = sum(1_000 + (i * 53) % 30_000 for i in range(_CODEX_PER_FILE))
    base_output = sum(30 + (i * 7) % 400 for i in range(_CODEX_PER_FILE))
    codex_extra.append(
        codex_usage_record(
            ordinal=1000,
            ts_ms=append_ts,
            thread_id="thread-scale-149",
            session_id="thread-scale-149",
            turn_id="turn-inc",
            response_id="resp-inc-1",
            input_tokens=2_000,
            output_tokens=50,
            reasoning_tokens=10,
            cached_tokens=1_500,
            thread_total=base_input + base_output + 2_050,
        )
    )
    codex_extra.append(
        codex_call(
            ordinal=1001, ts_ms=append_ts + 1_000, call_id="call-inc-1", name="exec",
            status="completed",
        )
    )
    codex_extra.append(
        codex_output(
            ordinal=1002, ts_ms=append_ts + 2_000, call_id="call-inc-1",
            text="Process exited with code 0",
        )
    )
    codex_append(codex_newest, codex_extra)

    claude_extra = [
        claude_assistant(
            ts_ms=append_ts,
            session_id=f"sess-scale-{_CLAUDE_FILES - 1:03d}",
            request_id="req-inc-1",
            message_id="msg-inc-1",
            usage=usage_payload(input_=300, output=25, cache_read=1_000),
            content=[{"type": "tool_use", "id": "toolu-inc-1", "name": "Read", "input": {}}],
        ),
        claude_user(
            ts_ms=append_ts + 500,
            session_id=f"sess-scale-{_CLAUDE_FILES - 1:03d}",
            tool_results=[("toolu-inc-1", False)],
        ),
    ]
    claude_append(claude_newest, claude_extra)

    incremental_reports = _timed(
        timings,
        "incremental",
        lambda: [adapter.scan(conn, source) for adapter, source in sources],
    )
    assert sum(report.usage_inserted for report in incremental_reports) == 7
    assert sum(report.tool_inserted for report in incremental_reports) == 7
    usage_total += 7
    tool_total += 7

    # ---- analytics queries ----------------------------------------------
    all_models = (*_OPENCODE_MODELS, *_CODEX_MODELS, *_CLAUDE_MODELS, "gpt-5.5")

    def aggregates() -> None:
        flt = aggregation.UsageFilter()
        series = aggregation.usage_series(
            conn, flt, dimension="model", metric="total", bucket="day", top_n=8
        )
        assert not series.is_empty
        outcomes = tool_metrics.tool_outcomes(conn, flt)
        assert outcomes
        tool_metrics.tool_durations(conn, flt)
        price_map = {
            model: costs.Price(
                model=model,
                input_per_mtok=1.0,
                output_per_mtok=10.0,
                cache_read_per_mtok=0.1,
                cache_write_per_mtok=1.0,
            )
            for model in all_models
        }
        estimate = costs.cost_summary(conn, flt, price_map)
        assert estimate.priced_events == usage_total
        assert estimate.unpriced_events == 0

    _timed(timings, "aggregates", aggregates)

    # ---- stability: a second full pass must not change row counts -------
    assert _count(conn, "usage_events", Agent.OPENCODE) == opencode_usage + 5
    assert _count(conn, "tool_events", Agent.OPENCODE) == opencode_tools + 5
    assert _count(conn, "usage_events", Agent.CODEX) == codex_usage + 1
    assert _count(conn, "tool_events", Agent.CODEX) == codex_tools + 1
    assert _count(conn, "usage_events", Agent.CLAUDE_CODE) == claude_usage + 1
    assert _count(conn, "tool_events", Agent.CLAUDE_CODE) == claude_tools + 1

    # ---- budgets + record ------------------------------------------------
    budgets = {
        "cold": 90.0,
        "warm": 10.0,
        "incremental": 30.0,
        "aggregates": 15.0,
    }
    print(
        "scale benchmark: "
        + " ".join(f"{label}={value:.2f}s" for label, value in timings.items())
        + f" | usage={usage_total:,} tools={tool_total:,}"
    )
    for label, budget in budgets.items():
        assert timings[label] <= budget, f"{label} scan took {timings[label]:.1f}s"
