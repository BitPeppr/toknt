"""Amp adapter (Sourcegraph thread files).

Amp stores one JSON object per thread under the XDG data directory
(``~/.local/share/amp/threads/*.json``). Current builds only materialize
thread files when a plugin exports them (or for older builds, always); the
files are rewritten whole, so scans re-parse on mtime/size changes with
stable ids keeping re-upserts idempotent.

Threads carry ``id``, ``title``, ``created`` (epoch ms), a
``env.initial.trees[0].uri`` workspace hint, and a ``messages`` array:
``user`` messages are stamped with ``meta.sentAt`` (epoch ms), ``assistant``
messages with ``usage`` — whose ``timestamp`` (ISO 8601) anchors the event.
``tool_use`` blocks pair with ``tool_result`` blocks carrying a ``run``
object (``result.output``, ``result.exitCode``, ``error``).

Token semantics: Amp follows the models.dev usage convention (verified for
the sibling pi adapter against a live corpus): ``input``/``output`` exclude
cache reads and writes, which are reported separately as
``cacheReadInputTokens``/``cacheCreationInputTokens`` — the additive
anthropic scheme. When a thread file carries no usage at all (common for
exported threads), the session and its tools still count and the missing
tokens stay missing.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from toknt.adapters.base import (
    Adapter,
    ScanReport,
    as_dict,
    categorize_tool,
    normalize_epoch_ms,
    opt_str,
    parse_iso8601_ms,
    token_int,
)
from toknt.adapters.cline_like import _load_json, _save_whole_file, _whole_file_changed
from toknt.discovery import DiscoveredSource
from toknt.models.events import (
    Harness,
    Outcome,
    OutcomeProvenance,
    Quality,
    SessionRecord,
    ToolEvent,
    UsageEvent,
    stable_event_id,
)
from toknt.models.usage import TokenCounts, TokenScheme
from toknt.storage.ingestion import (
    clear_quality_issues,
    reconcile_source_events,
    record_quality_issue,
    upsert_sessions,
    upsert_tool_events,
    upsert_usage_events,
)


class AmpAdapter(Adapter):
    source_id: ClassVar[str] = "amp:threads"
    agent: ClassVar[Harness] = Harness.AMP
    kind: ClassVar[str] = "json_dir"

    def discover(
        self,
        *,
        env: dict[str, str] | None = None,
        home: object | None = None,
        override: str | None = None,
    ) -> DiscoveredSource:
        from toknt.discovery import discover_amp

        return discover_amp(env=env, home=home, override=override)

    def scan(
        self,
        conn: sqlite3.Connection,
        source: DiscoveredSource,
        *,
        full: bool = False,
    ) -> ScanReport:
        mode = "full" if full else "incremental"
        root = Path(source.location)
        if not root.is_dir():
            return ScanReport(
                source_id=self.source_id,
                mode=mode,
                status="error",
                errors=1,
                notes=(f"threads directory not found: {root}",),
            )
        if full:
            clear_quality_issues(conn, source_id=self.source_id)

        usage_events: list[UsageEvent] = []
        tool_events: list[ToolEvent] = []
        sessions: dict[str, SessionRecord] = {}
        rows_seen = skipped = errors = files_scanned = 0
        notes: list[str] = []

        for path in sorted(root.glob("*.json")):
            relpath = path.relative_to(root).as_posix()
            changed, _checkpoint = _whole_file_changed(
                conn, self.source_id, relpath, path, full=full
            )
            if not changed:
                continue
            try:
                thread = as_dict(_load_json(path))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._quality_issue(
                    conn,
                    code="unparsable_file",
                    message=f"cannot read thread: {type(exc).__name__}",
                    sample_ref=relpath,
                )
                errors += 1
                continue
            thread_id = opt_str(thread.get("id")) or path.stem
            files_scanned += 1
            outcome = self._process_thread(thread, thread_id=thread_id)
            rows_seen += outcome.rows_seen
            skipped += outcome.skipped
            usage_events.extend(outcome.usage)
            tool_events.extend(outcome.tools)
            if outcome.session is not None:
                sessions[thread_id] = outcome.session
            _save_whole_file(conn, self.source_id, relpath, path)

        usage_stats = upsert_usage_events(conn, usage_events)
        tool_stats = upsert_tool_events(conn, tool_events)
        session_stats = upsert_sessions(conn, sessions.values())
        reconciled = 0
        if full:
            reconciled += reconcile_source_events(
                conn,
                source_id=self.source_id,
                table="usage_events",
                seen_ids=[event.id for event in usage_events],
            )
            reconciled += reconcile_source_events(
                conn,
                source_id=self.source_id,
                table="tool_events",
                seen_ids=[event.id for event in tool_events],
            )
        if files_scanned == 0:
            summary = "no thread files found"
        else:
            summary = f"{files_scanned} threads read"
        return ScanReport(
            source_id=self.source_id,
            mode=mode,
            status="partial" if errors else "ok",
            rows_seen=rows_seen,
            usage_inserted=usage_stats.inserted,
            usage_updated=usage_stats.updated,
            tool_inserted=tool_stats.inserted,
            tool_updated=tool_stats.updated,
            sessions_written=session_stats.total,
            skipped=skipped,
            errors=errors,
            reconciled=reconciled,
            notes=(summary, *notes),
        )

    def _process_thread(self, thread: dict, *, thread_id: str) -> _ThreadOutcome:
        result = _ThreadOutcome()
        messages = thread.get("messages")
        created = normalize_epoch_ms(thread.get("created"))
        project = _project_from_uri(thread)
        last_ts = created
        pending: dict[str, dict] = {}

        for index, message in enumerate(as_list(messages)):
            if not isinstance(message, dict):
                continue
            result.rows_seen += 1
            role = opt_str(message.get("role"))
            ts_ms = (
                normalize_epoch_ms(as_dict(message.get("meta")).get("sentAt"))
                or parse_iso8601_ms(as_dict(message.get("usage")).get("timestamp"))
                or last_ts
            )
            if ts_ms:
                last_ts = ts_ms

            for block in as_list(message.get("content")):
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                call_id = opt_str(block.get("id")) or opt_str(block.get("tool_use_id"))
                if block_type == "tool_use" and ts_ms is not None:
                    name = opt_str(block.get("name")) or "unknown"
                    key = call_id or f"{thread_id}:tool-{index}"
                    pending[key] = {"name": name, "ts_ms": ts_ms}
                    result.tools.append(
                        _tool_event(
                            key,
                            name=name,
                            start_ms=ts_ms,
                            end_ms=None,
                            run=None,
                            model=None,
                        )
                    )
                elif block_type == "tool_result":
                    key = call_id or f"{thread_id}:tool-{index}"
                    call = pending.pop(key, None)
                    start_ms = token_int(as_dict(call).get("ts_ms")) or ts_ms
                    if start_ms is None:
                        result.skipped += 1
                        continue
                    result.tools.append(
                        _tool_event(
                            key,
                            name=opt_str(as_dict(call).get("name")) or "unknown",
                            start_ms=start_ms,
                            end_ms=ts_ms,
                            run=block.get("run"),
                            model=opt_str(as_dict(call).get("model")),
                        )
                    )

            if role == "assistant":
                usage = as_dict(message.get("usage"))
                if usage and ts_ms is not None:
                    event = self._usage_event(
                        usage, thread_id=thread_id, index=index, ts_ms=ts_ms
                    )
                    if event is not None:
                        result.usage.append(event)
                elif usage:
                    result.skipped += 1

        result.session = SessionRecord(
            agent=self.agent,
            session_id=thread_id,
            source_id=self.source_id,
            project=project,
            directory=project,
            title=opt_str(thread.get("title")),
            time_created_ms=created,
            time_updated_ms=last_ts,
        )
        return result

    def _usage_event(
        self, usage: dict, *, thread_id: str, index: int, ts_ms: int
    ) -> UsageEvent | None:
        input_tokens = token_int(usage.get("inputTokens"))
        output_tokens = token_int(usage.get("outputTokens"))
        if input_tokens is None and output_tokens is None:
            return None
        counts = TokenCounts(
            input=input_tokens,
            output=output_tokens,
            reasoning=0,  # reasoning is billed inside output in this convention
            cache_read=token_int(usage.get("cacheReadInputTokens")),
            cache_write=token_int(usage.get("cacheCreationInputTokens")),
        )
        return _finalize_usage(
            UsageEvent(
                id=stable_event_id(
                    kind="usage",
                    agent=str(self.agent),
                    source_event_id=f"{thread_id}:msg-{index}",
                ),
                agent=self.agent,
                source_id=self.source_id,
                source_event_id=f"{thread_id}:msg-{index}",
                session_id=thread_id,
                ts_ms=ts_ms,
                tokens=counts,
                token_scheme=TokenScheme.ANTHROPIC,
                provider_total_tokens=token_int(usage.get("totalTokens")),
                source_event_type="thread.message",
            )
        )

    def _quality_issue(
        self, conn: sqlite3.Connection, *, code: str, message: str, sample_ref: str
    ) -> None:
        record_quality_issue(
            conn,
            agent=self.agent,
            source_id=self.source_id,
            code=code,
            message=message,
            sample_ref=sample_ref,
        )


def _project_from_uri(thread: dict) -> str | None:
    env = as_dict(thread.get("env"))
    initial = as_dict(env.get("initial"))
    trees = initial.get("trees")
    if isinstance(trees, list) and trees:
        uri = as_dict(trees[0]).get("uri")
        if isinstance(uri, str) and uri.startswith("file://"):
            path = uri[len("file://"):]
            return path.rstrip("/") or None
    return None


def _tool_event(
    key: str,
    *,
    name: str,
    start_ms: int,
    end_ms: int | None,
    run: object,
    model: str | None,
) -> ToolEvent:
    run_dict = as_dict(run)
    result = as_dict(run_dict.get("result"))
    outcome = Outcome.UNKNOWN
    confidence: float | None = None
    if end_ms is not None:
        exit_code = result.get("exitCode")
        if isinstance(exit_code, int) and not isinstance(exit_code, bool):
            outcome = Outcome.SUCCESS if exit_code == 0 else Outcome.FAILURE
            confidence = 0.9
        elif result.get("error"):
            outcome = Outcome.FAILURE
            confidence = 0.8
        elif result.get("output"):
            outcome = Outcome.SUCCESS
            confidence = 0.7
    elif end_ms is None:
        outcome = Outcome.PENDING
    duration = end_ms - start_ms if end_ms is not None and end_ms >= start_ms else None
    return ToolEvent(
        id=stable_event_id(kind="tool", agent=str(Harness.AMP), source_event_id=key),
        agent=Harness.AMP,
        source_id="amp:threads",
        source_event_id=key,
        call_id=key if key.startswith("call") or "-" in key else None,
        tool_name=name,
        tool_category=categorize_tool(name),
        ts_ms=start_ms,
        duration_ms=duration,
        outcome=outcome,
        outcome_provenance=OutcomeProvenance.RESULT_FIELD
        if outcome in (Outcome.SUCCESS, Outcome.FAILURE)
        else OutcomeProvenance.ABSENT,
        outcome_confidence=confidence,
        model=model,
        source_event_type="thread.tool_use",
    )


def _finalize_usage(event: UsageEvent) -> UsageEvent:
    derived = event.consistency_notes()
    if derived:
        quality = event.quality if event.quality is not Quality.OK else Quality.PARTIAL
        return dataclasses.replace(event, quality=quality, quality_notes=derived)
    return event


def as_list(value: object) -> list:
    return value if isinstance(value, list) else []


@dataclass(slots=True)
class _ThreadOutcome:
    rows_seen: int = 0
    skipped: int = 0
    usage: list[UsageEvent] = field(default_factory=list)
    tools: list[ToolEvent] = field(default_factory=list)
    session: SessionRecord | None = None
