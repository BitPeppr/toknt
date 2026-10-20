"""Grok Build adapter.

Grok keeps one directory of JSON state per session under
``~/.grok/sessions/<url-encoded-cwd>/<session-id>/``:

* ``usage.json`` — session totals plus a ``turns`` array, each turn carrying
  per-turn token counts (``inputTokens``, ``outputTokens``,
  ``cachedReadTokens``, ``cacheCreationTokens``, ``reasoningTokens``,
  ``totalTokens``) and a ``modelUsage`` map of per-model splits with an
  ``endedAt`` timestamp;
* ``summary.json`` — session metadata (id, cwd, created/updated, title);
* ``events.jsonl`` — an append-only event log with explicit tool events
  (``tool_started`` / ``tool_completed`` with ``outcome`` and
  ``duration_ms``).

Token semantics (verified against a real corpus on this machine):
``totalTokens == inputTokens + outputTokens``; ``cachedReadTokens`` is a
subset of input and ``reasoningTokens`` a subset of output — the openai
scheme. ``costUsdTicks`` exists but its tick scale is undocumented, so it is
never converted into a cost figure.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from pathlib import Path
from typing import ClassVar

from toknt.adapters.base import (
    Adapter,
    ScanReport,
    as_dict,
    categorize_tool,
    opt_str,
    parse_iso8601_ms,
    token_int,
)
from toknt.adapters.cline_like import _load_json
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
    load_checkpoint,
    reconcile_source_events,
    record_quality_issue,
    save_checkpoint,
    upsert_sessions,
    upsert_tool_events,
    upsert_usage_events,
)

_TOOL_OUTCOMES: dict[str, Outcome] = {
    "success": Outcome.SUCCESS,
    "completed": Outcome.SUCCESS,
    "failure": Outcome.FAILURE,
    "failed": Outcome.FAILURE,
    "error": Outcome.FAILURE,
    "denied": Outcome.DENIED,
    "cancelled": Outcome.CANCELLED,
    "canceled": Outcome.CANCELLED,
}


class GrokAdapter(Adapter):
    source_id: ClassVar[str] = "grok:sessions"
    agent: ClassVar[Harness] = Harness.GROK
    kind: ClassVar[str] = "json_dir"

    def discover(
        self,
        *,
        env: dict[str, str] | None = None,
        home: object | None = None,
        override: str | None = None,
    ) -> DiscoveredSource:
        from toknt.discovery import discover_grok

        return discover_grok(env=env, home=home, override=override)

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
                notes=(f"sessions directory not found: {root}",),
            )
        if full:
            clear_quality_issues(conn, source_id=self.source_id)

        usage_events: list[UsageEvent] = []
        tool_events: list[ToolEvent] = []
        sessions: dict[str, SessionRecord] = {}
        rows_seen = skipped = errors = files_scanned = 0
        notes: list[str] = []

        for directory in sorted(root.glob("*/*/")):
            usage_path = directory / "usage.json"
            summary_path = directory / "summary.json"
            if not usage_path.is_file() and not summary_path.is_file():
                continue
            relpath = directory.relative_to(root).as_posix()
            changed = (
                full
                or any(
                    self._file_changed(conn, f"{name}:{relpath}", directory / name)
                    for name in ("usage.json", "summary.json")
                )
            )
            if not changed:
                continue
            files_scanned += 1
            try:
                usage = _load_json(usage_path) if usage_path.is_file() else {}
                summary = _load_json(summary_path) if summary_path.is_file() else {}
            except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._quality_issue(
                    conn,
                    code="unparsable_file",
                    message=f"cannot read session state: {type(exc).__name__}",
                    sample_ref=relpath,
                )
                errors += 1
                continue

            session_id = (
                opt_str(as_dict(summary).get("info", {}).get("id")
                        if isinstance(as_dict(summary).get("info"), dict) else None)
                or opt_str(as_dict(usage).get("sessionId"))
                or directory.name
            )
            cwd = opt_str(as_dict(as_dict(summary).get("info")).get("cwd"))
            session = self._session_record(
                as_dict(summary), session_id=session_id, cwd=cwd
            )
            sessions[session_id] = session
            rows_seen += 1

            turn_usage, turn_skipped = self._usage_events(
                as_dict(usage), session_id=session_id, cwd=cwd
            )
            usage_events.extend(turn_usage)
            skipped += turn_skipped
            session_tools, tool_errors = self._tool_events(
                directory, session_id=session_id, cwd=cwd
            )
            tool_events.extend(session_tools)
            errors += tool_errors
            for name in ("usage.json", "summary.json"):
                self._save_file(conn, f"{name}:{relpath}", directory / name)

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
            summary = "no session directories"
        else:
            summary = f"{files_scanned} sessions read"
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

    # ------------------------------------------------------------- generators

    def _usage_events(
        self, usage: dict, *, session_id: str, cwd: str | None
    ) -> tuple[list[UsageEvent], int]:
        """One usage event per model per turn from ``usage.json``."""
        events: list[UsageEvent] = []
        skipped = 0
        turns = usage.get("turns")
        for turn in turns if isinstance(turns, list) else []:
            if not isinstance(turn, dict):
                skipped += 1
                continue
            ts_ms = parse_iso8601_ms(opt_str(turn.get("endedAt")))
            if ts_ms is None:
                skipped += 1
                continue
            model_usage = turn.get("modelUsage")
            if isinstance(model_usage, dict) and model_usage:
                models = model_usage
            elif turn.get("primaryModelId") is not None:
                # fall back to the turn-level aggregate under its primary model
                models = {str(turn["primaryModelId"]): turn}
            else:
                models = {}
            for model, buckets in models.items():
                if not isinstance(buckets, dict):
                    skipped += 1
                    continue
                turn_number = turn.get("turnNumber")
                event_id = f"{session_id}:t{turn_number}:{model}"
                events.append(
                    self._usage_event(
                        buckets,
                        event_id=event_id,
                        session_id=session_id,
                        cwd=cwd,
                        model=str(model),
                        ts_ms=ts_ms,
                    )
                )
        return events, skipped

    def _usage_event(
        self,
        buckets: dict,
        *,
        event_id: str,
        session_id: str,
        cwd: str | None,
        model: str,
        ts_ms: int,
    ) -> UsageEvent:
        counts = TokenCounts(
            input=token_int(buckets.get("inputTokens")),
            output=token_int(buckets.get("outputTokens")),
            reasoning=token_int(buckets.get("reasoningTokens")),
            cache_read=token_int(buckets.get("cachedReadTokens")),
            cache_write=token_int(buckets.get("cacheCreationTokens")),
        )
        return _finalize_usage(
            UsageEvent(
                id=stable_event_id(
                    kind="usage", agent=str(self.agent), source_event_id=event_id
                ),
                agent=self.agent,
                source_id=self.source_id,
                source_event_id=event_id,
                session_id=session_id,
                project=cwd,
                directory=cwd,
                model=model,
                ts_ms=ts_ms,
                tokens=counts,
                token_scheme=TokenScheme.OPENAI,
                provider_total_tokens=token_int(buckets.get("totalTokens")),
                source_event_type="usage.turns",
            )
        )

    def _tool_events(
        self, directory: Path, *, session_id: str, cwd: str | None
    ) -> tuple[list[ToolEvent], int]:
        """Tool calls from the session's ``events.jsonl`` event log."""
        events_path = directory / "events.jsonl"
        if not events_path.is_file():
            return [], 0
        events: list[ToolEvent] = []
        errors = 0
        try:
            lines = events_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            return [], 1
        pending: dict[str, dict] = {}
        finished: set[str] = set()
        for index, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            event_type = opt_str(record.get("type"))
            ts_ms = parse_iso8601_ms(opt_str(record.get("ts")))
            tool_name = opt_str(record.get("tool_name"))
            call_id = opt_str(record.get("tool_call_id"))
            if event_type == "tool_started":
                key = call_id or f"seq-{index}"
                pending[key] = {
                    "name": tool_name or "unknown",
                    "ts_ms": ts_ms,
                }
                continue
            if event_type == "tool_completed":
                key = call_id or f"seq-{index}"
                call = pending.pop(key, None) or {}
                finished.add(key)
                start_ms = token_int(call.get("ts_ms")) or ts_ms
                name = opt_str(call.get("name")) or tool_name or "unknown"
                if start_ms is None:
                    errors += 1
                    continue
                outcome_text = opt_str(record.get("outcome")) or ""
                outcome = _TOOL_OUTCOMES.get(outcome_text, Outcome.UNKNOWN)
                duration = token_int(record.get("duration_ms"))
                if duration is None and ts_ms is not None and ts_ms >= start_ms:
                    duration = ts_ms - start_ms
                events.append(
                    ToolEvent(
                        id=stable_event_id(
                            kind="tool", agent=str(self.agent), source_event_id=key
                        ),
                        agent=self.agent,
                        source_id=self.source_id,
                        source_event_id=key,
                        session_id=session_id,
                        call_id=call_id,
                        tool_name=name,
                        tool_category=categorize_tool(name),
                        ts_ms=start_ms,
                        duration_ms=duration,
                        outcome=outcome,
                        outcome_provenance=OutcomeProvenance.STATUS_FIELD,
                        outcome_confidence=0.9 if outcome is not Outcome.UNKNOWN else None,
                        source_event_type="events.tool_completed",
                    )
                )
        # calls whose completion never arrived stay visible as pending
        for key, call in pending.items():
            if key in finished:
                continue
            start_ms = token_int(call.get("ts_ms"))
            if start_ms is None:
                continue
            events.append(
                ToolEvent(
                    id=stable_event_id(
                        kind="tool", agent=str(self.agent), source_event_id=key
                    ),
                    agent=self.agent,
                    source_id=self.source_id,
                    source_event_id=key,
                    session_id=session_id,
                    call_id=key,
                    tool_name=opt_str(call.get("name")) or "unknown",
                    tool_category=categorize_tool(str(call.get("name"))),
                    ts_ms=start_ms,
                    outcome=Outcome.PENDING,
                    outcome_provenance=OutcomeProvenance.ABSENT,
                    source_event_type="events.tool_started",
                )
            )
        return events, errors

    def _session_record(
        self, summary: dict, *, session_id: str, cwd: str | None
    ) -> SessionRecord:
        info = as_dict(summary.get("info"))
        return SessionRecord(
            agent=self.agent,
            session_id=session_id,
            source_id=self.source_id,
            project=cwd or opt_str(info.get("cwd")),
            directory=cwd or opt_str(info.get("cwd")),
            title=opt_str(summary.get("session_summary")),
            model=opt_str(summary.get("current_model_id")),
            time_created_ms=parse_iso8601_ms(opt_str(summary.get("created_at"))),
            time_updated_ms=parse_iso8601_ms(opt_str(summary.get("updated_at"))),
        )

    # ----------------------------------------------------------------- helpers

    def _file_changed(self, conn: sqlite3.Connection, relpath: str, path: Path) -> bool:
        checkpoint = load_checkpoint(conn, self.source_id, relpath)
        try:
            stat = path.stat()
        except OSError:
            return False
        return not (
            checkpoint is not None
            and checkpoint.size == stat.st_size
            and checkpoint.mtime_ms == stat.st_mtime_ns // 1_000_000
        )

    def _save_file(self, conn: sqlite3.Connection, relpath: str, path: Path) -> None:
        try:
            stat = path.stat()
        except OSError:
            return
        save_checkpoint(
            conn,
            self.source_id,
            relpath,
            offset=0,
            size=stat.st_size,
            mtime_ms=stat.st_mtime_ns // 1_000_000,
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


def _finalize_usage(event: UsageEvent) -> UsageEvent:
    derived = event.consistency_notes()
    if derived:
        quality = event.quality if event.quality is not Quality.OK else Quality.PARTIAL
        return dataclasses.replace(event, quality=quality, quality_notes=derived)
    return event
