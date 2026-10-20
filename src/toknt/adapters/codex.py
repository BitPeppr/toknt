"""Codex CLI adapter.

Codex writes one JSONL rollout file per session under
``~/.codex/sessions/<year>/<month>/<day>/``. Two storage generations appear in
that tree, sometimes side by side:

* **Current**: ``token_usage_record`` events carry per-response ``usage``
  dicts; tool calls are ``custom_tool_call`` / ``function_call`` response
  items paired with their ``*_output`` counterparts by ``call_id``.
* **Legacy**: usage comes from ``event_msg.token_count`` events whose
  ``last_token_usage`` is the per-response amount (older builds only report
  the cumulative ``total_token_usage``, in which case the adapter derives
  increments and marks the event ``derived``).

A file that contains any ``token_usage_record`` uses only those records; the
``token_count`` mirror it also writes is ignored so usage is never counted
twice. If a file scanned before its first ``token_usage_record`` later turns
out to be current-generation, its legacy events are retracted and the file is
re-read from the start.

Model attribution follows ``turn_context.model`` (present in both
generations), falling back to ``thread_settings_applied.thread_settings.model``
for turns that never saw a context record. Subagent threads keep their own
thread id as the canonical session and point at the thread that spawned them.

Only accounting-relevant fields are read: message bodies, tool arguments and
tool outputs are never stored — tool outputs are classified in memory purely
to determine success/failure.
"""

from __future__ import annotations

import dataclasses
import json
import re
import sqlite3
from dataclasses import dataclass, field
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
from toknt.adapters.jsonl import JsonlBatch, read_jsonl_batch
from toknt.discovery import DiscoveredSource, discover_codex
from toknt.models.events import (
    Agent,
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
    delete_source_events,
    load_checkpoint,
    reconcile_source_events,
    record_quality_issue,
    save_checkpoint,
    upsert_sessions,
    upsert_tool_events,
    upsert_usage_events,
)

#: Legacy-generation usage events carry this source id prefix so they can be
#: retracted when a file turns out to be current-generation after all.
_LEGACY_PREFIX = "tc:"

_EXIT_CODE_RE = re.compile(r"Process exited with code (-?\d+)|Exit code:\s*(-?\d+)")
_SCRIPT_STATUS_RE = re.compile(r"^\s*(Script completed|Script failed)", re.M)
_FAILED_LINE_RE = re.compile(r"^\s*[\w.\-]+ (?:failed|errored):", re.M)

_TOOL_STATUS_OUTCOMES: dict[str, Outcome] = {
    "completed": Outcome.SUCCESS,
    "failed": Outcome.FAILURE,
    "error": Outcome.FAILURE,
    "errored": Outcome.FAILURE,
    "cancelled": Outcome.CANCELLED,
    "canceled": Outcome.CANCELLED,
    "in_progress": Outcome.PENDING,
    "pending": Outcome.PENDING,
    "queued": Outcome.PENDING,
}

_CALL_ITEM_TYPES = ("function_call", "custom_tool_call")
_OUTPUT_ITEM_TYPES = ("function_call_output", "custom_tool_call_output")


def _output_text(value: object) -> str | None:
    """Flatten a tool output (string or list of text blocks) to plain text."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for block in value:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else None
    if isinstance(value, dict) and isinstance(value.get("text"), str):
        return value["text"]
    return None


def _classify_output(
    text: str,
) -> tuple[Outcome, OutcomeProvenance, float | None]:
    """Determine a tool outcome from its output text.

    Exit codes are the strongest signal; script wrappers and explicit
    "failed:" headlines are classified with lower confidence. Anything else
    stays ``unknown`` with no evidence instead of guessing.
    """
    match = _EXIT_CODE_RE.search(text)
    if match:
        code = match.group(1) if match.group(1) is not None else match.group(2)
        outcome = Outcome.SUCCESS if code == "0" else Outcome.FAILURE
        return outcome, OutcomeProvenance.OUTPUT_TEXT, 0.8
    match = _SCRIPT_STATUS_RE.search(text)
    if match:
        outcome = (
            Outcome.SUCCESS if match.group(1) == "Script completed" else Outcome.FAILURE
        )
        return outcome, OutcomeProvenance.OUTPUT_TEXT, 0.7
    if _FAILED_LINE_RE.search(text):
        return Outcome.FAILURE, OutcomeProvenance.OUTPUT_TEXT, 0.6
    return Outcome.UNKNOWN, OutcomeProvenance.ABSENT, None


@dataclass(slots=True)
class _Batch:
    """Canonical records extracted from one batch of one rollout file."""

    usage: list[UsageEvent] = field(default_factory=list)
    tools: dict[str, ToolEvent] = field(default_factory=dict)  # keyed by call id
    orphan_outputs: int = 0
    skipped: int = 0
    malformed: int = 0


class CodexAdapter(Adapter):
    source_id: ClassVar[str] = "codex:sessions"
    agent: ClassVar[Agent] = Agent.CODEX
    kind: ClassVar[str] = "jsonl_dir"

    def discover(
        self,
        *,
        env: dict[str, str] | None = None,
        home: object | None = None,
        override: str | None = None,
    ) -> DiscoveredSource:
        return discover_codex(env=env, home=home, override=override)

    # ------------------------------------------------------------------ scan

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
            # A full rescan re-observes everything: stale issues would
            # otherwise survive fixed sources forever.
            clear_quality_issues(conn, source_id=self.source_id)

        paths = sorted(root.rglob("*.jsonl"))
        usage_events: list[UsageEvent] = []
        tool_events: list[ToolEvent] = []
        sessions: dict[str, SessionRecord] = {}
        rows_seen = skipped = errors = files_scanned = 0
        notes: list[str] = []

        for path in paths:
            outcome = self._scan_file(conn, root, path, full=full)
            if outcome is None:
                continue
            files_scanned += 1
            rows_seen += outcome.rows_seen
            skipped += outcome.skipped
            errors += outcome.errors
            usage_events.extend(outcome.usage)
            tool_events.extend(outcome.tools)
            if outcome.session is not None:
                self._merge_session(sessions, outcome.session)
            notes.extend(outcome.notes)

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

        if not paths:
            summary = "no rollout files found"
        elif files_scanned == 0:
            summary = "no changed files"
        else:
            summary = f"{files_scanned} files read"
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

    # -------------------------------------------------------------- one file

    def _scan_file(
        self,
        conn: sqlite3.Connection,
        root: Path,
        path: Path,
        *,
        full: bool,
    ) -> _FileOutcome | None:
        """Scan one rollout file. Returns ``None`` when the file is unchanged."""
        relpath = path.relative_to(root).as_posix()
        notes: list[str] = []
        try:
            stat = path.stat()
        except OSError as exc:
            self._quality_issue(
                conn,
                code="unreadable_file",
                message=f"cannot stat rollout file: {type(exc).__name__}",
                sample_ref=relpath,
            )
            return _FileOutcome(errors=1)

        checkpoint = None if full else load_checkpoint(conn, self.source_id, relpath)
        if (
            checkpoint is not None
            and checkpoint.size == stat.st_size
            and checkpoint.mtime_ms == stat.st_mtime_ns // 1_000_000
        ):
            return None

        state = {} if full else self._load_state(conn, relpath)
        batch = read_jsonl_batch(
            path,
            offset=checkpoint.offset if checkpoint else 0,
            fingerprint=checkpoint.fingerprint if checkpoint else None,
        )
        if batch.reset and state:
            state = {}
        if state.get("era") == "legacy" and any(
            record.get("type") == "token_usage_record" for record in batch.records
        ):
            # The file grew into the current format after we had already
            # ingested its legacy counters: retract them so usage is counted
            # exactly once (buffering in _process_batch skips the mirrors).
            delete_source_events(
                conn,
                source_id=self.source_id,
                table="usage_events",
                reference_prefix=f"{_LEGACY_PREFIX}{path.stem}:",
            )
            notes.append(f"re-ingested {relpath}: storage format changed mid-file")

        result = self._process_batch(state, batch, stem=path.stem)

        if batch.malformed:
            self._quality_issue(
                conn,
                code="unparsable_row",
                message="rollout contained unparsable JSON lines",
                occurrences=len(batch.malformed),
                sample_ref=relpath,
            )
        if result.missing_token_info:
            self._quality_issue(
                conn,
                code="missing_token_info",
                message="token_count event without usage details",
                occurrences=result.missing_token_info,
                sample_ref=relpath,
            )
        if result.batch.orphan_outputs:
            self._quality_issue(
                conn,
                code="orphan_tool_output",
                message="tool output without its matching call record",
                occurrences=result.batch.orphan_outputs,
                sample_ref=relpath,
            )

        save_checkpoint(
            conn,
            self.source_id,
            relpath,
            offset=batch.end_offset,
            size=stat.st_size,
            mtime_ms=stat.st_mtime_ns // 1_000_000,
            fingerprint=batch.fingerprint,
        )
        self._save_state(conn, relpath, state)

        return _FileOutcome(
            rows_seen=len(batch.records),
            skipped=result.batch.skipped + len(batch.malformed),
            errors=len(batch.malformed),
            usage=result.batch.usage,
            tools=list(result.batch.tools.values()),
            session=result.session,
            notes=tuple(notes),
        )

    # ----------------------------------------------------------- state machine

    def _process_batch(self, state: dict, batch: JsonlBatch, *, stem: str) -> _Processed:
        """Translate one batch of rollout records into canonical records.

        ``state`` is updated in place and persisted by the caller.
        """
        result = _Processed()
        current_model = opt_str(state.get("model"))
        pending_calls = {
            str(key): value for key, value in as_dict(state.get("pending")).items()
        }
        last_ts = token_int(state.get("last_ts_ms"))
        # Legacy counters are buffered until we know whether this batch also
        # contains current-format records: current-era files mirror every
        # response into token_count, sometimes *before* the first
        # token_usage_record, and those mirrors must never be ingested.
        legacy_usage: list[UsageEvent] = []
        legacy_missing = 0
        saw_current = state.get("era") == "current"

        for record in batch.records:
            record_type = record.get("type")
            payload = as_dict(record.get("payload"))
            ts_ms = parse_iso8601_ms(record.get("timestamp")) or last_ts

            if record_type == "session_meta":
                self._apply_session_meta(state, payload)
                if ts_ms:
                    last_ts = ts_ms
                continue

            if record_type == "turn_context":
                model = opt_str(payload.get("model"))
                if model:
                    current_model = model
                if ts_ms:
                    last_ts = ts_ms
                continue

            if record_type == "event_msg":
                event_type = payload.get("type")
                if event_type == "thread_settings_applied":
                    model = opt_str(
                        as_dict(payload.get("thread_settings")).get("model")
                    )
                    if model:
                        current_model = model
                    if ts_ms:
                        last_ts = ts_ms
                    continue
                if event_type == "token_count":
                    if saw_current:
                        continue  # mirrored in token_usage_record: skip
                    event, missing = self._legacy_usage_event(
                        state, payload, stem=stem, ts_ms=ts_ms, model=current_model
                    )
                    if event is not None:
                        legacy_usage.append(event)
                        last_ts = event.ts_ms
                    else:
                        legacy_missing += 1 if missing else 0
                    continue
                continue

            if record_type == "token_usage_record":
                saw_current = True
                event = self._usage_event(state, payload, ts_ms=ts_ms, model=current_model)
                if event is not None:
                    result.batch.usage.append(event)
                    last_ts = event.ts_ms
                else:
                    result.batch.skipped += 1
                    result.missing_token_info += 1
                continue

            if record_type == "response_item":
                item_type = payload.get("type")
                if item_type in _CALL_ITEM_TYPES:
                    call_id = opt_str(payload.get("call_id"))
                    if call_id is None or ts_ms is None:
                        result.batch.skipped += 1
                        continue
                    name = opt_str(payload.get("name")) or "unknown"
                    pending_calls[call_id] = {
                        "name": name,
                        "ts_ms": ts_ms,
                        "model": current_model,
                    }
                    result.batch.tools[call_id] = self._tool_event(
                        state,
                        call_id=call_id,
                        name=name,
                        start_ms=ts_ms,
                        end_ms=None,
                        status=opt_str(payload.get("status")),
                        output=None,
                        model=current_model,
                    )
                elif item_type in _OUTPUT_ITEM_TYPES:
                    call_id = opt_str(payload.get("call_id"))
                    if call_id is None:
                        result.batch.skipped += 1
                        continue
                    call = as_dict(pending_calls.pop(call_id, None))
                    if not call:
                        result.batch.orphan_outputs += 1
                        result.batch.skipped += 1
                        continue
                    start_ms = token_int(call.get("ts_ms"))
                    result.batch.tools[call_id] = self._tool_event(
                        state,
                        call_id=call_id,
                        name=opt_str(call.get("name")) or "unknown",
                        start_ms=start_ms or ts_ms or 1,
                        end_ms=ts_ms,
                        status=None,
                        output=_output_text(payload.get("output")),
                        model=opt_str(call.get("model")),
                    )
            if ts_ms:
                last_ts = ts_ms

        if batch.malformed:
            result.batch.malformed = len(batch.malformed)

        if saw_current:
            state["era"] = "current"
        elif legacy_usage or legacy_missing:
            state["era"] = "legacy"
            result.batch.usage.extend(legacy_usage)
            result.batch.skipped += legacy_missing
            result.missing_token_info += legacy_missing

        state["model"] = current_model
        state["pending"] = pending_calls
        state["last_ts_ms"] = last_ts

        # Session cache record: touched by every event this batch produced.
        session_id = opt_str(state.get("session_id"))
        if session_id is not None:
            event_ts = [
                *[event.ts_ms for event in result.batch.usage],
                *[event.ts_ms for event in result.batch.tools.values()],
            ]
            result.session = SessionRecord(
                agent=self.agent,
                session_id=session_id,
                source_id=self.source_id,
                parent_session_id=opt_str(state.get("parent_session_id")),
                project=opt_str(state.get("project")),
                directory=opt_str(state.get("directory")),
                agent_mode=opt_str(state.get("agent_mode")),
                model=current_model,
                provider=opt_str(state.get("provider")),
                time_created_ms=token_int(state.get("created_ms")),
                time_updated_ms=max(event_ts, default=None),
            )
        return result

    def _apply_session_meta(self, state: dict, meta: dict) -> None:
        own_id = opt_str(meta.get("id")) or opt_str(meta.get("session_id"))
        if own_id:
            state["session_id"] = own_id
        parent = opt_str(meta.get("parent_thread_id")) or opt_str(
            meta.get("forked_from_id")
        )
        meta_session = opt_str(meta.get("session_id"))
        if parent is None and meta_session is not None and meta_session != own_id:
            parent = meta_session
        state["parent_session_id"] = parent
        state["project"] = opt_str(meta.get("cwd"))
        state["directory"] = opt_str(meta.get("cwd"))
        state["provider"] = opt_str(meta.get("model_provider"))
        created = parse_iso8601_ms(meta.get("timestamp"))
        if created:
            state["created_ms"] = created

        thread_source = opt_str(meta.get("thread_source"))
        nickname = opt_str(meta.get("agent_nickname"))
        if thread_source in (None, "user"):
            state["agent_mode"] = None
        elif thread_source == "subagent" and nickname:
            state["agent_mode"] = f"subagent:{nickname}"
        else:
            state["agent_mode"] = thread_source

    # ------------------------------------------------------------- generators

    def _usage_event(
        self, state: dict, payload: dict, *, ts_ms: int | None, model: str | None
    ) -> UsageEvent | None:
        usage = as_dict(payload.get("usage"))
        if not usage or ts_ms is None:
            return None
        response_id = opt_str(payload.get("response_id"))
        thread_id = opt_str(payload.get("thread_id")) or opt_str(
            state.get("session_id")
        )
        payload_session = opt_str(payload.get("session_id"))
        parent = opt_str(state.get("parent_session_id"))
        if payload_session is not None and payload_session != thread_id:
            parent = payload_session

        fallback = None
        if response_id is None:
            fallback = {
                "thread": thread_id,
                "turn": opt_str(payload.get("turn_id")),
                "ts": ts_ms,
            }
        counts = TokenCounts(
            input=token_int(usage.get("input_tokens")),
            output=token_int(usage.get("output_tokens")),
            reasoning=token_int(usage.get("reasoning_output_tokens")),
            cache_read=token_int(usage.get("cached_input_tokens")),
            cache_write=token_int(usage.get("cache_write_input_tokens")),
        )
        event = UsageEvent(
            id=stable_event_id(
                kind="usage",
                agent=str(self.agent),
                source_event_id=response_id,
                fallback=fallback,
            ),
            agent=self.agent,
            source_id=self.source_id,
            source_event_id=response_id,
            session_id=thread_id,
            parent_session_id=parent,
            project=opt_str(state.get("project")),
            directory=opt_str(state.get("directory")),
            agent_mode=opt_str(state.get("agent_mode")),
            provider=opt_str(state.get("provider")),
            model=model,
            ts_ms=ts_ms,
            tokens=counts,
            token_scheme=TokenScheme.OPENAI,
            provider_total_tokens=token_int(usage.get("total_tokens")),
            source_event_type="token_usage_record",
        )
        return self._finalize_usage(event)

    def _legacy_usage_event(
        self,
        state: dict,
        payload: dict,
        *,
        stem: str,
        ts_ms: int | None,
        model: str | None,
    ) -> tuple[UsageEvent | None, bool]:
        """Translate a legacy ``token_count`` event; returns (event, missing)."""
        info = as_dict(payload.get("info"))
        total = as_dict(info.get("total_token_usage"))
        last = as_dict(info.get("last_token_usage"))
        previous = as_dict(state.get("cumulative"))
        if total:
            state["cumulative"] = {key: token_int(total.get(key)) for key in total}
        quality = Quality.OK
        notes: tuple[str, ...] = ()
        if not last:
            if not total:
                return None, True
            last = {
                key: max(0, token_int(total.get(key)) or 0)
                - max(0, token_int(previous.get(key)) or 0)
                for key in total
            }
            quality = Quality.DERIVED
            notes = ("delta_of_cumulative_totals",)
        ordinal = payload.get("ordinal")
        sequence = ordinal if isinstance(ordinal, int) else ts_ms
        if sequence is None or ts_ms is None:
            return None, True
        source_event_id = f"{_LEGACY_PREFIX}{stem}:{sequence}"
        counts = TokenCounts(
            input=token_int(last.get("input_tokens")),
            output=token_int(last.get("output_tokens")),
            reasoning=token_int(last.get("reasoning_output_tokens")),
            cache_read=token_int(last.get("cached_input_tokens")),
            cache_write=token_int(last.get("cache_write_input_tokens")),
        )
        event = UsageEvent(
            id=stable_event_id(
                kind="usage", agent=str(self.agent), source_event_id=source_event_id
            ),
            agent=self.agent,
            source_id=self.source_id,
            source_event_id=source_event_id,
            session_id=opt_str(state.get("session_id")),
            parent_session_id=opt_str(state.get("parent_session_id")),
            project=opt_str(state.get("project")),
            directory=opt_str(state.get("directory")),
            agent_mode=opt_str(state.get("agent_mode")),
            provider=opt_str(state.get("provider")),
            model=model,
            ts_ms=ts_ms,
            tokens=counts,
            token_scheme=TokenScheme.OPENAI,
            provider_total_tokens=token_int(last.get("total_tokens")),
            quality=quality,
            quality_notes=notes,
            source_event_type="event_msg.token_count",
        )
        return self._finalize_usage(event), False

    def _finalize_usage(self, event: UsageEvent) -> UsageEvent:
        derived = event.consistency_notes()
        if derived:
            quality = (
                event.quality if event.quality is not Quality.OK else Quality.PARTIAL
            )
            return dataclasses.replace(event, quality=quality, quality_notes=derived)
        return event

    def _tool_event(
        self,
        state: dict,
        *,
        call_id: str,
        name: str,
        start_ms: int,
        end_ms: int | None,
        status: str | None,
        output: str | None,
        model: str | None,
    ) -> ToolEvent:
        note: str | None = None
        quality = Quality.OK
        if output is not None:
            outcome, provenance, confidence = _classify_output(output)
            if outcome is Outcome.UNKNOWN:
                note = "no outcome signal in tool output"
        elif status is None:
            # The call record exists but no result has been written yet.
            outcome, provenance, confidence = (
                Outcome.PENDING,
                OutcomeProvenance.ABSENT,
                None,
            )
        elif status in _TOOL_STATUS_OUTCOMES:
            outcome = _TOOL_STATUS_OUTCOMES[status]
            provenance = OutcomeProvenance.STATUS_FIELD
            confidence = 0.8
        else:
            outcome, provenance, confidence = (
                Outcome.UNKNOWN,
                OutcomeProvenance.ABSENT,
                None,
            )
            note = f"unrecognized tool status: {status}"
            quality = Quality.AMBIGUOUS

        duration = None
        if end_ms is not None and end_ms >= start_ms:
            duration = end_ms - start_ms
        return ToolEvent(
            id=stable_event_id(
                kind="tool", agent=str(self.agent), source_event_id=call_id
            ),
            agent=self.agent,
            source_id=self.source_id,
            source_event_id=call_id,
            session_id=opt_str(state.get("session_id")),
            call_id=call_id,
            tool_name=name,
            tool_category=categorize_tool(name),
            ts_ms=start_ms,
            duration_ms=duration,
            outcome=outcome,
            outcome_provenance=provenance,
            outcome_confidence=confidence,
            model=model,
            agent_mode=opt_str(state.get("agent_mode")),
            quality=quality,
            quality_notes=(note,) if note else (),
            source_event_type="response_item.tool_call",
        )

    # ----------------------------------------------------------------- helpers

    def _merge_session(
        self, sessions: dict[str, SessionRecord], record: SessionRecord
    ) -> None:
        """Merge a per-file session record; resumed threads span files."""
        existing = sessions.get(record.session_id)
        if existing is None:
            sessions[record.session_id] = record
            return
        created = min(
            (ts for ts in (existing.time_created_ms, record.time_created_ms) if ts),
            default=None,
        )
        updated = max(
            (ts for ts in (existing.time_updated_ms, record.time_updated_ms) if ts),
            default=None,
        )
        sessions[record.session_id] = dataclasses.replace(
            record,
            parent_session_id=record.parent_session_id or existing.parent_session_id,
            project=record.project or existing.project,
            directory=record.directory or existing.directory,
            agent_mode=record.agent_mode or existing.agent_mode,
            model=record.model or existing.model,
            provider=record.provider or existing.provider,
            time_created_ms=created,
            time_updated_ms=updated,
        )

    def _quality_issue(
        self,
        conn: sqlite3.Connection,
        *,
        code: str,
        message: str,
        sample_ref: str,
        occurrences: int = 1,
    ) -> None:
        record_quality_issue(
            conn,
            agent=self.agent,
            source_id=self.source_id,
            code=code,
            message=message,
            occurrences=occurrences,
            sample_ref=sample_ref,
        )

    def _state_stream(self, relpath: str) -> str:
        return f"state:{relpath}"

    def _load_state(self, conn: sqlite3.Connection, relpath: str) -> dict:
        checkpoint = load_checkpoint(conn, self.source_id, self._state_stream(relpath))
        if checkpoint is None or not checkpoint.fingerprint:
            return {}
        try:
            data = json.loads(checkpoint.fingerprint)
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def _save_state(self, conn: sqlite3.Connection, relpath: str, state: dict) -> None:
        save_checkpoint(
            conn,
            self.source_id,
            self._state_stream(relpath),
            offset=0,
            fingerprint=json.dumps(state, separators=(",", ":"), sort_keys=True),
        )


@dataclass(slots=True)
class _Processed:
    batch: _Batch = field(default_factory=_Batch)
    session: SessionRecord | None = None
    missing_token_info: int = 0


@dataclass(slots=True)
class _FileOutcome:
    rows_seen: int = 0
    skipped: int = 0
    errors: int = 0
    usage: list[UsageEvent] = field(default_factory=list)
    tools: list[ToolEvent] = field(default_factory=list)
    session: SessionRecord | None = None
    notes: tuple[str, ...] = ()
