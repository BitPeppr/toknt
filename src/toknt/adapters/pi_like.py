"""pi-family session-tree adapter (pi, senpi, omp, gjc, prime, kimchi).

These harnesses write append-only JSONL session trees under
``<root>/agent/sessions/--<encoded-cwd>--/<timestamp>_<session-id>.jsonl``.
Each line is one entry of the session's entry tree, discriminated by ``type``:

* ``session`` — header: ``id`` (session id), ``cwd``, ``timestamp``;
* ``model_change`` / ``thinking_level_change`` — current model/provider;
* ``message`` — ``message.role`` in ``user``/``assistant``/``toolResult``.
  Assistant messages carry ``message.usage`` in models.dev shape
  (``input``, ``output``, ``cacheRead``, ``cacheWrite``, ``totalTokens``,
  ``cost``), content blocks of type ``text``/``thinking``/``toolCall``, and
  ``toolResult`` messages answer a pending call with ``isError``;
* ``custom`` — harness bookkeeping (plan-mode state and friends), ignored.

Token semantics (verified against a real 62-message corpus on this machine):
``totalTokens == input + output + cacheRead + cacheWrite``, i.e. cache reads
and writes are reported *separately* from ``input`` — the additive anthropic
scheme. Reasoning is billed inside ``output`` (the 5-token ``reasoning``
field is a subset of output in every observed sample), so it is recorded as
a structural zero exactly like Claude Code. ``cost.total`` is the provider's
own reported USD figure and is stored as ``cost_reported``.

Failed API calls (``stopReason == "error"``) carry all-zero usage and are
skipped, matching the Claude Code adapter's handling of API errors.
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
from toknt.adapters.jsonl import JsonlBatch, read_jsonl_batch
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


@dataclass(slots=True)
class _Processed:
    usage: list[UsageEvent] = field(default_factory=list)
    tools: list[ToolEvent] = field(default_factory=list)
    session: SessionRecord | None = None
    api_errors: int = 0
    skipped: int = 0


class PiFamilyAdapter(Adapter):
    """Base adapter for pi-shaped stores; subclasses pin harness and root."""

    source_id: ClassVar[str]
    agent: ClassVar[Harness]
    kind: ClassVar[str] = "jsonl_dir"

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
                sessions[outcome.session.session_id] = outcome.session
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
            summary = "no session files found"
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
        self, conn: sqlite3.Connection, root: Path, path: Path, *, full: bool
    ) -> _FileOutcome | None:
        relpath = path.relative_to(root).as_posix()
        try:
            stat = path.stat()
        except OSError as exc:
            self._quality_issue(
                conn,
                code="unreadable_file",
                message=f"cannot stat session: {type(exc).__name__}",
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

        result = self._process_batch(state, batch, stem=path.stem)
        if batch.malformed:
            self._quality_issue(
                conn,
                code="unparsable_row",
                message="session contained unparsable JSON lines",
                occurrences=len(batch.malformed),
                sample_ref=relpath,
            )
        if result.api_errors:
            self._quality_issue(
                conn,
                code="api_error_messages",
                message="assistant entries with a provider error (zero usage)",
                occurrences=result.api_errors,
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
            skipped=result.skipped + len(batch.malformed),
            errors=len(batch.malformed),
            usage=result.usage,
            tools=result.tools,
            session=result.session,
        )

    # ----------------------------------------------------------- state machine

    def _process_batch(self, state: dict, batch: JsonlBatch, *, stem: str) -> _Processed:
        result = _Processed()
        session_id = opt_str(state.get("session_id")) or stem.split("_")[-1]
        cwd = opt_str(state.get("cwd"))
        model = opt_str(state.get("model"))
        provider = opt_str(state.get("provider"))
        pending = {
            str(key): value for key, value in as_dict(state.get("pending")).items()
        }
        last_ts = token_int(state.get("last_ts_ms"))
        created_ms = token_int(state.get("created_ms"))
        event_ts: list[int] = []

        for record in batch.records:
            entry_type = record.get("type")
            entry_id = opt_str(record.get("id"))
            ts_ms = (
                normalize_epoch_ms(as_dict(record.get("message")).get("timestamp"))
                or parse_iso8601_ms(record.get("timestamp"))
                or last_ts
            )

            if entry_type == "session":
                session_id = opt_str(record.get("id")) or session_id
                cwd = opt_str(record.get("cwd")) or cwd
                state["session_id"] = session_id
                state["cwd"] = cwd
                if ts_ms:
                    created_ms = created_ms or ts_ms
                    state["created_ms"] = created_ms
                continue

            if entry_type == "model_change":
                model = opt_str(record.get("modelId")) or model
                provider = opt_str(record.get("provider")) or provider
                state["model"] = model
                state["provider"] = provider
                if ts_ms:
                    last_ts = ts_ms
                continue

            if entry_type != "message":
                if ts_ms:
                    last_ts = ts_ms
                continue

            message = as_dict(record.get("message"))
            role = opt_str(message.get("role"))
            if role == "assistant":
                model = opt_str(message.get("model")) or model
                provider = opt_str(message.get("provider")) or provider
                state["model"] = model
                state["provider"] = provider
                if opt_str(message.get("errorMessage")) or message.get(
                    "stopReason"
                ) == "error":
                    # failed API call: no usage was actually served
                    result.api_errors += 1
                    result.skipped += 1
                    continue
                content = message.get("content")
                if isinstance(content, list):
                    for block in content:
                        block = as_dict(block)
                        if block.get("type") != "toolCall":
                            continue
                        call_id = opt_str(block.get("id"))
                        if call_id is None or ts_ms is None:
                            result.skipped += 1
                            continue
                        name = opt_str(block.get("name")) or "unknown"
                        pending[call_id] = {
                            "name": name,
                            "ts_ms": ts_ms,
                            "model": model,
                        }
                        result.tools.append(
                            self._tool_event(
                                call_id=call_id,
                                name=name,
                                start_ms=ts_ms,
                                end_ms=None,
                                is_error=None,
                                model=model,
                            )
                        )
                usage = message.get("usage")
                if isinstance(usage, dict) and usage:
                    event = self._usage_event(
                        state,
                        usage,
                        message,
                        entry_id=entry_id or call_ref(record),
                        ts_ms=ts_ms,
                        model=model,
                        provider=provider,
                    )
                    if event is not None:
                        result.usage.append(event)
                        event_ts.append(event.ts_ms)
                elif usage is not None:
                    result.skipped += 1
                if ts_ms:
                    last_ts = ts_ms
                continue

            if role == "toolResult":
                call_id = opt_str(message.get("toolCallId"))
                call = as_dict(pending.pop(call_id, None)) if call_id else {}
                if not call or ts_ms is None:
                    result.skipped += 1
                else:
                    start_ms = token_int(call.get("ts_ms")) or ts_ms
                    result.tools.append(
                        self._tool_event(
                            call_id=str(call_id),
                            name=opt_str(call.get("name"))
                            or opt_str(message.get("toolName"))
                            or "unknown",
                            start_ms=start_ms,
                            end_ms=ts_ms,
                            is_error=message.get("isError"),
                            model=opt_str(call.get("model")),
                        )
                    )
                    event_ts.append(start_ms)
                if ts_ms:
                    last_ts = ts_ms
                continue

            if ts_ms:
                last_ts = ts_ms

        state["session_id"] = session_id
        state["cwd"] = cwd
        state["model"] = model
        state["provider"] = provider
        state["pending"] = pending
        state["last_ts_ms"] = last_ts
        state["created_ms"] = created_ms
        result.session = SessionRecord(
            agent=self.agent,
            session_id=session_id,
            source_id=self.source_id,
            project=cwd,
            directory=cwd,
            model=model,
            provider=provider,
            time_created_ms=created_ms,
            time_updated_ms=max(event_ts, default=None),
        )
        return result

    # ------------------------------------------------------------- generators

    def _usage_event(
        self,
        state: dict,
        usage: dict,
        message: dict,
        *,
        entry_id: str,
        ts_ms: int | None,
        model: str | None,
        provider: str | None,
    ) -> UsageEvent | None:
        if ts_ms is None:
            return None
        counts = TokenCounts(
            input=token_int(usage.get("input")),
            output=token_int(usage.get("output")),
            reasoning=0,  # billed inside output in this scheme
            cache_read=token_int(usage.get("cacheRead")),
            cache_write=token_int(usage.get("cacheWrite")),
        )
        cost = as_dict(usage.get("cost")).get("total")
        cost_reported = None
        if not isinstance(cost, bool) and isinstance(cost, (int, float)) and cost >= 0:
            cost_reported = float(cost)
        event = UsageEvent(
            id=stable_event_id(
                kind="usage", agent=str(self.agent), source_event_id=entry_id
            ),
            agent=self.agent,
            source_id=self.source_id,
            source_event_id=entry_id,
            session_id=opt_str(state.get("session_id")),
            project=opt_str(state.get("cwd")),
            directory=opt_str(state.get("cwd")),
            provider=provider,
            model=model,
            ts_ms=ts_ms,
            tokens=counts,
            token_scheme=TokenScheme.ANTHROPIC,
            provider_total_tokens=token_int(usage.get("totalTokens")),
            cost_reported=cost_reported,
            cost_currency="USD" if cost_reported is not None else None,
            finish_reason=opt_str(message.get("stopReason")),
            source_event_type="message",
        )
        return _finalize_usage(event)

    def _tool_event(
        self,
        *,
        call_id: str,
        name: str,
        start_ms: int,
        end_ms: int | None,
        is_error: object,
        model: str | None,
    ) -> ToolEvent:
        if is_error is True:
            outcome, provenance, confidence = (
                Outcome.FAILURE,
                OutcomeProvenance.RESULT_FIELD,
                0.9,
            )
        elif is_error is False:
            outcome, provenance, confidence = (
                Outcome.SUCCESS,
                OutcomeProvenance.RESULT_FIELD,
                0.9,
            )
        elif end_ms is None:
            outcome, provenance, confidence = (
                Outcome.PENDING,
                OutcomeProvenance.ABSENT,
                None,
            )
        else:
            outcome, provenance, confidence = (
                Outcome.UNKNOWN,
                OutcomeProvenance.ABSENT,
                None,
            )
        duration = end_ms - start_ms if end_ms is not None and end_ms >= start_ms else None
        return ToolEvent(
            id=stable_event_id(kind="tool", agent=str(self.agent), source_event_id=call_id),
            agent=self.agent,
            source_id=self.source_id,
            source_event_id=call_id,
            call_id=call_id,
            tool_name=name,
            tool_category=categorize_tool(name),
            ts_ms=start_ms,
            duration_ms=duration,
            outcome=outcome,
            outcome_provenance=provenance,
            outcome_confidence=confidence,
            model=model,
            source_event_type="message.toolCall",
        )

    # ----------------------------------------------------------------- helpers

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
            fingerprint=json.dumps(state, separators=(",", ":"), sort_keys=True, default=str),
        )


def call_ref(record: dict) -> str:
    """Stable identity for a message entry that carries no explicit id."""
    message = as_dict(record.get("message"))
    return f"{opt_str(record.get('timestamp')) or ''}:{opt_str(message.get('model')) or ''}"


def _finalize_usage(event: UsageEvent) -> UsageEvent:
    derived = event.consistency_notes()
    if derived:
        quality = event.quality if event.quality is not Quality.OK else Quality.PARTIAL
        return dataclasses.replace(event, quality=quality, quality_notes=derived)
    return event


@dataclass(slots=True)
class _FileOutcome:
    rows_seen: int = 0
    skipped: int = 0
    errors: int = 0
    usage: list[UsageEvent] = field(default_factory=list)
    tools: list[ToolEvent] = field(default_factory=list)
    session: SessionRecord | None = None
    notes: tuple[str, ...] = ()
