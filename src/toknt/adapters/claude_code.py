"""Claude Code adapter.

Claude Code stores append-only JSONL transcripts under
``~/.claude/projects/<project-slug>/``:

* ``<session-uuid>.jsonl`` — one file per session. ``assistant`` lines carry
  ``message.usage`` (Anthropic categories: ``input_tokens``,
  ``cache_creation_input_tokens``, ``cache_read_input_tokens``,
  ``output_tokens``), tool-use blocks and the model; ``user`` lines carry
  ``tool_result`` blocks with an explicit ``is_error`` flag.
* ``<session-uuid>/subagents/*.jsonl`` — subagent transcripts; the canonical
  session keeps its own session id and points at the session directory as its
  parent.

Streaming writes the same response several times with growing usage, so
assistant lines are deduplicated by ``requestId`` (falling back to
``message.id``) keeping the snapshot with the largest totals. Reasoning
tokens are reported as a structural zero: Claude Code bills extended
thinking inside ``output_tokens``, so a separate reasoning figure would
double count in the additive anthropic scheme.

Only accounting-relevant fields are read: message text, tool arguments and
tool outputs are never stored. API-error lines carry no usage and are
skipped; assistant lines that legitimately should have usage but do not are
flagged as ``missing_usage`` quality issues.
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
    opt_str,
    parse_iso8601_ms,
    token_int,
)
from toknt.adapters.jsonl import JsonlBatch, read_jsonl_batch
from toknt.discovery import DiscoveredSource, discover_claude_code
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
    load_checkpoint,
    reconcile_source_events,
    record_quality_issue,
    save_checkpoint,
    upsert_sessions,
    upsert_tool_events,
    upsert_usage_events,
)

_PROVIDER = "anthropic"

#: Directory that marks subagent transcripts inside a session directory.
_SUBAGENT_DIR = "subagents"

#: Fields that may carry a human-readable subagent name, in priority order.
_AGENT_HINT_FIELDS = ("agentNickname", "agentType", "agentId", "slug")


@dataclass(slots=True)
class _Batch:
    """Canonical records extracted from one batch of one transcript file."""

    usage: dict[str, UsageEvent] = field(default_factory=dict)  # keyed by request id
    tools: dict[str, ToolEvent] = field(default_factory=dict)  # keyed by tool-use id
    orphan_results: int = 0
    skipped: int = 0
    malformed: int = 0


class ClaudeCodeAdapter(Adapter):
    source_id: ClassVar[str] = "claude_code:projects"
    agent: ClassVar[Agent] = Agent.CLAUDE_CODE
    kind: ClassVar[str] = "jsonl_dir"

    def discover(
        self,
        *,
        env: dict[str, str] | None = None,
        home: object | None = None,
        override: str | None = None,
    ) -> DiscoveredSource:
        return discover_claude_code(env=env, home=home, override=override)

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
                notes=(f"projects directory not found: {root}",),
            )
        if full:
            # A full rescan re-observes everything: stale issues would
            # otherwise survive fixed sources forever.
            clear_quality_issues(conn, source_id=self.source_id)

        paths = self._transcript_paths(root)
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
            summary = "no transcript files found"
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
        """Scan one transcript file. Returns ``None`` when unchanged."""
        relpath = path.relative_to(root).as_posix()
        try:
            stat = path.stat()
        except OSError as exc:
            self._quality_issue(
                conn,
                code="unreadable_file",
                message=f"cannot stat transcript: {type(exc).__name__}",
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
        parent, is_subagent = self._layout(relpath)
        state["parent_session_id"] = parent
        if is_subagent:
            state["subagent"] = True

        batch = read_jsonl_batch(
            path,
            offset=checkpoint.offset if checkpoint else 0,
            fingerprint=checkpoint.fingerprint if checkpoint else None,
        )
        if batch.reset and state:
            state = {"parent_session_id": parent, "subagent": is_subagent}

        result = self._process_batch(state, batch, stem=path.stem)

        if batch.malformed:
            self._quality_issue(
                conn,
                code="unparsable_row",
                message="transcript contained unparsable JSON lines",
                occurrences=len(batch.malformed),
                sample_ref=relpath,
            )
        if result.missing_usage:
            self._quality_issue(
                conn,
                code="missing_usage",
                message="assistant line without usage details",
                occurrences=result.missing_usage,
                sample_ref=relpath,
            )
        if result.batch.orphan_results:
            self._quality_issue(
                conn,
                code="orphan_tool_result",
                message="tool result without its matching tool-use record",
                occurrences=result.batch.orphan_results,
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
            usage=list(result.batch.usage.values()),
            tools=list(result.batch.tools.values()),
            session=result.session,
        )

    def _layout(self, relpath: str) -> tuple[str | None, bool]:
        """Derive (parent session id, is subagent) from the project layout."""
        parts = Path(relpath).parts
        if _SUBAGENT_DIR in parts:
            index = parts.index(_SUBAGENT_DIR)
            if index >= 1:
                return parts[index - 1], True
        return None, False

    # ------------------------------------------------- format extension points

    def _transcript_paths(self, root: Path) -> list[Path]:
        """Every transcript file under the projects root.

        Claude-Code-shaped forks that keep sidecar files beside transcripts
        (Command Code's snapshots and backups) override this to filter them.
        """
        return sorted(root.rglob("*.jsonl"))

    def _record_type(self, record: dict) -> object:
        """The record discriminator; forks may wrap messages in an envelope."""
        return record.get("type")

    def _usage_key(self, record: dict, message: dict) -> str | None:
        """Streaming-dedupe identity for one assistant response."""
        return (
            opt_str(record.get("requestId"))
            or opt_str(message.get("id"))
            or opt_str(record.get("uuid"))
        )

    # ----------------------------------------------------------- state machine

    def _process_batch(
        self, state: dict, batch: JsonlBatch, *, stem: str
    ) -> _Processed:
        """Translate one batch of transcript lines into canonical records.

        ``state`` is updated in place and persisted by the caller.
        """
        result = _Processed()
        pending = {
            str(key): value for key, value in as_dict(state.get("pending")).items()
        }
        last_ts = token_int(state.get("last_ts_ms"))
        model = opt_str(state.get("model"))
        session_id = opt_str(state.get("session_id"))
        cwd = opt_str(state.get("project"))
        hint = opt_str(state.get("agent_hint"))

        for record in batch.records:
            ts_ms = parse_iso8601_ms(record.get("timestamp")) or last_ts
            session_ref = opt_str(record.get("sessionId"))
            if session_ref:
                session_id = session_ref
                state["session_id"] = session_ref
            cwd_ref = opt_str(record.get("cwd"))
            if cwd_ref:
                cwd = cwd_ref
                state["project"] = cwd
                state["directory"] = cwd
            for field_name in _AGENT_HINT_FIELDS:
                value = opt_str(record.get(field_name))
                if value:
                    hint = value
                    state["agent_hint"] = value
                    break
            if ts_ms and state.get("created_ms") is None:
                state["created_ms"] = ts_ms

            record_type = self._record_type(record)
            sidechain = bool(record.get("isSidechain"))
            mode = self._agent_mode(state, sidechain=sidechain, hint=hint)

            if record_type == "assistant":
                message = as_dict(record.get("message"))
                model = opt_str(message.get("model")) or model
                if record.get("isApiErrorMessage"):
                    # Failed API requests carry no usage.
                    if ts_ms:
                        last_ts = ts_ms
                    continue
                content = message.get("content")
                if isinstance(content, list):
                    for block in content:
                        block = as_dict(block)
                        if block.get("type") != "tool_use":
                            continue
                        call_id = opt_str(block.get("id"))
                        if call_id is None or ts_ms is None:
                            result.batch.skipped += 1
                            continue
                        name = opt_str(block.get("name")) or "unknown"
                        pending[call_id] = {
                            "name": name,
                            "ts_ms": ts_ms,
                            "model": model,
                            "mode": mode,
                        }
                        result.batch.tools[call_id] = self._tool_event(
                            state,
                            call_id=call_id,
                            name=name,
                            start_ms=ts_ms,
                            end_ms=None,
                            is_error=None,
                            model=model,
                            mode=mode,
                        )
                usage = message.get("usage")
                if not isinstance(usage, dict) or not usage:
                    result.missing_usage += 1
                    result.batch.skipped += 1
                    if ts_ms:
                        last_ts = ts_ms
                    continue
                key = self._usage_key(record, message)
                if key is None or ts_ms is None:
                    result.batch.skipped += 1
                    continue
                event = self._usage_event(
                    state,
                    usage,
                    message,
                    record=record,
                    key=key,
                    ts_ms=ts_ms,
                    model=model,
                    mode=mode,
                )
                existing = result.batch.usage.get(key)
                if existing is None or _snapshot_total(
                    event.tokens
                ) >= _snapshot_total(existing.tokens):
                    result.batch.usage[key] = event
                last_ts = ts_ms
                continue

            if record_type == "user":
                message = as_dict(record.get("message"))
                content = message.get("content")
                if not isinstance(content, list):
                    if ts_ms:
                        last_ts = ts_ms
                    continue
                for block in content:
                    block = as_dict(block)
                    if block.get("type") != "tool_result":
                        continue
                    call_id = opt_str(block.get("tool_use_id"))
                    if call_id is None:
                        result.batch.skipped += 1
                        continue
                    call = as_dict(pending.pop(call_id, None))
                    if not call:
                        result.batch.orphan_results += 1
                        result.batch.skipped += 1
                        continue
                    is_error = block.get("is_error")
                    if not isinstance(is_error, bool):
                        is_error = None
                    start_ms = token_int(call.get("ts_ms")) or ts_ms
                    if start_ms is None:
                        # Pending records always carry a timestamp; a missing
                        # one means unusable state — never fake a time.
                        result.batch.skipped += 1
                        continue
                    result.batch.tools[call_id] = self._tool_event(
                        state,
                        call_id=call_id,
                        name=opt_str(call.get("name")) or "unknown",
                        start_ms=start_ms,
                        end_ms=ts_ms,
                        is_error=is_error,
                        model=opt_str(call.get("model")),
                        mode=call.get("mode", mode),
                    )
            if ts_ms:
                last_ts = ts_ms

        if batch.malformed:
            result.batch.malformed = len(batch.malformed)

        state["session_id"] = session_id or stem
        state["model"] = model
        state["project"] = cwd
        state["directory"] = cwd
        state["pending"] = pending
        state["last_ts_ms"] = last_ts
        if hint:
            state["agent_hint"] = hint

        # Session cache record: touched by every event this batch produced.
        event_ts = [
            *[event.ts_ms for event in result.batch.usage.values()],
            *[event.ts_ms for event in result.batch.tools.values()],
        ]
        result.session = SessionRecord(
            agent=self.agent,
            session_id=state["session_id"],
            source_id=self.source_id,
            parent_session_id=opt_str(state.get("parent_session_id")),
            project=cwd,
            directory=cwd,
            agent_mode=self._base_mode(state, hint=hint),
            model=model,
            provider=_PROVIDER,
            time_created_ms=token_int(state.get("created_ms")),
            time_updated_ms=max(event_ts, default=None),
        )
        return result

    def _agent_mode(self, state: dict, *, sidechain: bool, hint: str | None) -> str | None:
        base = self._base_mode(state, hint=hint)
        if base is not None:
            return base
        return "sidechain" if sidechain else None

    def _base_mode(self, state: dict, *, hint: str | None) -> str | None:
        if not state.get("subagent"):
            return None
        return f"subagent:{hint}" if hint else "subagent"

    # ------------------------------------------------------------- generators

    def _usage_event(
        self,
        state: dict,
        usage: dict,
        message: dict,
        *,
        record: dict,
        key: str,
        ts_ms: int,
        model: str | None,
        mode: str | None,
    ) -> UsageEvent:
        counts = TokenCounts(
            input=token_int(usage.get("input_tokens")),
            output=token_int(usage.get("output_tokens")),
            # Structural zero: extended thinking is inside output_tokens in
            # the anthropic scheme, so reporting it separately would double
            # count. Never reported as unknown.
            reasoning=0,
            cache_read=token_int(usage.get("cache_read_input_tokens")),
            cache_write=token_int(usage.get("cache_creation_input_tokens")),
        )
        cost = _reported_cost(record.get("costUSD"))
        event = UsageEvent(
            id=stable_event_id(kind="usage", agent=str(self.agent), source_event_id=key),
            agent=self.agent,
            source_id=self.source_id,
            source_event_id=key,
            session_id=opt_str(state.get("session_id")),
            parent_session_id=opt_str(state.get("parent_session_id")),
            project=opt_str(state.get("project")),
            directory=opt_str(state.get("directory")),
            agent_mode=mode,
            provider=_PROVIDER,
            model=model,
            ts_ms=ts_ms,
            tokens=counts,
            token_scheme=TokenScheme.ANTHROPIC,
            provider_total_tokens=token_int(usage.get("total_tokens")),
            cost_reported=cost,
            cost_currency="USD" if cost is not None else None,
            finish_reason=opt_str(message.get("stop_reason")),
            source_event_type="assistant",
        )
        return self._finalize_usage(event)

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
        is_error: bool | None,
        model: str | None,
        mode: str | None,
    ) -> ToolEvent:
        note: str | None = None
        quality = Quality.OK
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
            # The call record exists but no result has been written yet.
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
            note = "tool result without an is_error flag"

        duration = None
        if end_ms is not None and end_ms >= start_ms:
            duration = end_ms - start_ms
        return ToolEvent(
            id=stable_event_id(kind="tool", agent=str(self.agent), source_event_id=call_id),
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
            agent_mode=mode,
            quality=quality,
            quality_notes=(note,) if note else (),
            source_event_type="assistant.tool_use",
        )

    # ----------------------------------------------------------------- helpers

    def _merge_session(
        self, sessions: dict[str, SessionRecord], record: SessionRecord
    ) -> None:
        """Merge a per-file session record; a session may span files."""
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


def _reported_cost(value: object) -> float | None:
    """A non-negative cost explicitly reported by the transcript."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number >= 0 else None


def _snapshot_total(counts: TokenCounts) -> int:
    """Sum of whatever components a usage snapshot reports.

    Only used to compare streaming snapshots of one response against each
    other — never as an accounting total.
    """
    return sum(
        value
        for value in (
            counts.input,
            counts.output,
            counts.reasoning,
            counts.cache_read,
            counts.cache_write,
        )
        if value is not None
    )


@dataclass(slots=True)
class _Processed:
    batch: _Batch = field(default_factory=_Batch)
    session: SessionRecord | None = None
    missing_usage: int = 0


@dataclass(slots=True)
class _FileOutcome:
    rows_seen: int = 0
    skipped: int = 0
    errors: int = 0
    usage: list[UsageEvent] = field(default_factory=list)
    tools: list[ToolEvent] = field(default_factory=list)
    session: SessionRecord | None = None
    notes: tuple[str, ...] = ()
