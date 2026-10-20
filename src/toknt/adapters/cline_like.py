"""Cline-family adapters (Cline, Roo Code, Kilo Code).

Two store generations exist, sometimes on the same machine:

* **Modern CLI/SDK** (Cline ≥ 3.x): ``~/.cline/data/sessions/<sessionId>/``
  holds ``<sessionId>.json`` — a manifest (session id, started/ended, cwd,
  model, provider, ``metadata.usage``/``totalCost``) — and
  ``<sessionId>.messages.json``, a dict whose ``messages`` list carries one
  entry per turn: ``id``/``role``/``ts`` plus ``content`` blocks. Assistant
  entries carry per-response ``metrics`` (``inputTokens``, ``outputTokens``,
  ``cacheReadTokens``, ``cacheWriteTokens``) and ``modelInfo``; tool calls
  are ``tool_use`` blocks answered by ``tool_result`` blocks whose
  ``content`` is a list of ``{query, result, error, success}`` entries.
* **Legacy VS Code extension** (Cline/Roo/Kilo): ``<globalStorage>/<ext
  id>/tasks/<taskId>/api_conversation_history.json``, an array of
  ``{role, content}`` messages with Claude-style blocks. Token counts are
  not part of this format, so those stores contribute sessions and tool
  events but no usage events.

Both stores are rewritten whole (not append-only), so scans re-parse a file
whenever its size or mtime changes; stable ids keep the re-upserts
idempotent. Metrics follow models.dev semantics — cache reads and writes
are reported separately from input — so usage uses the anthropic scheme
with reasoning as a structural zero (Cline bills thinking inside output).
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
from toknt.discovery import DiscoveredSource, discover_cline, discover_vscode_tasks
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

#: VS Code globalStorage extension ids, keyed by harness.
_EXTENSION_IDS: dict[Harness, str] = {
    Harness.CLINE: "saoudrizwan.claude-dev",
    Harness.ROO: "rooveterinaryinc.roo-cline",
    Harness.KILOCODE: "kilocode.kilo-code",
}


def _whole_file_changed(
    conn: sqlite3.Connection,
    source_id: str,
    relpath: str,
    path: Path,
    *,
    full: bool = False,
) -> tuple[bool, object]:
    """Checkpoint test for whole-file JSON stores (no byte offsets).

    Full scans must re-read everything regardless of checkpoints: their
    reconcile step treats the seen-id set as authoritative, so skipping
    unchanged files would make a full scan look like an empty source and
    wipe previously ingested events.
    """
    if full:
        return True, None
    checkpoint = load_checkpoint(conn, source_id, relpath)
    try:
        stat = path.stat()
    except OSError:
        return False, checkpoint
    if (
        checkpoint is not None
        and checkpoint.size == stat.st_size
        and checkpoint.mtime_ms == stat.st_mtime_ns // 1_000_000
    ):
        return False, checkpoint
    return True, checkpoint


def _save_whole_file(
    conn: sqlite3.Connection, source_id: str, relpath: str, path: Path
) -> None:
    try:
        stat = path.stat()
    except OSError:
        return
    save_checkpoint(
        conn,
        source_id,
        relpath,
        offset=0,
        size=stat.st_size,
        mtime_ms=stat.st_mtime_ns // 1_000_000,
    )


def _result_outcome(blocks: object) -> tuple[Outcome, OutcomeProvenance, float | None]:
    """Classify a modern-CLI tool_result from its structured entries.

    ``{query, result, error, success}`` entries carry an explicit success
    flag; a result with any failing entry is a failure. Results without
    entries stay unknown.
    """
    entries: list[dict] = []
    if isinstance(blocks, str):
        return Outcome.UNKNOWN, OutcomeProvenance.ABSENT, None
    if isinstance(blocks, list):
        entries = [block for block in blocks if isinstance(block, dict)]
    if not entries:
        return Outcome.UNKNOWN, OutcomeProvenance.ABSENT, None
    flags = [entry.get("success") for entry in entries]
    if any(entry.get("error") for entry in entries) or any(
        isinstance(flag, bool) and not flag for flag in flags
    ):
        return Outcome.FAILURE, OutcomeProvenance.RESULT_FIELD, 0.8
    if any(isinstance(flag, bool) and flag for flag in flags):
        return Outcome.SUCCESS, OutcomeProvenance.RESULT_FIELD, 0.9
    return Outcome.UNKNOWN, OutcomeProvenance.ABSENT, None


class ClineSessionsAdapter(Adapter):
    """Cline's modern CLI store (validated against a live 60-session corpus)."""

    source_id: ClassVar[str] = "cline:sessions"
    agent: ClassVar[Harness] = Harness.CLINE
    kind: ClassVar[str] = "json_dir"

    def discover(
        self,
        *,
        env: dict[str, str] | None = None,
        home: object | None = None,
        override: str | None = None,
    ) -> DiscoveredSource:
        return discover_cline(env=env, home=home, override=override)

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

        for directory in sorted(root.iterdir()):
            if not directory.is_dir():
                continue
            outcome = self._scan_session(conn, root, directory, full=full)
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

    def _scan_session(
        self, conn: sqlite3.Connection, root: Path, directory: Path, *, full: bool
    ) -> _SessionOutcome | None:
        relpath = directory.relative_to(root).as_posix()
        session_id = directory.name
        manifest_path = directory / f"{session_id}.json"
        messages_path = directory / f"{session_id}.messages.json"
        if not messages_path.is_file():
            return None

        changed, _checkpoint = _whole_file_changed(
            conn, self.source_id, f"messages:{relpath}", messages_path, full=full
        )
        if not changed:
            return None

        try:
            manifest = _load_json(manifest_path)
            payload = _load_json(messages_path)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._quality_issue(
                conn,
                code="unparsable_file",
                message=f"cannot read session: {type(exc).__name__}",
                sample_ref=relpath,
            )
            return _SessionOutcome(errors=1)

        usage_events: list[UsageEvent] = []
        tool_events: list[ToolEvent] = []
        pending: dict[str, dict] = {}
        last_model: str | None = None
        last_provider: str | None = None
        event_ts: list[int] = []
        rows = 0
        skipped = 0

        messages = as_dict(payload).get("messages")
        for message in messages if isinstance(messages, list) else []:
            rows += 1
            message_id = opt_str(message.get("id"))
            ts_ms = normalize_epoch_ms(message.get("ts"))
            role = opt_str(message.get("role"))
            model_info = as_dict(message.get("modelInfo"))
            model = opt_str(model_info.get("id")) or last_model
            provider = opt_str(model_info.get("provider")) or last_provider
            if model:
                last_model = model
            if provider:
                last_provider = provider

            if role == "assistant" and message_id and ts_ms:
                metrics = as_dict(message.get("metrics"))
                if metrics:
                    usage_events.append(
                        self._usage_event(
                            message, metrics, message_id=message_id, ts_ms=ts_ms,
                            model=model, provider=provider,
                        )
                    )
                    event_ts.append(ts_ms)
            for block in _blocks(message):
                block_type = block.get("type")
                call_id = opt_str(block.get("id")) or opt_str(block.get("tool_use_id"))
                if block_type == "tool_use" and call_id and ts_ms is not None:
                    name = opt_str(block.get("name")) or "unknown"
                    pending[call_id] = {"name": name, "ts_ms": ts_ms, "model": model}
                    tool_events.append(
                        _tool_event(
                            self.agent, self.source_id, call_id=call_id, name=name,
                            start_ms=ts_ms, end_ms=None, is_error=None,
                            model=model, confidence=None,
                        )
                    )
                elif block_type == "tool_result" and call_id:
                    call = pending.pop(call_id, None)
                    start_ms = token_int(as_dict(call).get("ts_ms")) or ts_ms
                    if start_ms is None:
                        skipped += 1
                        continue
                    outcome, provenance, confidence = _result_outcome(
                        block.get("content")
                    )
                    if isinstance(block.get("is_error"), bool):
                        outcome = Outcome.FAILURE if block["is_error"] else Outcome.SUCCESS
                        provenance = OutcomeProvenance.RESULT_FIELD
                        confidence = 0.9
                    tool_events.append(
                        _tool_event(
                            self.agent, self.source_id, call_id=call_id,
                            name=opt_str(as_dict(call).get("name"))
                            or opt_str(block.get("name"))
                            or "unknown",
                            start_ms=start_ms, end_ms=ts_ms, is_error=None,
                            model=opt_str(as_dict(call).get("model")),
                            confidence=confidence if outcome is not Outcome.UNKNOWN else None,
                            outcome=outcome, provenance=provenance,
                        )
                    )
                    event_ts.append(start_ms)

        session = self._session_record(manifest, session_id, event_ts, last_model)
        _save_whole_file(conn, self.source_id, f"messages:{relpath}", messages_path)
        return _SessionOutcome(
            rows_seen=rows,
            skipped=skipped,
            usage=usage_events,
            tools=tool_events,
            session=session,
        )

    def _usage_event(
        self,
        message: dict,
        metrics: dict,
        *,
        message_id: str,
        ts_ms: int,
        model: str | None,
        provider: str | None,
    ) -> UsageEvent:
        counts = TokenCounts(
            input=token_int(metrics.get("inputTokens")),
            output=token_int(metrics.get("outputTokens")),
            reasoning=0,  # billed inside outputTokens
            cache_read=token_int(metrics.get("cacheReadTokens")),
            cache_write=token_int(metrics.get("cacheWriteTokens")),
        )
        return _finalize_usage(
            UsageEvent(
                id=stable_event_id(
                    kind="usage", agent=str(self.agent), source_event_id=message_id
                ),
                agent=self.agent,
                source_id=self.source_id,
                source_event_id=message_id,
                session_id=opt_str(message.get("sessionId")),
                model=model,
                provider=provider,
                ts_ms=ts_ms,
                tokens=counts,
                token_scheme=TokenScheme.ANTHROPIC,
                source_event_type="assistant.metrics",
            )
        )

    def _session_record(
        self,
        manifest: dict,
        session_id: str,
        event_ts: list[int],
        last_model: str | None,
    ) -> SessionRecord:
        metadata = as_dict(manifest.get("metadata"))
        cost = metadata.get("totalCost")
        cost_reported = None
        if not isinstance(cost, bool) and isinstance(cost, (int, float)) and cost >= 0:
            cost_reported = float(cost)
        return SessionRecord(
            agent=self.agent,
            session_id=opt_str(manifest.get("session_id")) or session_id,
            source_id=self.source_id,
            project=opt_str(manifest.get("cwd")),
            directory=opt_str(manifest.get("workspace_root")),
            title=opt_str(metadata.get("title")),
            model=opt_str(manifest.get("model")) or last_model,
            provider=opt_str(manifest.get("provider")),
            time_created_ms=parse_iso8601_ms(manifest.get("started_at")),
            time_updated_ms=max(event_ts, default=None)
            or parse_iso8601_ms(manifest.get("ended_at")),
            cost_reported=cost_reported,
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


class VSCodeTaskAdapter(Adapter):
    """Cline-family VS Code task directories (sessions + tools, no usage)."""

    source_id: ClassVar[str]
    agent: ClassVar[Harness]
    extension_id: ClassVar[str]
    kind: ClassVar[str] = "json_dir"

    def discover(
        self,
        *,
        env: dict[str, str] | None = None,
        home: object | None = None,
        override: str | None = None,
    ) -> DiscoveredSource:
        return discover_vscode_tasks(
            self.agent, self.extension_id, env=env, home=home, override=override
        )

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
                notes=(f"tasks directory not found: {root}",),
            )
        if full:
            clear_quality_issues(conn, source_id=self.source_id)

        sessions: dict[str, SessionRecord] = {}
        tool_events: list[ToolEvent] = []
        rows_seen = skipped = errors = files_scanned = 0
        notes: list[str] = []

        for directory in sorted(root.iterdir()):
            history = directory / "api_conversation_history.json"
            if not history.is_file():
                continue
            relpath = history.relative_to(root).as_posix()
            changed, _checkpoint = _whole_file_changed(
                conn, self.source_id, relpath, history, full=full
            )
            if not changed:
                continue
            files_scanned += 1
            try:
                payload = _load_json(history)
            except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._quality_issue(
                    conn,
                    code="unparsable_file",
                    message=f"cannot read task history: {type(exc).__name__}",
                    sample_ref=relpath,
                )
                errors += 1
                continue
            task_id = directory.name
            pending: dict[str, dict] = {}
            created: int | None = None
            for message in payload if isinstance(payload, list) else []:
                role = opt_str(message.get("role"))
                ts_ms = normalize_epoch_ms(message.get("ts"))
                if created is None and ts_ms:
                    created = ts_ms
                if role != "assistant":
                    for block in _blocks(message):
                        call_id = opt_str(block.get("tool_use_id"))
                        if block.get("type") == "tool_result" and call_id:
                            call = pending.pop(call_id, None)
                            start_ms = token_int(as_dict(call).get("ts_ms")) or ts_ms
                            if start_ms is None:
                                skipped += 1
                                continue
                            is_error = block.get("is_error")
                            if not isinstance(is_error, bool):
                                is_error = None
                            tool_events.append(
                                _tool_event(
                                    self.agent, self.source_id, call_id=call_id,
                                    name=opt_str(as_dict(call).get("name"))
                                    or opt_str(block.get("name"))
                                    or "unknown",
                                    start_ms=start_ms, end_ms=ts_ms,
                                    is_error=is_error, model=None, confidence=None,
                                )
                            )
                    continue
                for block in _blocks(message):
                    if block.get("type") != "tool_use":
                        continue
                    call_id = opt_str(block.get("id"))
                    if call_id is None or ts_ms is None:
                        skipped += 1
                        continue
                    name = opt_str(block.get("name")) or "unknown"
                    pending[call_id] = {"name": name, "ts_ms": ts_ms}
                    tool_events.append(
                        _tool_event(
                            self.agent, self.source_id, call_id=call_id, name=name,
                            start_ms=ts_ms, end_ms=None, is_error=None, model=None,
                            confidence=None,
                        )
                    )
            sessions[task_id] = SessionRecord(
                agent=self.agent,
                session_id=task_id,
                source_id=self.source_id,
                time_created_ms=created,
                quality=Quality.PARTIAL,
                quality_notes=("no token counts in the VS Code task format",),
            )
            _save_whole_file(conn, self.source_id, relpath, history)

        tool_stats = upsert_tool_events(conn, tool_events)
        session_stats = upsert_sessions(conn, sessions.values())
        reconciled = 0
        if full:
            reconciled += reconcile_source_events(
                conn,
                source_id=self.source_id,
                table="tool_events",
                seen_ids=[event.id for event in tool_events],
            )
        if files_scanned == 0:
            summary = "no task directories"
        else:
            summary = f"{files_scanned} tasks read"
        return ScanReport(
            source_id=self.source_id,
            mode=mode,
            status="partial" if errors else "ok",
            rows_seen=rows_seen,
            tool_inserted=tool_stats.inserted,
            tool_updated=tool_stats.updated,
            sessions_written=session_stats.total,
            skipped=skipped,
            errors=errors,
            reconciled=reconciled,
            notes=(summary, "session/tool history only (no token counts)", *notes),
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


class ClineTaskAdapter(VSCodeTaskAdapter):
    source_id: ClassVar[str] = "cline:tasks"
    agent: ClassVar[Harness] = Harness.CLINE
    extension_id: ClassVar[str] = _EXTENSION_IDS[Harness.CLINE]


class RooTaskAdapter(VSCodeTaskAdapter):
    source_id: ClassVar[str] = "roo:tasks"
    agent: ClassVar[Harness] = Harness.ROO
    extension_id: ClassVar[str] = _EXTENSION_IDS[Harness.ROO]


class KiloCodeTaskAdapter(VSCodeTaskAdapter):
    source_id: ClassVar[str] = "kilocode:tasks"
    agent: ClassVar[Harness] = Harness.KILOCODE
    extension_id: ClassVar[str] = _EXTENSION_IDS[Harness.KILOCODE]


def _blocks(message: dict) -> list[dict]:
    """Content blocks of a message, flattened and filtered to dicts."""
    content = message.get("content")
    if isinstance(content, dict):
        return [content]
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, dict)]


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _tool_event(
    agent: Harness,
    source_id: str,
    *,
    call_id: str,
    name: str,
    start_ms: int,
    end_ms: int | None,
    is_error: bool | None,
    model: str | None,
    confidence: float | None,
    outcome: Outcome = Outcome.UNKNOWN,
    provenance: OutcomeProvenance = OutcomeProvenance.ABSENT,
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
    duration = end_ms - start_ms if end_ms is not None and end_ms >= start_ms else None
    return ToolEvent(
        id=stable_event_id(kind="tool", agent=str(agent), source_event_id=call_id),
        agent=agent,
        source_id=source_id,
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
        source_event_type="content_block",
    )


def _finalize_usage(event: UsageEvent) -> UsageEvent:
    derived = event.consistency_notes()
    if derived:
        quality = event.quality if event.quality is not Quality.OK else Quality.PARTIAL
        return dataclasses.replace(event, quality=quality, quality_notes=derived)
    return event


@dataclass(slots=True)
class _SessionOutcome:
    rows_seen: int = 0
    skipped: int = 0
    errors: int = 0
    usage: list[UsageEvent] = field(default_factory=list)
    tools: list[ToolEvent] = field(default_factory=list)
    session: SessionRecord | None = None
    notes: tuple[str, ...] = ()
