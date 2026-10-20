"""Gemini-CLI-shaped adapters (Gemini CLI, Qwen Code).

Gemini CLI records chats under ``~/.gemini/tmp/<project-hash>/chats``. Two
generations coexist: whole-session ``session-*.json`` objects
(``sessionId``/``startTime``/``lastUpdated`` + ``messages`` array) and
replayable ``session-*.jsonl`` files (metadata first line, then message and
control lines — ``$set`` metadata patches, ``$patch`` tool-result updates,
``$rewindTo`` history trims). Qwen Code, a Gemini CLI fork, writes the same
records under ``~/.qwen/projects/*/chats``.

Record mapping: ``type: "user"`` → user turns; ``gemini`` (or ``model``) →
assistant; other types (``info``/``error``/``warning``) are bookkeeping and
ignored. Assistant messages carry ``tokens`` in Gemini API categories —
``{input, output, cached, thoughts, tool, total}`` — and ``toolCalls`` with
a ``status`` per call.

Token semantics: in every observed sample
``total == input + output + thoughts``, and ``cached`` is a subset of
``input`` (the Gemini API's ``cachedContentTokenCount`` is part of
``promptTokenCount``). toknt's openai scheme matches the cached relationship
but not the additive ``thoughts``, so usage is recorded as openai-scheme
``input``/``output`` with ``thoughts`` surfaced in the ``reasoning`` field;
whenever ``thoughts > 0`` the consistency check records
``provider_total_mismatch`` — the reported total counts thoughts additively
while the scheme's effective total (input + output) deliberately does not.
No session on disk is miscounted silently: the discrepancy is flagged on
every affected event. ``tool`` prompt tokens are ignored (they are part of
the prompt count the API bills).
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
    "error": Outcome.FAILURE,
    "cancelled": Outcome.CANCELLED,
    "awaiting_approval": Outcome.PENDING,
    "validating": Outcome.PENDING,
    "scheduled": Outcome.PENDING,
    "executing": Outcome.PENDING,
}

_SKIP_SUFFIXES = (".checkpoints.jsonl", ".meta.json")


class GeminiFamilyAdapter(Adapter):
    """Base adapter for Gemini-shaped chat stores; subclasses pin the root."""

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
                notes=(f"chats directory not found: {root}",),
            )
        if full:
            clear_quality_issues(conn, source_id=self.source_id)

        usage_events: list[UsageEvent] = []
        tool_events: list[ToolEvent] = []
        sessions: dict[str, SessionRecord] = {}
        rows_seen = skipped = errors = files_scanned = 0
        notes: list[str] = []

        paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix in (".json", ".jsonl")
            and path.name.startswith("session-")
            and not path.name.endswith(_SKIP_SUFFIXES)
        )
        for path in paths:
            if path.suffix == ".jsonl":
                outcome = self._scan_jsonl(conn, root, path, full=full)
            else:
                outcome = self._scan_json(conn, root, path, full=full)
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

        # A resumed session can be rewritten from legacy .json to .jsonl; the
        # jsonl record wins because upserts are keyed by session id.
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

    # ---------------------------------------------------------- whole-session

    def _scan_json(
        self, conn: sqlite3.Connection, root: Path, path: Path, *, full: bool
    ) -> _FileOutcome | None:
        relpath = path.relative_to(root).as_posix()
        changed, _checkpoint = _whole_file_changed(
            conn, self.source_id, relpath, path, full=full
        )
        if not changed:
            return None
        try:
            record = _load_json(path)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._quality_issue(
                conn,
                code="unparsable_file",
                message=f"cannot read session: {type(exc).__name__}",
                sample_ref=relpath,
            )
            return _FileOutcome(errors=1)
        record = as_dict(record)
        result = _Processed()
        session_id = opt_str(record.get("sessionId"))
        if not session_id:
            return _FileOutcome(skipped=1, notes=("session without an id",))
        start_ms = parse_iso8601_ms(record.get("startTime"))
        updated_ms = parse_iso8601_ms(record.get("lastUpdated"))
        self._emit_messages(
            record.get("messages"),
            session_id=session_id,
            fallback_ts=start_ms,
            result=result,
        )
        session = SessionRecord(
            agent=self.agent,
            session_id=session_id,
            source_id=self.source_id,
            model=opt_str(record.get("model")),
            time_created_ms=start_ms,
            time_updated_ms=updated_ms,
        )
        _save_whole_file(conn, self.source_id, relpath, path)
        return _FileOutcome(
            rows_seen=len(as_dict_list(record.get("messages"))),
            usage=result.usage,
            tools=result.tools,
            session=session,
        )

    # ----------------------------------------------------------------- jsonl

    def _scan_jsonl(
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

        # Session files are replayable: patches and rewrites make byte-offset
        # continuation unsafe, so the file is always re-parsed in full.
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            self._quality_issue(
                conn,
                code="unparsable_file",
                message=f"cannot read session: {type(exc).__name__}",
                sample_ref=relpath,
            )
            return _FileOutcome(errors=1)

        result = _Processed()
        session_id: str | None = None
        start_ms: int | None = None
        updated_ms: int | None = None
        rows = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            rows += 1
            if "$set" in record:
                patch = as_dict(record.get("$set"))
                session_id = opt_str(patch.get("sessionId")) or session_id
                start_ms = parse_iso8601_ms(patch.get("startTime")) or start_ms
                updated_ms = parse_iso8601_ms(patch.get("lastUpdated")) or updated_ms
                self._emit_messages(
                    patch.get("messages"),
                    session_id=session_id or "",
                    fallback_ts=start_ms,
                    result=result,
                )
                continue
            if "$patch" in record or "$rewindTo" in record:
                continue  # tool-result updates / trims; last full message wins
            if session_id is None:
                session_id = opt_str(record.get("sessionId"))
                start_ms = parse_iso8601_ms(record.get("startTime")) or start_ms
                updated_ms = parse_iso8601_ms(record.get("lastUpdated")) or updated_ms
                continue
            if opt_str(record.get("id")) is None:
                continue
            self._emit_messages(
                [record],
                session_id=session_id,
                fallback_ts=start_ms,
                result=result,
            )

        session = (
            SessionRecord(
                agent=self.agent,
                session_id=session_id,
                source_id=self.source_id,
                time_created_ms=start_ms,
                time_updated_ms=updated_ms or max(
                    (event.ts_ms for event in result.usage), default=None
                ),
            )
            if session_id
            else None
        )
        save_checkpoint(
            conn,
            self.source_id,
            relpath,
            offset=stat.st_size,
            size=stat.st_size,
            mtime_ms=stat.st_mtime_ns // 1_000_000,
        )
        return _FileOutcome(rows_seen=rows, usage=result.usage, tools=result.tools, session=session)

    # ------------------------------------------------------------ processing

    def _emit_messages(
        self,
        messages: object,
        *,
        session_id: str,
        fallback_ts: int | None,
        result: _Processed,
    ) -> None:
        for message in as_dict_list(messages):
            message_type = opt_str(message.get("type"))
            ts_ms = (
                parse_iso8601_ms(message.get("timestamp"))
                or fallback_ts
            )
            if message_type == "gemini" or message_type == "model":
                self._assistant(message, session_id=session_id, ts_ms=ts_ms, result=result)
            elif message_type == "user":
                # user turns carry no usage; kept only for session timing
                if ts_ms:
                    result.event_ts.append(ts_ms)
            # info/error/warning and control lines are ignored

    def _assistant(
        self, message: dict, *, session_id: str, ts_ms: int | None, result: _Processed
    ) -> None:
        message_id = opt_str(message.get("id"))
        for call in as_dict_list(message.get("toolCalls")):
            call_id = opt_str(call.get("id"))
            name = opt_str(call.get("name"))
            status = opt_str(call.get("status")) or ""
            if call_id is None or name is None or ts_ms is None:
                result.skipped += 1
                continue
            outcome = _TOOL_OUTCOMES.get(status, Outcome.UNKNOWN)
            if outcome is Outcome.UNKNOWN and opt_str(call.get("result")) is not None:
                outcome = Outcome.SUCCESS
            provenance = (
                OutcomeProvenance.STATUS_FIELD
                if status in _TOOL_OUTCOMES
                else OutcomeProvenance.ABSENT
            )
            result.tools.append(
                ToolEvent(
                    id=stable_event_id(
                        kind="tool", agent=str(self.agent), source_event_id=call_id
                    ),
                    agent=self.agent,
                    source_id=self.source_id,
                    source_event_id=call_id,
                    session_id=session_id,
                    call_id=call_id,
                    tool_name=name,
                    tool_category=categorize_tool(name),
                    ts_ms=ts_ms,
                    outcome=outcome,
                    outcome_provenance=provenance,
                    outcome_confidence=0.9 if status in _TOOL_OUTCOMES else None,
                    model=opt_str(message.get("model")),
                    source_event_type="toolCalls",
                )
            )
        tokens = message.get("tokens")
        if message_id is None or ts_ms is None or not isinstance(tokens, dict):
            if message_id is not None:
                result.skipped += 1
            return
        counts = TokenCounts(
            input=token_int(tokens.get("input")),
            output=token_int(tokens.get("output")),
            reasoning=token_int(tokens.get("thoughts")),
            cache_read=token_int(tokens.get("cached")),
        )
        result.usage.append(
            _finalize_usage(
                UsageEvent(
                    id=stable_event_id(
                        kind="usage", agent=str(self.agent), source_event_id=message_id
                    ),
                    agent=self.agent,
                    source_id=self.source_id,
                    source_event_id=message_id,
                    session_id=session_id,
                    model=opt_str(message.get("model")),
                    ts_ms=ts_ms,
                    tokens=counts,
                    token_scheme=TokenScheme.OPENAI,
                    provider_total_tokens=token_int(tokens.get("total")),
                    source_event_type="message.gemini",
                )
            )
        )
        if ts_ms:
            result.event_ts.append(ts_ms)

    def _quality_issue(
        self,
        conn: sqlite3.Connection,
        *,
        code: str,
        message: str,
        sample_ref: str,
    ) -> None:
        record_quality_issue(
            conn,
            agent=self.agent,
            source_id=self.source_id,
            code=code,
            message=message,
            sample_ref=sample_ref,
        )


def as_dict_list(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _finalize_usage(event: UsageEvent) -> UsageEvent:
    derived = event.consistency_notes()
    if derived:
        quality = event.quality if event.quality is not Quality.OK else Quality.PARTIAL
        return dataclasses.replace(event, quality=quality, quality_notes=derived)
    return event


@dataclass(slots=True)
class _Processed:
    usage: list[UsageEvent] = field(default_factory=list)
    tools: list[ToolEvent] = field(default_factory=list)
    event_ts: list[int] = field(default_factory=list)
    skipped: int = 0


@dataclass(slots=True)
class _FileOutcome:
    rows_seen: int = 0
    skipped: int = 0
    errors: int = 0
    usage: list[UsageEvent] = field(default_factory=list)
    tools: list[ToolEvent] = field(default_factory=list)
    session: SessionRecord | None = None
    notes: tuple[str, ...] = ()


class GeminiAdapter(GeminiFamilyAdapter):
    source_id: ClassVar[str] = "gemini:chats"
    agent: ClassVar[Harness] = Harness.GEMINI

    def discover(
        self,
        *,
        env: dict[str, str] | None = None,
        home: object | None = None,
        override: str | None = None,
    ) -> DiscoveredSource:
        from toknt.discovery import discover_gemini

        return discover_gemini(env=env, home=home, override=override)


class QwenAdapter(GeminiFamilyAdapter):
    source_id: ClassVar[str] = "qwen:projects"
    agent: ClassVar[Harness] = Harness.QWEN

    def discover(
        self,
        *,
        env: dict[str, str] | None = None,
        home: object | None = None,
        override: str | None = None,
    ) -> DiscoveredSource:
        from toknt.discovery import discover_qwen

        return discover_qwen(env=env, home=home, override=override)
