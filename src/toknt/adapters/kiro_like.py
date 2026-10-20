"""Kiro adapter (CLI sessions, IDE/v3 session directories, headless store).

Kiro's CLI and IDE write different files, both JSONL:

* **CLI pair**: ``~/.kiro/sessions/cli/<id>.jsonl`` with a header
  ``<id>.json`` beside it. Records are ``{"kind": "Prompt"|"AssistantMessage",
  "data": {...}}`` with the text in ``data.content[].data`` and
  ``data.meta.timestamp`` in *seconds*. Streaming splits one reply across
  several ``AssistantMessage`` records sharing ``data.message_id``.
* **IDE / ``--v3`` directories**: ``~/.kiro/sessions/<workspace>/sess_<uuid>/``
  holding ``session.json`` (id, model, workspacePaths, timestamps) and
  ``messages.jsonl`` whose lines carry a ``payload`` wrapper (older builds
  wrote flat ``{"role", "content"}`` lines). ``tool_call`` records carry
  ``toolName`` and a ``status`` (``completed``/``failed``/``denied``).

Token counts are not part of any Kiro session format — usage is metered in
credits elsewhere — so this adapter contributes sessions and tool events
only. It never invents token numbers: the harness simply shows no token
data, which the UI labels explicitly.
"""

from __future__ import annotations

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
    stable_event_id,
)
from toknt.storage.ingestion import (
    clear_quality_issues,
    load_checkpoint,
    reconcile_source_events,
    record_quality_issue,
    save_checkpoint,
    upsert_sessions,
    upsert_tool_events,
)

_TOOL_STATUS: dict[str, Outcome] = {
    "completed": Outcome.SUCCESS,
    "failed": Outcome.FAILURE,
    "denied": Outcome.DENIED,
}


class KiroAdapter(Adapter):
    source_id: ClassVar[str] = "kiro:sessions"
    agent: ClassVar[Harness] = Harness.KIRO
    kind: ClassVar[str] = "jsonl_dir"

    def discover(
        self,
        *,
        env: dict[str, str] | None = None,
        home: object | None = None,
        override: str | None = None,
    ) -> DiscoveredSource:
        from toknt.discovery import discover_kiro

        return discover_kiro(env=env, home=home, override=override)

    def scan(
        self,
        conn: sqlite3.Connection,
        source: DiscoveredSource,
        *,
        full: bool = False,
    ) -> ScanReport:
        mode = "full" if full else "incremental"
        location = Path(source.location)
        if location.is_file():
            # headless kiro-cli store discovered directly
            if full:
                clear_quality_issues(conn, source_id=self.source_id)
            outcome = self._scan_headless(conn, location)
            tool_stats = upsert_tool_events(conn, [])
            session_stats = upsert_sessions(conn, outcome.sessions.values())
            return ScanReport(
                source_id=self.source_id,
                mode=mode,
                status="ok",
                rows_seen=outcome.rows_seen,
                tool_inserted=tool_stats.inserted,
                tool_updated=tool_stats.updated,
                sessions_written=session_stats.total,
                reconciled=0,
                notes=(
                    "headless kiro-cli store",
                    "no token counts in kiro session files",
                    *outcome.notes,
                ),
            )
        root = location
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

        sessions: dict[str, SessionRecord] = {}
        tool_events: list[ToolEvent] = []
        rows_seen = skipped = errors = files_scanned = 0
        notes: list[str] = []

        for path in sorted(root.rglob("*.jsonl")):
            outcome = self._scan_file(conn, root, path, full=full)
            if outcome is None:
                continue
            files_scanned += 1
            rows_seen += outcome.rows_seen
            skipped += outcome.skipped
            errors += outcome.errors
            tool_events.extend(outcome.tools)
            if outcome.session is not None:
                sessions[outcome.session.session_id] = outcome.session
            notes.extend(outcome.notes)

        # headless kiro-cli store: conversations_v2 rows, session metadata only
        headless = root.parent / "kiro-cli" / "data.sqlite3"
        if headless.is_file():
            outcome = self._scan_headless(conn, headless)
            sessions.update(outcome.sessions)
            rows_seen += outcome.rows_seen
            notes.extend(outcome.notes)

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
        if not files_scanned and not sessions:
            summary = "no session files found"
        else:
            summary = f"{files_scanned} files read"
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
            notes=(summary, "no token counts in kiro session files", *notes),
        )

    def _scan_file(
        self, conn: sqlite3.Connection, root: Path, path: Path, *, full: bool
    ) -> _FileOutcome | None:
        relpath = path.relative_to(root).as_posix()
        try:
            stat = path.stat()
        except OSError:
            return _FileOutcome(errors=1)
        checkpoint = None if full else load_checkpoint(conn, self.source_id, relpath)
        if (
            checkpoint is not None
            and checkpoint.size == stat.st_size
            and checkpoint.mtime_ms == stat.st_mtime_ns // 1_000_000
        ):
            return None

        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            self._quality_issue(
                conn,
                code="unparsable_file",
                message="cannot read session transcript",
                sample_ref=relpath,
            )
            return _FileOutcome(errors=1)

        session = self._session_from_directory(path, lines)
        tools: list[ToolEvent] = []
        rows = 0
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
            rows += 1
            tool = self._tool_from_record(record, session=session, seq=index)
            if tool is not None:
                tools.append(tool)

        save_checkpoint(
            conn,
            self.source_id,
            relpath,
            offset=stat.st_size,
            size=stat.st_size,
            mtime_ms=stat.st_mtime_ns // 1_000_000,
        )
        return _FileOutcome(rows_seen=rows, tools=tools, session=session)

    def _session_from_directory(self, path: Path, lines: list[str]) -> SessionRecord | None:
        """Session metadata from the pair/directory layout around a transcript."""
        workspace_header = path.with_suffix(".json")
        session_json = path.parent / "session.json"
        model = None
        created = None
        updated = None
        cwd = None
        session_id = path.stem
        header: dict = {}
        if session_json.is_file():
            try:
                header = as_dict(_load_json(session_json))
            except (OSError, json.JSONDecodeError):
                header = {}
            session_id = opt_str(header.get("id")) or session_id
            model = opt_str(header.get("model"))
            created = parse_iso8601_ms(
                opt_str(header.get("startTime") or header.get("created_at"))
            )
            updated = parse_iso8601_ms(
                opt_str(header.get("lastUpdated") or header.get("updated_at"))
            )
            paths = header.get("workspacePaths")
            if isinstance(paths, list) and paths and isinstance(paths[0], str):
                cwd = paths[0]
        elif workspace_header.is_file():
            try:
                header = as_dict(_load_json(workspace_header))
            except (OSError, json.JSONDecodeError):
                header = {}
            session_id = opt_str(header.get("sessionId")) or session_id
            cwd = opt_str(header.get("cwd"))
            created = parse_iso8601_ms(
                opt_str(header.get("timestamp") or header.get("started_at"))
            )
            updated = None
            if header.get("parent_session_id"):
                parent = opt_str(header.get("parent_session_id"))
                return self._session(
                    session_id, cwd=cwd, model=None, created=created,
                    updated=None, parent=parent,
                )
        else:
            updated = None
        return self._session(
            session_id, cwd=cwd, model=model, created=created, updated=updated
        )

    def _session(
        self,
        session_id: str,
        *,
        cwd: str | None,
        model: str | None,
        created: int | None,
        updated: int | None = None,
        parent: str | None = None,
    ) -> SessionRecord:
        return SessionRecord(
            agent=self.agent,
            session_id=session_id,
            source_id=self.source_id,
            parent_session_id=parent,
            project=cwd,
            directory=cwd,
            model=model,
            time_created_ms=created,
            time_updated_ms=updated,
            quality=Quality.PARTIAL,
            quality_notes=("no token counts in kiro session files",),
        )

    def _tool_from_record(
        self, record: dict, *, session: SessionRecord | None, seq: int
    ) -> ToolEvent | None:
        payload = as_dict(record.get("payload")) if "payload" in record else record
        record_type = payload.get("type") or payload.get("kind")
        if record_type != "tool_call" and record_type != "ToolUse":
            return None
        call_id = opt_str(payload.get("toolCallId")) or opt_str(payload.get("callId"))
        name = opt_str(payload.get("toolName")) or opt_str(payload.get("name")) or "unknown"
        status = opt_str(payload.get("status")) or ""
        ts_ms = normalize_epoch_ms(payload.get("timestamp")) or parse_iso8601_ms(
            opt_str(payload.get("timestamp"))
        )
        if ts_ms is None:
            return None
        event_id = call_id or f"{session.session_id if session else 'anon'}:t{seq}"
        outcome = _TOOL_STATUS.get(status, Outcome.UNKNOWN)
        return ToolEvent(
            id=stable_event_id(kind="tool", agent=str(self.agent), source_event_id=event_id),
            agent=self.agent,
            source_id=self.source_id,
            source_event_id=event_id,
            session_id=session.session_id if session else None,
            call_id=call_id,
            tool_name=name,
            tool_category=categorize_tool(name),
            ts_ms=ts_ms,
            outcome=outcome,
            outcome_provenance=(
                OutcomeProvenance.STATUS_FIELD
                if outcome is not Outcome.UNKNOWN
                else OutcomeProvenance.ABSENT
            ),
            outcome_confidence=0.85 if outcome is not Outcome.UNKNOWN else None,
            quality=Quality.PARTIAL,
            source_event_type="tool_call",
        )

    def _scan_headless(
        self, conn: sqlite3.Connection, db_path: Path
    ) -> _HeadlessOutcome:
        """Session rows from the headless CLI's ``conversations_v2`` table."""
        sessions: dict[str, SessionRecord] = {}
        rows = 0
        notes: tuple[str, ...] = ()
        try:
            uri = f"file:{db_path}?mode=ro"
            db = sqlite3.connect(uri, uri=True)
            try:
                cursor = db.execute(
                    "SELECT conversation_id, directory, created_at, updated_at "
                    "FROM conversations_v2"
                )
                for row in cursor.fetchall():
                    rows += 1
                    session_id = opt_str(row[0]) or f"headless-{rows}"
                    sessions[session_id] = SessionRecord(
                        agent=self.agent,
                        session_id=session_id,
                        source_id=self.source_id,
                        project=opt_str(row[1]),
                        directory=opt_str(row[1]),
                        quality=Quality.PARTIAL,
                        quality_notes=("headless store: no token counts",),
                    )
            finally:
                db.close()
        except Exception as exc:  # noqa: BLE001 — the store is optional
            notes = (f"headless store unreadable: {type(exc).__name__}",)
        return _HeadlessOutcome(sessions=sessions, rows_seen=rows, notes=notes)

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


@dataclass(slots=True)
class _FileOutcome:
    rows_seen: int = 0
    skipped: int = 0
    errors: int = 0
    tools: list[ToolEvent] = field(default_factory=list)
    session: SessionRecord | None = None
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class _HeadlessOutcome:
    sessions: dict[str, SessionRecord] = field(default_factory=dict)
    rows_seen: int = 0
    notes: tuple[str, ...] = ()
