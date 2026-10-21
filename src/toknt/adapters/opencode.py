"""OpenCode adapter.

OpenCode's local storage has three generations, all of which may coexist in
one data directory:

* **v2 SQLite** (current): ``session_v2`` / ``session_message`` tables. Tool
  calls are embedded inside assistant message ``content`` arrays.
* **v1 SQLite** (legacy): ``session`` / ``message`` / ``part`` tables. The
  same message IDs appear in both generations on migrated databases, so
  events are keyed by message ID and tool events by call ID — overlapping
  storage can never double count.
* **File storage** (oldest): ``storage/message/<session>/<msg>.json``.

Forked sessions: forking copies the parent's message history into the new
session under *new* message ids (id-level dedupe cannot see it), while the
fork's own ``session_v2.tokens_*`` aggregate starts from zero — opencode's
stats screen sums those aggregates, so summing every message row counts a
fork chain's shared history once per link (verified against a live corpus:
two fork chains inflated toknt's total 2×, 5.4B vs opencode's 2.7B). Rows
created before the fork's ``time_created`` are therefore skipped as copies;
the per-session reconciliation against ``session_v2.tokens_*`` then holds
within ~2% on every session.

Only accounting-relevant fields are read: the adapter never selects the
account, credential or permission tables, and never stores message content.
The database is opened strictly read-only (``mode=ro`` URI).

Incremental scans re-read a short trailing window of rows (``OVERLAP_MS``)
keyed on ``time_updated``, because a streaming assistant message gains its
final token counts *after* it was first written. A full scan additionally
reconciles deleted rows.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from pathlib import Path
from typing import ClassVar

from toknt.adapters.base import Adapter, ScanReport, categorize_tool, normalize_epoch_ms
from toknt.discovery import DiscoveredSource, discover_opencode
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

#: Trailing window re-read on incremental scans (see module docstring).
#: Measured on a live corpus of 30k assistant messages: the median delay
#: between a message being written and being finalised is 9 s, the 90th
#: percentile is 50 s, and only 4 messages in 30k were finalised more than
#: an hour later (the longest was 100 minutes). Six hours therefore covers
#: every late finalisation seen so far, while re-reading 1.4k rows instead
#: of 18k — the difference between a scan that takes seconds and one that
#: takes milliseconds.
OVERLAP_MS = 6 * 3_600_000

#: Each table gets its OWN checkpoint stream: sharing one key would let the
#: legacy table's stale max(time_updated) overwrite the current table's
#: watermark, silently shrinking every later incremental window.
_MESSAGE_STREAM = "session_message"

#: Checkpoint stream holding the *file state* of the source database (its
#: size and modification time, plus its write-ahead log). It is the cheap
#: "did anything change?" signal: the message table has no index on
#: ``time_updated``, so even a trailing-window query costs a full table
#: scan of a database with tens of thousands of JSON rows.
_DB_FILE_STREAM = "dbfile"
_LEGACY_STREAM = "message"
_PART_STREAM = "part"

_STATUS_OUTCOMES: dict[str, Outcome] = {
    "completed": Outcome.SUCCESS,
    "error": Outcome.FAILURE,
    "failed": Outcome.FAILURE,
    "pending": Outcome.PENDING,
    "running": Outcome.PENDING,
    "cancelled": Outcome.CANCELLED,
    "canceled": Outcome.CANCELLED,
    "denied": Outcome.DENIED,
}


def _token_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = int(value)
    return number if number >= 0 else None


def _opt_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _as_dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _database_file_state(db_path: Path) -> str:
    """A cheap fingerprint of the source database: size and mtime.

    The database file and its write-ahead log are the only things that
    change when opencode writes; stat'ing them costs microseconds, while
    re-reading the message table costs seconds. The shared-memory index
    (``-shm``) is deliberately left out: readers update it, and that would
    report a change where there is none.
    """
    parts: list[str] = []
    for path in (db_path, db_path.with_name(db_path.name + "-wal")):
        try:
            stat = path.stat()
            parts.append(f"{stat.st_mtime_ns}:{stat.st_size}")
        except OSError:
            parts.append("0:0")
    return "|".join(parts)


class OpenCodeAdapter(Adapter):
    source_id: ClassVar[str] = "opencode:db"
    agent: ClassVar[Agent] = Agent.OPENCODE
    kind: ClassVar[str] = "sqlite"

    def discover(
        self,
        *,
        env: dict[str, str] | None = None,
        home: object | None = None,
        override: str | None = None,
    ) -> DiscoveredSource:
        return discover_opencode(env=env, home=home, override=override)

    # ------------------------------------------------------------------ scan

    def scan(
        self,
        conn: sqlite3.Connection,
        source: DiscoveredSource,
        *,
        full: bool = False,
    ) -> ScanReport:
        data_dir = Path(source.location)
        db_path = data_dir / "opencode.db"
        if db_path.exists():
            return self._scan_database(conn, db_path, full=full)
        return self._scan_file_storage(conn, data_dir, full=full)

    # -------------------------------------------------------------- database

    def _scan_database(
        self, conn: sqlite3.Connection, db_path: Path, *, full: bool
    ) -> ScanReport:
        mode = "full" if full else "incremental"
        # The state is read *before* the scan and stored after it: a write
        # that lands while we are reading changes the state, so the next
        # scan notices and re-reads instead of trusting a half-read file.
        file_state = _database_file_state(db_path)
        if not full:
            checkpoint = load_checkpoint(conn, self.source_id, _DB_FILE_STREAM)
            if checkpoint is not None and checkpoint.fingerprint == file_state:
                # Nothing on disk changed since the last scan: re-reading
                # the message table would walk every row for no new data.
                return ScanReport(
                    source_id=self.source_id,
                    mode=mode,
                    status="ok",
                    rows_seen=0,
                    notes=("no changes since the last scan",),
                )
        try:
            src = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
        except sqlite3.Error as exc:
            return ScanReport(
                source_id=self.source_id,
                mode=mode,
                status="error",
                errors=1,
                notes=(f"cannot open database: {exc}",),
            )
        try:
            src.row_factory = sqlite3.Row
            src.execute("PRAGMA busy_timeout = 5000")
            if full:
                # A full rescan re-observes everything: stale issues would
                # otherwise survive fixed sources forever.
                clear_quality_issues(conn, source_id=self.source_id)
            report = self._ingest_database(conn, src, mode=mode, full=full)
        except sqlite3.Error as exc:
            return ScanReport(
                source_id=self.source_id,
                mode=mode,
                status="error",
                errors=1,
                notes=(f"database read failed: {exc}",),
            )
        finally:
            src.close()
        if report.status != "error":
            save_checkpoint(
                conn,
                self.source_id,
                _DB_FILE_STREAM,
                fingerprint=file_state,
            )
        return report

    def _ingest_database(
        self, conn: sqlite3.Connection, src: sqlite3.Connection, *, mode: str, full: bool
    ) -> ScanReport:
        tables = {
            row["name"]
            for row in src.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        if "session_message" not in tables and "message" not in tables:
            return ScanReport(
                source_id=self.source_id,
                mode=mode,
                status="error",
                errors=1,
                notes=("database has neither session_message nor message tables",),
            )

        sessions, projects = self._load_sessions(src, tables)
        session_records = self._session_records(sessions, projects)
        session_stats = upsert_sessions(conn, session_records)

        usage_events: list[UsageEvent] = []
        tool_events: list[ToolEvent] = []
        malformed: list[tuple[str, str]] = []  # (sample_ref, reason)
        error_notes: list[str] = []
        info_notes: list[str] = []
        rows_seen = 0
        skipped = 0
        fork_copies = 0
        watermark = self._watermark_ms(conn, _MESSAGE_STREAM) if not full else 0

        for table, era in (("session_message", "v2"), ("message", "v1")):
            if table not in tables:
                continue
            if table == "session_message":
                sql = (
                    "SELECT * FROM session_message "
                    "WHERE type = 'assistant' AND time_updated >= ?"
                )
            else:
                sql = "SELECT * FROM message WHERE time_updated >= ?"
            try:
                rows = src.execute(sql, (watermark,)).fetchall()
            except sqlite3.Error as exc:
                error_notes.append(f"{table}: {exc}")
                continue
            for row in rows:
                rows_seen += 1
                data = self._parse_json(row["data"], malformed, f"{table}:{row['id']}")
                if data is None:
                    continue
                if not self._row_matches_role(data, table):
                    continue
                meta = self._message_meta(row, data, sessions, projects, era)
                if self._is_fork_copy(meta, sessions):
                    # opencode forks copy the parent's message history into
                    # the new session under new ids; the fork's own aggregate
                    # starts at zero, so these rows would double count the
                    # parent's responses
                    fork_copies += 1
                    continue
                usage = self._usage_event(row["id"], data, meta, era)
                if usage is not None:
                    usage_events.append(usage)
                for event in self._content_tools(data, row["id"], meta):
                    tool_events.append(event)
        if fork_copies:
            info_notes.append(
                f"{fork_copies} forked-history rows skipped "
                "(copied into a fork by opencode)"
            )

        # v1 part table (tool calls of the legacy generation)
        if "part" in tables:
            part_watermark = self._watermark_ms(conn, _PART_STREAM) if not full else 0
            try:
                rows = src.execute(
                    "SELECT * FROM part WHERE time_updated >= ?", (part_watermark,)
                ).fetchall()
            except sqlite3.Error as exc:
                error_notes.append(f"part: {exc}")
                rows = []
            for row in rows:
                rows_seen += 1
                data = self._parse_json(row["data"], malformed, f"part:{row['id']}")
                if data is None or data.get("type") != "tool":
                    continue
                session = sessions.get(row["session_id"], {})
                meta = {
                    "message_id": row["message_id"],
                    "session_id": row["session_id"],
                    "parent_session_id": session.get("parent_id"),
                    "project": projects.get(session.get("project_id")) or session.get("directory"),
                    "directory": session.get("directory"),
                    "agent_mode": session.get("agent"),
                    "model": None,
                    "ts_ms": normalize_epoch_ms(row["time_created"]),
                    "source_event_type": "part.tool",
                }
                event = self._tool_from_part(data, row["id"], meta)
                if event is not None:
                    tool_events.append(event)

        usage_stats = upsert_usage_events(conn, usage_events)
        tool_stats = upsert_tool_events(conn, tool_events)

        malformed_issue_count = self._record_malformed(
            conn, [(ref, reason) for ref, reason in malformed]
        )
        skipped += len(malformed)

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

        fingerprint = self._database_fingerprint(src, tables)
        self._save_watermarks(conn, src, tables)

        errors = malformed_issue_count
        status = "partial" if (errors or error_notes) else "ok"
        if session_stats.inserted or session_stats.updated or session_records:
            pass  # sessions are upserted on every scan; nothing further to do
        return ScanReport(
            source_id=self.source_id,
            mode=mode,
            status=status,
            rows_seen=rows_seen,
            usage_inserted=usage_stats.inserted,
            usage_updated=usage_stats.updated,
            tool_inserted=tool_stats.inserted,
            tool_updated=tool_stats.updated,
            sessions_written=session_stats.total,
            skipped=skipped,
            errors=errors,
            reconciled=reconciled,
            fingerprint=fingerprint,
            notes=(*info_notes, *error_notes),
        )

    # ---------------------------------------------------------------- loading

    @staticmethod
    def _tables(src: sqlite3.Connection) -> set[str]:
        return {
            row["name"]
            for row in src.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }

    @staticmethod
    def _parse_json(
        raw: object, malformed: list[tuple[str, str]], sample_ref: str
    ) -> dict | None:
        if not isinstance(raw, str):
            malformed.append((sample_ref, "row data is not text"))
            return None
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            malformed.append((sample_ref, "row data is not valid JSON"))
            return None
        if not isinstance(value, dict):
            malformed.append((sample_ref, "row data is not a JSON object"))
            return None
        return value

    @staticmethod
    def _row_matches_role(data: dict, table: str) -> bool:
        if table == "session_message":
            return True  # the SQL query already filters type = 'assistant'
        return data.get("role") == "assistant"

    def _load_sessions(
        self, src: sqlite3.Connection, tables: set[str]
    ) -> tuple[dict[str, dict], dict[str, str]]:
        sessions: dict[str, dict] = {}
        for table in ("session", "session_v2"):  # v2 wins on overlap
            if table not in tables:
                continue
            for row in src.execute(f"SELECT * FROM {table}"):
                sessions[row["id"]] = dict(row)
        projects: dict[str, str] = {}
        if "project" in tables:
            for row in src.execute("SELECT id, worktree FROM project"):
                if isinstance(row["worktree"], str):
                    projects[row["id"]] = row["worktree"]
        return sessions, projects

    def _session_records(
        self, sessions: dict[str, dict], projects: dict[str, str]
    ) -> list[SessionRecord]:
        records = []
        for session_id, row in sessions.items():
            records.append(
                SessionRecord(
                    agent=self.agent,
                    session_id=session_id,
                    source_id=self.source_id,
                    parent_session_id=_opt_str(row.get("parent_id")),
                    project=projects.get(row.get("project_id")) or _opt_str(row.get("directory")),
                    directory=_opt_str(row.get("directory")),
                    title=_opt_str(row.get("title")),
                    agent_mode=_opt_str(row.get("agent")),
                    model=self._session_model(row.get("model")),
                    provider=None,
                    time_created_ms=normalize_epoch_ms(row.get("time_created")),
                    time_updated_ms=normalize_epoch_ms(row.get("time_updated")),
                    cost_reported=self._reported_cost(row.get("cost")),
                )
            )
        return records

    @staticmethod
    def _session_model(value: object) -> str | None:
        if isinstance(value, str):
            if value.startswith("{"):
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    return value
                if isinstance(parsed, dict):
                    return _opt_str(parsed.get("id")) or _opt_str(parsed.get("modelID"))
            return value or None
        if isinstance(value, dict):
            return _opt_str(value.get("id"))
        return None

    @staticmethod
    def _reported_cost(value: object) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value) if value >= 0 else None

    # --------------------------------------------------------------- mapping

    @staticmethod
    def _is_fork_copy(meta: dict, sessions: dict[str, dict]) -> bool:
        """Whether this message row is history copied in by an opencode fork.

        A forked session's ``fork_session_id`` is set and its ``time_created``
        is the fork instant; copied rows keep their original creation time,
        so anything strictly older than that instant arrived with the copy.
        """
        session = sessions.get(meta["session_id"], {})
        fork_of = _opt_str(session.get("fork_session_id"))
        if fork_of is None:
            return False
        fork_time = normalize_epoch_ms(session.get("time_created"))
        ts = meta.get("ts_ms")
        if fork_time is None or ts is None:
            return False
        return ts < fork_time

    def _message_meta(
        self,
        row: sqlite3.Row,
        data: dict,
        sessions: dict[str, dict],
        projects: dict[str, str],
        era: str,
    ) -> dict:
        session_id = row["session_id"]
        session = sessions.get(session_id, {})
        path_info = _as_dict(data.get("path"))
        time_info = _as_dict(data.get("time"))
        ts = normalize_epoch_ms(time_info.get("created")) or normalize_epoch_ms(
            row["time_created"]
        )
        return {
            "message_id": row["id"],
            "session_id": session_id,
            "parent_session_id": _opt_str(session.get("parent_id")),
            "project": (
                projects.get(session.get("project_id"))
                or _opt_str(session.get("directory"))
            ),
            "directory": _opt_str(session.get("directory")) or _opt_str(path_info.get("cwd")),
            "agent_mode": (
                _opt_str(data.get("agent"))
                or _opt_str(data.get("mode"))
                or _opt_str(session.get("agent"))
            ),
            "model": None,
            "ts_ms": ts,
            "source_event_type": f"{'session_message' if era == 'v2' else 'message'}.assistant",
        }

    def _usage_event(self, message_id: str, data: dict, meta: dict, era: str) -> UsageEvent | None:
        tokens = data.get("tokens")
        if not isinstance(tokens, dict):
            return None
        if meta["ts_ms"] is None:
            return None
        cache = _as_dict(tokens.get("cache"))
        counts = TokenCounts(
            input=_token_int(tokens.get("input")),
            output=_token_int(tokens.get("output")),
            reasoning=_token_int(tokens.get("reasoning")),
            cache_read=_token_int(cache.get("read", tokens.get("cache_read"))),
            cache_write=_token_int(cache.get("write", tokens.get("cache_write"))),
        )
        model_info = _as_dict(data.get("model"))
        event = UsageEvent(
            id=stable_event_id(kind="usage", agent=str(self.agent), source_event_id=message_id),
            agent=self.agent,
            source_id=self.source_id,
            source_event_id=message_id,
            session_id=meta["session_id"],
            parent_session_id=meta["parent_session_id"],
            project=meta["project"],
            directory=meta["directory"],
            agent_mode=meta["agent_mode"],
            provider=_opt_str(data.get("providerID")) or _opt_str(model_info.get("providerID")),
            model=_opt_str(data.get("modelID")) or _opt_str(model_info.get("id")),
            ts_ms=meta["ts_ms"],
            tokens=counts,
            token_scheme=TokenScheme.ANTHROPIC,
            provider_total_tokens=_token_int(tokens.get("total")),
            cost_reported=self._reported_cost(data.get("cost")),
            cost_currency="USD" if self._reported_cost(data.get("cost")) is not None else None,
            finish_reason=_opt_str(data.get("finish")),
            source_event_type=meta["source_event_type"],
        )
        derived = event.consistency_notes()
        if derived:
            quality = event.quality if event.quality is not Quality.OK else Quality.PARTIAL
            return dataclasses.replace(event, quality=quality, quality_notes=derived)
        return event

    def _content_tools(self, data: dict, message_id: str, meta: dict) -> list[ToolEvent]:
        events = []
        content = data.get("content")
        if not isinstance(content, list):
            return events
        for index, item in enumerate(content):
            if not isinstance(item, dict) or item.get("type") != "tool":
                continue
            state = _as_dict(item.get("state"))
            time_info = _as_dict(state.get("time"))
            start = normalize_epoch_ms(time_info.get("start"))
            end = normalize_epoch_ms(time_info.get("end"))
            call_id = _opt_str(item.get("id")) or _opt_str(item.get("callID"))
            event = self._tool_event(
                tool_name=_opt_str(item.get("name")) or _opt_str(item.get("tool")),
                call_id=call_id,
                fallback_ref=f"{message_id}:{index}",
                status=_opt_str(state.get("status")),
                start_ms=start or meta["ts_ms"],
                end_ms=end,
                meta=meta,
                source_event_type="message.content.tool",
            )
            if event is not None:
                events.append(event)
        return events

    def _tool_from_part(self, data: dict, part_id: str, meta: dict) -> ToolEvent | None:
        state = _as_dict(data.get("state"))
        time_info = _as_dict(state.get("time"))
        start = normalize_epoch_ms(time_info.get("start"))
        end = normalize_epoch_ms(time_info.get("end"))
        if start is None and isinstance(data.get("time"), dict):
            start = normalize_epoch_ms(_as_dict(data.get("time")).get("start"))
        return self._tool_event(
            tool_name=_opt_str(data.get("tool")),
            call_id=_opt_str(data.get("callID")),
            fallback_ref=part_id,
            status=_opt_str(state.get("status")),
            start_ms=start or meta["ts_ms"],
            end_ms=end,
            meta=meta,
            source_event_type="part.tool",
        )

    def _tool_event(
        self,
        *,
        tool_name: str | None,
        call_id: str | None,
        fallback_ref: str,
        status: str | None,
        start_ms: int | None,
        end_ms: int | None,
        meta: dict,
        source_event_type: str,
    ) -> ToolEvent | None:
        if start_ms is None:
            return None
        if status is None:
            outcome, provenance, confidence, note = (
                Outcome.UNKNOWN,
                OutcomeProvenance.ABSENT,
                None,
                None,
            )
        elif status in _STATUS_OUTCOMES:
            outcome = _STATUS_OUTCOMES[status]
            provenance = OutcomeProvenance.STATUS_FIELD
            confidence = 1.0
            note = None
        else:
            outcome, provenance, confidence = (
                Outcome.UNKNOWN,
                OutcomeProvenance.ABSENT,
                None,
            )
            note = f"unrecognized tool status: {status}"
        duration = None
        if end_ms is not None and end_ms >= start_ms:
            duration = end_ms - start_ms
        identity = call_id or fallback_ref
        event = ToolEvent(
            id=stable_event_id(kind="tool", agent=str(self.agent), source_event_id=identity),
            agent=self.agent,
            source_id=self.source_id,
            source_event_id=identity,
            session_id=meta["session_id"],
            call_id=call_id,
            tool_name=tool_name or "unknown",
            tool_category=categorize_tool(tool_name),
            ts_ms=start_ms,
            duration_ms=duration,
            outcome=outcome,
            outcome_provenance=provenance,
            outcome_confidence=confidence,
            model=meta.get("model"),
            agent_mode=meta["agent_mode"],
            quality=Quality.OK if note is None else Quality.AMBIGUOUS,
            quality_notes=(note,) if note else (),
            source_event_type=source_event_type,
        )
        return event

    # ------------------------------------------------------------- file storage

    def _scan_file_storage(
        self, conn: sqlite3.Connection, data_dir: Path, *, full: bool
    ) -> ScanReport:
        """Legacy file-based storage: ``storage/message/<session>/<message>.json``."""
        mode = "full" if full else "incremental"
        message_root = data_dir / "storage" / "message"
        if not message_root.is_dir():
            return ScanReport(
                source_id=self.source_id,
                mode=mode,
                status="error",
                errors=1,
                notes=(f"no database and no legacy message storage under {data_dir}",),
            )
        watermark = self._watermark_ms(conn, _MESSAGE_STREAM) if not full else 0
        if full:
            # A full rescan re-observes everything: stale issues would
            # otherwise survive fixed sources forever.
            clear_quality_issues(conn, source_id=self.source_id)
        usage_events: list[UsageEvent] = []
        malformed: list[tuple[str, str]] = []
        rows_seen = 0
        for path in sorted(message_root.rglob("*.json")):
            mtime_ms = path.stat().st_mtime_ns // 1_000_000
            if not full and mtime_ms < watermark:
                continue
            rows_seen += 1
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                malformed.append((str(path), "cannot read message file"))
                continue
            if not isinstance(data, dict):
                malformed.append((str(path), "message file is not a JSON object"))
                continue
            if data.get("role") != "assistant":
                continue
            message_id = _opt_str(data.get("id")) or path.stem
            meta = {
                "message_id": message_id,
                "session_id": path.parent.name,
                "parent_session_id": None,
                "project": None,
                "directory": _opt_str(_as_dict(data.get("path")).get("cwd")),
                "agent_mode": _opt_str(data.get("agent")) or _opt_str(data.get("mode")),
                "model": None,
                "ts_ms": normalize_epoch_ms(_as_dict(data.get("time")).get("created")) or mtime_ms,
                "source_event_type": "storage.message.assistant",
            }
            event = self._usage_event(message_id, data, meta, era="v1")
            if event is not None:
                usage_events.append(event)

        stats = upsert_usage_events(conn, usage_events)
        errors = self._record_malformed(conn, malformed)
        reconciled = 0
        if full:
            reconciled = reconcile_source_events(
                conn,
                source_id=self.source_id,
                table="usage_events",
                seen_ids=[event.id for event in usage_events],
            )
        watermark_after = max((event.ts_ms for event in usage_events), default=watermark)
        save_checkpoint(
            conn,
            self.source_id,
            _MESSAGE_STREAM,
            offset=watermark_after,
            size=rows_seen,
            fingerprint=f"{watermark_after}:{rows_seen}",
        )
        status = "partial" if errors else "ok"
        return ScanReport(
            source_id=self.source_id,
            mode=mode,
            status=status,
            rows_seen=rows_seen,
            usage_inserted=stats.inserted,
            usage_updated=stats.updated,
            skipped=len(malformed),
            errors=errors,
            reconciled=reconciled,
            notes=("legacy file storage",),
        )

    # -------------------------------------------------------------- helpers

    def _watermark_ms(self, conn: sqlite3.Connection, stream: str) -> int:
        checkpoint = load_checkpoint(conn, self.source_id, stream)
        if checkpoint is None or checkpoint.offset <= 0:
            return 0
        return max(0, checkpoint.offset - OVERLAP_MS)

    @staticmethod
    def _database_fingerprint(src: sqlite3.Connection, tables: set[str]) -> str:
        parts: list[str] = []
        for table in ("session_message", "message", "part"):
            if table not in tables:
                continue
            try:
                row = src.execute(
                    f"SELECT COUNT(*) AS n, COALESCE(MAX(time_updated), 0) AS m FROM {table}"
                ).fetchone()
                parts.append(f"{table}:{row['n']}:{row['m']}")
            except sqlite3.Error:
                continue
        return "|".join(parts)

    def _save_watermarks(
        self, conn: sqlite3.Connection, src: sqlite3.Connection, tables: set[str]
    ) -> None:
        streams = (
            ("session_message", _MESSAGE_STREAM),
            ("message", _LEGACY_STREAM),
            ("part", _PART_STREAM),
        )
        for table, stream in streams:
            if table not in tables:
                continue
            try:
                row = src.execute(
                    f"SELECT COUNT(*) AS n, COALESCE(MAX(time_updated), 0) AS m FROM {table}"
                ).fetchone()
            except sqlite3.Error:
                continue
            if row["m"] and row["m"] > 0:
                save_checkpoint(
                    conn,
                    self.source_id,
                    stream,
                    offset=int(row["m"]),
                    size=int(row["n"]),
                    fingerprint=f"{row['m']}:{row['n']}",
                )

    def _record_malformed(
        self, conn: sqlite3.Connection, malformed: list[tuple[str, str]]
    ) -> int:
        """One issue row per affected record (keyed by sample_ref).

        Rows are re-observed on every scan (overlapping incremental windows
        included), so per-record replace-on-rescan keeps doctor counts stable;
        the summary adds distinct records per reason.
        """
        for sample_ref, reason in malformed:
            record_quality_issue(
                conn,
                agent=self.agent,
                source_id=self.source_id,
                code="unparsable_row",
                message=f"Skipped row(s): {reason}",
                occurrences=1,
                sample_ref=sample_ref,
            )
        return len(malformed)
