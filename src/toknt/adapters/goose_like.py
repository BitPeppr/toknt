"""Goose adapter (Block) — SQLite session store plus the legacy JSONL tree.

Current Goose builds (≥ 1.10) keep sessions in a relational database at
``<data>/goose/sessions/sessions.db`` (``sessions`` joined to ``messages`` on
``session_id``; ``content_json`` holds a JSON array of content blocks, text
blocks only in v1). Older builds wrote one JSONL file per session: a
metadata header (``id``, ``working_dir``, ``created_at``, ``updated_at``)
followed by message records with ``role``/``created`` (unix seconds).

Neither store records per-response token usage, so this adapter contributes
sessions only — no usage events, no tool events. The session list still
feeds the sessions view, session counts and harness coverage; the missing
tokens stay missing rather than estimated.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import ClassVar

from toknt.adapters.base import (
    Adapter,
    ScanReport,
    normalize_epoch_ms,
    opt_str,
)
from toknt.discovery import DiscoveredSource
from toknt.models.events import Harness, Quality, SessionRecord
from toknt.storage.ingestion import (
    clear_quality_issues,
    load_checkpoint,
    reconcile_source_events,
    record_quality_issue,
    save_checkpoint,
    upsert_sessions,
)

_SESSION_NOTE = "goose stores carry no token counts; sessions only"


class GooseAdapter(Adapter):
    source_id: ClassVar[str] = "goose:sessions"
    agent: ClassVar[Harness] = Harness.GOOSE
    kind: ClassVar[str] = "sqlite"

    def discover(
        self,
        *,
        env: dict[str, str] | None = None,
        home: object | None = None,
        override: str | None = None,
    ) -> DiscoveredSource:
        from toknt.discovery import discover_goose

        return discover_goose(env=env, home=home, override=override)

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
            return self._scan_db(conn, location, mode=mode, full=full)
        if location.is_dir():
            return self._scan_legacy(conn, location, mode=mode, full=full)
        return ScanReport(
            source_id=self.source_id,
            mode=mode,
            status="error",
            errors=1,
            notes=(f"no sessions.db or JSONL store at {location}",),
        )

    def _scan_db(
        self, conn: sqlite3.Connection, db_path: Path, *, mode: str, full: bool
    ) -> ScanReport:
        if full:
            clear_quality_issues(conn, source_id=self.source_id)
        sessions: dict[str, SessionRecord] = {}
        rows = 0
        errors = 0
        notes: tuple[str, ...] = ()
        try:
            db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            try:
                columns = {
                    row[1] for row in db.execute("PRAGMA table_info(sessions)").fetchall()
                }
                working_dir = "working_dir" if "working_dir" in columns else None
                id_col = "id" if "id" in columns else None
                if id_col is None:
                    raise ValueError("sessions table has no id column")
                projection = f"{id_col}, {working_dir}" if working_dir else id_col
                for row in db.execute(f"SELECT {projection} FROM sessions"):
                    rows += 1
                    session_id = opt_str(row[0]) or f"goose-{rows}"
                    sessions[session_id] = SessionRecord(
                        agent=self.agent,
                        session_id=session_id,
                        source_id=self.source_id,
                        project=opt_str(row[1]) if working_dir else None,
                        directory=opt_str(row[1]) if working_dir else None,
                        quality=Quality.PARTIAL,
                        quality_notes=(_SESSION_NOTE,),
                    )
            finally:
                db.close()
        except Exception as exc:  # noqa: BLE001 — foreign DB shapes degrade
            errors = 1
            notes = (f"sessions.db unreadable: {type(exc).__name__}",)
        stats = upsert_sessions(conn, sessions.values())
        reconciled = 0
        if full:
            reconciled += reconcile_source_events(
                conn,
                source_id=self.source_id,
                table="sessions",
                seen_ids=[record.session_id for record in sessions.values()],
            )
        return ScanReport(
            source_id=self.source_id,
            mode=mode,
            status="partial" if errors else "ok",
            rows_seen=rows,
            sessions_written=stats.total,
            skipped=0,
            errors=errors,
            reconciled=reconciled,
            notes=(f"{rows} sessions read", _SESSION_NOTE, *notes),
        )

    def _scan_legacy(
        self, conn: sqlite3.Connection, root: Path, *, mode: str, full: bool
    ) -> ScanReport:
        if full:
            clear_quality_issues(conn, source_id=self.source_id)
        sessions: dict[str, SessionRecord] = {}
        rows = 0
        errors = 0
        for path in sorted(root.glob("*.jsonl")):
            relpath = path.name
            try:
                stat = path.stat()
            except OSError:
                errors += 1
                continue
            checkpoint = None if full else load_checkpoint(conn, self.source_id, relpath)
            if (
                checkpoint is not None
                and checkpoint.size == stat.st_size
                and checkpoint.mtime_ms == stat.st_mtime_ns // 1_000_000
            ):
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError) as exc:
                self._quality_issue(conn, relpath, exc)
                errors += 1
                continue
            header: dict = {}
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict) and not header and record.get("id"):
                    header = record
                    break
            if header:
                rows += 1
                session_id = opt_str(header.get("id")) or path.stem
                sessions[session_id] = SessionRecord(
                    agent=self.agent,
                    session_id=session_id,
                    source_id=self.source_id,
                    project=opt_str(header.get("working_dir")),
                    directory=opt_str(header.get("working_dir")),
                    title=opt_str(header.get("description")),
                    time_created_ms=normalize_epoch_ms(header.get("created_at")),
                    time_updated_ms=normalize_epoch_ms(header.get("updated_at")),
                    quality=Quality.PARTIAL,
                    quality_notes=(_SESSION_NOTE,),
                )
            save_checkpoint(
                conn,
                self.source_id,
                relpath,
                offset=stat.st_size,
                size=stat.st_size,
                mtime_ms=stat.st_mtime_ns // 1_000_000,
            )
        stats = upsert_sessions(conn, sessions.values())
        reconciled = 0
        if full:
            reconciled += reconcile_source_events(
                conn,
                source_id=self.source_id,
                table="sessions",
                seen_ids=[record.session_id for record in sessions.values()],
            )
        return ScanReport(
            source_id=self.source_id,
            mode=mode,
            status="partial" if errors else "ok",
            rows_seen=rows,
            sessions_written=stats.total,
            skipped=0,
            errors=errors,
            reconciled=reconciled,
            notes=(f"legacy JSONL: {rows} sessions", _SESSION_NOTE),
        )

    def _quality_issue(self, conn: sqlite3.Connection, relpath: str, exc: Exception) -> None:
        record_quality_issue(
            conn,
            agent=self.agent,
            source_id=self.source_id,
            code="unparsable_file",
            message=f"cannot read session file: {type(exc).__name__}",
            sample_ref=relpath,
        )

