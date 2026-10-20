"""Application services that coordinate discovery, adapters and storage.

The CLI is a thin shell over these functions, which keeps ingestion logic
testable without a terminal.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from toknt.adapters import registered_adapters
from toknt.adapters.base import ScanReport
from toknt.config import AppConfig
from toknt.discovery import DiscoveredSource
from toknt.storage import ingestion


@dataclass(frozen=True, slots=True)
class SourceOutcome:
    source: DiscoveredSource
    report: ScanReport | None = None
    skipped_reason: str | None = None


def run_scan(
    conn: sqlite3.Connection,
    *,
    full: bool = False,
    config: AppConfig | None = None,
) -> list[SourceOutcome]:
    """Discover enabled sources and ingest them.

    One adapter failure never aborts the others: errors are recorded as
    quality issues and reflected in the returned outcomes.
    """
    cfg = config if config is not None else AppConfig()
    mode = "full" if full else "incremental"
    outcomes: list[SourceOutcome] = []

    for adapter in registered_adapters():
        if adapter.agent not in cfg.enabled_harnesses:
            continue
        override = cfg.source_overrides.get(str(adapter.agent))
        source = adapter.discover(override=override)
        if not source.exists:
            outcomes.append(
                SourceOutcome(
                    source=source,
                    skipped_reason=source.note or f"source not found at {source.location}",
                )
            )
            continue

        ingestion.ensure_source(
            conn,
            source_id=source.source_id,
            agent=source.agent,
            kind=source.kind,
            location=source.location,
            status="scanning",
            note=source.note,
        )
        run_id = ingestion.start_scan_run(conn, source.source_id, mode)
        try:
            report = adapter.scan(conn, source, full=full)
        except Exception as exc:  # adapters translate errors, but never trust that
            message = f"{type(exc).__name__}: {exc}"[:300]
            ingestion.record_quality_issue(
                conn,
                agent=source.agent,
                source_id=source.source_id,
                code="scan_failed",
                message=message,
                severity="error",
            )
            report = ScanReport(
                source_id=source.source_id,
                mode=mode,
                status="error",
                errors=1,
                notes=(f"scan failed: {type(exc).__name__}",),
            )

        ingestion.finish_scan_run(
            conn,
            run_id,
            status=report.status,
            inserted=report.usage_inserted + report.tool_inserted,
            updated=report.usage_updated + report.tool_updated,
            skipped=report.skipped,
            errors=report.errors,
            notes="; ".join(report.notes) or None,
        )
        note_parts = [
            f"{report.usage_inserted + report.usage_updated} usage, "
            f"{report.tool_inserted + report.tool_updated} tool events"
        ]
        if report.reconciled:
            note_parts.append(f"{report.reconciled} reconciled")
        if report.skipped:
            note_parts.append(f"{report.skipped} skipped")
        note_parts.extend(report.notes)
        ingestion.ensure_source(
            conn,
            source_id=source.source_id,
            agent=source.agent,
            kind=source.kind,
            location=source.location,
            fingerprint=report.fingerprint,
            status=report.status,
            note="; ".join(note_parts)[:500],
        )
        outcomes.append(SourceOutcome(source=source, report=report))

    return outcomes
