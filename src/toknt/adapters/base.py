"""Adapter boundary: native agent formats never leak past this package.

Adapters translate a source's native storage into canonical records
(:mod:`toknt.models.events`). They must not aggregate, render or interpret
data-quality semantics beyond setting the documented fields — everything
downstream works on the canonical model.
"""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import ClassVar

from toknt.discovery import DiscoveredSource
from toknt.models.events import Agent, ToolCategory

#: Tool-name → category mapping shared by all adapters. Categories are
#: presentation-friendly groupings, never used for accounting.
_TOOL_CATEGORIES: dict[str, ToolCategory] = {
    "bash": ToolCategory.SHELL,
    "shell": ToolCategory.SHELL,
    "shell_command": ToolCategory.SHELL,
    "exec": ToolCategory.SHELL,
    "exec_command": ToolCategory.SHELL,
    "js": ToolCategory.SHELL,
    "write_stdin": ToolCategory.SHELL,
    "local_shell": ToolCategory.SHELL,
    "run": ToolCategory.SHELL,
    "terminal": ToolCategory.SHELL,
    "read": ToolCategory.FILE_READ,
    "read_file": ToolCategory.FILE_READ,
    "view": ToolCategory.FILE_READ,
    "view_image": ToolCategory.FILE_READ,
    "cat": ToolCategory.FILE_READ,
    "edit": ToolCategory.FILE_WRITE,
    "write": ToolCategory.FILE_WRITE,
    "write_file": ToolCategory.FILE_WRITE,
    "create_file": ToolCategory.FILE_WRITE,
    "apply_patch": ToolCategory.FILE_WRITE,
    "patch": ToolCategory.FILE_WRITE,
    "multiedit": ToolCategory.FILE_WRITE,
    "notebook_edit": ToolCategory.FILE_WRITE,
    "grep": ToolCategory.SEARCH,
    "glob": ToolCategory.SEARCH,
    "search": ToolCategory.SEARCH,
    "find": ToolCategory.SEARCH,
    "list": ToolCategory.SEARCH,
    "ls": ToolCategory.SEARCH,
    "list_dir": ToolCategory.SEARCH,
    "codebase_search": ToolCategory.SEARCH,
    "webfetch": ToolCategory.WEB,
    "web_search": ToolCategory.WEB,
    "websearch": ToolCategory.WEB,
    "fetch": ToolCategory.WEB,
    "browser": ToolCategory.WEB,
    "task": ToolCategory.TASK,
    "agent": ToolCategory.TASK,
    "spawn_agent": ToolCategory.TASK,
    "followup_task": ToolCategory.TASK,
    "wait_agent": ToolCategory.TASK,
    "subagent": ToolCategory.TASK,
    "delegate": ToolCategory.TASK,
}

_TOOL_PREFIXES: tuple[tuple[str, ToolCategory], ...] = (
    ("shell", ToolCategory.SHELL),
    ("exec", ToolCategory.SHELL),
    ("bash", ToolCategory.SHELL),
    ("read", ToolCategory.FILE_READ),
    ("cat", ToolCategory.FILE_READ),
    ("write", ToolCategory.FILE_WRITE),
    ("edit", ToolCategory.FILE_WRITE),
    ("patch", ToolCategory.FILE_WRITE),
    ("grep", ToolCategory.SEARCH),
    ("glob", ToolCategory.SEARCH),
    ("search", ToolCategory.SEARCH),
    ("list", ToolCategory.SEARCH),
    ("web", ToolCategory.WEB),
    ("task", ToolCategory.TASK),
    ("agent", ToolCategory.TASK),
)


def categorize_tool(name: str | None) -> ToolCategory:
    """Best-effort category for a tool name; unknown tools stay ``other``."""
    key = (name or "").strip().lower()
    if key in _TOOL_CATEGORIES:
        return _TOOL_CATEGORIES[key]
    for prefix, category in _TOOL_PREFIXES:
        if key.startswith(prefix):
            return category
    return ToolCategory.OTHER


def as_dict(value: object) -> dict:
    """Return ``value`` when it is a dict, otherwise an empty dict."""
    return value if isinstance(value, dict) else {}


def opt_str(value: object) -> str | None:
    """Return non-empty strings, otherwise ``None``."""
    return value if isinstance(value, str) and value else None


def token_int(value: object) -> int | None:
    """Coerce a token count; booleans and negatives are not counts."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = int(value)
    return number if number >= 0 else None


def parse_iso8601_ms(value: object) -> int | None:
    """Parse an ISO-8601 timestamp (optional ``Z``) into epoch milliseconds."""
    if not isinstance(value, str) or not value:
        return None
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    result = int(stamp.timestamp() * 1000)
    return result if result > 0 else None


def normalize_epoch_ms(value: object) -> int | None:
    """Coerce a source timestamp into UTC epoch milliseconds.

    Tolerates seconds, milliseconds, microseconds and nanoseconds so that a
    future format change degrades to a quality note instead of wrong dates.
    Values that cannot plausibly be timestamps return ``None``.
    """
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    number = int(value)
    if number <= 0:
        return None
    if number >= 100_000_000_000_000:  # ≥ 1e14: microseconds (1973+)
        return number // 1000
    if number >= 100_000_000_000:  # ≥ 1e11: milliseconds (1973+)
        return number
    if number >= 1_000_000_000:  # ≥ 1e9: seconds (2001+)
        return number * 1000
    return None


@dataclass(frozen=True, slots=True)
class ScanReport:
    """Outcome of one adapter scan over one source."""

    source_id: str
    mode: str  # "incremental" | "full"
    status: str = "ok"  # "ok" | "partial" | "error"
    rows_seen: int = 0
    usage_inserted: int = 0
    usage_updated: int = 0
    tool_inserted: int = 0
    tool_updated: int = 0
    sessions_written: int = 0
    skipped: int = 0
    errors: int = 0
    reconciled: int = 0
    fingerprint: str | None = None
    notes: tuple[str, ...] = ()


class Adapter(ABC):
    """Base class for source adapters.

    ``discover`` and ``scan`` must be side-effect free with respect to the
    source (read-only) and idempotent with respect to the canonical database
    (stable event ids, upsert semantics).
    """

    source_id: ClassVar[str]
    agent: ClassVar[Agent]
    kind: ClassVar[str]

    @abstractmethod
    def discover(
        self,
        *,
        env: dict[str, str] | None = None,
        home: object | None = None,
        override: str | None = None,
    ) -> DiscoveredSource:
        """Locate this adapter's source on disk."""

    @abstractmethod
    def scan(
        self,
        conn: sqlite3.Connection,
        source: DiscoveredSource,
        *,
        full: bool = False,
    ) -> ScanReport:
        """Ingest new/changed data from ``source`` into the canonical database."""
