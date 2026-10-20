"""Tool-call outcome analytics.

Success rate is defined as::

    confirmed successful calls / calls with known final outcomes

where "known final outcome" excludes pending calls and calls with unknown
outcomes. The observable fraction (known / all calls) is reported alongside,
so an adapter that fails to parse failures can never make every tool look
perfect — a tool with 4 calls of which 0 are observable shows an unknown
rate, not 100%.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import tzinfo

from toknt.analytics.aggregation import (
    Series,
    UsageFilter,
    _series_start,
    _tool_where,
    bucket_start_ms,
    bucket_starts,
    default_timezone,
)

_KNOWN_OUTCOMES: tuple[str, ...] = ("success", "failure", "denied", "cancelled")


@dataclass(frozen=True, slots=True)
class ToolOutcomeRow:
    tool: str
    category: str
    calls: int
    success: int
    failure: int
    denied: int
    cancelled: int
    pending: int
    unknown: int

    @property
    def known_outcomes(self) -> int:
        return self.success + self.failure + self.denied + self.cancelled

    @property
    def success_rate(self) -> float | None:
        """Confirmed successes over known final outcomes (``None`` if none known)."""
        if not self.known_outcomes:
            return None
        return self.success / self.known_outcomes

    @property
    def observability(self) -> float:
        """Fraction of calls with a known final outcome."""
        if not self.calls:
            return 0.0
        return self.known_outcomes / self.calls


def _fetch(conn: sqlite3.Connection, flt: UsageFilter, columns: str) -> list[sqlite3.Row]:
    where, params = _tool_where(flt)
    return conn.execute(f"SELECT {columns} FROM tool_events {where}", params).fetchall()


def tool_outcomes(conn: sqlite3.Connection, flt: UsageFilter) -> list[ToolOutcomeRow]:
    """Outcome counts per tool, ordered by call volume."""
    grouped: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "calls": 0,
            "success": 0,
            "failure": 0,
            "denied": 0,
            "cancelled": 0,
            "pending": 0,
            "unknown": 0,
        }
    )
    categories: dict[str, str] = {}
    for row in _fetch(conn, flt, "tool_name, tool_category, outcome"):
        entry = grouped[row["tool_name"]]
        entry["calls"] += 1
        outcome = row["outcome"] if row["outcome"] in (*_KNOWN_OUTCOMES, "pending") else "unknown"
        entry[outcome] += 1
        categories[row["tool_name"]] = row["tool_category"] or "other"

    result = [
        ToolOutcomeRow(
            tool=tool,
            category=categories.get(tool, "other"),
            calls=data["calls"],
            success=data["success"],
            failure=data["failure"],
            denied=data["denied"],
            cancelled=data["cancelled"],
            pending=data["pending"],
            unknown=data["unknown"],
        )
        for tool, data in grouped.items()
    ]
    result.sort(key=lambda item: (-item.calls, item.tool))
    return result


@dataclass(frozen=True, slots=True)
class DurationStats:
    count: int
    mean_ms: float
    p50_ms: int
    p90_ms: int
    max_ms: int


def tool_durations(conn: sqlite3.Connection, flt: UsageFilter) -> dict[str, DurationStats]:
    """Duration distribution per tool, only for calls with reliable timing."""
    durations: dict[str, list[int]] = defaultdict(list)
    for row in _fetch(conn, flt, "tool_name, duration_ms"):
        if row["duration_ms"] is not None:
            durations[row["tool_name"]].append(row["duration_ms"])

    result: dict[str, DurationStats] = {}
    for tool, values in durations.items():
        values = sorted(values)
        count = len(values)
        result[tool] = DurationStats(
            count=count,
            mean_ms=sum(values) / count,
            p50_ms=values[int(count * 0.5)],
            p90_ms=values[min(count - 1, int(count * 0.9))],
            max_ms=values[-1],
        )
    return result


@dataclass(slots=True)
class _Bucket:
    """Mutable per-bucket accumulator for :func:`tool_success_series`."""

    success: int = 0
    known: int = 0
    calls: int = 0


def tool_success_series(
    conn: sqlite3.Connection,
    flt: UsageFilter,
    *,
    bucket: str = "day",
    tz: tzinfo | None = None,
) -> Series:
    """Success rate (percent) per bucket and tool; ``None`` when unobservable."""
    tz = tz or default_timezone()
    rows = _fetch(conn, flt, "tool_name, outcome, ts_ms")
    if not rows:
        return Series((), {}, "tool", bucket)

    start = _series_start(flt.since_ms, min(row["ts_ms"] for row in rows))
    end = (flt.until_ms - 1) if flt.until_ms is not None else max(row["ts_ms"] for row in rows)
    buckets = bucket_starts(start, end, bucket, tz)

    stats: dict[str, dict[int, _Bucket]] = defaultdict(lambda: defaultdict(_Bucket))
    for row in rows:
        entry = stats[row["tool_name"]][bucket_start_ms(row["ts_ms"], bucket, tz)]
        entry.calls += 1
        if row["outcome"] in _KNOWN_OUTCOMES:
            entry.known += 1
            if row["outcome"] == "success":
                entry.success += 1

    series: dict[str, tuple[int | None, ...]] = {}
    ordered = sorted(stats, key=lambda name: -sum(entry.calls for entry in stats[name].values()))
    for tool in ordered:
        values: list[int | None] = []
        for bucket_ms in buckets:
            entry = stats[tool].get(bucket_ms)
            if entry is None or entry.known == 0:
                values.append(None)
            else:
                values.append(round(100 * entry.success / entry.known))
        series[tool] = tuple(values)
    return Series(tuple(buckets), series, "tool", bucket)
