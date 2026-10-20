"""Aggregate data for interactive views.

The TUI re-reads aggregates whenever the window, agent filter, metric or the
scanned corpus changes; everything is served straight from the SQLite store
via the shared analytics layer, so the TUI can never disagree with the
printed reports. Pure data in, plain dataclasses out — no painting happens
here.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass

from toknt.analytics import aggregation, model_names, tool_metrics
from toknt.config import AppConfig
from toknt.models.events import Harness
from toknt.models.usage import TokenScheme

__all__ = [
    "AgentToggles",
    "Analysis",
    "DailyTotals",
    "DEFAULT_WINDOW",
    "WINDOWS",
    "Window",
    "HarnessToggles",
    "HARNESS_NAMES",
]

HARNESS_NAMES = tuple(str(harness) for harness in Harness)

#: Historical names (the concept was renamed from agents to harnesses).
AGENT_NAMES = HARNESS_NAMES

_MISS = object()


@dataclass(frozen=True, slots=True)
class Window:
    label: str
    days: float | None  # None = everything; sub-day windows are "live"


WINDOWS: tuple[Window, ...] = (
    Window("3h", 3 / 24),
    Window("5h", 5 / 24),
    Window("1d", 1),
    Window("3d", 3),
    Window("7d", 7),
    Window("30d", 30),
    Window("90d", 90),
    Window("1y", 365),
    Window("all", None),
)

#: The window the explorer opens on (resolved by label, not by index, so
#: inserting a window never silently changes the default).
DEFAULT_WINDOW = next(index for index, window in enumerate(WINDOWS) if window.label == "90d")

#: Windows at most this many days long refresh themselves in the background
#: (the "live" views: hour-bucketed charts, periodic rescan).
LIVE_WINDOW_DAYS = 1.0

#: Charts for windows up to this many days are hour-bucketed; longer
#: windows switch to day buckets so the line stays readable.
HOUR_BUCKET_MAX_DAYS = 7.0


@dataclass(frozen=True, slots=True)
class HarnessToggles:
    """Which harnesses the views currently include (default: all known)."""

    active: frozenset[str] = frozenset(HARNESS_NAMES)

    def toggled(self, harness: str) -> HarnessToggles:
        if harness in self.active:
            return HarnessToggles(self.active - {harness})
        return HarnessToggles(self.active | {harness})

    def toggled_all_only(self, harness: str) -> HarnessToggles:
        """Focus exactly one harness (the 'h' focus cycle)."""
        return HarnessToggles(frozenset({harness}))


#: Historical name for :class:`HarnessToggles`.
AgentToggles = HarnessToggles


@dataclass(frozen=True, slots=True)
class DailyTotals:
    """Tokens per local day for the chosen window/agents."""

    days: tuple[int, ...]           # local-midnight bucket starts, ascending
    per_agent: dict[int, dict[str, int]]
    total: dict[int, int]


class Analysis:
    """Serves every view; memoized per state, invalidated on rescan.

    Interactive repaints repeat the same query repeatedly (arrow keys, help
    toggles) and building tens of thousands of per-event dataclasses costs
    ~200 ms on real corpora — long enough to feel laggy. Results are cached
    per exact state key and dropped by :meth:`invalidate` when a scan edits
    the corpus, so reports and TUI never disagree for stale data.
    """

    _CACHE_LIMIT = 16

    def __init__(self, conn: sqlite3.Connection, cfg: AppConfig) -> None:
        self.conn = conn
        self.cfg = cfg
        # one timezone object resolved once: day buckets and the heat grid
        # must agree exactly, or heatmap cells fall on the wrong keys.
        self.tz = aggregation.default_timezone()
        self._cache: dict[tuple, object] = {}

    def invalidate(self) -> None:
        """Forget memoized view data (call after any corpus-changing scan)."""
        self._cache.clear()

    def _memo(self, key: tuple, build) -> object:
        cached = self._cache.get(key, _MISS)
        if cached is _MISS:
            cached = build()
            self._cache[key] = cached
            while len(self._cache) > self._CACHE_LIMIT:
                self._cache.pop(next(iter(self._cache)))
        return cached

    # -- filter plumbing -----------------------------------------------------

    def usage_filter(self, since_ms: int, agents: AgentToggles) -> aggregation.UsageFilter:
        return aggregation.UsageFilter(
            since_ms=since_ms,
            agents=tuple(sorted(agents.active)),
            ignore_projects=self.cfg.ignore_projects,
        )

    def usage_rows(
        self, since_ms: int, agents: AgentToggles
    ) -> list[aggregation.UsageRow]:
        """The window's usage events, read once and shared.

        Building the per-event dataclasses costs ~120 ms on a real corpus,
        and the models view wants the same rows for the chart, the table and
        the rate estimate — one read, three answers.
        """
        key = ("usage_rows", since_ms, agents.active)
        return self._memo(key, lambda: aggregation.fetch_usage_rows(
            self.conn, self.usage_filter(since_ms, agents)
        ))

    # -- view data -----------------------------------------------------------

    def daily(self, since_ms: int, agents: AgentToggles) -> DailyTotals:
        key = ("daily", since_ms, agents.active)
        return self._memo(key, lambda: _build_daily(
            aggregation.fetch_usage_rows(self.conn, self.usage_filter(since_ms, agents)),
            self.tz,
        ))

    def model_series(
        self,
        since_ms: int,
        agents: AgentToggles,
        metric: str,
        bucket: str = "day",
        *,
        merge: bool = False,
    ) -> aggregation.Series:
        key = ("model_series", since_ms, agents.active, metric, bucket, merge)
        return self._memo(key, lambda: aggregation.usage_series(
            self.conn,
            self.usage_filter(since_ms, agents),
            dimension="model",
            metric=metric,
            bucket=bucket,
            tz=self.tz,
            top_n=8,
            canonical=model_names.canonical_model if merge else None,
            rows=self.usage_rows(since_ms, agents),
        ))

    def model_rate_series(
        self,
        since_ms: int,
        agents: AgentToggles,
        bucket: str = "day",
        *,
        merge: bool = False,
        lineup: Mapping[str, str] | None = None,
    ) -> aggregation.Series:
        """≈ tokens/s per model per bucket (see aggregation for the method).

        ``lineup`` is the token chart's display-name -> identity map: the
        rate chart then plots the same models in the same order, so the
        unit switch moves the lines, not the colours.
        """
        key = (
            "model_rate_series",
            since_ms,
            agents.active,
            bucket,
            merge,
            tuple(sorted(lineup.items())) if lineup else None,
        )
        return self._memo(key, lambda: aggregation.generation_rate_series(
            self.conn,
            self.usage_filter(since_ms, agents),
            bucket=bucket,
            tz=self.tz,
            top_n=8,
            canonical=model_names.canonical_model if merge else None,
            lineup=lineup,
            rows=self.usage_rows(since_ms, agents),
        ))

    def model_rows(
        self,
        since_ms: int,
        agents: AgentToggles,
        *,
        merge: bool = False,
        metric: str = "total",
    ) -> list[aggregation.BreakdownRow]:
        key = ("model_rows", since_ms, agents.active, merge, metric)
        return self._memo(key, lambda: list(aggregation.usage_breakdown(
            self.conn, self.usage_filter(since_ms, agents),
            canonical=model_names.canonical_model if merge else None,
            metric=metric,
            rows=self.usage_rows(since_ms, agents),
        )))

    def model_rate_totals(
        self, since_ms: int, agents: AgentToggles, *, merge: bool = False
    ) -> dict[str, float | None]:
        key = ("model_rate_totals", since_ms, agents.active, merge)
        return self._memo(key, lambda: aggregation.model_rate_totals(
            self.conn, self.usage_filter(since_ms, agents),
            canonical=model_names.canonical_model if merge else None,
            rows=self.usage_rows(since_ms, agents),
        ))

    def project_rows(
        self, since_ms: int, agents: AgentToggles
    ) -> list[aggregation.BreakdownRow]:
        key = ("project_rows", since_ms, agents.active)
        return self._memo(key, lambda: list(aggregation.usage_breakdown(
            self.conn, self.usage_filter(since_ms, agents), dimension="project"
        )))

    def cache_rates(
        self, since_ms: int, agents: AgentToggles, *, merge: bool = False
    ) -> list[aggregation.ModelCacheRate]:
        key = ("cache_rates", since_ms, agents.active, merge)
        return self._memo(key, lambda: aggregation.model_cache_rates(
            self.conn, self.usage_filter(since_ms, agents),
            canonical=model_names.canonical_model if merge else None,
        ))

    def tool_rows(self, since_ms: int, agents: AgentToggles) -> list[tool_metrics.ToolOutcomeRow]:
        key = ("tool_rows", since_ms, agents.active)
        return self._memo(key, lambda: list(tool_metrics.tool_outcomes(
            self.conn, self.usage_filter(since_ms, agents)
        )))

    def session_rows(
        self,
        since_ms: int,
        agents: AgentToggles,
        limit: int = 24,
        *,
        merge: bool = False,
        metric: str = "total",
    ) -> list[aggregation.SessionSummary]:
        key = ("sessions", since_ms, agents.active, limit, merge, metric)
        return self._memo(key, lambda: list(aggregation.session_ranking(
            self.conn, self.usage_filter(since_ms, agents), limit=limit,
            canonical=model_names.canonical_model if merge else None,
            metric=metric,
        )))

    def summary(self, since_ms: int, agents: AgentToggles) -> aggregation.OverallSummary:
        key = ("summary", since_ms, agents.active)
        return self._memo(key, lambda: aggregation.overall_summary(
            self.conn, self.usage_filter(since_ms, agents)
        ))


def _row_total(row: aggregation.UsageRow) -> int:
    scheme = row.scheme if isinstance(row.scheme, TokenScheme) else TokenScheme(str(row.scheme))
    try:
        total = aggregation.effective_total(row.tokens, scheme)
    except (ValueError, TypeError):
        return 0
    return int(total or 0)


def _build_daily(
    rows: list[aggregation.UsageRow],
    tz: object,
) -> DailyTotals:
    per_day: dict[int, dict[str, int]] = {}
    for row in rows:
        key = aggregation.bucket_start_ms(row.ts_ms, "day", tz)
        entry = per_day.setdefault(key, {})
        entry[row.agent] = entry.get(row.agent, 0) + _row_total(row)
    return DailyTotals(
        days=tuple(sorted(per_day)),
        per_agent=per_day,
        total={key: sum(entry.values()) for key, entry in per_day.items()},
    )
