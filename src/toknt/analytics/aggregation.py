"""Time-series and breakdown queries over canonical events.

Every function takes a connection plus a :class:`UsageFilter` and returns
plain dataclasses. Nothing here imports Rich or plotext, so the same queries
can back a future full-screen TUI.

Token totals follow the per-scheme semantics defined in
:mod:`toknt.models.usage`: rows are totalled individually with
``effective_total`` before summing, so partially reported events can never
silently deflate an aggregate.
"""

from __future__ import annotations

import dataclasses
import re
import sqlite3
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo

from toknt.analytics.model_names import CanonicalName, group_label
from toknt.models.usage import TokenCounts, TokenScheme, effective_total

BUCKETS: tuple[str, ...] = ("hour", "day", "week")
METRICS: tuple[str, ...] = (
    "total",
    "input",
    "output",
    "reasoning",
    "cache_read",
    "cache_write",
    "cached",
    "requests",
)

_BUCKET_STEPS: dict[str, timedelta] = {
    "hour": timedelta(hours=1),
    "day": timedelta(days=1),
    "week": timedelta(weeks=1),
}

METRIC_LABELS: dict[str, str] = {
    "total": "total tokens",
    "input": "input tokens",
    "output": "output tokens",
    "reasoning": "reasoning tokens",
    "cache_read": "cache-read tokens",
    "cache_write": "cache-write tokens",
    "cached": "cached tokens (read + write)",
    "requests": "requests",
}


# ------------------------------------------------------------------ timezone


def default_timezone() -> tzinfo:
    """Best-effort local timezone for bucket boundaries.

    Prefers ``$TZ``, then the zone name behind ``/etc/localtime``, and falls
    back to the current UTC offset. Bucketing needs *a* zone; the fallback
    just cannot model DST transitions for past dates.
    """
    import os

    tz_env = os.environ.get("TZ")
    if tz_env:
        try:
            return ZoneInfo(tz_env)
        except Exception:
            pass
    try:
        target = Path("/etc/localtime").resolve()
        parts = target.parts
        if "zoneinfo" in parts:
            index = parts.index("zoneinfo")
            name = "/".join(parts[index + 1 :])
            if name:
                return ZoneInfo(name)
    except Exception:
        pass
    return datetime.now().astimezone().tzinfo or ZoneInfo("UTC")


_RELATIVE_RE = re.compile(r"^(\d+)([hdw])$")


def parse_since(value: str, *, now_ms: int | None = None, tz: tzinfo | None = None) -> int:
    """Parse a ``--since`` value into epoch milliseconds.

    Accepts relative offsets (``12h``, ``30d``, ``2w``) and ISO dates
    (``2026-10-01``, interpreted as local midnight).
    """
    tz = tz or default_timezone()
    if value == "today":
        now = datetime.now(tz)
        return int(now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    match = _RELATIVE_RE.match(value)
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        now = now_ms if now_ms is not None else int(datetime.now(tz).timestamp() * 1000)
        delta = {"h": amount * 3_600_000, "d": amount * 86_400_000, "w": amount * 604_800_000}[unit]
        return now - delta
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"invalid --since value {value!r}; use 12h / 30d / 2w or an ISO date (2026-10-01)"
        ) from exc
    midnight = datetime(parsed.year, parsed.month, parsed.day, tzinfo=tz)
    return int(midnight.timestamp() * 1000)


def bucket_start_ms(ts_ms: int, bucket: str, tz: tzinfo) -> int:
    """Epoch milliseconds of the bucket containing ``ts_ms`` (local time)."""
    if bucket not in _BUCKET_STEPS:
        raise ValueError(f"unknown bucket {bucket!r}; expected one of {BUCKETS}")
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=tz)
    if bucket == "hour":
        dt = dt.replace(minute=0, second=0, microsecond=0)
    elif bucket == "day":
        dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    else:  # week starts Monday, local time
        dt = (dt - timedelta(days=dt.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    return int(dt.timestamp() * 1000)


def bucket_starts(start_ms: int, end_ms: int, bucket: str, tz: tzinfo) -> list[int]:
    """Contiguous bucket starts covering ``[start_ms, end_ms]`` (inclusive end)."""
    step = _BUCKET_STEPS[bucket]
    cursor = datetime.fromtimestamp(bucket_start_ms(start_ms, bucket, tz) / 1000, tz=tz)
    end = datetime.fromtimestamp(end_ms / 1000, tz=tz)
    buckets: list[int] = []
    while cursor <= end:
        buckets.append(int(cursor.timestamp() * 1000))
        cursor = cursor + step
    return buckets


# ------------------------------------------------------------------- filter


def _like_prefix(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return escaped


@dataclass(frozen=True, slots=True)
class UsageFilter:
    since_ms: int | None = None
    until_ms: int | None = None  # exclusive
    agents: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    projects: tuple[str, ...] = ()
    ignore_projects: tuple[str, ...] = ()


def _usage_where(flt: UsageFilter) -> tuple[str, list[object]]:
    clauses: list[str] = []
    params: list[object] = []
    if flt.since_ms is not None:
        clauses.append("ts_ms >= ?")
        params.append(flt.since_ms)
    if flt.until_ms is not None:
        clauses.append("ts_ms < ?")
        params.append(flt.until_ms)
    if flt.agents:
        clauses.append(f"agent IN ({', '.join('?' * len(flt.agents))})")
        params.extend(flt.agents)
    if flt.models:
        clauses.append(
            f"(model IN ({', '.join('?' * len(flt.models))}) OR "
            f"COALESCE(model, '(unknown)') IN ({', '.join('?' * len(flt.models))}))"
        )
        params.extend(flt.models)
        params.extend(flt.models)
    if flt.projects:
        sub = " OR ".join(
            "(project = ? OR project LIKE ? ESCAPE '\\')" for _ in flt.projects
        )
        clauses.append(f"({sub})")
        for project in flt.projects:
            params.extend([project, _like_prefix(project) + "/%"])
    for ignored in flt.ignore_projects:
        clauses.append("(project IS NULL OR (project != ? AND project NOT LIKE ? ESCAPE '\\'))")
        params.extend([ignored, _like_prefix(ignored) + "/%"])
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


@dataclass(frozen=True, slots=True)
class UsageRow:
    ts_ms: int
    agent: str
    model: str | None
    project: str | None
    session_id: str | None
    scheme: TokenScheme
    tokens: TokenCounts
    cost_reported: float | None


_SCHEME_VALUES = frozenset(item.value for item in TokenScheme)


def fetch_usage_rows(conn: sqlite3.Connection, flt: UsageFilter) -> list[UsageRow]:
    where, params = _usage_where(flt)
    sql = (
        "SELECT ts_ms, agent, model, project, session_id, token_scheme, "
        "input_tokens, output_tokens, reasoning_tokens, cache_read_tokens, "
        "cache_write_tokens, cost_reported FROM usage_events "
        f"{where} ORDER BY ts_ms"
    )
    rows = []
    for row in conn.execute(sql, params):
        scheme = (
            TokenScheme(row["token_scheme"])
            if row["token_scheme"] in _SCHEME_VALUES
            else TokenScheme.UNKNOWN
        )
        rows.append(
            UsageRow(
                ts_ms=row["ts_ms"],
                agent=row["agent"],
                model=row["model"],
                project=row["project"],
                session_id=row["session_id"],
                scheme=scheme,
                tokens=TokenCounts(
                    input=row["input_tokens"],
                    output=row["output_tokens"],
                    reasoning=row["reasoning_tokens"],
                    cache_read=row["cache_read_tokens"],
                    cache_write=row["cache_write_tokens"],
                ),
                cost_reported=row["cost_reported"],
            )
        )
    return rows


# ------------------------------------------------------------------ metrics


def metric_value(row: UsageRow, metric: str) -> int | None:
    """Value of one metric for one event, or ``None`` when undefined."""
    if metric == "requests":
        return 1
    if metric == "total":
        return effective_total(row.tokens, row.scheme)
    if metric == "cached":
        if row.tokens.cache_read is None or row.tokens.cache_write is None:
            return None
        return row.tokens.cache_read + row.tokens.cache_write
    return getattr(row.tokens, metric)


def metric_total(rows: Sequence[UsageRow], metric: str) -> int | None:
    """The aggregate of one metric over a group of events.

    ``total`` keeps the strict per-event semantics: an event whose total
    cannot be computed leaves the aggregate undefined rather than
    silently deflating it. The other metrics are summed the same way —
    a missing component makes the aggregate undefined, not zero.
    """
    if metric == "requests":
        return len(rows)
    total_acc = 0
    for row in rows:
        value = metric_value(row, metric)
        if value is None:
            return None
        total_acc += value
    return total_acc if rows else None


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


def _dimension_key(row: UsageRow, dimension: str) -> str:
    value = getattr(row, dimension)
    return value if value else "(unknown)"


def _display_key(key: str, spellings: Mapping[str, Sequence[str]]) -> str:
    """The label a merged group shows: its most common spelling."""
    if key not in spellings:
        return key
    return group_label(spellings[key])


# ------------------------------------------------------------------- series


@dataclass(frozen=True, slots=True)
class Series:
    """One value per (key, bucket); ``None`` marks a bucket without data."""

    buckets: tuple[int, ...]
    series: Mapping[str, tuple[int | None, ...]]
    dimension: str
    bucket: str = "day"

    @property
    def is_empty(self) -> bool:
        return not self.buckets or not any(
            value is not None for values in self.series.values() for value in values
        )


def _series_start(since_ms: int | None, first_ts_ms: int) -> int:
    """Window start for bucketing a series.

    The ``--since all`` sentinel (epoch 0) describes an unbounded filter, not
    a real window: anchoring buckets there would draw ~57 empty years of
    chart before the data. Clamp to the earliest observation instead.
    Explicit user windows keep their leading empty buckets.
    """
    if since_ms is None or since_ms <= 0:
        return first_ts_ms
    return since_ms


def usage_series(
    conn: sqlite3.Connection,
    flt: UsageFilter,
    *,
    dimension: str = "model",
    metric: str = "total",
    bucket: str = "day",
    tz: tzinfo | None = None,
    top_n: int | None = 8,
    other_label: str = "(other)",
    canonical: Callable[[str], CanonicalName] | None = None,
) -> Series:
    """Per-bucket series of ``metric``, split by ``dimension``.

    ``canonical`` maps a raw dimension value to its canonical identity, so
    values that name the same thing (a model served by two providers) share
    one series.
    """
    tz = tz or default_timezone()
    rows = fetch_usage_rows(conn, flt)
    if not rows:
        return Series((), {}, dimension, bucket)

    start = _series_start(flt.since_ms, min(row.ts_ms for row in rows))
    end = (flt.until_ms - 1) if flt.until_ms is not None else max(row.ts_ms for row in rows)
    buckets = bucket_starts(start, end, bucket, tz)

    per_key: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    totals: dict[str, int] = defaultdict(int)
    spellings: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        value = metric_value(row, metric)
        if value is None:
            continue
        raw = _dimension_key(row, dimension)
        if canonical is None:
            key, label = raw, raw
        else:
            name = canonical(raw)
            key, label = name.identity, name.label
        per_key[key][bucket_start_ms(row.ts_ms, bucket, tz)] += value
        totals[key] += value
        spellings[key].append(label)
    keys = sorted(totals, key=lambda key: (-totals[key], key))
    if top_n is not None and len(keys) > top_n:
        top = keys[:top_n]
        merged: dict[int, int] = defaultdict(int)
        for key in keys[top_n:]:
            for bucket_ms, value in per_key[key].items():
                merged[bucket_ms] += value
        series_keys = top + [other_label]
        per_key[other_label] = merged
    else:
        series_keys = keys

    series = {
        _display_key(key, spellings): tuple(
            per_key[key].get(bucket_ms) for bucket_ms in buckets
        )
        for key in series_keys
    }
    return Series(tuple(buckets), series, dimension, bucket)


def cumulative(series: Series) -> Series:
    """Running totals over a series; buckets before the first value stay None."""
    result: dict[str, tuple[int | None, ...]] = {}
    for key, values in series.series.items():
        running: int | None = None
        out: list[int | None] = []
        for value in values:
            if value is not None:
                running = (running or 0) + value
            out.append(running)
        result[key] = tuple(out)
    return Series(series.buckets, result, series.dimension, series.bucket)


def activity_series(
    conn: sqlite3.Connection,
    flt: UsageFilter,
    *,
    bucket: str = "day",
    tz: tzinfo | None = None,
) -> Series:
    """Requests and active sessions per bucket."""
    tz = tz or default_timezone()
    rows = fetch_usage_rows(conn, flt)
    if not rows:
        return Series((), {}, "activity", bucket)
    start = _series_start(flt.since_ms, min(row.ts_ms for row in rows))
    end = (flt.until_ms - 1) if flt.until_ms is not None else max(row.ts_ms for row in rows)
    buckets = bucket_starts(start, end, bucket, tz)

    requests: dict[int, int] = defaultdict(int)
    sessions: dict[int, set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        bucket_ms = bucket_start_ms(row.ts_ms, bucket, tz)
        requests[bucket_ms] += 1
        if row.session_id:
            sessions[bucket_ms].add((row.agent, row.session_id))
    return Series(
        tuple(buckets),
        {
            "requests": tuple(requests.get(bucket_ms, 0) for bucket_ms in buckets),
            "sessions": tuple(len(sessions.get(bucket_ms, ())) for bucket_ms in buckets),
        },
        "activity",
        bucket,
    )


# --------------------------------------------------------------- breakdown


@dataclass(frozen=True, slots=True)
class TokenSummary:
    input: int | None = None
    output: int | None = None
    reasoning: int | None = None
    cache_read: int | None = None
    cache_write: int | None = None
    total: int | None = None
    events: int = 0
    sessions: int = 0
    events_without_total: int = 0
    cost_reported: float | None = None

    @property
    def coverage(self) -> float:
        """Fraction of events whose total could be computed."""
        if not self.events:
            return 0.0
        return (self.events - self.events_without_total) / self.events


@dataclass(frozen=True, slots=True)
class BreakdownRow:
    key: str
    summary: TokenSummary
    schemes: tuple[str, ...]


def summarize(rows: Sequence[UsageRow]) -> TokenSummary:
    components: dict[str, int | None] = dict.fromkeys(
        ("input", "output", "reasoning", "cache_read", "cache_write")
    )
    total_acc = 0
    without_total = 0
    cost: float | None = None
    sessions: set[tuple[str, str]] = set()
    for row in rows:
        for name in components:
            value = getattr(row.tokens, name)
            if value is not None:
                components[name] = (components[name] or 0) + value
        effect = effective_total(row.tokens, row.scheme)
        if effect is None:
            without_total += 1
        else:
            total_acc += effect
        if row.cost_reported is not None:
            cost = (cost or 0.0) + row.cost_reported
        if row.session_id:
            sessions.add((row.agent, row.session_id))
    total = total_acc if rows and without_total == 0 else None
    return TokenSummary(
        **components,
        total=total,
        events=len(rows),
        sessions=len(sessions),
        events_without_total=without_total,
        cost_reported=cost,
    )


def usage_breakdown(
    conn: sqlite3.Connection,
    flt: UsageFilter,
    *,
    dimension: str = "model",
    canonical: Callable[[str], CanonicalName] | None = None,
    metric: str = "total",
) -> list[BreakdownRow]:
    """One row per distinct value of ``dimension``, ordered by total usage.

    ``canonical`` merges values that name the same thing (a model served by
    two providers) into one row; the merged row is summarized as a whole, so
    totals, event counts, sessions and reported cost stay consistent.
    ``metric`` decides what the row's total measures — the explorer follows
    the selected metric, so a breakdown of input tokens ranks and reports
    input tokens.
    """
    rows = fetch_usage_rows(conn, flt)
    grouped: dict[str, list[UsageRow]] = defaultdict(list)
    spellings: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        raw = _dimension_key(row, dimension)
        if canonical is None:
            key, label = raw, raw
        else:
            name = canonical(raw)
            key, label = name.identity, name.label
        grouped[key].append(row)
        spellings[key].append(label)

    result = []
    for key, group in grouped.items():
        schemes = tuple(sorted({str(row.scheme) for row in group}))
        summary = summarize(group)
        if metric != "total":
            summary = dataclasses.replace(summary, total=metric_total(group, metric))
        result.append(
            BreakdownRow(
                key=_display_key(key, spellings),
                summary=summary,
                schemes=schemes,
            )
        )
    result.sort(
        key=lambda item: (
            -(item.summary.total if item.summary.total is not None else -1),
            -item.summary.events,
            item.key,
        )
    )
    return result


@dataclass(frozen=True, slots=True)
class OverallSummary:
    summary: TokenSummary
    schemes: tuple[str, ...]
    agents: tuple[str, ...]
    first_ms: int | None
    last_ms: int | None


def overall_summary(conn: sqlite3.Connection, flt: UsageFilter) -> OverallSummary:
    rows = fetch_usage_rows(conn, flt)
    return OverallSummary(
        summary=summarize(rows),
        schemes=tuple(sorted({str(row.scheme) for row in rows})),
        agents=tuple(sorted({row.agent for row in rows})),
        first_ms=min((row.ts_ms for row in rows), default=None),
        last_ms=max((row.ts_ms for row in rows), default=None),
    )


# ---------------------------------------------------------------- sessions


@dataclass(frozen=True, slots=True)
class SessionSummary:
    session_id: str
    agent: str
    project: str | None
    title: str | None
    models: tuple[str, ...]
    summary: TokenSummary
    first_ms: int
    last_ms: int
    duration_ms: int | None
    tool_calls: int
    outlier: bool


def _session_models(
    group: Sequence[UsageRow],
    canonical: Callable[[str], CanonicalName] | None,
) -> tuple[str, ...]:
    """The distinct models a session used, optionally merged by identity."""
    names = [row.model for row in group if row.model]
    if canonical is None:
        return tuple(sorted(set(names)))
    spellings: dict[str, list[str]] = defaultdict(list)
    for name in names:
        canon = canonical(name)
        spellings[canon.identity].append(canon.label)
    return tuple(sorted({_display_key(key, spellings) for key in spellings}))


def session_ranking(
    conn: sqlite3.Connection,
    flt: UsageFilter,
    *,
    limit: int = 20,
    canonical: Callable[[str], CanonicalName] | None = None,
    metric: str = "total",
) -> list[SessionSummary]:
    """Sessions ranked by total usage, with outlier flags for spikes.

    ``canonical`` also collapses the model names listed for a session, so a
    model served by two providers is named once. ``metric`` decides what
    the ranking measures.
    """
    rows = [row for row in fetch_usage_rows(conn, flt) if row.session_id]
    grouped: dict[tuple[str, str], list[UsageRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.agent, row.session_id or "")].append(row)

    tool_counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in _tool_rows(conn, flt):
        if row["session_id"]:
            tool_counts[(row["agent"], row["session_id"])] += 1

    session_meta: dict[tuple[str, str], dict] = {}
    for row in conn.execute("SELECT * FROM sessions"):
        session_meta[(row["agent"], row["session_id"])] = dict(row)

    summaries: list[SessionSummary] = []
    for (agent, session_id), group in grouped.items():
        summary = summarize(group)
        if metric != "total":
            summary = dataclasses.replace(summary, total=metric_total(group, metric))
        meta = session_meta.get((agent, session_id), {})
        timestamps = [row.ts_ms for row in group]
        duration = None
        created, updated = meta.get("time_created_ms"), meta.get("time_updated_ms")
        if isinstance(created, int) and isinstance(updated, int) and updated >= created:
            duration = updated - created
        project = next((row.project for row in group if row.project), None)
        models = _session_models(group, canonical)
        summaries.append(
            SessionSummary(
                session_id=session_id,
                agent=agent,
                project=meta.get("project") or project,
                title=meta.get("title"),
                models=models,
                summary=summary,
                first_ms=meta.get("time_created_ms") or min(timestamps),
                last_ms=meta.get("time_updated_ms") or max(timestamps),
                duration_ms=duration,
                tool_calls=tool_counts.get((agent, session_id), 0),
                outlier=False,
            )
        )

    totals = [item.summary.total for item in summaries if item.summary.total is not None]
    if len(totals) >= 10:
        mean = sum(totals) / len(totals)
        variance = sum((value - mean) ** 2 for value in totals) / len(totals)
        threshold = mean + 2 * (variance**0.5)
        summaries = [
            dataclasses.replace(item, outlier=True)
            if item.summary.total is not None and item.summary.total > threshold
            else item
            for item in summaries
        ]

    summaries.sort(
        key=lambda item: (
            -(item.summary.total if item.summary.total is not None else -1),
            -item.summary.events,
        )
    )
    return summaries[:limit]


@dataclass(frozen=True, slots=True)
class ModelProvenance:
    """Who served a model: the providers and harnesses behind one identity."""

    providers: tuple[str, ...]
    agents: tuple[str, ...]


def model_provenance(
    conn: sqlite3.Connection,
    flt: UsageFilter,
    *,
    canonical: Callable[[str], CanonicalName] | None = None,
) -> dict[str, ModelProvenance]:
    """Providers and harnesses per model identity, over the filtered period.

    With ``canonical`` the spellings of one model are reported together, so
    a model served by three providers shows all three.
    """
    where, params = _usage_where(flt)
    rows = conn.execute(
        f"""
        SELECT COALESCE(model, '(unknown)') AS model,
               COALESCE(provider, '') AS provider,
               agent
        FROM usage_events {where}
        """,
        params,
    ).fetchall()
    providers: dict[str, set[str]] = defaultdict(set)
    agents: dict[str, set[str]] = defaultdict(set)
    spellings: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        raw = row["model"]
        if canonical is None:
            key, label = raw, raw
        else:
            name = canonical(raw)
            key, label = name.identity, name.label
        providers[key].add(row["provider"])
        agents[key].add(row["agent"])
        spellings[key].append(label)
    return {
        _display_key(key, spellings): ModelProvenance(
            providers=tuple(sorted(item for item in providers[key] if item)),
            agents=tuple(sorted(agents[key])),
        )
        for key in providers
    }


def model_inventory(conn: sqlite3.Connection) -> list[dict]:
    """All models seen in the database with coverage information."""
    rows = conn.execute(
        """
        SELECT COALESCE(model, '(unknown)') AS model,
               COALESCE(provider, '') AS provider,
               COUNT(DISTINCT agent) AS agents,
               COUNT(*) AS events,
               MIN(ts_ms) AS first_ms,
               MAX(ts_ms) AS last_ms
        FROM usage_events
        GROUP BY COALESCE(model, '(unknown)'), COALESCE(provider, '')
        ORDER BY events DESC, model
        """
    ).fetchall()
    return [dict(row) for row in rows]


@dataclass(frozen=True, slots=True)
class ModelCacheRate:
    """Cache-hit economics for one model over the selected period.

    ``rate`` is the fraction of prompt tokens served from cache, computed
    strictly per token scheme: anthropic-style stores report cache reads
    separately from input (rate = cache_read / (input + cache_read));
    openai-style stores report cached tokens as a subset of input
    (rate = cache_read / input). Models without a known scheme or without
    cache reporting get ``rate = None`` and never show a made-up number.
    """

    model: str
    cache_read: int
    prompt: int
    events: int

    @property
    def rate(self) -> float | None:
        if self.prompt <= 0:
            return None
        return min(self.cache_read / self.prompt, 1.0)


def model_cache_rates(
    conn: sqlite3.Connection,
    flt: UsageFilter,
    *,
    canonical: Callable[[str], CanonicalName] | None = None,
) -> list[ModelCacheRate]:
    """Per-model cache-read share, merged across the schemes a model ran on.

    ``canonical`` merges spellings of the same model before the rates are
    computed, so a model served by two providers reports one economy.
    """
    where, params = _usage_where(flt)
    rows = conn.execute(
        f"""
        SELECT COALESCE(model, '(unknown)') AS model, token_scheme,
               SUM(input_tokens) AS input_sum,
               SUM(cache_read_tokens) AS cache_read_sum,
               SUM(cache_write_tokens) AS cache_write_sum,
               COUNT(*) AS events
        FROM usage_events {where}
        GROUP BY COALESCE(model, '(unknown)'), token_scheme
        """,
        params,
    ).fetchall()
    merged: dict[str, list[sqlite3.Row]] = {}
    spellings: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        raw = row["model"]
        if canonical is None:
            key, label = raw, raw
        else:
            name = canonical(raw)
            key, label = name.identity, name.label
        merged.setdefault(key, []).append(row)
        spellings[key].append(label)
    rates: list[ModelCacheRate] = []
    for key, scheme_rows in merged.items():
        cache_read = 0
        prompt = 0
        events = 0
        known = False
        for row in scheme_rows:
            scheme = TokenScheme(str(row["token_scheme"]))
            input_sum = row["input_sum"] or 0
            read_sum = row["cache_read_sum"] or 0
            events += row["events"] or 0
            if scheme is TokenScheme.ANTHROPIC:
                known = True
                cache_read += read_sum
                prompt += input_sum + read_sum
            elif scheme is TokenScheme.OPENAI:
                known = True
                cache_read += read_sum
                prompt += input_sum
            else:
                prompt += input_sum
        rates.append(
            ModelCacheRate(
                model=_display_key(key, spellings),
                cache_read=cache_read if known else 0,
                prompt=prompt if known else 0,  # no rate without a known scheme
                events=events,
            )
        )
    rates.sort(key=lambda item: (-(item.rate or 0.0), -item.events))
    return rates


def _tool_where(flt: UsageFilter) -> tuple[str, list[object]]:
    clauses: list[str] = []
    params: list[object] = []
    if flt.since_ms is not None:
        clauses.append("ts_ms >= ?")
        params.append(flt.since_ms)
    if flt.until_ms is not None:
        clauses.append("ts_ms < ?")
        params.append(flt.until_ms)
    if flt.agents:
        clauses.append(f"agent IN ({', '.join('?' * len(flt.agents))})")
        params.extend(flt.agents)
    if flt.models:
        clauses.append(f"model IN ({', '.join('?' * len(flt.models))})")
        params.extend(flt.models)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


def _tool_rows(conn: sqlite3.Connection, flt: UsageFilter) -> list[sqlite3.Row]:
    where, params = _tool_where(flt)
    return conn.execute(f"SELECT * FROM tool_events {where} ORDER BY ts_ms", params).fetchall()
