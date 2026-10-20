"""Rich table builders and human formatting helpers.

Formatting rules that matter for honesty of the numbers:

* unknown values render as ``—``, never as 0;
* costs are shown with a currency and enough decimals to stay meaningful;
* percentages say what they are percentages *of* in the column header.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, tzinfo

from rich import box
from rich.table import Table

from toknt.analytics.aggregation import (
    BreakdownRow,
    SessionSummary,
    default_timezone,
)
from toknt.analytics.costs import CostEstimate
from toknt.analytics.quality import AgentQuality
from toknt.analytics.tool_metrics import DurationStats, ToolOutcomeRow

DASH = "—"


def humanize_int(value: int | None) -> str:
    if value is None:
        return DASH
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 10_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,}"


def humanize_cost(value: float | None, currency: str = "USD") -> str:
    if value is None:
        return DASH
    symbol = "$" if currency == "USD" else ""
    if value == 0:
        return f"{symbol}0.00"
    if abs(value) < 0.01:
        return f"{symbol}{value:.4f}"
    return f"{symbol}{value:,.2f}"


def humanize_percent(fraction: float | None) -> str:
    if fraction is None:
        return DASH
    return f"{fraction * 100:.1f}%"


def humanize_duration(ms: int | None) -> str:
    if ms is None:
        return DASH
    if ms < 1000:
        return f"{ms}ms"
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {remainder:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def humanize_epoch_datetime(ms: int | None, tz: tzinfo | None = None) -> str:
    if ms is None:
        return DASH
    tz = tz or default_timezone()
    return datetime.fromtimestamp(ms / 1000, tz=tz).strftime("%Y-%m-%d %H:%M")


def cost_note(estimate: CostEstimate | None) -> str:
    """One-line coverage note for an estimated cost."""
    if estimate is None or not (estimate.priced_events or estimate.unpriced_events):
        return "no priced events"
    return (
        f"{estimate.priced_events:,} of "
        f"{estimate.priced_events + estimate.unpriced_events:,} events priced "
        f"({estimate.coverage * 100:.0f}%)"
    )


def usage_table(
    rows: Sequence[BreakdownRow],
    *,
    title: str,
    dimension_label: str,
    display: Callable[[str], str] = str,
    cost_estimates: Mapping[str, CostEstimate] | None = None,
    limit: int | None = None,
) -> Table:
    table = Table(title=title, box=box.SIMPLE_HEAVY)
    table.add_column(dimension_label, overflow="fold")
    table.add_column("requests", justify="right")
    table.add_column("total", justify="right")
    table.add_column("input", justify="right")
    table.add_column("output", justify="right")
    table.add_column("cached", justify="right")
    table.add_column("share", justify="right")
    table.add_column("cost", justify="right")
    if cost_estimates is not None:
        table.add_column("est. cost", justify="right")

    visible = rows[:limit] if limit else rows
    grand_total = sum(
        row.summary.total for row in rows if row.summary.total is not None
    )
    for row in visible:
        summary = row.summary
        cached = None
        if summary.cache_read is not None and summary.cache_write is not None:
            cached = summary.cache_read + summary.cache_write
        share = None
        if summary.total is not None and grand_total:
            share = summary.total / grand_total
        cost_cell = humanize_cost(summary.cost_reported)
        if summary.total is None and summary.events_without_total:
            cost_cell += " *"
        cells = [
            display(row.key),
            f"{summary.events:,}" if summary.events else DASH,
            humanize_int(summary.total),
            humanize_int(summary.input),
            humanize_int(summary.output),
            humanize_int(cached),
            humanize_percent(share),
            cost_cell,
        ]
        if cost_estimates is not None:
            estimate = cost_estimates.get(row.key)
            cells.append(humanize_cost(estimate.estimated) if estimate else DASH)
        table.add_row(*cells)
    return table


def tool_table(
    rows: Sequence[ToolOutcomeRow],
    *,
    title: str = "Tool outcomes",
    durations: Mapping[str, DurationStats] | None = None,
    limit: int = 12,
) -> Table:
    table = Table(title=title, box=box.SIMPLE_HEAVY)
    table.add_column("tool", overflow="fold")
    table.add_column("calls", justify="right")
    table.add_column("success rate", justify="right")
    table.add_column("ok", justify="right")
    table.add_column("fail", justify="right")
    table.add_column("denied", justify="right")
    table.add_column("pending", justify="right")
    table.add_column("unknown", justify="right")
    table.add_column("observable", justify="right")
    if durations is not None:
        table.add_column("p90", justify="right")
    for row in rows[:limit]:
        cells = [
            row.tool,
            f"{row.calls:,}",
            humanize_percent(row.success_rate) if row.known_outcomes else DASH,
            f"{row.success:,}",
            f"{row.failure:,}",
            f"{row.denied:,}",
            f"{row.pending:,}",
            f"{row.unknown:,}",
            humanize_percent(row.observability),
        ]
        if durations is not None:
            stats = durations.get(row.tool)
            cells.append(humanize_duration(stats.p90_ms) if stats else DASH)
        table.add_row(*cells)
    return table


def session_table(
    rows: Sequence[SessionSummary],
    *,
    title: str = "Sessions",
    display: Callable[[str], str] = str,
) -> Table:
    table = Table(title=title, box=box.SIMPLE_HEAVY)
    table.add_column("session", overflow="fold")
    table.add_column("title", overflow="ellipsis", max_width=32)
    table.add_column("agent")
    table.add_column("model", overflow="fold")
    table.add_column("requests", justify="right")
    table.add_column("total", justify="right")
    table.add_column("tools", justify="right")
    table.add_column("cost", justify="right")
    table.add_column("last activity")
    for row in rows:
        models = ", ".join(display(model) for model in row.models[:2])
        if len(row.models) > 2:
            models += f" +{len(row.models) - 2}"
        marker = " !" if row.outlier else ""
        table.add_row(
            row.session_id[:20] + ("…" if len(row.session_id) > 20 else "") + marker,
            row.title or DASH,
            row.agent,
            models or DASH,
            f"{row.summary.events:,}",
            humanize_int(row.summary.total),
            f"{row.tool_calls:,}",
            humanize_cost(row.summary.cost_reported),
            humanize_epoch_datetime(row.last_ms),
        )
    return table


def quality_table(rows: Sequence[AgentQuality]) -> Table:
    table = Table(title="Adapter data coverage (all time)", box=box.SIMPLE_HEAVY)
    table.add_column("agent")
    table.add_column("events", justify="right")
    table.add_column("with model", justify="right")
    table.add_column("complete tokens", justify="right")
    table.add_column("reported cost", justify="right")
    table.add_column("tool calls", justify="right")
    table.add_column("known outcomes", justify="right")
    table.add_column("last ingest")
    for row in rows:
        table.add_row(
            row.agent,
            f"{row.usage_events:,}",
            humanize_percent(row.model_coverage) if row.usage_events else DASH,
            humanize_percent(row.token_coverage) if row.usage_events else DASH,
            (
                humanize_percent(row.events_with_reported_cost / row.usage_events)
                if row.usage_events
                else DASH
            ),
            f"{row.tool_calls:,}",
            humanize_percent(row.outcome_coverage) if row.tool_calls else DASH,
            humanize_epoch_datetime(row.last_ingest_ms),
        )
    return table
