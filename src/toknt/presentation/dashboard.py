"""The default overview screen: totals, trend, model mix, tools, warnings.

Read-only and side-effect free. Every number either comes from a canonical
event or is explicitly labelled as an estimate with its coverage.
"""

from __future__ import annotations

import sqlite3
from datetime import tzinfo

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from toknt.analytics import aggregation, costs, model_names, tool_metrics
from toknt.config import AppConfig
from toknt.presentation import charts, tables
from toknt.storage.ingestion import quality_issue_summary
from toknt.util import now_ms


def render_overview(
    conn: sqlite3.Connection,
    *,
    config: AppConfig,
    console: Console,
    since: str | None = None,
    bucket: str | None = None,
    tz: tzinfo | None = None,
    merge_models: bool = False,
) -> None:
    tz = tz or aggregation.default_timezone()
    bucket = bucket or config.default_bucket
    since_text = since or config.default_since

    total_events = conn.execute("SELECT COUNT(*) FROM usage_events").fetchone()[0]
    if total_events == 0:
        console.print(
            Panel(
                "No usage data has been imported yet.\n\n"
                "Run [bold]toknt scan[/bold] to import local agent history.\n"
                "Run [bold]toknt doctor[/bold] to see which sources were detected.",
                title="toknt — overview",
            )
        )
        return

    try:
        since_ms = aggregation.parse_since(since_text, tz=tz)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return
    flt = aggregation.UsageFilter(since_ms=since_ms, ignore_projects=config.ignore_projects)

    summary = aggregation.overall_summary(conn, flt)
    if summary.summary.events == 0:
        console.print(
            Panel(
                f"No usage events in the last [bold]{since_text}[/bold]. "
                "Try a wider window, e.g. [bold]toknt usage --since 180d[/bold].",
                title="toknt — overview",
            )
        )
        return

    price_map = costs.load_price_map(conn, config)
    estimate = costs.cost_summary(conn, flt, price_map)

    console.print(_totals_panel(summary, estimate, since_text))
    _print_trend(
        conn, flt, bucket, tz, config, console, merge_models=merge_models
    )
    _print_models(conn, flt, price_map, config, console, merge_models=merge_models)
    _print_tools(conn, flt, console)
    _print_notes(conn, summary, estimate, console)


def _totals_panel(
    summary: aggregation.OverallSummary,
    estimate: costs.CostEstimate,
    since_text: str,
) -> Panel:
    tokens = summary.summary
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="left")
    table.add_column(justify="right")
    cached = None
    if tokens.cache_read is not None and tokens.cache_write is not None:
        cached = tokens.cache_read + tokens.cache_write
    table.add_row("usage events", f"{tokens.events:,}")
    table.add_row("sessions", f"{tokens.sessions:,}")
    table.add_row("input tokens", tables.humanize_int(tokens.input))
    table.add_row("output tokens", tables.humanize_int(tokens.output))
    table.add_row("cached tokens (read + write)", tables.humanize_int(cached))
    table.add_row("reasoning tokens", tables.humanize_int(tokens.reasoning))
    table.add_row("[bold]total tokens[/bold]", f"[bold]{tables.humanize_int(tokens.total)}[/bold]")
    table.add_row("reported cost", tables.humanize_cost(tokens.cost_reported))
    table.add_row(
        "estimated cost",
        f"{tables.humanize_cost(estimate.estimated)} [dim]({tables.cost_note(estimate)})[/dim]",
    )
    if tokens.events_without_total:
        table.add_row(
            "[yellow]excluded from totals[/yellow]",
            f"[yellow]{tokens.events_without_total:,} event(s) with incomplete components[/yellow]",
        )
    return Panel(
        table,
        title=f"toknt — last {since_text}",
        subtitle=tables.humanize_epoch_datetime(now_ms()) + " local time",
    )


def _print_trend(
    conn: sqlite3.Connection,
    flt: aggregation.UsageFilter,
    bucket: str,
    tz: tzinfo,
    config: AppConfig,
    console: Console,
    *,
    merge_models: bool = False,
) -> None:
    canonical = model_names.canonical_model if merge_models else None
    series = aggregation.usage_series(
        conn, flt, dimension="model", metric="total", bucket=bucket, tz=tz, top_n=6,
        canonical=canonical,
    )
    renamed = aggregation.Series(
        series.buckets,
        {config.display_name(str(key)): values for key, values in series.series.items()},
        series.dimension,
        series.bucket,
    )
    chart = charts.line_chart(
        renamed,
        title=f"Token usage per {bucket} by model",
        ylabel="tokens",
        tz=tz,
        log=True,
    )
    if chart:
        console.print(chart)


def _print_models(
    conn: sqlite3.Connection,
    flt: aggregation.UsageFilter,
    price_map: dict[str, costs.Price],
    config: AppConfig,
    console: Console,
    *,
    merge_models: bool = False,
) -> None:
    canonical = model_names.canonical_model if merge_models else None
    breakdown = aggregation.usage_breakdown(
        conn, flt, dimension="model", canonical=canonical
    )
    if not breakdown:
        return
    estimates = costs.cost_by_dimension(
        conn, flt, price_map, dimension="model", canonical=canonical
    )
    console.print(
        tables.usage_table(
            breakdown,
            title="Token usage by model",
            dimension_label="model",
            display=config.display_name,
            cost_estimates=estimates,
            limit=8,
        )
    )


def _print_tools(conn: sqlite3.Connection, flt: aggregation.UsageFilter, console: Console) -> None:
    rows = tool_metrics.tool_outcomes(conn, flt)
    if not rows:
        return
    durations = tool_metrics.tool_durations(conn, flt)
    console.print(tables.tool_table(rows, durations=durations, limit=10))


def _print_notes(
    conn: sqlite3.Connection,
    summary: aggregation.OverallSummary,
    estimate: costs.CostEstimate,
    console: Console,
) -> None:
    notes: list[str] = []
    tokens = summary.summary
    if tokens.events_without_total:
        notes.append(
            f"{tokens.events_without_total:,} of {tokens.events:,} events lack complete token "
            "components; they are excluded from totals (details: toknt doctor)."
        )
    if estimate.unpriced_events:
        notes.append(
            f"Estimated cost covers {estimate.priced_events:,} of "
            f"{estimate.priced_events + estimate.unpriced_events:,} events; "
            "add prices with [bold]toknt pricing add[/bold]."
        )
    for issue in quality_issue_summary(conn)[:5]:
        notes.append(
            f"{issue['agent']}/{issue['code']}: {issue['message']} "
            f"({issue['occurrences']:,}×)"
        )
    if not notes:
        return
    panel = Table.grid()
    for note in notes:
        panel.add_row(f"• {note}")
    console.print(Panel(panel, title="Notes", border_style="yellow"))
