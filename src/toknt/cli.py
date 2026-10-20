"""Command-line interface for toknt.

A bare ``toknt`` renders the overview dashboard; subcommands cover ingestion
(``scan``), analytics (``usage``, ``models``, ``tools``, ``sessions``),
machine-readable exports (``export``), pricing for cost estimates
(``pricing``) and diagnostics (``doctor``). The CLI is a thin shell over
:mod:`toknt.service` and :mod:`toknt.analytics`; all real work happens there.
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import sys
from importlib import metadata
from typing import NoReturn

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from toknt import config, service
from toknt.adapters import registered_adapters
from toknt.analytics import aggregation, costs, model_names, tool_metrics
from toknt.config import AppConfig
from toknt.models.events import Harness
from toknt.models.usage import TokenCounts, TokenScheme, effective_total
from toknt.presentation import charts, dashboard, tables
from toknt.storage import database, pricing
from toknt.storage.ingestion import quality_issue_summary
from toknt.storage.migrations import schema_version

_HELP_CTX = {"help_option_names": ["-h", "--help"]}

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    context_settings=_HELP_CTX,
    help=(
        "Local-first usage analytics for AI coding harnesses "
        "(OpenCode, Codex CLI, Claude Code, and more)."
    ),
)
pricing_app = typer.Typer(
    help="Manage per-model prices used for cost estimates.",
    context_settings=_HELP_CTX,
)
app.add_typer(pricing_app, name="pricing")

console = Console()
error_console = Console(stderr=True)

_HARNESS_HELP = (
    "Restrict to a harness (repeatable): "
    + ", ".join(str(harness) for harness in Harness)
    + "."
)

#: ``--agent`` is kept as a hidden alias for the renamed ``--harness`` flag.
_AGENT_OPTION = typer.Option(None, "--harness", "--agent", help=_HARNESS_HELP)
_SINCE_HELP = (
    "Lookback window: 12h / 30d / 2w, 'today', an ISO date, or 'all' "
    "(default from config)."
)
_UNTIL_HELP = "Exclusive end: a lookback offset (7d) or a date (2026-10-01 = start of that day)."


# --------------------------------------------------------------------- helpers


def _installed_version() -> str:
    try:
        return metadata.version("toknt")
    except metadata.PackageNotFoundError:  # running from a source tree without install
        return "0.0.0+dev"


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"toknt {_installed_version()}")
        raise typer.Exit()


def _fail(message: str) -> NoReturn:
    error_console.print(Panel(message, title="toknt error", border_style="red"))
    raise typer.Exit(code=1)


def _open() -> sqlite3.Connection:
    try:
        return database.open_database()
    except sqlite3.Error as exc:
        _fail(f"Could not open the analytics database at {config.db_path()}:\n{exc}")


def _config() -> AppConfig:
    try:
        return config.load_config()
    except config.ConfigError as exc:
        _fail(str(exc))


def _period_filter(
    cfg: AppConfig,
    *,
    since: str | None,
    until: str | None,
    agents: list[str] | None = None,
    models: list[str] | None = None,
    projects: list[str] | None = None,
) -> aggregation.UsageFilter:
    tz = aggregation.default_timezone()
    since_value = since or cfg.default_since
    try:
        since_ms = 0 if since_value == "all" else aggregation.parse_since(since_value, tz=tz)
        until_ms = aggregation.parse_since(until, tz=tz) if until else None
    except ValueError as exc:
        _fail(str(exc))
    if until_ms is not None and since_ms > until_ms:
        _fail("--until is before --since")
    valid_harnesses = {str(harness) for harness in Harness}
    for name in agents or []:
        if name not in valid_harnesses:
            _fail(
                f"unknown harness {name!r}; expected one of {', '.join(sorted(valid_harnesses))}"
            )
    return aggregation.UsageFilter(
        since_ms=since_ms,
        until_ms=until_ms,
        agents=tuple(agents or ()),
        models=tuple(models or ()),
        projects=tuple(projects or ()),
        ignore_projects=cfg.ignore_projects,
    )


def _metric_choices() -> str:
    return ", ".join(aggregation.METRICS)


# ----------------------------------------------------------------------- root


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        help="Show the installed version and exit.",
        is_eager=True,
        callback=_version_callback,
    ),
) -> None:
    """Interactively explore usage (bare), or run a subcommand."""
    if ctx.invoked_subcommand is None:
        import os
        import sys

        if os.environ.get("TOKNT_TUI", "").casefold() in ("0", "off", "no"):
            overview()
            return
        if sys.stdin.isatty() and sys.stdout.isatty():
            from toknt.tui.app import run as run_tui

            run_tui(config.load_config())  # errors already boxed by _fail paths
        else:
            overview()


@app.command("overview")
def overview_cmd(
    merge_models: bool = typer.Option(
        False,
        "--merge-models",
        help="Merge the same model served by different providers (one row per model).",
    ),
) -> None:
    """Print the static overview dashboard (same as piping bare toknt)."""
    overview(merge_models=merge_models)


@app.command()
def graphics(
    backend: str = typer.Option(
        "auto",
        "--backend",
        help="auto|kitty|iterm2|sixel|off — force a transport for this test.",
    ),
    alt: bool = typer.Option(
        False,
        "--alt",
        help="Draw inside the alternate screen with raw mode, like the TUI does.",
    ),
) -> None:
    """Render a test chart to check whether this terminal shows images."""
    from toknt.presentation import graphics as gfx

    detected, reason = gfx.describe_backend(override=backend)
    console.print(f"[bold]graphics backend[/bold]: {detected}")
    console.print(f"[dim]{reason}[/dim]")
    if detected == "off":
        console.print(
            "[dim]charts will use Braille cells; set TOKNT_GRAPHICS=kitty|sixel|iterm2"
            " if your terminal does support images.[/dim]"
        )
        return
    width, height = 800, 200
    png, rgba = gfx.test_chart_png(width, height)
    columns, rows = 80, 20
    payload = gfx.encode_placement(
        detected, png, rgba, width, height, columns=columns, rows=rows, image_id=1
    )
    if alt:
        # same context as the TUI: alternate screen + raw mode
        import termios
        import tty

        saved = termios.tcgetattr(0)
        try:
            tty.setraw(0)
            sys.stdout.write("\x1b[?1049h\x1b[?25l\x1b[H\x1b[2J")
            sys.stdout.flush()
            sys.stdout.buffer.write(b"\x1b[1;1H")
            sys.stdout.buffer.write(payload)
            sys.stdout.buffer.flush()
            console.print(
                "[dim]image drawn in the alternate screen (raw mode) —"
                " press q to leave[/dim]"
            )
            import select

            while True:
                ready, _, _ = select.select([0], [], [], 30)
                if not ready:
                    break
                key = os.read(0, 1)
                if key in (b"q", b"\x03"):
                    break
        finally:
            termios.tcsetattr(0, termios.TCSADRAIN, saved)
            sys.stdout.write("\x1b[?25h\x1b[?1049l\x1b[0m")
            sys.stdout.flush()
        return
    sys.stdout.buffer.write(b"\x1b[1;1H")
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.write(b"\x1b[0m\r\n")
    sys.stdout.buffer.flush()
    console.print(
        f"[dim]sent {len(payload):,} bytes ({detected}); if no image appears above,"
        " your terminal does not implement this protocol.[/dim]"
    )


@app.command()
def scan(
    full: bool = typer.Option(
        False, "--full", help="Reindex sources from scratch instead of scanning incrementally."
    ),
) -> None:
    """Import usage data from detected local sources."""
    cfg = _config()
    conn = _open()
    mode = "full" if full else "incremental"
    try:
        outcomes = service.run_scan(conn, full=full, config=cfg)
    finally:
        conn.close()

    table = Table(title="toknt scan", box=box.SIMPLE_HEAVY)
    table.add_column("source")
    table.add_column("status")
    table.add_column("new", justify="right")
    table.add_column("updated", justify="right")
    table.add_column("skipped", justify="right")
    table.add_column("notes", overflow="fold")
    for outcome in outcomes:
        if outcome.report is None:
            table.add_row(
                outcome.source.source_id,
                "[dim]skipped[/dim]",
                "",
                "",
                "",
                outcome.skipped_reason or "",
            )
            continue
        report = outcome.report
        status_style = {"ok": "green", "partial": "yellow", "error": "red"}.get(
            report.status, "white"
        )
        inserted = report.usage_inserted + report.tool_inserted
        updated = report.usage_updated + report.tool_updated
        notes = "; ".join(report.notes)
        if report.reconciled:
            removed = f"{report.reconciled} removed"
            notes = f"{notes}; {removed}" if notes else removed
        table.add_row(
            outcome.source.source_id,
            f"[{status_style}]{report.status}[/{status_style}]",
            f"{inserted:,}",
            f"{updated:,}",
            f"{report.skipped:,}",
            notes,
        )
    console.print(table)
    console.print(f"Scan mode: {mode}. Database: [bold]{config.db_path()}[/bold]")
    if not outcomes:
        console.print(
            "[dim]No adapters enabled. Check adapters.enabled in the config file "
            "(toknt doctor shows the path).[/dim]"
        )
    elif all(outcome.report is None for outcome in outcomes):
        console.print(
            "[dim]No source data found. Harnesses are looked up in their standard "
            "locations; see toknt doctor for paths checked.[/dim]"
        )


def overview(*, merge_models: bool = False) -> None:
    """Print the overview dashboard (also the bare ``toknt`` behavior)."""
    cfg = _config()
    conn = _open()
    try:
        dashboard.render_overview(
            conn, config=cfg, console=console, merge_models=merge_models
        )
    finally:
        conn.close()


# ------------------------------------------------------------------ analytics


@app.command()
def usage(
    since: str | None = typer.Option(None, "--since", help=_SINCE_HELP),
    until: str | None = typer.Option(None, "--until", help=_UNTIL_HELP),
    by: str = typer.Option(
        "model", "--by", help="Breakdown dimension: model, harness or project."
    ),
    metric: str = typer.Option("total", "--metric", help=f"Chart metric: {_metric_choices()}."),
    bucket: str | None = typer.Option(
        None, "--bucket", help="hour, day or week (default from config)."
    ),
    chart: str = typer.Option("line", "--chart", help="line, bar, stacked or none."),
    cumulative: bool = typer.Option(False, "--cumulative", help="Plot running totals."),
    top: int = typer.Option(8, "--top", help="Series kept before grouping into (other)."),
    harness: list[str] | None = _AGENT_OPTION,
    model: list[str] | None = typer.Option(
        None, "--model", help="Restrict to a model (repeatable)."
    ),
    project: list[str] | None = typer.Option(
        None, "--project", help="Restrict to a project path (repeatable)."
    ),
    merge_models: bool = typer.Option(
        False,
        "--merge-models",
        help="Merge the same model served by different providers (one row per model).",
    ),
) -> None:
    """Token usage over time, broken down by model, harness or project."""
    cfg = _config()
    if by not in ("model", "harness", "agent", "project"):
        _fail("--by must be model, harness or project")
    by_label = "harness" if by == "agent" else by
    by_dimension = "agent" if by == "harness" else by  # the model field name
    if metric not in aggregation.METRICS:
        _fail(f"--metric must be one of {_metric_choices()}")
    if chart not in ("line", "bar", "stacked", "none"):
        _fail("--chart must be line, bar, stacked or none")
    if top < 1:
        _fail("--top must be at least 1")
    bucket = bucket or cfg.default_bucket
    if bucket not in aggregation.BUCKETS:
        _fail(f"--bucket must be one of {', '.join(aggregation.BUCKETS)}")
    flt = _period_filter(
        cfg, since=since, until=until, agents=harness, models=model, projects=project
    )

    conn = _open()
    try:
        tz = aggregation.default_timezone()
        canonical = (
            model_names.canonical_model
            if merge_models and by_dimension == "model"
            else None
        )
        if metric == "requests":
            activity = aggregation.activity_series(conn, flt, bucket=bucket, tz=tz)
            series = aggregation.Series(
                activity.buckets,
                {"requests": activity.series["requests"]},
                "requests",
                bucket,
            )
        else:
            series = aggregation.usage_series(
                conn, flt, dimension=by_dimension, metric=metric, bucket=bucket, tz=tz,
                top_n=top, canonical=canonical,
            )
            if cumulative:
                series = aggregation.cumulative(series)
        breakdown = aggregation.usage_breakdown(
            conn, flt, dimension=by_dimension, canonical=canonical
        )
        price_map = costs.load_price_map(conn, cfg)
        estimates = costs.cost_by_dimension(
            conn, flt, price_map, dimension=by_dimension, canonical=canonical
        )
        estimate = costs.cost_summary(conn, flt, price_map)
        overall = aggregation.overall_summary(conn, flt)
    finally:
        conn.close()

    if not breakdown:
        console.print("No usage in the selected period.")
        raise typer.Exit()

    if chart != "none":
        renamed = aggregation.Series(
            series.buckets,
            {cfg.display_name(str(key)): values for key, values in series.series.items()},
            series.dimension,
            series.bucket,
        )
        title = f"{aggregation.metric_label(metric).capitalize()} per {bucket}"
        ylabel = aggregation.metric_label(metric)
        if metric == "requests":
            title = f"Requests per {bucket}"
            ylabel = "requests"
        if chart == "line":
            rendered = charts.line_chart(renamed, title=title, ylabel=ylabel, tz=tz)
        elif chart == "bar":
            rendered = charts.bar_chart(renamed, title=title, ylabel=ylabel, tz=tz)
        else:
            rendered = charts.bar_chart(
                renamed, title=title, ylabel=ylabel, tz=tz, stacked=True
            )
        if rendered:
            console.print(rendered)

    console.print(
        tables.usage_table(
            breakdown,
            title=f"Usage by {by_label} — {aggregation.metric_label(metric)}",
            dimension_label=by_label,
            display=cfg.display_name,
            cost_estimates=estimates,
        )
    )
    overall_summary = overall.summary
    console.print(
        f"Total [bold]{tables.humanize_int(overall_summary.total)}[/bold] tokens across "
        f"{overall_summary.events:,} requests · reported cost "
        f"{tables.humanize_cost(overall_summary.cost_reported)} · estimated "
        f"{tables.humanize_cost(estimate.estimated)} "
        f"[dim]({tables.cost_note(estimate)})[/dim]"
    )


@app.command()
def models(
    since: str | None = typer.Option(None, "--since", help=_SINCE_HELP),
    until: str | None = typer.Option(None, "--until", help=_UNTIL_HELP),
    harness: list[str] | None = _AGENT_OPTION,
    top: int = typer.Option(10, "--top", help="Models shown in the chart."),
    chart: bool = typer.Option(True, "--chart/--no-chart", help="Show the ranked bar chart."),
    merge_models: bool = typer.Option(
        False,
        "--merge-models",
        help="Merge the same model served by different providers (one row per model).",
    ),
    tps: bool = typer.Option(
        False,
        "--tps",
        help="Add a ≈tok/s column (mean generation rate, estimated from message gaps).",
    ),
) -> None:
    """Model inventory with per-model usage and costs."""
    cfg = _config()
    flt = _period_filter(cfg, since=since, until=until, agents=harness)
    canonical = model_names.canonical_model if merge_models else None

    conn = _open()
    try:
        breakdown = aggregation.usage_breakdown(
            conn, flt, dimension="model", canonical=canonical
        )
        rates = (
            aggregation.model_rate_totals(conn, flt, canonical=canonical) if tps else {}
        )
        inventory = {
            row["model"]: row for row in aggregation.model_inventory(conn)
        }
        provenance = (
            aggregation.model_provenance(conn, flt, canonical=canonical)
            if merge_models
            else {}
        )
        price_map = costs.load_price_map(conn, cfg)
        estimates = costs.cost_by_dimension(
            conn, flt, price_map, dimension="model", canonical=canonical
        )
        estimate = costs.cost_summary(conn, flt, price_map)
    finally:
        conn.close()

    if not breakdown:
        console.print("No usage in the selected period.")
        raise typer.Exit()

    table = Table(title=f"Models — last {since or cfg.default_since}", box=box.SIMPLE_HEAVY)
    table.add_column("model", overflow="fold")
    table.add_column("provider")
    table.add_column("agents", justify="right")
    table.add_column("requests", justify="right")
    table.add_column("total", justify="right")
    table.add_column("reported cost", justify="right")
    table.add_column("est. cost", justify="right")
    if tps:
        # ≈ marks the estimate: a message's completion timestamp is
        # recorded, its duration is not — the rate pairs each message
        # with its predecessor's gap
        table.add_column("≈ tok/s", justify="right")
    for row in breakdown:
        info = inventory.get(row.key, {})
        est = estimates.get(row.key)
        if merge_models:
            origin = provenance.get(row.key)
            providers = " · ".join(origin.providers) if origin else tables.DASH
            agent_count = len(origin.agents) if origin else 0
        else:
            providers = str(info.get("provider") or tables.DASH)
            agent_count = int(info.get("agents", 0) or 0)
        cells = [
            cfg.display_name(row.key),
            providers,
            str(agent_count),
            f"{row.summary.events:,}",
            tables.humanize_int(row.summary.total),
            tables.humanize_cost(row.summary.cost_reported),
            tables.humanize_cost(est.estimated) if est else tables.DASH,
        ]
        if tps:
            rate = rates.get(row.identity)
            cells.append(f"{rate:.1f}" if rate is not None else tables.DASH)
        table.add_row(*cells)
    console.print(table)

    if chart:
        chart_rows = [row for row in breakdown if row.summary.total is not None][:top]
        rendered = charts.category_bar(
            [cfg.display_name(row.key) for row in chart_rows],
            [row.summary.total for row in chart_rows],
            title=f"Total tokens by model — last {since or cfg.default_since}",
        )
        if rendered:
            console.print(rendered)
    console.print(
        f"Estimated cost covers "
        f"{tables.cost_note(estimate)}; add or update prices with "
        f"[bold]toknt pricing add[/bold]."
    )


@app.command()
def tools(
    since: str | None = typer.Option(None, "--since", help=_SINCE_HELP),
    until: str | None = typer.Option(None, "--until", help=_UNTIL_HELP),
    harness: list[str] | None = _AGENT_OPTION,
    tool: list[str] | None = typer.Option(
        None, "--tool", help="Restrict to a tool name (repeatable)."
    ),
    bucket: str | None = typer.Option(
        None, "--bucket", help="hour, day or week (for --chart trend)."
    ),
    chart: str = typer.Option("rate", "--chart", help="rate, trend or none."),
    limit: int = typer.Option(12, "--limit", help="Number of tools to list."),
) -> None:
    """Tool-call outcomes and success rates."""
    cfg = _config()
    if chart not in ("rate", "trend", "none"):
        _fail("--chart must be rate, trend or none")
    bucket = bucket or cfg.default_bucket
    flt = _period_filter(cfg, since=since, until=until, agents=harness)

    conn = _open()
    try:
        rows = tool_metrics.tool_outcomes(conn, flt)
        if tool:
            wanted = set(tool)
            rows = [row for row in rows if row.tool in wanted]
        durations = tool_metrics.tool_durations(conn, flt)
        trend = None
        if chart == "trend":
            tz = aggregation.default_timezone()
            series = tool_metrics.tool_success_series(conn, flt, bucket=bucket, tz=tz)
            top_tools = [row.tool for row in rows[:3]]
            trend = aggregation.Series(
                series.buckets,
                {key: values for key, values in series.series.items() if key in top_tools},
                series.dimension,
                series.bucket,
            )
    finally:
        conn.close()

    if not rows:
        console.print("No tool calls in the selected period.")
        raise typer.Exit()

    if chart == "rate":
        rated = [row for row in rows[:15] if row.success_rate is not None]
        rendered = charts.category_bar(
            [row.tool for row in rated],
            [(row.success_rate or 0) * 100 for row in rated],
            title="Tool success rate — % of known outcomes",
            unit="%",
        )
        if rendered:
            console.print(rendered)
    elif chart == "trend" and trend is not None:
        rendered = charts.line_chart(
            trend,
            title=f"Tool success rate per {bucket} — top tools by calls",
            ylabel="success rate",
            percent=True,
        )
        if rendered:
            console.print(rendered)

    console.print(tables.tool_table(rows, durations=durations, limit=limit))
    if len(rows) > limit:
        console.print(
            f"[dim]… {len(rows) - limit:,} more tool(s) not shown; raise [bold]--limit[/bold] "
            "to list them all.[/dim]"
        )
    console.print(
        "[dim]Success rate = successes ÷ calls with known final outcomes "
        "(pending and unknown excluded); 'observable' shows how many calls have a "
        "known outcome at all.[/dim]"
    )


@app.command()
def sessions(
    since: str | None = typer.Option(None, "--since", help=_SINCE_HELP),
    until: str | None = typer.Option(None, "--until", help=_UNTIL_HELP),
    harness: list[str] | None = _AGENT_OPTION,
    limit: int = typer.Option(20, "--limit", help="Number of sessions to show."),
) -> None:
    """Sessions ranked by usage, with outlier flags."""
    cfg = _config()
    flt = _period_filter(cfg, since=since, until=until, agents=harness)

    conn = _open()
    try:
        rows = aggregation.session_ranking(conn, flt, limit=limit)
    finally:
        conn.close()

    if not rows:
        console.print("No sessions in the selected period.")
        raise typer.Exit()
    console.print(tables.session_table(rows, display=cfg.display_name))
    if any(row.outlier for row in rows):
        console.print(
            "[dim]! = session total above mean + 2σ for this period (possible spike).[/dim]"
        )


# --------------------------------------------------------------------- export

_USAGE_EXPORT_COLUMNS = (
    "id",
    "agent",
    "source_id",
    "source_event_id",
    "session_id",
    "parent_session_id",
    "project",
    "directory",
    "agent_mode",
    "provider",
    "model",
    "ts_ms",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "provider_total_tokens",
    "token_scheme",
    "effective_total_tokens",
    "cost_reported",
    "cost_currency",
    "finish_reason",
    "quality",
    "quality_notes",
    "source_event_type",
)

_TOOL_EXPORT_COLUMNS = (
    "id",
    "agent",
    "source_id",
    "source_event_id",
    "session_id",
    "call_id",
    "tool_name",
    "tool_category",
    "ts_ms",
    "duration_ms",
    "outcome",
    "outcome_provenance",
    "outcome_confidence",
    "model",
    "agent_mode",
    "quality",
    "quality_notes",
    "source_event_type",
)

_SESSION_EXPORT_COLUMNS = (
    "agent",
    "session_id",
    "source_id",
    "parent_session_id",
    "project",
    "directory",
    "title",
    "agent_mode",
    "model",
    "provider",
    "time_created_ms",
    "time_updated_ms",
    "cost_reported",
    "quality",
    "quality_notes",
)


def _export_usage(conn: sqlite3.Connection, flt: aggregation.UsageFilter) -> list[dict]:
    where, params = aggregation._usage_where(flt)
    rows = conn.execute(
        f"SELECT * FROM usage_events {where} ORDER BY ts_ms", params
    ).fetchall()
    out = []
    for row in rows:
        record = {
            name: row[name]
            for name in _USAGE_EXPORT_COLUMNS
            if name != "effective_total_tokens"
        }
        scheme_name = row["token_scheme"]
        scheme = (
            TokenScheme(scheme_name)
            if scheme_name in {item.value for item in TokenScheme}
            else TokenScheme.UNKNOWN
        )
        counts = TokenCounts(
            input=row["input_tokens"],
            output=row["output_tokens"],
            reasoning=row["reasoning_tokens"],
            cache_read=row["cache_read_tokens"],
            cache_write=row["cache_write_tokens"],
        )
        record["effective_total_tokens"] = effective_total(counts, scheme)
        ordered = {name: record.get(name) for name in _USAGE_EXPORT_COLUMNS}
        out.append(ordered)
    return out


def _export_tools(conn: sqlite3.Connection, flt: aggregation.UsageFilter) -> list[dict]:
    where, params = aggregation._tool_where(flt)
    rows = conn.execute(
        f"SELECT * FROM tool_events {where} ORDER BY ts_ms", params
    ).fetchall()
    return [{name: row[name] for name in _TOOL_EXPORT_COLUMNS} for row in rows]


def _export_sessions(conn: sqlite3.Connection, flt: aggregation.UsageFilter) -> list[dict]:
    clauses: list[str] = []
    params: list[object] = []
    if flt.agents:
        clauses.append(f"s.agent IN ({', '.join('?' * len(flt.agents))})")
        params.extend(flt.agents)
    if flt.since_ms is not None:
        clauses.append("e.ts_ms >= ?")
        params.append(flt.since_ms)
    if flt.until_ms is not None:
        clauses.append("e.ts_ms < ?")
        params.append(flt.until_ms)
    rows = conn.execute(
        f"""
        SELECT s.* FROM sessions s
        WHERE EXISTS (
            SELECT 1 FROM usage_events e
            WHERE e.agent = s.agent AND e.session_id = s.session_id
            {("AND " + " AND ".join(clauses)) if clauses else ""}
        )
        ORDER BY s.time_updated_ms
        """,
        params,
    ).fetchall()
    return [{name: row[name] for name in _SESSION_EXPORT_COLUMNS} for row in rows]


@app.command("export")
def export(
    what: str = typer.Option("usage", "--what", help="usage, tools or sessions."),
    fmt: str = typer.Option("csv", "--format", help="csv or json."),
    since: str | None = typer.Option(None, "--since", help=_SINCE_HELP),
    until: str | None = typer.Option(None, "--until", help=_UNTIL_HELP),
    harness: list[str] | None = _AGENT_OPTION,
    model: list[str] | None = typer.Option(
        None, "--model", help="usage only: restrict to a model."
    ),
    project: list[str] | None = typer.Option(
        None, "--project", help="usage only: restrict to a project."
    ),
) -> None:
    """Export canonical records as CSV or JSON (machine-readable, stdout)."""
    cfg = _config()
    if what not in ("usage", "tools", "sessions"):
        _fail("--what must be usage, tools or sessions")
    if fmt not in ("csv", "json"):
        _fail("--format must be csv or json")
    flt = _period_filter(
        cfg, since=since, until=until, agents=harness, models=model, projects=project
    )

    conn = _open()
    try:
        if what == "usage":
            rows = _export_usage(conn, flt)
            columns: tuple[str, ...] = _USAGE_EXPORT_COLUMNS
        elif what == "tools":
            rows = _export_tools(conn, flt)
            columns = _TOOL_EXPORT_COLUMNS
        else:
            rows = _export_sessions(conn, flt)
            columns = _SESSION_EXPORT_COLUMNS
    finally:
        conn.close()

    if fmt == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    else:
        sys.stdout.write(json.dumps(rows, default=str))
        sys.stdout.write("\n")


# -------------------------------------------------------------------- pricing


@pricing_app.command("list")
def pricing_list() -> None:
    """List configured price rows."""
    conn = _open()
    try:
        rows = pricing.list_prices(conn)
    finally:
        conn.close()
    if not rows:
        console.print(
            "No prices configured. Estimated costs stay unknown until prices are added "
            "(e.g. [bold]toknt pricing add gpt-5-codex --input 1.25 --output 10[/bold])."
        )
        return
    table = Table(title="Model prices (USD per million tokens)", box=box.SIMPLE_HEAVY)
    for column in ("model", "provider", "currency", "input", "output", "cache read", "cache write"):
        table.add_column(column)
    for row in rows:
        table.add_row(
            row.model,
            row.provider or tables.DASH,
            row.currency,
            _rate(row.input_per_mtok),
            _rate(row.output_per_mtok),
            _rate(row.cache_read_per_mtok),
            _rate(row.cache_write_per_mtok),
        )
    console.print(table)


def _rate(value: float | None) -> str:
    return tables.DASH if value is None else f"{value:g}"


@pricing_app.command("add")
def pricing_add(
    model: str = typer.Argument(..., help="Model id exactly as it appears in usage events."),
    input_rate: float | None = typer.Option(
        None, "--input", help="USD per million input tokens."
    ),
    output_rate: float | None = typer.Option(
        None, "--output", help="USD per million output tokens."
    ),
    cache_read_rate: float | None = typer.Option(
        None, "--cache-read", help="USD per million cache-read tokens."
    ),
    cache_write_rate: float | None = typer.Option(
        None, "--cache-write", help="USD per million cache-write tokens."
    ),
    provider: str = typer.Option("", "--provider", help="Provider id (optional)."),
    currency: str = typer.Option("USD", "--currency"),
    note: str | None = typer.Option(None, "--note"),
) -> None:
    """Add or update prices for one model."""
    if all(
        rate is None
        for rate in (input_rate, output_rate, cache_read_rate, cache_write_rate)
    ):
        _fail("provide at least one of --input, --output, --cache-read, --cache-write")
    conn = _open()
    try:
        pricing.add_price(
            conn,
            model=model,
            provider=provider,
            currency=currency,
            input_per_mtok=input_rate,
            output_per_mtok=output_rate,
            cache_read_per_mtok=cache_read_rate,
            cache_write_per_mtok=cache_write_rate,
            note=note,
        )
    finally:
        conn.close()
    console.print(f"Saved prices for [bold]{model}[/bold] ({currency} per million tokens).")


@pricing_app.command("remove")
def pricing_remove(
    model: str = typer.Argument(..., help="Model id to remove prices for."),
    provider: str | None = typer.Option(None, "--provider", help="Only remove this provider."),
) -> None:
    """Remove price rows for one model."""
    conn = _open()
    try:
        deleted = pricing.remove_price(conn, model=model, provider=provider)
    finally:
        conn.close()
    console.print(f"Removed {deleted} price row(s) for {model}.")


# --------------------------------------------------------------------- doctor


@app.command()
def doctor() -> None:
    """Explain configuration, sources, data coverage and quality."""
    cfg = _config()
    _print_config(cfg)
    conn = _open()
    try:
        _print_table_counts(conn)
        _print_discovered_sources(cfg)
        _print_sources(conn)
        from toknt.analytics.quality import harness_quality

        coverage = harness_quality(conn)
        if coverage:
            console.print(tables.quality_table(coverage))
        _print_quality_issues(conn)
        _print_pricing_hint(conn)
    finally:
        conn.close()


def _print_config(cfg: AppConfig) -> None:
    db_path = config.db_path()
    exists = db_path.exists()
    size = db_path.stat().st_size if exists else 0
    config_file = config.config_path()
    table = Table(title="Configuration", box=box.SIMPLE_HEAVY, show_header=False)
    table.add_column("Key", style="bold")
    table.add_column("Value")
    table.add_row("Data directory", str(config.data_dir()))
    table.add_row("Database", f"{db_path} ({'exists' if exists else 'missing'}, {size:,} bytes)")
    table.add_row(
        "Config file",
        f"{config_file} ({'exists' if config_file.exists() else 'missing — using defaults'})",
    )
    table.add_row("Enabled harnesses", ", ".join(str(h) for h in cfg.enabled_harnesses))
    table.add_row("Default window", f"since {cfg.default_since}, bucket {cfg.default_bucket}")
    if cfg.ignore_projects:
        table.add_row("Ignored projects", ", ".join(cfg.ignore_projects))
    from toknt.presentation import graphics as gfx

    backend, reason = gfx.describe_backend(override=cfg.ui_graphics)
    table.add_row(
        "Graphics",
        f"{backend} — {reason}" + ("" if backend != "off" else " (Braille cells)"),
    )
    console.print(table)


def _print_table_counts(conn: sqlite3.Connection) -> None:
    counts = database.table_counts(conn)
    table = Table(title=f"Database (schema v{schema_version(conn)})", box=box.SIMPLE_HEAVY)
    table.add_column("Table")
    table.add_column("Rows", justify="right")
    for name, count in counts.items():
        table.add_row(name, f"{count:,}")
    console.print(table)


def _print_discovered_sources(cfg: AppConfig) -> None:
    table = Table(title="Sources detected now", box=box.SIMPLE_HEAVY)
    table.add_column("source")
    table.add_column("status")
    table.add_column("location", overflow="fold")
    table.add_column("notes", overflow="fold")
    for adapter in registered_adapters():
        if adapter.agent not in cfg.enabled_harnesses:
            table.add_row(adapter.source_id, "[dim]disabled[/dim]", "", "")
            continue
        override = cfg.source_overrides.get(str(adapter.agent))
        source = adapter.discover(override=override)
        status = "[green]found[/green]" if source.exists else "[yellow]missing[/yellow]"
        table.add_row(adapter.source_id, status, source.location, source.note or "")
    console.print(table)


def _print_sources(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """
        SELECT id, agent, kind, location, last_scan_status, last_scan_at_ms, last_scan_note
        FROM sources ORDER BY agent, id
        """
    ).fetchall()
    if not rows:
        console.print("No sources registered yet.")
        return
    table = Table(title="Ingested sources", box=box.SIMPLE_HEAVY)
    table.add_column("source")
    table.add_column("harness")
    table.add_column("location", overflow="fold")
    table.add_column("last scan")
    table.add_column("status")
    table.add_column("notes", overflow="fold")
    for row in rows:
        table.add_row(
            row["id"],
            row["agent"],
            row["location"],
            tables.humanize_epoch_datetime(row["last_scan_at_ms"]),
            row["last_scan_status"] or "never",
            row["last_scan_note"] or "",
        )
    console.print(table)
    needs_rescan = [
        str(row["id"])
        for row in rows
        if str(row["last_scan_status"] or "") in ("partial", "error")
    ]
    if needs_rescan:
        console.print(
            "[dim]Hint: the last scan was partial or failed for "
            + ", ".join(needs_rescan)
            + " — after fixing the source, run [bold]toknt scan --full[/bold] to rebuild "
            "it from scratch.[/dim]"
        )


def _print_quality_issues(conn: sqlite3.Connection) -> None:
    issues = quality_issue_summary(conn)
    if not issues:
        return
    table = Table(title="Data-quality issues", box=box.SIMPLE_HEAVY)
    table.add_column("Harness")
    table.add_column("Code")
    table.add_column("Severity")
    table.add_column("Occurrences", justify="right")
    table.add_column("Message", overflow="fold")
    table.add_column("Sample", overflow="fold")
    for issue in issues:
        table.add_row(
            str(issue["agent"]),
            str(issue["code"]),
            str(issue["severity"]),
            f"{issue['occurrences']:,}",
            str(issue["message"]),
            str(issue["sample_ref"] or ""),
        )
    console.print(table)
    hint_codes = ("unparsable_row", "unparsable_line", "scan_failed")
    if any(str(issue["code"]) in hint_codes for issue in issues):
        console.print(
            "[dim]Hint: after fixing or updating a source, "
            "run [bold]toknt scan --full[/bold] to reindex it.[/dim]"
        )


def _print_pricing_hint(conn: sqlite3.Connection) -> None:
    try:
        count = conn.execute("SELECT COUNT(*) FROM pricing").fetchone()[0]
    except sqlite3.OperationalError:
        count = 0
    if count:
        console.print(
            f"[dim]{count} price row(s) configured; estimated costs use them and config "
            f"overrides. Reported costs (from sources) are always kept separate.[/dim]"
        )
    else:
        console.print(
            "[dim]No model prices configured: estimated costs will be unknown until you "
            "add them with [bold]toknt pricing add[/bold]. Reported costs from sources are "
            "shown regardless.[/dim]"
        )


def main() -> None:
    """Console-script entry point (pyproject: toknt = "toknt.cli:main")."""
    app()
