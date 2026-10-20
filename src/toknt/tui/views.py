"""Interactive views: shared state, chart pipeline, painting per view.

Views are pure-ish painters: given the terminal Painter, an :class:`AppState`
and an :class:`~toknt.tui.data.Analysis`, they redraw the whole screen.
Charts try the bitmap path first (producing an :class:`ImageRequest` the app
emits between text frames) and fall back to Braille/block cell rendering;
labels under charts always render as real terminal text.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from toknt.analytics import aggregation
from toknt.tui import widgets
from toknt.tui.data import (
    DEFAULT_WINDOW,
    HARNESS_NAMES,
    HOUR_BUCKET_MAX_DAYS,
    WINDOWS,
    Analysis,
    HarnessToggles,
)
from toknt.tui.painter import BOLD, DEFAULT_ATTRS, Attr, Painter
from toknt.tui.style import SERIES_HEX, heat_ramp

DAY_MS = 86_400_000

VIEW_TITLES = ("contributions", "models", "tools", "sessions")
METRICS = ("total", "input", "output", "reasoning", "cached")

#: Short column labels so a table can follow the selected metric.
METRIC_COLUMN_LABELS = {
    "total": "tokens",
    "input": "input",
    "output": "output",
    "reasoning": "reason",
    "cached": "cached",
}

HINT_PAIRS = (
    ("1-4", "view"),
    ("t", "window"),
    ("v", "metric"),
    ("m", "merge"),
    ("?", "help"),
    ("h", "isolate"),
    ("o/x/a", "harness"),
    ("p", "tok/s"),
    ("r", "rescan"),
    ("↑↓←→", "inspect"),
)  # "q quit" is pinned to the right edge by paint_hints; pairs are ordered
# most-essential-first so narrow terminals keep the important ones

TITLE_ATTR = Attr(None, None, BOLD)
ACCENT_ATTR = Attr(None, None, BOLD)
BRAND_ATTR = Attr(None, None, BOLD)
KEY_ATTR = Attr(None, None, BOLD)


@dataclass
class AppState:
    """Everything the current screen depends on."""

    view: int = 0                       # index into VIEW_TITLES
    window: int = DEFAULT_WINDOW         # index into WINDOWS (default 90d)
    metric: int = 0                     # index into METRICS
    agents: HarnessToggles = HarnessToggles()
    merge_models: bool = False          # merge one model served by many providers
    rate_mode: bool = False             # models chart: tokens/s instead of tokens
    cursor_day: int | None = None       # contributions inspect day
    session_row: int = 0
    tool_row: int = 0
    status: str = ""
    help: bool = False
    scanning: bool = False


@dataclass(frozen=True, slots=True)
class ImageRequest:
    """A chart the app should emit via the terminal's image protocol."""

    row: int
    col: int
    columns: int
    rows: int
    payload: bytes
    key: str                            # identity for skip-if-unchanged
    image_id: int = 1                   # stable per-slot (chart=1, donut=2)


@dataclass(frozen=True, slots=True)
class TerminalContext:
    """Snapshot of the image-protocol state the views render for."""

    backend: str = "off"                # kitty | iterm2 | sixel | off
    cell_w: int = 10
    cell_h: int = 20
    theme: str = "dark"                 # terminal background, "dark"|"light"
    bg_rgb: int | None = None           # exact OSC 11 answer, for erasing art


@dataclass(frozen=True, slots=True)
class ViewPaint:
    images: tuple[ImageRequest, ...] = ()


# ------------------------------------------------------------- state utils


def since_for(state: AppState, now_ms: int) -> int:
    window = WINDOWS[state.window]
    if window.days is None:
        return 0
    return now_ms - window.days * DAY_MS


def current_window_label(state: AppState) -> str:
    return WINDOWS[state.window].label


def current_bucket(state: AppState) -> str:
    """Chart bucket for the window: hours up close, days further out."""
    window = WINDOWS[state.window]
    if window.days is not None and window.days <= HOUR_BUCKET_MAX_DAYS:
        return "hour"
    return "day"


def current_metric(state: AppState) -> str:
    return METRICS[state.metric]


def metric_total(series: aggregation.Series) -> float:
    values = [value for values in series.series.values() for value in values if value is not None]
    return float(max(values)) if values else 0.0


def harnesses_line(state: AppState) -> str:
    if len(state.agents.active) == len(HARNESS_NAMES):
        return "harnesses: all"
    shown = ", ".join(sorted(state.agents.active)) or "none"
    if len(shown) > 28:
        return f"harnesses: {len(state.agents.active)} of {len(HARNESS_NAMES)}"
    return f"harnesses: {shown}"


#: Historical name (the concept was renamed from agents to harnesses).
agents_line = harnesses_line


def title_row(
    painter: Painter,
    state: AppState,
    *,
    rows: int,
    columns: int,
) -> None:
    """Top chrome: toknt brand, view tabs, window/metric/agent chips."""
    left, width = page_frame(columns)
    painter.text(0, left, "toknt", BRAND_ATTR)
    x = left + 6
    for index, name in enumerate(VIEW_TITLES):
        if index > 0:
            painter.text(0, x, "·", widgets.MUTED)
            x += 2
        label = f"{index + 1} {name}"
        if index == state.view:
            painter.text(0, x, label, Attr(SERIES_HEX[index % len(SERIES_HEX)], None, BOLD))
        else:
            painter.text(0, x, label, widgets.MUTED)
        x += len(label) + 2
    x += 1
    chips = (
        current_window_label(state),
        current_metric(state),
        harnesses_line(state),
    )
    if state.merge_models:
        chips = chips + ("merged",)
    for chip in chips:
        if x + len(chip) + 2 < left + width - 12:
            painter.text(0, x, "·", widgets.MUTED)
            x += 2
            painter.text(0, x, chip, TITLE_ATTR)
            x += len(chip) + 2
    if state.scanning:
        note = "‣ scanning…"
        painter.text(0, max(left + width - len(note) - 1, x), note, widgets.MUTED)
    elif state.status:
        # a one-line note at the page's right edge: scan results, the
        # selftest, off-view key feedback. Clipped to the room the chips
        # left, with an ellipsis when it does not fit — never allowed to
        # reach back into the chips.
        room = max(left + width - x - 3, 0)
        if room >= 4:
            note = state.status if len(state.status) <= room else state.status[: room - 1] + "…"
            painter.text(0, left + width - len(note) - 1, note, widgets.MUTED)
    paint_hints(painter, rows - 1, columns)


def paint_hints(painter: Painter, row: int, columns: int) -> None:
    """Bottom bar: bold keys + dim actions, no full-width highlight strip.

    ``q quit`` is pinned to the right edge so the quit key stays visible on
    narrow terminals; the remaining pairs fill left-to-right and drop from
    the middle inward when they no longer fit.
    """
    if row <= 0 or columns <= 4:
        return
    left, width = page_frame(columns)
    quit_key, quit_action = "q", "quit"
    quit_width = len(quit_key) + 1 + len(quit_action)
    quit_x = max(left + width - quit_width - 1, left)
    painter.text(row, quit_x, quit_key, KEY_ATTR)
    painter.text(row, quit_x + len(quit_key) + 1, quit_action, widgets.MUTED)
    x = left
    for key, action in HINT_PAIRS:
        if x + len(key) + 1 + len(action) > quit_x - 2:
            break
        painter.text(row, x, key, KEY_ATTR)
        x += len(key) + 1
        painter.text(row, x, action, widgets.MUTED)
        x += len(action) + 2


# ----------------------------------------------------------------- charts

_CHART_CACHE: dict[tuple, ImageRequest] = {}
_CHART_CACHE_LIMIT = 6


def _fingerprints(series: aggregation.Series) -> tuple:
    """Cheap equality token for the series data (C-speed tuple hashing)."""
    return (
        series.buckets,
        tuple(tuple(values) for values in series.series.values()),
    )


def paint_chart(
    painter: Painter,
    *,
    row: int,
    col: int,
    width_columns: int,
    height_rows: int,
    series: aggregation.Series,
    ctx: TerminalContext,
    identity: tuple = (),
) -> ImageRequest | None:
    """Bitmap chart when possible, Braille lines otherwise + text tick labels.

    The returned request, when not ``None``, must be emitted by the app right
    after the text frame flushes (the payload carries its own positioning).
    Identical charts are produced through a small render cache: the same
    view/window/metric/agents/geometry round-trips (arrow keys, rescan ticks)
    reuse the previous bitmap instead of rasterizing every keystroke.
    """
    if width_columns <= 8 or height_rows <= 4:
        return None
    if series.is_empty:
        painter.text(row, col, "no data for this window", DEFAULT_ATTRS)
        return None

    values: list[tuple[str, list[float | None]]] = [
        (name, [float(v) if v is not None else None for v in points])
        for name, points in series.series.items()
    ]
    plot_rows = height_rows - 1  # keep one row for tick labels

    if ctx.backend in ("kitty", "iterm2", "sixel") and plot_rows >= 6:
        pixel_w = width_columns * ctx.cell_w
        pixel_h = plot_rows * ctx.cell_h
        # A chart is drawn cell-sized; a bogus cell metric must not turn it
        # into a gigapixel image (the rasterizer is pure Python and would
        # freeze the UI for minutes).
        max_w, max_h = 4000, 2000
        if pixel_w > max_w or pixel_h > max_h:
            scale = min(max_w / pixel_w, max_h / pixel_h)
            pixel_w = max(int(pixel_w * scale), 1)
            pixel_h = max(int(pixel_h * scale), 1)
        cache_key = (
            identity,
            ctx.backend,
            ctx.cell_w,
            ctx.cell_h,
            pixel_w,
            pixel_h,
            _fingerprints(series),
        )
        request = _CHART_CACHE.get(cache_key)
        if request is None:
            png, rgba = _chart_png(pixel_w, pixel_h, values)
            request = ImageRequest(
                row=row,
                col=col,
                columns=width_columns,
                rows=plot_rows,
                payload=graphics_encode(
                    ctx.backend, png, rgba, pixel_w, pixel_h,
                    columns=width_columns, rows=plot_rows,
                ),
                key=f"{ctx.backend}/{pixel_w}x{pixel_h}/{abs(hash(cache_key)):#x}",
            )
            _CHART_CACHE[cache_key] = request
            while len(_CHART_CACHE) > _CHART_CACHE_LIMIT:
                _CHART_CACHE.pop(next(iter(_CHART_CACHE)))
        return _finish_chart(
            painter, row, col, width_columns, height_rows, series, request
        )

    widgets.draw_braille_lines(
        painter, row, col, width_columns, plot_rows, values, metric_total(series)
    )
    return _finish_chart(
        painter, row, col, width_columns, height_rows, series, None
    )


def _finish_chart(
    painter: Painter,
    row: int,
    col: int,
    width_columns: int,
    height_rows: int,
    series: aggregation.Series,
    request: ImageRequest | None,
) -> ImageRequest | None:
    tz = aggregation.default_timezone()
    for tick_col, text in chart_ticks(series.buckets, width_columns, tz):
        painter.text(row + height_rows - 1, col + tick_col, text, widgets.MUTED)
    _paint_legend(painter, row + height_rows, col, series, width=width_columns)
    return request


def _paint_legend(
    painter: Painter, row: int, col: int, series: aggregation.Series, *, width: int
) -> None:
    """Series swatches, clipped to the chart pane so side panes stay clean."""
    x = col
    limit = col + width
    for index, (name, _values) in enumerate(series.series.items()):
        if x + len(name[:11]) + 2 > limit:
            remaining = len(series.series) - index
            if remaining and x + 6 <= limit:
                painter.text(row, x, f"+{remaining} more", widgets.MUTED)
            break
        widgets.agent_swatch(painter, row, x, index, name[:11])
        x += 14


def chart_ticks(
    buckets: tuple[int, ...],
    columns: int,
    tz: object,
) -> list[tuple[int, str]]:
    """(col_offset, label) pairs evenly spread over the chart width."""
    if not buckets:
        return []
    wanted = max(2, min(6, columns // 12, len(buckets)))
    span_days = (buckets[-1] - buckets[0]) / DAY_MS if len(buckets) > 1 else 0
    if span_days > 18 * 30:
        fmt = "%b '%y"  # multi-year: month + year keeps chronology unambiguous
    elif span_days < 1.5:
        fmt = "%H:%M"   # sub-day windows: clock times, not four identical dates
    else:
        fmt = "%d %b"
    ticks: list[tuple[int, str]] = []
    used_end: int | None = None
    for index in range(wanted):
        position = round(index * (len(buckets) - 1) / max(wanted - 1, 1))
        col = round(index * (columns - 6) / max(wanted - 1, 1))
        moment = datetime.fromtimestamp(buckets[position] / 1000, tz=tz)
        label = moment.strftime(fmt)
        if used_end is not None and col <= used_end + len(
            ticks[-1][1] if ticks else ""
        ):
            continue
        ticks.append((col, label))
        used_end = col
    return ticks


# ------------------------------------------------------------- page layout

#: Columns of breathing room at each side of the page.
PAGE_MARGIN = 2

#: The page never stretches past this many columns: past that the layout
#: would have to stretch rows to fill it, and the side margins stay even.
#: Generous enough that an ultrawide monitor still feels used.
PAGE_MAX_WIDTH = 192


def page_frame(columns: int) -> tuple[int, int]:
    """The page block inside the terminal: (left column, width).

    On a narrow terminal the page simply fills the width; on a wide one it
    keeps a comfortable width and centers itself, so the margins on the
    left and the right always match.
    """
    width = max(min(columns - 2 * PAGE_MARGIN, PAGE_MAX_WIDTH), 40)
    left = max((columns - width) // 2, 0)
    return left, width


def paint_heading(painter: Painter, columns: int, text: str) -> None:
    """Panel chrome: bold heading one row under the tab bar, hairline rule."""
    left, width = page_frame(columns)
    painter.text(1, left, text[:width], TITLE_ATTR)
    painter.text(2, left, "─" * max(width, 0), widgets.MUTED)


def zones(columns: int, *, want_rail: bool = True) -> tuple[int, int, int]:
    """Split the page: (main_start_col, rail_col, rail_width).

    Wide layouts use a golden-ish two-zone split — the rail takes ~37% of
    the page (clamped 28–42 columns) with a 3-column gutter; narrower
    terminals fold back to a single full-width column. ``want_rail=False``
    forces the stacked layout.
    """
    left, body = page_frame(columns)
    # the wide layout is decided by the terminal, the split by the page
    if not want_rail or columns < 100:
        return left, None, 0
    rail_width = max(28, min(42, round(body * 0.37)))
    main_width = body - rail_width - 3
    if main_width < 40:  # degenerate split: keep everything stacked
        return left, None, 0
    rail_col = left + main_width + 3
    return left, rail_col, rail_width


def chart_budget(
    rows: int, *, min_plot: int = 5, max_plot: int = 15, page_width: int = 0
) -> int:
    """The line chart's image height: ~45% of the content rows, clamped.

    A wide page may afford a taller chart: without that, a 150-column
    chart would be a flat ribbon.
    """
    content_rows = max(rows - 4, 4)
    ceiling = max(max_plot, min(26, page_width // 10))
    return max(min_plot, min(ceiling, content_rows * 45 // 100))


# ----------------------------------------------------------------- views


def grid_weeks(state: AppState, columns: int, now_ms: int, tz: object) -> list[int]:
    """The week columns the contribution grid shows for the current window.

    Shared by the view and the cursor logic so the day the user can walk to
    is always a day the grid actually paints.
    """
    window = WINDOWS[state.window]
    max_weeks = max(4, (columns - 4) // 3)
    wanted = 26 if window.days is None else int(max(1, window.days // 7))
    return widgets.week_starts(now_ms, min(wanted, max_weeks), tz)


def paint_contribution(
    painter: Painter,
    state: AppState,
    analysis: Analysis,
    *,
    rows: int,
    columns: int,
    ctx: TerminalContext,
    now_ms: int,
) -> ViewPaint:
    tz = analysis.tz
    since_ms = since_for(state, now_ms)
    daily = analysis.daily(since_ms, state.agents)
    window = WINDOWS[state.window]
    heading = "contributions — tokens per day"
    if window.days is None:
        heading += " (recent half-year)"
    paint_heading(painter, columns, heading)

    weeks = grid_weeks(state, columns, now_ms, tz)
    grid_width = widgets.heatmap_width(weeks)
    page_col, body = page_frame(columns)
    wide = columns >= 100
    if wide:
        # The grid hugs its weekday labels, so the zone that holds it is
        # as wide as the grid needs; everything else lives in the rail.
        left_width = max(grid_width + 2, 40)
        rail_col = page_col + left_width + 3
        rail_width = max(body - left_width - 3, 30)
    else:
        left_width = body
        rail_col = None
        rail_width = 0

    ramp = heat_ramp(ctx.theme)
    if state.cursor_day is None and daily.total:
        state.cursor_day = daily.days[-1]
    # pad every day the window covers with its (possibly zero) total so the
    # grid is a solid rectangle: empty days get the ramp's floor color
    # instead of leaving holes that read as "the grid doesn't fill the
    # zone". Days outside the window (the rest of the current week) stay
    # unpainted, so they are not mistaken for quiet days.
    grid_totals: dict[int, int] = {}
    for week in weeks:
        for slot in range(7):
            day_ms = week + slot * widgets.DAY_MS
            if day_ms > now_ms:
                continue
            if since_ms and day_ms + widgets.DAY_MS <= since_ms:
                continue
            grid_totals[day_ms] = daily.total.get(day_ms, 0)
    widgets.draw_heatmap(
        painter,
        3,
        page_col,
        weeks=weeks,
        totals=grid_totals,
        ramp=ramp,
        tz=tz,
        cursor=state.cursor_day,
        width=left_width,
    )
    per_harness: dict[str, int] = {}
    present: set[str] = set()
    for per_day in daily.per_agent.values():
        for agent, amount in per_day.items():
            if amount:
                present.add(agent)
                per_harness[agent] = per_harness.get(agent, 0) + amount
    total = sum(daily.total.values())
    days_active = len(daily.days)

    if wide:
        # Wide layout: the grid and the days it inspects sit together on
        # the left; the rail carries the totals and the share lists.
        _paint_day_cards(
            painter,
            state,
            daily,
            row=3 + widgets.HEAT_ROWS + 1,
            col=page_col,
            width=left_width,
            tz=tz,
            bottom=rows - 2,
        )
        projects = {
            row.key: (row.summary.total or 0)
            for row in analysis.project_rows(since_ms, state.agents)
            if row.summary.total
        }
        _paint_contribution_rail(
            painter,
            rail_col,
            rail_width,
            state=state,
            per_harness=per_harness,
            projects=projects,
            total=total,
            days_active=days_active,
            rows=rows,
        )
        return ViewPaint()

    # Stacked layout: everything flows in one column with breathing gaps.
    entries, _idle = _harness_legend_entries(state, present)
    _paint_contribution_lower(
        painter,
        state,
        daily,
        present,
        entries,
        _legend_line_count(entries, body),
        per_harness=per_harness,
        weeks=weeks,
        tz=tz,
        rows=rows,
        col=page_col,
        columns=body,
        total=total,
        days_active=days_active,
    )
    return ViewPaint()


def _paint_day_cards(
    painter: Painter,
    state: AppState,
    daily,
    *,
    row: int,
    col: int,
    width: int,
    tz: object,
    bottom: int,
) -> None:
    """The picked day and the peak day, under the grid they belong to."""
    cursor = row
    if state.cursor_day is not None and cursor + 1 <= bottom:
        day_total = daily.total.get(state.cursor_day, 0)
        moment = datetime.fromtimestamp(state.cursor_day / 1000, tz=tz)
        painter.text(cursor, col, "picked day", widgets.MUTED)
        painter.text(
            cursor + 1,
            col,
            f"{widgets.compact_number(day_total)} tokens",
            ACCENT_ATTR,
        )
        painter.text(
            cursor + 1,
            col + width - 12,
            moment.strftime("%a %d %b"),
            widgets.MUTED,
        )
        cursor += 2
        for agent, amount in sorted(
            daily.per_agent.get(state.cursor_day, {}).items(), key=lambda kv: -kv[1]
        ):
            if cursor > bottom:
                break
            index = HARNESS_NAMES.index(agent) if agent in HARNESS_NAMES else 0
            widgets.agent_swatch(painter, cursor, col, index, agent[:11])
            painter.text(
                cursor,
                col + width - 9,
                widgets.compact_number(amount).rjust(9),
                widgets.MUTED,
            )
            cursor += 1
        cursor += 1
    peak_day = max(daily.total.items(), key=lambda kv: kv[1], default=None)
    if peak_day and cursor + 2 <= bottom:
        moment = datetime.fromtimestamp(peak_day[0] / 1000, tz=tz)
        painter.text(cursor, col, "peak day", widgets.MUTED)
        painter.text(
            cursor + 1,
            col,
            f"{widgets.compact_number(peak_day[1])} tokens",
            ACCENT_ATTR,
        )
        painter.text(cursor + 2, col, moment.strftime("%a %d %b"), widgets.MUTED)


def _project_label(path: str) -> str:
    """The tail of a project path: '…/Project/kbd'.

    Two projects rarely differ in their first segments, so the head of a
    path says the same thing on every row — the tail is what tells them
    apart.
    """
    parts = [part for part in path.split("/") if part]
    if not parts:
        return path
    tail = "/".join(parts[-2:])
    return f"\u2026/{tail}" if len(parts) > 2 else tail


def _paint_contribution_rail(
    painter: Painter,
    col: int,
    width: int,
    *,
    state: AppState,
    per_harness: dict[str, int],
    projects: dict[str, int],
    total: int,
    days_active: int,
    rows: int,
) -> None:
    """Right zone of the contributions view.

    Stack: window total, the share by project, the share by harness, and a
    quiet note about toggled-off harnesses — the rail is the state panel,
    the left zone is the calendar and the days it inspects.
    """
    bottom = rows - 2
    cursor = 3
    painter.text(cursor, col, "window total", widgets.MUTED)
    painter.text(
        cursor + 1, col, f"{widgets.compact_number(total)} tokens", ACCENT_ATTR
    )
    avg = total // max(days_active, 1)
    painter.text(
        cursor + 2,
        col,
        f"{days_active} active days · avg {widgets.compact_number(avg)}/day"[: width - 2],
        widgets.MUTED,
    )
    cursor += 4
    # a wide rail splits into two columns: the projects on the left, the
    # harnesses on the right. Extra width becomes more readable rows
    # instead of longer empty ones.
    two_columns = width >= 96
    project_col = col
    project_width = width - (width // 2 + 3) if two_columns else width
    harness_col = col + width - width // 2 if two_columns else col
    harness_width = width // 2 if two_columns else width

    def paint_list(
        heading: str,
        entries: dict[str, int],
        list_col: int,
        list_width: int,
        index_of,
        name_width: int | None,
        label=None,
    ) -> int:
        painter.text(cursor, list_col, heading, widgets.MUTED)
        painted = widgets.draw_share_rows(
            painter,
            cursor + 1,
            list_col,
            per_harness=entries,
            total=total,
            width=list_width,
            max_rows=min(10, bottom - cursor - 1),
            index_of=index_of,
            name_width=name_width,
            label=label,
        )
        return painted

    if projects and cursor <= bottom - 1:
        painted = paint_list(
            "share by project",
            {path: amount for path, amount in projects.items()},
            project_col,
            project_width,
            None,
            None,
            label=_project_label,
        )
        if not two_columns:  # side by side, the columns share one band
            cursor += 1 + painted + 1
    if per_harness and cursor <= bottom - 1:
        painted = paint_list(
            "share by harness",
            per_harness,
            harness_col,
            harness_width,
            lambda name: (
                HARNESS_NAMES.index(name) if name in HARNESS_NAMES else 0
            ),
            14 if two_columns else None,
        )
        if not two_columns:
            cursor += 1 + painted + 1
    # keep the toggle state visible on wide screens too (the old legend
    # carried it; here it is one quiet line)
    off = [name for name in HARNESS_NAMES if name not in state.agents.active]
    if off and cursor <= bottom:
        if len(off) <= 3 and sum(len(name) for name in off) + 10 <= width:
            note = "· " + ", ".join(off) + " (off)"
        else:
            note = f"· {len(off)} harnesses off (o/x/a)"
        painter.text(cursor, col, note[: width - 2], widgets.MUTED)


def _harness_legend_entries(
    state: AppState, present: set[str]
) -> tuple[list[tuple[int, str, int, bool]], int]:
    """Compact legend entries as (series_index, text, width, is_swatch).

    Harnesses with data get colored swatches; toggled-off ones stay visible
    as muted chips; active-but-idle harnesses collapse into one quiet tail
    chip — the old legend spent four rows announcing 18 "(no data)" names.
    Returns (entries, idle_count).
    """
    entries: list[tuple[int, str, int, bool]] = []
    idle = 0
    for index, name in enumerate(HARNESS_NAMES):
        active = name in state.agents.active
        if active and name in present:
            entries.append((index, name, len(name) + 3, True))
        elif not active:
            entries.append((index, f"\u00b7 {name} (off)", len(name) + 8, False))
        else:
            idle += 1
    if idle:
        entries.append((-1, f"\u00b7 +{idle} idle", 11, False))
    return entries, idle


def _legend_line_count(entries: list[tuple[int, str, int, bool]], columns: int) -> int:
    """How many wrapped rows the legend needs (pure; mirrors the painter)."""
    if columns < 12:
        return 1
    x, lines = 2, 1
    max_x = columns - 2
    for _index, _text, width, _swatch in entries:
        if x + width > max_x and x > 2:
            lines += 1
            x = 2
        x += width
    return lines


def _paint_legend_entries(
    painter: Painter,
    row: int,
    entries: list[tuple[int, str, int, bool]],
    col: int,
    columns: int,
) -> None:
    x = col
    line = 0
    max_x = columns - 2
    for index, text, width, swatch in entries:
        if x + width > max_x and x > col:
            line += 1
            x = col
        if swatch:
            widgets.agent_swatch(painter, row + line, x, index, text)
        else:
            painter.text(row + line, x, text, widgets.MUTED)
        x += width


def _paint_contribution_lower(
    painter: Painter,
    state: AppState,
    daily,
    present: set[str],
    entries: list[tuple[int, str, int, bool]],
    legend_lines: int,
    *,
    per_harness: dict[str, int],
    weeks: list[int],
    tz: object,
    rows: int,
    col: int,
    columns: int,
    total: int,
    days_active: int,
) -> None:
    """Everything below the heat grid, laid out to the available height.

    Tiers, most complete first: detail + total + full legend; detail + full
    legend; full legend alone; detail + one-line summary; summary alone.
    The share strip paints only when the tier above left room, so
    nothing collides with the hint bar or disappears off the bottom.
    """
    free_top = 3 + widgets.HEAT_ROWS      # first row below the grid block
    bottom = rows - 2                     # last row above the hint bar
    detail_row = free_top + 1             # one breathing row after the key

    def paint_detail_and_total(row: int) -> None:
        _paint_detail_line(painter, state, daily, row, col=col, columns=columns, tz=tz)
        painter.text(
            row + 1,
            col,
            f"window total {widgets.compact_number(total)} tokens"
            f" across {days_active} active days",
            widgets.MUTED,
        )

    if detail_row > bottom:
        # the grid fills the terminal: a one-line summary takes the last row
        if free_top <= bottom:
            _paint_summary_line(painter, free_top, present, col=col, columns=columns)
        return

    if detail_row + 1 + legend_lines <= bottom:
        # roomy: detail, total, full legend (legend right after the total)
        paint_detail_and_total(detail_row)
        _paint_legend_entries(painter, detail_row + 3, entries, col, columns)
        legend_bottom = detail_row + 2 + legend_lines
    elif detail_row + legend_lines <= bottom:
        # tight: detail, full legend (no total, no gap)
        _paint_detail_line(
            painter, state, daily, detail_row, col=col, columns=columns, tz=tz
        )
        _paint_legend_entries(painter, detail_row + 2, entries, col, columns)
        legend_bottom = detail_row + 1 + legend_lines
    elif free_top + legend_lines - 1 <= bottom:
        # legend-only: the full legend replaces the detail block
        _paint_legend_entries(painter, free_top, entries, col, columns)
        legend_bottom = free_top + legend_lines - 1
    else:
        # detail + one-line summary (the full legend cannot fit)
        _paint_detail_line(
            painter, state, daily, detail_row, col=col, columns=columns, tz=tz
        )
        _paint_summary_line(painter, detail_row + 1, present, col=col, columns=columns)
        legend_bottom = detail_row + 1

    # Share strip: window tokens split by harness with inline bars.
    share_row = legend_bottom + 2
    if share_row <= bottom - 1 and per_harness:
        _paint_share_strip(
            painter, share_row, per_harness, total, state, col=col, columns=columns
        )


def _paint_detail_line(
    painter: Painter,
    state: AppState,
    daily,
    row: int,
    *,
    col: int,
    columns: int,
    tz: object,
) -> None:
    if state.cursor_day is None:
        return
    moment = datetime.fromtimestamp(state.cursor_day / 1000, tz=tz)
    day_total = daily.total.get(state.cursor_day, 0)
    if day_total:
        parts = [
            moment.strftime("%a %d %b %Y"),
            f"{widgets.compact_number(day_total)} tokens",
        ]
        for agent, amount in sorted(daily.per_agent.get(state.cursor_day, {}).items()):
            parts.append(f"{agent} {widgets.compact_number(amount)}")
    else:
        parts = [moment.strftime("%a %d %b %Y"), "no tokens"]
    painter.text(row, col, " \u00b7 ".join(parts)[:columns], ACCENT_ATTR)


def _paint_summary_line(
    painter: Painter, row: int, present: set[str], *, col: int, columns: int
) -> None:
    """One-line stand-in for the full legend at very small heights."""
    with_data = [name for name in HARNESS_NAMES if name in present]
    x = col
    shown = 0
    for name in with_data[:3]:
        index = HARNESS_NAMES.index(name)
        if x + len(name) + 3 > columns - 22:
            break
        widgets.agent_swatch(painter, row, x, index, name)
        x += len(name) + 3
        shown += 1
    remaining = len(with_data) - shown
    note = f"+{remaining} more" if remaining > 0 else "no other data"
    if x < columns - 2:
        painter.text(row, x, f"\u00b7 {note} (o/x/a/h)"[: columns - x - 1], widgets.MUTED)


def _paint_share_strip(
    painter: Painter,
    row: int,
    per_harness: dict[str, int],
    total: int,
    state: AppState,
    *,
    col: int,
    columns: int,
) -> None:
    """One line: 'harness ████ 42%' per harness with tokens, wrapped."""
    painter.text(row, col, "share by harness", widgets.MUTED)
    x = col + 18  # past the label
    max_x = columns - 6
    for name, amount in sorted(per_harness.items(), key=lambda kv: -kv[1]):
        index = HARNESS_NAMES.index(name) if name in HARNESS_NAMES else 0
        share = amount / total if total else 0.0
        label = f"{name[:14]} {widgets.bar_glyphs(share, 10)} {share * 100:>3.0f}%"
        if x + len(label) > max_x:
            painter.text(row, min(x + 1, columns - 2), "…", widgets.MUTED)
            break
        color = SERIES_HEX[index % len(SERIES_HEX)]
        painter.text(row, x, label, Attr(color, None, 0))
        x += len(label) + 3


def paint_models(
    painter: Painter,
    state: AppState,
    analysis: Analysis,
    *,
    rows: int,
    columns: int,
    ctx: TerminalContext,
    now_ms: int,
) -> ViewPaint:
    since_ms = since_for(state, now_ms)
    metric = current_metric(state)
    token_series = analysis.model_series(
        since_ms,
        state.agents,
        metric,
        bucket=current_bucket(state),
        merge=state.merge_models,
    )
    if state.rate_mode:
        # the graph switches units; the table keeps the token breakdown
        series = analysis.model_rate_series(
            since_ms,
            state.agents,
            bucket=current_bucket(state),
            merge=state.merge_models,
            lineup=token_series.identities,
        )
        heading = "≈ tokens/s by model — output+reasoning over message gaps"
    else:
        series = token_series
        heading = f"tokens by model — {aggregation.metric_label(metric)}"
    if state.merge_models:
        heading += " (merged across providers)"
    paint_heading(painter, columns, heading)
    rates = analysis.model_rate_totals(since_ms, state.agents, merge=state.merge_models)
    breakdown = analysis.model_rows(
        since_ms, state.agents, merge=state.merge_models, metric=metric
    )
    cache_rates = analysis.cache_rates(since_ms, state.agents, merge=state.merge_models)
    shares = [
        (entry.key, float(entry.summary.total))
        for entry in breakdown[:6]
        if entry.summary.total
    ]
    main_col, rail_col, rail_width = zones(columns, want_rail=bool(shares))
    _page_col, page_width = page_frame(columns)
    main_width = page_width - (rail_width + 3 if rail_width else 0)

    plot_rows = chart_budget(rows, page_width=page_width)  # ~45% of the height
    image = paint_chart(
        painter,
        row=3,
        col=main_col,
        width_columns=main_width if rail_width else page_width,
        height_rows=plot_rows + 1,  # P image rows, then ticks, then legend
        series=series,
        ctx=ctx,
        identity=(
            state.window,
            state.metric,
            tuple(sorted(state.agents.active)),
            state.rate_mode,
        ),
    )
    pie_image = None
    # the table's extent decides how far the rail's cache list may run: the
    # two blocks end on the same row, so the page reads as one frame
    chart_bottom = 3 + plot_rows + 2 + (1 if rows >= 24 else 0)
    table_max_rows = max(rows - 2 - chart_bottom, 0)
    model_row_count = min(len(breakdown), 24, table_max_rows)
    table_last_row = chart_bottom + model_row_count
    if rail_width:
        pie_image = _paint_model_side(
            painter,
            rail_col,
            rail_width,
            shares,
            cache_rates,
            row=3,
            rows=rows,
            ctx=ctx,
            list_bottom=table_last_row,
            identity=(
                state.window,
                tuple(sorted(state.agents.active)),
                round(sum(value for _key, value in shares)),
            ),
        )
    table_width = main_width if rail_width else page_width
    include_share = False
    include_rate = False
    share_width = 0
    if columns >= 78:
        # full breakdown: model, tokens, events, sessions, cost (+ a
        # relative-magnitude bar when the zone has room to breathe)
        # a rate column only when the zone can afford a comfortable name
        include_rate = table_width >= 80 and any(
            value is not None for value in rates.values()
        )
        fixed = 10 + 8 + 9 + 8 + (7 if include_rate else 0)
        gutter = 2 * (6 if include_rate else 5)
        share_width = 9
        name_width = table_width - fixed - gutter - share_width
        if name_width < 16:  # no room for the bar: the name wins
            share_width = 0
            name_width = table_width - fixed - gutter
        name_width = max(min(name_width, 56), 14)
        slack = table_width - (name_width + fixed + gutter + share_width)
        share_width = min(36, share_width + max(slack, 0))
        include_share = share_width > 0
        columns_spec = [
            ("model", name_width, 0),
            (">" + METRIC_COLUMN_LABELS.get(metric, "tokens"), 10, 1),
            (">events", 8, 1),
            (">sessions", 9, 1),
            (">cost", 8, 1),
        ]
        if include_rate:
            # ≈ marks the estimate: a message's completion timestamp is
            # recorded, its duration is not
            columns_spec.append((">≈tok/s", 7, 1))
        columns_spec += [("vs max", share_width, 0)] if include_share else []
    else:
        # narrow: model, the metric, events (total ~54 columns)
        columns_spec = [
            ("model", min(max(table_width - 26, 10), 30), 0),
            (">" + METRIC_COLUMN_LABELS.get(metric, "tokens"), 10, 1),
            (">events", 8, 1),
        ]
    peak_tokens = max(
        (entry.summary.total or 0 for entry in breakdown[:24]), default=0
    )
    table_rows: list[list[str]] = []
    for entry in breakdown[:24]:
        summary = entry.summary
        cells = [
            entry.key[:30],
            widgets.compact_number(summary.total),
        ]
        if columns >= 78:
            cells.extend(
                [
                    str(summary.events),
                    str(summary.sessions),
                    _cost_text(summary.cost_reported),
                ]
            )
            if include_rate:
                rate = rates.get(entry.identity)
                cells.append(f"{rate:.1f}" if rate is not None else "—")
            if include_share:
                cells.append(
                    widgets.bar_glyphs(
                        (summary.total or 0) / peak_tokens if peak_tokens else 0.0,
                        max(share_width - 2, 3),
                    )
                )
        else:
            cells.append(str(summary.events))
        table_rows.append(cells)
    widgets.draw_table(
        painter,
        chart_bottom,
        main_col,
        columns_spec,
        table_rows,
        selected=None,
        max_rows=rows - 2 - chart_bottom,
    )
    images = (image,) if image else ()
    return ViewPaint(images + ((pie_image,) if pie_image else ()))


def _paint_model_side(
    painter: Painter,
    col: int,
    width: int,
    shares: list[tuple[str, float]],
    cache_rates: list[aggregation.ModelCacheRate],
    *,
    row: int,
    rows: int,
    ctx: TerminalContext,
    identity: tuple,
    list_bottom: int | None = None,
) -> ImageRequest | None:
    """Right zone of the models view: donut, legend under it, cache bars.

    The blocks stack vertically with hard row guards, so the legend never
    lands on the donut and nothing spills into the hint bar; a breathing
    row separates each block. Terminals without image support get a small
    stacked-block share ribbon instead of the bitmap donut. ``list_bottom``
    is the row the model table ends on: the cache list stops there, so the
    two blocks finish together instead of running past each other.
    """
    bottom = rows - 2
    pie_image: ImageRequest | None = None
    cursor = row
    pie_rows = max(8, min(13, rows // 3))
    if ctx.backend in ("kitty", "iterm2", "sixel") and pie_rows >= 6 and rows >= 16:
        from toknt.presentation import graphics

        square_px = min((width - 1) * ctx.cell_w, pie_rows * ctx.cell_h)
        pie_cols = max(square_px // max(ctx.cell_w, 1), 6)
        pie_image_rows = max(square_px // max(ctx.cell_h, 1), 1)
        png, rgba = graphics.pie_png(
            width=square_px, height=square_px, weights=[value for _key, value in shares]
        )
        # left-aligned with the legend chips and cache bars below it:
        # a donut floating in the middle of the rail looks detached
        donut_col = col
        pie_image = ImageRequest(
            row=row,
            col=donut_col,
            columns=pie_cols,
            rows=pie_image_rows,
            payload=graphics_encode(
                ctx.backend, png, rgba, square_px, square_px,
                columns=pie_cols, rows=pie_image_rows, image_id=2,
            ),
            image_id=2,
            key=(
                f"pie/{square_px}/"
                f"{abs(hash(identity + tuple(round(value) for _key, value in shares))):#x}"
            ),
        )
        cursor = row + pie_image_rows + 1  # one breathing row under the donut
    elif shares and rows >= 14:
        # no image transport: a colored stacked ribbon still shows the mix
        total_shares = sum(value for _key, value in shares) or 1.0
        ribbon_width = max(width - 2, 8)
        for ribbon_row in range(row, min(row + 3, bottom + 1)):
            x = col
            for index, (_key, value) in enumerate(shares):
                segment = max(round(value / total_shares * ribbon_width), 1)
                color = Attr(SERIES_HEX[index % len(SERIES_HEX)], None, 0)
                painter.text(
                    ribbon_row, x, "█" * min(segment, ribbon_width - (x - col)), color
                )
                x += segment
                if x - col >= ribbon_width:
                    break
        cursor = row + 3
    # legend below the donut: chips are 15 columns wide (swatch + 12-char
    # name) so a 28–42-column zone fits two per row
    chip_width = 15
    per_row = max(width // chip_width, 1)
    legend_lines = 0
    for index, (key, _value) in enumerate(shares):
        slot = index % per_row
        line = index // per_row
        paint_row = cursor + line
        if paint_row > bottom:
            break
        widgets.agent_swatch(
            painter, paint_row, col + slot * chip_width, index, key[:12]
        )
        legend_lines = max(legend_lines, line + 1)
    cursor += legend_lines + 1  # blank before the cache block
    # cache-rate bars below the legend block
    if cursor <= bottom and cache_rates:
        limit = bottom if list_bottom is None else min(bottom, list_bottom)
        # the bars sit at cursor+1 .. cursor+budget, so a budget of
        # limit - cursor puts the last one exactly on `limit`
        budget = max(limit - cursor, 0)
        if budget > 0:
            painter.text(cursor, col, "cache hit rate", TITLE_ATTR)
            shown = 0
            for entry in cache_rates:
                if entry.rate is None or shown >= budget:
                    continue
                label = f"{entry.model[:12]:12}"
                bar = widgets.bar_glyphs(entry.rate, 8)
                painter.text(cursor + 1 + shown, col, label, widgets.MUTED)
                painter.text(
                    cursor + 1 + shown,
                    col + 13,
                    f"{bar}{entry.rate * 100:>4.0f}%",
                    DEFAULT_ATTRS,
                )
                shown += 1
            if shown == 0:
                painter.text(cursor + 1, col, "no cache data in window", widgets.MUTED)
    return pie_image



def paint_tools(
    painter: Painter,
    state: AppState,
    analysis: Analysis,
    *,
    rows: int,
    columns: int,
    ctx: TerminalContext,
    now_ms: int,
) -> ViewPaint:
    since_ms = since_for(state, now_ms)
    outcomes = analysis.tool_rows(since_ms, state.agents)
    paint_heading(painter, columns, "tool outcomes (known-result calls only get rates)")
    page_col, page_width = page_frame(columns)
    if page_width >= 112:
        tool_width = min(max(page_width - 100, 12), 40)
        # the outcome bar absorbs the ultrawide slack (capped, so bars stay
        # readable instead of becoming billboards)
        bar_width = min(28, max(12, page_width - tool_width - 84))
        columns_spec = [
            ("tool", tool_width, 0),
            ("category", 12, 0),
            (">calls", 8, 1),
            (">ok", 7, 1),
            (">fail", 7, 1),
            (">deny", 7, 1),
            (">cancel", 8, 1),
            (">unknown", 9, 1),
            (">rate", 8, 1),
            ("", bar_width, 0),
        ]
    else:
        # narrow terminals: outcome details fold into the rate column
        tool_width = min(max(page_width - 51, 10), 34)
        bar_width = 12
        columns_spec = [
            ("tool", tool_width, 0),
            ("category", 12, 0),
            (">calls", 8, 1),
            (">rate", 8, 1),
            ("", bar_width, 0),
        ]
    detailed = page_width >= 112
    body: list[list[str]] = []
    for outcome in outcomes[:64]:
        rate = outcome.success_rate
        cells = [
            outcome.tool[:tool_width],
            outcome.category[:12],
            str(outcome.calls),
        ]
        if detailed:
            cells.extend(
                [
                    str(outcome.success),
                    str(outcome.failure),
                    str(outcome.denied),
                    str(outcome.cancelled),
                    str(outcome.unknown),
                    f"{rate * 100:.0f}%" if rate is not None else "—",
                    widgets.bar_glyphs(rate, bar_width - 2),
                ]
            )
        else:
            cells.extend(
                [
                    f"{rate * 100:.0f}%" if rate is not None else "—",
                    widgets.bar_glyphs(rate, 10),
                ]
            )
        body.append(cells)
    offset = max(0, min(state.tool_row, len(body) - (rows - 4)))
    widgets.draw_table(
        painter, 3, page_col, columns_spec, body,
        selected=state.tool_row,
        max_rows=rows - 6,
        row_offset=offset,
    )
    return ViewPaint()


def paint_sessions(
    painter: Painter,
    state: AppState,
    analysis: Analysis,
    *,
    rows: int,
    columns: int,
    ctx: TerminalContext,
    now_ms: int,
) -> ViewPaint:
    since_ms = since_for(state, now_ms)
    metric = current_metric(state)
    ranked = analysis.session_rows(
        since_ms,
        state.agents,
        limit=200,
        merge=state.merge_models,
        metric=metric,
    )
    paint_heading(painter, columns, "sessions by tokens (arrows to inspect)")
    page_col, page_width = page_frame(columns)
    peak_tokens = max(
        (entry.summary.total or 0 for entry in ranked), default=0
    )
    if page_width >= 102:
        # the project column absorbs the page's slack so the table fills
        # the width instead of leaving a ragged hole on the right
        columns_spec = [
            ("when", 16, 0),
            ("harness", 12, 0),
            ("project", max(min(page_width - 80, 96), 8), 0),
            (">" + METRIC_COLUMN_LABELS.get(metric, "tokens"), 10, 1),
            ("vs max", 11, 0),
            (">events", 8, 1),
            (">tools", 7, 1),
        ]
        detailed = True
    elif page_width >= 70:
        # medium: drop the events/tools columns, keep the share bars
        columns_spec = [
            ("when", 16, 0),
            ("harness", 12, 0),
            ("project", max(min(page_width - 61, 56), 8), 0),
            (">" + METRIC_COLUMN_LABELS.get(metric, "tokens"), 10, 1),
            ("vs max", 11, 0),
        ]
        detailed = False
    else:
        # minimum width: when, harness, project, tokens
        columns_spec = [
            ("when", 16, 0),
            ("harness", 12, 0),
            ("project", max(min(page_width - 48, 44), 8), 0),
            (">" + METRIC_COLUMN_LABELS.get(metric, "tokens"), 10, 1),
        ]
        detailed = False
    body: list[list[str]] = []
    for entry in ranked:
        summary = entry.summary
        when = datetime.fromtimestamp(entry.last_ms / 1000, tz=analysis.tz)
        # bars read better scaled to the largest session: the top row is
        # full, everything else shows its magnitude against it
        magnitude = (
            summary.total / peak_tokens if peak_tokens and summary.total else None
        )
        project = (entry.project or "—")
        cells = [
            when.strftime("%d %b %H:%M"),
            entry.agent[:12],
            project[: columns_spec[2][1]] if len(project) > columns_spec[2][1] else project,
            widgets.compact_number(summary.total),
        ]
        if columns >= 70:
            cells.append(widgets.bar_glyphs(magnitude, 9))
        if detailed:
            cells.extend([str(summary.events), str(entry.tool_calls)])
        body.append(cells)
    offset = max(0, min(state.session_row, len(body) - (rows - 7)))
    widgets.draw_table(
        painter, 3, page_col, columns_spec, body,
        selected=state.session_row,
        max_rows=rows - 7,
        row_offset=offset,
    )
    selected = ranked[min(state.session_row, len(ranked) - 1)] if ranked else None
    if selected is not None:
        notes = [f"{selected.agent}", selected.project or "—"]
        if selected.models:
            notes.append(", ".join(name for name in selected.models[:3]))
        if selected.duration_ms:
            minutes = selected.duration_ms / 60000
            notes.append(f"{minutes:.0f} min")
        painter.text(rows - 2, page_col, " · ".join(notes)[:page_width], ACCENT_ATTR)
    return ViewPaint()


HELP_LINES = (
    "toknt interactive — keys",
    "",
    "1-4     contributions · models · tools · sessions",
    "t       window: 3h/5h/1d/3d/7d/30d/90d/1y/all",
    "        (short windows auto-refresh)",
    "v       metric: total/input/output/reasoning/…",
    "h       isolate one harness: all → each → all",
    "        (o/x/a toggle harnesses one by one)",
    "p       tok/s ⇄ tokens (models chart, ≈ estimated)",
    "m       merge models served by different providers",
    "        (glm-5.3-flash == z-ai/glm-5.3-flash)",
    "o/x/a   toggle opencode · codex · claude_code",
    "h       isolate one harness: all → each → all",
    "arrows  inspect day / pick table row",
    "r       rescan sources in the background",
    "g       graphics self-test (draws a test image)",
    "?       this help (chart stays put underneath)",
    "q       quit",
)


def paint_help(painter: Painter, painter_rows: int, painter_columns: int) -> None:
    """Overlay a centered key-reference card over whatever is on screen."""
    lines = list(HELP_LINES)
    available = max(painter_rows - 3, 3)
    if len(lines) > available:
        # a short terminal drops the indented examples first, then the
        # least-used entries: the card must never run off the screen
        compact = [line for line in lines if not line.startswith("        ")]
        lines = compact if len(compact) <= available else lines[:available]
    width = min(max(len(line) for line in lines) + 4, painter_columns - 4)
    height = len(lines) + 2
    top = max((painter_rows - height) // 2, 1)
    left = max((painter_columns - width) // 2, 1)
    for y, line in enumerate(lines):
        painter.clear(top + 1 + y, left, width, 1)
        attr = TITLE_ATTR if y == 0 else DEFAULT_ATTRS
        painter.text(top + 1 + y, left, line.ljust(width), attr)


def _cost_text(value: float | None) -> str:
    if value is None:
        return "—"
    return f"${value:.2f}"


def _chart_png(
    pixel_w: int, pixel_h: int, values: list[tuple[str, list[float | None]]]
):
    from toknt.presentation import graphics

    series = {name: point_list for name, point_list in values}
    return graphics.chart_png(width=pixel_w, height=pixel_h, kind="line", series=series)


def graphics_encode(
    backend: str,
    png: bytes,
    rgba: bytes,
    pixel_w: int,
    pixel_h: int,
    *,
    columns: int,
    rows: int,
    image_id: int = 1,
) -> bytes:
    from toknt.presentation import graphics

    return graphics.encode_placement(
        backend, png, rgba, pixel_w, pixel_h, columns=columns, rows=rows,
        image_id=image_id,
    )
