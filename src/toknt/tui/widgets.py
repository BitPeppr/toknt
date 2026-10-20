"""Cell widgets for the TUI: contributions grid, tables, chart fallbacks.

Everything here paints into :class:`~toknt.tui.painter.Painter` cells and is
the always-available rendering path. Chart widgets have two flavors:

* :func:`draw_braille_lines` renders multi-series lines with Braille dots
  (4 rows × 2 cols of sub-pixels per cell — sharp, colorful, portable);
* :func:`draw_stacked` renders stacked area/bars with the partial-block
  glyphs (▁▂▃▄▅▆▇█, 1/8-cell vertical resolution).

When the terminal also supports an image protocol the TUI swaps these for
anti-aliased PNG placements; these widgets stay independent of that path
and remain the fallback.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timedelta

from toknt.tui.painter import BOLD, DEFAULT_ATTRS, DIM, Attr, Painter
from toknt.tui.style import SERIES_HEX, rgb_luminance

DAY_MS = 86_400_000
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
MUTED = Attr(None, None, DIM)
LABEL = Attr(None, None, BOLD)
BOLD_ATTR = Attr(None, None, BOLD)

_BLOCK_GLYPHS = "▁▂▃▄▅▆▇█"

#: 1/8-cell partial blocks that fill from the left — the right glyphs for
#: horizontal bars (see :func:`bar_glyphs`).
_BAR_PARTIALS = "▏▎▍▌▋▊▉"
_HEAT_LEVEL_ENTRIES = ("less", "more")


def compact_number(value: int | float | None) -> str:
    """Axis-tag formatting: 1234 → '1.2K', 3.4e6 → '3.4M', None → '—'."""
    if value is None:
        return "—"
    number = abs(float(value))
    sign = "-" if value < 0 else ""
    for divisor, unit in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if number >= divisor:
            return f"{sign}{number / divisor:.1f}{unit}"
    if value == int(value):
        return f"{sign}{int(value)}"
    return f"{sign}{number:.1f}"


# ---------------------------------------------------------------- heat map


def week_starts(now_ms: int, count: int, tz: object) -> list[int]:
    """Ascending Monday bucket starts; the last one contains ``now_ms``.

    Snapped to local midnight exactly like ``aggregation.bucket_start_ms``,
    so heatmap cells line up with data bucket keys across DST shifts.
    """
    today = datetime.fromtimestamp(now_ms / 1000, tz=tz)
    monday = (today - timedelta(days=today.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    oldest = monday - timedelta(weeks=count - 1)
    return [
        int((oldest + timedelta(weeks=index)).timestamp() * 1000)
        for index in range(count)
    ]


def heat_levels(values: dict[int, int], *, levels: int = 5) -> dict[int, int]:
    """Map day totals to 0..levels-1 ramp levels with a sqrt curve."""
    peak = max(values.values(), default=0)
    out: dict[int, int] = {}
    if peak <= 0:
        return {key: 0 for key in values}
    for key, value in values.items():
        if value > 0:
            out[key] = min(int((value / peak) ** 0.5 * (levels - 1) + 0.5), levels - 1)
        else:
            out[key] = 0
    return out


HEAT_ROWS = 9  # month labels + 7 day rows + ramp key


def draw_heatmap(
    painter: Painter,
    row: int,
    col: int,
    *,
    weeks: list[int],
    totals: dict[int, int],
    ramp: tuple[int, ...],
    tz: object,
    cursor: int | None = None,
    width: int | None = None,
) -> tuple[int, int]:
    """GitHub-style contributions grid; returns (rows, cols) consumed.

    Rows are Mon..Sun top to bottom, columns are weeks. The viewport heading
    is painted by the caller; this block only draws month labels, the grid,
    and the ramp key. The cursor day is highlighted with the brightest ramp
    color (plus bold) instead of moving the terminal cursor, so rounding
    never steals focus.

    The grid hugs its weekday labels: week columns are two cells wide with a
    single space between them, and the block is never centered away from
    the labels (a calendar floating in the middle of the page reads as a
    chart, not as a calendar).
    """
    levels = heat_levels(totals)
    label_margin = 4
    # one space between week columns; drop it when the zone is too narrow
    gap = 1 if width is None or width >= label_margin + len(weeks) * 3 - 1 else 0
    stride = 2 + gap
    used_cols = label_margin + len(weeks) * 2 + (len(weeks) - 1) * gap

    # Month labels, centered over each month's span of week columns and
    # drawn only when the 3-character label fits without touching the
    # previous one — adjacent labels never overlap ("SeOct" stays impossible).
    last_end: int | None = None
    index = 0
    while index < len(weeks):
        moment = datetime.fromtimestamp(weeks[index] / 1000, tz=tz)
        month_key = moment.strftime("%Y-%m")
        end = index
        while (
            end + 1 < len(weeks)
            and datetime.fromtimestamp(weeks[end + 1] / 1000, tz=tz).strftime("%Y-%m")
            == month_key
        ):
            end += 1
        span_columns = (end - index + 1) * stride - gap
        if span_columns >= 3:
            label = moment.strftime("%b").center(3)
            label_x = col + label_margin + index * stride
            label_x += max((span_columns - 3) // 2, 0)
            if (
                label_x + 3 <= col + used_cols
                and (last_end is None or label_x >= last_end + 1)
            ):
                painter.text(row, label_x, label, MUTED)
                last_end = label_x + 3
        index = end + 1

    day_row = row + 1
    for slot, name in enumerate(WEEKDAYS):
        painter.text(day_row + slot, col, name, MUTED)

    cursor_bg = ramp[-1]
    cursor_fg = 0x111418 if rgb_luminance(cursor_bg) > 128 else 0xE6EDF3
    for index, week in enumerate(weeks):
        monday = datetime.fromtimestamp(week / 1000, tz=tz)
        for slot in range(7):
            day_ms = int((monday + timedelta(days=slot)).timestamp() * 1000)
            level = levels.get(day_ms)
            if level is None:  # day is outside the current window
                continue
            bg = ramp[level]
            fg = 0x111418 if rgb_luminance(bg) > 128 else 0xE6EDF3
            mask = BOLD if cursor == day_ms else 0
            if cursor == day_ms:
                bg, fg = cursor_bg, cursor_fg
            painter.text(
                day_row + slot, col + label_margin + index * stride, " ",
                Attr(fg, bg, mask),
            )

    draw_legend(painter, day_row + 7, col + label_margin, ramp)
    return HEAT_ROWS, used_cols


def heatmap_width(weeks: list[int], *, gap: int = 1) -> int:
    """Columns the grid occupies (weekday labels + week columns)."""
    if not weeks:
        return 4
    return 4 + len(weeks) * 2 + (len(weeks) - 1) * gap


def draw_legend(painter: Painter, row: int, col: int, ramp: tuple[int, ...]) -> None:
    """'less ▢▢▢▢▢ more' ramp key."""
    painter.text(row, col, "less", MUTED)
    x = col + 5
    for level, _unused in enumerate(ramp):
        painter.text(row, x, " ", Attr(None, ramp[level], 0))
        x += 1
    painter.text(row, x, " more", MUTED)


# ------------------------------------------------------------- line chart


def draw_braille_lines(
    painter: Painter,
    row: int,
    col: int,
    width: int,
    height: int,
    series: Sequence[tuple[str, list[float | None]]],
    y_max: float,
) -> None:
    """Multi-series lines via Braille sub-pixels (⣿); colors follow SERIES_HEX.

    ``None`` values break the line exactly like the bitmap renderer. The
    bottom-most dot in each cell decides the cell's ink color.
    """
    if width <= 0 or height <= 0 or not series:
        return
    sub_rows = height * 4
    sub_cols = width * 2
    slots = max(len(values) for _, values in series)
    if slots <= 0:
        return

    pixels: dict[tuple[int, int], tuple[int, int]] = {}  # (x, y) → (series_index, y)

    def paint_dot(px: int, py: int, index: int) -> None:
        if 0 <= px < sub_cols and 0 <= py < sub_rows:
            pixels[(px, py)] = (index, py)  # later series paint over earlier

    for index, (_name, values) in enumerate(series):
        prev: tuple[float, float] | None = None
        for slot, value in enumerate(values):
            if value is None:
                prev = None
                continue
            sx = (slot + 0.5) * sub_cols / slots
            sy = (1 - value / max(y_max, 1e-9)) * (sub_rows - 1)
            point = (sx, sy)
            if prev is not None:
                steps = int(max(abs(point[0] - prev[0]), abs(point[1] - prev[1])))
                for step in range(steps + 1):
                    t = step / steps if steps else 0.0
                    px = prev[0] + (point[0] - prev[0]) * t
                    py = prev[1] + (point[1] - prev[1]) * t
                    paint_dot(int(round(px)), int(round(py)), index)
            prev = point

    for cy in range(height):
        for cx in range(width):
            bits = 0
            chosen = None
            top_y = -1
            for dy in range(4):
                for dx in range(2):
                    hit = pixels.get((cx * 2 + dx, cy * 4 + dy))
                    if hit is None:
                        continue
                    bits |= 1 << (dy * 2 + dx)
                    if hit[1] >= top_y:
                        top_y = hit[1]
                        chosen = hit[0]
            if bits and chosen is not None:
                painter.text(
                    row + cy,
                    col + cx,
                    chr(0x2800 + bits),
                    Attr(SERIES_HEX[chosen % len(SERIES_HEX)], None, 0),
                )


# ------------------------------------------------------------ stacked bars


def draw_stacked(
    painter: Painter,
    row: int,
    col: int,
    width: int,
    height: int,
    series: Sequence[tuple[str, list[float | None]]],
    y_max: float,
) -> None:
    """Stacked area/bars per slot with 1/8-cell vertical resolution.

    When the window holds more slots than columns, slots are strided so the
    shape always fits: gaps read as background.
    """
    if width <= 0 or height <= 0 or not series:
        return
    slots = max(len(values) for _, values in series)
    stride = max(1, math.ceil(slots / width))
    columns = []
    for slot in range(0, slots, stride):
        layers: list[tuple[int, float, float]] = []
        base = 0.0
        for index, (_name, values) in enumerate(series):
            value = values[slot] if slot < len(values) else None
            if value is None:
                continue
            units = max(value, 0.0) / max(y_max, 1e-9) * height * 8
            layers.append((index, base, base + units))
            base += units
        columns.append((slot, layers))

    column_layers = [layers for _slot, layers in columns]
    for r in range(height):
        top = (height - r) * 8
        bottom = top - 8
        x = 0
        while x < len(column_layers):
            pick = _covering(column_layers[x], bottom, top)
            if pick is None:
                x += 1
                continue
            start = x
            while x < len(column_layers) and _covering(column_layers[x], bottom, top) == pick:
                x += 1
            index, level = pick
            glyph = _BLOCK_GLYPHS[level - 1] * (x - start)
            color = SERIES_HEX[index % len(SERIES_HEX)]
            painter.text(row + r, col + start, glyph, Attr(color, None, 0))


def _covering(
    layers: list[tuple[int, float, float]],
    bottom: float,
    top: float,
) -> tuple[int, int] | None:
    """(series_index, glyph_level 1..8) of the first layer covering a row."""
    for index, layer_bottom, layer_top in layers:
        if layer_top <= bottom or layer_bottom >= top:
            continue
        filled = min(layer_top, top) - max(layer_bottom, bottom)
        return (index, min(max(int(filled), 1), 8))
    return None


# ------------------------------------------------------------------ tables


def bar_glyphs(fraction: float | None, width: int) -> str:
    """An inline share/rate bar: '█████▋    ' — colour only on the used part.

    The unused portion is unpainted space: halftone shade characters (``░``)
    puddle into noise over a transparent terminal background and read as a
    third colour elsewhere. ``None`` (no data) renders as an empty bar;
    fractions clip to [0, 1]. The partial cell fills from the *left*
    (``_BAR_PARTIALS``): the bottom-anchored ladder (``_BLOCK_GLYPHS``)
    belongs to stacked columns, where each text row is a vertical slice —
    in a horizontal bar it would read as a stray character rather than as
    part of the fill.
    """
    if width <= 0:
        return ""
    if fraction is None:
        return " " * width
    fraction = max(0.0, min(1.0, float(fraction)))
    units = fraction * width * 8
    full = int(units // 8)
    remainder = int(units - full * 8 + 0.5)  # round the partial block
    if remainder > 7:  # rounded up into a full block
        full += 1
        remainder = 0
    glyphs = "█" * full
    if remainder:
        glyphs += _BAR_PARTIALS[remainder - 1]
    glyphs += " " * max(width - len(glyphs), 0)
    return glyphs


def draw_share_rows(
    painter: Painter,
    row: int,
    col: int,
    *,
    per_harness: dict[str, int],
    total: int,
    width: int,
    max_rows: int,
    index_of,
    name_width: int | None = None,
    label=None,
) -> int:
    """'■ name ▮▮▮▮ 42%' per entry, one row each; rows painted returned.

    The vertical counterpart of the old one-line share strip: readable
    bars with room for the token totals when the rail is wide enough.
    ``index_of`` maps a name to its series color index (the harness
    catalog lives in toknt.tui.data, which must not be imported from
    here); ``None`` paints the rows in the neutral color, which is what
    projects want - they are not a series. ``label`` rewrites the shown
    name (project paths show their tail).
    """
    painted = 0
    ranked = sorted(per_harness.items(), key=lambda kv: -kv[1])
    if name_width is None:
        name_width = max(min(width - 20, 40), 8)
    for name, amount in ranked:
        if painted >= max_rows:
            painter.text(
                row + painted, col, f"+{len(ranked) - painted} more", MUTED
            )
            break
        share = amount / total if total else 0.0
        # the bar follows the name, so a short name leaves no void; the
        # whole row (name, bar, percent, total) stays inside the zone
        bar_width = max(min(width - name_width - 16, 24), 6)
        color = (
            Attr(SERIES_HEX[index_of(name) % len(SERIES_HEX)], None, 0)
            if index_of is not None
            else DEFAULT_ATTRS
        )
        painter.text(row + painted, col, "\u25a0", color)
        shown_name = label(name) if label is not None else name
        painter.text(row + painted, col + 2, shown_name[:name_width], MUTED)
        painter.text(
            row + painted, col + 3 + name_width, bar_glyphs(share, bar_width), color
        )
        painter.text(
            row + painted,
            col + 4 + name_width + bar_width,
            f"{share * 100:>3.0f}%",
            MUTED,
        )
        if width >= 40:
            painter.text(
                row + painted,
                col + 9 + name_width + bar_width,
                compact_number(amount).rjust(7),
                MUTED,
            )
        painted += 1
    return painted


def _compose_row(
    columns: Sequence[tuple[str, int, int]],
    cells: Sequence[str],
) -> str:
    """Left/right-aligned, clipped one-line row text (trailing pad per col)."""
    pieces: list[str] = []
    for position, (_label, width, align) in enumerate(columns):
        raw = cells[position] if position < len(cells) else ""
        text = str(raw)
        if align:  # right
            text = text[-width:].rjust(width)
        else:
            text = text[:width].ljust(width)
        pieces.append(text + "  ")
    return "".join(pieces).rstrip()


def draw_table(
    painter: Painter,
    row: int,
    col: int,
    columns: Sequence[tuple[str, int, int]],
    rows: Sequence[Sequence[str]],
    *,
    selected: int | None = None,
    max_rows: int | None = None,
    row_offset: int = 0,
) -> int:
    """Header + clipped body; returns the number of table rows painted.

    Column triples are ``(label, width, align)`` with ``align`` 0 = left,
    1 = right. ``selected`` is an absolute row index; ``row_offset`` scrolls
    the window through longer lists.
    """
    header_cells: list[str] = []
    for label, _width, align in columns:
        clean = label[1:] if align and label.startswith(">") else label
        header_cells.append(clean)
    painter.text(row, col, _compose_row(columns, header_cells), LABEL)
    visible = rows[row_offset:]
    if max_rows is not None:
        visible = visible[:max_rows]
    for index, cells in enumerate(visible):
        absolute = row_offset + index
        attrs = BOLD_ATTR if absolute == selected else DEFAULT_ATTRS
        painter.text(row + 1 + index, col, _compose_row(columns, cells), attrs)
    return len(visible)


# ------------------------------------------------------------- metric tags


def agent_swatch(painter: Painter, row: int, col: int, index: int, label: str) -> None:
    """'■ label' legend entry in the series' color (colorblind-safe hex set)."""
    color = SERIES_HEX[index % len(SERIES_HEX)]
    painter.text(row, col, "■", Attr(color, None, 0))
    painter.text(row, col + 2, label, MUTED)
