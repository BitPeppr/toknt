"""Widget behavior: heat grid, block/braille charts, tables, helpers."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from toknt.analytics import aggregation
from toknt.tui.painter import BOLD, Painter
from toknt.tui.style import SERIES_HEX, heat_ramp
from toknt.tui.widgets import (
    compact_number,
    draw_braille_lines,
    draw_heatmap,
    draw_stacked,
    draw_table,
    heat_levels,
    week_starts,
)

UTC = ZoneInfo("UTC")
NOW = int(datetime(2026, 6, 19, 12, tzinfo=UTC).timestamp() * 1000)


def test_compact_number() -> None:
    assert compact_number(None) == "—"
    assert compact_number(999) == "999"
    assert compact_number(1234) == "1.2K"
    assert compact_number(3_400_000) == "3.4M"
    assert compact_number(2_500_000_000) == "2.5B"


def test_week_starts_are_monday_midnight_and_match_aggregation() -> None:
    weeks = week_starts(NOW, 9, UTC)
    assert len(weeks) == 9
    monday = datetime.fromtimestamp(weeks[0] / 1000, tz=UTC)
    assert monday.weekday() == 0
    assert monday.hour == monday.minute == monday.second == 0
    # grid keys must equal aggregation's day buckets (DST-safe matching)
    for week in weeks:
        for slot in range(7):
            moment = datetime.fromtimestamp(week / 1000, tz=UTC)
            day_key = int((moment.timestamp() + 86_400 * slot) * 1000)
            expected = aggregation.bucket_start_ms(week + slot * 86_400_000, "day", UTC)
            # wall-clock arithmetic can differ only across DST boundaries;
            # in UTC they must be identical
            assert day_key == expected


def test_heat_levels_sqrt_mapping() -> None:
    levels = heat_levels({1: 0, 2: 100, 3: 1000, 4: 500})
    assert levels[1] == 0
    assert levels[3] == 4  # peak
    assert 1 <= levels[2] <= 3
    assert heat_levels({}) == {}
    assert all(value == 0 for value in heat_levels({5: 0, 6: 0}).values())


def test_draw_heatmap_paints_cells_and_cursor() -> None:
    painter = Painter()
    painter.resize(14, 120)
    painter.frame()
    weeks = week_starts(NOW, 4, UTC)
    totals = {week: 1000 * index for index, week in enumerate(weeks)}
    rows, cols = draw_heatmap(
        painter, 1, 1, weeks=weeks, totals=totals, ramp=heat_ramp("dark"),
        tz=UTC, cursor=weeks[2],
    )
    # the block is month labels + 7 day rows + the ramp key; the viewport
    # heading belongs to the view, not to the grid
    # two cells per week plus one space between weeks: 4 label columns +
    # 4 weeks * 2 cells + 3 gaps
    assert (rows, cols) == (9, 4 + 8 + 3)
    painted = [
        (y, x)
        for y in range(2, 9)
        for x in range(5, 5 + cols - 4)
        if painter.buffer[y][x].bg is not None
    ]
    assert painted, "no heat cells painted"
    cursor_cells = [
        (y, x)
        for y in range(2, 9)
        for x in range(0, 120)
        if painter.buffer[y][x].mask & BOLD
    ]
    assert cursor_cells, "cursor cell not highlighted"
    assert all(
        painter.buffer[y][x].bg == heat_ramp("dark")[-1] for y, x in cursor_cells
    )


def test_draw_heatmap_skips_days_outside_window() -> None:
    painter = Painter()
    painter.resize(14, 60)
    painter.frame()
    weeks = week_starts(NOW, 4, UTC)
    draw_heatmap(
        painter, 1, 1, weeks=weeks, totals={weeks[0]: 5}, ramp=heat_ramp("dark"),
        tz=UTC,
    )
    with_bg = sum(
        1
        for y in range(2, 9)
        for x in range(0, 60)
        if painter.buffer[y][x].bg is not None
    )
    assert with_bg == 1  # only the Monday of week 0 carries data


def test_draw_heatmap_breathes_and_centres_in_a_wide_zone() -> None:
    painter = Painter()
    painter.resize(14, 120)
    painter.frame()
    weeks = week_starts(NOW, 4, UTC)
    totals = {week + slot * 86_400_000: 10 for week in weeks for slot in range(7)}
    rows, cols = draw_heatmap(
        painter, 1, 1, weeks=weeks, totals=totals, ramp=heat_ramp("dark"),
        tz=UTC, width=60,
    )
    assert rows == 9
    # a wider zone adds breathing space between weeks and centres the grid
    assert cols > 4 + 8
    painted_x = [
        x
        for x in range(0, 120)
        if any(painter.buffer[y][x].bg is not None for y in range(2, 9))
    ]
    assert painted_x
    assert painted_x[0] > 1  # not glued to the left edge
    assert painted_x[-1] < 119


def test_braille_lines_paint_colored_subpixels() -> None:
    painter = Painter()
    painter.resize(12, 40)
    painter.frame()
    draw_braille_lines(
        painter, 0, 0, 30, 6,
        [("a", [0, 5, 2, 8]), ("b", [1, None, 3, 0])],
        8,
    )
    braille = [cell for line in painter.buffer for cell in line if "⠀" <= cell.char <= "⣿"]
    assert braille
    assert {cell.fg for cell in braille} == {SERIES_HEX[0], SERIES_HEX[1]}  # both series
    assert painter.frame() != b""


def test_stacked_blocks_fill_bottom_up() -> None:
    painter = Painter()
    painter.resize(12, 40)
    painter.frame()
    draw_stacked(
        painter, 1, 1, 10, 6,
        [("a", [4, 0, 8]), ("b", [4, None, 2])],
        10,
    )
    middle = painter.buffer[6][1:]  # row near the bottom
    chars = [cell.char for cell in middle]
    assert any(glyph in "▁▂▃▄▅▆▇█" for glyph in chars)
    empty_slot = painter.buffer[6][2]  # slot 1 has no data → blank
    assert empty_slot.char in (" ", "\x00") or empty_slot.bg is not None


def test_draw_table_aligns_and_selects() -> None:
    painter = Painter()
    painter.resize(10, 80)
    painter.frame()
    columns = [("model", 8, 0), (">tokens", 8, 1), (">share", 6, 1)]
    drawn = draw_table(
        painter, 0, 0, columns,
        [["gpt-5", "12000", "60%"], ["o3", "8000", "40%"]],
        selected=1,
    )
    assert drawn == 2
    header = "".join(cell.char for cell in painter.buffer[0]).rstrip()
    row0 = "".join(cell.char for cell in painter.buffer[1]).rstrip()
    row1 = "".join(cell.char for cell in painter.buffer[2]).rstrip()
    assert "model" in header and "tokens" in header and "share" in header
    assert "gpt-5" in row0 and "12000" in row0
    assert "o3" in row1 and "8000" in row1
    assert painter.buffer[2][3].mask & BOLD  # selection highlight
    assert not painter.buffer[1][3].mask & BOLD  # unselected rows stay plain
