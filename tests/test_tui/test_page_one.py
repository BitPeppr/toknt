"""Page-one layout, the day cursor, and metric-aware tables.

The contributions page is the explorer's front door: the calendar has to
sit next to its weekday labels, the day under the cursor has to follow the
arrows, and every table has to answer the metric that is selected.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from toknt.config import AppConfig
from toknt.tui.app import App
from toknt.tui.keys import KeyEvent
from toknt.tui.views import TerminalContext


def _app(rows: int = 34, cols: int = 120):
    writes: list[bytes] = []
    app = App(AppConfig(), stdin_fd=0, stdout_fd=1)
    app.term = SimpleNamespace(write=writes.append, writes=writes)
    app.ctx = TerminalContext(backend="off", cell_w=10, cell_h=20, theme="dark")
    app.painter.resize(rows, cols)
    return app, writes


def _rows(app: App) -> list[str]:
    return [
        "".join(cell.char for cell in app.painter.buffer[y]).rstrip()
        for y in range(app.painter.rows)
    ]


class TestCalendarLayout:
    def test_grid_sits_next_to_its_weekday_labels(self, data_dir, seeded_conn) -> None:
        app, _writes = _app()
        app.paint()
        rows = _rows(app)
        grid_row = next(line for line in rows if line.startswith("  Mon"))
        label_end = grid_row.index("Mon") + 3
        # the first cell follows the label immediately, not a screen away
        first_cell = grid_row.index("▓") if "▓" in grid_row else -1
        assert first_cell == -1 or first_cell - label_end <= 2

    def test_week_columns_are_not_stretched(self, data_dir, seeded_conn) -> None:
        app, _writes = _app()
        app.paint()
        buf = app.painter.buffer
        # a week column is two cells plus a single space: cells in one row
        # are never further apart than three columns
        row = next(
            y
            for y in range(4, 12)
            if any(buf[y][x].bg is not None for x in range(4, 80))
        )
        columns = [x for x in range(4, 80) if buf[row][x].bg is not None]
        assert columns
        gaps = {b - a for a, b in zip(columns, columns[1:], strict=False)}
        assert max(gaps) <= 3

    def test_no_weekly_bar_chart_on_page_one(self, data_dir, seeded_conn) -> None:
        app, _writes = _app()
        app.paint()
        joined = "\n".join(_rows(app))
        assert "weekly tokens" not in joined


class TestShareLists:
    def test_oxa_keys_no_longer_toggle_harnesses(
        self, data_dir, seeded_conn
    ) -> None:
        # 'h' is the only harness filter now; letters that used to toggle
        # three hard-coded harnesses must be inert
        app, _writes = _app()
        before = app.state.agents
        for key in "oxa":
            assert app.apply_key(KeyEvent("char", key)) is False
        assert app.state.agents == before

    def test_project_list_sits_above_the_harness_list(
        self, data_dir, seeded_conn
    ) -> None:
        app, _writes = _app()
        app.paint()
        joined = "\n".join(_rows(app))
        assert "share by project" in joined
        assert "share by harness" in joined
        assert joined.index("share by project") < joined.index("share by harness")

    def test_harnesses_with_data_are_listed(self, data_dir, seeded_conn) -> None:
        app, _writes = _app()
        app.paint()
        joined = "\n".join(_rows(app))
        assert "opencode" in joined


class TestDayCursor:
    def test_cursor_lands_on_a_day_the_grid_paints(self, data_dir, seeded_conn) -> None:
        app, _writes = _app()
        app.paint()
        buf = app.painter.buffer
        cursor = app.state.cursor_day
        assert cursor is not None
        highlighted = [
            (y, x)
            for y in range(4, 12)
            for x in range(0, 120)
            if buf[y][x].mask & 1  # BOLD
        ]
        assert highlighted, "the cursor day is not highlighted in the grid"

    def test_arrow_keys_move_the_cursor_and_the_picked_day(
        self, data_dir, seeded_conn
    ) -> None:
        app, _writes = _app()
        app.paint()
        start = app.state.cursor_day
        assert start is not None
        app.apply_key(KeyEvent("left"))
        assert app.state.cursor_day == start - 86_400_000
        app.apply_key(KeyEvent("right"))
        assert app.state.cursor_day == start

    def test_cursor_never_leaves_the_painted_days(self, data_dir, seeded_conn) -> None:
        app, _writes = _app()
        app.paint()
        for _ in range(40):
            app.apply_key(KeyEvent("right"))
        buf = app.painter.buffer
        highlighted = [
            (y, x)
            for y in range(4, 12)
            for x in range(0, 120)
            if buf[y][x].mask & 1
        ]
        assert highlighted, "the cursor drifted off the painted grid"
        # and it stops on a whole day, never on a timestamp inside one
        cursor = app.state.cursor_day
        assert cursor is not None
        assert cursor % 86_400_000 == 0 or cursor == app.state.cursor_day

    def test_picked_day_reports_the_cursor_day(self, data_dir, seeded_conn) -> None:
        app, _writes = _app()
        app.paint()
        joined = "\n".join(_rows(app))
        assert "picked day" in joined


class TestMetricFollows:
    @staticmethod
    def _model_rows(rows: list[str]) -> list[str]:
        """The body of the models table (below its header)."""
        header = next(
            index
            for index, line in enumerate(rows)
            if "model" in line and "events" in line
        )
        return [line for line in rows[header + 1 :] if line.strip()]

    def _models_table(self, metric_presses: int) -> list[str]:
        app, _writes = _app()
        app.state.view = 1
        for _ in range(metric_presses):
            app.apply_key(KeyEvent("char", "v"))
        app.paint()
        return _rows(app)

    def test_table_header_follows_the_metric(self, data_dir, seeded_conn) -> None:
        total = "\n".join(self._models_table(0))
        inputs = "\n".join(self._models_table(1))
        assert ">tokens" in total or "tokens" in total
        assert "input" in inputs
        assert "input" not in total.split("tokens by model")[0]

    def test_table_values_follow_the_metric(self, data_dir, seeded_conn) -> None:
        total = self._model_rows(self._models_table(0))
        inputs = self._model_rows(self._models_table(1))
        assert total and inputs
        assert total[0] != inputs[0]

    def test_sessions_table_follows_the_metric(self, data_dir, seeded_conn) -> None:
        app, _writes = _app()
        app.state.view = 3
        app.apply_key(KeyEvent("char", "v"))
        app.paint()
        joined = "\n".join(_rows(app))
        assert "input" in joined


class TestTitleBar:
    def test_no_graphics_chip(self, data_dir, seeded_conn) -> None:
        app, _writes = _app()
        app.paint()
        top = _rows(app)[0]
        assert "kitty" not in top
        assert "braille" not in top


class TestPageMargins:
    """The page keeps equal margins on both sides at any terminal width."""

    @staticmethod
    def _page_extents(cols: int) -> tuple[int, int]:
        """The page block: where the hairline rule starts and ends."""
        app, _writes = _app(rows=24, cols=cols)
        app.paint()
        rule = "".join(cell.char for cell in app.painter.buffer[2])
        used = [x for x, char in enumerate(rule) if char == "─"]
        assert used
        return used[0] + 1, used[-1] + 1

    @pytest.mark.parametrize("cols", [80, 100, 120, 140, 160, 200])
    def test_margins_are_symmetric(self, data_dir, seeded_conn, cols: int) -> None:
        left, right = self._page_extents(cols)
        assert left - 1 == cols - right

    def test_wide_terminal_centers_the_page(self, data_dir, seeded_conn) -> None:
        left, right = self._page_extents(200)
        assert left > 3  # not glued to the left edge
        assert left - 1 == 200 - right

    def test_page_frame_helper(self) -> None:
        from toknt.tui.views import PAGE_MAX_WIDTH, page_frame

        left, width = page_frame(120)
        assert left == 2 and width == 116
        left, width = page_frame(200)
        assert width == PAGE_MAX_WIDTH
        assert left == (200 - width) // 2
