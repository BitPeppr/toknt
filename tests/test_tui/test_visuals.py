"""Round-6 visual regression tests: month labels, donut, bars, harness focus.

Covers the user-visible fixes: calendar month labels that never collide
("SeOct" is impossible), the help overlay leaving the chart bitmap alone,
the 'h' harness-focus cycle, inline bar glyphs, the models donut and the
per-scheme cache-rate aggregation.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from tests.support.models import at, tool, usage
from toknt.config import AppConfig
from toknt.models.usage import TokenScheme
from toknt.presentation import graphics
from toknt.storage import ingestion
from toknt.tui import views, widgets
from toknt.tui.app import App
from toknt.tui.keys import KeyEvent
from toknt.tui.painter import Painter
from toknt.tui.style import SERIES_HEX, heat_ramp
from toknt.tui.views import AppState, TerminalContext, title_row


def _screen(app: App, rows: int, cols: int) -> dict[int, str]:
    """Rebuild the painted screen from the app's captured writes."""
    frame = b"".join(app.term.writes).decode("utf-8", "replace")  # type: ignore[union-attr]
    screen: dict[int, list[str]] = {}
    cursor = [1, 1]
    i = 0
    while i < len(frame):
        m = re.match(r"\x1b\[(\d+);(\d+)H", frame[i:])
        if m:
            cursor = [int(m.group(1)), int(m.group(2))]
            i += m.end()
            continue
        if frame[i] == "\x1b":
            m2 = re.match(r"\x1b\[[0-9;?]*[a-zA-Z]", frame[i:])
            if m2:
                i += m2.end()
                continue
            i += 1
            continue
        line = screen.setdefault(cursor[0], [" "] * cols)
        if cursor[1] <= cols:
            line[cursor[1] - 1] = frame[i]
        cursor[1] += 1
        i += 1
    return {row: "".join(cells).rstrip() for row, cells in screen.items()}


class TestTitleStatus:
    """The title bar's right edge: a one-line note, never over the chips."""

    def _screen(self, columns: int = 120, **state_kwargs) -> list[str]:
        painter = Painter()
        painter.resize(14, columns)
        state = AppState(**state_kwargs)
        title_row(painter, state, rows=14, columns=columns)
        return [
            "".join(cell.char for cell in painter.buffer[row]).rstrip()
            for row in range(14)
        ]

    def test_status_renders_right_aligned(self) -> None:
        lines = self._screen(columns=200, status="scan done: 5 new events")
        assert lines[0].rstrip().endswith("scan done: 5 new events")

    def test_scanning_note_wins_over_status(self) -> None:
        lines = self._screen(status="scan done: 5", scanning=True)
        assert "‣ scanning…" in lines[0]
        assert "scan done" not in lines[0]

    def test_status_too_long_is_clipped_with_an_ellipsis(self) -> None:
        lines = self._screen(columns=120, status="scan done: 5 new events")
        # the message is too long for the space the chips left: it moves to
        # the hint bar whole rather than being cut to nothing
        assert "scan done: 5 new events" in "\n".join(lines)

    def test_status_falls_back_to_the_hint_bar_when_it_does_not_fit(self) -> None:
        # at 60 columns the tab strip fills the title row: the note moves
        # to the bottom bar rather than disappearing
        lines = self._screen(columns=60, status="tok/s applies to the models view")
        assert "tok/s applies" in lines[13]
        assert "q quit" in lines[13]

    def test_long_status_never_collides_with_chips(self) -> None:
        lines = self._screen(status="x" * 300)
        # the chips must survive intact: a note never evicts them
        assert "harnesses: all" in lines[0]
        assert "3d" in lines[0]

    def test_status_stays_inside_the_page_frame(self) -> None:
        lines = self._screen(columns=120, status="y" * 40)
        page_left = 2
        page_right = 118
        for line in lines:
            assert len(line) <= page_right + 1
        joined = "\n".join(lines)
        assert page_left <= joined.index("y")


class TestModelRailAlignment:
    """The cache bars mirror the model table row for row.

    The rail sits beside the table, so each bar must describe the model next
    to it: same models, same order, same first and last row. A list that is
    merely the same length but ranked differently would look aligned while
    saying something else.
    """

    def _lines(self, rows: int = 30, cols: int = 160) -> list[str]:
        writes: list[bytes] = []
        app = App(AppConfig(), stdin_fd=0, stdout_fd=1)
        app.term = SimpleNamespace(write=writes.append, writes=writes)
        app.ctx = TerminalContext(backend="off", cell_w=10, cell_h=20, theme="dark")
        app.painter.resize(rows, cols)
        app.state.view = 1
        app.paint()
        screen = _screen(app, rows, cols)
        return [line for _row, line in sorted(screen.items())]

    @staticmethod
    def _rows(lines: list[str]) -> tuple[int, int, list[int], list[int]]:
        heading = next(
            index for index, line in enumerate(lines) if "cache hit rate" in line
        )
        rail_col = lines[heading].index("cache hit rate")
        header = next(
            index
            for index, line in enumerate(lines)
            if "events" in line[:rail_col] and "model" in line[:rail_col]
        )
        body = [
            index
            for index, line in enumerate(lines)
            if index > header and "q quit" not in line and line[:rail_col].strip()
        ]
        cache = [
            index
            for index, line in enumerate(lines)
            if index > header and "q quit" not in line and line[rail_col:].strip()
        ]
        return rail_col, header, body, cache

    @pytest.mark.parametrize("rows", [24, 30, 40])
    def test_cache_rows_mirror_the_model_rows(
        self, rows: int, data_dir, seeded_conn
    ) -> None:
        # a corpus with more cache rows than the table can show: the bars
        # must stop with the table, not run past it
        ingestion.upsert_usage_events(
            seeded_conn,
            [
                usage(
                    id=f"usage:opencode:cache-{index:02d}",
                    ts_ms=at(2026, 10, 2, 9) + index * 1000,
                    model=f"cache-model-{index:02d}",
                    input=100,
                    output=50,
                    reasoning=10,
                    cache_read=5,
                    cache_write=0,
                )
                for index in range(40)
            ],
        )
        lines = self._lines(rows=rows)
        rail_col, header, body, cache = self._rows(lines)
        heading = next(
            index for index, line in enumerate(lines) if "cache hit rate" in line
        )
        # the cache heading shares the table's header row, so the bars and
        # the model rows line up exactly
        assert heading == header
        assert body
        assert body == cache
        assert len(body) <= 24
        for index in body:
            model = lines[index][:rail_col].split()[0]
            label = lines[index][rail_col:].split()[0]
            assert model.startswith(label), (model, label, lines[index])

    def test_every_share_keeps_a_legend_chip_in_the_narrow_rail(
        self, data_dir, seeded_conn
    ) -> None:
        # the narrowest rail (the tok/s column borrows width from it) must
        # still name every slice of the donut
        ingestion.upsert_usage_events(
            seeded_conn,
            [
                usage(
                    id=f"usage:opencode:share-{index:02d}",
                    ts_ms=at(2026, 10, 2, 9) + index * 1000,
                    model=f"share-model-{index:02d}",
                    input=100 + index,
                    output=50,
                )
                for index in range(6)
            ],
        )
        writes: list[bytes] = []
        app = App(AppConfig(), stdin_fd=0, stdout_fd=1)
        app.term = SimpleNamespace(write=writes.append, writes=writes)
        app.ctx = TerminalContext(backend="off", cell_w=10, cell_h=20, theme="dark")
        app.painter.resize(24, 100)
        app.state.view = 1
        app.paint()
        screen = _screen(app, 24, 100)
        lines = [line for _row, line in sorted(screen.items())]
        heading = next(
            index for index, line in enumerate(lines) if "cache hit rate" in line
        )
        rail_col = lines[heading].index("cache hit rate")
        chips = sum(line[rail_col:].count("■") for line in lines[:heading])
        assert chips == 6
        app.conn.close()

    def test_model_without_cache_reporting_gets_a_dash_in_its_row(
        self, data_dir, seeded_conn
    ) -> None:
        ingestion.upsert_usage_events(
            seeded_conn,
            [
                usage(
                    id="usage:opencode:cache-known",
                    ts_ms=at(2026, 10, 2, 9),
                    model="known-cache-model",
                    input=100,
                    output=50,
                    cache_read=100,
                    cache_write=0,
                ),
                usage(
                    id="usage:opencode:cache-unknown",
                    ts_ms=at(2026, 10, 2, 9, 1),
                    model="unknown-cache-model",
                    scheme=TokenScheme.UNKNOWN,
                    input=100,
                    output=10,
                    cache_read=0,
                    cache_write=0,
                ),
            ],
        )
        lines = self._lines(rows=30)
        rail_col, _header, _body, _cache = self._rows(lines)
        unknown_row = next(
            line for line in lines if line[rail_col:].startswith("unknown-cach")
        )
        # no cache-reporting scheme → no invented percentage
        assert "—" in unknown_row[rail_col:]
        known_row = next(
            line for line in lines if line[rail_col:].startswith("known-cache-")
        )
        assert "50%" in known_row[rail_col:]


class TestSeriesPalette:
    def test_text_and_raster_share_one_palette(self) -> None:
        """A legend swatch and its bitmap line must be the same colour."""
        assert SERIES_HEX is graphics.SERIES_PALETTE
        assert graphics.SERIES_PALETTE[0] == 0x4E8FA8
        assert 0x56B4E9 not in graphics.SERIES_PALETTE  # the old bright sky


class TestModelRailLegend:
    """Every donut slice keeps a name, even when the rail is narrow."""

    @staticmethod
    def _rail(rows: int, list_top: int) -> tuple[str, object]:
        painter = Painter()
        painter.resize(rows, 100)
        shares = [(f"model-{index:02d}", float(10 - index)) for index in range(6)]
        image = views._paint_model_side(
            painter,
            70,
            28,
            shares,
            [],
            row=3,
            rows=rows,
            ctx=TerminalContext(
                backend="kitty", cell_w=10, cell_h=20, theme="dark"
            ),
            identity=(),
            list_top=list_top,
        )
        text = "\n".join("".join(cell.char for cell in line) for line in painter.buffer)
        return text, image

    @pytest.mark.parametrize(("rows", "list_top"), [(24, 15), (16, 10)])
    def test_all_six_shares_are_named(
        self, rows: int, list_top: int
    ) -> None:
        text, image = self._rail(rows, list_top)
        for index in range(6):
            assert f"model-{index:02d}" in text
        if rows == 24:
            assert image is not None  # the donut still fits above the cache
        else:
            assert image is None  # too short: the ribbon replaces the donut

    def test_narrow_rail_uses_two_columns_not_four_rows(self) -> None:
        text, _image = self._rail(24, 15)
        legend_rows = [
            line
            for line in text.splitlines()
            if "model-00" in line or "model-02" in line or "model-04" in line
        ]
        assert len(legend_rows) == 3
        assert all(
            line.count("model-") == 2 for line in legend_rows
        )


class TestSessionsBottomMargin:
    def test_tall_terminal_keeps_a_bottom_margin_and_scrolls(
        self, data_dir, seeded_conn
    ) -> None:
        base = at(2026, 10, 10, 9, 0)
        ingestion.upsert_usage_events(
            seeded_conn,
            [
                usage(
                    id=f"usage:opencode:many-{index:02d}",
                    ts_ms=base + index * 60_000,
                    session_id=f"ses_many_{index:02d}",
                    project=f"/work/session-{index:02d}",
                    model="model-a",
                )
                for index in range(30)
            ],
        )
        writes: list[bytes] = []
        app = App(AppConfig(), stdin_fd=0, stdout_fd=1)
        app.term = SimpleNamespace(write=writes.append, writes=writes)
        app.ctx = TerminalContext(backend="off", cell_w=10, cell_h=20, theme="dark")
        app.painter.resize(40, 120)
        app.state.view = 3
        app.paint()
        screen = _screen(app, 40, 120)
        time_re = re.compile(r"\d{2} [A-Z][a-z]{2} \d{2}:\d{2}")
        rows = [row for row, line in screen.items() if time_re.search(line)]
        assert len(rows) == 24  # capped like the models table
        assert max(rows) <= 32  # a real bottom margin above the notes/hints
        # the 24-row window must not cap the cursor: scrolling still walks
        # the whole ranked list
        for _ in range(29):
            assert app.apply_key(KeyEvent("down"))
        assert app.state.session_row == 29
        app.paint()
        screen = _screen(app, 40, 120)
        first = min(row for row, line in screen.items() if time_re.search(line))
        assert "session-29" not in screen[first]  # newest row has scrolled off


class TestSessionsScroll:
    """The highlight walks down the rows; the list follows at the edge."""

    @staticmethod
    def _selected_row(app: App) -> int:
        for row in range(4, app.painter.rows):
            if any(cell.mask & 1 for cell in app.painter.buffer[row] if cell.char.strip()):
                return row
        raise AssertionError("no selected row was painted")

    def test_the_window_scrolls_only_when_the_cursor_reaches_the_edge(
        self, data_dir, seeded_conn
    ) -> None:
        base = at(2026, 10, 10, 9, 0)
        ingestion.upsert_usage_events(
            seeded_conn,
            [
                usage(
                    id=f"usage:opencode:scroll-{index:02d}",
                    ts_ms=base + index * 60_000,
                    session_id=f"ses_scroll_{index:02d}",
                    project=f"/work/scroll-{index:02d}",
                    model="model-a",
                )
                for index in range(40)
            ],
        )
        app = App(AppConfig(), stdin_fd=0, stdout_fd=1)
        app.term = SimpleNamespace(write=lambda _data: None, writes=[])
        app.ctx = TerminalContext(backend="off", cell_w=10, cell_h=20, theme="dark")
        app.painter.resize(40, 120)
        app.state.view = 3
        app.paint()
        assert self._selected_row(app) == 4  # the first body row
        app.apply_key(KeyEvent("down"))
        app.paint()
        # the highlight moves; the list does not scroll under it
        assert self._selected_row(app) == 5
        for _ in range(23):
            app.apply_key(KeyEvent("down"))
        app.paint()
        # now the cursor is at the bottom edge and the window follows
        assert self._selected_row(app) == 27
        app.conn.close()


class TestToolsOutcomes:
    def test_the_cursor_never_walks_past_the_painted_rows(
        self, data_dir, seeded_conn
    ) -> None:
        # the table paints a capped number of rows; the cursor must not
        # walk into the invisible tail
        base = at(2026, 10, 10, 9, 0)
        ingestion.upsert_tool_events(
            seeded_conn,
            [
                tool(
                    id=f"tool:opencode:many-{index:02d}",
                    ts_ms=base + index * 1000,
                    tool_name=f"tool-{index:02d}",
                )
                for index in range(70)
            ],
        )
        app = App(AppConfig(), stdin_fd=0, stdout_fd=1)
        app.term = SimpleNamespace(write=lambda _data: None, writes=[])
        app.ctx = TerminalContext(backend="off", cell_w=10, cell_h=20, theme="dark")
        app.painter.resize(30, 160)
        app.state.view = 2
        for _ in range(80):
            app.apply_key(KeyEvent("down"))
        assert app.state.tool_row == views.TOOL_ROW_CAP - 1
        app.conn.close()

    def test_pending_calls_are_counted_in_the_unknown_column(
        self, data_dir, seeded_conn
    ) -> None:
        # the fixture's 'exec' call is pending: it has no rate, but it must
        # still appear in an outcome column so calls always add up
        app = App(AppConfig(), stdin_fd=0, stdout_fd=1)
        app.term = SimpleNamespace(write=lambda _data: None, writes=[])
        app.ctx = TerminalContext(backend="off", cell_w=10, cell_h=20, theme="dark")
        app.painter.resize(20, 160)
        app.state.view = 2
        app.paint()
        lines = [
            "".join(cell.char for cell in app.painter.buffer[row]).rstrip()
            for row in range(app.painter.rows)
        ]
        row = next(line for line in lines if line.strip().startswith("exec "))
        assert re.search(
            r"exec\s+shell\s+1\s+0\s+0\s+0\s+0\s+1\s+—", row
        ), row
        app.conn.close()


class TestMonthLabels:
    def test_adjacent_months_never_overlap(self) -> None:
        painter = Painter()
        painter.resize(14, 120)
        tz = UTC
        # a 12-week grid where a month starts two weeks after the previous
        # label was drawn — the old renderer stamped both labels at their
        # first week and they collided ("SeOct")
        base = datetime(2026, 8, 10, tzinfo=tz)
        weeks = [
            int((base + __import__("datetime").timedelta(weeks=index)).timestamp() * 1000)
            for index in range(12)
        ]
        totals = {week + day * 86_400_000: 100 for week in weeks for day in range(7)}
        widgets.draw_heatmap(
            painter, 1, 1, weeks=weeks, totals=totals, ramp=heat_ramp("dark"),
            tz=tz, cursor=None,
        )
        labels = painter.buffer[2]  # month-label row (title is row 1)
        text = "".join(cell.char for cell in labels)
        month_names = re.findall(r"[A-Z][a-z]{2}", text)
        # every label present, none merged into a 6+ character run like SeOct
        assert month_names, "no month labels rendered"
        for name in month_names:
            assert text.count(name) >= 1
        # labels are separated by at least one blank column
        positions = [m.start() for m in re.finditer(r"[A-Z][a-z]{2}", text)]
        for a, b in zip(positions, positions[1:], strict=False):
            assert b - a >= 3, f"labels too close: {text!r}"


class TestBarGlyphs:
    def test_fraction_renders_partial_blocks(self) -> None:
        assert widgets.bar_glyphs(1.0, 4) == "████"
        # an unfilled bar carries no shade pattern: the track is empty space
        # so transparency in the terminal is respected and no third colour
        # appears — colour belongs to the used portion only
        assert widgets.bar_glyphs(0.0, 4) == "    "
        assert widgets.bar_glyphs(0.5, 4) == "██  "
        assert widgets.bar_glyphs(0.53125, 2) == "█▏"  # 8.5/16 units → full + 1/8 partial
        assert widgets.bar_glyphs(None, 3) == "   "
        assert widgets.bar_glyphs(1.5, 2) == "██"  # clips
        assert widgets.bar_glyphs(-0.5, 2) == "  "
        # a NaN share is not a share: it must not read as a full bar
        assert widgets.bar_glyphs(float("nan"), 4) == "    "

    def test_partial_block_fills_from_the_left(self) -> None:
        # a horizontal bar must not use the bottom-anchored ladder: that
        # glyph reads as a stray character instead of part of the fill
        horizontal = "▏▎▍▌▋▊▉"
        for eighth in range(1, 8):
            fraction = eighth / 8
            bar = widgets.bar_glyphs(fraction, 1)
            assert bar[0] in horizontal, f"{bar!r} is not a left-anchored partial"

    def test_stacked_columns_keep_the_vertical_ladder(self) -> None:
        # the bottom-anchored ladder belongs to stacked columns, where
        # every text row is a vertical slice
        from toknt.tui.widgets import _BLOCK_GLYPHS

        assert _BLOCK_GLYPHS[0] == "▁"


class TestPie:
    def test_donut_covers_expected_area(self) -> None:
        png, rgba = graphics.pie_png(width=80, height=80, weights=[3, 2, 1])
        opaque = sum(1 for i in range(3, len(rgba), 4) if rgba[i] > 0)
        expected = 3.14159 / 4 * (1 - 0.45**2)
        assert opaque / (80 * 80) == pytest.approx(expected, abs=0.05)

    def test_empty_weights_render_blank(self) -> None:
        png, rgba = graphics.pie_png(width=40, height=40, weights=[])
        assert all(rgba[i] == 0 for i in range(3, len(rgba), 4))

    def test_zero_only_weights_render_blank(self) -> None:
        _png, rgba = graphics.pie_png(width=40, height=40, weights=[0, 0])
        assert all(rgba[i] == 0 for i in range(3, len(rgba), 4))


class TestCacheRates:
    def test_schemes_use_their_own_denominator(self, conn) -> None:
        from toknt.analytics import aggregation
        from toknt.models.usage import TokenCounts, TokenScheme
        from toknt.storage import ingestion
        from toknt.util import now_ms

        events = []
        for index, (scheme, input_tokens, cache_read) in enumerate([
            (TokenScheme.ANTHROPIC, 100, 300),   # rate 300/400 = 75%
            (TokenScheme.OPENAI, 400, 100),      # rate 100/400 = 25%
            (TokenScheme.UNKNOWN, 500, 0),       # no rate: unknown scheme
        ]):
            events.append(
                __import__("toknt.models.events", fromlist=["UsageEvent"]).UsageEvent(
                    id=f"u{index}",
                    agent="opencode",
                    source_id="opencode:db",
                    session_id=f"s{index}",
                    ts_ms=now_ms(),
                    model=f"m{index}",
                    tokens=TokenCounts(input=input_tokens, output=10, cache_read=cache_read),
                    token_scheme=scheme,
                )
            )
        ingestion.upsert_usage_events(conn, events)
        rates = {
            rate.model: rate
            for rate in aggregation.model_cache_rates(conn, aggregation.UsageFilter())
        }
        assert rates["m0"].rate == pytest.approx(0.75)
        assert rates["m1"].rate == pytest.approx(0.25)
        assert rates["m2"].rate is None  # unknown scheme never invents a rate
        # the canonical key is what the table rows join on; the label is not
        assert rates["m0"].identity == "m0"
        assert rates["m2"].identity == "m2"


class TestHarnessFocus:
    def make_app(self) -> tuple[App, list[bytes]]:
        writes: list[bytes] = []
        app = App(AppConfig(), stdin_fd=0, stdout_fd=1)
        app.term = SimpleNamespace(write=writes.append, writes=writes)
        app.ctx = TerminalContext(backend="off", theme="dark")
        return app, writes

    def test_h_key_cycles_to_a_harness_with_data(self, data_dir, seeded_conn) -> None:
        app, _writes = self.make_app()
        assert app.state.help is False
        # 'h' walks the harnesses that have tokens in the window (the seeded
        # claude_code event is incomplete → zero visible tokens → excluded)
        # and returns to 'all' after the last one.
        assert app.apply_key(KeyEvent("char", "h")) is True
        assert app.state.agents.active == frozenset({"opencode"})
        app.apply_key(KeyEvent("char", "h"))
        assert app.state.agents.active == frozenset({"codex"})
        app.apply_key(KeyEvent("char", "h"))  # last harness with data → all
        assert len(app.state.agents.active) == 22  # back to all

    def test_help_overlay_keeps_chart_images(self, data_dir, seeded_conn) -> None:
        app, writes = self.make_app()
        app.ctx = TerminalContext(
            backend="kitty", cell_w=10, cell_h=20, theme="dark", bg_rgb=0x101010
        )
        app.painter.resize(34, 120)
        app.state.view = 1  # models: the only view with a bitmap chart
        app.paint()
        placed = sum(1 for w in writes if b"a=T,f=100" in w)
        assert placed == 2  # chart + donut bitmaps on screen
        writes.clear()
        app.state.help = True
        app.paint()
        # the help text is painted but the charts are NOT erased or re-emitted
        assert not any(b"a=d" in w for w in writes)
        assert not any(b"a=T,f=100" in w for w in writes)
        screen = _screen(app, 34, 120)
        assert any("toknt interactive" in line for line in screen.values())


class TestContributionsStrips:
    def make_app(self) -> tuple[App, list[bytes]]:
        writes: list[bytes] = []
        app = App(AppConfig(), stdin_fd=0, stdout_fd=1)
        app.term = SimpleNamespace(write=writes.append, writes=writes)
        app.ctx = TerminalContext(backend="off", theme="dark")
        return app, writes

    def test_share_lists_render(self, data_dir, seeded_conn) -> None:
        app, _writes = self.make_app()
        app.painter.resize(34, 120)
        app.paint()
        screen = _screen(app, 34, 120)
        joined = "\n".join(screen.values())
        assert "share by harness" in joined
        assert "share by project" in joined
        assert "opencode" in joined

    def test_legend_lists_harnesses_with_data(self, data_dir, seeded_conn) -> None:
        app, _writes = self.make_app()
        app.painter.resize(34, 120)
        app.paint()
        joined = "\n".join(_screen(app, 34, 120).values())
        for harness in ("opencode", "codex", "claude_code"):
            assert harness in joined, f"{harness} missing from the view"
        # idle harnesses are collapsed instead of announced one by one
        assert "(no data)" not in joined

    def test_narrow_legend_collapses_idle_harnesses(self, data_dir, seeded_conn) -> None:
        app, _writes = self.make_app()
        app.painter.resize(34, 80)
        app.paint()
        joined = "\n".join(_screen(app, 34, 80).values())
        assert "opencode" in joined
        assert "idle" in joined
        assert "(no data)" not in joined
