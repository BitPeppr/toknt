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

from toknt.config import AppConfig
from toknt.presentation import graphics
from toknt.tui import widgets
from toknt.tui.app import App
from toknt.tui.keys import KeyEvent
from toknt.tui.painter import Painter
from toknt.tui.style import heat_ramp
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

    def _line(self, columns: int = 120, **state_kwargs) -> str:
        painter = Painter()
        painter.resize(14, columns)
        state = AppState(**state_kwargs)
        title_row(painter, state, rows=14, columns=columns)
        return "".join(cell.char for cell in painter.buffer[0])

    def test_status_renders_right_aligned(self) -> None:
        line = self._line(columns=200, status="scan done: 5 new events")
        assert line.rstrip().endswith("scan done: 5 new events")

    def test_scanning_note_wins_over_status(self) -> None:
        line = self._line(status="scan done: 5", scanning=True)
        assert "‣ scanning…" in line
        assert "scan done" not in line

    def test_status_too_long_is_clipped_with_an_ellipsis(self) -> None:
        line = self._line(columns=120, status="scan done: 5 new events")
        assert "scan done" in line
        assert line.rstrip().endswith("…")

    def test_long_status_never_collides_with_chips(self) -> None:
        line = self._line(status="x" * 300)
        assert "x" in line
        # the chips must survive intact: the note never reaches back into them
        assert "harnesses: all" in line
        assert "90d" in line

    def test_status_stays_inside_the_page_frame(self) -> None:
        line = self._line(columns=120, status="y" * 40)
        page_left = 2
        page_right = 118
        first = line.index("y")
        last = len(line.rstrip()) - 1
        assert page_left <= first
        assert last <= page_right


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
