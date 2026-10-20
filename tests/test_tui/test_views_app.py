"""View chrome, chart caching, and the app's bitmap lifecycle."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from toknt.config import AppConfig
from toknt.presentation.graphics import png_encode
from toknt.tui import views
from toknt.tui.app import App
from toknt.tui.data import HARNESS_NAMES, AgentToggles, Analysis
from toknt.tui.painter import BOLD, REVERSE, Painter
from toknt.tui.views import AppState, ImageRequest, TerminalContext, paint_chart, title_row

UTC = ZoneInfo("UTC")
NOW = int(datetime(2026, 6, 19, 12, tzinfo=UTC).timestamp() * 1000)


def _series(*values: float):
    """Build a one-series Series without touching the storage."""
    import dataclasses

    from toknt.analytics import aggregation

    base = aggregation.Series(buckets=(), series={}, dimension="model")
    series_map = {"m": list(values)}  # type: ignore[dict-item]
    buckets = tuple(NOW + index * 86_400_000 for index in range(len(values)))
    return dataclasses.replace(base, buckets=buckets, series=series_map)  # type: ignore[arg-type]


def _fake_png(monkeypatch: pytest.MonkeyPatch, calls: list[tuple[int, int]]) -> None:
    def fake_png(pixel_w: int, pixel_h: int, values):
        calls.append((pixel_w, pixel_h))
        blank = bytes(pixel_w * pixel_h * 4)
        return png_encode(pixel_w, pixel_h, blank), blank

    monkeypatch.setattr(views, "_chart_png", fake_png)


class TestChartCache:
    def ctx(self, backend: str = "kitty") -> TerminalContext:
        return TerminalContext(backend=backend, cell_w=10, cell_h=20, theme="dark")

    def test_repeat_paint_reuses_render(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(views, "_CHART_CACHE", {})
        painter = Painter()
        painter.resize(30, 120)
        calls: list[tuple[int, int]] = []
        _fake_png(monkeypatch, calls)
        first = paint_chart(
            painter, row=2, col=2, width_columns=100, height_rows=12,
            series=_series(1, 5, 3), ctx=self.ctx(), identity=(),
        )
        second = paint_chart(
            painter, row=2, col=2, width_columns=100, height_rows=12,
            series=_series(1, 5, 3), ctx=self.ctx(), identity=(),
        )
        assert first is not None and second is not None
        assert first.payload == second.payload
        assert first.key == second.key
        assert len(calls) == 1, "cache hit must skip the rasterizer"

    def test_changed_geometry_or_state_goes_through_cache_miss(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(views, "_CHART_CACHE", {})
        painter = Painter()
        painter.resize(30, 120)
        calls: list[tuple[int, int]] = []
        _fake_png(monkeypatch, calls)
        common = dict(row=2, col=2, width_columns=100, height_rows=12, ctx=self.ctx())
        paint_chart(painter, series=_series(1, 5, 3), identity=(), **common)
        paint_chart(painter, series=_series(1, 5, 3), identity=(), **{
            **common, "width_columns": 90,
        })  # different geometry
        paint_chart(painter, series=_series(1, 5, 3), identity=(2,), **common
                    )  # different identity (e.g. metric switched)
        paint_chart(painter, series=_series(9, 9, 9), identity=(), **common
                    )  # different data
        assert len(calls) == 4

    def test_cache_is_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(views, "_CHART_CACHE", {})
        painter = Painter()
        painter.resize(30, 120)
        calls: list[tuple[int, int]] = []
        _fake_png(monkeypatch, calls)
        common = dict(row=2, col=2, width_columns=100, height_rows=12, ctx=self.ctx())
        for mark in range(10):
            paint_chart(painter, series=_series(mark, mark), identity=(), **common)
        assert len(views._CHART_CACHE) <= views._CHART_CACHE_LIMIT


class TestChrome:
    def test_title_tabs_and_hints_without_reverse(self) -> None:
        painter = Painter()
        painter.resize(14, 110)
        painter.frame()
        state = AppState()
        title_row(painter, state, rows=14, columns=110)
        header = "".join(cell.char for cell in painter.buffer[0])
        assert "toknt" in header
        for index, name in enumerate(views.VIEW_TITLES):
            assert f"{index + 1} {name}" in header
        hints = "".join(cell.char for cell in painter.buffer[13]).rstrip()
        assert "view" in hints and "quit" in hints and "rescan" in hints
        assert not any(
            cell.mask & REVERSE for line in painter.buffer for cell in line
        ), "chrome must not use reverse video anywhere"

    def test_active_tab_is_bold_others_dim(self) -> None:
        painter = Painter()
        painter.resize(14, 110)
        painter.frame()
        state = AppState(view=1)
        title_row(painter, state, rows=14, columns=110)
        header = "".join(cell.char for cell in painter.buffer[0])
        position = header.find("2 models")
        assert position >= 0
        assert all(
            painter.buffer[0][position + offset].mask & BOLD
            for offset in range(len("2 models"))
        )
        inactive = header.find("1 contributions")
        assert inactive >= 0
        assert not any(
            painter.buffer[0][inactive + offset].mask & BOLD
            for offset in range(len("1 contributions"))
        )

    def test_scanning_chip_appears_when_scanning(self) -> None:
        painter = Painter()
        painter.resize(14, 110)
        painter.frame()
        state = AppState(scanning=True)
        title_row(painter, state, rows=14, columns=110)
        header = "".join(cell.char for cell in painter.buffer[0])
        assert "scanning…" in header


class TestAgentLegend:
    def _paint(
        self, seeded_conn, state: AppState | None = None, columns: int = 150
    ) -> str:
        analysis = Analysis(seeded_conn, AppConfig())
        painter = Painter()
        painter.resize(30, columns)
        painter.frame()
        views.paint_contribution(
            painter, state or AppState(), analysis, rows=30, columns=columns,
            ctx=TerminalContext(backend="off"), now_ms=NOW,
        )
        return "".join(cell.char for line in painter.buffer for cell in line)

    def test_share_rail_lists_harnesses_with_data(self, seeded_conn) -> None:
        text = self._paint(seeded_conn)
        assert "opencode" in text
        assert "codex" in text
        assert "claude_code" in text
        assert "(no data)" not in text  # idle harnesses stay quiet

    def test_toggled_off_agents_get_a_quiet_note(self, seeded_conn) -> None:
        text = self._paint(seeded_conn, AppState(agents=AgentToggles(
            frozenset({"opencode"}),
        )))
        assert "harnesses off" in text
        assert "(o/x/a)" in text

    def test_few_toggled_off_agents_are_named(self, seeded_conn) -> None:
        active = frozenset(
            name for name in HARNESS_NAMES if name not in {"gemini", "grok", "pi"}
        )
        text = self._paint(seeded_conn, AppState(agents=AgentToggles(active)))
        assert "(off)" in text
        assert "gemini" in text

    def test_narrow_legend_collapses_idle_harnesses(self, seeded_conn) -> None:
        text = self._paint(seeded_conn, columns=80)
        assert "opencode" in text
        assert "idle" in text
        assert "(no data)" not in text


class TestAnalysisMemo:
    def test_repeat_state_memoizes_the_query(self, seeded_conn) -> None:
        analysis = Analysis(seeded_conn, AppConfig())
        agents = AgentToggles()
        assert analysis.daily(0, agents) is analysis.daily(0, agents)
        assert analysis.model_series(0, agents, "total") is analysis.model_series(
            0, agents, "total"
        )
        assert analysis.model_rows(0, agents) is analysis.model_rows(0, agents)
        assert analysis.tool_rows(0, agents) is analysis.tool_rows(0, agents)
        assert analysis.session_rows(0, agents, 24) is analysis.session_rows(0, agents, 24)

    def test_state_change_bypasses_memo(self, seeded_conn) -> None:
        analysis = Analysis(seeded_conn, AppConfig())
        everything = AgentToggles()
        only_open = AgentToggles(frozenset({"opencode"}))
        assert analysis.daily(0, everything) is not analysis.daily(0, only_open)

    def test_invalidate_drops_memo_after_scan(self, seeded_conn) -> None:
        analysis = Analysis(seeded_conn, AppConfig())
        agents = AgentToggles()
        first = analysis.daily(0, agents)
        analysis.invalidate()
        assert analysis.daily(0, agents) is not first

    def test_memo_is_bounded(self, seeded_conn) -> None:
        analysis = Analysis(seeded_conn, AppConfig())
        agents = AgentToggles()
        for offset in range(10):
            analysis.daily(offset * 1000, agents)
        assert len(analysis._cache) <= analysis._CACHE_LIMIT


class TestImageLifecycle:
    def make_app(self) -> tuple[App, list[bytes]]:
        writes: list[bytes] = []
        app = App(AppConfig(), stdin_fd=0, stdout_fd=1)
        app.term = SimpleNamespace(write=writes.append)
        app.ctx = TerminalContext(
            backend="kitty", cell_w=10, cell_h=20, theme="dark", bg_rgb=0xABCDEF
        )
        return app, writes

    def test_kitty_places_skips_and_deletes(self, data_dir) -> None:
        app, writes = self.make_app()
        request = ImageRequest(
            row=2, col=2, columns=20, rows=8, payload=b"IMG", key="chart-1",
        )
        app._sync_images_begin((request,))
        app._sync_images_end((request,))
        assert writes == [b"\x1b[3;3HIMG"]
        # identical chart → nothing emitted
        app._sync_images_begin((request,))
        app._sync_images_end((request,))
        assert len(writes) == 1
        # chart leaves → explicit, scoped delete
        app._sync_images_begin(())
        app._sync_images_end(())
        assert writes[1:] == [b"\x1b_Ga=d,d=i,i=1,C=1,q=1\x1b\\"]

    def test_kitty_rect_change_new_payload_self_deletes(self, data_dir) -> None:
        app, writes = self.make_app()
        first = ImageRequest(
            row=2, col=2, columns=20, rows=8, payload=b"IMG", key="chart-1"
        )
        app._sync_images_begin((first,))
        app._sync_images_end((first,))
        second = ImageRequest(
            row=2, col=2, columns=30, rows=10, payload=b"IMG2", key="chart-1"
        )
        app._sync_images_begin((second,))
        app._sync_images_end((second,))
        # the placement payload embeds its own delete; no separate erase write
        assert writes == [b"\x1b[3;3HIMG", b"\x1b[3;3HIMG2"]

    def test_models_view_places_two_images_and_erases_both(
        self, data_dir, seeded_conn
    ) -> None:
        """The models view owns two slots (chart + donut); both erase cleanly."""
        import re as _re

        app, writes = self.make_app()
        app.ctx = TerminalContext(
            backend="kitty", cell_w=10, cell_h=20, theme="dark", bg_rgb=0x101010
        )
        app.painter.resize(34, 120)
        app.state.view = 1
        app.paint()
        payloads = [w for w in writes if b"a=T,f=100" in w]
        ids = sorted(
            _re.search(rb"i=(\d+)", w).group(1).decode() for w in payloads
        )
        assert ids == ["1", "2"]  # chart and donut, distinct kitty image ids
        writes.clear()
        app.state.view = 2  # tools: no images → both slots erased
        app.paint()
        deletes = sorted(
            _re.search(rb"i=(\d+)", w).group(1).decode()
            for w in writes
            if b"a=d" in w
        )
        assert deletes == ["1", "2"]

    def test_sixel_erase_uses_queried_background(self, data_dir) -> None:
        app, writes = self.make_app()
        app.ctx = TerminalContext(
            backend="sixel", cell_w=10, cell_h=20, theme="dark", bg_rgb=0x001122
        )
        first = ImageRequest(
            row=2, col=2, columns=20, rows=8, payload=b"IMG", key="chart-1"
        )
        app._sync_images_begin((first,))
        app._sync_images_end((first,))
        assert writes == [b"\x1b[3;3HIMG"]
        app._sync_images_begin(())  # erase happens before the text frame
        app._sync_images_end(())
        erase = writes[1]
        assert erase.startswith(b"\x1b[3;3H\x1bPq")  # opaque bg-colored image
        assert b"0;17;34" in erase  # exact rgb(0, 17, 34) palette entry
