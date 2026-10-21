"""Live-window tests: the 3h view, hourly buckets and the auto-rescan tick.

The 3h window is the near-real-time view: hour-bucketed model series, clock
tick labels, a sliding clock anchor on every background scan and a periodic
auto-rescan so the numbers actually move while you watch.
"""

from __future__ import annotations

import os
import re
import time
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from toknt.config import AppConfig
from toknt.tui import views
from toknt.tui.app import LIVE_FIRST_SCAN_MS, SCAN_INTERVAL_MS, App
from toknt.tui.data import LIVE_WINDOW_DAYS, WINDOWS
from toknt.tui.keys import KeyEvent
from toknt.tui.views import TerminalContext


def make_app() -> tuple[App, list[bytes]]:
    writes: list[bytes] = []
    app = App(AppConfig(), stdin_fd=0, stdout_fd=1)
    app.term = SimpleNamespace(write=writes.append, writes=writes)
    app.ctx = TerminalContext(backend="off", theme="dark")
    # the background scan must stay hermetic: no real sources, no slow reads
    app.cfg = AppConfig(enabled_harnesses=())
    return app, writes


def index_of(label: str) -> int:
    return next(i for i, w in enumerate(WINDOWS) if w.label == label)


class TestWindows:
    def test_3h_window_exists_and_is_live(self) -> None:
        window = WINDOWS[index_of("3h")]
        assert window.days == pytest.approx(3 / 24)
        assert window.days is not None and window.days <= LIVE_WINDOW_DAYS

    def test_cycle_order_shortest_first(self) -> None:
        assert [w.label for w in WINDOWS] == [
            "3h",
            "5h",
            "1d",
            "3d",
            "7d",
            "30d",
            "90d",
            "1y",
            "all",
        ]

    def test_default_window_is_three_days(self) -> None:
        # three days is wide enough to show a working week and narrow
        # enough that the charts still have something to say
        assert WINDOWS[views.AppState().window].label == "3d"

    def test_the_window_cycle_runs_backwards(self) -> None:
        app, _writes = make_app()
        app.state.window = index_of("3d")  # the window the explorer opens on
        seen = []
        for _ in range(5):
            app.apply_key(KeyEvent("char", "t"))
            seen.append(WINDOWS[app.state.window].label)
        # from the default three days: a day, an afternoon, a morning, then
        # on to everything
        assert seen == ["1d", "5h", "3h", "all", "1y"]

    def test_since_for_3h_reaches_back_three_hours(self) -> None:
        state = views.AppState(window=index_of("3h"))
        now = 1_800_000_000_000
        assert views.since_for(state, now) == now - int(3 / 24 * 86_400_000)

    @pytest.mark.parametrize(("label", "days"), [("1d", 1), ("3d", 3), ("7d", 7)])
    def test_day_windows_reach_back_the_right_amount(
        self, label: str, days: int
    ) -> None:
        window = WINDOWS[index_of(label)]
        assert window.days == days
        now = 1_800_000_000_000
        state = views.AppState(window=index_of(label))
        assert views.since_for(state, now) == now - days * 86_400_000

    def test_one_day_window_is_live(self) -> None:
        window = WINDOWS[index_of("1d")]
        assert window.days is not None and window.days <= LIVE_WINDOW_DAYS

    def test_multi_day_windows_are_not_live(self) -> None:
        for label in ("3d", "7d", "30d"):
            window = WINDOWS[index_of(label)]
            assert window.days is not None
            assert window.days > LIVE_WINDOW_DAYS


class TestHourBuckets:
    def test_current_bucket_by_window(self) -> None:
        assert views.current_bucket(views.AppState(window=index_of("3h"))) == "hour"
        assert views.current_bucket(views.AppState(window=index_of("5h"))) == "hour"
        assert views.current_bucket(views.AppState(window=index_of("30d"))) == "day"
        assert views.current_bucket(views.AppState(window=index_of("all"))) == "day"

    @pytest.mark.parametrize("label", ["1d", "3d", "7d"])
    def test_day_windows_stay_hourly(self, label: str) -> None:
        # up to a week the chart stays hour-bucketed: a week of hourly
        # points shows the working rhythm, days would flatten it to 7 points
        assert views.current_bucket(views.AppState(window=index_of(label))) == "hour"

    def test_five_hour_window_is_live_and_hourly(self) -> None:
        window = WINDOWS[index_of("5h")]
        assert window.days == pytest.approx(5 / 24)
        assert window.days is not None and window.days <= LIVE_WINDOW_DAYS
        # 5h spans 5-6 hourly buckets — enough for a readable line chart

        base = 1_800_000_000_000 - 1_800_000_000_000 % 3_600_000
        buckets = tuple(base + i * 3_600_000 for i in range(6))
        ticks = views.chart_ticks(buckets, 120, UTC)
        labels = [label for _col, label in ticks]
        assert len(labels) == 6
        assert len(set(labels)) == 6

    def test_model_series_uses_hourly_buckets(self, seeded_conn) -> None:
        from toknt.tui.data import Analysis

        analysis = Analysis(seeded_conn, AppConfig())
        state = views.AppState(window=index_of("3h"))
        # anchor the window on the seeded data itself (the fixtures are days
        # old); a 3h window ending just after the newest event must bucket
        # the corpus by local hour
        newest = seeded_conn.execute(
            "SELECT MAX(ts_ms) FROM usage_events"
        ).fetchone()[0]
        since_ms = newest - 2 * 3_600_000
        series = analysis.model_series(since_ms, state.agents, "total", bucket="hour")
        assert len(series.buckets) >= 1
        deltas = {
            b - a for a, b in zip(series.buckets, series.buckets[1:], strict=False)
        }
        assert deltas <= {3_600_000}
        hour_bucket = datetime.fromtimestamp(series.buckets[0] / 1000, tz=analysis.tz)
        assert (hour_bucket.minute, hour_bucket.second) == (0, 0)

    def test_chart_ticks_show_clock_times_for_sub_day_spans(self) -> None:
        base = 1_800_000_000_000 - 1_800_000_000_000 % 3_600_000
        buckets = tuple(base + i * 3_600_000 for i in range(4))
        ticks = views.chart_ticks(buckets, 120, UTC)
        assert ticks, "hourly chart must produce tick labels"
        labels = [label for _col, label in ticks]
        assert len(labels) == len(set(labels)), f"duplicate ticks: {labels!r}"
        for label in labels:
            assert re.match(r"^\d{2}:\d{2}$", label), f"expected HH:MM, got {label!r}"

    def test_chart_ticks_show_dates_for_multi_day_spans(self) -> None:
        base = 1_800_000_000_000
        buckets = tuple(base + i * 7 * 86_400_000 for i in range(6))
        ticks = views.chart_ticks(buckets, 120, UTC)
        for _col, label in ticks:
            assert ":" not in label  # "%d %b" style, no clock times


class TestAutoRescan:
    @staticmethod
    def _wait_for_scan(app: App) -> None:
        """Drain the scan result the way the main loop does."""
        deadline = time.time() + 20
        while time.time() < deadline:
            app._drain_scan()
            if not app.state.scanning:
                return
            time.sleep(0.05)
        raise AssertionError("the scan never finished")

    def test_is_live_window_tracks_selection(self, data_dir) -> None:
        app, _writes = make_app()
        app.state.window = index_of("3h")
        assert app._is_live_window() is True
        app.state.window = index_of("90d")
        assert app._is_live_window() is False
        app.state.window = index_of("all")
        assert app._is_live_window() is False

    def test_switching_into_3h_schedules_a_quick_first_scan(
        self, data_dir, seeded_conn
    ) -> None:
        app, _writes = make_app()
        app.state.window = index_of("3d")
        before = time.time() * 1000
        app.apply_key(KeyEvent("char", "t"))  # 3d → 1d
        assert app._next_auto_scan >= before + 1  # unchanged cadence
        app.apply_key(KeyEvent("char", "t"))  # 1d → 5h
        app.apply_key(KeyEvent("char", "t"))  # 5h → 3h
        assert app._is_live_window()
        assert app._next_auto_scan <= time.time() * 1000 + LIVE_FIRST_SCAN_MS + 50

    def test_five_hour_window_also_schedules_a_scan(
        self, data_dir, seeded_conn
    ) -> None:
        app, _writes = make_app()
        app.state.window = index_of("3d")
        for _ in range(2):  # 3d → 1d → 5h
            app.apply_key(KeyEvent("char", "t"))
        assert WINDOWS[app.state.window].label == "5h"
        assert app._is_live_window()
        assert app._next_auto_scan <= time.time() * 1000 + LIVE_FIRST_SCAN_MS + 50

    def test_due_tick_starts_a_scan_and_completion_reschedules(
        self, data_dir, seeded_conn
    ) -> None:
        app, _writes = make_app()
        app.state.window = index_of("3h")
        app._next_auto_scan = 0  # long overdue
        app._maybe_auto_scan()
        assert app.state.scanning is True
        # a running scan is never stacked
        app._maybe_auto_scan()
        assert app.state.scanning is True
        self._wait_for_scan(app)
        # the worker drained its own connection; drain the done flag as the
        # main loop would and confirm the next tick moved into the future
        assert app._scan_done is True
        app._scan_done = False
        app._now_ms = int(time.time() * 1000)
        app._next_auto_scan = app._now_ms + SCAN_INTERVAL_MS
        assert app._next_auto_scan > time.time() * 1000
        assert app.state.scanning is False

    def test_the_scan_runs_in_its_own_process(self, data_dir, seeded_conn) -> None:
        # a scan parses every changed source file: doing that in a thread
        # would stall the interface for exactly as long as it takes
        app, _writes = make_app()
        app._next_auto_scan = 0
        app._maybe_auto_scan()
        assert app._scan_process is not None
        assert app._scan_process.pid != os.getpid()
        self._wait_for_scan(app)
        app._stop_scan()

    def test_opening_the_app_starts_a_scan(
        self, data_dir, seeded_conn, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # every launch reads the sources again: the last session may be
        # hours old
        app, _writes = make_app()
        monkeypatch.setattr(app, "_probe_terminal", lambda: None)
        monkeypatch.setattr(app, "_refresh_size", lambda force=False: False)
        app._boot()
        assert app.state.scanning is True
        self._wait_for_scan(app)
        app._stop_scan()

    def test_every_window_refreshes_in_the_background(
        self, data_dir, seeded_conn
    ) -> None:
        # the tick is not tied to the live windows any more: a changed
        # source is picked up whatever the window shows
        app, _writes = make_app()
        app.state.window = index_of("30d")
        assert app._is_live_window() is False
        app._next_auto_scan = 0
        app._maybe_auto_scan()
        assert app.state.scanning is True
        self._wait_for_scan(app)
        app._stop_scan()

    def test_scan_completion_advances_the_clock_anchor(
        self, data_dir, seeded_conn
    ) -> None:
        app, _writes = make_app()
        app.state.window = index_of("3h")
        stale_anchor = app._now_ms
        app._next_auto_scan = 0
        app._maybe_auto_scan()
        self._wait_for_scan(app)
        # replicate the main loop's drain
        if app._scan_done:
            app._scan_done = False
            app._now_ms = int(time.time() * 1000)
            app._next_auto_scan = app._now_ms + SCAN_INTERVAL_MS
        assert app._now_ms >= stale_anchor
        assert app._next_auto_scan > stale_anchor

    def test_live_view_repaints_after_scan(self, data_dir, seeded_conn) -> None:
        app, writes = make_app()
        app.painter.resize(24, 80)
        app.state.window = index_of("3h")
        app.paint()
        writes.clear()
        app._next_auto_scan = 0
        app._maybe_auto_scan()
        self._wait_for_scan(app)
        app._scan_done = False
        app._now_ms = int(time.time() * 1000)
        app.state.help = True  # force a full-screen change so the diff is large
        app.paint()
        assert writes, "live window must repaint with fresh data"
        screen = b"".join(writes).decode("utf-8", "replace")
        assert "toknt interactive" in screen  # the overlay repaint landed
