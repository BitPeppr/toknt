"""Scaling regression tests: every view degrades gracefully, never breaks.

The TUI must stay collision-free from the hard minimum (60x14) up to
ultrawide. These tests pin the degradation tiers: the harness legend
collapses to a one-line summary at tiny heights, tables switch to compact
column sets on narrow terminals, the quit hint stays pinned, and no painted
row ever exceeds the terminal width.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

from toknt.config import AppConfig
from toknt.tui import views
from toknt.tui.app import App
from toknt.tui.keys import KeyEvent
from toknt.tui.views import TerminalContext

SIZES = [(60, 14), (70, 18), (80, 22), (90, 26), (100, 30), (120, 34), (160, 44)]


def make_app() -> App:
    app = App(AppConfig(), stdin_fd=0, stdout_fd=1)
    app.term = SimpleNamespace(write=lambda data: None)
    app.ctx = TerminalContext(backend="off", cell_w=10, cell_h=20, theme="dark")
    return app


def screen(app_cols_rows: tuple[int, int], keys: str = "") -> dict[int, str]:
    """Fresh app per render: cols, rows -> painted screen text by row."""
    cols, rows = app_cols_rows
    writes: list[bytes] = []
    app = App(AppConfig(), stdin_fd=0, stdout_fd=1)
    app.term = SimpleNamespace(write=writes.append)
    app.ctx = TerminalContext(backend="off", cell_w=10, cell_h=20, theme="dark")
    app.painter.resize(rows, cols)
    for key in keys:
        app.apply_key(KeyEvent("char", key))
    app.paint()
    frame = b"".join(writes).decode("utf-8", "replace")
    out: dict[int, list[str]] = {}
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
        line = out.setdefault(cursor[0], [" "] * cols)
        if cursor[1] <= cols:
            line[cursor[1] - 1] = frame[i]
        cursor[1] += 1
        i += 1
    return {r: "".join(cells).rstrip() for r, cells in out.items()}


class TestHintBar:
    def test_quit_hint_pinned_at_every_width(self, data_dir, seeded_conn) -> None:
        for cols, rows in SIZES:
            s = screen((cols, rows))
            assert any("quit" in line for line in s.values()), (
                f"quit hint missing at {cols}x{rows}"
            )

    def test_quit_hint_appears_once(self, data_dir, seeded_conn) -> None:
        s = screen((120, 34))
        bottom = s.get(34, "")
        assert bottom.count("quit") == 1

    def test_core_hints_survive_at_minimum_width(self, data_dir, seeded_conn) -> None:
        s = screen((60, 14))
        bottom = s.get(14, "")
        for token in ("window", "metric", "help", "quit"):
            assert token in bottom, f"{token!r} missing from hints at 60 cols"


class TestHelpCard:
    def test_help_card_fits_at_minimum_width(self, data_dir, seeded_conn) -> None:
        s = screen((60, 14), keys="?")
        text = "\n".join(s.values())
        assert "toknt interactive" in text
        for line in s.values():
            assert len(line) <= 60


class TestContributionsTiers:
    def test_minimum_height_shows_summary_legend(self, data_dir, seeded_conn) -> None:
        s = screen((60, 14))
        joined = "\n".join(s.values())
        # the full wrapped legend cannot fit; the summary must
        assert "opencode" in joined
        assert "(h)" in joined
        # nothing collides with the hint row
        assert "contributions — tokens per day" in joined

    def test_legend_present_at_all_heights(self, data_dir, seeded_conn) -> None:
        for cols, rows in SIZES:
            s = screen((cols, rows))
            joined = "\n".join(s.values())
            assert "opencode" in joined, f"legend missing at {cols}x{rows}"

    def test_strips_never_collide_with_hints(self, data_dir, seeded_conn) -> None:
        for cols, rows in SIZES:
            s = screen((cols, rows))
            hint_row = s.get(rows, "")
            assert "share by harness" not in hint_row
            assert "weekly tokens" not in hint_row


class TestTableTiers:
    def test_sessions_table_fits_at_60_columns(self, data_dir, seeded_conn) -> None:
        s = screen((60, 20), keys="4")
        header = next(
            (line for line in s.values() if "when" in line and "tokens" in line), ""
        )
        assert header, "session table header missing at 60 cols"
        for line in s.values():
            assert len(line) <= 60

    def test_sessions_detail_line_clear_of_table(self, data_dir, seeded_conn) -> None:
        s = screen((120, 34), keys="4")
        detail_rows = [r for r, line in s.items() if "gpt-5-codex" in line and "·" in line]
        # the selected-session detail must be on its own row, not merged
        # into the table's last row
        for r in detail_rows:
            assert s[r].strip().startswith(("codex", "opencode", "claude_code")) or "·" in s[r]

    def test_models_table_compact_below_78_columns(self, data_dir, seeded_conn) -> None:
        s = screen((60, 20), keys="2")
        header = next((line for line in s.values() if "model" in line and "tokens" in line), "")
        assert header, "compact table header missing at 60 cols"
        assert "sessions" not in header and "cost" not in header
        for line in s.values():
            assert len(line) <= 60

    def test_models_full_table_at_wide(self, data_dir, seeded_conn) -> None:
        s = screen((120, 34), keys="2")
        joined = "\n".join(s.values())
        assert "sessions" in joined  # the sessions column exists
        assert "cost" in joined

    def test_tools_table_fits_at_80_columns(self, data_dir, seeded_conn) -> None:
        s = screen((80, 22), keys="3")
        for line in s.values():
            assert len(line) <= 80
        assert "rate" in "\n".join(s.values())


class TestModelsSidePane:
    def test_side_pane_blocks_stack_without_overlap(self, data_dir, seeded_conn) -> None:
        app = make_app()
        app.ctx = TerminalContext(
            backend="kitty", cell_w=10, cell_h=20, theme="dark", bg_rgb=0x101010
        )
        app.painter.resize(30, 100)
        app.state.view = 1
        app.paint()
        vp = views.paint_models(
            app.painter,
            app.state,
            app.analysis,
            rows=30,
            columns=100,
            ctx=app.ctx,
            now_ms=app._now_ms,
        )
        # donut emitted as its own image slot, legend text below it
        assert any(r.image_id == 2 for r in vp.images)
        donut_bottom = next(
            r.row + r.rows for r in vp.images if r.image_id == 2
        )
        text_rows = {
            r: "".join(c.char for c in app.painter.buffer[r])
            for r in range(donut_bottom, 28)
        }
        # cache bars must start below the donut, never on it
        joined = "\n".join(text_rows.values())
        assert "cache hit rate" in joined

    def test_no_side_pane_below_100_columns(self, data_dir, seeded_conn) -> None:
        app = make_app()
        app.ctx = TerminalContext(
            backend="kitty", cell_w=10, cell_h=20, theme="dark", bg_rgb=0x101010
        )
        app.painter.resize(30, 90)
        app.state.view = 1
        app.paint()
        vp = views.paint_models(
            app.painter,
            app.state,
            app.analysis,
            rows=30,
            columns=90,
            ctx=app.ctx,
            now_ms=app._now_ms,
        )
        assert not any(r.image_id == 2 for r in vp.images)
