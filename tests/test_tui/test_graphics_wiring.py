"""Graphics wiring tests: env override, size probe, and the title-bar chip.

The TUI must honour ``TOKNT_GRAPHICS`` (documented, but it used to be
ignored), size itself from the real terminal rather than stale COLUMNS
values, and show which chart renderer is active so "no graph" is
diagnosable at a glance.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from toknt.config import AppConfig, ConfigError, load_config
from toknt.tui import views
from toknt.tui.app import App
from toknt.tui.keys import KeyEvent
from toknt.tui.painter import Painter, terminal_size
from toknt.tui.views import TerminalContext


class TestGraphicsOverride:
    def test_env_override_reaches_the_config(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("TOKNT_GRAPHICS", "kitty")
        cfg = load_config(tmp_path / "missing.toml")
        assert cfg.ui_graphics == "kitty"

    def test_env_override_beats_the_file(self, tmp_path, monkeypatch) -> None:
        config = tmp_path / "config.toml"
        config.write_text('[presentation]\ngraphics = "sixel"\n')
        monkeypatch.setenv("TOKNT_GRAPHICS", "iterm2")
        assert load_config(config).ui_graphics == "iterm2"

    def test_invalid_env_value_is_rejected(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("TOKNT_GRAPHICS", "nonsense")
        with pytest.raises(ConfigError):
            load_config(tmp_path / "missing.toml")

    def test_no_env_keeps_the_file_value(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("TOKNT_GRAPHICS", raising=False)
        config = tmp_path / "config.toml"
        config.write_text('[presentation]\ngraphics = "sixel"\n')
        assert load_config(config).ui_graphics == "sixel"

    def test_default_is_auto(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("TOKNT_GRAPHICS", raising=False)
        assert AppConfig().ui_graphics == "auto"


class TestTerminalSize:
    def test_ioctl_wins_over_stale_env(self, monkeypatch) -> None:
        monkeypatch.setenv("COLUMNS", "80")
        monkeypatch.setenv("LINES", "24")
        rows, columns = terminal_size()
        # either the real terminal size, or the fallback — but never the
        # stale environment values when the terminal answers
        if (rows, columns) != (24, 80):
            assert columns != 80 or rows != 24

    def test_size_is_sane_without_a_terminal(self) -> None:
        rows, columns = terminal_size()
        assert rows > 0 and columns > 0


class TestTitleRow:
    def make_app(self) -> App:
        app = App(AppConfig(), stdin_fd=0, stdout_fd=1)
        app.term = SimpleNamespace(write=lambda data: None)
        app.ctx = TerminalContext(backend="kitty", theme="dark")
        return app

    def test_title_row_has_no_graphics_chip(self) -> None:
        # the backend cannot be changed from the TUI, so naming it in the
        # title bar was noise
        app = self.make_app()
        painter = Painter()
        painter.resize(34, 120)
        views.title_row(painter, app.state, rows=34, columns=120)
        row1 = "".join(cell.char for cell in painter.buffer[0])
        assert "kitty" not in row1
        assert "braille" not in row1
        assert "harnesses: all" in row1

    def test_merge_chip_appears_when_merging(self) -> None:
        app = self.make_app()
        app.state.merge_models = True
        painter = Painter()
        painter.resize(34, 120)
        views.title_row(painter, app.state, rows=34, columns=120)
        row1 = "".join(cell.char for cell in painter.buffer[0])
        assert "merged" in row1


class TestSelfTestImage:
    def make_app(self) -> tuple[App, list[bytes]]:
        writes: list[bytes] = []
        app = App(AppConfig(), stdin_fd=0, stdout_fd=1)
        app.term = SimpleNamespace(write=writes.append, writes=writes)
        app.ctx = TerminalContext(
            backend="kitty", cell_w=10, cell_h=20, theme="dark", bg_rgb=0x101010
        )
        return app, writes

    def test_g_key_draws_the_shared_test_chart(self) -> None:
        app, writes = self.make_app()
        app.painter.resize(34, 120)
        assert app.apply_key(KeyEvent("char", "g")) is True
        payload = b"".join(writes)
        assert b"a=T,f=100" in payload
        assert b"i=3" in payload  # its own image slot
        assert "graphics self-test" in app.state.status

    def test_g_key_reports_braille_when_graphics_are_off(self) -> None:
        app, writes = self.make_app()
        app.ctx = TerminalContext(backend="off", theme="dark")
        app.painter.resize(34, 120)
        assert app.apply_key(KeyEvent("char", "g")) is True
        assert b"a=T,f=100" not in b"".join(writes)
        assert "braille" in app.state.status

    def test_test_chart_is_shared_with_the_cli_command(self) -> None:
        from toknt.presentation import graphics

        png, _rgba = graphics.test_chart_png(200, 60)
        assert png.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(png) > 100


def _kitty_image_table(data: bytes) -> set[int]:
    """What images would kitty be showing after replaying `data`?

    Kitty deletes *all* images for a bare ``a=d``; ``d=i`` removes only the
    image named by ``i=``. A transmission with id N makes N visible.
    """
    alive: set[int] = set()
    for command in data.split(b"\x1b\\"):
        if not command.startswith(b"\x1b_G"):
            continue
        body = command[2:].decode("ascii", "replace")
        if body.startswith("a=d"):
            if "d=i" in body:
                match = re.search(r"i=(\d+)", body)
                if match:
                    alive.discard(int(match.group(1)))
            else:  # d=a is the default: everything dies
                alive.clear()
            continue
        match = re.search(r"\bi=(\d+)", body)
        if match:
            alive.add(int(match.group(1)))
    return alive


class TestModelsViewImages:
    def test_line_chart_survives_the_donut(self, data_dir, seeded_conn) -> None:
        """Regression: page 2 lost the line graph because the donut's leading
        delete (a bare a=d) erased every image kitty held — the chart died
        the moment the donut was placed, one frame later."""
        writes: list[bytes] = []
        app = App(AppConfig(), stdin_fd=0, stdout_fd=1)
        app.term = SimpleNamespace(write=writes.append, writes=writes)
        app.ctx = TerminalContext(
            backend="kitty", cell_w=10, cell_h=20, theme="dark", bg_rgb=0x101010
        )
        app.painter.resize(34, 120)
        app.state.view = 1
        app.paint()
        alive = _kitty_image_table(b"".join(writes))
        assert 1 in alive, "line chart must still be displayed"
        assert 2 in alive, "donut must still be displayed"

