"""The 'm' toggle: merge one model served by many providers."""

from __future__ import annotations

import re
from types import SimpleNamespace

from tests.support.models import at, usage
from toknt.config import AppConfig
from toknt.storage import ingestion
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


def _screen(app: App, writes: list[bytes], rows: int, cols: int) -> dict[int, str]:
    frame = b"".join(writes).decode("utf-8", "replace")
    out: dict[int, list[str]] = {}
    cursor = [1, 1]
    index = 0
    while index < len(frame):
        move = re.match(r"\x1b\[(\d+);(\d+)H", frame[index:])
        if move:
            cursor = [int(move.group(1)), int(move.group(2))]
            index += move.end()
            continue
        if frame[index] == "\x1b":
            other = re.match(r"\x1b\[[0-9;?]*[a-zA-Z]", frame[index:])
            if other:
                index += other.end()
                continue
            index += 1
            continue
        line = out.setdefault(cursor[0], [" "] * cols)
        if cursor[1] <= cols:
            line[cursor[1] - 1] = frame[index]
        cursor[1] += 1
        index += 1
    return {row: "".join(cells).rstrip() for row, cells in out.items()}


def _second_spelling(seeded_conn) -> None:
    """Add the same model under a provider prefix to the seeded corpus."""
    ingestion.upsert_usage_events(
        seeded_conn,
        [
            usage(
                id="usage:cline:merge-1",
                ts_ms=at(2026, 10, 2, 9),
                agent="cline",
                model="z-ai/glm-5.3-flash",
                input=1010,
                output=0,
                reasoning=0,
                cache_read=0,
                cache_write=0,
            )
        ],
    )


class TestMergeToggle:
    def test_state_defaults_to_split(self, data_dir, seeded_conn) -> None:
        app, _writes = _app()
        assert app.state.merge_models is False

    def test_m_key_toggles_the_state(self, data_dir, seeded_conn) -> None:
        app, _writes = _app()
        app.apply_key(KeyEvent("char", "m"))
        assert app.state.merge_models is True
        app.apply_key(KeyEvent("char", "m"))
        assert app.state.merge_models is False

    def test_models_view_reports_the_merge_in_the_heading(
        self, data_dir, seeded_conn
    ) -> None:
        app, writes = _app()
        app.apply_key(KeyEvent("char", "2"))
        app.apply_key(KeyEvent("char", "m"))
        app.paint()
        joined = "\n".join(_screen(app, writes, 34, 120).values())
        assert "merged across providers" in joined
        assert "· merged" in joined  # the title-bar chip

    def test_split_view_keeps_raw_names(self, data_dir, seeded_conn) -> None:
        app, writes = _app()
        app.apply_key(KeyEvent("char", "2"))
        app.paint()
        joined = "\n".join(_screen(app, writes, 34, 120).values())
        assert "merged across providers" not in joined
        assert "· merged" not in joined


class TestMergedView:
    """The seeded corpus holds one model under two spellings."""

    def _rows(self, merge: bool) -> list[str]:
        app, writes = _app()
        app.apply_key(KeyEvent("char", "2"))
        if merge:
            app.apply_key(KeyEvent("char", "m"))
        app.paint()
        screen = _screen(app, writes, 34, 120)
        return [line for _, line in sorted(screen.items())]

    def _table_rows(self, rows: list[str]) -> list[str]:
        """The body of the models table (the chart legend is above it)."""
        header = next(
            index
            for index, line in enumerate(rows)
            if "model" in line and "tokens" in line and "events" in line
        )
        return [line for line in rows[header + 1 :] if line.strip()]

    def test_split_view_keeps_both_spellings(self, data_dir, seeded_conn) -> None:
        _second_spelling(seeded_conn)
        body = self._table_rows(self._rows(False))
        matching = [line for line in body if "glm-5.3-fla" in line]
        assert len(matching) == 2

    def test_merged_view_shows_one_row_for_the_model(self, data_dir, seeded_conn) -> None:
        _second_spelling(seeded_conn)
        body = self._table_rows(self._rows(True))
        matching = [line for line in body if "glm-5.3-fla" in line]
        assert len(matching) == 1
        assert "z-ai/" not in matching[0]
        assert "free" not in matching[0]

    def test_merged_view_sums_the_tokens(self, data_dir, seeded_conn) -> None:
        _second_spelling(seeded_conn)
        body = self._table_rows(self._rows(True))
        merged = next(line for line in body if "glm-5.3-fla" in line)
        assert "20.0K" in merged  # 18,990 from the seed + 1,010 from the prefix


class TestHelpCard:
    """The help card must fit even on the smallest supported terminal."""

    def test_help_card_fits_at_60x14(self, data_dir, seeded_conn) -> None:
        app, writes = _app(rows=14, cols=60)
        app.apply_key(KeyEvent("char", "?"))
        app.paint()
        screen = _screen(app, writes, 14, 60)
        assert any("toknt interactive" in line for line in screen.values())
        for row, line in screen.items():
            assert len(line) <= 60
            assert row <= 14

    def test_help_mentions_the_merge_toggle(self, data_dir, seeded_conn) -> None:
        app, writes = _app()
        app.apply_key(KeyEvent("char", "?"))
        app.paint()
        joined = "\n".join(_screen(app, writes, 34, 120).values())
        assert "merge models" in joined
