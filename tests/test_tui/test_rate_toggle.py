"""The 'p' toggle: the models chart switches tokens ⇄ tokens per second."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from tests.support.models import at, usage
from toknt.config import AppConfig
from toknt.storage import ingestion
from toknt.tui.app import App
from toknt.tui.keys import KeyEvent
from toknt.tui.views import TerminalContext


def _app(rows: int = 34, cols: int = 136):
    writes: list[bytes] = []
    app = App(AppConfig(), stdin_fd=0, stdout_fd=1)
    app.term = SimpleNamespace(write=writes.append, writes=writes)
    app.ctx = TerminalContext(backend="off", cell_w=10, cell_h=20, theme="dark")
    app.painter.resize(rows, cols)
    return app, writes


def _screen(app: App, writes: list[bytes], rows: int, cols: int) -> dict[int, str]:
    """Rebuild the painted screen from the app's captured writes."""
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


def _joined(app: App, writes: list[bytes]) -> str:
    screen = _screen(app, writes, app.painter.rows, app.painter.columns)
    return "\n".join(line for _, line in sorted(screen.items()))


def _rate_pair(seeded_conn) -> None:
    """Two messages of one session, 10 seconds apart at a steady 12 tok/s."""
    base = at(2026, 10, 10, 12, 0)
    ingestion.upsert_usage_events(
        seeded_conn,
        [
            usage(
                id="usage:opencode:rate-1",
                ts_ms=base,
                session_id="ses_rate_pair",
                model="rate-model",
                input=0,
                output=0,
                reasoning=0,
                cache_read=0,
                cache_write=0,
            ),
            usage(
                id="usage:opencode:rate-2",
                ts_ms=base + 10_000,
                session_id="ses_rate_pair",
                model="rate-model",
                input=10,
                output=120,
                reasoning=0,
                cache_read=0,
                cache_write=0,
            ),
        ],
    )


def _decorated_pair(seeded_conn) -> None:
    """A pair whose model name merges: the label ('muse-spark-1.3') and the
    identity ('muse-spark-1-3') are different strings."""
    base = at(2026, 10, 10, 12, 0)
    ingestion.upsert_usage_events(
        seeded_conn,
        [
            usage(
                id="usage:opencode:rate-merge-1",
                ts_ms=base,
                session_id="ses_rate_merge",
                model="muse-spark-1.3-contributor-free",
                input=0,
                output=0,
                reasoning=0,
                cache_read=0,
                cache_write=0,
            ),
            usage(
                id="usage:opencode:rate-merge-2",
                ts_ms=base + 10_000,
                session_id="ses_rate_merge",
                model="muse-spark-1.3-contributor-free",
                input=10,
                output=120,
                reasoning=0,
                cache_read=0,
                cache_write=0,
            ),
        ],
    )


class TestRateToggleState:
    def test_off_by_default(self, data_dir, seeded_conn) -> None:
        app, _writes = _app()
        assert app.state.rate_mode is False

    def test_p_toggles_only_on_the_models_view(self, data_dir, seeded_conn) -> None:
        app, _writes = _app()
        app.apply_key(KeyEvent("char", "2"))  # models
        assert app.apply_key(KeyEvent("char", "p")) is True
        assert app.state.rate_mode is True
        assert app.apply_key(KeyEvent("char", "p")) is True
        assert app.state.rate_mode is False

    def test_other_views_get_feedback_not_a_toggle(self, data_dir, seeded_conn) -> None:
        app, _writes = _app()
        app.apply_key(KeyEvent("char", "1"))  # contributions
        assert app.apply_key(KeyEvent("char", "p")) is True  # repaint: status shows
        assert app.state.rate_mode is False
        assert "models" in app.state.status

    def test_status_clears_when_the_view_changes(self, data_dir, seeded_conn) -> None:
        # a note about the old view must not linger on the new one
        app, _writes = _app()
        app.apply_key(KeyEvent("char", "1"))
        app.apply_key(KeyEvent("char", "p"))
        assert app.state.status
        app.apply_key(KeyEvent("char", "2"))  # models
        assert app.state.status == ""

    def test_ctrl_c_quits(self, data_dir, seeded_conn) -> None:
        # the terminal is in raw mode, so the signal never arrives on its
        # own: the key reference promises ctrl+c quits
        app, _writes = _app()
        with pytest.raises(SystemExit):
            app.apply_key(KeyEvent("ctrl+c"))


def _table_header(joined: str) -> str:
    return next(
        line
        for line in joined.splitlines()
        if "model" in line and "events" in line
    )


class TestRateScreen:
    """136 columns: the table zone is wide enough for the tok/s column."""

    def test_heading_switches_with_the_toggle(self, data_dir, seeded_conn) -> None:
        _rate_pair(seeded_conn)
        app, writes = _app()
        app.apply_key(KeyEvent("char", "2"))
        app.paint()
        assert "tokens by model" in _joined(app, writes)
        assert "tokens per second" not in _joined(app, writes)
        app.apply_key(KeyEvent("char", "p"))
        app.paint()
        joined = _joined(app, writes)
        assert "tokens per second by model" in joined
        assert "output+reasoning" not in joined
        assert "tokens by model" not in joined

    def test_table_carries_the_estimated_rate_column(
        self, data_dir, seeded_conn
    ) -> None:
        _rate_pair(seeded_conn)
        app, writes = _app()
        app.apply_key(KeyEvent("char", "2"))
        app.paint()
        joined = _joined(app, writes)
        assert "tok/s" in _table_header(joined)  # visible before the toggle too
        assert "12.0" in joined  # 120 generated tokens / 10 s

    def test_no_pairs_means_no_rate_column_and_no_data(
        self, data_dir, seeded_conn
    ) -> None:
        # the seeded corpus holds one message per session: nothing pairs
        app, writes = _app()
        app.apply_key(KeyEvent("char", "2"))
        assert app.apply_key(KeyEvent("char", "p")) is True
        app.paint()
        joined = _joined(app, writes)
        assert "tokens per second by model" in joined
        # the window is not empty — only the rate is unmeasurable, and the
        # message says so
        assert "no rate data" in joined
        assert "no data for this window" not in joined
        assert "tok/s" not in _table_header(joined)

    def test_help_card_mentions_p(self, data_dir, seeded_conn) -> None:
        app, writes = _app()
        app.apply_key(KeyEvent("char", "?"))
        app.paint()
        assert any("tok/s" in line for line in _joined(app, writes).splitlines())

    def test_merged_view_keeps_the_rate(self, data_dir, seeded_conn) -> None:
        # the merged row's label and its identity differ; the rate must be
        # found through the identity, or the column reads a row of dashes
        _decorated_pair(seeded_conn)
        app, writes = _app()
        app.apply_key(KeyEvent("char", "2"))
        app.apply_key(KeyEvent("char", "m"))
        app.paint()
        lines = _joined(app, writes).splitlines()
        header = next(
            index
            for index, line in enumerate(lines)
            if "events" in line and "tok/s" in line
        )
        row = next(
            line for line in lines[header + 1 :] if "muse-spark" in line and line.strip()
        )
        assert "12.0" in row
