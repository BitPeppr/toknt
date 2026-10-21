"""Shared test fixtures.

Everything here is synthetic: tests never read the developer's real
coding-agent history. The `data_dir` fixture points ``TOKNT_DATA_DIR`` at a
temporary directory and ``TOKNT_CONFIG`` at a path that does not exist, so
every test gets its own hermetic database *and* default configuration.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from toknt import config
from toknt.models.events import ToolEvent, UsageEvent
from toknt.models.usage import TokenCounts
from toknt.storage import database, ingestion

FIXTURES_DIR = Path(__file__).parent / "fixtures"

#: The clock the TUI tests run at. The corpus is stamped at fixed dates
#: (2-10 October 2026) while the windows are relative to the clock: with a
#: real clock the corpus would quietly fall out of the window as the
#: calendar moves on. Everything the views show is then deterministic.
FIXED_NOW_MS = int(datetime(2026, 10, 12, 12, 0, tzinfo=UTC).timestamp() * 1000)


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "toknt-data"
    monkeypatch.setenv(config.DATA_DIR_ENV, str(path))
    monkeypatch.setenv(config.CONFIG_FILE_ENV, str(tmp_path / "toknt-config.toml"))
    return path


def write_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, text: str) -> Path:
    """Write a config file and point ``TOKNT_CONFIG`` at it for one test."""
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setenv(config.CONFIG_FILE_ENV, str(path))
    return path


@pytest.fixture
def make_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Factory fixture: ``make_config("[adapters]\\nenabled = []")``."""

    def _make(text: str) -> Path:
        return write_config(monkeypatch, tmp_path, text)

    return _make


@pytest.fixture
def conn(data_dir: Path) -> Iterator[sqlite3.Connection]:
    connection = database.open_database()
    yield connection
    connection.close()


def _load_jsonl(name: str) -> list[dict]:
    text = (FIXTURES_DIR / name).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_usage_events() -> list[UsageEvent]:
    events = []
    for record in _load_jsonl("canonical_usage_events.jsonl"):
        record = dict(record)
        record["tokens"] = TokenCounts(**record.get("tokens", {}))
        record["quality_notes"] = tuple(record.get("quality_notes", ()))
        events.append(UsageEvent(**record))
    return events


def load_tool_events() -> list[ToolEvent]:
    events = []
    for record in _load_jsonl("canonical_tool_events.jsonl"):
        record = dict(record)
        record["quality_notes"] = tuple(record.get("quality_notes", ()))
        events.append(ToolEvent(**record))
    return events


@pytest.fixture
def usage_events() -> list[UsageEvent]:
    return load_usage_events()


@pytest.fixture
def tool_events() -> list[ToolEvent]:
    return load_tool_events()


@pytest.fixture
def seeded_conn(
    conn: sqlite3.Connection,
    usage_events: list[UsageEvent],
    tool_events: list[ToolEvent],
) -> sqlite3.Connection:
    """Connection pre-populated with the canonical fixture events."""
    ingestion.upsert_usage_events(conn, usage_events)
    ingestion.upsert_tool_events(conn, tool_events)
    return conn


@pytest.fixture(autouse=True)
def pinned_app_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the TUI at a fixed moment, on a window that holds the corpus.

    The explorer opens on a three-day window; the test corpus is stamped
    days before the pinned clock, so the app under test asks for a wider
    window. The default itself is asserted where it belongs — on
    ``AppState``.
    """
    from toknt.tui.app import App
    from toknt.tui.data import WINDOWS

    window = next(
        index for index, item in enumerate(WINDOWS) if item.label == "30d"
    )
    original = App.__init__

    def pinned(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self._now_ms = FIXED_NOW_MS
        self.state.window = window

    monkeypatch.setattr(App, "__init__", pinned)
