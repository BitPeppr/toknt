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
from pathlib import Path

import pytest

from toknt import config
from toknt.models.events import ToolEvent, UsageEvent
from toknt.models.usage import TokenCounts
from toknt.storage import database, ingestion

FIXTURES_DIR = Path(__file__).parent / "fixtures"


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
