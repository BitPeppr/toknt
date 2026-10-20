"""CLI smoke tests: initialization, empty states and doctor output."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from typer.testing import CliRunner

from toknt import config
from toknt.cli import app

runner = CliRunner()


def invoke(*args: str):
    return runner.invoke(app, list(args))


class TestVersion:
    def test_version_flag(self) -> None:
        result = invoke("--version")
        assert result.exit_code == 0
        assert "toknt" in result.stdout


class TestScan:
    def test_scan_with_no_adapters_is_clean(self, data_dir: Path, make_config) -> None:
        make_config("[adapters]\nenabled = []\n")
        result = invoke("scan")
        assert result.exit_code == 0
        assert config.db_path().exists()
        assert "No adapters enabled" in result.stdout

    def test_scan_full_reports_the_mode(self, data_dir: Path, make_config) -> None:
        make_config("[adapters]\nenabled = []\n")
        result = invoke("scan", "--full")
        assert result.exit_code == 0
        assert "mode: full" in result.stdout.lower()


class TestDoctor:
    def test_doctor_reports_schema_and_tables(self, data_dir: Path) -> None:
        result = invoke("doctor")
        assert result.exit_code == 0
        from toknt.storage.migrations import MIGRATIONS

        latest = max(migration.version for migration in MIGRATIONS)
        assert f"schema v{latest}" in result.stdout
        for table in ("usage_events", "tool_events", "sources"):
            assert table in result.stdout

    def test_doctor_reports_no_sources(self, data_dir: Path) -> None:
        result = invoke("doctor")
        assert "No sources registered yet" in result.stdout


class TestOverview:
    def test_empty_state_guides_the_user(self, data_dir: Path) -> None:
        result = invoke()
        assert result.exit_code == 0
        assert "No usage data" in result.stdout
        assert "toknt scan" in result.stdout

    def test_with_data_shows_totals(self, data_dir: Path, seeded_conn: sqlite3.Connection) -> None:
        result = invoke()
        assert result.exit_code == 0
        assert "usage events" in result.stdout
        assert "total tokens" in result.stdout
