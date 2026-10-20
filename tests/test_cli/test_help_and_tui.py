"""CLI: `-h` everywhere, static overview paths, TUI escape hatches."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from typer.testing import CliRunner

from tests.support.models import at, usage
from toknt.cli import app
from toknt.storage import ingestion

runner = CliRunner()
WIDE = {"COLUMNS": "200", "LINES": "50"}


def invoke(*args: str):
    return runner.invoke(app, list(args), env=WIDE)


class TestHelpFlag:
    def test_root_help(self) -> None:
        result = invoke("-h")
        assert result.exit_code == 0
        assert "Usage:" in result.stdout
        assert "--help" in result.stdout  # both spellings advertised
        assert "-h" in result.stdout

    def test_every_command_accepts_dash_h(self) -> None:
        commands = (
            ("scan",), ("usage",), ("models",), ("tools",), ("sessions",),
            ("export",), ("doctor",), ("overview",),
        )
        for command in commands:
            result = invoke(*command, "-h")
            assert result.exit_code == 0, command
            assert "Usage:" in result.stdout, command

    def test_pricing_subcommands_accept_dash_h(self) -> None:
        for command in ("add", "list", "remove"):
            result = invoke("pricing", command, "-h")
            assert result.exit_code == 0, command
            assert "Usage:" in result.stdout, command

    def test_help_shows_command_list(self) -> None:
        result = invoke("-h")
        for name in ("scan", "usage", "models", "tools", "sessions",
                     "export", "doctor", "pricing"):
            assert name in result.stdout


class TestOverviewPaths:
    def test_tui_disabled_env_prints_static_dashboard(
        self, data_dir: Path, monkeypatch
    ) -> None:
        monkeypatch.setenv("TOKNT_TUI", "0")
        result = invoke()  # no subcommand: static fallback in the runner
        assert result.exit_code == 0
        assert "usage events" in result.stdout or "toknt —" in result.stdout

    def test_overview_command_prints_dashboard(self, data_dir: Path) -> None:
        result = invoke("overview")
        assert result.exit_code == 0
        assert "toknt —" in result.stdout
        assert (
            "No usage data has been imported yet" in result.stdout
            or "usage events" in result.stdout
        )

    def test_overview_can_merge_models(
        self, data_dir: Path, seeded_conn: sqlite3.Connection
    ) -> None:
        ingestion.upsert_usage_events(
            seeded_conn,
            [
                usage(
                    id="usage:cline:merge-3",
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
        result = invoke("overview", "--merge-models")
        assert result.exit_code == 0
        assert "glm-5.3-flash-free" not in result.stdout
        assert "glm-5.3-flash" in result.stdout
