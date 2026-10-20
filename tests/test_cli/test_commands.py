"""CLI command tests for analytics, export and pricing (seeded fixtures)."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from pathlib import Path

from typer.testing import CliRunner

from tests.support.models import at, usage
from toknt.cli import app
from toknt.storage import ingestion

runner = CliRunner()

# Fixed period that contains the canonical fixture events (Oct 2026) while
# staying independent of the machine clock.
PERIOD = ("--since", "2026-01-01", "--until", "2027-01-01")

# Wide pseudo-terminal so Rich never folds model/session names mid-word.
WIDE = {"COLUMNS": "220", "LINES": "60"}


def invoke(*args: str):
    return runner.invoke(app, list(args), env=WIDE)


def combined_output(result) -> str:
    """stdout + stderr: Rich error panels go to stderr."""
    return (result.stdout or "") + (result.stderr or "")


class TestUsageCommand:
    def test_table_and_footer(self, data_dir: Path, seeded_conn: sqlite3.Connection) -> None:
        result = invoke("usage", *PERIOD, "--chart", "none")
        assert result.exit_code == 0
        assert "glm-5.3-flash-free" in result.stdout
        assert "gpt-5-codex" in result.stdout
        assert "reported cost" in result.stdout
        assert "estimated" in result.stdout
        # No prices configured: nothing is estimable (never guessed).
        assert "0 of 3 events priced" in result.stdout

    def test_line_chart_renders(self, data_dir: Path, seeded_conn: sqlite3.Connection) -> None:
        result = invoke("usage", *PERIOD)
        assert result.exit_code == 0
        assert "┤" in result.stdout  # the chart axis glyph
        assert "by model" in result.stdout

    def test_filter_by_agent(self, data_dir: Path, seeded_conn: sqlite3.Connection) -> None:
        result = invoke("usage", *PERIOD, "--chart", "none", "--agent", "codex")
        assert result.exit_code == 0
        assert "gpt-5-codex" in result.stdout
        assert "glm-5.3-flash-free" not in result.stdout

    def test_filter_by_harness_flag_and_by_dimension(
        self, data_dir: Path, seeded_conn: sqlite3.Connection
    ) -> None:
        """--harness is the renamed --agent; --by harness labels the table."""
        harness_result = invoke(
            "usage", *PERIOD, "--chart", "none", "--harness", "codex"
        )
        assert harness_result.exit_code == 0
        assert "gpt-5-codex" in harness_result.stdout
        by_harness = invoke("usage", *PERIOD, "--chart", "none", "--by", "harness")
        assert by_harness.exit_code == 0
        assert "Usage by harness" in by_harness.stdout
        # the historical --by agent spelling keeps working
        by_agent = invoke("usage", *PERIOD, "--chart", "none", "--by", "agent")
        assert by_agent.exit_code == 0
        assert "Usage by harness" in by_agent.stdout

    def test_requests_metric_and_unknown_model_last(
        self, data_dir: Path, seeded_conn: sqlite3.Connection
    ) -> None:
        result = invoke("usage", *PERIOD, "--chart", "none", "--metric", "requests")
        assert result.exit_code == 0
        assert "requests" in result.stdout

    def test_invalid_options_fail(self, data_dir: Path, seeded_conn: sqlite3.Connection) -> None:
        assert invoke("usage", "--by", "nonsense").exit_code == 1
        assert invoke("usage", "--agent", "not_a_harness").exit_code == 1
        assert invoke("usage", "--bucket", "fortnight").exit_code == 1
        assert invoke("usage", "--metric", "nonsense").exit_code == 1

    def test_empty_period_message(self, data_dir: Path, seeded_conn: sqlite3.Connection) -> None:
        result = invoke("usage", "--since", "2000-01-01", "--until", "2000-02-01")
        assert result.exit_code == 0
        assert "No usage in the selected period" in result.stdout

    def test_since_all_covers_full_history(
        self, data_dir: Path, seeded_conn: sqlite3.Connection
    ) -> None:
        ingestion.upsert_usage_events(
            seeded_conn,
            [
                usage(
                    id="usage:opencode:ancient",
                    ts_ms=at(2000, 1, 1),
                    model="ancient-model",
                    input=10**12,  # ranks first, so table truncation cannot hide it
                    output=0,
                    reasoning=0,
                    cache_read=0,
                    cache_write=0,
                )
            ],
        )
        recent = invoke("usage", "--since", "30d", "--chart", "none")
        assert "ancient-model" not in recent.stdout
        full = invoke("usage", "--since", "all", "--chart", "none")
        assert full.exit_code == 0
        assert "ancient-model" in full.stdout


class TestModelsCommand:
    def test_inventory_table(self, data_dir: Path, seeded_conn: sqlite3.Connection) -> None:
        result = invoke("models", *PERIOD)
        assert result.exit_code == 0
        assert "gpt-5-codex" in result.stdout
        assert "openai" in result.stdout
        assert "glm-5.3-flash-free" in result.stdout

    def test_no_chart_flag(self, data_dir: Path, seeded_conn: sqlite3.Connection) -> None:
        result = invoke("models", *PERIOD, "--no-chart")
        assert result.exit_code == 0
        assert "Total tokens by model" not in result.stdout

    def test_merge_models_collapses_spellings(
        self, data_dir: Path, seeded_conn: sqlite3.Connection
    ) -> None:
        ingestion.upsert_usage_events(
            seeded_conn,
            [
                usage(
                    id="usage:cline:merge-1",
                    ts_ms=at(2026, 10, 2, 9),
                    agent="cline",
                    model="z-ai/glm-5.3-flash",
                    provider="nvidia",
                    input=1010,
                    output=0,
                    reasoning=0,
                    cache_read=0,
                    cache_write=0,
                )
            ],
        )
        result = invoke("models", *PERIOD, "--merge-models")
        assert result.exit_code == 0
        assert "glm-5.3-flash-free" not in result.stdout
        assert "z-ai/glm-5.3-flash" not in result.stdout
        assert "glm-5.3-flash" in result.stdout
        # the merged row names every provider that served the model
        assert "opencode" in result.stdout
        assert "nvidia" in result.stdout

    def test_no_tps_column_by_default(
        self, data_dir: Path, seeded_conn: sqlite3.Connection
    ) -> None:
        result = invoke("models", *PERIOD, "--no-chart")
        assert result.exit_code == 0
        assert "tok/s" not in result.stdout

    def test_tps_column_shows_estimated_rates(
        self, data_dir: Path, seeded_conn: sqlite3.Connection
    ) -> None:
        base = at(2026, 10, 8, 12, 0)
        ingestion.upsert_usage_events(
            seeded_conn,
            [
                usage(
                    id="usage:opencode:tps-1",
                    ts_ms=base,
                    session_id="ses_tps_pair",
                    model="rate-model",
                    input=0,
                    output=0,
                    reasoning=0,
                    cache_read=0,
                    cache_write=0,
                ),
                usage(
                    id="usage:opencode:tps-2",
                    ts_ms=base + 10_000,
                    session_id="ses_tps_pair",
                    model="rate-model",
                    output=120,
                    reasoning=0,
                    cache_read=0,
                    cache_write=0,
                ),
            ],
        )
        result = invoke("models", *PERIOD, "--no-chart", "--tps")
        assert result.exit_code == 0
        assert "tok/s" in result.stdout
        assert "12.0" in result.stdout  # 120 generated tokens / 10 s

    def test_tps_with_merge_models_joins_the_rate(
        self, data_dir: Path, seeded_conn: sqlite3.Connection
    ) -> None:
        # a merged model's label ("muse-spark-1.3") is not its identity
        # ("muse-spark-1-3"): the rate must be found through the identity
        base = at(2026, 10, 8, 12, 0)
        ingestion.upsert_usage_events(
            seeded_conn,
            [
                usage(
                    id="usage:opencode:tps-merge-1",
                    ts_ms=base,
                    session_id="ses_tps_merge",
                    model="muse-spark-1.3-contributor-free",
                    input=0,
                    output=0,
                    reasoning=0,
                    cache_read=0,
                    cache_write=0,
                ),
                usage(
                    id="usage:opencode:tps-merge-2",
                    ts_ms=base + 10_000,
                    session_id="ses_tps_merge",
                    model="muse-spark-1.3-contributor-free",
                    input=10,
                    output=120,
                    reasoning=0,
                    cache_read=0,
                    cache_write=0,
                ),
            ],
        )
        result = invoke("models", *PERIOD, "--no-chart", "--tps", "--merge-models")
        assert result.exit_code == 0
        assert "12.0" in result.stdout


class TestUsageMergeModels:
    def test_usage_merges_model_rows(
        self, data_dir: Path, seeded_conn: sqlite3.Connection
    ) -> None:
        ingestion.upsert_usage_events(
            seeded_conn,
            [
                usage(
                    id="usage:cline:merge-2",
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
        result = invoke("usage", *PERIOD, "--chart", "none", "--merge-models")
        assert result.exit_code == 0
        assert "glm-5.3-flash-free" not in result.stdout
        assert "glm-5.3-flash" in result.stdout


class TestToolsCommand:
    def test_outcome_table(self, data_dir: Path, seeded_conn: sqlite3.Connection) -> None:
        result = invoke("tools", *PERIOD, "--chart", "none")
        assert result.exit_code == 0
        assert "bash" in result.stdout
        assert "edit" in result.stdout
        # bash: 1 success, no failures -> 100.0%
        assert "100.0%" in result.stdout
        # exec: only a pending call -> no rate, dash rendered
        assert "—" in result.stdout

    def test_rate_chart_and_trend(self, data_dir: Path, seeded_conn: sqlite3.Connection) -> None:
        bar = invoke("tools", *PERIOD, "--chart", "rate")
        assert bar.exit_code == 0
        assert "success rate" in bar.stdout.lower()
        trend = invoke("tools", *PERIOD, "--chart", "trend", "--bucket", "day")
        assert trend.exit_code == 0

    def test_tool_filter(self, data_dir: Path, seeded_conn: sqlite3.Connection) -> None:
        result = invoke("tools", *PERIOD, "--chart", "none", "--tool", "bash")
        assert result.exit_code == 0
        assert "bash" in result.stdout
        assert "edit" not in result.stdout


class TestSessionsCommand:
    def test_sessions_table(self, data_dir: Path, seeded_conn: sqlite3.Connection) -> None:
        result = invoke("sessions", *PERIOD)
        assert result.exit_code == 0
        assert "ses_san_opencode_1" in result.stdout
        assert "Sessions" in result.stdout


class TestExportCommand:
    def test_usage_csv(self, data_dir: Path, seeded_conn: sqlite3.Connection) -> None:
        result = invoke("export", "--what", "usage", "--format", "csv", *PERIOD)
        assert result.exit_code == 0
        rows = list(csv.DictReader(io.StringIO(result.stdout)))
        assert len(rows) == 3
        by_id = {row["id"]: row for row in rows}
        assert by_id["usage:opencode:msg_san_001"]["effective_total_tokens"] == "18990"
        assert by_id["usage:codex:resp_san_002"]["effective_total_tokens"] == "32683"
        # Unknown-scheme event: no effective total, exported as empty, not zero.
        assert by_id["usage:claude_code:req_san_003"]["effective_total_tokens"] == ""

    def test_tools_json(self, data_dir: Path, seeded_conn: sqlite3.Connection) -> None:
        result = invoke("export", "--what", "tools", "--format", "json", *PERIOD)
        assert result.exit_code == 0
        rows = json.loads(result.stdout)
        assert len(rows) == 3
        assert {row["tool_name"] for row in rows} == {"bash", "edit", "exec"}
        assert rows[0]["outcome_provenance"] is not None

    def test_sessions_csv_has_headers_when_empty(
        self, data_dir: Path, seeded_conn: sqlite3.Connection
    ) -> None:
        result = invoke(
            "export", "--what", "sessions", "--format", "csv", "--since", "2000-01-01",
            "--until", "2000-02-01",
        )
        assert result.exit_code == 0
        header = result.stdout.splitlines()[0]
        assert "session_id" in header

    def test_invalid_options(self, data_dir: Path) -> None:
        assert invoke("export", "--what", "nonsense").exit_code == 1
        assert invoke("export", "--format", "xml").exit_code == 1


class TestPricingCommands:
    def test_empty_list_hint(self, data_dir: Path) -> None:
        result = invoke("pricing", "list")
        assert result.exit_code == 0
        assert "No prices configured" in result.stdout

    def test_add_list_remove(self, data_dir: Path) -> None:
        added = invoke(
            "pricing", "add", "gpt-5-codex", "--input", "1.25", "--output", "10",
            "--cache-read", "0.125",
        )
        assert added.exit_code == 0
        assert "gpt-5-codex" in added.stdout
        listed = invoke("pricing", "list")
        assert "gpt-5-codex" in listed.stdout
        assert "1.25" in listed.stdout
        removed = invoke("pricing", "remove", "gpt-5-codex")
        assert removed.exit_code == 0
        assert "1" in removed.stdout
        assert "No prices configured" in invoke("pricing", "list").stdout

    def test_add_requires_a_rate(self, data_dir: Path) -> None:
        result = invoke("pricing", "add", "some-model")
        assert result.exit_code == 1
        assert "at least one" in combined_output(result)

    def test_prices_upgrade_usage_footer(
        self, data_dir: Path, seeded_conn: sqlite3.Connection
    ) -> None:
        before = invoke("usage", *PERIOD, "--chart", "none")
        assert "0 of 3 events priced" in before.stdout
        # Input/output alone cannot price the codex event: it also uses
        # cache-read tokens, so the estimate stays unknown until that rate
        # exists. (Strictness: a partially-priced event is never estimated.)
        invoke("pricing", "add", "gpt-5-codex", "--input", "1.25", "--output", "10")
        strict = invoke("usage", *PERIOD, "--chart", "none")
        assert "0 of 3 events priced" in strict.stdout
        invoke(
            "pricing", "add", "gpt-5-codex",
            "--input", "1.25", "--output", "10", "--cache-read", "0.125",
        )
        after = invoke("usage", *PERIOD, "--chart", "none")
        assert after.exit_code == 0
        assert "1 of 3 events priced" in after.stdout
        assert "estimated $" in after.stdout
