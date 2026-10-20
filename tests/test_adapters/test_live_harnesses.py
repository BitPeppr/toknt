"""Live-data regression tests for harnesses installed on this machine.

These run only when the real store exists; they scan the actual home
directory into a temporary database and compare what the adapters produced
against the source files themselves. They are the guard against silent
format drift: if Cline, pi or Grok changes its on-disk shape, these fail
before any number in the TUI can be quietly wrong.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from toknt.adapters import registered_adapters
from toknt.config import AppConfig
from toknt.models.events import Harness
from toknt.storage import database

HOME = Path.home()


def _live_conn(tmp_path: Path) -> sqlite3.Connection:
    import os

    os.environ["TOKNT_DATA_DIR"] = str(tmp_path)
    try:
        return database.open_database()
    finally:
        del os.environ["TOKNT_DATA_DIR"]


def _scan_all(conn: sqlite3.Connection, harnesses: set[str]) -> None:
    from toknt.service import run_scan

    cfg = AppConfig(
        enabled_harnesses=tuple(
            harness for harness in Harness if harness.value in harnesses
        )
    )
    run_scan(conn, full=True, config=cfg)


def _totals(conn: sqlite3.Connection, harness: str) -> tuple[int, int, int]:
    row = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0)"
        " FROM usage_events WHERE agent = ?",
        (harness,),
    ).fetchone()
    return int(row[0]), int(row[1]), int(row[2])


# ----------------------------------------------------------------------- pi

PI_ROOT = HOME / ".pi" / "agent" / "sessions"
pytestmark_pi = pytest.mark.skipif(not PI_ROOT.is_dir(), reason="no live pi corpus")


@pytest.mark.skipif(not PI_ROOT.is_dir(), reason="no live pi corpus")
class TestLivePi:
    def test_source_totals_match_transcripts(self, tmp_path: Path) -> None:
        conn = _live_conn(tmp_path)
        _scan_all(conn, {"pi"})
        events, input_total, output_total = _totals(conn, "pi")
        assert events > 0
        source_in = source_out = 0
        for path in PI_ROOT.glob("*/*.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("type") != "message":
                    continue
                message = entry.get("message") or {}
                if message.get("role") != "assistant":
                    continue
                if message.get("stopReason") == "error":
                    continue  # failed API calls carry no served usage
                usage = message.get("usage") or {}
                source_in += usage.get("input") or 0
                source_out += usage.get("output") or 0
        assert (input_total, output_total) == (source_in, source_out)
        conn.close()


# -------------------------------------------------------------------- cline

CLINE_ROOT = HOME / ".cline" / "data" / "sessions"


@pytest.mark.skipif(not CLINE_ROOT.is_dir(), reason="no live cline corpus")
class TestLiveCline:
    def test_source_totals_match_message_metrics(self, tmp_path: Path) -> None:
        conn = _live_conn(tmp_path)
        _scan_all(conn, {"cline"})
        events, input_total, output_total = _totals(conn, "cline")
        assert events > 0
        source_in = source_out = 0
        for manifest_path in CLINE_ROOT.glob("*/[0-9]*.json"):
            messages_path = manifest_path.with_name(
                manifest_path.name.replace(".json", ".messages.json")
            )
            if not messages_path.is_file():
                continue
            try:
                messages = json.loads(messages_path.read_text(encoding="utf-8")).get(
                    "messages", []
                )
            except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                continue
            for message in messages:
                if message.get("role") != "assistant":
                    continue
                metrics = message.get("metrics") or {}
                source_in += metrics.get("inputTokens") or 0
                source_out += metrics.get("outputTokens") or 0
        assert (input_total, output_total) == (source_in, source_out)
        conn.close()


# --------------------------------------------------------------------- grok

GROK_ROOT = HOME / ".grok" / "sessions"


@pytest.mark.skipif(not GROK_ROOT.is_dir(), reason="no live grok corpus")
class TestLiveGrok:
    def test_source_totals_match_usage_files(self, tmp_path: Path) -> None:
        conn = _live_conn(tmp_path)
        _scan_all(conn, {"grok"})
        events, input_total, output_total = _totals(conn, "grok")
        assert events > 0
        source_in = source_out = 0
        for usage_path in GROK_ROOT.glob("*/*/usage.json"):
            try:
                payload = json.loads(usage_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                continue
            session_totals = payload.get("session") or {}
            source_in += session_totals.get("inputTokens") or 0
            source_out += session_totals.get("outputTokens") or 0
        assert (input_total, output_total) == (source_in, source_out)
        # every completed tool event in the logs became a tool event row
        completed = 0
        for events_path in GROK_ROOT.glob("*/*/events.jsonl"):
            for line in events_path.read_text(encoding="utf-8").splitlines():
                if '"tool_completed"' in line:
                    completed += 1
        assert conn.execute(
            "SELECT COUNT(*) FROM tool_events WHERE agent = 'grok'"
        ).fetchone()[0] >= completed
        conn.close()


# ---------------------------------------------------------------- opencode

OPENCODE_DB = HOME / ".local" / "share" / "opencode" / "opencode.db"


@pytest.mark.skipif(not OPENCODE_DB.is_file(), reason="no live opencode.db")
class TestLiveOpencode:
    def test_totals_match_opencodes_own_session_aggregates(
        self, tmp_path: Path
    ) -> None:
        """toknt must agree with opencode's own stats screen.

        opencode's stats sums ``session_v2.tokens_*`` (verified against its
        CLI source); the adapter skips fork-copied history rows so summing
        per-response events lands on the same number (within the small
        drift of responses straddling a fork instant).
        """
        import sqlite3 as _sqlite3

        conn = _live_conn(tmp_path)
        _scan_all(conn, {"opencode"})
        _, input_total, output_total = _totals(conn, "opencode")
        assert input_total + output_total > 0
        src = _sqlite3.connect(f"file:{OPENCODE_DB}?mode=ro", uri=True)
        try:
            aggregate = src.execute(
                "SELECT SUM(tokens_input + tokens_output + tokens_reasoning"
                " + tokens_cache_read + tokens_cache_write) FROM session_v2"
            ).fetchone()[0]
        finally:
            src.close()
        toknt_total = conn.execute(
            "SELECT SUM(input_tokens + output_tokens + reasoning_tokens"
            " + cache_read_tokens + cache_write_tokens)"
            " FROM usage_events WHERE agent = 'opencode'"
        ).fetchone()[0]
        # the fork-time cut makes the two accountings agree within a few
        # percent; before the fix toknt reported ~2x opencode's number
        drift = abs(toknt_total - aggregate) / aggregate
        assert drift < 0.03, (
            f"toknt {toknt_total:,} vs opencode's own aggregate {aggregate:,} "
            f"({drift:.1%} drift)"
        )
        conn.close()


# ------------------------------------------------------- all live harnesses

@pytest.mark.skipif(
    not (PI_ROOT.is_dir() and CLINE_ROOT.is_dir() and GROK_ROOT.is_dir()),
    reason="not all live corpora present",
)
class TestLiveHarnessCoverage:
    def test_every_live_harness_appears_in_sessions(self, tmp_path: Path) -> None:
        conn = _live_conn(tmp_path)
        _scan_all(conn, {"pi", "cline", "grok"})
        agents = {
            row[0]
            for row in conn.execute("SELECT DISTINCT agent FROM sessions").fetchall()
        }
        assert {"pi", "cline", "grok"} <= agents
        # the adapters are all registered and each discovered source resolves
        adapters = registered_adapters()
        assert {Harness.PI, Harness.CLINE, Harness.GROK} <= {
            adapter.agent for adapter in adapters
        }
        conn.close()
