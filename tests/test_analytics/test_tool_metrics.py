"""Tool outcome analytics: rates, observability, durations, trend series."""

from __future__ import annotations

import sqlite3
from zoneinfo import ZoneInfo

from tests.support.models import at, tool
from toknt.analytics import aggregation, tool_metrics
from toknt.models.events import Outcome, OutcomeProvenance
from toknt.storage import ingestion

UTC = ZoneInfo("UTC")


def seed(conn: sqlite3.Connection, events) -> None:
    ingestion.upsert_tool_events(conn, events)


class TestToolOutcomes:
    def test_outcome_accounting(self, conn: sqlite3.Connection) -> None:
        seed(
            conn,
            [
                tool(id="tool:opencode:1", ts_ms=at(2026, 1, 5), outcome=Outcome.SUCCESS),
                tool(id="tool:opencode:2", ts_ms=at(2026, 1, 5), outcome=Outcome.SUCCESS),
                tool(id="tool:opencode:3", ts_ms=at(2026, 1, 5), outcome=Outcome.FAILURE),
                tool(id="tool:opencode:4", ts_ms=at(2026, 1, 5), outcome=Outcome.PENDING),
                tool(
                    id="tool:opencode:5",
                    ts_ms=at(2026, 1, 5),
                    outcome=Outcome.UNKNOWN,
                    provenance=OutcomeProvenance.ABSENT,
                    confidence=None,
                ),
            ],
        )
        rows = tool_metrics.tool_outcomes(conn, aggregation.UsageFilter())
        assert len(rows) == 1
        row = rows[0]
        assert row.calls == 5
        assert row.success == 2
        assert row.failure == 1
        assert row.pending == 1
        assert row.unknown == 1
        assert row.known_outcomes == 3
        assert row.success_rate == 2 / 3
        assert row.observability == 3 / 5

    def test_success_rate_is_none_without_known_outcomes(
        self, conn: sqlite3.Connection
    ) -> None:
        seed(
            conn,
            [
                tool(id="tool:opencode:1", ts_ms=at(2026, 1, 5), outcome=Outcome.PENDING),
                tool(
                    id="tool:opencode:2",
                    ts_ms=at(2026, 1, 5),
                    outcome=Outcome.UNKNOWN,
                    provenance=OutcomeProvenance.ABSENT,
                    confidence=None,
                ),
            ],
        )
        row = tool_metrics.tool_outcomes(conn, aggregation.UsageFilter())[0]
        assert row.success_rate is None
        assert row.observability == 0.0

    def test_grouping_and_ordering(self, conn: sqlite3.Connection) -> None:
        seed(
            conn,
            [
                tool(id="tool:opencode:1", ts_ms=at(2026, 1, 5), tool_name="bash"),
                tool(id="tool:opencode:2", ts_ms=at(2026, 1, 5), tool_name="bash"),
                tool(id="tool:opencode:3", ts_ms=at(2026, 1, 5), tool_name="edit"),
            ],
        )
        rows = tool_metrics.tool_outcomes(conn, aggregation.UsageFilter())
        assert [row.tool for row in rows] == ["bash", "edit"]
        assert rows[0].calls == 2


class TestToolDurations:
    def test_percentiles_ignore_unmeasured_calls(self, conn: sqlite3.Connection) -> None:
        events = [
            tool(id=f"tool:opencode:{index}", ts_ms=at(2026, 1, 5), duration_ms=index * 10)
            for index in range(1, 11)
        ]
        events.append(tool(id="tool:opencode:no-duration", ts_ms=at(2026, 1, 5), duration_ms=None))
        seed(conn, events)
        stats = tool_metrics.tool_durations(conn, aggregation.UsageFilter())
        assert "bash" in stats
        bash = stats["bash"]
        assert bash.count == 10
        assert bash.max_ms == 100
        assert bash.p50_ms == 60
        assert bash.p90_ms == 100
        assert bash.mean_ms == 55.0


class TestToolSuccessSeries:
    def test_rate_per_bucket_with_gaps(self, conn: sqlite3.Connection) -> None:
        seed(
            conn,
            [
                tool(id="tool:opencode:1", ts_ms=at(2026, 1, 5, 10), outcome=Outcome.SUCCESS),
                tool(id="tool:opencode:2", ts_ms=at(2026, 1, 5, 11), outcome=Outcome.FAILURE),
                # Jan 6 has only a pending call: not observable -> None
                tool(id="tool:opencode:3", ts_ms=at(2026, 1, 6, 9), outcome=Outcome.PENDING),
                tool(id="tool:opencode:4", ts_ms=at(2026, 1, 7, 9), outcome=Outcome.SUCCESS),
            ],
        )
        flt = aggregation.UsageFilter(since_ms=at(2026, 1, 5), until_ms=at(2026, 1, 8))
        series = tool_metrics.tool_success_series(conn, flt, bucket="day", tz=None)
        values = series.series["bash"]
        assert values[0] == 50
        assert values[1] is None
        assert values[2] == 100

    def test_since_all_clamps_buckets_to_first_tool(self, conn: sqlite3.Connection) -> None:
        """Epoch-0 (unbounded) filters bucket from the earliest event, not 1970."""
        seed(
            conn,
            [tool(id="tool:opencode:1", ts_ms=at(2026, 1, 5, 10), outcome=Outcome.SUCCESS)],
        )
        series = tool_metrics.tool_success_series(
            conn, aggregation.UsageFilter(since_ms=0), bucket="day", tz=UTC
        )
        assert series.buckets == (at(2026, 1, 5),)

    def test_empty_when_no_tools(self, conn: sqlite3.Connection) -> None:
        series = tool_metrics.tool_success_series(conn, aggregation.UsageFilter())
        assert series.is_empty
