"""Aggregation tests: time parsing, bucketing, series, breakdowns, sessions.

All data is synthetic canonical events inserted through the storage layer, so
the tests exercise the same path a real scan would.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from tests.support.models import at, tool, usage
from toknt.analytics import aggregation
from toknt.models.events import SessionRecord
from toknt.models.usage import TokenScheme
from toknt.storage import ingestion

UTC = ZoneInfo("UTC")
NY = ZoneInfo("America/New_York")
BERLIN = ZoneInfo("Europe/Berlin")


def seed(conn: sqlite3.Connection, events) -> None:
    ingestion.upsert_usage_events(conn, events)


class TestParseSince:
    def test_hours_days_weeks(self) -> None:
        now = at(2026, 10, 19, 12, 0)
        assert aggregation.parse_since("24h", now_ms=now, tz=UTC) == now - 24 * 3_600_000
        assert aggregation.parse_since("30d", now_ms=now, tz=UTC) == now - 30 * 86_400_000
        assert aggregation.parse_since("2w", now_ms=now, tz=UTC) == now - 2 * 604_800_000

    def test_iso_date_is_local_midnight(self) -> None:
        result = aggregation.parse_since("2026-10-01", tz=NY)
        expected = int(datetime(2026, 10, 1, tzinfo=NY).timestamp() * 1000)
        assert result == expected

    def test_today_resolves_to_local_midnight(self) -> None:
        result = aggregation.parse_since("today", tz=BERLIN)
        local = datetime.fromtimestamp(result / 1000, tz=BERLIN)
        assert (local.hour, local.minute, local.second) == (0, 0, 0)

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid --since"):
            aggregation.parse_since("fortnight", tz=UTC)


class TestBuckets:
    def test_hour_bucket(self) -> None:
        moment = at(2026, 1, 5, 10, 47)
        assert aggregation.bucket_start_ms(moment, "hour", UTC) == at(2026, 1, 5, 10, 0)

    def test_day_bucket_respects_local_timezone(self) -> None:
        moment = at(2026, 1, 6, 3, 30)  # 2026-01-05 22:30 in New York
        expected = int(datetime(2026, 1, 5, tzinfo=NY).timestamp() * 1000)
        assert aggregation.bucket_start_ms(moment, "day", NY) == expected

    def test_week_starts_monday(self) -> None:
        wednesday = at(2026, 1, 7, 15, 0)
        assert aggregation.bucket_start_ms(wednesday, "week", UTC) == at(2026, 1, 5, 0, 0)

    def test_unknown_bucket_raises(self) -> None:
        with pytest.raises(ValueError):
            aggregation.bucket_start_ms(at(2026, 1, 5), "fortnight", UTC)

    def test_day_buckets_are_dst_safe(self) -> None:
        # Europe/Berlin springs forward on 2026-03-29 at 02:00.
        start = int(datetime(2026, 3, 28, 12, tzinfo=BERLIN).timestamp() * 1000)
        end = int(datetime(2026, 3, 30, 12, tzinfo=BERLIN).timestamp() * 1000)
        buckets = aggregation.bucket_starts(start, end, "day", BERLIN)
        days = [
            datetime.fromtimestamp(bucket / 1000, tz=BERLIN).strftime("%Y-%m-%d")
            for bucket in buckets
        ]
        assert days == ["2026-03-28", "2026-03-29", "2026-03-30"]
        for bucket in buckets:
            local = datetime.fromtimestamp(bucket / 1000, tz=BERLIN)
            assert (local.hour, local.minute) == (0, 0)


def _three_day_events() -> list:
    return [
        usage(
            id="usage:opencode:u1",
            ts_ms=at(2026, 1, 5, 10),
            model="big",
            input=300,
            output=0,
            reasoning=0,
            cache_read=0,
            cache_write=0,
        ),
        usage(
            id="usage:opencode:u2",
            ts_ms=at(2026, 1, 5, 11),
            model="small",
            input=10,
            output=0,
            reasoning=0,
            cache_read=0,
            cache_write=0,
        ),
        usage(
            id="usage:opencode:u3",
            ts_ms=at(2026, 1, 7, 9),
            model="small",
            input=20,
            output=0,
            reasoning=0,
            cache_read=0,
            cache_write=0,
        ),
    ]


class TestSeries:
    def test_gaps_are_none_and_keys_ordered_by_total(self, conn: sqlite3.Connection) -> None:
        seed(conn, _three_day_events())
        flt = aggregation.UsageFilter(since_ms=at(2026, 1, 5), until_ms=at(2026, 1, 8))
        series = aggregation.usage_series(conn, flt, metric="total", bucket="day", tz=UTC)
        assert series.buckets == (at(2026, 1, 5), at(2026, 1, 6), at(2026, 1, 7))
        assert tuple(series.series) == ("big", "small")
        assert series.series["big"] == (300, None, None)
        assert series.series["small"] == (10, None, 20)

    def test_top_n_groups_the_rest_into_other(self, conn: sqlite3.Connection) -> None:
        events = [
            usage(id="usage:opencode:a", ts_ms=at(2026, 1, 5), model="a", input=100,
                  output=0, reasoning=0, cache_read=0, cache_write=0),
            usage(id="usage:opencode:b", ts_ms=at(2026, 1, 5), model="b", input=50,
                  output=0, reasoning=0, cache_read=0, cache_write=0),
            usage(id="usage:opencode:c", ts_ms=at(2026, 1, 5), model="c", input=10,
                  output=0, reasoning=0, cache_read=0, cache_write=0),
        ]
        seed(conn, events)
        series = aggregation.usage_series(conn, aggregation.UsageFilter(), top_n=2, tz=UTC)
        assert tuple(series.series) == ("a", "b", "(other)")
        assert series.series["(other)"] == (10,)

    def test_cumulative_running_total(self, conn: sqlite3.Connection) -> None:
        seed(conn, _three_day_events())
        flt = aggregation.UsageFilter(since_ms=at(2026, 1, 5), until_ms=at(2026, 1, 8))
        series = aggregation.usage_series(conn, flt, metric="total", bucket="day", tz=UTC)
        running = aggregation.cumulative(series)
        assert running.series["big"] == (300, 300, 300)
        assert running.series["small"] == (10, 10, 30)

    def test_requests_metric_counts_events(self, conn: sqlite3.Connection) -> None:
        seed(conn, _three_day_events())
        flt = aggregation.UsageFilter(since_ms=at(2026, 1, 5), until_ms=at(2026, 1, 8))
        series = aggregation.usage_series(conn, flt, metric="requests", bucket="day", tz=UTC)
        assert series.series["big"] == (1, None, None)
        assert series.series["small"] == (1, None, 1)

    def test_activity_series_counts_requests_and_sessions(
        self, conn: sqlite3.Connection
    ) -> None:
        seed(conn, _three_day_events())
        flt = aggregation.UsageFilter(since_ms=at(2026, 1, 5), until_ms=at(2026, 1, 8))
        activity = aggregation.activity_series(conn, flt, bucket="day", tz=UTC)
        assert activity.series["requests"] == (2, 0, 1)
        assert activity.series["sessions"] == (1, 0, 1)


class TestSeriesWindowStart:
    """The ``--since all`` sentinel must not anchor buckets at epoch 0."""

    def test_since_all_clamps_buckets_to_first_event(
        self, conn: sqlite3.Connection
    ) -> None:
        seed(conn, _three_day_events())
        series = aggregation.usage_series(
            conn, aggregation.UsageFilter(since_ms=0), metric="total", bucket="day", tz=UTC
        )
        assert series.buckets == (at(2026, 1, 5), at(2026, 1, 6), at(2026, 1, 7))

    def test_activity_since_all_clamps_to_first_event(
        self, conn: sqlite3.Connection
    ) -> None:
        seed(conn, _three_day_events())
        activity = aggregation.activity_series(
            conn, aggregation.UsageFilter(since_ms=0), bucket="day", tz=UTC
        )
        assert activity.buckets[0] == at(2026, 1, 5)

    def test_explicit_since_keeps_leading_empty_buckets(
        self, conn: sqlite3.Connection
    ) -> None:
        seed(conn, _three_day_events())
        flt = aggregation.UsageFilter(since_ms=at(2026, 1, 1), until_ms=at(2026, 1, 8))
        series = aggregation.usage_series(conn, flt, metric="total", bucket="day", tz=UTC)
        assert series.buckets[0] == at(2026, 1, 1)
        assert len(series.buckets) == 7
        assert series.series["small"][:2] == (None, None)


class TestBreakdownAndSummary:
    def test_summary_flags_events_without_total(self, conn: sqlite3.Connection) -> None:
        events = [
            usage(id="usage:opencode:full", ts_ms=at(2026, 1, 5), model="m",
                  input=100, output=50, reasoning=10, cache_read=5, cache_write=0),
            usage(id="usage:opencode:partial", ts_ms=at(2026, 1, 5), model="m",
                  input=5, output=2, reasoning=None, cache_read=0, cache_write=0),
        ]
        seed(conn, events)
        summary = aggregation.overall_summary(conn, aggregation.UsageFilter()).summary
        assert summary.events == 2
        assert summary.events_without_total == 1
        assert summary.total is None  # one un-totalable event poisons the sum
        assert summary.input == 105  # components still sum when reported
        assert summary.coverage == 0.5

    def test_breakdown_orders_by_total_and_keeps_partial_last(
        self, conn: sqlite3.Connection
    ) -> None:
        events = [
            usage(id="usage:opencode:big-a", ts_ms=at(2026, 1, 5), model="big-a",
                  input=1000, output=0, reasoning=0, cache_read=0, cache_write=0),
            usage(id="usage:opencode:small", ts_ms=at(2026, 1, 5), model="small",
                  input=10, output=0, reasoning=0, cache_read=0, cache_write=0),
            usage(id="usage:opencode:partial", ts_ms=at(2026, 1, 5), model="partial",
                  input=5, output=2, reasoning=None, cache_read=0, cache_write=0),
        ]
        seed(conn, events)
        rows = aggregation.usage_breakdown(conn, aggregation.UsageFilter())
        assert [row.key for row in rows] == ["big-a", "small", "partial"]
        assert rows[0].summary.total == 1000
        assert rows[-1].summary.total is None
        assert rows[-1].summary.events_without_total == 1

    def test_breakdown_reports_mixed_schemes(self, conn: sqlite3.Connection) -> None:
        events = [
            usage(id="usage:opencode:x", ts_ms=at(2026, 1, 5), model="m",
                  scheme=TokenScheme.ANTHROPIC, input=1, output=1, reasoning=1,
                  cache_read=1, cache_write=0),
            usage(id="usage:codex:y", ts_ms=at(2026, 1, 5), model="m",
                  agent="codex", source_id="codex:sessions", scheme=TokenScheme.OPENAI,
                  input=10, output=5, reasoning=3, cache_read=0, cache_write=None),
        ]
        seed(conn, events)
        rows = aggregation.usage_breakdown(conn, aggregation.UsageFilter())
        assert rows[0].schemes == ("anthropic", "openai")
        assert rows[0].summary.total == 4 + 15

    def test_overall_summary_reports_period_and_agents(self, conn: sqlite3.Connection) -> None:
        seed(conn, _three_day_events())
        overall = aggregation.overall_summary(conn, aggregation.UsageFilter())
        assert overall.agents == ("opencode",)
        assert overall.first_ms == at(2026, 1, 5, 10)
        assert overall.last_ms == at(2026, 1, 7, 9)


class TestFilters:
    def test_agent_filter(self, conn: sqlite3.Connection) -> None:
        seed(conn, _three_day_events())
        flt = aggregation.UsageFilter(agents=("codex",))
        assert aggregation.fetch_usage_rows(conn, flt) == []

    def test_project_prefix_filter(self, conn: sqlite3.Connection) -> None:
        events = [
            usage(id="usage:opencode:p1", ts_ms=at(2026, 1, 5), project="/work/alpha"),
            usage(id="usage:opencode:p2", ts_ms=at(2026, 1, 5), project="/work/beta"),
            usage(id="usage:opencode:p3", ts_ms=at(2026, 1, 5), project=None),
        ]
        seed(conn, events)
        flt = aggregation.UsageFilter(projects=("/work/alpha",))
        rows = aggregation.fetch_usage_rows(conn, flt)
        assert [row.project for row in rows] == ["/work/alpha"]

    def test_ignore_projects_excludes_subpaths_but_keeps_null(
        self, conn: sqlite3.Connection
    ) -> None:
        events = [
            usage(id="usage:opencode:ig1", ts_ms=at(2026, 1, 5), project="/work/alpha"),
            usage(id="usage:opencode:ig2", ts_ms=at(2026, 1, 5), project="/work/alpha/sub"),
            usage(id="usage:opencode:ig3", ts_ms=at(2026, 1, 5), project="/work/beta"),
            usage(id="usage:opencode:ig4", ts_ms=at(2026, 1, 5), project=None),
        ]
        seed(conn, events)
        flt = aggregation.UsageFilter(ignore_projects=("/work/alpha",))
        projects = [row.project for row in aggregation.fetch_usage_rows(conn, flt)]
        assert projects == ["/work/beta", None]

    def test_unknown_model_matches_null_rows(self, conn: sqlite3.Connection) -> None:
        events = [
            usage(id="usage:opencode:n1", ts_ms=at(2026, 1, 5), model=None),
            usage(id="usage:opencode:n2", ts_ms=at(2026, 1, 5), model="m"),
        ]
        seed(conn, events)
        flt = aggregation.UsageFilter(models=("(unknown)",))
        rows = aggregation.fetch_usage_rows(conn, flt)
        assert len(rows) == 1
        assert rows[0].model is None


class TestSessionRanking:
    def _seed_sessions(self, conn: sqlite3.Connection) -> None:
        events = []
        records = []
        for index in range(1, 11):
            session_id = f"session-{index:02d}"
            events.append(
                usage(
                    id=f"usage:opencode:s{index}",
                    ts_ms=at(2026, 1, 5) + index * 1000,
                    session_id=session_id,
                    input=index * 100,
                    output=0,
                    reasoning=0,
                    cache_read=0,
                    cache_write=0,
                )
            )
            records.append(
                SessionRecord(
                    agent="opencode",
                    session_id=session_id,
                    source_id="opencode:db",
                    title=f"Session {index}",
                    time_created_ms=at(2026, 1, 5) + index * 1000,
                    time_updated_ms=at(2026, 1, 5) + index * 60_000,
                )
            )
        events.append(
            usage(
                id="usage:opencode:spike",
                ts_ms=at(2026, 1, 6),
                session_id="session-spike",
                input=50_000,
                output=0,
                reasoning=0,
                cache_read=0,
                cache_write=0,
            )
        )
        records.append(
            SessionRecord(
                agent="opencode",
                session_id="session-spike",
                source_id="opencode:db",
                title="Spike",
                time_created_ms=at(2026, 1, 6),
                time_updated_ms=at(2026, 1, 6, 2),
            )
        )
        seed(conn, events)
        ingestion.upsert_sessions(conn, records)
        ingestion.upsert_tool_events(
            conn,
            [
                tool(id=f"tool:opencode:c{index}", ts_ms=at(2026, 1, 6), session_id="session-spike")
                for index in range(3)
            ],
        )

    def test_ranking_with_outlier_flag(self, conn: sqlite3.Connection) -> None:
        self._seed_sessions(conn)
        ranking = aggregation.session_ranking(conn, aggregation.UsageFilter())
        assert ranking[0].session_id == "session-spike"
        assert ranking[0].outlier is True
        assert ranking[0].tool_calls == 3
        assert ranking[0].title == "Spike"
        assert ranking[0].duration_ms == 2 * 3_600_000
        # The spike dwarfs the regular sessions: none of them is an outlier.
        assert all(not item.outlier for item in ranking[1:])

    def test_ranking_respects_limit_and_order(self, conn: sqlite3.Connection) -> None:
        self._seed_sessions(conn)
        ranking = aggregation.session_ranking(conn, aggregation.UsageFilter(), limit=3)
        assert len(ranking) == 3
        assert [item.session_id for item in ranking[:3]] == [
            "session-spike",
            "session-10",
            "session-09",
        ]
