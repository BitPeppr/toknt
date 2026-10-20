"""Merging the same model served by different providers.

The analytics layer takes an optional ``canonical`` callable; when it is
given, values that name the same model are grouped before anything is
counted, so totals, events, sessions and cost stay consistent.
"""

from __future__ import annotations

import sqlite3

import pytest

from tests.support.models import at, usage
from toknt.analytics import aggregation, model_names
from toknt.models.usage import TokenScheme
from toknt.storage import ingestion

UTC = None


def seed(conn: sqlite3.Connection, events) -> None:
    ingestion.upsert_usage_events(conn, events)


def _same_model_events() -> list:
    """Three spellings of one model, plus a different model entirely."""
    return [
        usage(
            id="usage:opencode:u1",
            ts_ms=at(2026, 1, 5, 10),
            model="z-ai/glm-5.3-flash",
            agent="opencode",
            session_id="session-a",
            input=300,
            output=0,
            reasoning=0,
            cache_read=0,
            cache_write=0,
        ),
        usage(
            id="usage:cline:u2",
            ts_ms=at(2026, 1, 5, 11),
            model="glm-5.3-flash-free",
            agent="cline",
            session_id="session-b",
            input=100,
            output=0,
            reasoning=0,
            cache_read=0,
            cache_write=0,
        ),
        usage(
            id="usage:opencode:u3",
            ts_ms=at(2026, 1, 6, 9),
            model="z-ai/glm-5.3",
            agent="opencode",
            session_id="session-c",
            input=50,
            output=0,
            reasoning=0,
            cache_read=0,
            cache_write=0,
        ),
    ]


CANONICAL = model_names.canonical_model


class TestBreakdownMerging:
    def test_split_keeps_every_spelling(self, conn: sqlite3.Connection) -> None:
        seed(conn, _same_model_events())
        flt = aggregation.UsageFilter(since_ms=at(2026, 1, 5), until_ms=at(2026, 1, 7))
        rows = aggregation.usage_breakdown(conn, flt)
        assert {row.key for row in rows} == {
            "z-ai/glm-5.3-flash",
            "glm-5.3-flash-free",
            "z-ai/glm-5.3",
        }

    def test_merged_rows_are_summarized_together(self, conn: sqlite3.Connection) -> None:
        seed(conn, _same_model_events())
        flt = aggregation.UsageFilter(since_ms=at(2026, 1, 5), until_ms=at(2026, 1, 7))
        rows = aggregation.usage_breakdown(conn, flt, canonical=CANONICAL)
        by_key = {row.key: row for row in rows}
        assert set(by_key) == {"glm-5.3-flash", "glm-5.3"}
        merged = by_key["glm-5.3-flash"]
        assert merged.summary.total == 400  # 300 + 100 across both spellings
        assert merged.summary.events == 2
        assert merged.summary.sessions == 2  # one per harness

    def test_merged_rows_are_ranked_by_total(self, conn: sqlite3.Connection) -> None:
        seed(conn, _same_model_events())
        flt = aggregation.UsageFilter(since_ms=at(2026, 1, 5), until_ms=at(2026, 1, 7))
        rows = aggregation.usage_breakdown(conn, flt, canonical=CANONICAL)
        assert rows[0].key == "glm-5.3-flash"
        assert rows[0].summary.total == 400

    def test_different_models_stay_apart(self, conn: sqlite3.Connection) -> None:
        seed(conn, _same_model_events())
        flt = aggregation.UsageFilter(since_ms=at(2026, 1, 5), until_ms=at(2026, 1, 7))
        rows = aggregation.usage_breakdown(conn, flt, canonical=CANONICAL)
        by_key = {row.key: row for row in rows}
        assert by_key["glm-5.3"].summary.total == 50

    def test_reported_cost_is_summed(self, conn: sqlite3.Connection) -> None:
        seed(
            conn,
            [
                usage(
                    id="usage:opencode:c1",
                    ts_ms=at(2026, 1, 5, 10),
                    model="z-ai/glm-5.3-flash",
                    cost_reported=1.25,
                ),
                usage(
                    id="usage:cline:c2",
                    ts_ms=at(2026, 1, 5, 11),
                    model="glm-5.3-flash-free",
                    cost_reported=0.75,
                ),
            ],
        )
        flt = aggregation.UsageFilter(since_ms=at(2026, 1, 5), until_ms=at(2026, 1, 7))
        rows = aggregation.usage_breakdown(conn, flt, canonical=CANONICAL)
        assert len(rows) == 1
        assert rows[0].summary.cost_reported == 2.0


class TestSeriesMerging:
    def test_series_share_one_bucket_line(self, conn: sqlite3.Connection) -> None:
        seed(conn, _same_model_events())
        flt = aggregation.UsageFilter(since_ms=at(2026, 1, 5), until_ms=at(2026, 1, 7))
        series = aggregation.usage_series(
            conn, flt, metric="total", bucket="day", tz=UTC, canonical=CANONICAL
        )
        assert tuple(series.series) == ("glm-5.3-flash", "glm-5.3")
        assert series.series["glm-5.3-flash"] == (400, None, None)
        assert series.series["glm-5.3"] == (None, 50, None)

    def test_series_without_canonical_keeps_both_lines(self, conn: sqlite3.Connection) -> None:
        seed(conn, _same_model_events())
        flt = aggregation.UsageFilter(since_ms=at(2026, 1, 5), until_ms=at(2026, 1, 7))
        series = aggregation.usage_series(
            conn, flt, metric="total", bucket="day", tz=UTC
        )
        assert "z-ai/glm-5.3-flash" in series.series
        assert "glm-5.3-flash-free" in series.series


class TestCacheRateMerging:
    def test_rates_merge_across_spellings(self, conn: sqlite3.Connection) -> None:
        seed(
            conn,
            [
                usage(
                    id="usage:opencode:r1",
                    ts_ms=at(2026, 1, 5, 10),
                    model="z-ai/glm-5.3-flash",
                    scheme=TokenScheme.ANTHROPIC,
                    input=100,
                    cache_read=100,
                ),
                usage(
                    id="usage:cline:r2",
                    ts_ms=at(2026, 1, 5, 11),
                    model="glm-5.3-flash-free",
                    scheme=TokenScheme.ANTHROPIC,
                    input=100,
                    cache_read=300,
                ),
            ],
        )
        flt = aggregation.UsageFilter(since_ms=at(2026, 1, 5), until_ms=at(2026, 1, 7))
        rates = aggregation.model_cache_rates(conn, flt, canonical=CANONICAL)
        assert len(rates) == 1
        assert rates[0].model == "glm-5.3-flash"
        assert rates[0].cache_read == 400
        assert rates[0].prompt == 600
        assert rates[0].rate == pytest.approx(400 / 600)

    def test_unmerged_rates_stay_per_spelling(self, conn: sqlite3.Connection) -> None:
        seed(conn, _same_model_events())
        flt = aggregation.UsageFilter(since_ms=at(2026, 1, 5), until_ms=at(2026, 1, 7))
        rates = aggregation.model_cache_rates(conn, flt)
        assert len(rates) == 3


class TestSessionModelNames:
    def test_session_models_are_deduplicated(self, conn: sqlite3.Connection) -> None:
        seed(
            conn,
            [
                usage(
                    id="usage:opencode:s1",
                    ts_ms=at(2026, 1, 5, 10),
                    model="z-ai/glm-5.3-flash",
                    session_id="session-a",
                ),
                usage(
                    id="usage:opencode:s2",
                    ts_ms=at(2026, 1, 5, 11),
                    model="glm-5.3-flash-free",
                    session_id="session-a",
                ),
            ],
        )
        flt = aggregation.UsageFilter(since_ms=at(2026, 1, 5), until_ms=at(2026, 1, 7))
        sessions = aggregation.session_ranking(conn, flt, canonical=CANONICAL)
        assert sessions[0].models == ("glm-5.3-flash",)

    def test_session_models_keep_raw_names_without_canonical(
        self, conn: sqlite3.Connection
    ) -> None:
        seed(conn, _same_model_events())
        flt = aggregation.UsageFilter(since_ms=at(2026, 1, 5), until_ms=at(2026, 1, 7))
        sessions = aggregation.session_ranking(conn, flt)
        assert sessions[0].models == ("z-ai/glm-5.3-flash",)
