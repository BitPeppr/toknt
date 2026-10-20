"""Cost estimation tests: strictness, scheme semantics, overrides, coverage."""

from __future__ import annotations

import sqlite3

from tests.support.models import at, usage
from toknt.analytics import aggregation, costs
from toknt.config import AppConfig, PricingOverride
from toknt.models.usage import TokenCounts, TokenScheme
from toknt.storage import ingestion, pricing


def price(
    *,
    input_rate: float | None = 1.0,
    output_rate: float | None = 10.0,
    cache_read_rate: float | None = 0.25,
    cache_write_rate: float | None = 1.25,
) -> costs.Price:
    return costs.Price(
        model="model-a",
        input_per_mtok=input_rate,
        output_per_mtok=output_rate,
        cache_read_per_mtok=cache_read_rate,
        cache_write_per_mtok=cache_write_rate,
    )


class TestEstimateCost:
    def test_all_categories_priced(self) -> None:
        counts = TokenCounts(
            input=1_000_000, output=1_000_000, reasoning=0, cache_read=2_000_000, cache_write=0
        )
        result = costs.estimate_cost(counts, TokenScheme.ANTHROPIC, price())
        assert result == 1.0 + 10.0 + 0.5

    def test_reasoning_billed_at_output_rate_for_anthropic(self) -> None:
        counts = TokenCounts(
            input=0, output=1_000_000, reasoning=1_000_000, cache_read=0, cache_write=0
        )
        result = costs.estimate_cost(counts, TokenScheme.ANTHROPIC, price())
        assert result == 20.0

    def test_openai_reasoning_is_a_subset_of_output(self) -> None:
        counts = TokenCounts(
            input=1_000_000, output=1_000_000, reasoning=500_000, cache_read=0, cache_write=None
        )
        result = costs.estimate_cost(counts, TokenScheme.OPENAI, price())
        assert result == 11.0

    def test_openai_cached_input_is_billed_once(self) -> None:
        counts = TokenCounts(
            input=3_000_000, output=0, reasoning=0, cache_read=2_000_000, cache_write=0
        )
        # 1M non-cached input at the input rate + 2M cached at the cache rate.
        result = costs.estimate_cost(counts, TokenScheme.OPENAI, price())
        assert result == 1.0 + 0.5

    def test_openai_cache_writes_are_ordinary_input(self) -> None:
        counts = TokenCounts(
            input=1_000_000, output=0, reasoning=0, cache_read=0, cache_write=1_000_000
        )
        # Write tokens are already inside input: billed at the input rate
        # exactly once, never at the cache-write rate.
        result = costs.estimate_cost(counts, TokenScheme.OPENAI, price())
        assert result == 1.0

    def test_openai_cached_input_without_cache_rate_is_not_guessed(self) -> None:
        counts = TokenCounts(
            input=3_000_000, output=0, reasoning=0, cache_read=2_000_000, cache_write=0
        )
        partial = price(cache_read_rate=None)
        assert costs.estimate_cost(counts, TokenScheme.OPENAI, partial) is None

    def test_openai_cached_above_input_never_bills_negative_input(self) -> None:
        counts = TokenCounts(
            input=100, output=0, reasoning=0, cache_read=1_000_000, cache_write=0
        )
        result = costs.estimate_cost(counts, TokenScheme.OPENAI, price())
        assert result == 0.25

    def test_missing_rate_for_used_category_is_not_guessed(self) -> None:
        counts = TokenCounts(input=0, output=5, reasoning=0, cache_read=0, cache_write=0)
        partial_price = price(output_rate=None)
        assert costs.estimate_cost(counts, TokenScheme.ANTHROPIC, partial_price) is None

    def test_missing_rate_for_unused_category_is_fine(self) -> None:
        counts = TokenCounts(input=1_000_000, output=0, reasoning=0, cache_read=0, cache_write=0)
        partial_price = price(output_rate=None)
        assert costs.estimate_cost(counts, TokenScheme.ANTHROPIC, partial_price) == 1.0

    def test_unknown_scheme_or_price_is_none(self) -> None:
        counts = TokenCounts(input=5, output=5, reasoning=0, cache_read=0, cache_write=0)
        assert costs.estimate_cost(counts, TokenScheme.UNKNOWN, price()) is None
        assert costs.estimate_cost(counts, TokenScheme.ANTHROPIC, None) is None

    def test_zero_usage_costs_zero(self) -> None:
        counts = TokenCounts(input=0, output=0, reasoning=0, cache_read=0, cache_write=0)
        assert costs.estimate_cost(counts, TokenScheme.ANTHROPIC, price()) == 0.0


class TestPriceLoading:
    def test_config_overrides_merge_field_wise_with_table(self, conn: sqlite3.Connection) -> None:
        pricing.add_price(conn, model="model-a", input_per_mtok=2.0, output_per_mtok=20.0)
        config = AppConfig(pricing={"model-a": PricingOverride(input_per_mtok=5.0)})
        price_map = costs.load_price_map(conn, config)
        assert price_map["model-a"].input_per_mtok == 5.0  # config rate wins
        assert price_map["model-a"].output_per_mtok == 20.0  # table rate kept
        assert price_map["model-a"].source == "config"

    def test_latest_effective_row_wins_without_config(self, conn: sqlite3.Connection) -> None:
        pricing.add_price(conn, model="model-a", input_per_mtok=1.0, output_per_mtok=10.0)
        pricing.add_price(
            conn, model="model-a", effective_from_ms=at(2026, 1, 1), input_per_mtok=9.0
        )
        price_map = costs.load_price_map(conn, AppConfig())
        assert price_map["model-a"].input_per_mtok == 9.0
        # Rows are versioned wholesale: a newer row that omits a rate means
        # "unknown", not "keep the old value".
        assert price_map["model-a"].output_per_mtok is None
        assert price_map["model-a"].source == "table"


class TestCostSummaries:
    def _seed(self, conn: sqlite3.Connection) -> None:
        ingestion.upsert_usage_events(
            conn,
            [
                usage(
                    id="usage:opencode:known-a",
                    ts_ms=at(2026, 1, 5),
                    model="model-a",
                    input=1_000_000,
                    output=0,
                    reasoning=0,
                    cache_read=0,
                    cache_write=0,
                ),
                usage(
                    id="usage:opencode:known-b",
                    ts_ms=at(2026, 1, 5),
                    model="model-a",
                    input=1_000_000,
                    output=0,
                    reasoning=0,
                    cache_read=0,
                    cache_write=0,
                ),
                usage(
                    id="usage:opencode:unpriced",
                    ts_ms=at(2026, 1, 5),
                    model="never-priced",
                    input=1_000_000,
                    output=0,
                    reasoning=0,
                    cache_read=0,
                    cache_write=0,
                ),
            ],
        )

    def test_coverage_counts(self, conn: sqlite3.Connection) -> None:
        self._seed(conn)
        price_map = {"model-a": price()}
        estimate = costs.cost_summary(conn, aggregation.UsageFilter(), price_map)
        assert estimate.estimated == 2.0
        assert estimate.priced_events == 2
        assert estimate.unpriced_events == 1
        assert estimate.coverage == 2 / 3

    def test_by_dimension(self, conn: sqlite3.Connection) -> None:
        self._seed(conn)
        price_map = {"model-a": price()}
        grouped = costs.cost_by_dimension(
            conn, aggregation.UsageFilter(), price_map, dimension="model"
        )
        assert set(grouped) == {"model-a", "never-priced"}
        assert grouped["model-a"].estimated == 2.0
        assert grouped["never-priced"].estimated is None

    def test_no_prices_at_all_yields_no_estimate(self, conn: sqlite3.Connection) -> None:
        self._seed(conn)
        estimate = costs.cost_summary(conn, aggregation.UsageFilter(), {})
        assert estimate.estimated is None
        assert estimate.priced_events == 0
        assert estimate.unpriced_events == 3
