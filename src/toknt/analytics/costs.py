"""Cost estimation from versioned pricing data.

Estimated costs are computed at query time from the ``pricing`` table plus
optional per-model overrides in the configuration file. Nothing derived from
prices is ever written to event rows, so correcting a price never rewrites
history, and source-reported costs (``cost_reported``) are always kept
distinct from estimates.

An estimate is only produced when **every used token category has a price**:
known usage with unknown prices yields ``None`` — never a guess. Callers are
expected to surface coverage (how many events were estimable) next to any
estimate total.

Scheme semantics are honoured when billing:

* Anthropic-style: ``input``, ``output``, ``reasoning``, ``cache_read`` and
  ``cache_write`` are all additive; reasoning is priced at the output rate.
* OpenAI-style: cached input is a **subset** of ``input`` (billed at the
  cache-read rate), cache writes are ordinary input billed at the input
  rate, and reasoning is already inside ``output``. Billing the cached
  portion again would double count it.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from toknt.analytics.aggregation import (
    UsageFilter,
    _dimension_key,
    _display_key,
    fetch_usage_rows,
)
from toknt.analytics.model_names import CanonicalName
from toknt.config import AppConfig
from toknt.models.usage import TokenCounts, TokenScheme


@dataclass(frozen=True, slots=True)
class Price:
    """Per-million-token prices for one model."""

    model: str
    input_per_mtok: float | None = None
    output_per_mtok: float | None = None
    cache_read_per_mtok: float | None = None
    cache_write_per_mtok: float | None = None
    currency: str = "USD"
    source: str = "table"  # "table" | "config"


def load_price_map(
    conn: sqlite3.Connection,
    config: AppConfig | None = None,
) -> dict[str, Price]:
    """Prices keyed by model id; the most recent table row per model wins.

    Configuration overrides replace individual rates of a model's entry;
    rates the configuration does not mention fall back to the table (if the
    model has a row there).
    """
    prices: dict[str, Price] = {}
    try:
        rows = conn.execute(
            """
            SELECT model, currency, input_per_mtok, output_per_mtok,
                   cache_read_per_mtok, cache_write_per_mtok, effective_from_ms
            FROM pricing ORDER BY effective_from_ms
            """
        ).fetchall()
    except sqlite3.OperationalError:  # pre-migration database
        rows = []
    for row in rows:
        prices[row["model"]] = Price(
            model=row["model"],
            input_per_mtok=row["input_per_mtok"],
            output_per_mtok=row["output_per_mtok"],
            cache_read_per_mtok=row["cache_read_per_mtok"],
            cache_write_per_mtok=row["cache_write_per_mtok"],
            currency=row["currency"] or "USD",
            source="table",
        )
    if config is not None:
        for model, override in config.pricing.items():
            base = prices.get(model)
            prices[model] = Price(
                model=model,
                input_per_mtok=_pick(override.input_per_mtok, base and base.input_per_mtok),
                output_per_mtok=_pick(override.output_per_mtok, base and base.output_per_mtok),
                cache_read_per_mtok=_pick(
                    override.cache_read_per_mtok, base and base.cache_read_per_mtok
                ),
                cache_write_per_mtok=_pick(
                    override.cache_write_per_mtok, base and base.cache_write_per_mtok
                ),
                currency=base.currency if base else "USD",
                source="config",
            )
    return prices


def _pick(override: float | None, base: float | None) -> float | None:
    return override if override is not None else base


def estimate_cost(
    counts: TokenCounts,
    scheme: TokenScheme,
    price: Price | None,
) -> float | None:
    """Estimated cost in ``price.currency``, or ``None`` when not estimable."""
    if price is None or scheme is TokenScheme.UNKNOWN:
        return None

    input_tokens = counts.input
    cache_read = counts.cache_read
    cache_write = counts.cache_write
    output_tokens = counts.output
    if scheme is TokenScheme.ANTHROPIC:
        if counts.reasoning is not None and counts.reasoning > 0:
            if output_tokens is None:
                return None
            output_tokens += counts.reasoning
    elif scheme is TokenScheme.OPENAI:
        # Cached input is a subset of input and carries its own (discounted)
        # rate; cache writes are ordinary input billed at the input rate;
        # reasoning is already inside output.
        if input_tokens is not None:
            input_tokens = max(input_tokens - (cache_read or 0), 0)
        cache_write = None

    total = 0.0
    for value, rate in (
        (input_tokens, price.input_per_mtok),
        (output_tokens, price.output_per_mtok),
        (cache_read, price.cache_read_per_mtok),
        (cache_write, price.cache_write_per_mtok),
    ):
        if value is None or value == 0:
            continue
        if rate is None:
            return None
        total += value / 1_000_000 * rate
    return total


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """Sum of estimable costs plus coverage information."""

    estimated: float | None
    priced_events: int
    unpriced_events: int

    @property
    def coverage(self) -> float:
        total = self.priced_events + self.unpriced_events
        if not total:
            return 0.0
        return self.priced_events / total


def _estimates(rows: Iterable, price_map: Mapping[str, Price]) -> CostEstimate:
    estimated: float | None = None
    priced = 0
    unpriced = 0
    for row in rows:
        price = price_map.get(row.model) if row.model else None
        value = estimate_cost(row.tokens, row.scheme, price)
        if value is None:
            unpriced += 1
        else:
            priced += 1
            estimated = (estimated or 0.0) + value
    return CostEstimate(estimated=estimated, priced_events=priced, unpriced_events=unpriced)


def cost_summary(
    conn: sqlite3.Connection,
    flt: UsageFilter,
    price_map: Mapping[str, Price],
) -> CostEstimate:
    """Estimated cost over every event matching ``flt``."""
    return _estimates(fetch_usage_rows(conn, flt), price_map)


def cost_by_dimension(
    conn: sqlite3.Connection,
    flt: UsageFilter,
    price_map: Mapping[str, Price],
    *,
    dimension: str = "model",
    canonical: Callable[[str], CanonicalName] | None = None,
) -> dict[str, CostEstimate]:
    """Estimated cost per distinct value of ``dimension``.

    ``canonical`` merges values that name the same thing (a model served by
    two providers) into one entry.
    """
    grouped: dict[str, list] = {}
    spellings: dict[str, list[str]] = defaultdict(list)
    for row in fetch_usage_rows(conn, flt):
        raw = _dimension_key(row, dimension)
        if canonical is None:
            key, label = raw, raw
        else:
            name = canonical(raw)
            key, label = name.identity, name.label
        grouped.setdefault(key, []).append(row)
        spellings[key].append(label)
    return {
        _display_key(key, spellings): _estimates(rows, price_map)
        for key, rows in grouped.items()
    }
