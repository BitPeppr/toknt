"""Token accounting semantics — the correctness core of the whole tool."""

from __future__ import annotations

import pytest

from toknt.models.usage import (
    TokenCounts,
    TokenScheme,
    additive_fields,
    effective_total,
    is_complete,
    missing_components,
)


class TestTokenCounts:
    def test_defaults_are_all_none(self) -> None:
        counts = TokenCounts()
        assert counts.input is None
        assert counts.output is None
        assert counts.reasoning is None
        assert counts.cache_read is None
        assert counts.cache_write is None

    def test_rejects_negative_values(self) -> None:
        with pytest.raises(ValueError, match="must not be negative"):
            TokenCounts(input=-1)

    def test_rejects_non_integer_values(self) -> None:
        with pytest.raises(ValueError, match="must be an int or None"):
            TokenCounts(input=12.5)  # type: ignore[arg-type]

    def test_rejects_booleans(self) -> None:
        with pytest.raises(ValueError, match="must be an int or None"):
            TokenCounts(output=True)  # type: ignore[arg-type]

    def test_zero_is_a_valid_reported_count(self) -> None:
        assert TokenCounts(cache_write=0).cache_write == 0


class TestEffectiveTotal:
    def test_anthropic_total_matches_observed_opencode_data(self) -> None:
        # Real response: provider reported total 18990.
        counts = TokenCounts(
            input=451, output=1162, reasoning=133, cache_read=17244, cache_write=0
        )
        assert effective_total(counts, TokenScheme.ANTHROPIC) == 18990

    def test_anthropic_total_includes_all_additive_components(self) -> None:
        counts = TokenCounts(input=1, output=2, reasoning=3, cache_read=4, cache_write=5)
        assert effective_total(counts, TokenScheme.ANTHROPIC) == 15

    def test_openai_total_matches_observed_codex_data(self) -> None:
        # Real response: provider reported total 32683; cached input is a subset
        # of input and reasoning a subset of output, so neither is added again.
        counts = TokenCounts(
            input=32377, output=306, reasoning=123, cache_read=31488, cache_write=0
        )
        assert effective_total(counts, TokenScheme.OPENAI) == 32683

    def test_openai_never_adds_cache_or_reasoning_to_total(self) -> None:
        counts = TokenCounts(input=100, output=10, reasoning=90, cache_read=80)
        assert effective_total(counts, TokenScheme.OPENAI) == 110

    def test_unknown_scheme_has_no_total(self) -> None:
        counts = TokenCounts(input=100, output=10)
        assert effective_total(counts, TokenScheme.UNKNOWN) is None

    def test_incomplete_events_refuse_a_total_by_default(self) -> None:
        counts = TokenCounts(input=100, output=None)
        assert effective_total(counts, TokenScheme.OPENAI) is None

    def test_incomplete_events_can_opt_into_a_partial_total(self) -> None:
        counts = TokenCounts(input=100, output=None)
        assert effective_total(counts, TokenScheme.OPENAI, allow_partial=True) == 100

    def test_empty_counts_produce_no_total(self) -> None:
        assert effective_total(TokenCounts(), TokenScheme.OPENAI, allow_partial=True) is None
        assert effective_total(TokenCounts(), TokenScheme.ANTHROPIC, allow_partial=True) is None


class TestComponentIntrospection:
    def test_additive_fields_per_scheme(self) -> None:
        assert additive_fields(TokenScheme.ANTHROPIC) == (
            "input",
            "output",
            "reasoning",
            "cache_read",
            "cache_write",
        )
        assert additive_fields(TokenScheme.OPENAI) == ("input", "output")
        assert additive_fields(TokenScheme.UNKNOWN) == ()

    def test_missing_components_reports_each_absent_field(self) -> None:
        counts = TokenCounts(input=1, output=2)
        assert missing_components(counts, TokenScheme.ANTHROPIC) == (
            "reasoning",
            "cache_read",
            "cache_write",
        )
        assert missing_components(counts, TokenScheme.OPENAI) == ()

    def test_is_complete(self) -> None:
        complete = TokenCounts(input=1, output=2, reasoning=0, cache_read=0, cache_write=0)
        assert is_complete(complete, TokenScheme.ANTHROPIC)
        assert is_complete(complete, TokenScheme.OPENAI)
        assert not is_complete(TokenCounts(input=1), TokenScheme.OPENAI)
        assert not is_complete(TokenCounts(), TokenScheme.UNKNOWN)
