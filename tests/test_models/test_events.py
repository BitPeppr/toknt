"""Canonical event model behaviour."""

from __future__ import annotations

import pytest

from toknt.models.events import (
    Agent,
    Outcome,
    OutcomeProvenance,
    Quality,
    SessionRecord,
    ToolCategory,
    ToolEvent,
    UsageEvent,
    stable_event_id,
)
from toknt.models.usage import TokenCounts, TokenScheme


def make_usage_event(**overrides: object) -> UsageEvent:
    base: dict = {
        "id": "usage:opencode:msg_test",
        "agent": "opencode",
        "source_id": "opencode:db",
        "session_id": "ses_test",
        "ts_ms": 1_790_941_680_162,
        "tokens": TokenCounts(
            input=451, output=1162, reasoning=133, cache_read=17244, cache_write=0
        ),
        "token_scheme": "anthropic",
        "provider_total_tokens": 18990,
        "model": "glm-5.3-flash-free",
        "provider": "opencode",
    }
    base.update(overrides)
    return UsageEvent(**base)


def make_tool_event(**overrides: object) -> ToolEvent:
    base: dict = {
        "id": "tool:opencode:call_test",
        "agent": "opencode",
        "source_id": "opencode:db",
        "tool_name": "bash",
        "ts_ms": 1_790_941_710_000,
        "outcome": "success",
        "outcome_provenance": "status_field",
        "outcome_confidence": 1.0,
    }
    base.update(overrides)
    return ToolEvent(**base)


class TestStableEventId:
    def test_prefers_the_source_identifier(self) -> None:
        event_id = stable_event_id(kind="usage", agent="opencode", source_event_id="msg_123")
        assert event_id == "usage:opencode:msg_123"

    def test_matches_the_fixture_id_convention(self) -> None:
        assert (
            stable_event_id(kind="usage", agent="opencode", source_event_id="msg_san_001")
            == "usage:opencode:msg_san_001"
        )

    def test_fallback_is_deterministic_for_identical_records(self) -> None:
        fallback = {"session_id": "ses_1", "ts_ms": 123, "model": "m"}
        first = stable_event_id(
            kind="tool", agent="codex", source_event_id=None, fallback=fallback
        )
        second = stable_event_id(
            kind="tool", agent="codex", source_event_id=None, fallback=fallback
        )
        assert first == second

    def test_fallback_differs_when_the_record_differs(self) -> None:
        first = stable_event_id(
            kind="tool", agent="codex", source_event_id=None, fallback={"ts_ms": 1}
        )
        second = stable_event_id(
            kind="tool", agent="codex", source_event_id=None, fallback={"ts_ms": 2}
        )
        assert first != second

    def test_fallback_requires_supplied_identity(self) -> None:
        with pytest.raises(ValueError, match="no fallback identity"):
            stable_event_id(kind="usage", agent="codex", source_event_id=None)


class TestUsageEvent:
    def test_coerces_strings_to_enums(self) -> None:
        event = make_usage_event(agent="codex", token_scheme="openai", quality="partial")
        assert event.agent is Agent.CODEX
        assert event.token_scheme is TokenScheme.OPENAI
        assert event.quality is Quality.PARTIAL

    def test_rejects_invalid_enum_values(self) -> None:
        with pytest.raises(ValueError, match="invalid agent"):
            make_usage_event(agent="not-an-agent")

    def test_rejects_non_positive_timestamps(self) -> None:
        with pytest.raises(ValueError, match="ts_ms"):
            make_usage_event(ts_ms=0)
        with pytest.raises(ValueError, match="ts_ms"):
            make_usage_event(ts_ms="now")

    def test_rejects_negative_reported_cost(self) -> None:
        with pytest.raises(ValueError, match="cost_reported"):
            make_usage_event(cost_reported=-0.01)

    def test_quality_notes_are_normalised_to_a_tuple(self) -> None:
        event = make_usage_event(quality_notes=["a", "b"])
        assert event.quality_notes == ("a", "b")


class TestConsistencyNotes:
    def test_healthy_event_has_no_notes(self) -> None:
        assert make_usage_event().consistency_notes() == ()

    def test_flags_provider_total_mismatch(self) -> None:
        notes = make_usage_event(provider_total_tokens=12345).consistency_notes()
        assert any("provider_total_mismatch" in note for note in notes)

    def test_flags_missing_components_for_a_known_scheme(self) -> None:
        event = make_usage_event(
            tokens=TokenCounts(input=1, output=2), provider_total_tokens=None
        )
        notes = event.consistency_notes()
        assert any(note.startswith("missing_components:") for note in notes)

    def test_flags_cached_input_exceeding_input_for_openai_scheme(self) -> None:
        event = make_usage_event(
            token_scheme="openai",
            tokens=TokenCounts(input=10, output=1, cache_read=11),
            provider_total_tokens=None,
            quality="ambiguous",
        )
        assert "cache_read_exceeds_input" in event.consistency_notes()

    def test_preserves_adapter_notes(self) -> None:
        event = make_usage_event(quality_notes=["adapter_note"])
        assert event.consistency_notes()[0] == "adapter_note"


class TestToolEvent:
    def test_defaults_to_unknown_outcome_without_evidence(self) -> None:
        event = ToolEvent(
            id="tool:opencode:call_x",
            agent="opencode",
            source_id="opencode:db",
            tool_name="bash",
            ts_ms=1,
        )
        assert event.outcome is Outcome.UNKNOWN
        assert event.outcome_provenance is OutcomeProvenance.ABSENT
        assert event.tool_category is ToolCategory.OTHER

    def test_accepts_string_enums(self) -> None:
        event = make_tool_event(
            outcome="failure", outcome_provenance="output_text", tool_category="shell"
        )
        assert event.outcome is Outcome.FAILURE
        assert event.outcome_provenance is OutcomeProvenance.OUTPUT_TEXT
        assert event.tool_category is ToolCategory.SHELL

    def test_rejects_confidence_outside_unit_interval(self) -> None:
        with pytest.raises(ValueError, match="outcome_confidence"):
            make_tool_event(outcome_confidence=1.5)

    def test_rejects_negative_duration(self) -> None:
        with pytest.raises(ValueError, match="duration_ms"):
            make_tool_event(duration_ms=-1)

    def test_requires_a_tool_name(self) -> None:
        with pytest.raises(ValueError, match="tool_name"):
            make_tool_event(tool_name="")


class TestSessionRecord:
    def test_minimal_record(self) -> None:
        record = SessionRecord(agent="codex", session_id="s1", source_id="codex:sessions")
        assert record.agent is Agent.CODEX
        assert record.parent_session_id is None

    def test_requires_session_id(self) -> None:
        with pytest.raises(ValueError, match="session_id"):
            SessionRecord(agent="codex", session_id="", source_id="codex:sessions")
