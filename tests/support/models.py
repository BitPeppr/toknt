"""Builders for synthetic canonical events in analytics tests."""

from __future__ import annotations

from datetime import UTC, datetime

from toknt.models.events import Outcome, OutcomeProvenance, ToolCategory, ToolEvent, UsageEvent
from toknt.models.usage import TokenCounts, TokenScheme

UTC = UTC


def at(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> int:
    """UTC epoch milliseconds for a calendar moment (tests are timezone-free)."""
    return int(datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp() * 1000)


def usage(
    *,
    id: str,
    ts_ms: int,
    agent: str = "opencode",
    source_id: str = "opencode:db",
    model: str | None = "model-a",
    provider: str | None = "provider-a",
    session_id: str | None = "session-1",
    parent_session_id: str | None = None,
    project: str | None = "/work/alpha",
    agent_mode: str | None = "build",
    scheme: TokenScheme | str = TokenScheme.ANTHROPIC,
    input: int | None = 100,
    output: int | None = 50,
    reasoning: int | None = 10,
    cache_read: int | None = 5,
    cache_write: int | None = 0,
    provider_total: int | None = None,
    cost_reported: float | None = None,
    quality: str = "ok",
    quality_notes: tuple[str, ...] = (),
) -> UsageEvent:
    """A canonical usage event with sane anthropic-style defaults."""
    return UsageEvent(
        id=id,
        agent=agent,
        source_id=source_id,
        source_event_id=id.rsplit(":", 1)[-1],
        session_id=session_id,
        parent_session_id=parent_session_id,
        project=project,
        directory=project,
        agent_mode=agent_mode,
        provider=provider,
        model=model,
        ts_ms=ts_ms,
        tokens=TokenCounts(
            input=input,
            output=output,
            reasoning=reasoning,
            cache_read=cache_read,
            cache_write=cache_write,
        ),
        token_scheme=scheme,
        provider_total_tokens=provider_total,
        cost_reported=cost_reported,
        cost_currency="USD" if cost_reported is not None else None,
        quality=quality,
        quality_notes=quality_notes,
        source_event_type="test",
    )


def tool(
    *,
    id: str,
    ts_ms: int,
    tool_name: str = "bash",
    agent: str = "opencode",
    source_id: str = "opencode:db",
    session_id: str = "session-1",
    category: ToolCategory | str = ToolCategory.SHELL,
    outcome: Outcome | str = Outcome.SUCCESS,
    provenance: OutcomeProvenance | str = OutcomeProvenance.STATUS_FIELD,
    confidence: float | None = 1.0,
    duration_ms: int | None = None,
    model: str | None = "model-a",
) -> ToolEvent:
    return ToolEvent(
        id=id,
        agent=agent,
        source_id=source_id,
        source_event_id=id.rsplit(":", 1)[-1],
        session_id=session_id,
        call_id=id.rsplit(":", 1)[-1],
        tool_name=tool_name,
        tool_category=category,
        ts_ms=ts_ms,
        duration_ms=duration_ms,
        outcome=outcome,
        outcome_provenance=provenance,
        outcome_confidence=confidence,
        model=model,
        agent_mode="build",
        source_event_type="test",
    )
