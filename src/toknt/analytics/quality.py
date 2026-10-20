"""Data-coverage reporting used by ``toknt doctor``.

Answers, per agent: how much of the ingested data actually carries the
fields analytics needs? Token totals silently omit events with incomplete
components by design, so the fraction of complete events must be visible
next to every aggregate.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

_COMPLETE_TOKENS_SQL = """
CASE token_scheme
    WHEN 'anthropic' THEN
        input_tokens IS NOT NULL AND output_tokens IS NOT NULL
        AND reasoning_tokens IS NOT NULL AND cache_read_tokens IS NOT NULL
        AND cache_write_tokens IS NOT NULL
    WHEN 'openai' THEN
        input_tokens IS NOT NULL AND output_tokens IS NOT NULL
    ELSE 0
END
"""

_KNOWN_OUTCOME_SQL = "outcome IN ('success', 'failure', 'denied', 'cancelled')"


@dataclass(frozen=True, slots=True)
class HarnessQuality:
    agent: str
    usage_events: int
    events_with_model: int
    events_with_complete_tokens: int
    events_with_reported_cost: int
    first_ms: int | None
    last_ms: int | None
    last_ingest_ms: int | None
    tool_calls: int
    tool_calls_with_known_outcome: int

    @property
    def token_coverage(self) -> float:
        if not self.usage_events:
            return 0.0
        return self.events_with_complete_tokens / self.usage_events

    @property
    def model_coverage(self) -> float:
        if not self.usage_events:
            return 0.0
        return self.events_with_model / self.usage_events

    @property
    def outcome_coverage(self) -> float:
        if not self.tool_calls:
            return 0.0
        return self.tool_calls_with_known_outcome / self.tool_calls


def harness_quality(conn: sqlite3.Connection) -> list[HarnessQuality]:
    """Per-harness field coverage over all ingested history."""
    usage_rows = conn.execute(
        f"""
        SELECT agent,
               COUNT(*) AS usage_events,
               SUM(CASE WHEN model IS NOT NULL THEN 1 ELSE 0 END) AS events_with_model,
               SUM({_COMPLETE_TOKENS_SQL}) AS events_with_complete_tokens,
               SUM(CASE WHEN cost_reported IS NOT NULL THEN 1 ELSE 0 END)
                   AS events_with_reported_cost,
               MIN(ts_ms) AS first_ms,
               MAX(ts_ms) AS last_ms,
               MAX(ingested_at_ms) AS last_ingest_ms
        FROM usage_events
        GROUP BY agent
        """
    ).fetchall()
    tool_rows = conn.execute(
        f"""
        SELECT agent,
               COUNT(*) AS tool_calls,
               SUM(CASE WHEN {_KNOWN_OUTCOME_SQL} THEN 1 ELSE 0 END)
                   AS tool_calls_with_known_outcome
        FROM tool_events
        GROUP BY agent
        """
    ).fetchall()
    tools_by_harness = {row["agent"]: row for row in tool_rows}

    result: list[HarnessQuality] = []
    for row in usage_rows:
        tool_row = tools_by_harness.get(row["agent"])
        result.append(
            HarnessQuality(
                agent=row["agent"],
                usage_events=row["usage_events"] or 0,
                events_with_model=row["events_with_model"] or 0,
                events_with_complete_tokens=row["events_with_complete_tokens"] or 0,
                events_with_reported_cost=row["events_with_reported_cost"] or 0,
                first_ms=row["first_ms"],
                last_ms=row["last_ms"],
                last_ingest_ms=row["last_ingest_ms"],
                tool_calls=(tool_row["tool_calls"] or 0) if tool_row else 0,
                tool_calls_with_known_outcome=(
                    (tool_row["tool_calls_with_known_outcome"] or 0) if tool_row else 0
                ),
            )
        )
    for agent, tool_row in tools_by_harness.items():
        if not any(item.agent == agent for item in result):
            result.append(
                HarnessQuality(
                    agent=agent,
                    usage_events=0,
                    events_with_model=0,
                    events_with_complete_tokens=0,
                    events_with_reported_cost=0,
                    first_ms=None,
                    last_ms=None,
                    last_ingest_ms=None,
                    tool_calls=tool_row["tool_calls"] or 0,
                    tool_calls_with_known_outcome=(
                        tool_row["tool_calls_with_known_outcome"] or 0
                    ),
                )
            )
    result.sort(key=lambda item: item.agent)
    return result


#: Historical name for :class:`HarnessQuality`.
AgentQuality = HarnessQuality
agent_quality = harness_quality
