"""Canonical event model shared by every adapter and every analytics query.

Adapters translate native source formats into these records; storage persists
them; analytics never sees a source-specific shape. Records keep enough
provenance (agent, source id, source event type, quality flags) to trace any
number back to where it came from.

Only accounting-relevant fields are represented on purpose: no prompt bodies,
tool arguments or tool outputs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from toknt.models.usage import (
    TokenCounts,
    TokenScheme,
    effective_total,
    missing_components,
)

_E = type[StrEnum]


def _coerce(enum_cls: _E, value: Any, field_name: str) -> Any:
    """Accept str values for enum fields (adapters parse from JSON anyway)."""
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except ValueError as exc:
        allowed = ", ".join(repr(member.value) for member in enum_cls)
        raise ValueError(f"invalid {field_name}: {value!r} (allowed: {allowed})") from exc


def _coerce_notes(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    notes = tuple(str(note) for note in value)
    return notes


class Harness(StrEnum):
    """The coding harness a record came from.

    Values are stored in the database (``usage_events.agent``), so they are
    append-only: never renamed, only added. The word "harness" replaces
    "agent" in user-facing language because every other tool already calls
    plan/build modes "agents"; a harness is the CLI/IDE program that talks
    to the model.
    """

    OPENCODE = "opencode"
    CODEX = "codex"
    CLAUDE_CODE = "claude_code"
    # Claude-Code-shaped JSONL harnesses
    COMMANDCODE = "commandcode"
    CODEBUDDY = "codebuddy"
    ZCODE = "zcode"
    # Codex-CLI-shaped JSONL harnesses
    TRAE = "trae"
    # Gemini-CLI-shaped checkpoint stores
    GEMINI = "gemini"
    QWEN = "qwen"
    # Cline-shaped stores (modern CLI JSON pair + VS Code task dirs)
    CLINE = "cline"
    ROO = "roo"
    KILOCODE = "kilocode"
    # pi-shaped session trees
    PI = "pi"
    SENPI = "senpi"
    OMP = "omp"
    GJC = "gjc"
    PRIME = "prime"
    KIMCHI = "kimchi"
    # Distinct formats
    KIRO = "kiro"
    AMP = "amp"
    GOOSE = "goose"
    GROK = "grok"


#: Historical name for :class:`Harness` (the concept was renamed; the wire
#: values are identical). Kept as an alias so imports keep working.
Agent = Harness


class Quality(StrEnum):
    """Reliability of a record's fields."""

    OK = "ok"
    DERIVED = "derived"  # values derived (e.g. computed increments) rather than reported
    PARTIAL = "partial"  # some fields missing
    AMBIGUOUS = "ambiguous"  # source semantics unclear
    UNKNOWN = "unknown"


class Outcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"
    CANCELLED = "cancelled"
    PENDING = "pending"
    UNKNOWN = "unknown"


class OutcomeProvenance(StrEnum):
    """How an outcome was determined — required to interpret success rates."""

    STATUS_FIELD = "status_field"  # source carried an explicit status
    RESULT_FIELD = "result_field"  # source carried a structured result/error object
    OUTPUT_TEXT = "output_text"  # classified from result text (lower confidence)
    INFERRED = "inferred"  # inferred from context (e.g. next event seen)
    ABSENT = "absent"  # no evidence; outcome unknown/pending


class ToolCategory(StrEnum):
    SHELL = "shell"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    SEARCH = "search"
    WEB = "web"
    TASK = "task"
    OTHER = "other"


def stable_event_id(
    *,
    kind: str,
    agent: str,
    source_event_id: str | None,
    fallback: Mapping[str, Any] | None = None,
) -> str:
    """Deterministic identity for a canonical event.

    Prefers the source's own identifier. When a source has none, a hash of the
    supplied fallback fields (session, timestamp, model, sequence, ...) keeps
    repeated scans idempotent. Callers using the fallback should record a
    ``derived_event_id`` quality note.
    """
    if source_event_id:
        return f"{kind}:{agent}:{source_event_id}"
    if fallback is None:
        raise ValueError("source_event_id is None and no fallback identity was supplied")
    canonical = json.dumps(fallback, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]
    return f"{kind}:{agent}:derived:{digest}"


@dataclass(frozen=True, slots=True)
class UsageEvent:
    """One model response's token usage and reported cost."""

    id: str
    agent: Agent | str
    source_id: str
    session_id: str | None
    ts_ms: int
    source_event_id: str | None = None
    parent_session_id: str | None = None
    project: str | None = None
    directory: str | None = None
    agent_mode: str | None = None
    provider: str | None = None
    model: str | None = None
    tokens: TokenCounts = field(default_factory=TokenCounts)
    token_scheme: TokenScheme | str = TokenScheme.UNKNOWN
    provider_total_tokens: int | None = None
    cost_reported: float | None = None
    cost_currency: str | None = None
    finish_reason: str | None = None
    quality: Quality | str = Quality.OK
    quality_notes: tuple[str, ...] = ()
    source_event_type: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent", _coerce(Agent, self.agent, "agent"))
        object.__setattr__(
            self, "token_scheme", _coerce(TokenScheme, self.token_scheme, "token_scheme")
        )
        object.__setattr__(self, "quality", _coerce(Quality, self.quality, "quality"))
        object.__setattr__(self, "quality_notes", _coerce_notes(self.quality_notes))
        if not self.id:
            raise ValueError("usage event requires an id")
        if not self.source_id:
            raise ValueError("usage event requires a source_id")
        if not isinstance(self.ts_ms, int) or isinstance(self.ts_ms, bool) or self.ts_ms <= 0:
            raise ValueError(f"ts_ms must be positive epoch milliseconds, got {self.ts_ms!r}")
        if self.provider_total_tokens is not None and self.provider_total_tokens < 0:
            raise ValueError("provider_total_tokens must not be negative")
        if self.cost_reported is not None and self.cost_reported < 0:
            raise ValueError("cost_reported must not be negative")

    def consistency_notes(self) -> tuple[str, ...]:
        """Observations that make this event's numbers suspicious.

        Never rewrites values; surfaces discrepancies so ingestion can attach
        them as quality notes.
        """
        notes = list(self.quality_notes)
        scheme = TokenScheme(self.token_scheme)
        missing = missing_components(self.tokens, scheme)
        if scheme is not TokenScheme.UNKNOWN and missing:
            notes.append(f"missing_components: {', '.join(missing)}")

        if scheme is TokenScheme.OPENAI:
            if (
                self.tokens.cache_read is not None
                and self.tokens.input is not None
                and self.tokens.cache_read > self.tokens.input
            ):
                notes.append("cache_read_exceeds_input")

        computed = effective_total(self.tokens, scheme)
        if (
            computed is not None
            and self.provider_total_tokens is not None
            and computed != self.provider_total_tokens
        ):
            notes.append(
                "provider_total_mismatch: "
                f"reported {self.provider_total_tokens} vs components {computed}"
            )
        return tuple(notes)


@dataclass(frozen=True, slots=True)
class ToolEvent:
    """One tool call with its outcome and outcome provenance."""

    id: str
    agent: Agent | str
    source_id: str
    tool_name: str
    ts_ms: int
    source_event_id: str | None = None
    session_id: str | None = None
    call_id: str | None = None
    tool_category: ToolCategory | str = ToolCategory.OTHER
    duration_ms: int | None = None
    outcome: Outcome | str = Outcome.UNKNOWN
    outcome_provenance: OutcomeProvenance | str = OutcomeProvenance.ABSENT
    outcome_confidence: float | None = None
    model: str | None = None
    agent_mode: str | None = None
    quality: Quality | str = Quality.OK
    quality_notes: tuple[str, ...] = ()
    source_event_type: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent", _coerce(Agent, self.agent, "agent"))
        object.__setattr__(self, "outcome", _coerce(Outcome, self.outcome, "outcome"))
        object.__setattr__(
            self,
            "outcome_provenance",
            _coerce(OutcomeProvenance, self.outcome_provenance, "outcome_provenance"),
        )
        object.__setattr__(
            self, "tool_category", _coerce(ToolCategory, self.tool_category, "tool_category")
        )
        object.__setattr__(self, "quality", _coerce(Quality, self.quality, "quality"))
        object.__setattr__(self, "quality_notes", _coerce_notes(self.quality_notes))
        if not self.id:
            raise ValueError("tool event requires an id")
        if not self.source_id:
            raise ValueError("tool event requires a source_id")
        if not self.tool_name:
            raise ValueError("tool event requires a tool_name")
        if not isinstance(self.ts_ms, int) or isinstance(self.ts_ms, bool) or self.ts_ms <= 0:
            raise ValueError(f"ts_ms must be positive epoch milliseconds, got {self.ts_ms!r}")
        if self.duration_ms is not None and self.duration_ms < 0:
            raise ValueError("duration_ms must not be negative")
        if self.outcome_confidence is not None and not 0.0 <= self.outcome_confidence <= 1.0:
            raise ValueError(
                f"outcome_confidence must be in [0, 1], got {self.outcome_confidence!r}"
            )


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """Session metadata cache (no message content)."""

    agent: Agent | str
    session_id: str
    source_id: str
    parent_session_id: str | None = None
    project: str | None = None
    directory: str | None = None
    title: str | None = None
    agent_mode: str | None = None
    model: str | None = None
    provider: str | None = None
    time_created_ms: int | None = None
    time_updated_ms: int | None = None
    cost_reported: float | None = None
    quality: Quality | str = Quality.OK
    quality_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent", _coerce(Agent, self.agent, "agent"))
        object.__setattr__(self, "quality", _coerce(Quality, self.quality, "quality"))
        object.__setattr__(self, "quality_notes", _coerce_notes(self.quality_notes))
        if not self.session_id:
            raise ValueError("session record requires a session_id")
        if not self.source_id:
            raise ValueError("session record requires a source_id")
