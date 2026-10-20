"""Token accounting primitives.

Providers report token usage with different, overlapping categories, so the
canonical model does **not** define one universal total. Every usage event
records which scheme it came from, and totals are computed per scheme.

Verified against real local data (2026-10):

* Anthropic-style (OpenCode): ``total`` = input + output + reasoning +
  cache_read + cache_write. Cache reads are reported separately from
  ``input``. Observed: 451 + 1162 + 133 + 17244 + 0 == 18990, matching the
  provider-reported total exactly.
* OpenAI-style (Codex CLI): ``total`` = input + output. Cached input is a
  *subset* of ``input``, and reasoning is a subset of ``output``. Observed:
  32377 + 306 == 32683, matching the provider-reported total exactly.

Neither scheme's categories are interchangeable with the other's. Analytics
must never add fields from different schemes into one number.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class TokenScheme(StrEnum):
    """How a source's token categories relate to each other."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    UNKNOWN = "unknown"


#: Field names that add up to the effective total for each scheme.
_ADDITIVE_FIELDS: dict[TokenScheme, tuple[str, ...]] = {
    TokenScheme.ANTHROPIC: ("input", "output", "reasoning", "cache_read", "cache_write"),
    TokenScheme.OPENAI: ("input", "output"),
    TokenScheme.UNKNOWN: (),
}


@dataclass(frozen=True, slots=True)
class TokenCounts:
    """Token categories for one usage event.

    ``None`` means "the source did not report this", never zero.
    """

    input: int | None = None
    output: int | None = None
    reasoning: int | None = None
    cache_read: int | None = None
    cache_write: int | None = None

    def __post_init__(self) -> None:
        # direct attribute access on the slots; dataclasses.fields() per
        # instance is too slow at 10^4 rows per interactive repaint
        for name in ("input", "output", "reasoning", "cache_read", "cache_write"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} tokens must be an int or None, got {value!r}")
            if value < 0:
                raise ValueError(f"{name} tokens must not be negative, got {value}")


def additive_fields(scheme: TokenScheme) -> tuple[str, ...]:
    """Names of the token fields that make up the effective total."""
    return _ADDITIVE_FIELDS[scheme]


def missing_components(counts: TokenCounts, scheme: TokenScheme) -> tuple[str, ...]:
    """Additive components the source did not report."""
    return tuple(name for name in additive_fields(scheme) if getattr(counts, name) is None)


def is_complete(counts: TokenCounts, scheme: TokenScheme) -> bool:
    """Whether every additive component for the scheme is present."""
    return bool(additive_fields(scheme)) and not missing_components(counts, scheme)


def effective_total(
    counts: TokenCounts,
    scheme: TokenScheme,
    *,
    allow_partial: bool = False,
) -> int | None:
    """Sum of the scheme's additive components, or ``None`` when undefined.

    By default this refuses to total incomplete events: a total built from
    "whatever happens to be present" would silently understate usage. Callers
    that knowingly accept a partial total (and annotate quality accordingly)
    can pass ``allow_partial=True``.
    """
    names = additive_fields(scheme)
    if not names:
        return None
    values = [getattr(counts, name) for name in names]
    present = [value for value in values if value is not None]
    if not present:
        return None
    if len(present) != len(values) and not allow_partial:
        return None
    return sum(present)
