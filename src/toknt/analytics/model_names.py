"""Canonical model identity across providers, harnesses and routing tiers.

The same underlying model shows up under different names depending on who
served it:

===========================  ===========================================
recorded name                what makes it different
===========================  ===========================================
``z-ai/glm-5.3-flash``       vendor prefix
``cline-free/muse-spark-1.3`` harness prefix
``glm-5.3-flash-free``       provider's free tier
``muse-spark-1.3-contributor`` vendor tier marker
``inkling:free``             routing tag
``llama-3-70b-q4_k_m``       quantization
===========================  ===========================================

This module derives one *identity* per model so the explorer can merge the
same model across sources — without collapsing genuinely different models.
``glm-5.3`` and ``glm-5.3-flash`` must never merge: version numbers, size
tiers (``flash``, ``mini``, ``ultra``) and capability markers (``vl``,
``instruct``) are part of the identity.

The rules, in order:

1. Drop a leading ``vendor/`` or ``harness/`` segment — who served a model
   is not part of what it is.
2. Drop a routing tag after ``:`` (``inkling:free`` → ``inkling``).
3. Drop decoration tokens — free tiers, vendor tier markers and
   quantization suffixes — which describe *how* a model was served.
4. Lower-case and split what is left into tokens (``muse-spark-1.3`` →
   ``muse``, ``spark``, ``1.3``).

The *label* is the readable spelling of the cleaned name; the *identity* is
the token list with number separators flattened, so ``gpt4o`` and
``gpt-4o`` share one identity while ``glm-5.3-flash`` and ``glm-5.3`` do
not.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

#: Tokens describing how a model was served, never which model it is.
DECORATION_TOKENS = frozenset(
    {
        # free tiers and vendor programmes
        "free",
        "contributor",
        "tier",
        "latest",
        # quantization spelled out
        "quant",
        "quantized",
    }
)

#: Trailing quantization markers ("llama-3-70b-q4_k_m" → "llama-3-70b").
#: They are stripped as a suffix, before separators are normalized, so tags
#: with underscores ("q4_k_m") survive intact long enough to be recognized.
_QUANT_SUFFIX_RE = re.compile(
    r"[-_.](?:"
    r"q\d+(?:_k_m|_k_s|_k|_m)?"
    r"|fp\d+"
    r"|mxfp\d+"
    r"|nf4"
    r"|bnb4"
    r"|awq"
    r"|gptq"
    r"|exl2"
    r"|int[348]"
    r"|gguf"
    r"|ggml"
    r")(?![a-z0-9])"
)

#: Vendor spellings that mean the same family. Applied to identity tokens
#: only — the readable label keeps whatever spelling the source used.
TOKEN_ALIASES = {
    "sglm": "glm",  # SiliconFlow's spelling of GLM
}

_TOKEN_RE = re.compile(r"[a-z]+|\d+(?:\.\d+)*")
_DECORATION_RE = re.compile(
    r"(?<![a-z0-9])(?:" + "|".join(sorted(DECORATION_TOKENS)) + r")(?![a-z0-9])"
)


@dataclass(frozen=True, slots=True)
class CanonicalName:
    """One model as the explorer sees it.

    ``identity`` decides what merges with what; ``label`` is what the UI
    shows. Two spellings of the same model share an identity but may carry
    slightly different labels (the group picks the most common one).
    """

    identity: str
    label: str


def _strip_prefix(text: str) -> str:
    """Drop a leading ``vendor/`` or ``harness/`` segment."""
    if "/" in text:
        return text.split("/", 1)[1]
    return text


def _strip_tag(text: str) -> str:
    """Drop a routing tag (``inkling:free`` → ``inkling``)."""
    if ":" in text:
        return text.split(":", 1)[0]
    return text


def clean_label(name: str) -> str:
    """The readable spelling of a model name, without provider decoration."""
    text = name.strip().lower()
    text = _strip_tag(_strip_prefix(text))
    stripped = None
    while stripped != text:  # a name may carry several markers ("-fp8-q4")
        stripped = text
        text = _QUANT_SUFFIX_RE.sub("", text)
    text = text.replace("_", "-").replace(" ", "-")
    text = _DECORATION_RE.sub("", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text


def identity_tokens(label: str) -> list[str]:
    """Tokens of a cleaned label, with number separators flattened."""
    tokens = _TOKEN_RE.findall(label)
    return [TOKEN_ALIASES.get(token, token).replace(".", "-") for token in tokens]


def canonical_model(name: str) -> CanonicalName:
    """Canonical identity and label for one recorded model name."""
    label = clean_label(name)
    tokens = identity_tokens(label)
    return CanonicalName(identity="-".join(tokens), label=label)


def group_label(labels: Iterable[str]) -> str:
    """Pick the label for a merged group.

    The most common spelling wins; ties go to the shortest, then the
    alphabetically first, so the result is stable across repaints.
    """
    counts = Counter(labels)
    if not counts:
        return ""
    return min(counts, key=lambda text: (-counts[text], len(text), text))
