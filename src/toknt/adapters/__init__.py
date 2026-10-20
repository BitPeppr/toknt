"""Adapter registry.

Adapters are instantiated per scan; each one declares which harness it
reads. Formats that share a shape (Claude-Code-style, Codex-style, Gemini-
style, Cline-style, pi-style) reuse one parser parameterized by harness
identity and discovery root.
"""

from __future__ import annotations

from toknt.adapters.amp_like import AmpAdapter
from toknt.adapters.base import Adapter, ScanReport, categorize_tool, normalize_epoch_ms
from toknt.adapters.claude_code import ClaudeCodeAdapter
from toknt.adapters.claude_like import CodeBuddyAdapter, CommandCodeAdapter, ZCodeAdapter
from toknt.adapters.cline_like import (
    ClineSessionsAdapter,
    ClineTaskAdapter,
    KiloCodeTaskAdapter,
    RooTaskAdapter,
)
from toknt.adapters.codex import CodexAdapter
from toknt.adapters.codex_like import TraeAdapter
from toknt.adapters.gemini_like import GeminiAdapter, QwenAdapter
from toknt.adapters.goose_like import GooseAdapter
from toknt.adapters.grok_like import GrokAdapter
from toknt.adapters.kiro_like import KiroAdapter
from toknt.adapters.opencode import OpenCodeAdapter
from toknt.adapters.pi_like import PiFamilyAdapter

__all__ = [
    "Adapter",
    "ScanReport",
    "categorize_tool",
    "normalize_epoch_ms",
    "registered_adapters",
]


def _pi_subclass(harness, source_id: str) -> type[PiFamilyAdapter]:
    """One adapter class per pi-family harness (identity is class state)."""
    name = str(harness)

    def discover(
        self,
        *,
        env=None,
        home=None,
        override=None,
    ):
        from toknt.discovery import discover_pi_family

        return discover_pi_family(self.agent, env=env, home=home, override=override)

    return type(
        f"{name.capitalize()}PiAdapter",
        (PiFamilyAdapter,),
        {
            "source_id": source_id,
            "agent": harness,
            "discover": discover,
        },
    )


def registered_adapters() -> tuple[Adapter, ...]:
    """Every adapter the scanner knows about, in stable display order."""
    from toknt.models.events import Harness

    pi_family = tuple(
        _pi_subclass(harness, f"{str(harness)}:sessions")()
        for harness in (
            Harness.PI,
            Harness.SENPI,
            Harness.OMP,
            Harness.GJC,
            Harness.PRIME,
            Harness.KIMCHI,
        )
    )
    return (
        OpenCodeAdapter(),
        CodexAdapter(),
        ClaudeCodeAdapter(),
        CommandCodeAdapter(),
        CodeBuddyAdapter(),
        ZCodeAdapter(),
        TraeAdapter(),
        GeminiAdapter(),
        QwenAdapter(),
        ClineSessionsAdapter(),
        ClineTaskAdapter(),
        RooTaskAdapter(),
        KiloCodeTaskAdapter(),
        *pi_family,
        KiroAdapter(),
        AmpAdapter(),
        GooseAdapter(),
        GrokAdapter(),
    )
