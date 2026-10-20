"""Codex-CLI-shaped harness adapters (TRAE CLI).

TRAE CLI writes the same rollout format as Codex CLI — ``rollout-*.jsonl``
under its own home — so the Codex parser is reused wholesale; only the
discovery root and the harness identity differ. Both storage generations
(current ``token_usage_record`` events and legacy ``event_msg.token_count``
events) are handled by the base adapter.
"""

from __future__ import annotations

from typing import ClassVar

from toknt.adapters.codex import CodexAdapter
from toknt.discovery import DiscoveredSource, discover_trae
from toknt.models.events import Harness


class TraeAdapter(CodexAdapter):
    source_id: ClassVar[str] = "trae:sessions"
    agent: ClassVar[Harness] = Harness.TRAE
    kind: ClassVar[str] = "jsonl_dir"

    def discover(
        self,
        *,
        env: dict[str, str] | None = None,
        home: object | None = None,
        override: str | None = None,
    ) -> DiscoveredSource:
        return discover_trae(env=env, home=home, override=override)
