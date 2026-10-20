"""Claude-Code-shaped harness adapters (Command Code, CodeBuddy, ZCode).

These harnesses write Claude Code's transcript layout: one JSONL file per
session under ``<root>/projects/<slug>/``, assistant messages carrying
``message.usage`` in Anthropic categories and tool-use/tool-result blocks.

Differences that matter:

* **Command Code** wraps every message in a ``{"type":"message", ...}``
  envelope after a ``{"type":"session", ...}`` header (format v3), and keeps
  sidecar streams beside each transcript — ``<session>.checkpoints.jsonl``
  (state snapshots), ``<session>.prompts.jsonl`` and the client's own
  ``<session>.v2.bak`` backup. Without the filter these add phantom
  sessions. Its oldest format (v2) was flat
  ``{"role","content","timestamp","sessionId"}`` lines with no usage; those
  are read as turns but produce no usage events.
* **CodeBuddy Code / ZCode** are byte-for-byte Claude Code transcripts in
  their own ``~/.<name>/projects`` tree, so they reuse the base parser
  unchanged.

Tool names are each harness's own (``shell_command``, ``read_file``,
``edit_file``, ``write_file`` …); the shared category mapper handles them by
prefix.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from toknt.adapters.base import opt_str
from toknt.adapters.claude_code import ClaudeCodeAdapter
from toknt.discovery import (
    DiscoveredSource,
    discover_claude_shaped,
    discover_commandcode,
)
from toknt.models.events import Harness

#: Sidecar files Command Code keeps beside real transcripts.
_SIDECAR_SUFFIXES = (".checkpoints.jsonl", ".prompts.jsonl", ".v2.bak")


class CommandCodeAdapter(ClaudeCodeAdapter):
    source_id: ClassVar[str] = "commandcode:projects"
    agent: ClassVar[Harness] = Harness.COMMANDCODE
    kind: ClassVar[str] = "jsonl_dir"

    def discover(
        self,
        *,
        env: dict[str, str] | None = None,
        home: object | None = None,
        override: str | None = None,
    ) -> DiscoveredSource:
        return discover_commandcode(env=env, home=home, override=override)

    def _transcript_paths(self, root: Path) -> list[Path]:
        return sorted(
            path
            for path in root.rglob("*.jsonl")
            if not path.name.endswith(_SIDECAR_SUFFIXES)
        )

    def _record_type(self, record: dict) -> object:
        kind = record.get("type")
        if kind == "message":
            role = record.get("message", {})
            role = role.get("role") if isinstance(role, dict) else None
            return role if role in ("user", "assistant") else None
        if kind is None and record.get("role") == "user":
            return "user"  # v2 flat lines: turns only, no usage payload
        return None

    def _usage_key(self, record: dict, message: dict) -> str | None:
        # Command Code v3 envelopes carry the entry id at the top level.
        return opt_str(record.get("id")) or super()._usage_key(record, message)


class ClonedProjectAdapter(ClaudeCodeAdapter):
    """A harness whose ``~/.<name>/projects`` tree is Claude Code's format."""

    harness: ClassVar[Harness]

    def discover(
        self,
        *,
        env: dict[str, str] | None = None,
        home: object | None = None,
        override: str | None = None,
    ) -> DiscoveredSource:
        return discover_claude_shaped(self.harness, env=env, home=home, override=override)


class CodeBuddyAdapter(ClonedProjectAdapter):
    source_id: ClassVar[str] = "codebuddy:projects"
    agent: ClassVar[Harness] = Harness.CODEBUDDY
    harness: ClassVar[Harness] = Harness.CODEBUDDY


class ZCodeAdapter(ClonedProjectAdapter):
    source_id: ClassVar[str] = "zcode:projects"
    agent: ClassVar[Harness] = Harness.ZCODE
    harness: ClassVar[Harness] = Harness.ZCODE
