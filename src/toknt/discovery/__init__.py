"""Locate installed harnesses and their local telemetry stores.

Paths are resolved at scan time from environment variables and conventional
locations, in this order: an explicit override (from toknt's config file),
the harness's own environment variable, XDG locations, then the platform
default. Nothing here requires the harness to be running, and everything is
read-only.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from toknt.models.events import Harness

OPENCODE_SOURCE_ID = "opencode:db"
CODEX_SOURCE_ID = "codex:sessions"
CLAUDE_CODE_SOURCE_ID = "claude_code:projects"

#: Harnesses whose on-disk formats are recognized but deliberately not read
#: yet (encrypted, zstd-compressed or per-project SQLite stores). The doctor
#: command lists them so the UI can say "known, not readable" instead of
#: "unknown harness".
KNOWN_UNREADABLE: tuple[str, ...] = (
    "aider",
    "antigravity",
    "cherrystudio",
    "codewhale",
    "copilot",
    "crush",
    "cursor",
    "deepseek",
    "devin",
    "hermes",
    "jetbrains",
    "junie",
    "muse",
    "openclaw",
    "reasonix",
    "windsurf",
    "zed",
)


@dataclass(frozen=True, slots=True)
class DiscoveredSource:
    source_id: str
    agent: Harness
    kind: str  # "sqlite" | "jsonl_dir" | "json_dir"
    location: str  # absolute path of the source root or file
    exists: bool
    note: str | None = None


def _env(env: Mapping[str, str] | None, key: str) -> str | None:
    source = os.environ if env is None else env
    value = source.get(key)
    return value if value else None


def _home(home: Path | None) -> Path:
    return Path.home() if home is None else home


def _first_existing(candidates: Sequence[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def discover_opencode(
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    override: str | None = None,
) -> DiscoveredSource:
    """Locate the OpenCode data directory and its database / legacy storage."""
    base = _home(home)
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override).expanduser())
    data_env = _env(env, "OPENCODE_DATA_DIR")
    if data_env:
        candidates.append(Path(data_env).expanduser())
    xdg = _env(env, "XDG_DATA_HOME")
    if xdg:
        candidates.append(Path(xdg) / "opencode")
    candidates.append(base / ".local" / "share" / "opencode")
    candidates.append(base / "Library" / "Application Support" / "opencode")

    chosen = _first_existing(candidates) or candidates[-1]
    db_path = chosen / "opencode.db"
    legacy_dir = chosen / "storage" / "message"
    exists = db_path.exists() or legacy_dir.is_dir()
    note: str | None = None
    if not exists:
        note = f"no opencode.db or legacy storage under {chosen}"
    elif not db_path.exists():
        note = "database missing; using legacy file storage"
    elif legacy_dir.is_dir():
        note = "database present; legacy file storage also found"
    return DiscoveredSource(
        source_id=OPENCODE_SOURCE_ID,
        agent=Harness.OPENCODE,
        kind="sqlite",
        location=str(chosen),
        exists=exists,
        note=note,
    )


def discover_codex(
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    override: str | None = None,
) -> DiscoveredSource:
    """Locate Codex's session rollout directory."""
    base = _home(home)
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override).expanduser())
    codex_home = _env(env, "CODEX_HOME")
    if codex_home:
        candidates.append(Path(codex_home).expanduser())
    candidates.append(base / ".codex")

    chosen = _first_existing(candidates) or candidates[-1]
    sessions = chosen / "sessions"
    archived = chosen / "archived_sessions"
    exists = sessions.is_dir()
    note: str | None = None
    if not exists:
        note = f"no sessions directory under {chosen}"
    elif archived.is_dir() and any(archived.glob("*.jsonl")):
        note = "archived_sessions also present"
    return DiscoveredSource(
        source_id=CODEX_SOURCE_ID,
        agent=Harness.CODEX,
        kind="jsonl_dir",
        location=str(sessions),
        exists=exists,
        note=note,
    )


def discover_claude_code(
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    override: str | None = None,
) -> DiscoveredSource:
    """Locate Claude Code's project transcript directory."""
    base = _home(home)
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override).expanduser())
    claude_home = _env(env, "CLAUDE_CONFIG_DIR")
    if claude_home:
        candidates.append(Path(claude_home).expanduser())
    candidates.append(base / ".claude")

    chosen = _first_existing(candidates) or candidates[-1]
    projects = chosen / "projects"
    exists = projects.is_dir()
    note = None if exists else f"no projects directory under {chosen}"
    return DiscoveredSource(
        source_id=CLAUDE_CODE_SOURCE_ID,
        agent=Harness.CLAUDE_CODE,
        kind="jsonl_dir",
        location=str(projects),
        exists=exists,
        note=note,
    )


def _resolve(
    *,
    env: Mapping[str, str] | None,
    home: Path | None,
    override: str | None,
    env_vars: tuple[str, ...],
    defaults: tuple[str, ...],
) -> Path:
    """First existing candidate, else the last default (matches discovery order)."""
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override).expanduser())
    for name in env_vars:
        value = _env(env, name)
        if value:
            candidates.append(Path(value).expanduser())
    candidates.extend(Path(item) for item in defaults)
    return _first_existing(candidates) or candidates[-1]


def _dir_source(
    source_id: str,
    harness: Harness,
    root: Path,
    *,
    kind: str = "jsonl_dir",
    note: str | None = None,
) -> DiscoveredSource:
    return DiscoveredSource(
        source_id=source_id,
        agent=harness,
        kind=kind,
        location=str(root),
        exists=root.is_dir(),
        note=note or (None if root.is_dir() else f"no sessions directory at {root}"),
    )


# ------------------------------------------------------- Claude-shaped harnesses


def discover_commandcode(
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    override: str | None = None,
) -> DiscoveredSource:
    """Command Code writes Claude-Code-shaped transcripts per project."""
    base = _home(home)
    root = _resolve(
        env=env,
        home=home,
        override=override,
        env_vars=("COMMANDCODE_HOME",),
        defaults=(str(base / ".commandcode" / "projects"),),
    )
    return _dir_source("commandcode:projects", Harness.COMMANDCODE, root)


def discover_claude_shaped(
    harness: Harness,
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    override: str | None = None,
) -> DiscoveredSource:
    """CodeBuddy Code / ZCode: ``~/.<name>/projects/<slug>/<session>.jsonl``."""
    name = str(harness)
    base = _home(home)
    root = _resolve(
        env=env,
        home=home,
        override=override,
        env_vars=(f"{name.upper()}_CONFIG_DIR",),
        defaults=(str(base / f".{name}" / "projects"),),
    )
    return _dir_source(f"{name}:projects", harness, root)


# --------------------------------------------------------- Codex-shaped harnesses


def discover_trae(
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    override: str | None = None,
) -> DiscoveredSource:
    """TRAE CLI keeps Codex-style rollout files under ``~/.trae/cli/sessions``."""
    base = _home(home)
    root = _resolve(
        env=env,
        home=home,
        override=override,
        env_vars=("TRAE_HOME",),
        defaults=(str(base / ".trae" / "cli" / "sessions"),),
    )
    return _dir_source("trae:sessions", Harness.TRAE, root)


# ------------------------------------------------------- Gemini-shaped harnesses


def discover_gemini(
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    override: str | None = None,
) -> DiscoveredSource:
    """Gemini CLI chats live under ``~/.gemini/tmp/<hash>/chats``.

    ``GEMINI_CLI_HOME`` moves the whole ``.gemini`` directory; the chats tree
    is ``<home>/tmp/*/chats`` under either root.
    """
    base = _home(home)
    cli_home = _env(env, "GEMINI_CLI_HOME")
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override).expanduser())
    roots = [base / ".gemini"]
    if cli_home:
        roots.insert(0, Path(cli_home).expanduser() / ".gemini")
    for root_dir in roots:
        candidates.append(root_dir / "tmp")
    chosen = _first_existing(candidates) or candidates[-1]
    return DiscoveredSource(
        source_id="gemini:chats",
        agent=Harness.GEMINI,
        kind="jsonl_dir",
        location=str(chosen),
        exists=chosen.is_dir(),
        note=None if chosen.is_dir() else f"no chats directory at {chosen}",
    )


def discover_qwen(
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    override: str | None = None,
) -> DiscoveredSource:
    """Qwen Code (a Gemini CLI fork) writes ``~/.qwen/projects/*/chats``."""
    base = _home(home)
    root = _resolve(
        env=env,
        home=home,
        override=override,
        env_vars=("QWEN_HOME",),
        defaults=(str(base / ".qwen" / "projects"),),
    )
    return _dir_source("qwen:projects", Harness.QWEN, root)


# ---------------------------------------------------------- Cline-shaped stores


def _vscode_global_storage(base: Path) -> list[Path]:
    """globalStorage roots for every VS Code-family host on this platform."""
    support = base / "Library" / "Application Support"
    hosts = (
        "Code",
        "Code - Insiders",
        "VSCodium",
        "Cursor",
        "Windsurf",
        "Kiro",
        "Positron",
    )
    roots = [support / host / "User" / "globalStorage" for host in hosts]
    config = base / ".config"
    roots.extend(config / host / "User" / "globalStorage" for host in hosts)
    return roots


def discover_cline(
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    override: str | None = None,
) -> DiscoveredSource:
    """Cline's modern CLI sessions: ``~/.cline/data/sessions/<id>/``."""
    base = _home(home)
    cline_data = _env(env, "CLINE_DATA_DIR")
    cline_dir = _env(env, "CLINE_DIR")
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override).expanduser())
    if _env(env, "CLINE_SESSION_DATA_DIR"):
        candidates.append(Path(_env(env, "CLINE_SESSION_DATA_DIR")).expanduser())
    if cline_data:
        candidates.append(Path(cline_data).expanduser() / "sessions")
    if cline_dir:
        candidates.append(Path(cline_dir).expanduser() / "data" / "sessions")
    candidates.append(base / ".cline" / "data" / "sessions")
    chosen = _first_existing(candidates) or candidates[-1]
    return _dir_source("cline:sessions", Harness.CLINE, chosen)


def discover_vscode_tasks(
    harness: Harness,
    extension_id: str,
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    override: str | None = None,
) -> DiscoveredSource:
    """A Cline-family VS Code extension's task directory, if the host exists."""
    base = _home(home)
    name = str(harness)
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override).expanduser())
    for storage in _vscode_global_storage(base):
        candidates.append(storage / extension_id / "tasks")
    for candidate in candidates:
        if candidate.is_dir():
            return DiscoveredSource(
                source_id=f"{name}:tasks",
                agent=harness,
                kind="json_dir",
                location=str(candidate),
                exists=True,
            )
    return _dir_source(f"{name}:tasks", harness, candidates[0] if candidates else base)


# ------------------------------------------------------------- pi-family trees


def discover_pi_family(
    harness: Harness,
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    override: str | None = None,
) -> DiscoveredSource:
    """pi-shaped session trees: ``<root>/agent/sessions/**/*.jsonl``.

    Kimchi Coding nests under the XDG config directory instead of ``$HOME``.
    """
    name = str(harness)
    base = _home(home)
    defaults = [f"{base}/.{name}/agent/sessions"]
    if name == "kimchi":
        xdg_config = _env(env, "XDG_CONFIG_HOME")
        defaults = [
            f"{Path(xdg_config).expanduser() / 'kimchi' / 'harness' / 'sessions'}"
            if xdg_config
            else f"{base}/.config/kimchi/harness/sessions"
        ]
    if name == "prime":
        defaults = [f"{base}/.prime/agent/sessions"]
    root = _resolve(
        env=env,
        home=home,
        override=override,
        env_vars=(f"{name.upper()}_CODING_AGENT_DIR",),
        defaults=tuple(defaults),
    )
    return _dir_source(f"{name}:sessions", harness, root)


# --------------------------------------------------------- single-format stores


def discover_kiro(
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    override: str | None = None,
) -> DiscoveredSource:
    """Kiro CLI/IDE sessions and the headless CLI's SQLite store."""
    base = _home(home)
    root = _resolve(
        env=env,
        home=home,
        override=override,
        env_vars=("KIRO_HOME",),
        defaults=(str(base / ".kiro" / "sessions"),),
    )
    source = _dir_source("kiro:sessions", Harness.KIRO, root)
    if not source.exists:
        support = base / "Library" / "Application Support"
        headless = support / "kiro-cli" / "data.sqlite3"
        if headless.is_file():
            return DiscoveredSource(
                source_id="kiro:headless",
                agent=Harness.KIRO,
                kind="sqlite",
                location=str(headless),
                exists=True,
            )
    return source


def discover_amp(
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    override: str | None = None,
) -> DiscoveredSource:
    """Amp thread files under the XDG data directory."""
    base = _home(home)
    xdg = _env(env, "XDG_DATA_HOME")
    defaults: list[Path] = []
    if xdg:
        defaults.append(Path(xdg).expanduser() / "amp" / "threads")
    else:
        defaults.append(base / ".local" / "share" / "amp" / "threads")
    root = _resolve(
        env=env,
        home=home,
        override=override,
        env_vars=("AMP_HOME",),
        defaults=tuple(defaults),
    )
    return _dir_source("amp:threads", Harness.AMP, root)


def discover_goose(
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    override: str | None = None,
) -> DiscoveredSource:
    """Goose sessions.db (>= 1.10) with the legacy JSONL tree as fallback."""
    base = _home(home)
    xdg = _env(env, "XDG_DATA_HOME")
    goose_root = _env(env, "GOOSE_PATH_ROOT")
    roots: list[Path] = []
    if goose_root:
        roots.append(Path(goose_root).expanduser() / "data" / "sessions")
    if xdg:
        roots.append(Path(xdg).expanduser() / "goose" / "sessions")
    roots.extend(
        [
            base / ".local" / "share" / "goose" / "sessions",
            base / ".local" / "share" / "Block" / "goose" / "sessions",
            base / "Library" / "Application Support" / "goose" / "sessions",
            base / "Library" / "Application Support" / "Block" / "goose" / "sessions",
        ]
    )
    if override:
        roots.insert(0, Path(override).expanduser())
    for root in roots:
        db = root / "sessions.db"
        if db.is_file():
            return DiscoveredSource(
                source_id="goose:sessions",
                agent=Harness.GOOSE,
                kind="sqlite",
                location=str(db),
                exists=True,
            )
    for root in roots:
        if root.is_dir() and any(root.glob("*.jsonl")):
            return _dir_source("goose:sessions", Harness.GOOSE, root, note="legacy JSONL store")
    return _dir_source("goose:sessions", Harness.GOOSE, roots[0])


def discover_grok(
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    override: str | None = None,
) -> DiscoveredSource:
    """Grok Build keeps one directory of JSON state per session."""
    base = _home(home)
    root = _resolve(
        env=env,
        home=home,
        override=override,
        env_vars=("GROK_HOME",),
        defaults=(str(base / ".grok" / "sessions"),),
    )
    return _dir_source("grok:sessions", Harness.GROK, root)
