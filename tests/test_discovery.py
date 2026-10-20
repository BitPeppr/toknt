"""Discovery tests: resolution order for each agent's data location."""

from __future__ import annotations

from pathlib import Path

from toknt.discovery import (
    discover_claude_code,
    discover_codex,
    discover_opencode,
)


def _touch_opencode(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "opencode.db").write_bytes(b"")
    return root


class TestOpenCodeDiscovery:
    def test_explicit_override_wins(self, tmp_path: Path) -> None:
        override = _touch_opencode(tmp_path / "elsewhere" / "opencode")
        found = discover_opencode(override=str(override), home=tmp_path / "home")
        assert found.exists
        assert found.location == str(override)
        assert found.agent == "opencode"

    def test_env_var_used_when_no_override(self, tmp_path: Path) -> None:
        data = _touch_opencode(tmp_path / "env-data")
        found = discover_opencode(
            env={"OPENCODE_DATA_DIR": str(data)}, home=tmp_path / "home"
        )
        assert found.exists
        assert found.location == str(data)

    def test_xdg_data_home_checked(self, tmp_path: Path) -> None:
        xdg = tmp_path / "xdg"
        _touch_opencode(xdg / "opencode")
        found = discover_opencode(env={"XDG_DATA_HOME": str(xdg)}, home=tmp_path / "home")
        assert found.exists
        assert found.location == str(xdg / "opencode")

    def test_macos_fallback_location(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        _touch_opencode(home / "Library" / "Application Support" / "opencode")
        found = discover_opencode(env={}, home=home)
        assert found.exists
        assert found.location == str(home / "Library" / "Application Support" / "opencode")

    def test_legacy_storage_only(self, tmp_path: Path) -> None:
        root = tmp_path / "opencode"
        (root / "storage" / "message" / "ses_x").mkdir(parents=True)
        found = discover_opencode(home=tmp_path / "home", override=str(root))
        assert found.exists
        assert found.note is not None
        assert "legacy" in found.note

    def test_missing_source_reports_note(self, tmp_path: Path) -> None:
        found = discover_opencode(env={}, home=tmp_path / "home")
        assert not found.exists
        assert found.note is not None
        assert "no opencode.db" in found.note


class TestCodexDiscovery:
    def test_codex_home_env(self, tmp_path: Path) -> None:
        codex_home = tmp_path / "codex-home"
        (codex_home / "sessions").mkdir(parents=True)
        found = discover_codex(env={"CODEX_HOME": str(codex_home)}, home=tmp_path / "home")
        assert found.exists
        assert found.location == str(codex_home / "sessions")
        assert found.agent == "codex"

    def test_default_home_location(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        (home / ".codex" / "sessions").mkdir(parents=True)
        found = discover_codex(env={}, home=home)
        assert found.exists
        assert found.location == str(home / ".codex" / "sessions")

    def test_missing_sessions_reports_note(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        (home / ".codex").mkdir(parents=True)
        found = discover_codex(env={}, home=home)
        assert not found.exists
        assert found.note is not None


class TestClaudeCodeDiscovery:
    def test_config_dir_env(self, tmp_path: Path) -> None:
        claude_home = tmp_path / "claude-home"
        (claude_home / "projects").mkdir(parents=True)
        found = discover_claude_code(
            env={"CLAUDE_CONFIG_DIR": str(claude_home)}, home=tmp_path / "home"
        )
        assert found.exists
        assert found.location == str(claude_home / "projects")
        assert found.agent == "claude_code"

    def test_default_home_location(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        (home / ".claude" / "projects").mkdir(parents=True)
        found = discover_claude_code(env={}, home=home)
        assert found.exists
        assert found.location == str(home / ".claude" / "projects")

    def test_missing_projects_reports_note(self, tmp_path: Path) -> None:
        found = discover_claude_code(env={}, home=tmp_path / "home")
        assert not found.exists
        assert found.note is not None
