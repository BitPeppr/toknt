"""Configuration loading tests: defaults, full file, validation errors."""

from __future__ import annotations

from pathlib import Path

import pytest

from toknt import config
from toknt.models.events import Harness


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


class TestDefaults:
    def test_missing_file_gives_defaults(self, tmp_path: Path) -> None:
        cfg = config.load_config(tmp_path / "nope.toml")
        assert cfg.enabled_harnesses == tuple(Harness)
        assert cfg.default_since == "30d"
        assert cfg.default_bucket == "day"
        assert cfg.ignore_projects == ()
        assert cfg.aliases == {}
        assert cfg.pricing == {}
        assert cfg.source_overrides == {}


class TestFullConfig:
    def test_every_section_is_read(self, tmp_path: Path) -> None:
        path = write(
            tmp_path,
            """
[general]
since = "7d"
bucket = "week"

[adapters]
enabled = ["codex"]
[adapters.paths]
codex = "/data/codex"

[display]
[display.aliases]
"gpt-5-codex" = "Codex"

[privacy]
ignore_projects = ["/secret"]

[pricing."gpt-5-codex"]
input = 1.5
output = 12.0
cache_read = 0.15
""",
        )
        cfg = config.load_config(path)
        assert cfg.enabled_harnesses == (Harness.CODEX,)
        assert cfg.source_overrides == {"codex": "/data/codex"}
        assert cfg.default_since == "7d"
        assert cfg.default_bucket == "week"
        assert cfg.ignore_projects == ("/secret",)
        assert cfg.aliases == {"gpt-5-codex": "Codex"}
        assert cfg.pricing["gpt-5-codex"].input_per_mtok == 1.5
        assert cfg.pricing["gpt-5-codex"].output_per_mtok == 12.0
        assert cfg.pricing["gpt-5-codex"].cache_read_per_mtok == 0.15
        assert cfg.pricing["gpt-5-codex"].cache_write_per_mtok is None

    def test_display_name_uses_aliases(self, tmp_path: Path) -> None:
        cfg = config.AppConfig(aliases={"m": "Nice name"})
        assert cfg.display_name("m") == "Nice name"
        assert cfg.display_name("other") == "other"
        assert cfg.display_name(None) == "(unknown)"

    def test_empty_enabled_list_disables_every_adapter(self, tmp_path: Path) -> None:
        path = write(tmp_path, "[adapters]\nenabled = []\n")
        cfg = config.load_config(path)
        assert cfg.enabled_harnesses == ()

    def test_enabled_subset_preserves_order(self, tmp_path: Path) -> None:
        path = write(tmp_path, '[adapters]\nenabled = ["claude_code", "opencode"]\n')
        cfg = config.load_config(path)
        assert cfg.enabled_harnesses == (Harness.CLAUDE_CODE, Harness.OPENCODE)


class TestValidation:
    def test_unknown_agent_rejected(self, tmp_path: Path) -> None:
        path = write(tmp_path, '[adapters]\nenabled = ["not_a_harness"]\n')
        with pytest.raises(config.ConfigError, match="unknown harnesses"):
            config.load_config(path)

    def test_unknown_path_override_rejected(self, tmp_path: Path) -> None:
        path = write(tmp_path, '[adapters.paths]\nnot_a_harness = "/x"\n')
        with pytest.raises(config.ConfigError, match="unknown harnesses"):
            config.load_config(path)

    def test_bad_bucket_rejected(self, tmp_path: Path) -> None:
        path = write(tmp_path, '[general]\nbucket = "fortnight"\n')
        with pytest.raises(config.ConfigError, match="hour, day or week"):
            config.load_config(path)

    def test_negative_price_rejected(self, tmp_path: Path) -> None:
        path = write(tmp_path, '[pricing."m"]\ninput = -1\n')
        with pytest.raises(config.ConfigError, match="non-negative"):
            config.load_config(path)

    def test_type_errors_rejected(self, tmp_path: Path) -> None:
        path = write(tmp_path, '[display.aliases]\nm = 1\n')
        with pytest.raises(config.ConfigError, match="table of strings"):
            config.load_config(path)
        path = write(tmp_path, "[adapters]\nenabled = [1, 2]\n")
        with pytest.raises(config.ConfigError, match="list of strings"):
            config.load_config(path)

    def test_invalid_toml_rejected(self, tmp_path: Path) -> None:
        path = write(tmp_path, "this is not [ toml\n")
        with pytest.raises(config.ConfigError, match="cannot read config"):
            config.load_config(path)


class TestLocations:
    def test_env_overrides_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path / "data"))
        monkeypatch.setenv(config.CONFIG_FILE_ENV, str(tmp_path / "cfg.toml"))
        assert config.data_dir() == tmp_path / "data"
        assert config.db_path() == tmp_path / "data" / "toknt.db"
        assert config.config_path() == tmp_path / "cfg.toml"

    def test_ensure_data_dir_creates_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "data-sub"
        monkeypatch.setenv(config.DATA_DIR_ENV, str(target))
        assert config.ensure_data_dir() == target
        assert target.is_dir()
