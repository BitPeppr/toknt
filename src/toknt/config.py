"""Application configuration and filesystem locations.

Everything is stored locally:

* the SQLite analytics index lives under the platform data directory
  (``~/Library/Application Support/toknt`` on macOS, XDG on Linux);
* the optional configuration file lives under the platform config directory
  (``~/Library/Application Support/toknt/config.toml`` on macOS, XDG on Linux);
* ``TOKNT_DATA_DIR`` and ``TOKNT_CONFIG`` override both, which tests use to
  stay hermetic.

The application name is centralized here (and in ``pyproject.toml``) so the
working name stays easy to change.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from platformdirs import user_config_path, user_data_path

from toknt.models.events import Harness

APP_NAME = "toknt"
DATA_DIR_ENV = "TOKNT_DATA_DIR"
CONFIG_FILE_ENV = "TOKNT_CONFIG"
DB_FILENAME = "toknt.db"


class ConfigError(Exception):
    """Raised when the configuration file exists but cannot be used."""


@dataclass(frozen=True, slots=True)
class PricingOverride:
    """Per-million-token prices for one model, supplied by the user."""

    input_per_mtok: float | None = None
    output_per_mtok: float | None = None
    cache_read_per_mtok: float | None = None
    cache_write_per_mtok: float | None = None


@dataclass(frozen=True, slots=True)
class AppConfig:
    enabled_harnesses: tuple[Harness, ...] = tuple(Harness)
    source_overrides: Mapping[str, str] = field(default_factory=dict)
    default_since: str = "30d"
    default_bucket: str = "day"
    ignore_projects: tuple[str, ...] = ()
    aliases: Mapping[str, str] = field(default_factory=dict)
    pricing: Mapping[str, PricingOverride] = field(default_factory=dict)
    ui_graphics: str = "auto"  # auto | kitty | iterm2 | sixel | off

    #: Historical name (the concept was renamed to harnesses).
    @property
    def enabled_agents(self) -> tuple[Harness, ...]:
        return self.enabled_harnesses

    def display_name(self, model: str | None) -> str:
        if model is None:
            return "(unknown)"
        return self.aliases.get(model, model)


def data_dir() -> Path:
    """Directory that holds the local analytics database."""
    override = os.environ.get(DATA_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return user_data_path(APP_NAME, appauthor=False)


def db_path() -> Path:
    """Path of the local analytics SQLite database."""
    return data_dir() / DB_FILENAME


def config_path() -> Path:
    """Path of the optional configuration file."""
    override = os.environ.get(CONFIG_FILE_ENV)
    if override:
        return Path(override).expanduser()
    return user_config_path(APP_NAME, appauthor=False) / "config.toml"


def ensure_data_dir() -> Path:
    """Create the data directory if needed and return it."""
    directory = data_dir()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _as_str_list(value: object, what: str, path: Path) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ConfigError(f"{path}: {what} must be a list of strings")
    return tuple(value)


def _as_str_map(value: object, what: str, path: Path) -> Mapping[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in value.items()
    ):
        raise ConfigError(f"{path}: {what} must be a table of strings")
    return dict(value)


def _as_float(value: object, what: str, path: Path) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ConfigError(f"{path}: {what} must be a non-negative number")
    return float(value)


def _parse_pricing(raw: object, path: Path) -> Mapping[str, PricingOverride]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: pricing must be a table of model tables")
    pricing: dict[str, PricingOverride] = {}
    for model, table in raw.items():
        if not isinstance(model, str) or not isinstance(table, dict):
            raise ConfigError(f"{path}: pricing entries must be [pricing.\"<model>\"] tables")
        pricing[model] = PricingOverride(
            input_per_mtok=_as_float(table.get("input"), f"pricing.{model}.input", path),
            output_per_mtok=_as_float(table.get("output"), f"pricing.{model}.output", path),
            cache_read_per_mtok=_as_float(
                table.get("cache_read"), f"pricing.{model}.cache_read", path
            ),
            cache_write_per_mtok=_as_float(
                table.get("cache_write"), f"pricing.{model}.cache_write", path
            ),
        )
    return pricing



def _graphics_setting(raw: object, config_file: Path | None) -> str:
    """The graphics transport: ``TOKNT_GRAPHICS`` beats the config file.

    The environment wins because it is the documented per-invocation
    override, and the TUI reads the config rather than the environment.
    """
    valid = ("auto", "kitty", "iterm2", "sixel", "off")
    env_graphics = os.environ.get("TOKNT_GRAPHICS")
    if env_graphics is not None:
        if env_graphics.casefold() not in valid:
            raise ConfigError(
                "TOKNT_GRAPHICS must be one of " + ", ".join(valid)
            )
        return env_graphics.casefold()
    if raw is None:
        return "auto"
    if not isinstance(raw, str) or raw.casefold() not in valid:
        where = f"{config_file}: " if config_file else ""
        raise ConfigError(
            f"{where}presentation.graphics must be one of " + ", ".join(valid)
        )
    return raw.casefold()

def load_config(path: Path | None = None) -> AppConfig:
    """Load the configuration file; a missing file means defaults."""
    config_file = path if path is not None else config_path()
    if not config_file.exists():
        # The environment override still applies without a config file:
        # TOKNT_GRAPHICS is the documented per-invocation switch.
        return AppConfig(ui_graphics=_graphics_setting(None, None))
    try:
        raw = tomllib.loads(config_file.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError) as exc:
        raise ConfigError(f"{config_file}: cannot read config: {exc}") from exc

    general = raw.get("general", {})
    adapters = raw.get("adapters", {})
    display = raw.get("display", {})
    privacy = raw.get("privacy", {})
    for name, section in (
        ("general", general),
        ("adapters", adapters),
        ("display", display),
        ("privacy", privacy),
        ("presentation", raw.get("presentation", {})),
    ):
        if not isinstance(section, dict):
            raise ConfigError(f"{config_file}: [{name}] must be a table")

    since = general.get("since", "30d")
    bucket = general.get("bucket", "day")
    if not isinstance(since, str) or not isinstance(bucket, str):
        raise ConfigError(f"{config_file}: general.since and general.bucket must be strings")
    if bucket not in ("hour", "day", "week"):
        raise ConfigError(f"{config_file}: general.bucket must be hour, day or week")

    enabled_raw = _as_str_list(adapters.get("enabled"), "adapters.enabled", config_file)
    valid_harnesses = {str(harness) for harness in Harness}
    if "enabled" in adapters:
        unknown = set(enabled_raw) - valid_harnesses
        if unknown:
            raise ConfigError(
                f"{config_file}: adapters.enabled contains unknown harnesses: {sorted(unknown)}"
            )
        enabled = tuple(Harness(name) for name in enabled_raw)
    else:
        enabled = AppConfig().enabled_harnesses

    overrides = _as_str_map(adapters.get("paths"), "adapters.paths", config_file)
    unknown_paths = set(overrides) - valid_harnesses
    if unknown_paths:
        raise ConfigError(
            f"{config_file}: adapters.paths contains unknown harnesses: {sorted(unknown_paths)}"
        )

    graphics_raw = raw.get("presentation", {}).get("graphics")
    graphics_value = _graphics_setting(graphics_raw, config_file)

    return AppConfig(
        enabled_harnesses=enabled,
        source_overrides=overrides,
        default_since=since,
        default_bucket=bucket,
        ignore_projects=_as_str_list(
            privacy.get("ignore_projects"), "privacy.ignore_projects", config_file
        ),
        aliases=_as_str_map(display.get("aliases"), "display.aliases", config_file),
        pricing=_parse_pricing(raw.get("pricing"), config_file),
        ui_graphics=graphics_value,
    )
