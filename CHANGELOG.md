# Changelog

All notable changes to `toknt` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] — 2026-10-20

### Fixed

- Two tests only passed on a machine in one timezone: the CLI help assertions
  now look at the rendered text instead of the raw escape sequences, and the
  merged-series test buckets its (UTC-stamped) events in UTC.

## [0.1.0] — 2026-10-20

The first public release: local-first token, cost and tool analytics for AI
coding harnesses.

### Added

- **Interactive TUI** — contribution graph of tokens per day, per-model
  series with a token-share donut, cache-hit-rate bars that mirror the model
  table, tool outcomes, session rankings, and live 3h/5h/1d windows that
  rescan in the background.
- **Logarithmic token charts** — token counts plot on a log axis, so a
  5M-token model stays visible next to a 250M one. tok/s stays linear.
- **True image charts** — anti-aliased PNG output over the kitty, iTerm2 and
  sixel protocols, with a Braille-cell fallback everywhere else.
- **21 harness sources** — OpenCode, Codex CLI, Claude Code, Cline, pi, Grok
  Build, Gemini CLI and more, normalized into a canonical model.
- **Strict token semantics** — incomplete counts are never completed by
  guessing, and reported costs stay separate from estimated ones.
- **Commands** — `overview`, `scan`, `usage`, `models`, `tools`, `sessions`,
  `export`, `doctor`, `graphics`.
- **Derived tok/s** — mean generation rate per model, derived from message
  gaps with the sources' own reporting schemes.
- `toknt doctor` for source discovery, coverage and data-quality issues.

[0.1.1]: https://github.com/BitPeppr/toknt/releases/tag/v0.1.1
[0.1.0]: https://github.com/BitPeppr/toknt/releases/tag/v0.1.0
