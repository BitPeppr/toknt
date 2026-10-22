# Changelog

All notable changes to `toknt` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] — 2026-10-22

### Fixed

- A second read of the README caught what the first pass missed: the tools
  view reports a **p90** call duration (not "duration percentiles"), the `g`
  key was missing from the key table, the token-scheme notes were wrong for
  OpenCode (reasoning is a separate category there, not a structural zero)
  and for Gemini/Qwen (thoughts are additive, not inside output), the tok/s
  rate's treatment of those thoughts is now stated, `[general] since` is
  described as the CLI default (the explorer ignores it), the performance
  section now describes the skip check accurately (size + mtime everywhere,
  database plus write-ahead log for OpenCode) instead of claiming watermarks
  for every SQLite store, the test suite is no longer called "hermetic"
  (live-corpus tests read real stores when present), and `Plan.md` is
  spelled with its actual case.

## [0.2.1] — 2026-10-22

### Fixed

- The README was audited against the code and corrected where it had drifted:
  the harness count (22, not 21), the window cycle and the default window,
  the rescan behaviour, the command reference (`toknt graphics` was missing),
  the period options (`today`), the quality-issue codes, the source-support
  table (which harnesses are regression-tested against a live corpus), the
  privacy list (call durations and tool categories), the performance figures
  (re-measured) and the sixel terminal list.

## [0.2.0] — 2026-10-21

### Changed

- The explorer opens on the **3-day window** (was 90 days): wide enough to
  show a working week, narrow enough that the charts still say something.
- The `t` key walks the window cycle **backwards** — `3d → 1d → 5h → 3h →
  all → 1y → 90d → 30d → 7d` — so the recent windows are one keystroke
  away and everything else is one keystroke further.
- **Every launch rescans the sources**, and every window refreshes in the
  background from now on. A source whose files have not changed is skipped
  in milliseconds.
- The rescan runs in **its own process**: reading a busy source no longer
  stalls the interface while it happens.
- `python -m toknt` runs the command line interface.

### Fixed

- The OpenCode adapter re-read a seven-day window of messages on every
  scan — 18k rows of JSON, seconds of work. The window is now six hours
  (measured against a live corpus: no message was ever finalised later
  than that), and an unchanged database is not read at all. A refresh of an
  idle machine went from 2.4 s to 5 ms.

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

[0.2.2]: https://github.com/BitPeppr/toknt/releases/tag/v0.2.2
[0.2.1]: https://github.com/BitPeppr/toknt/releases/tag/v0.2.1
[0.2.0]: https://github.com/BitPeppr/toknt/releases/tag/v0.2.0
[0.1.1]: https://github.com/BitPeppr/toknt/releases/tag/v0.1.1
[0.1.0]: https://github.com/BitPeppr/toknt/releases/tag/v0.1.0
