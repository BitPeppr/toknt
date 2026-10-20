# toknt

Local-first usage analytics for AI coding harnesses. `toknt` reads the native
telemetry of **OpenCode**, **OpenAI Codex CLI**, **Claude Code**, **Cline**,
**pi**, **Grok Build**, **Gemini CLI** and more (21 harnesses, see
[Source support](#source-support)) from your machine, normalizes it into a
canonical model, keeps a local SQLite index, and renders token, cost and
tool-call analytics in the terminal:

- **an interactive TUI** — booting into a GitHub-style contribution graph of
  tokens per day, with per-model series, a token-share donut, cache-hit-rate
  bars, tool outcomes and session rankings one keypress away,
- **live 3h/5h/1d views** — hour-bucketed model charts with an auto-rescanning
  background scan, for watching token counts accumulate in near real time,
- **true image charts** — anti-aliased, high-fidelity graphics via the
  kitty / iTerm2 / sixel protocols where the terminal supports them
  (previously-impossible resolution: no botched character cells),
- daily token usage by model, harness or project (line, bar, stacked, cumulative),
- cached-input share, reasoning share and requests over time,
- per-harness and per-project token share, plus the day cards (picked day,
  peak day) on the contributions view,
- model inventory with reported _and_ estimated costs, kept strictly separate,
- tool-call outcomes with success-rate denominators, rate bars and duration
  percentiles,
- session rankings with per-session token share, outlier flags and
  parent/subagent attribution,
- CSV/JSON exports of the canonical records,
- `toknt doctor` for source discovery, coverage and data-quality issues.

Everything stays on disk. `toknt` never uploads telemetry, never authenticates
to providers, and makes no network requests at any point.

## Install

Requires Python 3.12+.

```bash
uv tool install .        # isolated install; pipx install . also works
toknt doctor             # what toknt found on this machine
toknt scan               # import local history (incremental)
toknt                    # interactive: usage explorer, boot view = contributions
toknt -h                 # all commands accept the short help flag
```

From a checkout, `uv sync` and prefix commands with `uv run` (e.g.
`uv run toknt doctor`).

## Interactive mode

Running bare `toknt` in a terminal enters the explorer instead of printing a
static dashboard. Every view opens with a bold heading and a hairline rule,
so the content reads as one page. The page fills the terminal up to 192
columns and centers itself beyond that, so the margins on the left and the
right always match — and a big monitor gets more content rather than
stretched rows.

It starts on the **contributions** view: every day of the chosen window
laid out like GitHub's contribution graph, colored by total tokens, with
weekday labels and month labels centered over their own month (never
colliding, even on dense windows). The calendar stays tight and sits
directly against its weekday labels — week columns are two cells wide with
a single space between them, never stretched across the page. On wide
terminals the page splits in two: the left holds the calendar and the days
it inspects (**picked day** under the cursor with its per-harness amounts,
then the **peak day**), while the right rail carries the **window total**
with its daily average, the **share-by-project** list, the
**share-by-harness** list (both with inline bars) and a quiet note about
toggled-off harnesses. On a wide screen the rail splits into two columns —
projects beside harnesses — and project paths show their tail
(`…/Project/kbd`), which is what tells two projects apart. Narrow terminals fold back to a single column, and
the legend there names the harnesses that carry data as colored swatches,
keeps toggled-off ones visible (`codex (off)`) and collapses the idle ones
into a single `+N idle` tail. A compact tab bar (view chips,
window/metric/harness state) marks the active view in its series color; the
bottom bar pairs each key with a dim action label.

| Key                        | Action                                                              |
| -------------------------- | ------------------------------------------------------------------- |
| `1`–`4`                    | contributions · tokens by model · tools · sessions                  |
| `t`                        | cycle window: `3h → 5h → 1d → 3d → 7d → 30d → 90d → 1y → all`    |
| `v`                        | cycle metric: total · input · output · reasoning · cached           |
| `m`                        | merge the same model served by different providers (one row each)   |
| `o` / `x` / `a`            | toggle OpenCode / Codex / Claude Code in every view                 |
| `h`                        | focus one harness (cycles all → each harness that has data → all)   |
| `←` `→` `↑` `↓`            | move the day cursor (contributions) or table selection              |
| `r`                        | rescan all sources in the background, then show fresh data          |
| `?`                        | key reference overlay (the chart underneath stays put)              |
| `q`, `Ctrl+C`              | quit                                                                |

The **3h, 5h and 1d windows are the live views**: the tokens-by-model chart
switches to hour buckets with clock-time tick labels (`19:00 20:00 …`), and
the app rescans every source in the background every ~30 seconds (quickly on
entry, then steadily), sliding the window forward with each completed scan —
token counts, the donut and the cache-rate bars move on their own while you
watch. The `‣ scanning…` chip shows when a refresh is in flight; the UI
stays interactive throughout.

Windows up to a week (`3h`, `5h`, `1d`, `3d`, `7d`) are charted by the
hour, so the day-scale windows still show the working rhythm instead of a
handful of flat daily totals; from `30d` on, the charts switch to day
buckets. The contribution grid always draws calendar days, so a short window
shows the days it covers inside the current week.

The models view renders a chart of the current metric split by the top 8
models beside a right rail that holds a **donut of the models' token share**
(on image backends; a colored share ribbon elsewhere), the model's legend
chips, and **cache hit-rate bars per model** (scheme-aware: cache reads are
a separate denominator for anthropic-style sources, a subset for openai-style
ones, and absent entirely for unknown schemes). Below, the per-model
breakdown table fills the zone width and ends with a `vs max` bar showing
each model's size relative to the busiest one. The selected **metric**
drives all of it — the chart, the donut's shares, the table's numbers and
the ranking — so switching to `input` re-ranks the table by input tokens
and reports input tokens in every column. Piping or scripting (non-tty
stdout) gets the old static print; `toknt overview` forces that dashboard,
and `TOKNT_TUI=0` disables the interactive boot entirely.

### Merging models across providers

The same model is recorded under different names depending on who served it:
`z-ai/glm-5.3-flash` (vendor prefix), `cline-free/muse-spark-1.3-contributor`
(harness prefix), `glm-5.3-flash-free` (provider's free tier),
`thinkingmachines/inkling:free` (routing tag), `llama-3-70b-q4_k_m`
(quantization). Pressing **`m`** merges those spellings into one row, one
chart series and one cache-rate bar — `muse-spark-1.3` from Cline and from
OpenCode become one model, and `z-ai/glm-5.3-flash` and `glm-5.3-flash-free`
become one. The same merge is available on the command line with
`toknt usage --merge-models` (also `toknt models --merge-models` and
`toknt overview --merge-models`).

The merge is deliberately narrow, because merging the wrong models would
quietly distort every number:

- **Dropped** (they describe how a model was served, not which model it is):
  a leading `vendor/` or `harness/` segment, a `:free`-style routing tag,
  free-tier and vendor-programme markers (`-free`, `-contributor`) and
  quantization suffixes (`-fp8`, `-q4_k_m`, `-awq`).
- **Kept** (they identify the model): version numbers, size tiers (`flash`,
  `mini`, `ultra`), capability markers (`vl`, `instruct`), release stages
  (`preview`) and exact build tags.

So `glm-5.3` and `glm-5.3-flash` stay apart, `muse-spark-1.3` and
`muse-spark-1.2` stay apart, and `nemotron-3-ultra` does not swallow
`nemotron-3-ultra-550b-a55b`. Merged rows are summarized as a whole, so
totals, event counts, sessions and reported cost stay consistent.

## Terminal graphics

When the terminal understands an image protocol, the models chart and the
token-share donut are rendered as anti-aliased bitmaps — soft-edged strokes,
transparent background, gridlines — and placed inline (kitty:
kitty/ghostty/WezTerm/Warp/rio, iTerm2: OSC 1337, sixel: xterm/foot).
Identical charts are re-rendered from a small cache, so navigating never
re-encodes the bitmap; when an image's slot disappears (view switch, smaller
window) it is erased with the terminal's own background color, so no ghost
graphs live behind the text. The help overlay is plain text and deliberately
leaves the chart bitmap on screen. Everything else degrades to sharp
Braille-dot charts, so the same data views work in a plain `screen`.

Check what your terminal reports — and whether it actually draws images:

```bash
toknt doctor          # the "Graphics" row names the backend and the reason
toknt graphics        # draws a test chart; force a transport with --backend
toknt graphics --alt  # same image inside the alternate screen (like the TUI)
```

`toknt graphics` prints the detected transport (kitty / iterm2 / sixel /
off) with the reason, then emits a real chart. If no image appears, your
terminal does not implement the detected protocol — force another one with
`TOKNT_GRAPHICS=kitty|sixel|iterm2` or `[presentation] graphics = "…"`.
Inside the TUI the same test is one keypress away: **`g`** draws the
identical test image.

Terminals without image support (Apple Terminal, Alacritty, most tmux
setups) use the Braille charts; toknt refuses to send images inside a
multiplexer, where they would be swallowed silently.

Detection can be forced:

```toml
[presentation]
graphics = "auto"   # auto | kitty | iterm2 | sixel | off
```

`TOKNT_GRAPHICS` overrides the same setting per invocation (`off` restores
character-cell charts everywhere). Non-tty output never emits images.

## Source support

A **harness** is the CLI/IDE program that talks to the model (the word
"agent" now commonly means plan/build modes, so toknt doesn't use it for
this). Every harness below feeds the same canonical model; adapters are
parameterized by format family where stores share a shape. Sources are
opened read-only and never modified.

| Harness                                           | Default location                                                                             | Status                                                            |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| OpenCode                                          | `~/.local/share/opencode` (or `OPENCODE_DATA_DIR`, `XDG_DATA_HOME`)                          | validated against real data (2026-10)                              |
| Codex CLI                                         | `~/.codex/sessions` (or `CODEX_HOME`)                                                        | validated against real data (2026-10)                              |
| Claude Code                                       | `~/.claude/projects` (or `CLAUDE_CONFIG_DIR`)                                                | fixtures only (not installed on the dev machine)                   |
| Cline (CLI)                                       | `~/.cline/data/sessions` (or `CLINE_DATA_DIR`)                                               | validated against real data (2026-10)                              |
| Cline / Roo Code / Kilo Code (VS Code)            | `<globalStorage>/<ext id>/tasks` across Code/Insiders/VSCodium/Cursor/Windsurf hosts         | documented format; sessions + tools (the format carries no tokens) |
| pi                                                | `~/.pi/agent/sessions`                                                                       | validated against real data (2026-10)                              |
| Grok Build                                        | `~/.grok/sessions` (or `GROK_HOME`)                                                          | validated against real data (2026-10)                              |
| Command Code                                      | `~/.commandcode/projects`                                                                    | Claude-shaped transcripts; fixtures only                           |
| CodeBuddy Code / ZCode                            | `~/.codebuddy/projects`, `~/.zcode/projects`                                                 | Claude-shaped transcripts; fixtures only                           |
| TRAE CLI                                          | `~/.trae/cli/sessions` (or `TRAE_HOME`)                                                      | Codex-shaped rollouts; fixtures only                               |
| Gemini CLI                                        | `~/.gemini/tmp/*/chats` (or `GEMINI_CLI_HOME`)                                               | documented checkpoint format; fixtures only                        |
| Qwen Code                                         | `~/.qwen/projects` (or `QWEN_HOME`)                                                          | Gemini-shaped; fixtures only                                       |
| pi-family (senpi, omp, gajae-code, prime, kimchi) | `~/.<name>/agent/sessions` (`~/.config/kimchi/harness/sessions` for kimchi)                  | pi-shaped session trees; fixtures only                             |
| Kiro (CLI/IDE/headless)                           | `~/.kiro/sessions`, `<app support>/kiro-cli/data.sqlite3`                                    | documented formats; sessions + tools (no token counts)             |
| Amp (Sourcegraph)                                 | `~/.local/share/amp/threads` (or `XDG_DATA_HOME`)                                            | thread JSON files; fixtures only                                   |
| Goose (Block)                                     | `<data>/goose/sessions/sessions.db`, legacy JSONL tree                                       | sessions only (the store carries no token counts)                  |

Recognized but deliberately unreadable (encrypted, zstd-compressed or
per-project stores without a stable schema): aider, antigravity,
cherry studio, codewhale, copilot, crush, cursor, deepseek, devin, hermes,
jetbrains, junie, muse, openclaw, reasonix, windsurf, zed — `toknt doctor`
lists them so "known" is never confused with "readable".

Config: `[adapters] enabled = [...]` accepts any of the harness ids above
(default: all); `paths` overrides a harness's root, e.g.
`paths = { cline = "/Volumes/backup/cline" }`.

## Commands

Period options accept `30d`, `12h`, `8w`, calendar dates (`2026-10-01`) and
`all`; `--harness` (alias `--agent`), `--model` and `--project` filters
repeat.

| Command                           | What it does                                                                                                                                                                                                                 |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `toknt`                           | interactive explorer (static overview print when stdout is not a tty)                                                                                                                                                        |
| `toknt overview`                  | static overview dashboard: totals, per-day chart by model, model and tool tables; `--merge-models`                                                                 |
| `toknt scan [--full]`             | incremental import; `--full` reindexes and reconciles deleted rows                                                                                                                                                           |
| `toknt usage`                     | token series: `--by model\|harness\|project` (`agent` kept as an alias), `--metric total\|input\|output\|reasoning\|cache_read\|cache_write\|cached\|requests`, `--bucket hour\|day\|week`, `--chart line\|bar\|stacked\|none`, `--cumulative`, `--top N`, `--merge-models` |
| `toknt models`                    | per-model requests, tokens, reported and estimated cost                                                                                                                                                                      |
| `toknt tools`                     | per-tool calls, success rate (known outcomes only), `--limit`                                                                                                                                                                |
| `toknt sessions`                  | sessions ranked by usage, `!` outliers, `--limit`                                                                                                                                                                            |
| `toknt export`                    | `--what usage\|tools\|sessions --format csv\|json` to stdout                                                                                                                                                                 |
| `toknt doctor`                    | configuration, detected/ingested sources, table counts, quality issues, pricing hint                                                                                                                                         |
| `toknt pricing list\|add\|remove` | per-model prices for cost estimates                                                                                                                                                                                          |

Examples:

```bash
toknt usage --since 90d --by model --metric cached --chart stacked
toknt tools --since 30d --chart trend --bucket week
toknt export --what usage --format csv --since 7d > usage.csv
toknt pricing add gpt-6-luna --input 1.25 --output 10 --cache-read 0.125
```

## Token accounting

Token categories overlap differently across providers, so every usage event
stores a `token_scheme` and totals are computed per scheme:

| Scheme                                                | How tokens add up                                     | Notes                                                                  |
| ----------------------------------------------------- | ----------------------------------------------------- | ---------------------------------------------------------------------- |
| `anthropic` (OpenCode, Claude Code, Cline, pi, Amp)   | input + output + reasoning + cache_read + cache_write | cache reads/writes are _separate_ from input; reasoning is inside output (structural zero) |
| `openai` (Codex CLI, TRAE, Gemini CLI, Qwen, Grok)    | input + output                                        | cached input is a **subset** of input; reasoning is a subset of output |

Source-specific rules enforced by the adapters:

- **OpenCode** stores a dual schema (`session`/`message` plus
  `session_v2`/`session_message`); records appearing in both are deduplicated
  by message id and tool calls by call id. Streaming rewrites update the same
  event instead of adding a second one. **Forked sessions** copy the parent's
  message history under new ids while the fork's own `session_v2.tokens_*`
  aggregate starts at zero — those copied rows are skipped (and counted in
  the scan note), so toknt's totals match opencode's own stats screen
  (verified: 2.68B vs opencode's 2.6B display; before the fix toknt showed
  2× because two fork chains re-counted their shared history).
- **Codex CLI** has two rollout generations. Files containing
  `token_usage_record` use only those per-response figures (the adjacent
  cumulative mirrors are ignored); older files derive per-response increments
  from `last_token_usage` (or from cumulative totals as a fallback, flagged
  with `quality=derived`). A file that switches generation mid-file triggers a
  retraction of its legacy rows instead of double counting.
- **Claude Code** transcripts contain the same response several times while
  streaming; snapshots are deduplicated by `requestId` keeping the largest
  usage. Extended thinking is billed inside `output_tokens`, so `reasoning` is
  reported as a structural zero — it is never guessed. Claude-shaped forks
  (Command Code, CodeBuddy, ZCode) reuse the parser; Command Code's sidecar
  streams (`.checkpoints.jsonl`, `.prompts.jsonl`, `.v2.bak`) are skipped so
  they never become phantom sessions.
- **Cline** per-response metrics come from each assistant message
  (`metrics`/`modelInfo`); the manifest's session-level `metadata.usage` is
  deliberately ignored because it is stale after compaction (verified against
  a live corpus). `tool_result` entries with `{query, result, error, success}`
  records classify outcomes from their structured success flags.
- **pi-family** usage follows the models.dev convention (verified: `total ==
  input + output + cacheRead + cacheWrite` across a live corpus); failed API
  calls (`stopReason == "error"`) are skipped like Claude Code's API errors.
- **Grok** per-turn usage comes from each session's `usage.json` (`turns[]`
  with per-model splits); `total == input + output` (verified), so cached
  reads map as a subset of input. `costUsdTicks` has no documented scale and
  is never converted into a cost.
- **Gemini CLI / Qwen** record `total == input + output + thoughts` with
  `cached` inside input; toknt records the openai scheme and surfaces the
  thoughts as `reasoning`, flagging `provider_total_mismatch` on every event
  where thoughts > 0 instead of quietly folding them into a total.
- **Kiro, Cline-VS-Code, Goose** formats carry no token counts at all: their
  sessions and tool calls still count, and the missing tokens stay missing
  (quality-flagged), never estimated.
- Every usage event keeps its **provider-reported total** when the source
  reports one; when component math disagrees, the mismatch is recorded as a
  quality note rather than "fixed". Unknown schemes produce no total at all.
- Session identities keep their source ids; subagent sessions record a
  `parent_session_id` and an `agent_mode` (e.g. `subagent:Linnaeus`,
  `sidechain`) instead of being merged into the parent.

## Costs: reported vs estimated

- `reported cost` is what a source itself recorded per response (OpenCode's
  `cost`, Claude Code's `costUSD` when present). It is stored with the event.
- `estimated cost` is computed **at query time** from the pricing table and
  config overrides, so price updates never rewrite history. Codex rollouts
  carry no cost, so Codex models only ever show estimates.
- An estimate is produced only when every non-zero token category has a
  price; otherwise the value stays `—` and coverage is shown next to totals
  (e.g. `3 of 17,649 events priced (0%)`). Nothing is invented.
- Scheme-aware billing: anthropic-style events bill `reasoning` at the output
  rate; OpenAI-style events bill non-cached input at the input rate and
  cached input at the cache-read rate (never double-billed).

## Configuration

The optional config file lives in the platform config directory — run
`toknt doctor` to see the exact path (`~/.config/toknt/config.toml` when
`XDG_CONFIG_HOME` is set, `~/Library/Application Support/toknt/config.toml`
otherwise on macOS, XDG on Linux), or point `TOKNT_CONFIG` anywhere:

```toml
[general]
since = "30d"        # default window for all commands
bucket = "day"       # hour | day | week

[adapters]
enabled = ["opencode", "codex", "claude_code"]   # [] disables scanning entirely
paths = { codex = "/Volumes/external/codex" }    # per-agent source overrides

[display]
aliases = { "claude-sonnet-4-5" = "Sonnet 4.5" } # display only; ids stay exact

[privacy]
ignore_projects = ["/work/secret"]   # excluded from analytics

[pricing."gpt-6-luna"]               # override or extend table prices, field-wise
input = 1.25
output = 10
cache_read = 0.125
```

`TOKNT_DATA_DIR` relocates the database (default: platform application-data
directory, `~/Library/Application Support/toknt/toknt.db` on macOS).

## Diagnostics and data quality

`toknt doctor` explains, without reading any source content:

- where the database and config live, and which adapters are enabled,
- which sources exist _now_ and their notes (e.g. archived sessions found),
- what has been ingested, when it was last scanned and with which result,
- table row counts and per-agent coverage (requests, outcome observability),
- aggregated quality issues (`toknt doctor` footer), recording code,
  occurrence count and a safe sample reference — never prompt text, tool
  arguments or outputs. Counts describe the source as last observed: the
  same defect is never counted twice, different affected records are added
  up, and a `toknt scan --full` refreshes them instead of accumulating.

Quality codes: `unparsable_row`, `unreadable_file`, `missing_usage`,
`missing_token_info`, `orphan_tool_output`, `orphan_tool_result`,
`scan_failed`. When a source was last scanned with status `partial` or
`error`, doctor suggests `toknt scan --full` to rebuild it from scratch.

Tool success rates use only calls with a known final outcome; `pending` and
`unknown` calls are excluded from the denominator and shown separately, so a
success rate can never flatter the data.

## Privacy

Stored per event: timestamps, token counts, cost fields, model/provider/agent
ids, project paths, session ids, tool names, outcome status and quality notes.
**Never** stored: prompt bodies, model responses, tool arguments, tool
outputs, file contents, credentials.

## Performance

`toknt` reads sources incrementally (checkpoints with byte offsets and
fingerprints for JSONL, per-source watermarks for SQLite) and indexes
SQLite for timestamp/model/session/agent/tool queries. Unchanged files are
skipped by a stat check.

`tests/test_performance.py` builds a synthetic six-month history — three
harnesses, 30k usage events and 15k tool calls, including duplicate legacy
OpenCode rows, both Codex format generations and Claude snapshot dedupe —
and measures (Apple Silicon laptop, SSD, 2026-10):

| Scenario                                                                       | Wall time |
| ------------------------------------------------------------------------------ | --------- |
| Cold scan (everything, all agents)                                             | ≈2 s      |
| Warm scan (no changes)                                                         | ~0.07 s   |
| Incremental scan after appending a small batch per agent                       | ~0.07 s   |
| Analytics queries (daily series, tool outcomes, cost estimate over 30k events) | ~0.5 s    |

Real-machine full rescan of ~20k OpenCode events with full-size payloads plus
the Codex rollouts takes ≈5 s; synthetic JSON payloads are compact, so treat
cold-scan figures as a lower bound. Run `uv run pytest tests/test_performance.py -s -q`
to reproduce; the test also asserts per-agent row counts exactly, so repeated
scans can never inflate totals.

## Known limitations

- **Claude Code** and the fixture-validated harnesses above are covered by
  synthetic tests only (see the source-support table); treat their numbers as
  best effort until verified on a machine with real transcripts. The three
  harnesses marked "validated against real data" are regression-tested
  against the live corpus on the development machine in
  `tests/test_adapters/test_live_harnesses.py` (skipped automatically when a
  store is absent).
- Harnesses whose formats carry no token counts (Kiro, Cline's VS Code task
  dirs, Goose) show sessions and tool outcomes but no token/cost figures —
  by design, never estimated.
- Codex rollouts carry no reported cost; cost columns for Codex are estimates
  once you add prices, otherwise `—`.
- A reported cost of `$0.00` means _the source recorded zero_ (typical for
  free/subscription models), not that toknt estimated zero.
- Models with unknown token schemes show no total; partial usage stays
  partial (flagged), never completed by guessing.
- Estimated costs ignore provider pricing subtleties beyond the documented
  four categories (e.g. batch discounts, long-context surcharges).
- Charts render with `plotext`; very narrow terminals fold tables but charts
  need roughly 80+ columns for full legibility. Time-series charts start at
  your earliest stored event (even with `--since all`) and, once the span
  exceeds one year, tick labels carry the two-digit year so the chronology
  stays unambiguous.

## Development

```bash
uv sync                        # create the venv and install dependencies
uv run pytest                  # full suite (hermetic; never reads real history)
uv run pytest tests/test_performance.py -s -q   # benchmark with timings
uv run ruff check .            # lint
uv run toknt doctor            # inspect local database and source detection
```

Test fixtures are synthetic and sanitized (`tests/fixtures`,
`tests/support`); the suite covers parsing pitfalls for every source format
family, dedupe and rescans, truncated/replaced files, subagents, outcome
correlation, pricing strictness, timezone/bucket boundaries, exports and CLI
behavior on empty databases. Milestones and their completion gates are
recorded in `plan.md`.
