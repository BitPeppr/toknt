# Implementation Plan: Agent Usage Analytics CLI

## 1. Objective

Build a local-first CLI tool that aggregates usage telemetry from multiple AI coding agents and presents detailed historical analytics directly in the terminal.

The tool should go substantially beyond OpenCode's `/stats` equivalent. It should answer questions such as:

- How many tokens do I consume each day, broken down by model, agent and project?
- How much of my input is cached, and how has cache usage changed over time?
- Which models account for the most usage and estimated cost?
- Which tools succeed or fail most frequently? Are their failure rates increasing?
- How frequently do I invoke each tool, model or agent?
- Which sessions are unusually expensive, long-running or tool-call-heavy?
- How much of the collected data is reliable, incomplete, duplicated or unavailable?

The finished tool should operate offline, discover supported agents automatically, persist historical data between runs, and remain responsive as log collections grow.

**Primary target platform:** macOS, with Linux support designed in from the start.

**Implementation language:** Python.

**Working package name:** `toknt`. Keep the project name easy to change.

## 2. Technology stack

Use a small, conventional Python stack:

- **Python 3.12+** for the implementation.
- **Typer** for command-line parsing and subcommands.
- **Rich** for terminal tables, panels, formatting and progress indicators.
- **plotext** for charts rendered directly in the terminal.
- **SQLite (`sqlite3`)** for persistent local analytics and incremental indexing.
- **platformdirs** for locating application data and configuration directories.
- **pytest** for unit, integration and regression testing.
- **ruff** for linting and formatting.

Use `uv` for development and dependency management if the repository does not already prescribe another tool.

Avoid Pandas, a web server, cloud services, heavyweight visualization stacks and unnecessary abstractions in the initial implementation. The data volume and analytical operations involved do not justify a large dependency tree.

Keep the chart-rendering layer independent of the analytics layer, so an interactive full-screen TUI can be added later without rewriting the data model.

## 3. Architecture

Separate the implementation into six layers:

1. **Source discovery:** locate installed agents and their local telemetry stores.
2. **Adapters:** parse each agent's native format and extract supported events.
3. **Canonical data model:** normalize common fields without discarding provider-specific information.
4. **Storage and indexing:** deduplicate events and maintain an incremental SQLite index.
5. **Analytics:** calculate time-series buckets, model breakdowns, tool outcomes and usage metrics.
6. **Presentation:** produce terminal charts, tables, summaries and machine-readable exports.

A suggested structure:

```text
src/toknt/
    __init__.py
    cli.py
    config.py

    discovery/
        __init__.py

    adapters/
        __init__.py
        base.py
        opencode.py
        codex.py
        claude_code.py

    models/
        events.py
        usage.py

    storage/
        database.py
        migrations.py
        ingestion.py

    analytics/
        aggregation.py
        tokens.py
        costs.py
        tool_metrics.py
        quality.py

    presentation/
        charts.py
        tables.py
        dashboard.py

    diagnostics.py

tests/
    fixtures/
    test_adapters/
    test_storage/
    test_analytics/
    test_cli/
```

Treat this as a starting point, not a command to create empty files for every conceivable future feature. Add modules when they have real responsibilities.

### Core architectural rule

Adapters must not implement their own graphs, aggregation logic or CLI output. They should extract source events and translate them into canonical records. All analytics should operate on the canonical model.

## 4. Phase 1: Inspect the sources and define the data model

Before implementing an adapter, inspect the actual local formats and create small, sanitized fixtures from representative sessions. Verify field meanings against current source code, available documentation and sample data.

Initial integrations, in priority order:

**A. OpenCode**

- Discover the local data directory and current SQLite database, respecting relevant environment variables and platform-specific paths.
- Read the database strictly in read-only mode.
- Extract session metadata, assistant-message usage, model/provider identifiers, timestamps and tool-call/result information where available.
- Support legacy file-based session storage when practical.
- Avoid counting the same history twice when both a database and legacy files exist.
- Investigate the native export format as a potential fallback.

Reference: https://www.open-code.ai/en/docs/cli

**B. OpenAI Codex CLI**

- Discover rollout JSONL files under the Codex data directory, including date-based session directories.
- Parse session events and token-count records.
- Distinguish cumulative session counters from counters describing a single request or turn.
- Account for subagent sessions and investigate whether inherited history or repeated counter records would inflate aggregates.
- Extract function/tool calls and results wherever their event types support meaningful classification.

Reference: https://github.com/openai/codex

**C. Claude Code**

- Discover session JSONL files, including the project and subagent layouts.
- Extract model identifiers, timestamps, message/request identifiers, token usage, and tool-call/result events.
- Investigate repeated snapshots, duplicate request identifiers and incomplete usage records.
- Do not assume every `message.usage` field is an authoritative final count. Validate against available usage summaries and current versions of the source format.

Reference: https://github.com/anthropics/claude-code

Keep each adapter independent and version-tolerant. The formats are not equally stable, and the application must remain useful when an unfamiliar event or field appears.

### Canonical records

At minimum, represent the following concepts:

**Usage event**

- Stable internal event ID and source-specific event ID, when available
- Source agent and source file/database identifier
- Event timestamp, stored consistently in UTC
- Session ID and parent-session ID, when known
- Project or working directory, when available
- Provider and model ID
- Input tokens, output tokens, cached input tokens, cache-write tokens and reasoning tokens, where available
- Provider-reported total tokens, if provided
- Reported cost and currency, if provided
- Estimated cost, kept separate from reported cost
- Data-quality status and parsing notes

**Tool event**

- Stable event ID
- Agent, session, timestamp and model context when available
- Tool name and tool category
- Call ID used to associate the call with its result
- Outcome: success, failure, denied/cancelled, pending or unknown
- Duration, if reliably available
- Outcome provenance and confidence

Keep the source-specific event type and enough provenance to trace surprising analytics back to their origin. Do not store prompt bodies, complete transcripts or tool arguments merely to calculate these metrics.

## 5. Phase 2: Build reliable ingestion and persistence

Create a local SQLite database in the appropriate application-data directory.

Use schema migrations from the beginning. Do not treat the database schema as disposable implementation detail.

### Incremental ingestion

Implement an ingestion service that:

- Discovers supported sources and reports which ones were found.
- Performs a full initial import.
- Stores source checkpoints, such as file offsets, source fingerprints and last-processed identifiers.
- Incrementally processes appended data on subsequent runs.
- Detects changed, truncated or replaced files and safely reindexes them.
- Handles partially written JSONL lines and malformed records without aborting the entire scan.
- Supports an explicit full-rescan/reconciliation command.
- Upserts records using stable identities to prevent duplicates.

For OpenCode's SQLite database, use read-only connections and safe read transactions. Account for concurrent writes and WAL mode. Do not modify the source database.

A full rescan should produce the same aggregate totals as a correctly completed initial import.

### Data-quality safeguards

Token accounting must be explicit rather than based on blindly summing whatever fields happen to exist.

- Do not sum cumulative counters once per log entry.
- Where only cumulative counters are available, derive increments carefully and detect resets or discontinuities.
- Deduplicate repeated events using source IDs and validated request/event identity rules.
- Investigate copied history in child sessions before aggregating parent and child totals.
- Do not replace unavailable token counts with invented estimates.
- Preserve unknown fields for diagnostics without interpreting them automatically.
- Keep source-reported and locally estimated costs distinct.

Define consistent semantics for input, cached input, cache-write input, output, reasoning and total tokens. These categories can overlap or differ between providers, so **never calculate a supposedly universal total by adding every available field together**.

Track data quality by adapter, event type and metric. Where a source is ambiguous, expose the limitation instead of silently counting it as exact.

## 6. Phase 3: Implement analytics

Implement the analytics independently of terminal rendering. Each query should accept a time range and relevant filters and return structured results that are easy to test.

### A. Token usage

Support:

- Input, output, cache-read, cache-write and reasoning-token trends.
- Token usage by model, provider, agent and project.
- Daily and weekly totals, with hourly aggregation when timestamp resolution supports it.
- Model comparisons and model-mix percentages.
- Session-level consumption and rankings of the highest-usage sessions.
- Usage over time, including cumulative totals where meaningful.

For mixed-model graphs, use one series per selected model. Support a single metric at a time and avoid placing too many models on a chart by default. Allow filtering to a selected set of models.

### B. Cost analysis

Support:

- Reported cost where available.
- Estimated cost using configurable model/provider pricing data.
- Cost over time and cost breakdowns by model, project and agent.
- Cost per session and average cost per session.
- Estimated cost per output token and other explicitly defined efficiency metrics.

Pricing data must be versioned or timestamped where practical. Unknown model prices should produce an unknown estimate, not a guessed number.

Never present an estimated figure as billing-confirmed cost.

### C. Tool-call analytics

Calculate:

- Total tool calls, grouped by tool, agent and model.
- Successful calls, failed calls, cancellations and calls with unknown outcomes.
- Success rate by tool, across all tools and over time.
- Most frequently failing tools.
- Tool duration distributions when duration is available.
- Tool-call activity per session or model.
- Common failure categories when the source provides enough evidence to classify them.

Define success rate as:

`confirmed successful calls / calls with known final outcomes`

Exclude pending calls and unknown outcomes from the denominator, but report their counts and the fraction of calls with an observable outcome. Otherwise, an adapter that fails to parse errors can deceptively make every tool appear successful.

Do not equate a non-zero shell exit code, an agent-level error and a user-denied action unless the source format supports that interpretation.

### D. Activity and comparisons

Add:

- Sessions and model requests over time.
- Activity by day of week and hour of day.
- Average tokens per request/session.
- Session duration and tool-call intensity, when measurable.
- Comparisons across agents, models and projects.
- Outlier detection for unusually large sessions or usage spikes.

Clearly label approximate or provider-specific metrics. Prioritise defensible measurements over the number of statistics displayed.

## 7. Phase 4: Design the CLI and visualisations

Use a clear command structure. The exact syntax can be refined while implementing, but preserve these capabilities:

```bash
toknt
toknt scan
toknt scan --full
toknt doctor

toknt usage
toknt usage --by model
toknt usage --by agent
toknt usage --metric output --since 30d
toknt usage --bucket week

toknt models
toknt tools
toknt sessions
toknt export --format csv
```

The default command should display a useful overview rather than dumping a wall of logs.

### Terminal charts

Use `plotext` to implement reusable chart functions.

Initial chart types:

1. Multi-series line chart: tokens consumed per day by model.
2. Stacked or grouped chart: input versus output and cache token categories, with non-overlapping definitions.
3. Line chart: tool-call success rate over time.
4. Bar chart: success rate by tool.
5. Bar chart: usage or estimated cost by model.
6. Activity chart: sessions or requests per day.
7. Ranked table: sessions with the highest usage or cost.

Provide sensible defaults for time range and bucket size, plus filters for agent, provider, model and project.

Charts should have readable labels, units, legends and dates. Handle empty data, a single data point, large model counts, terminal resizing and non-interactive output.

Use Rich for tables and summaries. Plain-text mode must remain useful when stdout is piped or the terminal does not support interactive rendering.

### Overview layout

Aim for a compact terminal overview containing:

- Total recorded usage within the selected period
- Input/output/cache breakdown where comparable
- Total reported cost and estimated cost shown separately
- Active sessions and requests
- Model-usage breakdown
- Token-usage trend graph
- Tool success rate and tools with the most failures
- Data-quality warnings

A full-screen keyboard-navigable TUI can follow once the query and chart layers are stable. Do not make the entire project dependent on implementing that more complex interface first.

## 8. Phase 5: Configuration, privacy and diagnostics

Support a simple configuration file for:

- Enabled adapters
- Custom source paths
- Default time range and graph bucket
- Display preferences
- Model aliases, without losing the original provider/model ID
- Pricing overrides
- Optional ignored projects or directories

The database and configuration must remain local. Never upload telemetry, authenticate to providers or make network requests during ordinary scans.

Provide a `doctor` command that explains:

- Which agents were detected
- Which data sources are accessible
- How many sessions and usage records were imported
- How many records were malformed, duplicated, ambiguous or skipped
- Which metrics are unavailable for each adapter
- Whether a source needs a full rescan
- Whether displayed cost numbers are reported or estimated

Errors must identify the affected source and record where possible, without printing prompts, secrets or sensitive tool arguments.

## 9. Phase 6: Testing and validation

Testing accounting correctness is more important than testing chart aesthetics.

Build sanitized fixture datasets for every adapter, including ordinary sessions, tool failures, duplicate records, partial files, unknown models, missing usage fields and subagent sessions.

Required tests:

- Parsing of supported event types and unknown future event types.
- Correct attribution of model and timestamp.
- Correct handling of cumulative versus per-request token usage.
- Deduplication across repeated scans.
- Detection and handling of truncated or replaced source files.
- Parent/subagent history handling.
- Tool-call correlation and success-rate denominators.
- Cost calculations with known, unknown and overridden prices.
- Time-range, timezone and bucket-boundary correctness.
- Stable aggregates after repeated ingestion.
- CLI output for empty databases and unavailable sources.
- CSV/JSON exports matching underlying query results.

Add integration tests using synthetic OpenCode SQLite data and representative JSONL fixtures. Do not require the test suite to read the developer's actual coding-agent history.

Validate aggregates against native usage summaries wherever reliable comparisons are available. Document discrepancies rather than tuning the implementation to match an unreliable value.

Performance requirements:

- Process log files incrementally instead of loading complete histories into memory.
- Keep warm scans efficient by avoiding full reparsing of unchanged sources.
- Use SQLite indexes appropriate to timestamp, model, session, agent and tool queries.
- Benchmark with realistically large synthetic datasets and record actual results.

## 10. Implementation order and completion gates

**Milestone 1: Foundation**

Inspect the repository, establish the package/dependency setup, define the canonical schema, add database migrations and create fixture-based tests.

Completion gate: a working CLI that creates its local database and runs tests successfully.

**Milestone 2: One end-to-end adapter**

Implement OpenCode ingestion first. Build the normalized usage model, persistence, aggregation and first token-by-model line graph.

Completion gate: real local session history can be ingested repeatedly without inflating totals, and a useful graph appears in the terminal.

**Milestone 3: Additional adapters**

Add Codex and Claude Code independently. Create regression fixtures for their source-specific accounting pitfalls and implement data-quality reporting.

Completion gate: all adapters feed the same analytics interface without embedding provider-specific assumptions in the chart code.

**Milestone 4: Analytics and CLI**

Implement the remaining token breakdowns, tool metrics, model comparisons, date filtering and exports.

Completion gate: every advertised metric has documented semantics and automated tests.

**Milestone 5: Polish and hardening**

Improve chart legibility, configuration, diagnostics, performance, error handling, documentation and installation.

Completion gate: a new user can install the tool, discover local sources, generate an overview and understand the limitations of their data without reading the source code.

Do not attempt every milestone in one enormous code-generation pass. Implement each stage, run the tests, inspect the changes and only then continue.

## 11. Definition of done

The first usable release is complete when:

- Three adapters can discover and ingest supported OpenCode, Codex CLI and Claude Code data.
- Historical usage persists locally between runs.
- The user can graph daily token usage by model and inspect tool-call success rates by tool.
- Source-specific deduplication and token-accounting rules are tested.
- Estimated costs, unknown data and ambiguous records are clearly identified.
- `toknt doctor` explains problems with source discovery and data quality.
- Exports work independently of terminal rendering.
- The application never needs to transmit local session data.
- Installation, usage, source support and known limitations are documented.

## 12. Final implementation instructions

Start by inspecting the existing repository and the current source formats. Do not assume the project is empty or overwrite existing conventions.

Write a brief technical design and identify any unresolved source-format questions before implementation. Then implement the milestones in order, running the relevant tests after every stage.

Keep the first release focused on local historical analytics. Live monitoring, API billing integrations, web dashboards, model-price scraping, advanced forecasting and plugin systems are future extensions, not prerequisites.

The central product requirement is simple: **every graph should be useful, every metric should have a clear definition, and every number should have a defensible source.**
