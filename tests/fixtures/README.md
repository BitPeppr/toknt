# Test fixtures

All fixtures in this directory are **synthetic and sanitized**. The test suite
must never read the developer's real coding-agent history.

- `canonical_usage_events.jsonl` — canonical usage events covering the
  anthropic and openai token schemes, plus a partial event with missing fields
  and an unknown model.
- `canonical_tool_events.jsonl` — canonical tool events covering a success, a
  failure and a pending call.

The canonical fixtures double as documentation of the canonical record shape:
every key maps directly to a keyword argument of
`toknt.models.events.UsageEvent` / `ToolEvent`.
