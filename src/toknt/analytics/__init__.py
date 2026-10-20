"""Analytics: pure queries over canonical events.

The only module coupling in this package runs one way —
``costs``/``quality``/``tool_metrics`` may use ``aggregation`` primitives, but
``aggregation`` depends on nothing above storage. Presentation layers are
never imported here.
"""

from toknt.analytics.aggregation import (
    BUCKETS,
    METRICS,
    BreakdownRow,
    OverallSummary,
    Series,
    SessionSummary,
    TokenSummary,
    UsageFilter,
    UsageRow,
    activity_series,
    bucket_start_ms,
    bucket_starts,
    cumulative,
    default_timezone,
    fetch_usage_rows,
    metric_label,
    metric_value,
    model_cache_rates,
    model_inventory,
    overall_summary,
    parse_since,
    session_ranking,
    summarize,
    usage_breakdown,
    usage_series,
)
from toknt.analytics.costs import (
    CostEstimate,
    Price,
    cost_by_dimension,
    cost_summary,
    estimate_cost,
    load_price_map,
)
from toknt.analytics.quality import HarnessQuality, harness_quality
from toknt.analytics.tool_metrics import (
    DurationStats,
    ToolOutcomeRow,
    tool_durations,
    tool_outcomes,
    tool_success_series,
)

__all__ = [
    "BUCKETS",
    "METRICS",
    "AgentQuality",
    "BreakdownRow",
    "CostEstimate",
    "DurationStats",
    "OverallSummary",
    "Price",
    "Series",
    "SessionSummary",
    "TokenSummary",
    "ToolOutcomeRow",
    "UsageFilter",
    "UsageRow",
    "activity_series",
    "agent_quality",  # historical name
    "HarnessQuality",
    "harness_quality",
    "bucket_start_ms",
    "bucket_starts",
    "cost_by_dimension",
    "cost_summary",
    "cumulative",
    "default_timezone",
    "estimate_cost",
    "fetch_usage_rows",
    "load_price_map",
    "metric_label",
    "metric_value",
    "ModelCacheRate",
    "model_cache_rates",
    "model_inventory",
    "overall_summary",
    "parse_since",
    "session_ranking",
    "summarize",
    "tool_durations",
    "tool_outcomes",
    "tool_success_series",
    "usage_breakdown",
    "usage_series",
]
