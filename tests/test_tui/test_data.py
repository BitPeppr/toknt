"""The TUI data layer against the canonical fixture events."""

from __future__ import annotations

import sqlite3

import pytest

from toknt.tui.data import (
    AGENT_NAMES,
    DEFAULT_WINDOW,
    WINDOWS,
    AgentToggles,
    Analysis,
)

pytestmark = pytest.mark.usefixtures("data_dir")


@pytest.fixture
def analysis(seeded_conn: sqlite3.Connection) -> Analysis:
    from toknt import config

    return Analysis(seeded_conn, config.load_config())


def test_windows_labels() -> None:
    assert [window.label for window in WINDOWS] == [
        "3h",
        "5h",
        "1d",
        "3d",
        "7d",
        "30d",
        "90d",
        "1y",
        "all",
    ]
    assert WINDOWS[-1].days is None


def test_default_window_is_resolved_by_label() -> None:
    assert WINDOWS[DEFAULT_WINDOW].label == "3d"


def test_agent_toggles_toggle() -> None:
    base = AgentToggles()
    assert len(base.active) == len(AGENT_NAMES)
    removed = base.toggled("opencode")
    assert "opencode" not in removed.active
    back = removed.toggled("opencode")
    assert back == base
    assert "claude_code" in base.toggled("claude_code").active is not None or (
        "claude_code" not in base.toggled("claude_code").active
    )


def test_daily_days_are_sorted_and_agents_split(analysis: Analysis) -> None:
    daily = analysis.daily(0, AgentToggles())
    assert daily.days == tuple(sorted(daily.days))
    assert daily.total
    for day, entry in daily.per_agent.items():
        assert day in daily.total
        assert sum(entry.values()) == daily.total[day]


def test_agent_filter_shrinks_totals(analysis: Analysis) -> None:
    everyone = analysis.daily(0, AgentToggles())
    per_agent = analysis.daily(0, AgentToggles())
    subset = analysis.daily(0, AgentToggles.toggled(AgentToggles(), "codex"))
    assert set(subset.per_agent.get(subset.days[0], {})) <= {"claude_code", "opencode"}
    # every filtered total is ≤ the everyone-total for the same day
    for day, value in subset.total.items():
        assert value <= everyone.total[day]
    del per_agent


def test_model_series_and_breakdown_agree(analysis: Analysis) -> None:
    series = analysis.model_series(0, AgentToggles(), "total")
    rows = analysis.model_rows(0, AgentToggles())
    assert series.series
    top_keys = list(series.series)
    for entry in rows:
        assert entry.summary.total is None or entry.summary.total >= 0
    assert top_keys or not rows


def test_tool_rows_have_outcome_counts(analysis: Analysis) -> None:
    outcomes = analysis.tool_rows(0, AgentToggles())
    for outcome in outcomes:
        assert outcome.calls >= outcome.success + outcome.failure


def test_session_rows_limit(analysis: Analysis) -> None:
    ranked = analysis.session_rows(0, AgentToggles(), limit=5)
    assert len(ranked) <= 5
    for entry in ranked:
        assert entry.summary.events >= 1
        assert entry.first_ms <= entry.last_ms
