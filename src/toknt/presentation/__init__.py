"""Terminal presentation: Rich tables, plotext charts and the dashboard.

Nothing in this package reads the database or reinterprets accounting
semantics: it renders analytics dataclasses as strings/tables. This is also
the only place plotext and Rich are imported.
"""

from toknt.presentation import charts, dashboard, tables

__all__ = ["charts", "dashboard", "tables"]
