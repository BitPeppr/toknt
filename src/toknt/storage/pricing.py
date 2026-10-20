"""Versioned model pricing rows.

Prices live in the analytics database (``pricing`` table) and are managed via
``toknt pricing``. They are consulted at query time by
:mod:`toknt.analytics.costs`; event rows never bake in estimated costs.

Rows are keyed by ``(model, provider, effective_from_ms)``; the analytics
layer uses the most recent row per model, so adding a corrected price does not
require touching history.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PriceRow:
    model: str
    provider: str
    currency: str
    input_per_mtok: float | None
    output_per_mtok: float | None
    cache_read_per_mtok: float | None
    cache_write_per_mtok: float | None
    effective_from_ms: int
    note: str | None


def add_price(
    conn: sqlite3.Connection,
    *,
    model: str,
    provider: str = "",
    currency: str = "USD",
    input_per_mtok: float | None = None,
    output_per_mtok: float | None = None,
    cache_read_per_mtok: float | None = None,
    cache_write_per_mtok: float | None = None,
    effective_from_ms: int = 0,
    note: str | None = None,
) -> None:
    """Insert or replace a price row."""
    conn.execute(
        """
        INSERT INTO pricing (
            model, provider, currency, input_per_mtok, output_per_mtok,
            cache_read_per_mtok, cache_write_per_mtok, effective_from_ms, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(model, provider, effective_from_ms) DO UPDATE SET
            currency = excluded.currency,
            input_per_mtok = excluded.input_per_mtok,
            output_per_mtok = excluded.output_per_mtok,
            cache_read_per_mtok = excluded.cache_read_per_mtok,
            cache_write_per_mtok = excluded.cache_write_per_mtok,
            note = excluded.note
        """,
        (
            model,
            provider,
            currency,
            input_per_mtok,
            output_per_mtok,
            cache_read_per_mtok,
            cache_write_per_mtok,
            effective_from_ms,
            note,
        ),
    )
    conn.commit()


def list_prices(conn: sqlite3.Connection) -> list[PriceRow]:
    rows = conn.execute(
        """
        SELECT model, provider, currency, input_per_mtok, output_per_mtok,
               cache_read_per_mtok, cache_write_per_mtok, effective_from_ms, note
        FROM pricing ORDER BY model, provider, effective_from_ms
        """
    ).fetchall()
    return [PriceRow(**dict(row)) for row in rows]


def remove_price(
    conn: sqlite3.Connection,
    *,
    model: str,
    provider: str | None = None,
    effective_from_ms: int | None = None,
) -> int:
    """Delete price rows for a model; returns the number of deleted rows."""
    clauses = ["model = ?"]
    params: list[object] = [model]
    if provider is not None:
        clauses.append("provider = ?")
        params.append(provider)
    if effective_from_ms is not None:
        clauses.append("effective_from_ms = ?")
        params.append(effective_from_ms)
    cursor = conn.execute(f"DELETE FROM pricing WHERE {' AND '.join(clauses)}", params)
    conn.commit()
    return cursor.rowcount if cursor.rowcount > 0 else 0
