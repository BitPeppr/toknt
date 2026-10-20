"""Small shared helpers."""

from __future__ import annotations

import time


def now_ms() -> int:
    """Current UTC time as Unix epoch milliseconds.

    All timestamps in the database are UTC epoch milliseconds; local time
    zones only ever apply at query/presentation time.
    """
    return time.time_ns() // 1_000_000
