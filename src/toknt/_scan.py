"""The TUI's rescan worker.

The explorer runs its rescan in a child process: parsing every changed
source file is seconds of work, and the interface must not stall while it
happens. This module is that child — ``python -m toknt._scan`` — and it is
private on purpose: the TUI is its only caller.

The configuration travels in the ``TOKNT_SCAN_CONFIG`` environment
variable as JSON, because the child has to scan exactly the sources the
TUI would have scanned (and nothing else). The TUI also puts its own
package directory on the child's import path, so the child always runs the
same code as the parent.
"""

from __future__ import annotations

import json
import os
import sys

from toknt.config import AppConfig
from toknt.models.events import Harness
from toknt.service import run_scan
from toknt.storage.database import open_database

#: Environment variable carrying the scan configuration as JSON.
CONFIG_ENV = "TOKNT_SCAN_CONFIG"


def config_from_payload(payload: dict) -> AppConfig:
    """Rebuild the configuration the TUI handed over."""
    return AppConfig(
        enabled_harnesses=tuple(
            Harness(name) for name in payload.get("enabled_harnesses", [])
        ),
        source_overrides=dict(payload.get("source_overrides", {})),
        default_since=payload.get("default_since", "30d"),
        default_bucket=payload.get("default_bucket", "day"),
        ignore_projects=tuple(payload.get("ignore_projects", [])),
    )


def main(argv: list[str] | None = None) -> int:
    raw = os.environ.get(CONFIG_ENV, "{}")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {}
    config = config_from_payload(payload)
    conn = open_database()
    try:
        run_scan(conn, full=False, config=config)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
