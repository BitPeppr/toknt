"""``python -m toknt`` runs the command line interface.

The TUI also starts its rescan through this entry point (in a child
process), so the two always run the same code.
"""

from __future__ import annotations

from toknt.cli import main

if __name__ == "__main__":
    main()
