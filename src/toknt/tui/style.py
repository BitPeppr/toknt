"""Interactive mode config: key layout, cell model,themes and SGR encoding."""

from __future__ import annotations

import unicodedata

from toknt.palette import SERIES_PALETTE


def char_width(char: str) -> int:
    """Display width in terminal cells (2 for East-Asian wide, else 1)."""
    return 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1


def rgb_luminance(value: int) -> float:
    """0..255 RRGGBB int → approximate relative luminance."""
    r = (value >> 16) & 255
    g = (value >> 8) & 255
    b = value & 255
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def mix(a: int, b: int, t: float) -> int:
    """Blend two RRGGBB colors; ``t=0`` → ``a``, ``t=1`` → ``b``."""
    ar, ag, ab = a >> 16, (a >> 8) & 255, a & 255
    br, bg, bb = b >> 16, (b >> 8) & 255, b & 255
    r = round(ar + (br - ar) * t)
    g = round(ag + (bg - ag) * t)
    c = round(ab + (bb - ab) * t)
    return (r << 16) | (g << 8) | c


#: Text charts, legend swatches and raster charts all draw from the same
#: palette. Keeping this as an alias (not a copy) prevents the two render
#: paths from drifting apart and pairing a muted legend with a bright line.
SERIES_HEX = SERIES_PALETTE


def heat_ramp(background: str) -> tuple[int, ...]:
    """GitHub-style contributions ramp for the background level.

    Level 0 is the empty-day cell, level 4 the busiest; ordering preserved in
    both light and dark modes.
    """
    if background == "light":
        return (0xEBEDF0, 0x9BE9A8, 0x40C463, 0x30A14E, 0x216E39)
    return (0x1B2A22, 0x0E4429, 0x006D32, 0x26A641, 0x39D353)
