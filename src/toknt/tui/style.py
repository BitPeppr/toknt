"""Interactive mode config: key layout, cell model,themes and SGR encoding."""

from __future__ import annotations

import unicodedata


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


#: One muted, mid-tone character for every series: the old sky blue was
#: the brightest entry and made page one's tab and share bar shout while
#: the other pages stayed quiet. Hues still separate the series; the
#: saturation and lightness stay in one narrow band.
SERIES_HEX = (
    0x4E8FA8,  # dusty blue
    0xBC8A3C,  # ochre
    0x3F6F8F,  # steel blue
    0x4F8A72,  # sage
    0xA9748C,  # mauve
    0xB2633F,  # terracotta
    0x8C8C8C,  # grey
)


def heat_ramp(background: str) -> tuple[int, ...]:
    """GitHub-style contributions ramp for the background level.

    Level 0 is the empty-day cell, level 4 the busiest; ordering preserved in
    both light and dark modes.
    """
    if background == "light":
        return (0xEBEDF0, 0x9BE9A8, 0x40C463, 0x30A14E, 0x216E39)
    return (0x1B2A22, 0x0E4429, 0x006D32, 0x26A641, 0x39D353)
