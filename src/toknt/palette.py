"""The shared series palette.

One palette for every chart path: the raster charts (kitty/iterm2/sixel PNG
output) and the text charts (Braille lines, share strips, legend swatches).
Keeping the tuple in its own tiny module means both sides import the same
object without dragging the whole presentation package (and plotext) into a
leaf import.

The hues separate the series while saturation and lightness stay in one
narrow band, so a page never shouts: the values stay legible on dark and
light terminal backgrounds alike.
"""

from __future__ import annotations

__all__ = ["SERIES_PALETTE"]

#: Muted, mid-luminance series colours (RRGGBB).
SERIES_PALETTE: tuple[int, ...] = (
    0x4E8FA8,  # dusty blue
    0xBC8A3C,  # ochre
    0x3F6F8F,  # steel blue
    0x4F8A72,  # sage
    0xA9748C,  # mauve
    0xB2633F,  # terracotta
    0x8C8C8C,  # grey
)
