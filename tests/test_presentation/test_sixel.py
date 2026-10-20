"""Sixel round-trip test: decode our own payload and compare pixels.

Sixel is positional: each colour's data continues from where the previous
colour stopped, so a missing carriage return silently shifts every colour
off-grid. Decoding the emitted stream back to a bitmap catches that class
of bug without needing a sixel terminal.
"""

from __future__ import annotations

import re

import pytest

from toknt.presentation import graphics


def decode_sixel(payload: bytes) -> tuple[int, int, dict[tuple[int, int], tuple[int, int, int]]]:
    """Decode a sixel stream into (width, height, {pixel: rgb})."""
    text = payload.decode("latin-1")
    assert text.startswith("\x1bPq")
    body = text[3:]
    assert body.endswith("\x1b\\")
    body = body[:-2]

    # raster attributes: "Pan;Pad;width;height
    raster = re.match(r'"(\d+);(\d+);(\d+);(\d+)', body)
    assert raster, "missing raster attributes"
    width = int(raster.group(3))
    height = int(raster.group(4))
    body = body[raster.end() :]

    # palette: #n;2;r;g;b (until the first plain colour introducer)
    palette: dict[int, tuple[int, int, int]] = {}
    while True:
        match = re.match(r"#(\d+);2;(\d+);(\d+);(\d+)", body)
        if not match:
            break
        palette[int(match.group(1))] = (
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(4)),
        )
        body = body[match.end() :]

    pixels: dict[tuple[int, int], tuple[int, int, int]] = {}
    color = 0
    x = 0
    band = 0
    index = 0
    while index < len(body):
        ch = body[index]
        if ch == "#":
            match = re.match(r"#(\d+)", body[index:])
            assert match, "malformed colour introducer"
            color = int(match.group(1))
            index += match.end()
            continue
        if ch == "!":
            match = re.match(r"!(\d+)(.)", body[index:])
            assert match, "malformed repeat"
            count = int(match.group(1))
            char = match.group(2)
            if char != "?":
                for step in range(count):
                    for sub in range(6):
                        if char_isset(char, sub):
                            pixels[(x + step, band * 6 + sub)] = palette.get(
                                color, (255, 255, 255)
                            )
            x += count
            index += match.end()
            continue
        if ch == "?":
            x += 1
            index += 1
            continue
        if ch == "$":
            x = 0
            index += 1
            continue
        if ch == "-":
            band += 1
            x = 0
            index += 1
            continue
        # a plain sixel char: sets six vertical pixels
        for sub in range(6):
            if char_isset(ch, sub):
                pixels[(x, band * 6 + sub)] = palette.get(color, (255, 255, 255))
        x += 1
        index += 1
    return width, height, pixels


def char_isset(char: str, sub: int) -> bool:
    return bool(ord(char) - 63 & (1 << sub))


def test_sixel_round_trip_places_pixels_on_the_original_grid() -> None:
    width, height = 120, 60
    rgba = bytearray(width * height * 4)
    for y in range(height):
        for x in range(width):
            offset = (y * width + x) * 4
            if 20 <= x < 60 and 10 <= y < 30:
                rgba[offset : offset + 4] = bytes((220, 40, 60, 255))
            elif 70 <= x < 100 and 30 <= y < 50:
                rgba[offset : offset + 4] = bytes((30, 120, 220, 255))
            else:
                rgba[offset : offset + 4] = bytes((0, 0, 0, 0))

    payload = graphics.sixel_bytes(width, height, bytes(rgba))
    dec_width, dec_height, pixels = decode_sixel(payload)
    assert (dec_width, dec_height) == (width, height)

    # the red block must land exactly where it was painted
    assert pixels.get((30, 15)) is not None
    assert pixels.get((30, 15)) != pixels.get((80, 40))
    # and both blocks must be present, not shifted off the grid
    reds = [p for p, c in pixels.items() if c[0] > 150 and c[1] < 120]
    blues = [p for p, c in pixels.items() if c[2] > 150 and c[0] < 120]
    assert reds and blues
    assert all(20 <= x < 60 and 10 <= y < 30 for x, y in reds)
    assert all(70 <= x < 100 and 30 <= y < 50 for x, y in blues)


def test_sixel_palette_defines_every_color_it_uses() -> None:
    rgba = bytes((200, 100, 50, 255)) * (40 * 20)
    payload = graphics.sixel_bytes(40, 20, rgba).decode("latin-1")
    body = payload[3:]  # strip the DCS introducer
    raster = re.match(r'"\d+;\d+;\d+;\d+', body)
    assert raster
    body = body[raster.end() :]

    palette_part = body
    defined = set()
    while True:
        match = re.match(r"#(\d+);2;(\d+);(\d+);(\d+)", palette_part)
        if not match:
            break
        defined.add(int(match.group(1)))
        palette_part = palette_part[match.end() :]
    assert defined, "the payload must define its palette"
    used = {int(m) for m in re.findall(r"#(\d+)(?![;\d])", palette_part)}
    assert used, "sixel body must select colours"
    assert used <= defined, f"undefined colour registers used: {sorted(used - defined)}"


@pytest.mark.parametrize("width,height", [(40, 20), (61, 13)])
def test_sixel_height_is_padded_to_a_sixel_band(width: int, height: int) -> None:
    payload = graphics.sixel_bytes(width, height, bytes((10, 20, 30, 255)) * width * height)
    text = payload.decode("latin-1")
    raster = re.search(r'"2;1;(\d+);(\d+)', text)
    assert raster
    assert int(raster.group(2)) % 6 == 0
    assert int(raster.group(2)) >= height
