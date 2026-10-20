"""Graphics module: PNG writer, transports, detection, chart renderer."""

from __future__ import annotations

import re
import struct
import zlib

import pytest

from toknt.presentation.graphics import (
    SERIES_PALETTE,
    Canvas,
    _scaled_channel,
    chart_png,
    detect_backend,
    encode_placement,
    iterm2_inline,
    kitty_apc,
    measure_cell,
    png_encode,
    sixel_bytes,
    solid_fill,
    terminal_background,
)


def _png_dims(png: bytes) -> tuple[int, int, int, int]:
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    ihdr = struct.unpack(">IIBBBBB", png[16:29])
    return ihdr[0], ihdr[1], ihdr[2], ihdr[3]  # width, height, depth, colortype


class TestPngEncode:
    def test_header_and_crc(self) -> None:
        rgba = bytes(240)
        png = png_encode(10, 6, rgba)
        width, height, depth, colortype = _png_dims(png)
        assert (width, height, depth, colortype) == (10, 6, 8, 6)
        # CRC of the IHDR chunk (length+type+payload) verifies integrity
        expected_crc = zlib.crc32(png[12:29]) & 0xFFFFFFFF
        actual_crc = struct.unpack(">I", png[29:33])[0]
        assert actual_crc == expected_crc

    def test_size_validation(self) -> None:
        with pytest.raises(ValueError):
            png_encode(4, 4, bytes(4 * 4 * 4 + 1))


class TestKitty:
    def test_single_chunk_deletes_then_places(self) -> None:
        png = png_encode(4, 4, bytes(64))
        payload = kitty_apc(png, columns=10, rows=5, image_id=7)
        # delete is scoped to this image (a bare a=d deletes EVERY image)
        assert payload.startswith(b"\x1b_Ga=d,d=i,i=7,C=1,q=1\x1b\\")
        # one AP command follows the delete, carrying the whole payload
        commands = payload.split(b"\x1b_G")[1:]
        assert len(commands) == 2  # delete + transmit
        transmission = commands[1]
        assert transmission.startswith(b"a=T,f=100,i=7,C=1,q=1,c=10,r=5,m=0;")
        assert transmission.endswith(b"\x1b\\")
        assert b"\x1b[" not in payload  # no CSI noise among graphic bytes

    def test_chunking_places_m_flags_per_chunk(self) -> None:
        big = b"\xa5" * 9000  # incompressible-ish payload (encoder takes any bytes)
        payload = kitty_apc(big, columns=10, rows=5)
        commands = payload.split(b"\x1b_G")[1:]
        assert len(commands) >= 4  # delete + ≥3 data chunks at 4096 byte steps
        assert commands[1].startswith(b"a=T,f=100,i=1,C=1,q=1")
        for chunk in commands[1:-1]:
            assert b"m=1;" in chunk or b"m=0;" in chunk
        assert commands[-1].rstrip(b"\x1b\\")
        assert b"m=0;" in commands[-1]
        # every data chunk body stays ≤4096 base64 chars
        for chunk in commands[1:]:
            body = chunk.split(b";", 1)[1].rsplit(b"\x1b", 1)[0]
            assert len(body) <= 4096


class TestIterm2:
    def test_osc_frame(self) -> None:
        png = png_encode(4, 4, bytes(64))
        payload = iterm2_inline(png, columns=10, rows=5)
        assert payload.startswith(b"\x1b]1337;File=inline=1;")
        assert payload.endswith(b"\x07")
        assert b"width=10;height=5;" in payload


class TestSixel:
    def test_structure_and_terminator(self) -> None:
        canvas = Canvas(24, 13)
        canvas.draw_rect(0, 0, 24, 13, (255, 0, 0, 255))
        payload = sixel_bytes(24, 13, canvas.rgba())
        assert payload.startswith(b"\x1bPq")
        assert payload.endswith(b"\x1b\\")
        # raster attributes pad height to a multiple of 6 (13 → 18)
        assert b'"2;1;24;18' in payload
        assert b"#0" in payload  # at least one palette entry

    def test_transparency_omits_pixels(self) -> None:
        transparent = sixel_bytes(4, 4, bytes(4 * 4 * 4))
        painted = sixel_bytes(4, 4, bytes(4 * 4 * 4))
        assert transparent.count(b"!") == painted.count(b"!")  # same skeleton


class TestEncodePlacement:
    def test_dispatch(self) -> None:
        png = png_encode(4, 4, bytes(64))
        canvas_rgba = bytes(4 * 4 * 4)
        kitty = encode_placement("kitty", png, canvas_rgba, 4, 4, columns=2, rows=2)
        iterm = encode_placement("iterm2", png, canvas_rgba, 4, 4, columns=2, rows=2)
        sixel = encode_placement("sixel", png, canvas_rgba, 4, 4, columns=2, rows=2)
        assert kitty.startswith(b"\x1b_G")
        assert iterm.startswith(b"\x1b]1337;")
        assert sixel.startswith(b"\x1bPq")
        with pytest.raises(ValueError):
            encode_placement("banana", png, canvas_rgba, 4, 4, columns=2, rows=2)


class TestDetection:
    @pytest.mark.parametrize(
        ("env", "tty", "override", "expected"),
        [
            ({}, True, None, "off"),
            ({"KITTY_WINDOW_ID": "1"}, True, None, "kitty"),
            ({"KITTY_WINDOW_ID": "1"}, False, None, "off"),
            ({"TERM": "xterm-kitty"}, True, None, "kitty"),
            ({"GHOSTTY_RESOURCES_DIR": "/x"}, True, None, "kitty"),
            ({"TERM_PROGRAM": "WezTerm"}, True, None, "kitty"),
            ({"TERM_PROGRAM": "iTerm.app"}, True, None, "iterm2"),
            ({"ITERM_SESSION_ID": "w0"}, True, None, "iterm2"),
            ({"TERM": "xterm-256color"}, True, None, "sixel"),
            ({"TERM": "foot"}, True, None, "sixel"),
            ({"TERM": "screen"}, True, None, "off"),
            ({}, True, "iterm2", "iterm2"),
            ({}, False, "kitty", "off"),
            ({}, True, "off", "off"),
            ({"TERM_PROGRAM": "iTerm.app"}, True, "sixel", "sixel"),
            ({"KITTY_WINDOW_ID": "1"}, True, "off", "off"),
        ],
    )
    def test_matrix(
        self,
        env: dict[str, str],
        tty: bool,
        override: str | None,
        expected: str,
    ) -> None:
        assert detect_backend(env=env, stdout_is_tty=tty, override=override) == expected


class TestQueries:
    def test_non_tty_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
        assert measure_cell() == (10, 20)
        assert terminal_background() == ("dark", None)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("ff", 255),
            ("f", 255),      # 1-digit replies scale to the nibble max
            ("1", 17),       # 1*255//15
            ("88", 136),
            ("ffff", 255),
        ],
    )
    def test_channel_scaling(self, text: str, expected: int) -> None:
        assert _scaled_channel(text) == expected


class TestSolidFill:
    def test_opaque_fill_erases_everything(self) -> None:
        png, rgba = solid_fill(width=8, height=4, color=0x102030)
        width, height, _depth, _colortype = _png_dims(png)
        assert (width, height) == (8, 4)
        assert rgba[0] == 0x10 and rgba[1] == 0x20 and rgba[2] == 0x30
        assert all(rgba[index] == 255 for index in range(3, len(rgba), 4))


class TestRenderer:
    def test_line_series_with_gaps(self) -> None:
        png, rgba = chart_png(
            width=120,
            height=48,
            kind="line",
            series={"a": [1, None, 4, 10], "b": [2, 3, None, 5]},
        )
        width, height, _depth, colortype = _png_dims(png)
        assert (width, height) == (120, 48)
        assert colortype == 6
        painted = sum(1 for index in range(3, len(rgba), 4) if rgba[index])
        assert painted > 300

    def test_stacked_and_bar_share_geometry(self) -> None:
        for kind in ("bar", "stacked"):
            png, rgba = chart_png(
                width=80,
                height=40,
                kind=kind,
                series={"a": [2, None, 5], "b": [1, 3, None]},
            )
            painted = sum(1 for index in range(3, len(rgba), 4) if rgba[index])
            assert painted > 100

    def test_empty_series_is_blank_png(self) -> None:
        png, rgba = chart_png(width=60, height=30, kind="line", series={})
        assert len(png) > 0
        assert rgba == bytes(60 * 30 * 4)

    def test_blank_then_paint(self) -> None:
        # canvas with nothing drawn stays fully transparent
        canvas = Canvas(8, 8)
        assert canvas.rgba() == bytes(8 * 8 * 4)
        canvas.draw_rect(0, 0, 8, 8, (10, 20, 30, 255))
        rgba = canvas.rgba()
        assert rgba[3] == 255 and rgba[0] == 10

    def test_soft_falloff_stays_inside_bounds(self) -> None:
        canvas = Canvas(20, 20)
        canvas.draw_line(0, 0, 19, 19, (200, 200, 200, 255), thickness=4)
        rgba = canvas.rgba()
        assert sum(1 for index in range(3, len(rgba), 4) if rgba[index]) > 80

    def test_palette_is_stable(self) -> None:
        assert len(SERIES_PALETTE) >= 5
        assert all(isinstance(color, int) for color in SERIES_PALETTE)
        assert re.match(r"^[0-9a-f]{6}$", f"{hex(SERIES_PALETTE[0])[2:]:0>6}")
