"""Terminal-graphics protocol tests: payload structure and detection.

These pin the details that decide whether a chart actually appears:

* the kitty protocol requires the image id on **every** transmission chunk
  (a chunk without it is orphaned and the terminal renders nothing),
* iTerm2's ``size=`` is the payload's byte count, not the cell count,
* the PNG we emit is a real PNG,
* backend detection knows the common terminals and refuses to use images
  inside a multiplexer (tmux/screen swallow them).
"""

from __future__ import annotations

import re
import struct
import zlib

import pytest

from toknt.presentation import graphics


def sample_png(width: int = 400, height: int = 120) -> bytes:
    png, _rgba = graphics.chart_png(
        width=width,
        height=height,
        kind="line",
        series={"a": [1, 4, 2, 7, 3], "b": [3, 1, 5, 2, 6]},
    )
    return png


class TestKittyPayload:
    def test_every_chunk_carries_the_image_id(self) -> None:
        png = sample_png()
        payload = graphics.kitty_apc(png, columns=40, rows=12)
        chunks = [
            chunk
            for chunk in payload.split(b"\x1b\\")
            if b"m=" in chunk and b"a=d" not in chunk
        ]
        assert len(chunks) > 1, "the sample must span several chunks"
        for index, chunk in enumerate(chunks):
            assert b"i=1" in chunk, f"chunk {index} lost its image id: {chunk[:60]!r}"
            assert b"C=1" in chunk
        # first chunk opens the transmission, the last closes it
        assert b"a=T" in chunks[0]
        assert b"m=0" in chunks[-1]
        assert b"m=1" in chunks[0]

    def test_placement_carries_cell_size(self) -> None:
        payload = graphics.kitty_apc(sample_png(), columns=37, rows=9)
        assert b"c=37" in payload and b"r=9" in payload

    def test_image_id_is_per_slot(self) -> None:
        payload = graphics.kitty_apc(sample_png(), columns=10, rows=5, image_id=2)
        chunks = [c for c in payload.split(b"\x1b\\") if b"m=" in c and b"a=d" not in c]
        assert all(b"i=2" in chunk for chunk in chunks)

    def test_delete_targets_only_its_own_image(self) -> None:
        """``a=d`` without ``d=i`` means *delete every image* in kitty.

        The TUI places two images per frame (line chart + donut); with the
        old delete, the donut's leading delete erased the chart that had
        just been placed — the line graph never survived to be shown.
        """
        payload_1 = graphics.kitty_apc(sample_png(), columns=10, rows=5, image_id=1)
        payload_2 = graphics.kitty_apc(sample_png(), columns=10, rows=5, image_id=2)

        # replay the stream against a model of kitty's image table
        alive: set[int] = set()
        for payload in (payload_1, payload_2):
            for command in payload.split(b"\x1b\\"):
                if not command.startswith(b"\x1b_G"):
                    continue
                body = command[2:].decode("ascii")
                if not body.startswith("a=d"):
                    alive.add(1 if "i=1" in body else 2)
                    continue
                assert "d=i" in body, f"delete must be scoped: {body!r}"
                assert "q=1" in body, "delete must be quiet (no OK reply)"
                image_id = int(re.search(r"i=(\d+)", body).group(1))
                alive.discard(image_id)  # d=i deletes only this image

        assert alive == {1, 2}, f"both images must survive: {alive=}"


class TestIterm2Payload:
    def test_size_is_the_payload_byte_count(self) -> None:
        png = sample_png()
        payload = graphics.iterm2_inline(png, columns=40, rows=12)
        start = payload.find(b"size=") + len(b"size=")
        end = payload.find(b";", start)
        assert int(payload[start:end]) == len(png)

    def test_uses_the_inline_file_control_sequence(self) -> None:
        payload = graphics.iterm2_inline(sample_png(), columns=10, rows=5)
        assert payload.startswith(b"\x1b]1337;File=inline=1;")
        assert payload.endswith(b"\x07")


class TestPng:
    def test_emitted_png_is_a_valid_rgba_image(self) -> None:
        png = sample_png(120, 60)
        assert png.startswith(b"\x89PNG\r\n\x1a\n")
        # IHDR: length, type, width, height, bit depth, colour type, ...
        length, kind = struct.unpack(">I4s", png[8:16])
        assert kind == b"IHDR" and length == 13
        width, height, depth, colour = struct.unpack(">IIBB", png[16:26])
        assert (width, height) == (120, 60)
        assert depth == 8 and colour == 6  # 8-bit RGBA
        # every chunk's CRC must check out
        offset = 8
        seen = []
        while offset < len(png):
            (size,) = struct.unpack(">I", png[offset : offset + 4])
            chunk_type = png[offset + 4 : offset + 8]
            body = png[offset + 8 : offset + 8 + size]
            (crc,) = struct.unpack(">I", png[offset + 8 + size : offset + 12 + size])
            assert zlib.crc32(chunk_type + body) & 0xFFFFFFFF == crc
            seen.append(chunk_type)
            offset += 12 + size
        assert seen == [b"IHDR", b"IDAT", b"IEND"]


class TestDetection:
    @pytest.mark.parametrize(
        ("env", "expected"),
        [
            ({"KITTY_WINDOW_ID": "1"}, "kitty"),
            ({"TERM": "xterm-kitty"}, "kitty"),
            ({"TERM_PROGRAM": "ghostty"}, "kitty"),
            ({"GHOSTTY_RESOURCES_DIR": "/x"}, "kitty"),
            ({"TERM_PROGRAM": "WezTerm"}, "kitty"),
            ({"TERM_PROGRAM": "iTerm.app"}, "iterm2"),
            ({"ITERM_SESSION_ID": "w0t0p"}, "iterm2"),
            ({"TERM": "xterm-256color"}, "sixel"),
            ({"TERM": "foot"}, "sixel"),
            ({"TERM": "xterm-kitty", "TMUX": "/tmp/tmux"}, "off"),
            ({"TERM": "xterm-256color", "STY": "1"}, "off"),
            ({"TERM": "xterm-256color"}, "sixel"),
            ({"TERM": "dumb"}, "off"),
            ({"TERM_PROGRAM": "Apple_Terminal"}, "off"),
        ],
    )
    def test_matrix(self, env: dict[str, str], expected: str) -> None:
        backend, reason = graphics.describe_backend(env, stdout_is_tty=True)
        assert backend == expected, reason
        assert reason, "every detection must explain itself"

    def test_non_tty_is_off(self) -> None:
        backend, reason = graphics.describe_backend(
            {"KITTY_WINDOW_ID": "1"}, stdout_is_tty=False
        )
        assert backend == "off"
        assert "not a terminal" in reason

    def test_override_is_honoured(self) -> None:
        backend, reason = graphics.describe_backend(
            {}, stdout_is_tty=True, override="kitty"
        )
        assert backend == "kitty"
        assert "forced" in reason

    def test_override_off(self) -> None:
        backend, _reason = graphics.describe_backend(
            {"KITTY_WINDOW_ID": "1"}, stdout_is_tty=True, override="off"
        )
        assert backend == "off"
