"""Key decoding matrix for the TUI."""

from __future__ import annotations

import pytest

from toknt.tui.keys import KeyEvent, decode, read_key


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"q", KeyEvent("char", "q")),
        (b"5", KeyEvent("char", "5")),
        (b" ", KeyEvent("char", " ")),
        (b"\r", KeyEvent("enter")),
        (b"\n", KeyEvent("enter")),
        (b"\x03", KeyEvent("ctrl+c")),
        (b"\x04", KeyEvent("ctrl+d")),
        (b"\x09", KeyEvent("tab")),
        (b"\x1b", KeyEvent("esc")),  # no continuation source → bare ESC
        (b"\x1b[A", KeyEvent("up")),
        (b"\x1b[B", KeyEvent("down")),
        (b"\x1b[C", KeyEvent("right")),
        (b"\x1b[D", KeyEvent("left")),
        (b"\x1bOA", KeyEvent("up")),
        (b"\x1bOD", KeyEvent("left")),
        (b"\x1b[5~", KeyEvent("pgup")),
        (b"\x1b[6~", KeyEvent("pgdown")),
        (b"\x1b[3~", KeyEvent("delete")),
        (b"\x1b[H", KeyEvent("home")),
        (b"\x1b[F", KeyEvent("end")),
        (b"\x1b[9;9x", KeyEvent("none")),  # unknown sequence: swallowed
        (b"\x00", KeyEvent("none")),
        (b"\xff\xff", KeyEvent("none")),
    ],
)
def test_decode_matrix(data: bytes, expected: KeyEvent) -> None:
    assert decode(data) == expected


def test_is_char_ignores_case() -> None:
    assert decode(b"T").is_char("t")
    assert not decode(b"t").is_char("x")


def test_escape_continuation_waits_for_fd() -> None:
    # decode([], fd=None) is 'none'; a lone ESC with fd None is 'esc'
    assert decode(b"\x1b").name == "esc"


def test_read_key_none_on_quiet_fd() -> None:
    import os

    rd, write_end = os.pipe()
    os.close(write_end)  # EOF: read() returns b"" and the decoder sees nothing
    try:
        assert read_key(rd, poll_wait=0.01) is None
    finally:
        os.close(rd)
