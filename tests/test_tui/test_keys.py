"""Key decoding matrix for the TUI."""

from __future__ import annotations

import os
import time

import pytest

from toknt.tui.keys import _ESC_CONT_WAIT, KeyEvent, decode, read_key


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


def _pipe_with(payload: bytes) -> int:
    read_fd, write_fd = os.pipe()
    os.write(write_fd, payload)
    os.close(write_fd)
    return read_fd


def test_split_arrow_sequence_is_completed_not_leaked() -> None:
    """ESC + '[' in one read, 'A' in the next: still an up-arrow.

    The final byte must never surface as a printable character: 'A' folds
    to 'a' in the app, and a split sequence used to flash the harness chip
    between 'all' and '21 of 22' while scrolling.
    """
    read_fd = _pipe_with(b"A")
    try:
        assert decode(b"\x1b[", fd=read_fd) == KeyEvent("up")
    finally:
        os.close(read_fd)


def test_split_home_sequence_is_completed() -> None:
    read_fd = _pipe_with(b"H")
    try:
        assert decode(b"\x1b[", fd=read_fd) == KeyEvent("home")
    finally:
        os.close(read_fd)


def test_split_escape_after_bare_esc_is_completed() -> None:
    read_fd = _pipe_with(b"[A")
    try:
        assert decode(b"\x1b", fd=read_fd) == KeyEvent("up")
    finally:
        os.close(read_fd)


def test_incomplete_sequence_is_swallowed_not_typed() -> None:
    read_fd, write_fd = os.pipe()
    os.close(write_fd)  # EOF: no continuation is coming
    try:
        assert decode(b"\x1b[", fd=read_fd) == KeyEvent("none")
    finally:
        os.close(read_fd)


@pytest.mark.parametrize(
    ("final", "expected"),
    [(b"A", KeyEvent("up")), (b"H", KeyEvent("home"))],
)
def test_late_split_sequence_is_reassembled_across_reads(
    final: bytes, expected: KeyEvent
) -> None:
    """A final byte after the 50 ms wait must not become a printable key.

    An 80 ms split used to leave ``A``/``H`` in the tty buffer; the next read
    decoded them as characters. The prefix is now carried to the next read.
    """
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, b"\x1b[")
        assert read_key(read_fd, poll_wait=0.01) == KeyEvent("none")
        time.sleep(_ESC_CONT_WAIT + 0.1)
        os.write(write_fd, final)
        assert read_key(read_fd, poll_wait=0.5) == expected
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_hostile_csi_without_a_final_byte_is_dropped() -> None:
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, b"\x1b[" + b"0" * 300)
        started = time.monotonic()
        assert read_key(read_fd, poll_wait=0.5) == KeyEvent("none")
        assert time.monotonic() - started < 0.25  # dropped, not spun on
        assert _pending(read_fd) is None  # nothing left to confuse the next key
    finally:
        os.close(read_fd)
        os.close(write_fd)


def _pending(fd: int) -> bytes | None:
    from toknt.tui import keys

    entry = keys._PENDING.get(fd)
    return entry[0] if entry else None


def test_a_stalled_sequence_does_not_eat_the_next_keystroke() -> None:
    """A stalled ``ESC [`` is ambiguous; a key that cannot complete it wins.

    The final byte of a real arrow is ``A``/``H``/…, but a letter like ``q``
    cannot complete a sequence — decoding it as a keystroke keeps the quit
    key working after a dropped sequence.
    """
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, b"\x1b[")
        assert read_key(read_fd, poll_wait=0.01) == KeyEvent("none")
        assert _pending(read_fd) == b"\x1b["  # the prefix waits
        os.write(write_fd, b"q")
        assert read_key(read_fd, poll_wait=0.5) == KeyEvent("char", "q")
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_an_abandoned_prefix_expires() -> None:
    """A sequence interrupted long ago must not shape the next key."""
    from toknt.tui import keys

    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, b"\x1b[")
        assert read_key(read_fd, poll_wait=0.01) == KeyEvent("none")
        # pretend the prefix has been waiting for a very long time
        keys._PENDING[read_fd] = (b"\x1b[", time.monotonic() - 60.0)
        os.write(write_fd, b"q")
        assert read_key(read_fd, poll_wait=0.5) == KeyEvent("char", "q")
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_read_key_none_on_quiet_fd() -> None:
    rd, write_end = os.pipe()
    os.close(write_end)  # EOF: read() returns b"" and the decoder sees nothing
    try:
        assert read_key(rd, poll_wait=0.01) is None
    finally:
        os.close(rd)
