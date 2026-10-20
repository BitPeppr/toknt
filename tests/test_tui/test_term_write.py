"""Terminal writer tests: short writes must not truncate image payloads.

A single ``os.write`` can accept fewer bytes than offered (the pty buffer
is far smaller than an image payload). Silently dropping the remainder
truncates the kitty/iTerm2/sixel payload, and the terminal then renders no
image at all — the chart simply never appears.
"""

from __future__ import annotations

import os

from toknt.tui.term import Terminal


class _ShortWriter:
    """A file object that accepts at most ``chunk`` bytes per write."""

    def __init__(self, chunk: int) -> None:
        self.chunk = chunk
        self.data = bytearray()

    def write(self, payload: bytes) -> int:
        piece = bytes(payload[: self.chunk])
        self.data += piece
        return len(piece)


def test_write_loops_until_everything_is_written(monkeypatch) -> None:
    writer = _ShortWriter(chunk=7)
    monkeypatch.setattr(os, "write", lambda fd, payload: writer.write(payload))
    term = Terminal(stdin_fd=0, stdout_fd=1)
    payload = b"\x1b_Ga=T,f=100,i=1,C=1,m=0;" + b"x" * 5000
    term.write(payload)
    assert bytes(writer.data) == payload


def test_write_of_empty_payload_is_a_noop(monkeypatch) -> None:
    called = []
    monkeypatch.setattr(os, "write", lambda fd, payload: called.append(payload) or 0)
    term = Terminal(stdin_fd=0, stdout_fd=1)
    term.write(b"")
    assert not called


def test_write_survives_a_closed_descriptor(monkeypatch) -> None:
    def boom(fd: int, payload: bytes) -> int:
        raise OSError(9, "Bad file descriptor")

    monkeypatch.setattr(os, "write", boom)
    term = Terminal(stdin_fd=0, stdout_fd=1)
    term.write(b"payload")  # must not raise
