"""Raw keyboard decoding for the interactive mode.

Keys are read from a raw-mode fd one event at a time. The decoder covers
everything the TUI binds: letters/digits/space, enter, a bare ESC, ctrl
keys, arrows, Home/End, delete and page up/down. Anything unrecognised
degrades to ``'none'`` — the app ignores keys it does not bind, so exotic
escape sequences stay harmless. ``decode`` is pure (given a continuation
source); ``read_key`` is its I/O wrapper.
"""

from __future__ import annotations

import os
import select
from dataclasses import dataclass

__all__ = ["KeyEvent", "read_key", "decode"]

_ESC_CONT_WAIT = 0.05  # seconds to wait for the rest of an escape sequence


@dataclass(frozen=True, slots=True)
class KeyEvent:
    name: str  # symbolic: 'char', 'enter', 'up', 'esc', 'ctrl+c', ...
    char: str | None = None  # payload for 'char'

    def is_char(self, ch: str) -> bool:
        return self.name == "char" and (self.char or "").casefold() == ch


def read_key(fd: int, *, poll_wait: float = 0.0) -> KeyEvent | None:
    """Return the next key event, or ``None`` when none is queued."""
    first = _read_bytes(fd, poll_wait)
    if not first:
        return None
    return decode(first, fd=fd)


def _read_bytes(fd: int, wait: float) -> bytes:
    readable, _, _ = select.select([fd], [], [], wait)
    if not readable:
        return b""
    try:
        return os.read(fd, 64)
    except OSError:
        return b""


_ARROW_SEQUENCES = {
    b"[A": "up",
    b"[B": "down",
    b"[C": "right",
    b"[D": "left",
    b"OA": "up",
    b"OB": "down",
    b"OC": "right",
    b"OD": "left",
}


def decode(data: bytes, *, fd: int | None = None) -> KeyEvent:
    """Decode one event from ``data``; ``fd`` supplies continuation bytes."""
    if not data:
        return KeyEvent("none")
    if data[0] == 0x1B:
        return _decode_escape(data[1:], fd)
    if data[0] in (13, 10):
        return KeyEvent("enter")
    if data[0] == 3:
        return KeyEvent("ctrl+c")
    if data[0] == 4:
        return KeyEvent("ctrl+d")
    if data[0] == 9:
        return KeyEvent("tab")
    if data[0] == 0x20:
        return KeyEvent("char", " ")
    try:
        text = data.decode("utf-8", "strict")
    except UnicodeDecodeError:
        return KeyEvent("none")
    first = text[:1]
    if not first.isprintable():
        return KeyEvent("none")
    return KeyEvent("char", first)


def _decode_escape(body: bytes, fd: int | None) -> KeyEvent:
    if not body:
        more = _read_bytes(fd, _ESC_CONT_WAIT) if fd is not None else b""
        if more:
            return decode(more, fd=fd)
        return KeyEvent("esc")
    if body[:2] in _ARROW_SEQUENCES:
        return KeyEvent(_ARROW_SEQUENCES[body[:2]])
    if body.startswith(b"[5~"):
        return KeyEvent("pgup")
    if body.startswith(b"[6~"):
        return KeyEvent("pgdown")
    if body.startswith(b"[3~"):
        return KeyEvent("delete")
    if body.startswith(b"[1~") or body.startswith(b"[H"):
        return KeyEvent("home")
    if body.startswith(b"[4~") or body.startswith(b"[F"):
        return KeyEvent("end")
    return KeyEvent("none")  # unrecognized sequence: swallow for safety
