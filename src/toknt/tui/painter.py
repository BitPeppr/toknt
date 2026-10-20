"""Full-screen cell painter with frame diffing.

The TUI is a pure cell model plus an escape-sequence emitter:

* cells carry a char, foreground/background colors (or ``None`` = terminal
  default) and an attribute mask (bold, underline, reverse),
* :meth:`Painter.frame` walks the grid against the last-emitted snapshot and
  emits only changed runs, each prefixed with an absolute cursor position —
  so image transports that leave the cursor anywhere unknown are safe:
  every run repositions anyway,
* double-width characters reserve a sentinel continuation cell so later
  overwrites and erases stay aligned; the emitted text skips sentinels and
  lets the terminal's implicit advance happen.

Terminal lifecycle helpers (alternate screen, raw modes, cursor) live in
:mod:`toknt.tui.term` — this module stays pure cell math and encoding.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass

from toknt.tui.style import char_width

BOLD = 1
UNDERLINE = 2
REVERSE = 4
DIM = 8


@dataclass(frozen=True, slots=True, eq=True)
class Attr:
    fg: int | None
    bg: int | None
    mask: int

    def copy(self, **updates: int | None | object) -> Attr:
        values = {"fg": self.fg, "bg": self.bg, "mask": self.mask}
        values.update(updates)
        return Attr(values["fg"], values["bg"], int(values["mask"]))


DEFAULT_ATTRS = Attr(None, None, 0)


@dataclass(slots=True, eq=True)
class Cell:
    char: str
    fg: int | None
    bg: int | None
    mask: int
    sentinel: bool = False


def _blank() -> Cell:
    return Cell(" ", None, None, 0)


def _sentinel(attrs: Attr) -> Cell:
    return Cell("", attrs.fg, attrs.bg, attrs.mask, True)


def _initial_snapshot_cell() -> Cell:
    # impossible cell so the first frame paints everything
    return Cell("\x00", -1, -1, -1, True)


def terminal_size() -> tuple[int, int]:
    """(rows, columns) of the controlling terminal.

    The ioctl is authoritative: ``shutil.get_terminal_size`` prefers the
    COLUMNS/LINES environment variables, which are frequently stale (or set
    by a parent process) and would size the whole UI to the wrong width —
    pushing the right-hand panes off screen.
    """
    for stream in (sys.stdout, sys.stdin, sys.stderr):
        try:
            size = os.get_terminal_size(stream.fileno())
        except (AttributeError, ValueError, OSError):
            continue
        if size.columns > 0 and size.lines > 0:
            return size.lines, size.columns
    columns, rows = shutil.get_terminal_size()
    return rows, columns


class Painter:
    """Diff-painting buffer over the whole terminal."""

    def __init__(self) -> None:
        self.rows = 0
        self.columns = 0
        self.buffer: list[list[Cell]] = []
        self.snapshot: list[list[Cell]] = []

    # -- lifecycle -----------------------------------------------------------

    def resize(self, rows: int, columns: int) -> None:
        """Adopt a new size; caller must request a full repaint."""
        self.rows = max(rows, 0)
        self.columns = max(columns, 0)
        self.buffer = [[_blank() for _ in range(self.columns)] for _ in range(self.rows)]
        self.snapshot = [
            [_initial_snapshot_cell() for _ in range(self.columns)]
            for _ in range(self.rows)
        ]

    # -- drawing -------------------------------------------------------------

    def clear(self, row: int, col: int, width: int, height: int) -> None:
        """Overwrite a rectangle with default cells (erases images too)."""
        for y in range(row, min(row + height, self.rows)):
            line = self.buffer[y]
            for x in range(col, min(col + width, self.columns)):
                line[x] = _blank()

    def text(self, row: int, col: int, content: str, attrs: Attr = DEFAULT_ATTRS) -> None:
        """Write ``content`` clipped to the row, honoring wide characters."""
        if row < 0 or row >= self.rows or col >= self.columns:
            return
        line = self.buffer[row]
        index = col
        for char in content:
            width = char_width(char)
            if index + width > self.columns:
                break
            line[index] = Cell(char, attrs.fg, attrs.bg, attrs.mask)
            if width == 2:
                line[index + 1] = _sentinel(attrs)
                index += 2
            else:
                index += 1

    def box(self, row: int, col: int, width: int, height: int, attrs: Attr) -> None:
        """Single-line box with a title row shared with tables: not used yet."""
        top = "─" * max(width - 2, 0)
        self.text(row, col, "╭" + top + "╮", attrs)
        for y in range(row + 1, row + height):
            self.text(y, col, "│" + " " * (width - 2) + "│", attrs)
        self.text(row + height - 1, col, "╰" + top + "╯", attrs)

    # -- encoding ------------------------------------------------------------

    def frame(self) -> bytes:
        """Escape bytes for every changed run; snapshot becomes current."""
        if not self.buffer:
            return b""
        out: list[bytes] = []
        for y in range(self.rows):
            line = self.buffer[y]
            snap = self.snapshot[y]
            x = 0
            while x < self.columns:
                key = (line[x].fg, line[x].bg, line[x].mask)
                if key == (snap[x].fg, snap[x].bg, snap[x].mask) and line[x] == snap[x]:
                    x += 1
                    continue
                start = x
                chars: list[str] = []
                while x < self.columns:
                    ahead = line[x]
                    if (ahead.fg, ahead.bg, ahead.mask) != key:
                        break
                    if not ahead.sentinel:
                        chars.append(ahead.char)
                    snap[x] = Cell(ahead.char, ahead.fg, ahead.bg, ahead.mask, ahead.sentinel)
                    x += 1
                out.append(
                    _position(y, start).encode("ascii")
                    + _encode_attrs(key).encode("ascii")
                    + "".join(chars).encode("utf-8")
                )
        if out:
            out.append(b"\x1b[0m")  # leave clean state behind the frame
        return b"".join(out)


def _position(row: int, col: int) -> str:
    return f"\x1b[{row + 1};{col + 1}H"


def _encode_attrs(key: tuple[int | None, int | None, int]) -> str:
    """Encode a run style over an ABSOLUTE SGR state.

    Every attribute the terminal remembers is either turned on or explicitly
    off in each run (``22`` cancels bold+dim, ``24`` underline, ``27``
    reverse); otherwise attributes from previous runs — e.g. the reversed
    hint bar — would leak into unrelated runs and wash the screen grey.
    """
    fg, bg, mask = key
    codes: list[str] = []
    # attribute set / clear (state-changing bits first)
    if mask & BOLD:
        codes.append("1")
    elif mask & DIM:
        codes.append("2")
    else:
        codes.append("22")
    codes.append("4" if mask & UNDERLINE else "24")
    codes.append("7" if mask & REVERSE else "27")
    # colors
    if fg is None:
        codes.append("39")
    else:
        codes.append(f"38;2;{(fg >> 16) & 255};{(fg >> 8) & 255};{fg & 255}")
    if bg is None:
        codes.append("49")
    else:
        codes.append(f"48;2;{(bg >> 16) & 255};{(bg >> 8) & 255};{bg & 255}")
    return "\x1b[" + ";".join(codes) + "m"
