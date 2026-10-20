"""Terminal lifecycle for the TUI: alt screen, raw input, image-friendly output.

Painting goes through :class:`Terminal`, which owns the raw-mode fd, the
alternate-screen handoff and the final cleanup paths. All writes are raw
bytes on fd 1: styled text cells produce no Rich involvement and image
payloads (kitty's APC, iTerm2's OSC, sixel's DCS) pass through untouched.
"""

from __future__ import annotations

import atexit
import os
import select
import sys
import termios
import tty

__all__ = ["Terminal"]


class Terminal:
    """Own raw mode + alternate screen and paints bytes on fd 1."""

    def __init__(self, *, stdin_fd: int = 0, stdout_fd: int = 1) -> None:
        self.stdin_fd = stdin_fd
        self.stdout_fd = stdout_fd
        self._saved: list | None = None  # termios tcgetattr attribute list
        self.leave_requested = False

    # -- lifecycle -----------------------------------------------------------

    def enter(self) -> None:  # pragma: no cover - needs a real terminal
        self._saved = termios.tcgetattr(self.stdin_fd)
        tty.setraw(self.stdin_fd)  # clears ISIG: Ctrl+C arrives as keypress 0x03
        sys.stdout.write("\x1b[?1049h\x1b[?25l\x1b[H\x1b[2J\x1b[3J")
        sys.stdout.flush()
        atexit.register(self.leave)

    def leave(self) -> None:  # pragma: no cover - needs a real terminal
        if self._saved is not None:
            termios.tcsetattr(self.stdin_fd, termios.TCSADRAIN, self._saved)
            self._saved = None
        sys.stdout.write("\x1b[?25h\x1b[?1049l\x1b[0m")
        sys.stdout.flush()

    # -- output --------------------------------------------------------------

    def write(self, data: bytes) -> None:
        """Write every byte, even when the terminal takes only part of it.

        A single ``os.write`` may accept fewer bytes than requested (the
        pty buffer is small compared to an image payload). Dropping the
        remainder truncates the kitty/iTerm2/sixel payload and the terminal
        then renders no image at all — so the write is looped until the
        buffer is empty, waiting on EAGAIN when stdout is non-blocking.
        """
        if not data:
            return
        view = memoryview(data)
        while view:
            try:
                written = os.write(self.stdout_fd, view)
            except BlockingIOError:
                select.select([], [self.stdout_fd], [], 0.05)
                continue
            except OSError:
                return
            if written <= 0:
                return
            view = view[written:]

    # -- signals -------------------------------------------------------------

