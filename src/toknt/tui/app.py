"""Interactive application loop: keys, rescan worker, painting, teardown.

Runs at terminal-event speed — there is no frame timer. Each iteration polls
the keyboard (blocking up to ~100 ms), applies key semantics, and repaints
only when state changed or the terminal resized. Bitmap charts are emitted
right after each text diff so draw order stays deterministic (spaces under,
image above); the graphics payload carries its own positioning.

The rescan runs in its own process: parsing every changed source file is
seconds of work, and the interface must stay responsive while it happens.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from toknt import _scan
from toknt.analytics import aggregation
from toknt.config import AppConfig
from toknt.presentation import graphics
from toknt.storage.database import open_database
from toknt.tui import views
from toknt.tui.data import LIVE_WINDOW_DAYS, WINDOWS, Analysis
from toknt.tui.keys import KeyEvent, read_key
from toknt.tui.painter import Painter, terminal_size
from toknt.tui.term import Terminal
from toknt.tui.views import AppState, ImageRequest, TerminalContext

DAY_MS = 86_400_000
MIN_ROWS = 14
MIN_COLS = 60

#: How long after switching into a live window the first refresh runs: the
#: live views should show what just happened, not what was true a minute ago.
LIVE_FIRST_SCAN_MS = 2_000

#: How often the sources are re-read in the background. The scan skips
#: sources whose files have not changed, so this costs a few milliseconds
#: when nothing happened and a real read when something did.
SCAN_INTERVAL_MS = 30_000

#: Historical name (the interval used to be tied to the live windows).
LIVE_SCAN_INTERVAL_MS = SCAN_INTERVAL_MS

#: The rescan runs in a child process: the same code the command line
#: uses, and no risk of a child importing (and re-running) the script that
#: started the TUI. The child's import path is set to *this* package, so a
#: source checkout and an installed release scan with the same code.
SCAN_CODE = (
    "import sys; sys.path.insert(0, {root!r}); "
    "from toknt._scan import main; main()"
)


def _clamp(value: int, lo: int, hi: int) -> int:
    return lo if value < lo else hi if value > hi else value


class App:
    """Owns the terminal, painter, state, and the background scan worker."""

    def __init__(self, cfg: AppConfig, *, stdin_fd: int = 0, stdout_fd: int = 1) -> None:
        self.cfg = cfg
        self.stdin_fd = stdin_fd
        self.stdout_fd = stdout_fd
        self.conn = open_database()
        self.analysis = Analysis(self.conn, cfg)
        self.painter = Painter()
        self.state = AppState()
        self.term = Terminal(stdin_fd=stdin_fd, stdout_fd=stdout_fd)
        # Environment-only detection here; cell-size and background queries
        # happen in run() after raw mode, since replies arrive on stdin.
        self.ctx = TerminalContext(
            backend=graphics.detect_backend(override=cfg.ui_graphics)
        )
        self._scan_process: subprocess.Popen | None = None
        self._scan_log: object | None = None  # the child's stderr
        self._scan_mark = 0              # scan_runs id before the last scan
        self._scan_done = False          # main loop drains this into a repaint
        self._last_images: dict[int, tuple] = {}  # image_id → placed slot
        # one clock anchor for the whole session: window since-values must be
        # byte-stable or the data memo (and chart cache) never repeat. Live
        # windows advance it when a background scan completes, so the window
        # slides forward with fresh data instead of freezing at boot time.
        self._now_ms = int(time.time() * 1000)
        self._next_auto_scan = self._now_ms + SCAN_INTERVAL_MS

    # -- lifecycle -----------------------------------------------------------

    def run(self) -> int:
        self.term.enter()
        try:
            self._boot()
            return self._loop()
        finally:
            self._stop_scan()
            self.conn.close()
            self.term.leave()

    def _boot(self) -> None:
        """Prepare the terminal and start the first refresh.

        Every open reads the sources again: the last session may have been
        hours (or days) ago, and the numbers on screen should be the
        numbers as of now.
        """
        self._probe_terminal()
        self._refresh_size(force=True)
        self._start_rescan()

    def _probe_terminal(self) -> None:
        """Query cell metrics + background AFTER raw mode (replies hit stdin)."""
        ctx = self.ctx
        theme, bg_rgb = graphics.terminal_background()
        if ctx.backend != "off":
            cell_w, cell_h = graphics.measure_cell()
            self.ctx = dataclasses.replace(
                ctx, cell_w=cell_w, cell_h=cell_h, theme=theme, bg_rgb=bg_rgb
            )
        else:
            self.ctx = dataclasses.replace(ctx, theme=theme, bg_rgb=bg_rgb)

    def _refresh_size(self, *, force: bool) -> bool:
        rows, columns = terminal_size()
        if force or (rows, columns) != (self.painter.rows, self.painter.columns):
            self.painter.resize(rows, columns)
            self.term.write(b"\x1b[H\x1b[2J")
            return True
        return False

    # -- loop ----------------------------------------------------------------

    def _loop(self) -> int:
        dirty = True
        while True:
            if self._refresh_size(force=False):
                dirty = True
            event = read_key(self.stdin_fd, poll_wait=0.1)
            if event is not None and self.apply_key(event):
                dirty = True
            self._drain_scan()
            if self._scan_done:
                self._scan_done = False
                # the corpus changed: slide the clock anchor so live windows
                # move forward (cache keys shift exactly once per scan)
                self._now_ms = int(time.time() * 1000)
                self._next_auto_scan = self._now_ms + SCAN_INTERVAL_MS
                dirty = True  # finished scans repaint even without keypresses
            self._maybe_auto_scan()
            if dirty:
                self.paint()
                dirty = False

    def _drain_scan(self) -> None:
        """Pick up a finished scan's result (the main loop's job)."""
        process = self._scan_process
        if process is None or process.poll() is None:
            return  # still running
        self._scan_process = None
        self.analysis.invalidate()  # the corpus changed; drop memoized views
        self.state.scanning = False
        if process.returncode:
            reason = self._scan_error()
            self.state.status = f"scan failed{': ' + reason if reason else ''}"
        else:
            inserted = self.conn.execute(
                "SELECT COALESCE(SUM(events_inserted), 0) FROM scan_runs WHERE id > ?",
                (self._scan_mark,),
            ).fetchone()
            count = int(inserted[0]) if inserted else 0
            self.state.status = f"scan done: {count} new events"
        if self._scan_log is not None:
            self._scan_log.close()
            self._scan_log = None
        self._scan_done = True

    def _is_live_window(self) -> bool:
        window = WINDOWS[self.state.window]
        return window.days is not None and window.days <= LIVE_WINDOW_DAYS

    def _draw_graphics_selftest(self) -> bool:
        """Draw the shared test chart in the TUI's own context ('g' key).

        Same image as ``toknt graphics`` — if that command shows a chart but
        this does not, the difference is the TUI's context (alternate
        screen, raw mode) rather than the payload.
        """
        backend = self.ctx.backend
        if backend == "off":
            self.state.status = "graphics off (braille charts)"
            return True
        # the test image fits the screen it is drawn on: a fixed 80x20
        # image is larger than a small terminal and lands clipped
        columns = max(min(80, self.painter.columns - 4), 8)
        rows = max(min(20, self.painter.rows - 4), 4)
        width_px = columns * self.ctx.cell_w
        height_px = rows * self.ctx.cell_h
        png, rgba = graphics.test_chart_png(width_px, height_px)
        payload = graphics.encode_placement(
            backend,
            png,
            rgba,
            width_px,
            height_px,
            columns=columns,
            rows=rows,
            image_id=3,
        )
        self.term.write(b"\x1b[3;3H" + payload)
        self._last_images[3] = ("selftest", 2, 2, columns, rows)
        self.state.status = f"graphics self-test drawn ({backend})"
        return True

    def _maybe_auto_scan(self) -> None:
        """Start the background rescan when the tick comes due.

        Every window refreshes: the scan skips the sources whose files have
        not changed, so a tick on an idle machine costs milliseconds, and a
        tick while a harness is busy brings in the numbers it just wrote.
        """
        if self.state.scanning:
            return
        if time.time() * 1000 >= self._next_auto_scan:
            self._start_rescan()

    # -- application of keys ---------------------------------------------------

    def apply_key(self, event: KeyEvent) -> bool:
        """Mutate state for one key; True when the screen needs a repaint."""
        if event.name == "ctrl+c":
            # the key reference promises ctrl+c quits: the terminal is in
            # raw mode, so the signal never arrives on its own
            raise SystemExit(0)
        if event.name in ("ctrl+d", "esc"):
            return False
        if event.is_char("q"):
            raise SystemExit(0)
        if event.name == "char":
            char = (event.char or "").casefold()
            if char == "?":
                self.state.help = not self.state.help
                return True
            if char.isdigit() and char != "0":
                index = int(char) - 1
                if index < len(views.VIEW_TITLES):
                    self.state.view = index
                    self.state.status = ""  # a note about the old view is stale
                    return True
            if char == "t":
                # the window cycle runs backwards: from the default three
                # days down to a day, an afternoon, a morning, and then on
                # to everything
                self.state.window = (self.state.window - 1) % len(WINDOWS)
                self.state.cursor_day = None
                self.state.status = ""
                if self._is_live_window():
                    # fresh data on arrival: scan soon, then keep ticking
                    self._next_auto_scan = time.time() * 1000 + LIVE_FIRST_SCAN_MS
                return True
            if char == "v":
                self.state.metric = (self.state.metric + 1) % len(views.METRICS)
                self.state.status = ""
                return True
            if char == "p":
                if self.state.view != 1:  # only the models chart has rates
                    self.state.status = "tok/s applies to the models view"
                    return True  # repaint so the status shows
                self.state.rate_mode = not self.state.rate_mode
                self.state.status = ""
                return True
            if char == "m":
                self.state.merge_models = not self.state.merge_models
                return True
            if char == "h":
                return self._cycle_harness_focus()
            if char == "g":
                return self._draw_graphics_selftest()
            if char == "r":
                self._start_rescan()
                return True
        if event.name in ("up", "down", "left", "right"):
            return self._move_cursor(event.name)
        return False

    def _cycle_harness_focus(self) -> bool:
        """'h' cycles the harness filter: all → each harness with data → all.

        Only harnesses that have tokens in the current window participate, so
        the cycle never walks 20 empty entries; with no data at all it is a
        no-op back to 'all'.
        """
        daily = self.analysis.daily(self._since(), views.HarnessToggles())
        order = [
            name
            for name in views.HARNESS_NAMES
            if any(daily.per_agent.get(day, {}).get(name) for day in daily.days)
        ]
        current_all = len(self.state.agents.active) == len(views.HARNESS_NAMES)
        if current_all or not order:
            if not order:
                return False
            self.state.agents = self.state.agents.toggled_all_only(order[0])
        else:
            active = sorted(self.state.agents.active)
            if len(active) == 1 and active[0] in order:
                index = order.index(active[0])
                if index + 1 < len(order):
                    self.state.agents = self.state.agents.toggled_all_only(order[index + 1])
                else:
                    self.state.agents = views.HarnessToggles()  # back to all
            else:
                self.state.agents = self.state.agents.toggled_all_only(order[0])
        self.state.cursor_day = None
        return True

    # -- rescan process --------------------------------------------------------

    def _start_rescan(self) -> None:
        """Start the rescan in a child process.

        Parsing every changed source file is seconds of work. In a thread
        that work would hold the GIL and the interface would stutter
        exactly when the user is looking at fresh numbers, so the scan runs
        as its own process; the outcome is picked up from the database.
        """
        if self.state.scanning:
            self.state.status = "scan already running"
            return
        self.state.scanning = True
        self.state.status = "scanning sources…"
        self._scan_mark = self._last_scan_run_id()
        root = str(Path(_scan.__file__).resolve().parent.parent)
        self._scan_log = tempfile.TemporaryFile()
        try:
            self._scan_process = subprocess.Popen(
                [sys.executable, "-c", SCAN_CODE.format(root=root)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=self._scan_log,
                env={**os.environ, _scan.CONFIG_ENV: json.dumps(self._scan_payload())},
            )
        except OSError as exc:  # no interpreter, no permissions
            self.state.scanning = False
            self.state.status = f"scan failed: {type(exc).__name__}"
            self._scan_process = None
            self._scan_log.close()
            self._scan_log = None

    def _scan_error(self) -> str:
        """The child's last words, when it had any."""
        if self._scan_log is None:
            return ""
        try:
            self._scan_log.seek(0)
            text = self._scan_log.read().decode("utf-8", "replace").strip()
        except OSError:  # pragma: no cover - the log file is gone
            return ""
        if not text:
            return ""
        last = text.splitlines()[-1].strip()
        return last[:120]

    def _scan_payload(self) -> dict:
        """The configuration the child needs to scan exactly our sources."""
        cfg = self.cfg
        return {
            "enabled_harnesses": [str(harness) for harness in cfg.enabled_harnesses],
            "source_overrides": dict(cfg.source_overrides),
            "default_since": cfg.default_since,
            "default_bucket": cfg.default_bucket,
            "ignore_projects": list(cfg.ignore_projects),
        }

    def _last_scan_run_id(self) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM scan_runs"
        ).fetchone()
        return int(row[0]) if row else 0

    def _stop_scan(self) -> None:
        """Tear the scan process down (the app is closing)."""
        process = self._scan_process
        self._scan_process = None
        if process is not None and process.poll() is None:
            process.terminate()
        if self._scan_log is not None:
            self._scan_log.close()
            self._scan_log = None
        self.state.scanning = False

    # -- painting --------------------------------------------------------------

    def paint(self) -> None:
        rows, columns = self.painter.rows, self.painter.columns
        self.painter.clear(0, 0, columns, rows)
        if rows < MIN_ROWS or columns < MIN_COLS:
            self.painter.text(0, 0, f"terminal too small: need ≥{MIN_ROWS} rows")
            self.painter.text(1, 0, f"× {MIN_COLS} columns")
            self._sync_images_begin(())
            self.term.write(self.painter.frame())
            self._sync_images_end(())
            return
        views.title_row(
            self.painter,
            self.state,
            rows=rows,
            columns=columns,
        )
        painters = (
            views.paint_contribution,
            views.paint_models,
            views.paint_tools,
            views.paint_sessions,
        )
        view_paint = painters[self.state.view % len(painters)](
            self.painter,
            self.state,
            self.analysis,
            rows=rows,
            columns=columns,
            ctx=self.ctx,
            now_ms=self._now_ms,
        )
        # Help overlays as text; charts stay on screen underneath (the image
        # protocols composite text over transparent pixels, and the overlay
        # box only covers the middle of the chart, never erasing it).
        requests = view_paint.images
        if self.state.help:
            views.paint_help(self.painter, rows, columns)
        self._sync_images_begin(requests)
        self.term.write(self.painter.frame())
        self._sync_images_end(requests)

    # -- image lifecycle ---------------------------------------------------------

    def _image_slots(self, requests: tuple[ImageRequest, ...]) -> dict[int, tuple]:
        """image_id → (key, row, col, columns, rows) for a frame's images."""
        if self.ctx.backend not in ("kitty", "iterm2", "sixel"):
            return {}
        slots: dict[int, tuple] = {}
        for request in requests:
            slots[request.image_id] = (
                request.key,
                request.row,
                request.col,
                request.columns,
                request.rows,
            )
        return slots

    def _sync_images_begin(self, requests: tuple[ImageRequest, ...]) -> None:
        """Erase art that won't survive this frame, BEFORE text is emitted.

        Text cell runs repaint cells but never images, and sixel/iTerm2
        payloads only paint where pixels are opaque — stale art would show
        through gaps or linger behind text without this pass.
        """
        previous = self._last_images
        if not previous:
            return
        backend = self.ctx.backend
        upcoming = self._image_slots(requests)
        vanished = [image_id for image_id in previous if image_id not in upcoming]
        if not vanished:
            return
        for image_id in vanished:
            key, row, col, columns, rows = previous[image_id]
            if backend == "kitty":
                # a new placement deletes its image id itself; a vanished one
                # needs an explicit delete
                # d=i: delete only this image (a bare a=d deletes ALL,
                # which erased surviving charts when a sibling vanished);
                # q=1: keep the terminal's OK reply out of our stdin
                self.term.write(
                    b"\x1b_Ga=d,d=i,i=%d,C=1,q=1\x1b\\" % image_id
                )
            elif backend in ("sixel", "iterm2"):
                # any change in content or geometry leaves stale/ghost
                # pixels: opaque background fill over the exact old rectangle
                self._erase_bitmap(row, col, columns, rows)
        self._last_images = {
            image_id: slot for image_id, slot in previous.items() if image_id in upcoming
        }

    def _sync_images_end(self, requests: tuple[ImageRequest, ...]) -> None:
        """Place new/changed bitmaps after the text frame, skipping no-ops."""
        slots = self._image_slots(requests)
        for image_id, slot in slots.items():
            if self._last_images.get(image_id) == slot:
                continue
            request = next(r for r in requests if r.image_id == image_id)
            self.term.write(
                b"\x1b[%d;%dH" % (request.row + 1, request.col + 1) + request.payload
            )
        if set(slots) != set(self._last_images):
            self._last_images = {k: v for k, v in self._last_images.items() if k in slots}
        self._last_images.update(slots)

    def _erase_bitmap(self, row: int, col: int, columns: int, rows: int) -> None:
        """Paint background-colored opaque pixels over a former image rect."""
        bg = self.ctx.bg_rgb
        if bg is None:
            bg = 0x101010 if self.ctx.theme == "dark" else 0xEDEDED
        width_px = max(columns * self.ctx.cell_w, 1)
        height_px = max(rows * self.ctx.cell_h, 1)
        png, rgba = graphics.solid_fill(width=width_px, height=height_px, color=bg)
        if self.ctx.backend == "sixel":
            # exact color: full 256-bin quantization keeps the palette entry
            # equal to the queried background, so the fill is truly invisible
            payload = graphics.sixel_bytes(width_px, height_px, rgba, levels=255)
        else:
            payload = graphics.iterm2_inline(
                png, columns=max(columns, 1), rows=max(rows, 1)
            )
        self.term.write(b"\x1b[%d;%dH" % (row + 1, col + 1) + payload)

    # -- cursor / selection ------------------------------------------------------

    def _since(self) -> int:
        return views.since_for(self.state, self._now_ms)

    def _grid_bounds(self) -> tuple[int, int] | None:
        """The day range the contribution grid can show.

        The upper bound is today's midnight, not "now": the grid paints
        whole days, so a cursor parked on a timestamp inside today would
        match no cell — the highlight would vanish and the picked day
        would read zero tokens.
        """
        window = WINDOWS[self.state.window]
        max_weeks = max(4, (self.painter.columns - 8) // 2)
        wanted = 26 if window.days is None else int(max(1, window.days // 7))
        weeks = views.widgets.week_starts(
            self._now_ms, min(wanted, max_weeks), self.analysis.tz
        )
        if not weeks:
            return None
        today = aggregation.bucket_start_ms(self._now_ms, "day", self.analysis.tz)
        return weeks[0], max(today, weeks[0])

    def _move_cursor(self, name: str) -> bool:
        state = self.state
        if state.view == 0:
            step = {
                "left": -DAY_MS,
                "right": DAY_MS,
                "up": -7 * DAY_MS,
                "down": 7 * DAY_MS,
            }[name]
            bounds = self._grid_bounds()
            if state.cursor_day is None:
                daily = self.analysis.daily(self._since(), state.agents)
                state.cursor_day = daily.days[-1] if daily.days else None
            if state.cursor_day is None:
                return False
            if bounds is not None:
                low, high = bounds
                state.cursor_day = _clamp(state.cursor_day + step, low, high)
            else:
                state.cursor_day += step
            return True
        if state.view == 2:
            count = len(self.analysis.tool_rows(self._since(), state.agents))
            # the table paints a capped number of rows: the cursor must not
            # walk past the rows that are actually on screen
            count = min(count, views.TOOL_ROW_CAP)
            state.tool_row = _clamp(
                state.tool_row + (1 if name == "down" else -1), 0, max(count - 1, 0)
            )
            return count > 0
        if state.view == 3:
            # the view paints up to 200 sessions through a 24-row window;
            # clamp the cursor against that same list or scrolling stops
            # after the first page
            count = len(
                self.analysis.session_rows(
                    self._since(),
                    state.agents,
                    limit=200,
                    merge=state.merge_models,
                    metric=views.current_metric(state),
                )
            )
            state.session_row = _clamp(
                state.session_row + (1 if name == "down" else -1), 0, max(count - 1, 0)
            )
            return count > 0
        return False


def run(cfg: AppConfig, *, stdin_fd: int = 0, stdout_fd: int = 1) -> None:
    """Entry point used by the CLI: boot into the default contributions view."""
    App(cfg, stdin_fd=stdin_fd, stdout_fd=stdout_fd).run()
