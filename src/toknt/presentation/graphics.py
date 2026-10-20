"""High-fidelity in-terminal images: kitty graphics, iTerm2 inline, sixel.

Charts no longer have to be drawn from character cells alone. When the
terminal understands an image protocol, toknt renders its charts into an
RGBA bitmap with soft-edged anti-aliased strokes and crisp fills, encodes
the bitmap as PNG (stdlib ``zlib`` only — no third-party dependencies) and
shows it where a lower-resolution block chart used to be.

Three transports are supported:

* **kitty graphics protocol** (kitty, ghostty, WezTerm, ...) — APC payload,
  chunked, transparent PNG, never moves the cursor.
* **iTerm2 inline images** (iTerm2) — OSC 1337 with a base64 PNG.
* **sixel** (xterm, foot, ...) — palette-indexed columnar encoding of a
  quantised copy of the canvas; transparency is composited by omission.

Detection is environment-driven; ``TOKNT_GRAPHICS`` forces a transport
(``auto|kitty|sixel|iterm2|off``). Everything degrades gracefully: an
unrecognised terminal simply keeps the character-cell charts.
"""

from __future__ import annotations

import base64
import math
import os
import re
import select
import struct
import sys
import termios
import time
import zlib
from collections.abc import Mapping, Sequence

__all__ = [
    "SERIES_PALETTE",
    "Canvas",
    "detect_backend",
    "encode_placement",
    "iterm2_inline",
    "kitty_apc",
    "measure_cell",
    "sixel_bytes",
    "terminal_background",
]


# Okabe–Ito subset: colorblind-safe, mid-luminance, legible on dark and light
# terminal backgrounds alike.
SERIES_PALETTE: tuple[int, ...] = (
    0x56B4E9,  # sky
    0xE69F00,  # orange
    0x0072B2,  # blue
    0x009E73,  # green
    0xCC79A7,  # pink
    0xD55E00,  # vermillion
    0x8C8C8C,  # grey
)


# ----------------------------------------------------------------- detection


def detect_backend(
    env: Mapping[str, str] | None = None,
    *,
    stdout_is_tty: bool | None = None,
    override: str | None = None,
) -> str:
    """Return ``kitty|sixel|iterm2|off`` for the running terminal.

    ``override`` mirrors the ``TOKNT_GRAPHICS`` value: ``auto`` probes the
    environment, an explicit transport is honoured verbatim (so users can
    work around misdetection), and ``off`` keeps the character-cell charts.
    """
    return describe_backend(
        env, stdout_is_tty=stdout_is_tty, override=override
    )[0]


def describe_backend(
    env: Mapping[str, str] | None = None,
    *,
    stdout_is_tty: bool | None = None,
    override: str | None = None,
) -> tuple[str, str]:
    """The detected backend plus a human-readable reason.

    The reason matters when charts do not appear: it names the terminal
    that was detected (or explains why none was).
    """
    env = env if env is not None else os.environ
    stdout_is_tty = sys.stdout.isatty() if stdout_is_tty is None else stdout_is_tty
    term = env.get("TERM", "")
    term_program = env.get("TERM_PROGRAM", "")
    if override and override != "auto":
        if override in ("kitty", "sixel", "iterm2") and stdout_is_tty:
            return override, f"forced by --backend/TOKNT_GRAPHICS ({override})"
        return "off", "forced off by --backend/TOKNT_GRAPHICS"
    if not stdout_is_tty:
        return "off", "stdout is not a terminal"
    # Multiplexers do not pass image protocols through to the outer
    # terminal, so images would silently vanish.
    if env.get("TMUX") or env.get("STY"):
        return "off", "running inside a terminal multiplexer (tmux/screen)"
    if env.get("KITTY_WINDOW_ID") or "kitty" in term or term_program == "kitty":
        return "kitty", "kitty protocol (kitty)"
    if (
        term_program == "ghostty"
        or env.get("GHOSTTY_RESOURCES_DIR")
        or term_program == "Ghostty"
    ):
        return "kitty", "kitty protocol (ghostty)"
    if term_program == "WezTerm":
        return "kitty", "kitty protocol (WezTerm)"
    if term_program in ("WarpTerminal", "warp", "rio"):
        return "kitty", f"kitty protocol ({term_program})"
    if term_program == "iTerm.app" or env.get("ITERM_SESSION_ID"):
        return "iterm2", "iTerm2 inline images"
    if "sixel" in term or term.startswith(("xterm", "foot", "st-256color", "mintty")):
        return "sixel", f"sixel (TERM={term})"
    if term_program:
        return "off", f"{term_program} has no image protocol"
    return "off", f"no image protocol detected (TERM={term or 'unset'})"


# ------------------------------------------------------------- PNG encoding


def png_encode(width: int, height: int, rgba: bytes | bytearray) -> bytes:
    """Minimal RGBA PNG writer (stdlib only); ``rgba`` is W*H*4 bytes."""
    stride = width * 4
    if len(rgba) != stride * height:
        raise ValueError(f"rgba buffer is {len(rgba)} bytes, expected {stride * height}")
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter None per scanline (correctness over compression)
        raw += rgba[y * stride : (y + 1) * stride]
    idat = zlib.compress(bytes(raw), 6)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
    )


# ------------------------------------------------------------- transporters


def kitty_apc(png: bytes, *, columns: int, rows: int, image_id: int = 1) -> bytes:
    """Display ``png`` across ``columns``/``rows`` cells via the kitty protocol.

    Data travels in <=4096-byte base64 chunks; ``m=1`` means "more coming"
    and ``m=0`` closes the transmission. The previous placement is deleted
    first (``a=d``) so redrawing the same image id never leaks stale
    artwork, and ``C=1`` keeps the cursor where the painter left it.
    """
    payload = base64.b64encode(png).decode("ascii")
    chunks = [payload[i : i + 4096] for i in range(0, len(payload), 4096)] or [""]
    out = bytearray()
    # d=i is REQUIRED: a bare a=d defaults to d=a — delete every image —
    # so with two charts (line + donut) the second delete erased the first.
    out += f"\x1b_Ga=d,d=i,i={image_id},C=1,q=1\x1b\\".encode("ascii")
    for n, piece in enumerate(chunks):
        last = n == len(chunks) - 1
        more = "0" if last else "1"
        if n == 0:
            params = (
                f"a=T,f=100,i={image_id},C=1,q=1,c={columns},r={rows},m={more}"
            )
        else:
            # continuation chunks repeat the image id: the protocol
            # associates them with their transmission by id, and a chunk
            # without it is orphaned (the terminal renders nothing at all)
            params = f"i={image_id},C=1,m={more}"
        out += f"\x1b_G{params};{piece}\x1b\\".encode("ascii")
    return bytes(out)


def iterm2_inline(png: bytes, *, columns: int, rows: int) -> bytes:
    """iTerm2 inline image: OSC 1337 ``File=`` with a base64 PNG payload."""
    encoded = base64.b64encode(png).decode("ascii")
    return (
        f"\x1b]1337;File=inline=1;size={len(png)};"
        f"width={columns};height={rows};preserveAspectRatio=false:{encoded}\x07"
    ).encode("ascii")


def sixel_bytes(width: int, height: int, rgba: bytes | bytearray, *, levels: int = 4) -> bytes:
    """Encode the canvas as sixel; height is padded up to a multiple of 6.

    Pixels with alpha < 50% stay unpainted so the terminal background shows
    through. Colors are quantised to ``levels`` steps per channel (with the
    top bin pinned at 255) so anti-aliased gradients stay compact.
    """
    bins = max(levels, 2)
    stride = width * 4
    six_rows = (height + 5) // 6

    entries: list[tuple[int, int, int]] = []
    entry_index: dict[tuple[int, int, int], int] = {}

    def entry_for(r: int, g: int, b: int) -> int:
        key = (r, g, b)
        index = entry_index.get(key)
        if index is None:
            index = len(entries)
            entries.append(key)
            entry_index[key] = index
        return index

    def quantize(channel: int) -> int:
        if bins <= 1:
            return channel
        step = 255 // (bins - 1)
        return min(round(channel / step) * step, 255)

    # Pass 1: per band, per color index, per column → a sixel char whose six
    # low bits are the painted pixels. Every color's stream carries one char
    # per column ('?' for "no pixels here") so the colours stay aligned on
    # the x axis — a missing leading column shifts the whole image left.
    bands: list[dict[int, list[str]]] = []
    for band in range(six_rows):
        columns: list[dict[int, int]] = []
        colors: set[int] = set()
        for x in range(width):
            seen: dict[int, int] = {}
            for sub in range(6):
                y = band * 6 + sub
                if y >= height:
                    continue
                offset = y * stride + x * 4
                if rgba[offset + 3] < 128:
                    continue
                code = entry_for(
                    quantize(rgba[offset]),
                    quantize(rgba[offset + 1]),
                    quantize(rgba[offset + 2]),
                )
                seen[code] = seen.get(code, 0) | (1 << sub)
            columns.append(seen)
            colors.update(seen)
        runs: dict[int, list[str]] = {index: [] for index in colors}
        for seen in columns:
            for index in colors:
                runs[index].append(chr(63 + seen.get(index, 0)))
        bands.append(runs)

    header = "\x1bPq"
    raster = f'"2;1;{width};{six_rows * 6}'
    palette = "".join(
        f"#{i};2;{r};{g};{b}" for i, (r, g, b) in enumerate(entries)
    )

    def emit_runs(chars: list[str]) -> str:
        pieces: list[str] = []
        i = 0
        while i < len(chars):
            j = i
            while j < len(chars) and chars[j] == chars[i]:
                j += 1
            run = j - i
            if run > 3:
                pieces.append(f"!{run}{chars[i]}")
            else:
                pieces.append(chars[i] * run)
            i = j
        return "".join(pieces)

    body: list[str] = []
    for band_number, per_color in enumerate(bands):
        if band_number:
            body.append("-")
        first = True
        for index, run in per_color.items():
            if any(ch != "?" for ch in run):
                if not first:
                    # sixel data is positional: after a colour's run the
                    # cursor sits at the right edge, so the next colour
                    # needs a carriage return or its pixels land off-grid
                    body.append("$")
                first = False
                body.append(f"#{index}" + emit_runs(run))
    return (header + raster + palette + "".join(body) + "\x1b\\").encode("ascii")


# ------------------------------------------------------------ chart render


def encode_placement(
    backend: str,
    png: bytes,
    rgba: bytes,
    pixel_w: int,
    pixel_h: int,
    *,
    columns: int,
    rows: int,
    image_id: int = 1,
) -> bytes:
    """Encode ``png`` for the backend, tagged with a stable kitty image id."""
    if backend == "kitty":
        return kitty_apc(png, columns=columns, rows=rows, image_id=image_id)
    if backend == "iterm2":
        return iterm2_inline(png, columns=columns, rows=rows)
    if backend == "sixel":
        return sixel_bytes(pixel_w, pixel_h, rgba)
    raise ValueError(f"unknown graphics backend {backend!r}")


def _nice_ceiling(value: float) -> float:
    """Round an axis maximum up to a human number (1/2/5 × 10**k)."""
    if value <= 0:
        return 1.0
    exponent = 10 ** math.floor(math.log10(value))
    for factor in (1.0, 2.0, 5.0, 10.0):
        if value <= exponent * factor:
            return exponent * factor
    return 10.0 * exponent


def _unpack(color: int, alpha: int = 255) -> tuple[int, int, int, int]:
    return ((color >> 16) & 255, (color >> 8) & 255, color & 255, alpha)


def chart_png(
    *,
    width: int,
    height: int,
    kind: str,  # "line" | "bar" | "stacked"
    series: Mapping[str, list[float | None]],
    palette: Sequence[int] = SERIES_PALETTE,
) -> tuple[bytes, bytes]:
    """Render series data as an anti-aliased transparent PNG + RGBA bitmap.

    Axis maxima use 1/2/5 ceilings, gridlines are faint white, lines get a
    soft 3px stroke. ``None`` values cut lines into segments and leave bar
    slots empty, matching the character-cell renderer's gap semantics.
    Words such as titles and tick labels are NOT drawn here: text renders as
    real terminal text around the image, where it stays crisp wherever the
    terminal lives.
    """
    canvas = Canvas(width, height)
    blank = bytes(width * height * 4)
    all_values = [value for values in series.values() for value in values if value is not None]
    if not all_values:
        return png_encode(width, height, blank), blank

    pad = 4.0
    y_max = _nice_ceiling(max(all_values))
    plot_w = float(width) - 2 * pad
    plot_h = float(height) - 2 * pad
    count = max(len(values) for values in series.values())
    if count <= 0:
        return png_encode(width, height, blank), blank
    step = plot_w / count

    grid = (200, 200, 200, 26)
    baseline = (200, 200, 200, 48)
    for quarter in range(1, 5):
        gy = pad + (1 - quarter / 4) * plot_h
        canvas.draw_line(pad, gy, width - pad, gy, grid, thickness=1)
    base_y = pad + plot_h
    canvas.draw_line(pad, base_y, width - pad, base_y, baseline, thickness=1)

    def y_of(value: float) -> float:
        return pad + (1 - value / y_max) * plot_h

    if kind == "line":
        for n, key in enumerate(series):
            color = _unpack(palette[n % len(palette)])
            open_run = False
            prev: tuple[float, float] | None = None
            for index, value in enumerate(series[key]):
                if value is None:
                    open_run = False
                    prev = None
                    continue
                point = (pad + index * step + step / 2, y_of(value))
                if open_run and prev is not None:
                    canvas.draw_line(prev[0], prev[1], point[0], point[1], color, thickness=3)
                open_run = True
                prev = point
    else:  # "bar" and "stacked" share the slot geometry
        stacked = kind == "stacked"
        heights = [0.0] * count
        for n, key in enumerate(series):
            color = _unpack(palette[n % len(palette)])
            for index, value in enumerate(series[key]):
                if value is None:
                    continue
                h_px = plot_h * value / y_max
                x = pad + index * step + step * 0.12
                slot_w = max(step * 0.76, 1.0)
                if stacked:
                    bottom = base_y - heights[index]
                    canvas.draw_rect(x, bottom - h_px, slot_w, h_px, color)
                    heights[index] += h_px
                else:
                    canvas.draw_rect(x, base_y - h_px, slot_w, h_px, color)
    bitmap = canvas.straight()  # one unpremultiply pass, shared by both outputs
    return png_encode(width, height, bitmap), bitmap


def test_chart_png(width: int = 800, height: int = 200) -> tuple[bytes, bytes]:
    """A self-test chart (PNG + RGBA) used by ``toknt graphics`` and the
    TUI's ``g`` key — one image, so both paths can be compared directly."""
    return chart_png(
        width=width,
        height=height,
        kind="line",
        series={
            "series one": [3, 7, 4, 9, 6, 11, 8],
            "series two": [1, 2, 6, 3, 8, 5, 10],
        },
    )


def solid_fill(*, width: int, height: int, color: int) -> tuple[bytes, bytes]:
    """An opaque single-color image used to erase stale sixel/iTerm2 art."""
    canvas = Canvas(width, height)
    canvas.draw_rect(0, 0, float(width), float(height), _unpack(color, 255))
    bitmap = canvas.straight()
    return png_encode(width, height, bitmap), bitmap


def pie_png(
    *,
    width: int,
    height: int,
    weights: Sequence[float],
    palette: Sequence[int] = SERIES_PALETTE,
    hole: float = 0.45,
) -> tuple[bytes, bytes]:
    """A donut chart of relative weights as an anti-aliased transparent PNG.

    Slices follow :data:`SERIES_PALETTE` order, matching every other chart's
    series colors. Wedge and radial edges are blended over ~1px; the bitmap
    stays premultiplied like every other Canvas product. Returns the encoded
    PNG and the straight RGBA bitmap (sixel input), same contract as
    :func:`chart_png`.
    """
    canvas = Canvas(width, height)
    blank = bytes(width * height * 4)
    clean = [float(value) for value in weights if isinstance(value, (int, float)) and value > 0]
    if not clean or sum(clean) <= 0:
        return png_encode(width, height, blank), blank

    total = sum(clean)
    fractions = [value / total for value in clean]
    outer = min(width, height) / 2 - 1
    _draw_pie(canvas, width / 2, height / 2, outer, outer * float(hole), fractions, palette)
    bitmap = canvas.straight()
    return png_encode(width, height, bitmap), bitmap


def _draw_pie(
    canvas: Canvas,
    cx: float,
    cy: float,
    outer: float,
    inner: float,
    fractions: Sequence[float],
    palette: Sequence[int],
    *,
    hole: float = 0.0,
) -> None:
    """Paint one donut into ``canvas`` (premultiplied per-pixel compositing)."""
    if inner <= 0:
        inner = outer * float(hole)
    buf = canvas.buf
    stride = canvas.stride
    width = canvas.width
    height = canvas.height

    # Wedge boundaries in [0, 2pi); slices start at 12 o'clock like most tools.
    boundaries = [0.0]
    angle = 0.0
    for fraction in fractions:
        angle += float(fraction) * 2 * math.pi
        boundaries.append(angle)
    boundaries[-1] = 2 * math.pi

    y0 = max(int(cy - outer) - 1, 0)
    y1 = min(int(cy + outer) + 2, height)
    x0 = max(int(cx - outer) - 1, 0)
    x1 = min(int(cx + outer) + 2, width)
    for y in range(y0, y1):
        row = y * stride
        dy = y + 0.5 - cy
        for x in range(x0, x1):
            dx = x + 0.5 - cx
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > outer or dist < inner:
                continue
            theta = math.atan2(dy, dx) % (2 * math.pi)
            # slice index: last boundary <= theta (last slice wraps at 2pi)
            lo, hi = 0, len(boundaries) - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if boundaries[mid] <= theta:
                    lo = mid
                else:
                    hi = mid - 1
            slice_index = min(lo, len(fractions) - 1)
            color = palette[slice_index % len(palette)]
            cr, cg, cb = (color >> 16) & 255, (color >> 8) & 255, color & 255

            # radial coverage (soft outer/inner rim)
            coverage = min(1.0, outer - dist + 0.5, dist - inner + 0.5)
            # angular coverage: blend with the neighboring slice near boundaries
            lower_gap = theta - boundaries[slice_index]
            upper_gap = boundaries[slice_index + 1] - theta
            edge_gap = min(lower_gap, upper_gap)
            edge_pixels = edge_gap * dist if dist > 0 else 0.0
            if edge_pixels < 1.0:
                neighbor = (
                    slice_index - 1
                    if lower_gap < upper_gap
                    else slice_index + 1
                ) % len(fractions)
                other = palette[neighbor % len(palette)]
                orr, ogg, obb = (other >> 16) & 255, (other >> 8) & 255, other & 255
                blend = 1.0 - max(edge_pixels, 0.0)
                cr = int(cr * (1 - blend) + orr * blend)
                cg = int(cg * (1 - blend) + ogg * blend)
                cb = int(cb * (1 - blend) + obb * blend)
            if coverage <= 0:
                continue
            alpha = int(255 * max(0.0, min(coverage, 1.0)))
            if alpha <= 0:
                continue
            base = row + x * 4
            # premultiplied compositing onto whatever is already there
            inv = 255 - alpha
            buf[base] = (cr * alpha + buf[base] * inv) // 255
            buf[base + 1] = (cg * alpha + buf[base + 1] * inv) // 255
            buf[base + 2] = (cb * alpha + buf[base + 2] * inv) // 255
            buf[base + 3] = (alpha * 255 + buf[base + 3] * inv) // 255


# ------------------------------------------------------------ terminal
# queries


def _read_reply(timeout: float) -> bytes:
    """Read one terminal reply from stdin (raw mode assumed)."""
    deadline = time.monotonic() + timeout
    reply = bytearray()
    stdin_fd = sys.stdin.fileno()
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        readable, _, _ = select.select([stdin_fd], [], [], max(remaining, 0.01))
        if not readable:
            break
        try:
            chunk = os.read(stdin_fd, 256)
        except OSError:
            break
        if not chunk:
            break
        reply += chunk
        # XTWINOPS replies end with 't'; OSC replies with BEL or ESC \.
        if b"t" in reply or b"\x07" in reply or reply.endswith(b"\x1b\\"):
            break
    return bytes(reply)


_CELL_REPLY = re.compile(r"\x1b\[6;(\d+);(\d+)t")
_RGB_REPLY = re.compile(r"rgb:([0-9a-fA-F]{1,4})/([0-9a-fA-F]{1,4})/([0-9a-fA-F]{1,4})")
_FLUSH = getattr(termios, "TCIFLUSH", 0)


def _scaled_channel(text: str) -> int:
    """X11 color component (1-4 hex digits) scaled to 0..255."""
    top = (1 << (4 * len(text))) - 1  # each response width has its own max
    return int(text, 16) * 255 // top


def measure_cell(timeout: float = 0.25) -> tuple[int, int]:
    """(cell_width, cell_height) in pixels via XTWINOPS 18; (10, 20) fallback.

    The reply is range-checked: a terminal that answers with a nonsense size
    (or a stray reply that happens to match the pattern) must not turn the
    next chart into a gigapixel image that freezes the UI.
    """
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return 10, 20
    try:
        termios.tcflush(0, _FLUSH)
    except (termios.error, OSError):
        pass
    try:
        os.write(1, b"\x1b[16t")
    except OSError:
        return 10, 20
    match = _CELL_REPLY.match(_read_reply(timeout).decode("ascii", "replace"))
    if match:
        try:
            height, width = int(match.group(1)), int(match.group(2))
        except ValueError:
            height = width = 0
        if 2 <= width <= 40 and 4 <= height <= 80:
            return width, height
    return 10, 20


def terminal_background(timeout: float = 0.25) -> tuple[str, int | None]:
    """(mode, rgb) via the OSC 11 background query; dark on timeout.

    ``rgb`` is an 0xRRGGBB int or None when the terminal did not answer —
    exact enough to paint opaque over stale graphics for erasure.
    """
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return "dark", None
    try:
        termios.tcflush(0, _FLUSH)
    except (termios.error, OSError):
        pass
    try:
        os.write(1, b"\x1b]11;?\x1b\\")
    except OSError:
        return "dark", None
    match = _RGB_REPLY.search(_read_reply(timeout).decode("ascii", "replace"))
    if not match:
        return "dark", None
    rgb = [_scaled_channel(text) for text in match.groups()]
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    packed = (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]
    return ("light" if luminance >= 128 else "dark"), packed


# ------------------------------------------------------------ raster canvas


class Canvas:
    """A premultiplied-RGBA surface, composited "over".

    Keeping everything premultiplied makes alpha math blameless: painting is
    ``dst = src + dst*(255-src_a)/255`` per channel. Anti-aliasing comes from
    the soft-edged brush itself rather than supersampling, so the output pass
    is a cheap unpremultiply instead of a 2x2 downscale.
    """

    SCALE = 1

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.big_w = width * self.SCALE
        self.big_h = height * self.SCALE
        self.stride = self.big_w * 4
        self.buf = bytearray(self.big_h * self.stride)

    # -- primitives ----------------------------------------------------------

    def draw_line(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        color: Sequence[int],
        *,
        thickness: float = 2.0,
    ) -> None:
        """A stroke with a soft falloff, in small-pixel coordinates.

        Solid strokes stamp an overlapping disc brush (full-alpha over
        compositing is idempotent, so repeat stamps compound to full
        coverage). Translucent strokes (gridlines, faded series) rasterize
        by exact distance-to-segment in one pass instead: translucent stamps
        would compound toward opacity, and any paint-once heuristic leaves
        noisy halos. One sample per pixel keeps the declared alpha exact.
        """
        radius = max(thickness, 1.0) * self.SCALE / 2
        bx0, by0 = x0 * self.SCALE, y0 * self.SCALE
        bx1, by1 = x1 * self.SCALE, y1 * self.SCALE
        dx = bx1 - bx0
        dy = by1 - by0
        length = (dx * dx + dy * dy) ** 0.5
        cr, cg, cb = color[0], color[1], color[2]
        ca = color[3]
        if ca <= 0:
            return
        buf = self.buf
        stride = self.stride
        big_w = self.big_w
        big_h = self.big_h

        if ca == 255:
            steps = max(int(length / max(radius * 1.25, 0.35)), 1)
            for step in range(steps + 1):
                t = step / steps
                px = bx0 + dx * t
                py = by0 + dy * t
                lo_x = max(int(px - radius), 0)
                hi_x = min(int(px + radius) + 2, big_w)
                lo_y = max(int(py - radius), 0)
                hi_y = min(int(py + radius) + 2, big_h)
                for y in range(lo_y, hi_y):
                    dyv = y - py
                    dist_y = dyv * dyv
                    row = y * stride
                    for x in range(lo_x, hi_x):
                        dxv = x - px
                        dd = dxv * dxv + dist_y
                        if dd >= radius * radius:
                            continue
                        sa = int(255 * (radius - dd**0.5) / max(radius, 1e-6))
                        if sa <= 0:
                            continue
                        src_inv = 255 - sa
                        base = row + x * 4
                        value = (cr * sa + buf[base] * src_inv) // 255
                        buf[base] = value if value < 255 else 255
                        value = (cg * sa + buf[base + 1] * src_inv) // 255
                        buf[base + 1] = value if value < 255 else 255
                        value = (cb * sa + buf[base + 2] * src_inv) // 255
                        buf[base + 2] = value if value < 255 else 255
                        buf[base + 3] = 255
            return

        #-- translucent: exact single-pass coverage over the segment bbox ------
        core = radius - 1.0
        if core < 0.0:
            core = 0.0
        falloff = max(radius - core, 1e-6)
        len2 = dx * dx + dy * dy
        lo_x = max(int(min(bx0, bx1) - radius), 0)
        hi_x = min(int(max(bx0, bx1) + radius) + 2, big_w)
        lo_y = max(int(min(by0, by1) - radius), 0)
        hi_y = min(int(max(by0, by1) + radius) + 2, big_h)
        for y in range(lo_y, hi_y):
            row = y * stride
            vy = y - by0
            for x in range(lo_x, hi_x):
                if len2:
                    vx = x - bx0
                    t = (vx * dx + vy * dy) / len2
                    if t < 0.0:
                        t = 0.0
                    elif t > 1.0:
                        t = 1.0
                    ref_x = bx0 + dx * t
                    ref_y = by0 + dy * t
                else:
                    ref_x, ref_y = bx0, by0
                dd = ((x - ref_x) ** 2 + (y - ref_y) ** 2) ** 0.5
                if dd >= radius:
                    continue
                if dd <= core:
                    sa = ca
                else:
                    sa = int(ca * (radius - dd) / falloff)
                    if sa <= 0:
                        continue
                src_inv = 255 - sa
                base = row + x * 4
                value = (cr * sa + buf[base] * src_inv) // 255
                buf[base] = value if value < 255 else 255
                value = (cg * sa + buf[base + 1] * src_inv) // 255
                buf[base + 1] = value if value < 255 else 255
                value = (cb * sa + buf[base + 2] * src_inv) // 255
                buf[base + 2] = value if value < 255 else 255
                dst_a = (sa * 255 + buf[base + 3] * src_inv) // 255
                buf[base + 3] = dst_a if dst_a < 255 else 255

    def draw_rect(self, x: float, y: float, w: float, h: float, color: Sequence[int]) -> None:
        bx = int(x * self.SCALE)
        by = int(y * self.SCALE)
        bw = max(1, int(w * self.SCALE))
        bh = max(1, int(h * self.SCALE))
        cr, cg, cb = color[0], color[1], color[2]
        src_a = color[3]
        if src_a <= 0:
            return
        inv = 255 - src_a
        buf = self.buf
        for yy in range(max(by, 0), min(by + bh, self.big_h)):
            base = yy * self.stride
            for xx in range(max(bx, 0), min(bx + bw, self.big_w)):
                offset = base + xx * 4
                value = (cr * src_a + buf[offset] * inv) // 255
                buf[offset] = value if value < 255 else 255
                value = (cg * src_a + buf[offset + 1] * inv) // 255
                buf[offset + 1] = value if value < 255 else 255
                value = (cb * src_a + buf[offset + 2] * inv) // 255
                buf[offset + 2] = value if value < 255 else 255
                value = (src_a * 255 + buf[offset + 3] * inv) // 255
                buf[offset + 3] = value if value < 255 else 255

    def _straight(self) -> bytes:
        """The bitmap unpremultiplied (shared by png() and rgba())."""
        out = bytearray(self.buf)
        for base in range(0, len(out), 4):
            alpha = out[base + 3]
            if alpha == 255:
                continue
            if alpha == 0:
                out[base] = out[base + 1] = out[base + 2] = 0
                continue
            for i in range(3):
                out[base + i] = min(out[base + i] * 255 // alpha, 255)
        return bytes(out)

    def png(self) -> bytes:
        """Encode the final PNG."""
        return png_encode(self.width, self.height, self._straight())

    def straight(self) -> bytes:
        """The bitmap, unpremultiplied (sixel input)."""
        return self._straight()

    def rgba(self) -> bytes:
        """The bitmap, unpremultiplied ('rgba' kept for call compatibility)."""
        return self._straight()
