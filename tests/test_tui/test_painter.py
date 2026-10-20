"""Cell painter: frame diffing, wide chars, SGR encoding."""

from __future__ import annotations

from toknt.tui.painter import (
    BOLD,
    REVERSE,
    Attr,
    Cell,
    Painter,
    _encode_attrs,
    _position,
)


def _painter(rows: int = 6, columns: int = 40) -> Painter:
    painter = Painter()
    painter.resize(rows, columns)
    painter.frame()  # throwaway: establishes an aligned snapshot
    return painter


def test_first_frame_paints_everything_then_quiets() -> None:
    painter = Painter()
    painter.resize(4, 20)
    first = painter.frame()
    assert b"\x1b[1;1H" in first
    assert painter.frame() == b""  # snapshot aligned: no-op second frame


def test_single_cell_change_emits_one_run() -> None:
    painter = _painter()
    painter.text(2, 3, "hi", Attr(0x00FF00, None, BOLD))
    frame = painter.frame()
    assert b"\x1b[3;4H" in frame
    assert b"38;2;0;255;0" in frame  # truecolor fg
    assert b"\x1b[1;" in frame  # bold on (with reverse/underline explicitly cleared)
    assert b"hi" in frame
    # unchanged rest of the screen stays quiet
    assert frame.count(b"\x1b[") == 3  # CUP + SGR pair per run + trailing reset
    assert painter.frame() == b""


def test_plain_runs_explicitly_clear_state_from_previous_runs() -> None:
    """Every run carries absolute SGR state: on- or off-codes for each attr.

    Regression: the reversed hint bar previously leaked into later runs,
    washing most of the screen with a grey highlight.
    """
    plain = _encode_attrs((None, None, 0))
    assert plain.startswith("\x1b[22;24;27;39;49m")  # bold+dim/underline/reverse off
    bold = _encode_attrs((None, None, BOLD))
    assert bold.startswith("\x1b[1;24;27;39;49m")
    dim = _encode_attrs((None, None, 8))
    assert dim.startswith("\x1b[2;24;27;39;49m")
    reverse = _encode_attrs((None, None, 4))
    assert reverse.startswith("\x1b[22;24;7;39;49m")


def test_frame_ends_with_a_reset_for_outside_paint() -> None:
    painter = _painter()
    painter.text(0, 0, "hello")
    frame = painter.frame()
    assert frame.endswith(b"\x1b[0m")


def test_clear_writes_fresh_blank_run() -> None:
    painter = _painter()
    painter.text(0, 0, "aaaaaaaa")
    painter.frame()
    painter.clear(0, 0, 8, 1)
    frame = painter.frame()
    assert b"\x1b[1;1H" in frame
    assert b"        " in frame  # eight blanks erase the text


def test_wide_char_occupies_two_cells_without_leaking_text() -> None:
    painter = _painter()
    painter.text(0, 0, "汉")
    frame = painter.frame()
    assert "汉".encode() in frame
    assert frame.count("汉".encode()) == 1  # no phantom pad char
    # a full-width overwrite then erase stays aligned
    painter.text(0, 1, "x")
    painter.frame()
    painter.clear(0, 0, 40, 1)
    assert painter.frame() != b""


def test_reverse_attr_for_cursor_style_cells() -> None:
    painter = _painter()
    painter.text(1, 1, " ", Attr(None, None, REVERSE))
    frame = painter.frame()
    assert b"[22;24;7;39;49m" in frame  # reverse on; bold/dim + underline cleared


def test_position_and_attrs_helpers() -> None:
    assert _position(0, 0) == "\x1b[1;1H"
    attrs = _encode_attrs((0x112233, 0x445566, BOLD))
    assert "38;2;17;34;51" in attrs
    assert "48;2;68;85;102" in attrs
    assert attrs.startswith("\x1b[1;24;27;")  # bold first, off-codes explicit


def test_paint_beyond_edges_is_clipped_not_crashing() -> None:
    painter = _painter(rows=3, columns=10)
    painter.text(-2, 0, "out of bounds", Attr(None, None, 0))
    painter.text(0, 8, "0123456789", Attr(None, None, 0))  # overflows → clipped
    assert painter.frame() != b""  # wrote what fit without raising


def test_buffer_defaults_are_blank_cells() -> None:
    painter = _painter()
    painter.clear(0, 0, painter.columns, 1)
    cell = painter.buffer[0][0]
    assert cell == Cell(" ", None, None, 0)
