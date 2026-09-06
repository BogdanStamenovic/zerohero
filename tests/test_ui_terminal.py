from __future__ import annotations

import io
import sys
import time

from zerohero.events import Frame, Hand, Landmark
from zerohero.ui.keys import KeyReader
from zerohero.ui.overlay import OverlayState
from zerohero.ui.terminal import render

COLS, ROWS = 80, 24


def _hand(side: str, cx: float, cy: float) -> Hand:
    # 21 plausible landmarks fanned out from a palm centre, roughly hand-shaped
    # enough to exercise every connection without needing real MediaPipe data.
    landmarks = [Landmark(x=cx, y=cy, z=0.0)]
    for finger in range(5):
        base_x = cx + (finger - 2) * 0.03
        for joint in range(4):
            landmarks.append(Landmark(x=base_x, y=cy - 0.05 - joint * 0.04, z=0.0))
    return Hand(side=side, score=0.95, landmarks=landmarks)


def _state(index: int = 1) -> OverlayState:
    return OverlayState(
        mode="guitar",
        chords=["Am", "G", "C", "F"],
        index=index,
        vibe=0.5,
        tracks={
            "left": (0.3, 0.5, -0.4, 0.2, True),
            "right": (0.7, 0.4, 1.2, -0.6, False),
        },
        last_event="strum down",
        last_event_t=0.0,
        fps=29.5,
        link="lead (1 follower)",
    )


def _two_hand_frame(t: float = 0.05) -> Frame:
    return Frame(t=t, width=640, height=480, hands=[_hand("left", 0.3, 0.5), _hand("right", 0.7, 0.45)])


def _visible_width(line: str) -> int:
    # No ANSI is emitted with color=False, so plain len() already is the
    # visible width -- this helper exists to make that assumption explicit.
    return len(line)


def test_render_shape_and_width() -> None:
    lines = render(_two_hand_frame(), _state(), COLS, ROWS, color=False)
    assert len(lines) == ROWS
    for line in lines:
        assert _visible_width(line) == COLS


def test_render_no_ansi_when_color_false() -> None:
    lines = render(_two_hand_frame(), _state(), COLS, ROWS, color=False)
    assert all("\x1b" not in line for line in lines)


def test_canvas_contains_braille() -> None:
    lines = render(_two_hand_frame(), _state(), COLS, ROWS, color=False)
    text = "\n".join(lines)
    assert any(0x2800 <= ord(ch) <= 0x28FF for ch in text)


def test_chord_strip_brackets_current_chord() -> None:
    state = _state(index=1)  # chords[1] == "G"
    lines = render(_two_hand_frame(), state, COLS, ROWS, color=False)
    assert any("[ G ]" in line for line in lines)


def test_ascii_fallback_is_pure_ascii() -> None:
    lines = render(_two_hand_frame(), _state(), COLS, ROWS, color=False, ascii_mode=True)
    text = "\n".join(lines)
    assert all(ord(ch) < 128 for ch in text)
    # and no braille leaked in either
    assert not any(0x2800 <= ord(ch) <= 0x28FF for ch in text)


def test_render_performance_budget() -> None:
    frame = _two_hand_frame()
    state = _state()
    n = 100
    start = time.perf_counter()
    for _ in range(n):
        render(frame, state, COLS, ROWS, color=True)
    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / n) * 1000
    assert avg_ms < 8.0, f"average render took {avg_ms:.3f} ms"


def test_render_no_hands() -> None:
    frame = Frame(t=0.0, width=640, height=480, hands=[])
    state = OverlayState(mode="guitar", chords=[], index=0, vibe=0.0, tracks={})
    lines = render(frame, state, COLS, ROWS, color=False)
    assert len(lines) == ROWS
    assert all(len(line) == COLS for line in lines)


def test_render_image_none() -> None:
    frame = Frame(t=0.0, width=640, height=480, hands=[_hand("left", 0.5, 0.5)], image=None)
    lines = render(frame, _state(), COLS, ROWS, color=False)
    assert len(lines) == ROWS


def test_render_small_terminal_does_not_crash() -> None:
    lines = render(_two_hand_frame(), _state(), 20, 6, color=False)
    assert len(lines) == 6
    assert all(len(line) == 20 for line in lines)


# ---- KeyReader ------------------------------------------------------------


def test_key_reader_non_tty_returns_none(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("x"))
    with KeyReader() as kr:
        assert kr.poll() is None


def test_key_reader_context_manager_does_not_raise(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    kr = KeyReader()
    kr.__enter__()
    kr.poll()
    kr.__exit__(None, None, None)
