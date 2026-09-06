from __future__ import annotations

import time

import numpy as np

from zerohero.config import Config
from zerohero.events import Frame, Hand, Landmark
from zerohero.ui.overlay import Overlay, OverlayState


def _state() -> OverlayState:
    return OverlayState(
        mode="guitar",
        chords=["Am", "G", "C", "F"],
        index=1,
        vibe=0.6,
        tracks={
            "left": (0.3, 0.5, -0.2, 0.1, True),
            "right": (0.7, 0.4, 1.5, -0.5, False),
        },
        last_event="strum down",
        last_event_t=0.0,
        fps=29.5,
        link="lead (2 followers)",
    )


def _frame_with_hand() -> Frame:
    landmarks = [Landmark(x=0.3 + i * 0.01, y=0.5 + i * 0.01, z=0.0) for i in range(21)]
    return Frame(t=0.1, width=640, height=480, hands=[Hand(side="left", score=0.9, landmarks=landmarks)])


def test_draw_on_none_image_synthesizes_canvas() -> None:
    overlay = Overlay(Config())
    frame = Frame(t=0.0, width=320, height=240, hands=[], image=None)
    out = overlay.draw(None, frame, _state())
    assert out.shape == (240, 320, 3)


def test_draw_on_real_image_with_hand() -> None:
    overlay = Overlay(Config())
    frame = _frame_with_hand()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    out = overlay.draw(image, frame, _state())
    assert out.shape == (480, 640, 3)
    # Something was actually drawn, not a silent no-op.
    assert out.sum() > 0


def test_draw_performance_budget() -> None:
    overlay = Overlay(Config())
    frame = _frame_with_hand()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    state = _state()

    n = 50
    start = time.perf_counter()
    for _ in range(n):
        overlay.draw(image.copy(), frame, state)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / n) * 1000
    assert avg_ms < 5.0, f"overlay draw averaged {avg_ms:.2f}ms over {n} calls"
