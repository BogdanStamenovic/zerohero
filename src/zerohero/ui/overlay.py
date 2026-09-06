"""Overlay drawing (hand skeleton, chord strip, meters, HUD) and the display window.

Kept to cheap cv2 primitives only -- no per-pixel work -- since this runs on
the main loop every frame alongside gesture processing and music. See
ARCHITECTURE.md "Pipeline" and "engine" for the key bindings shown in the
help corner.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from zerohero.config import Config
from zerohero.events import Frame, Side

# 21-point hand skeleton, wrist (0) rooted. MediaPipe's own `solutions` module
# (which used to export this) isn't part of the tasks API used here, so the
# topology is just hardcoded against the landmark indices in events.py.
_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

_COLOR: dict[str, tuple[int, int, int]] = {"left": (255, 140, 0), "right": (0, 200, 255)}  # BGR
_DEFAULT_COLOR = (200, 200, 200)
_EVENT_FLASH_S = 0.4
# Pixels per hand-width/s of palm velocity, purely for arrow visibility.
_VELOCITY_ARROW_SCALE = 22.0


@dataclass
class OverlayState:
    mode: str
    chords: list[str]
    index: int
    vibe: float
    # side -> (palm_x, palm_y, vx, vy, closed), palm in 0..1 image coords.
    tracks: dict[str, tuple[float, float, float, float, bool]] = field(default_factory=dict)
    last_event: str = ""
    last_event_t: float = 0.0
    fps: float = 0.0
    link: str = ""


class Overlay:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def draw(self, image: np.ndarray | None, frame: Frame, state: OverlayState) -> np.ndarray:
        if image is None:
            image = np.zeros((frame.height, frame.width, 3), dtype=np.uint8)
        h, w = image.shape[:2]

        for hand in frame.hands:
            track = state.tracks.get(hand.side)
            closed = track[4] if track is not None else False
            self._draw_hand(image, hand, closed, w, h)
        for side, (px, py, vx, vy, _closed) in state.tracks.items():
            self._draw_velocity_arrow(image, side, px, py, vx, vy, w, h)

        self._draw_chord_strip(image, state, w, h)
        self._draw_vibe_bar(image, state, w, h)
        self._draw_hud(image, frame, state, w, h)
        return image

    def _draw_hand(self, image: np.ndarray, hand, closed: bool, w: int, h: int) -> None:
        color = _COLOR.get(hand.side, _DEFAULT_COLOR)
        pts = [(int(p.x * w), int(p.y * h)) for p in hand.landmarks]
        for a, b in _CONNECTIONS:
            cv2.line(image, pts[a], pts[b], color, 2, cv2.LINE_AA)
        radius, fill = (6, -1) if closed else (4, 2)
        for p in pts:
            cv2.circle(image, p, radius, color, fill, cv2.LINE_AA)

    def _draw_velocity_arrow(
        self, image: np.ndarray, side: Side, px: float, py: float, vx: float, vy: float, w: int, h: int
    ) -> None:
        color = _COLOR.get(side, _DEFAULT_COLOR)
        origin = (int(px * w), int(py * h))
        tip = (int(px * w + vx * _VELOCITY_ARROW_SCALE), int(py * h + vy * _VELOCITY_ARROW_SCALE))
        cv2.arrowedLine(image, origin, tip, color, 2, cv2.LINE_AA, tipLength=0.3)

    def _draw_chord_strip(self, image: np.ndarray, state: OverlayState, w: int, h: int) -> None:
        if not state.chords:
            return
        y = h - 30
        slot = w / len(state.chords)
        for i, chord in enumerate(state.chords):
            x0, x1 = int(i * slot), int((i + 1) * slot)
            bg = (0, 160, 0) if i == state.index else (60, 60, 60)
            cv2.rectangle(image, (x0 + 2, y - 20), (x1 - 2, y + 10), bg, -1)
            cv2.putText(image, chord, (x0 + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_vibe_bar(self, image: np.ndarray, state: OverlayState, w: int, h: int) -> None:
        bar_w, bar_h = 24, int(h * 0.5)
        x0, y0 = w - bar_w - 12, int(h * 0.25)
        cv2.rectangle(image, (x0, y0), (x0 + bar_w, y0 + bar_h), (60, 60, 60), -1)
        fill_h = int(bar_h * max(0.0, min(1.0, state.vibe)))
        cv2.rectangle(image, (x0, y0 + bar_h - fill_h), (x0 + bar_w, y0 + bar_h), (0, 100, 255), -1)
        cv2.rectangle(image, (x0, y0), (x0 + bar_w, y0 + bar_h), (200, 200, 200), 1)

    def _draw_hud(self, image: np.ndarray, frame: Frame, state: OverlayState, w: int, h: int) -> None:
        link_text = state.link if state.link else "none"
        cv2.putText(
            image,
            f"{state.fps:4.1f} fps  link: {link_text}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # frame.t, not wall clock: keeps the flash timing correct under replay.
        if state.last_event and (frame.t - state.last_event_t) < _EVENT_FLASH_S:
            cv2.putText(image, state.last_event, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        help_text = "q quit  n/p chord  space strum  r reset"
        (tw, _th), _ = cv2.getTextSize(help_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        help_color = (150, 150, 150)
        cv2.putText(image, help_text, (w - tw - 10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, help_color, 1, cv2.LINE_AA)


class Window:
    def __init__(self, title: str) -> None:
        self.title = title
        self._opened = False

    def show(self, image: np.ndarray) -> None:
        if not self._opened:
            cv2.namedWindow(self.title, cv2.WINDOW_AUTOSIZE)
            self._opened = True
        cv2.imshow(self.title, image)

    def poll_key(self) -> str | None:
        key = cv2.waitKey(1) & 0xFF
        if key == 0xFF:  # cv2's sentinel for "no key" (raw -1, masked)
            return None
        if key == 27:
            return "esc"
        return chr(key) if 32 <= key < 127 else None

    def close(self) -> None:
        if self._opened:
            cv2.destroyWindow(self.title)
            self._opened = False
