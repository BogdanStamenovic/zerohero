"""Terminal UI: a live braille-canvas hand-tracking view, no GUI required.

Renders the same information as `ui.overlay.Overlay` (see ARCHITECTURE.md
"Pipeline", "Coordinate conventions", "vision", "engine") but onto a terminal
grid instead of a cv2 window, so a headless SSH session can still show the
skeleton, chord strip, vibe meter and HUD. `TerminalView` mirrors the OpenCV
`Window` API (`start`/`draw`/`poll_key`/`close`) so `engine/app.py` can pick
either at runtime.

Landmarks are `x, y` normalised 0..1 with y down, exactly as documented in
ARCHITECTURE.md's "Coordinate conventions" -- no pixel image is needed to draw
them, which is what makes this backend work over a headless connection.
"""

from __future__ import annotations

import math
import shutil
import sys
import time

from zerohero.config import Config
from zerohero.events import Frame
from zerohero.ui.keys import KeyReader, enable_vt
from zerohero.ui.overlay import OverlayState

# 21-point hand skeleton, matching events.py's landmark indices. Kept
# independent of overlay.py's cv2 connection list (that one is a simplified
# stand-in topology); this is the full MediaPipe layout including palm ties,
# which read better at braille-dot resolution.
_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # index
    (9, 10), (10, 11), (11, 12),  # middle
    (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (5, 9), (9, 13), (13, 17),  # palm
)

# Braille cell = 2 dots wide x 4 dots tall, U+2800 + this bitmask.
# Bit layout is fixed by the Unicode block, not a free choice:
#   (0,0) (1,0)      0x01 0x08
#   (0,1) (1,1)  ->  0x02 0x10
#   (0,2) (1,2)      0x04 0x20
#   (0,3) (1,3)      0x40 0x80
_BRAILLE_BASE = 0x2800
_BRAILLE_BITS: dict[tuple[int, int], int] = {
    (0, 0): 0x01, (0, 1): 0x02, (0, 2): 0x04, (1, 0): 0x08,
    (1, 1): 0x10, (1, 2): 0x20, (0, 3): 0x40, (1, 3): 0x80,
}

_TRUECOLOR: dict[str, tuple[int, int, int]] = {
    "left": (0, 230, 230),  # cyan
    "right": (255, 140, 0),  # orange
    "closed": (230, 0, 230),  # magenta -- overrides side color while fisted
}

HUD_LINES = 3
_VIBE_BAR_LEN = 12
_MAX_FPS = 30.0
_RESIZE_POLL_S = 0.5
# Dots per hand-width/s for the palm velocity line, and how far (in dots) it
# is allowed to stretch so a fast swipe doesn't paint clear off the canvas.
_VELOCITY_DOT_SCALE = 6.0
_VELOCITY_CAP_FRAC = 0.3


def _supports_braille() -> bool:
    enc = sys.stdout.encoding or "ascii"
    try:
        "⣿".encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def _letterbox(dot_w: int, dot_h: int, aspect: float) -> tuple[float, float, float, float]:
    """Largest `aspect`-shaped rect centred in a dot_w x dot_h box.

    A terminal cell is roughly half as wide as it is tall, and braille packs
    2x4 dots into that cell, so a dot is close to square -- the aspect ratio
    can be applied directly in dot space without a further correction factor.
    """
    if dot_w <= 0 or dot_h <= 0:
        return 0.0, 0.0, float(max(dot_w, 0)), float(max(dot_h, 0))
    if dot_w / dot_h > aspect:
        h = float(dot_h)
        w = dot_h * aspect
        return (dot_w - w) / 2, 0.0, w, h
    w = float(dot_w)
    h = dot_w / aspect
    return 0.0, (dot_h - h) / 2, w, h


def _velocity_tip(px: float, py: float, vx: float, vy: float, dot_w: int, dot_h: int) -> tuple[float, float]:
    tx, ty = vx * _VELOCITY_DOT_SCALE, vy * _VELOCITY_DOT_SCALE
    mag = math.hypot(tx, ty)
    cap = _VELOCITY_CAP_FRAC * min(dot_w, dot_h) if dot_w and dot_h else 0.0
    if cap > 0 and mag > cap:
        tx, ty = tx / mag * cap, ty / mag * cap
    return px + tx, py + ty


def _plot_point(
    bits: list[list[int]] | None,
    cell_char: list[list[str | None]],
    cell_color: list[list[str | None]],
    w: int,
    h: int,
    x: float,
    y: float,
    color: str,
    ascii_mode: bool,
    char: str,
) -> None:
    ix, iy = int(round(x)), int(round(y))
    if ascii_mode:
        if 0 <= ix < w and 0 <= iy < h:
            cell_char[iy][ix] = char
            cell_color[iy][ix] = color
        return
    cx, cy = ix // 2, iy // 4
    if 0 <= cx < w and 0 <= cy < h and bits is not None:
        bits[cy][cx] |= _BRAILLE_BITS[(ix % 2, iy % 4)]
        cell_color[cy][cx] = color


def _plot_line(
    bits: list[list[int]] | None,
    cell_char: list[list[str | None]],
    cell_color: list[list[str | None]],
    w: int,
    h: int,
    p0: tuple[float, float],
    p1: tuple[float, float],
    color: str,
    ascii_mode: bool,
    char: str,
) -> None:
    x0, y0 = int(round(p0[0])), int(round(p0[1]))
    x1, y1 = int(round(p1[0])), int(round(p1[1]))
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        _plot_point(bits, cell_char, cell_color, w, h, x, y, color, ascii_mode, char)
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def _plot_label(
    cell_char: list[list[str | None]],
    cell_color: list[list[str | None]],
    w: int,
    h: int,
    x_dot: float,
    y_dot: float,
    ascii_mode: bool,
    letter: str,
    color: str,
) -> None:
    cx = int(round(x_dot)) if ascii_mode else int(round(x_dot)) // 2
    cy = int(round(y_dot)) if ascii_mode else int(round(y_dot)) // 4
    cx = max(0, min(w - 1, cx - 1))  # one cell left of the wrist dot
    cy = max(0, min(h - 1, cy))
    if w > 0 and h > 0:
        cell_char[cy][cx] = letter
        cell_color[cy][cx] = color


def _row_to_string(chars: list[str], colors: list[str | None], color: bool) -> str:
    if not color:
        return "".join(chars)
    parts: list[str] = []
    cur: str | None = None
    for ch, col in zip(chars, colors, strict=True):
        if col != cur:
            if cur is not None:
                parts.append("\x1b[0m")
            if col is not None:
                r, g, b = _TRUECOLOR[col]
                parts.append(f"\x1b[38;2;{r};{g};{b}m")
            cur = col
        parts.append(ch)
    if cur is not None:
        parts.append("\x1b[0m")
    return "".join(parts)


def _render_canvas(
    frame: Frame, state: OverlayState, canvas_w: int, canvas_h: int, *, color: bool, ascii_mode: bool
) -> list[str]:
    if canvas_w <= 0 or canvas_h <= 0:
        return [" " * max(canvas_w, 0) for _ in range(max(canvas_h, 0))]

    dot_w = canvas_w if ascii_mode else canvas_w * 2
    dot_h = canvas_h if ascii_mode else canvas_h * 4
    bits: list[list[int]] | None = None if ascii_mode else [[0] * canvas_w for _ in range(canvas_h)]
    cell_char: list[list[str | None]] = [[None] * canvas_w for _ in range(canvas_h)]
    cell_color: list[list[str | None]] = [[None] * canvas_w for _ in range(canvas_h)]

    aspect = frame.width / frame.height if frame.width and frame.height else 4 / 3
    x0, y0, dw, dh = _letterbox(dot_w, dot_h, aspect)

    def to_dot(lx: float, ly: float) -> tuple[float, float]:
        return (x0 + lx * dw, y0 + ly * dh)

    for hand in frame.hands:
        track = state.tracks.get(hand.side)
        closed = bool(track[4]) if track is not None else False
        cname = "closed" if closed else hand.side
        pts = [to_dot(lm.x, lm.y) for lm in hand.landmarks]
        for a, b in _CONNECTIONS:
            if a < len(pts) and b < len(pts):
                _plot_line(bits, cell_char, cell_color, canvas_w, canvas_h, pts[a], pts[b], cname, ascii_mode, ".")
        for p in pts:
            _plot_point(bits, cell_char, cell_color, canvas_w, canvas_h, p[0], p[1], cname, ascii_mode, "o")
        if pts:
            label = ("L" if hand.side == "left" else "R") + ("!" if hand.gesture == "Closed_Fist" else "")
            _plot_label(cell_char, cell_color, canvas_w, canvas_h, pts[0][0], pts[0][1], ascii_mode, label, cname)

    for side, track in state.tracks.items():
        px, py, vx, vy, closed = track
        origin = to_dot(px, py)
        tip = _velocity_tip(origin[0], origin[1], vx, vy, dot_w, dot_h)
        vcname = "closed" if closed else side
        _plot_line(bits, cell_char, cell_color, canvas_w, canvas_h, origin, tip, vcname, ascii_mode, "#")

    rows: list[str] = []
    for cy in range(canvas_h):
        chars: list[str] = []
        colors: list[str | None] = []
        for cx in range(canvas_w):
            cell_label = cell_char[cy][cx]
            if cell_label is not None:
                chars.append(cell_label)
            elif ascii_mode:
                chars.append(" ")
            else:
                assert bits is not None
                b = bits[cy][cx]
                chars.append(chr(_BRAILLE_BASE + b) if b else " ")
            colors.append(cell_color[cy][cx])
        rows.append(_row_to_string(chars, colors, color))
    return rows


def _chord_line(state: OverlayState, width: int, *, color: bool) -> str:
    if not state.chords:
        return " " * width
    tokens = [f"[ {c} ]" if i == state.index else f"  {c}  " for i, c in enumerate(state.chords)]
    text = " ".join(tokens)
    start = end = -1
    if 0 <= state.index < len(state.chords):
        needle = f"[ {state.chords[state.index]} ]"
        pos = text.find(needle)
        if pos != -1:
            start, end = pos, pos + len(needle)
    text = text[:width].ljust(width)
    if color and start != -1 and start < width:
        end = min(end, width)
        text = text[:start] + f"\x1b[1;7m{text[start:end]}\x1b[0m" + text[end:]
    return text


def _vibe_line(frame: Frame, state: OverlayState, width: int, *, ascii_mode: bool) -> str:
    filled = int(round(max(0.0, min(1.0, state.vibe)) * _VIBE_BAR_LEN))
    fill_ch, empty_ch = ("#", "-") if ascii_mode else ("▮", "▯")
    left = f"vibe {fill_ch * filled}{empty_ch * (_VIBE_BAR_LEN - filled)}"
    right = state.last_event if state.last_event and (frame.t - state.last_event_t) < 0.6 else ""
    if right:
        pad = max(1, width - len(left) - len(right))
        line = left + " " * pad + right
    else:
        line = left
    return line[:width].ljust(width)


def _status_line(state: OverlayState, width: int) -> str:
    left = f"{state.fps:4.1f} fps  {state.mode}  link: {state.link or 'none'}"
    help_text = "q quit  n/p chord  space strum  r reset"
    pad = max(1, width - len(left) - len(help_text))
    line = left + " " * pad + help_text
    return line[:width].ljust(width)


def render(
    frame: Frame,
    state: OverlayState,
    cols: int,
    rows: int,
    *,
    color: bool = True,
    ascii_mode: bool = False,
) -> list[str]:
    """Pure render: one Frame/OverlayState -> exactly `rows` strings of `cols` visible chars.

    No cursor-positioning escapes are emitted here (only truecolor codes, and
    only when `color=True`) so this is safe to call from tests or to diff.
    """
    cols = max(cols, 20)
    rows = max(rows, 6)
    interior_w = cols - 2
    interior_h = rows - 2
    hud_n = min(HUD_LINES, interior_h)
    canvas_h = interior_h - hud_n

    canvas_rows = _render_canvas(frame, state, interior_w, canvas_h, color=color, ascii_mode=ascii_mode)
    hud_all = [
        _chord_line(state, interior_w, color=color),
        _vibe_line(frame, state, interior_w, ascii_mode=ascii_mode),
        _status_line(state, interior_w),
    ]
    hud_rows = hud_all[:hud_n]

    tl, tr, bl, br, hb, vb = ("+", "+", "+", "+", "-", "|") if ascii_mode else ("┌", "┐", "└", "┘", "─", "│")

    lines = [tl + hb * interior_w + tr]
    lines.extend(vb + r + vb for r in canvas_rows)
    lines.extend(vb + r + vb for r in hud_rows)
    lines.append(bl + hb * interior_w + br)
    return lines


class TerminalView:
    """Terminal-based stand-in for `ui.overlay.Window`. Same start/draw/poll_key/close shape."""

    def __init__(self, cfg: Config, ascii: bool = False) -> None:  # noqa: A002 -- matches the spec'd kwarg name
        self.cfg = cfg
        self._ascii = ascii or not _supports_braille()
        self._keys: KeyReader | None = None
        self._started = False
        self._last_draw_t = 0.0
        self._last_size = (80, 24)
        self._last_size_check = 0.0

    def start(self) -> None:
        enable_vt()
        sys.stdout.write("\x1b[?1049h\x1b[?25l")
        sys.stdout.flush()
        self._keys = KeyReader()
        self._keys.__enter__()
        self._last_size = shutil.get_terminal_size()
        self._last_size_check = time.monotonic()
        self._started = True

    def draw(self, frame: Frame, state: OverlayState) -> None:
        now = time.monotonic()
        if now - self._last_draw_t < 1.0 / _MAX_FPS:
            return
        self._last_draw_t = now

        if now - self._last_size_check >= _RESIZE_POLL_S:
            self._last_size_check = now
            size = shutil.get_terminal_size()
            if size != self._last_size:
                self._last_size = size
                sys.stdout.write("\x1b[2J")

        cols, rows = self._last_size
        lines = render(frame, state, cols, rows, color=True, ascii_mode=self._ascii)
        body = "\r\n".join(line + "\x1b[K" for line in lines)
        sys.stdout.write("\x1b[H" + body)
        sys.stdout.flush()

    def poll_key(self) -> str | None:
        if self._keys is None:
            return None
        return self._keys.poll()

    def close(self) -> None:
        if self._keys is not None:
            self._keys.__exit__(None, None, None)
            self._keys = None
        if self._started:
            sys.stdout.write("\x1b[?25h\x1b[?1049l")
            sys.stdout.flush()
            self._started = False
