"""Per-side history: smoothed position, width and velocity for one hand.

Frames arrive at irregular intervals, so velocity is always computed from
`frame.t`, never an assumed fixed dt. See ARCHITECTURE.md "gestures".
"""

from __future__ import annotations

import math
from collections import deque

from zerohero.config import GestureConfig
from zerohero.events import Side
from zerohero.gestures.features import HandFeatures

# Below this dt the two samples are effectively the same instant (duplicate
# timestamps from a replay file, or two frames landing in the same tick);
# dividing by it would spike velocity to nonsense, so the step is skipped.
_MIN_DT = 0.001
# Above this dt, treat the gap as a discontinuity rather than motion: the
# hand could have moved anywhere in that time, so a computed velocity would
# be fabricated, not measured.
_MAX_DT = 0.25


class HandTrack:
    """Tracks one hand (left or right) across frames.

    State machine is just "have we seen this hand recently": `update()` feeds
    a detection, `mark_missing()` is called on frames where the hand was not
    found. Once the gap since the last detection exceeds `cfg.lost_after`,
    the velocity history is dropped so the next detection starts at rest
    instead of reporting a phantom velocity spike across the gap.
    """

    def __init__(self, side: Side, cfg: GestureConfig) -> None:
        self.side = side
        self.cfg = cfg
        self.present = False
        self.palm: tuple[float, float] = (0.0, 0.0)
        self.width = 0.0
        self.velocity: tuple[float, float] = (0.0, 0.0)
        self.speed = 0.0
        self.closed_score = 0.0
        # Fingertip movement relative to the palm, hand widths/s, EMA: the
        # "fingers wiggling" signal, independent of where the hand travels.
        self.finger_activity = 0.0
        self._tips: tuple[tuple[float, float], ...] = ()
        # Recent per-frame fingertip steps along the finger axis (t, step) for
        # the wiggle detector: a wiggle is a run of direction reversals.
        self.tip_steps: deque[tuple[float, float]] = deque()
        self.last_t: float | None = None
        self._smoothed_width = 0.0

    def update(self, features: HandFeatures, t: float) -> None:
        alpha = self.cfg.velocity_smoothing
        dt = None if self.last_t is None else t - self.last_t

        if dt is not None and dt < _MIN_DT:
            pass  # too small/noisy to measure; keep the previous velocity and last_t
        elif dt is None or dt > _MAX_DT:
            # First detection, or a gap long enough that motion across it is
            # unknowable: start at rest rather than compute a fabricated jump.
            self._smoothed_width = features.width
            self.velocity = (0.0, 0.0)
            self.last_t = t
        else:
            prev_palm = self.palm
            self._smoothed_width = alpha * self._smoothed_width + (1 - alpha) * features.width
            width_for_norm = self._smoothed_width if self._smoothed_width > 1e-6 else features.width
            dx = (features.palm[0] - prev_palm[0]) / width_for_norm
            dy = (features.palm[1] - prev_palm[1]) / width_for_norm
            if math.hypot(dx, dy) > self.cfg.teleport_widths:
                self.velocity = (0.0, 0.0)  # glitch, not motion: restart from rest
            else:
                old_vx, old_vy = self.velocity
                self.velocity = (alpha * old_vx + (1 - alpha) * dx / dt, alpha * old_vy + (1 - alpha) * dy / dt)
            self.last_t = t

        self.speed = math.hypot(*self.velocity)
        if dt is not None and _MIN_DT <= dt <= _MAX_DT and self._tips and len(self._tips) == len(features.tips):
            pairs = list(zip(self._tips, features.tips, strict=True))
            moved = sum(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in pairs) / len(pairs)
            # Signed step along the finger axis, averaged over the four fingers
            # (the thumb moves sideways). Curling and uncurling flip its sign.
            step = sum(b[1] - a[1] for a, b in pairs[1:]) / (len(pairs) - 1)
            beta = 0.6
            if abs(step) > 1.0:
                # Fingertips do not move a whole hand width in one frame: the
                # tracker re-fitted the hand. Forget the recent steps so the
                # glitch cannot count as a reversal.
                self.tip_steps.clear()
                self.finger_activity = 0.0
            else:
                self.finger_activity = beta * self.finger_activity + (1 - beta) * moved / dt
                self.tip_steps.append((t, step))
            while self.tip_steps and t - self.tip_steps[0][0] > 0.5:
                self.tip_steps.popleft()
        else:
            self.finger_activity = 0.0
            self.tip_steps.clear()
        self._tips = features.tips
        self.palm = features.palm
        self.width = features.width
        self.closed_score = features.closed_score
        self.present = True

    def mark_missing(self, t: float) -> None:
        if self.last_t is not None and (t - self.last_t) > self.cfg.lost_after:
            self.velocity = (0.0, 0.0)
            self.speed = 0.0
            self.finger_activity = 0.0
            self._tips = ()
            self.tip_steps.clear()
            self._smoothed_width = 0.0
            self.last_t = None
        self.present = False


def reversals(steps: deque[tuple[float, float]], now: float, window: float, min_step: float) -> int:
    """Direction reversals of the fingertip motion inside the last `window` seconds.

    Steps smaller than `min_step` (hand widths) are jitter and are skipped, so
    a still hand scores 0 however noisy the tracker is; a wiggle alternates
    sign every few frames and scores high.
    """
    last_sign = 0
    count = 0
    for t, step in steps:
        if now - t > window or abs(step) < min_step:
            continue
        sign = 1 if step > 0 else -1
        if last_sign and sign != last_sign:
            count += 1
        last_sign = sign
    return count
