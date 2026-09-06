"""Per-side history: smoothed position, width and velocity for one hand.

Frames arrive at irregular intervals, so velocity is always computed from
`frame.t`, never an assumed fixed dt. See ARCHITECTURE.md "gestures".
"""

from __future__ import annotations

import math

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
            raw_vx = (features.palm[0] - prev_palm[0]) / dt / width_for_norm
            raw_vy = (features.palm[1] - prev_palm[1]) / dt / width_for_norm
            old_vx, old_vy = self.velocity
            self.velocity = (alpha * old_vx + (1 - alpha) * raw_vx, alpha * old_vy + (1 - alpha) * raw_vy)
            self.last_t = t

        self.speed = math.hypot(*self.velocity)
        self.palm = features.palm
        self.width = features.width
        self.closed_score = features.closed_score
        self.present = True

    def mark_missing(self, t: float) -> None:
        if self.last_t is not None and (t - self.last_t) > self.cfg.lost_after:
            self.velocity = (0.0, 0.0)
            self.speed = 0.0
            self._smoothed_width = 0.0
            self.last_t = None
        self.present = False
