"""Onset detector for conducting hits (piano mode): 2D speed, dominant-axis
direction. Same onset/hysteresis shape as `StrumDetector`, see its docstring."""

from __future__ import annotations

from zerohero.config import GestureConfig
from zerohero.events import Beat, BeatDirection, Side
from zerohero.gestures._onset import onset_intensity, predict_peak
from zerohero.gestures.track import HandTrack


def _dominant_direction(vx: float, vy: float) -> BeatDirection:
    if abs(vx) >= abs(vy):
        return "left" if vx < 0 else "right"
    return "up" if vy < 0 else "down"


class BeatDetector:
    """Onset/hysteresis on 2D speed instead of `strum`'s vertical-only speed.

    Direction is read off the velocity at the moment of onset (the dominant
    axis), not tracked afterwards, so a hit that curves after triggering
    still reports the direction that actually caused it to fire.
    """

    def __init__(self, side: Side, cfg: GestureConfig) -> None:
        self.side = side
        self.cfg = cfg
        self._locked = False
        self._last_fire_t: float | None = None
        self._prev: tuple[float, float] | None = None  # (t, speed) from the previous frame

    def update(self, track: HandTrack, t: float, closed: bool) -> Beat | None:
        cfg = self.cfg
        if not track.present:
            self._prev = None
            return None

        speed = track.speed
        prev, self._prev = self._prev, (t, speed)

        if self._locked:
            if speed <= cfg.beat_speed_off:
                self._locked = False
            return None

        if speed < cfg.beat_speed_on:
            return None
        if self._last_fire_t is not None and (t - self._last_fire_t) < cfg.beat_refractory:
            return None

        self._locked = True
        self._last_fire_t = t
        vx, vy = track.velocity
        direction = _dominant_direction(vx, vy)
        peak = predict_peak(speed, prev[1], t - prev[0]) if prev else speed
        intensity = onset_intensity(peak, cfg.beat_speed_on, cfg.beat_speed_full)
        return Beat(t=t, hand=self.side, direction=direction, intensity=intensity, closed=closed)
