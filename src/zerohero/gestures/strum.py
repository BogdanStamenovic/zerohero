"""Onset detector for guitar strums: vertical hand speed with hysteresis."""

from __future__ import annotations

from zerohero.config import GestureConfig
from zerohero.events import Side, Strum, StrumDirection
from zerohero.gestures._onset import onset_intensity, predict_peak
from zerohero.gestures.track import HandTrack

# A sideways wave should never register as a strum. If the horizontal speed
# dominates the vertical by this much, the motion is a wave, not a pick
# stroke, regardless of how fast |vy| is.
_SIDEWAYS_RATIO = 1.5


class StrumDetector:
    """One-shot onset/hysteresis state machine on `|vy|`.

    Armed -> fires once `|vy|` crosses `strum_speed_on` (and the refractory
    period has elapsed and the motion isn't sideways) -> locked until `|vy|`
    drops back under `strum_speed_off`, at which point it re-arms. This is
    the same shape used by `FistDetector`/`BeatDetector`: hysteresis avoids
    re-firing while the hand rides above a single threshold, and onset
    firing (not peak or release) keeps latency low.
    """

    def __init__(self, side: Side, cfg: GestureConfig) -> None:
        self.side = side
        self.cfg = cfg
        self._locked = False
        self._last_fire_t: float | None = None
        self._prev: tuple[float, float] | None = None  # (t, |vy|) from the previous frame

    def update(self, track: HandTrack, t: float) -> Strum | None:
        cfg = self.cfg
        if not track.present:
            self._prev = None
            return None

        vx, vy = track.velocity
        avy = abs(vy)
        prev, self._prev = self._prev, (t, avy)

        if self._locked:
            if avy <= cfg.strum_speed_off:
                self._locked = False
            return None

        if avy < cfg.strum_speed_on:
            return None
        if self._last_fire_t is not None and (t - self._last_fire_t) < cfg.strum_refractory:
            return None
        if abs(vx) > _SIDEWAYS_RATIO * avy:
            return None  # sideways wave, not a strum

        self._locked = True
        self._last_fire_t = t
        direction: StrumDirection = "down" if vy > 0 else "up"
        peak = predict_peak(avy, prev[1], t - prev[0]) if prev else avy
        intensity = onset_intensity(peak, cfg.strum_speed_on, cfg.strum_speed_full)
        return Strum(t=t, hand=self.side, direction=direction, intensity=intensity)
