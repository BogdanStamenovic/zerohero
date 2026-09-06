"""Per-hand pianist detectors for piano mode. See ARCHITECTURE.md "piano".

One `Pianist` per hand watches its `HandTrack` every frame and emits:

- `ChordHit`    the open hand hit downward
- `SweepStep`   below the limiter, the hand moving sideways crossed the next
                keyboard slot
- `WiggleNote`  the fingers are wiggling and the next zigzag note is due

The limiter is the keyboard surface: above it the hand only travels (sideways
motion does nothing), below it sideways motion sweeps the keys. Hits use the
onset/hysteresis scheme shared with the strum detector, with the peak-speed
prediction so the accent does not depend on which frame crossed the
threshold. Sweeps are continuous: the hand's x is quantised onto the chord
lattice and a step fires whenever the slot changes in the direction of travel.
Wiggle is fingertip motion relative to the palm with hysteresis; while it is
on, notes are due at a rate set by how lively the fingers are.
"""

from __future__ import annotations

from typing import Literal

from zerohero.config import PianoConfig
from zerohero.events import ChordHit, PianistEvent, Side, SweepStep, WiggleNote
from zerohero.gestures._onset import onset_intensity, predict_peak
from zerohero.gestures.track import HandTrack


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


class Pianist:
    def __init__(self, side: Side, cfg: PianoConfig) -> None:
        self.side = side
        self.cfg = cfg
        self._hit_locked = False
        self._last_hit_t: float | None = None
        self._prev: tuple[float, float] | None = None  # (t, speed) previous frame
        self._slot: int | None = None
        self._last_sweep_t = 0.0
        self.wiggling = False
        self._next_wiggle_t = 0.0
        self._wiggle_since: float | None = None  # when activity first rose above wiggle_on

    def below_limiter(self, track: HandTrack) -> bool:
        return track.palm[1] >= self.cfg.limiter_y

    # The mode owns the lattice (it depends on the chord); it tells us where
    # the hand currently sits on it.
    def update(self, track: HandTrack, t: float, slot: int | None) -> list[PianistEvent]:
        cfg = self.cfg
        out: list[PianistEvent] = []
        if not track.present:
            self._prev = None
            self._slot = None
            self._hit_locked = False
            self.wiggling = False
            self._wiggle_since = None
            return out

        vx, vy = track.velocity
        speed = track.speed
        prev, self._prev = self._prev, (t, speed)

        # ---- chord: open hand hit downward ----
        if self._hit_locked:
            if speed <= cfg.hit_speed_off:
                self._hit_locked = False
        elif speed >= cfg.hit_speed_on and (self._last_hit_t is None or t - self._last_hit_t >= cfg.hit_refractory):
            downward = vy > 0 and vy >= abs(vx)
            if downward:
                peak = predict_peak(speed, prev[1], t - prev[0]) if prev else speed
                intensity = onset_intensity(peak, cfg.hit_speed_on, cfg.hit_speed_full)
                self._hit_locked = True
                self._last_hit_t = t
                out.append(ChordHit(t=t, hand=self.side, x=track.palm[0], intensity=intensity))

        # ---- wiggle: fingers moving relative to the palm ----
        act = track.finger_activity
        if not self.wiggling:
            if act >= cfg.wiggle_on:
                self._wiggle_since = self._wiggle_since if self._wiggle_since is not None else t
                if t - self._wiggle_since >= cfg.wiggle_min_hold:
                    self.wiggling = True
                    self._next_wiggle_t = t  # first note right away
            else:
                self._wiggle_since = None
        elif act <= cfg.wiggle_off:
            self.wiggling = False
            self._wiggle_since = None
        if self.wiggling and t >= self._next_wiggle_t:
            level = (act - cfg.wiggle_on) / max(cfg.wiggle_full - cfg.wiggle_on, 1e-6)
            drift: Literal["left", "right", "none"] = "right" if vx > 1.0 else "left" if vx < -1.0 else "none"
            activity = max(0.0, min(1.0, level))
            out.append(WiggleNote(t=t, hand=self.side, x=track.palm[0], drift=drift, activity=activity))
            self._next_wiggle_t = t + _lerp(cfg.wiggle_gap_slow, cfg.wiggle_gap_fast, level)

        # ---- sweep: sideways below the limiter ----
        if slot is None or not self.below_limiter(track):
            self._slot = None  # lifting off the keys forgets the position; a new sweep starts fresh
            return out
        sideways = abs(vx) >= cfg.sweep_speed_on and abs(vx) >= abs(vy)
        if not sideways:
            self._slot = slot  # resting on the keys: resync so the next sweep starts from here
            return out
        if self._slot is None:
            self._slot = slot
            return out
        direction = 1 if vx > 0 else -1
        # Only slots crossed in the direction of travel count; jitter backwards is ignored.
        while (slot - self._slot) * direction > 0 and t - self._last_sweep_t >= cfg.sweep_min_gap:
            self._slot += direction
            self._last_sweep_t = t
            out.append(
                SweepStep(
                    t=t, hand=self.side, slot=self._slot, direction="right" if direction > 0 else "left", speed=abs(vx)
                )
            )
        return out
