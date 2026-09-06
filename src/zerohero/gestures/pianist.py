"""Per-hand pianist detectors for piano mode. See ARCHITECTURE.md "piano".

One `Pianist` per hand watches its `HandTrack` every frame and emits:

- `GripHit`     a sharp movement onset while the hand is gripped
- `GripRelease` the grip opened (the mode releases the held chord)
- `SweepStep`   an open hand moving sideways crossed the next keyboard slot
- `Passage`     an open hand hit vertically, or moved erratically

Grip uses hysteresis on `closed_score`. Hits use the same onset/hysteresis
scheme as the strum detector, with the peak-speed prediction so the accent
does not depend on which frame crossed the threshold. Sweeps are continuous:
the hand's x position is quantised onto the chord-tone lattice and a step
fires whenever the slot changes in the direction of travel.
"""

from __future__ import annotations

from collections import deque

from zerohero.config import PianoConfig
from zerohero.events import GripHit, GripRelease, Passage, PianistEvent, Side, SweepStep
from zerohero.gestures._onset import onset_intensity, predict_peak
from zerohero.gestures.track import HandTrack


class Pianist:
    def __init__(self, side: Side, cfg: PianoConfig) -> None:
        self.side = side
        self.cfg = cfg
        self.gripped = False
        self._hit_locked = False
        self._last_hit_t: float | None = None
        self._prev: tuple[float, float] | None = None  # (t, speed) previous frame
        self._slot: int | None = None
        self._last_sweep_t = 0.0
        self._vx_sign_history: deque[tuple[float, int]] = deque()
        self._was_present = False

    # The mode owns the lattice (it depends on the chord); it tells us how many
    # slots there are and where the hand currently is on it.
    def update(self, track: HandTrack, t: float, slot: int | None) -> list[PianistEvent]:
        cfg = self.cfg
        out: list[PianistEvent] = []
        if not track.present:
            if self.gripped:
                self.gripped = False
                out.append(GripRelease(t=t, hand=self.side))
            self._prev = None
            self._slot = None
            self._hit_locked = False
            self._vx_sign_history.clear()
            self._was_present = False
            return out
        self._was_present = True

        # ---- grip ----
        score = track.closed_score
        if not self.gripped and score >= cfg.grip_on:
            self.gripped = True
            self._slot = None  # a grip is not a sweep; forget the lattice position
        elif self.gripped and score <= cfg.grip_off:
            self.gripped = False
            out.append(GripRelease(t=t, hand=self.side))

        vx, vy = track.velocity
        speed = track.speed
        prev, self._prev = self._prev, (t, speed)

        # ---- erratic motion bookkeeping: horizontal direction reversals ----
        sign = 1 if vx > 0.6 else -1 if vx < -0.6 else 0
        if sign and (not self._vx_sign_history or self._vx_sign_history[-1][1] != sign):
            self._vx_sign_history.append((t, sign))
        while self._vx_sign_history and t - self._vx_sign_history[0][0] > cfg.erratic_window:
            self._vx_sign_history.popleft()
        erratic = len(self._vx_sign_history) - 1 >= cfg.erratic_reversals

        # ---- hit onset (grip -> chord, open + vertical/erratic -> passage) ----
        if self._hit_locked:
            if speed <= cfg.hit_speed_off:
                self._hit_locked = False
        elif speed >= cfg.hit_speed_on and (self._last_hit_t is None or t - self._last_hit_t >= cfg.hit_refractory):
            peak = predict_peak(speed, prev[1], t - prev[0]) if prev else speed
            intensity = onset_intensity(peak, cfg.hit_speed_on, cfg.hit_speed_full)
            vertical = abs(vy) >= abs(vx)
            if self.gripped:
                self._hit_locked = True
                self._last_hit_t = t
                out.append(GripHit(t=t, hand=self.side, x=track.palm[0], intensity=intensity, downward=vy > 0))
            elif vertical or erratic:
                self._hit_locked = True
                self._last_hit_t = t
                out.append(
                    Passage(
                        t=t,
                        hand=self.side,
                        x=track.palm[0],
                        direction="down" if vy > 0 else "up",
                        intensity=intensity,
                        erratic=erratic,
                    )
                )
                self._vx_sign_history.clear()
            # open hand moving sideways fast: that is a sweep, handled below

        # ---- sweep: open hand moving sideways ----
        if self.gripped or slot is None:
            return out
        sideways = abs(vx) >= cfg.sweep_speed_on and abs(vx) >= abs(vy)
        if not sideways:
            self._slot = slot  # resync while resting so the next sweep starts from here
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
