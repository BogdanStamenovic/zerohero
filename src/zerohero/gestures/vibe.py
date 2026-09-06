"""Overall energy meter: how vigorously the hands are moving right now."""

from __future__ import annotations

import math
from collections import deque

from zerohero.config import GestureConfig
from zerohero.gestures.track import HandTrack


class VibeMeter:
    """Rolling RMS of hand speed over `cfg.vibe_window` seconds, in `0..1`.

    Each `update()` records one sample: the RMS speed across present hands
    (0 if none are present). Samples older than `vibe_window` are dropped by
    wall-clock time, not sample count, so the meter decays smoothly and at a
    predictable rate once motion stops, regardless of frame rate -- once the
    last high-speed sample ages out of the window (~`vibe_window` seconds),
    only zero samples remain and the value settles at 0.
    """

    def __init__(self, cfg: GestureConfig) -> None:
        self.cfg = cfg
        self.value = 0.0
        self._samples: deque[tuple[float, float]] = deque()

    def update(self, tracks: list[HandTrack], t: float) -> None:
        present_speeds = [tr.speed for tr in tracks if tr.present]
        sample = math.sqrt(sum(s * s for s in present_speeds) / len(present_speeds)) if present_speeds else 0.0

        self._samples.append((t, sample))
        cutoff = t - self.cfg.vibe_window
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

        if not self._samples:
            self.value = 0.0
            return
        rms = math.sqrt(sum(s * s for _, s in self._samples) / len(self._samples))
        self.value = max(0.0, min(1.0, rms / self.cfg.vibe_speed_full))
