"""Shared onset-intensity mapping used by `strum.py` and `conduct.py`."""

from __future__ import annotations


def onset_intensity(speed: float, speed_on: float, speed_full: float) -> float:
    """Map an onset speed in `[speed_on, speed_full]` to `[0.15, 1.0]`.

    A stroke that barely clears the threshold should still read as a
    deliberate strum/hit, not a near-zero one, so the floor is 0.15 rather
    than 0.
    """
    span = max(speed_full - speed_on, 1e-6)
    ratio = max(0.0, min(1.0, (speed - speed_on) / span))
    return 0.15 + 0.85 * ratio


# Typical duration of one strum / conducting stroke. Only the ratio
# acceleration * duration matters below, so being off by 30% shifts intensity
# a little; it does not break the phase independence.
STROKE_DURATION = 0.15


def predict_peak(speed: float, prev_speed: float, dt: float) -> float:
    """Estimate a stroke's peak speed from the speed and acceleration at onset.

    Firing at the threshold crossing keeps latency low, but the speed sampled
    there depends on which frame happened to cross: the same stroke read
    0.2 or 1.0 depending on phase. Modelling the stroke as a half-sine
    `V sin(pi t / T)`: at onset `speed = V sin(phi)` and
    `accel = V (pi / T) cos(phi)`, so `V = sqrt(speed^2 + (accel T / pi)^2)`
    regardless of phase.
    """
    if dt <= 0:
        return speed
    accel = max(0.0, (speed - prev_speed) / dt)
    return max(speed, (speed * speed + (accel * STROKE_DURATION / 3.141592653589793) ** 2) ** 0.5)
