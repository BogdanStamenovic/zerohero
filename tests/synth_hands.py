"""Synthetic hand trajectories for gesture tests: no camera, no MediaPipe.

Builds `Frame` sequences with plausible 21-point landmark geometry (open hand
vs. fist) moving along a caller-specified palm path, at ~30 fps with jitter
on both frame timing and every landmark, mimicking real tracker noise.

Landmark geometry is expressed as local offsets (in hand-widths, x right,
y down, matching `Frame`'s normalised image coords) around the wrist at the
origin, calibrated so:
  - `dist(wrist, middle_mcp) == 1.0` hand-width by construction, matching
    `HandFeatures.width`.
  - open-hand fingertips are well past their PIPs (extended); fist fingertips
    curl back closer to the wrist than their PIPs (curled).
  - the open thumb tip sits far from the pinky MCP; the fist thumb tip tucks
    in close to it.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass

from zerohero.events import Frame, Hand, Landmark, Side

PointFn = Callable[[float], tuple[float, float]]
ShapeFn = Callable[[float], str]  # t -> "open" | "fist" | "absent"

# local offsets, in hand-widths, keyed by MediaPipe landmark index
_OPEN_TEMPLATE: dict[int, tuple[float, float]] = {
    0: (0.00, 0.00),  # wrist
    1: (-0.15, -0.10),  # thumb_cmc
    2: (-0.35, -0.20),  # thumb_mcp
    3: (-0.55, -0.35),  # thumb_ip
    4: (-0.80, -0.50),  # thumb_tip (far from pinky_mcp -> extended)
    5: (-0.30, -0.90),  # index_mcp
    6: (-0.30, -1.25),  # index_pip
    7: (-0.30, -1.45),  # index_dip
    8: (-0.30, -1.65),  # index_tip
    9: (0.00, -1.00),  # middle_mcp (defines hand width)
    10: (0.00, -1.40),  # middle_pip
    11: (0.00, -1.65),  # middle_dip
    12: (0.00, -1.90),  # middle_tip
    13: (0.28, -0.92),  # ring_mcp
    14: (0.28, -1.27),  # ring_pip
    15: (0.28, -1.47),  # ring_dip
    16: (0.28, -1.67),  # ring_tip
    17: (0.50, -0.80),  # pinky_mcp
    18: (0.50, -1.10),  # pinky_pip
    19: (0.50, -1.28),  # pinky_dip
    20: (0.50, -1.45),  # pinky_tip
}

# Same base joints, fingertips curled back toward the palm instead of extended.
_FIST_TEMPLATE: dict[int, tuple[float, float]] = dict(_OPEN_TEMPLATE)
_FIST_TEMPLATE.update(
    {
        3: (-0.55, -0.35),  # thumb_ip unchanged
        4: (0.10, -0.55),  # thumb_tip tucked in near pinky_mcp -> curled
        6: (-0.30, -1.10),  # index_pip (slight bend)
        7: (-0.30, -0.95),  # index_dip folds back down
        8: (-0.28, -0.80),  # index_tip: closer to wrist than pip -> curled
        10: (0.00, -1.25),
        11: (0.00, -1.05),
        12: (0.05, -0.85),
        14: (0.28, -1.15),
        15: (0.28, -0.97),
        16: (0.25, -0.80),
        18: (0.50, -1.00),
        19: (0.48, -0.85),
        20: (0.45, -0.70),
    }
)

_PALM_LANDMARKS = (0, 5, 9, 13, 17)


def _build_hand(
    side: Side,
    template: dict[int, tuple[float, float]],
    palm: tuple[float, float],
    width: float,
    rng: random.Random,
    noise_sigma: float,
) -> Hand:
    local = {i: (ox * width, oy * width) for i, (ox, oy) in template.items()}
    local_palm_x = sum(local[i][0] for i in _PALM_LANDMARKS) / len(_PALM_LANDMARKS)
    local_palm_y = sum(local[i][1] for i in _PALM_LANDMARKS) / len(_PALM_LANDMARKS)
    shift = (palm[0] - local_palm_x, palm[1] - local_palm_y)

    landmarks = []
    for i in range(21):
        lx, ly = local[i]
        landmarks.append(
            Landmark(
                x=lx + shift[0] + rng.gauss(0.0, noise_sigma),
                y=ly + shift[1] + rng.gauss(0.0, noise_sigma),
                z=rng.gauss(0.0, noise_sigma),
            )
        )
    return Hand(side=side, score=0.95, landmarks=landmarks)


def open_hand(
    side: Side, palm: tuple[float, float], width: float, rng: random.Random | None = None, noise_sigma: float = 0.003
) -> Hand:
    return _build_hand(side, _OPEN_TEMPLATE, palm, width, rng or random.Random(0), noise_sigma)


def fist(
    side: Side, palm: tuple[float, float], width: float, rng: random.Random | None = None, noise_sigma: float = 0.003
) -> Hand:
    return _build_hand(side, _FIST_TEMPLATE, palm, width, rng or random.Random(0), noise_sigma)


# ---- palm paths -----------------------------------------------------------


def hold_path(pos: tuple[float, float]) -> PointFn:
    return lambda _t: pos


def drift_path(p0: tuple[float, float], velocity: tuple[float, float], t0: float) -> PointFn:
    """Constant-velocity motion from `p0` starting at `t0` (units/sec)."""
    return lambda t: (p0[0] + velocity[0] * max(0.0, t - t0), p0[1] + velocity[1] * max(0.0, t - t0))


def stroke_path(p0: tuple[float, float], delta: tuple[float, float], t0: float, duration: float) -> PointFn:
    """A single stroke from `p0` to `p0 + delta` over `[t0, t0 + duration]`.

    Velocity follows a raised cosine (`1 - cos(2*pi*frac)`): zero at both
    ends, peaking at the midpoint at twice the average speed. This is what a
    real strum/beat onset looks like -- a clean accelerate-decelerate, not an
    instant jump. Clamped to the endpoints outside the window so strokes can
    be composed back-to-back.
    """

    def f(t: float) -> tuple[float, float]:
        if t <= t0:
            return p0
        if t >= t0 + duration:
            return (p0[0] + delta[0], p0[1] + delta[1])
        frac = (t - t0) / duration
        s = frac - math.sin(2 * math.pi * frac) / (2 * math.pi)
        return (p0[0] + delta[0] * s, p0[1] + delta[1] * s)

    return f


def peak_speed_hw(delta_hw: float, duration: float) -> float:
    """Peak speed (hand-widths/sec) that `stroke_path` reaches for a stroke
    covering `delta_hw` hand-widths in `duration` seconds. Lets tests derive
    a displacement/duration pair from a target peak speed instead of guessing."""
    return 2.0 * delta_hw / duration


def piecewise(pieces: list[tuple[float, float, PointFn]]) -> PointFn:
    """Picks whichever `(t0, t1, path_fn)` window covers `t`; outside all
    windows, holds at the nearest window's boundary value."""

    def f(t: float) -> tuple[float, float]:
        for t0, t1, fn in pieces:
            if t0 <= t <= t1:
                return fn(t)
        before = [p for p in pieces if p[1] <= t]
        if before:
            t0, t1, fn = before[-1]
            return fn(t1)
        t0, t1, fn = pieces[0]
        return fn(t0)

    return f


# ---- frame sequence generation --------------------------------------------


@dataclass
class HandSpec:
    side: Side
    width: float
    palm: PointFn
    shape: ShapeFn = lambda _t: "open"  # noqa: E731


def motion(
    hand_specs: list[HandSpec],
    t_start: float,
    duration: float,
    fps: float = 30.0,
    jitter: float = 0.15,
    noise_sigma: float = 0.003,
    rng: random.Random | None = None,
) -> list[Frame]:
    """Build a `Frame` sequence spanning `[t_start, t_start + duration)`.

    Frame timing jitters by up to `jitter` fraction of the nominal 1/fps
    period, like a real camera thread does not deliver perfectly even
    intervals. Each present hand gets fresh landmark noise per frame.
    """
    rng = rng or random.Random(0)
    dt_nominal = 1.0 / fps

    times = []
    t = t_start
    while t < t_start + duration:
        times.append(t)
        t += dt_nominal * (1.0 + rng.uniform(-jitter, jitter))

    frames = []
    for t in times:
        hands = []
        for spec in hand_specs:
            shape = spec.shape(t)
            if shape == "absent":
                continue
            palm = spec.palm(t)
            builder = open_hand if shape == "open" else fist
            hands.append(builder(spec.side, palm, spec.width, rng=rng, noise_sigma=noise_sigma))
        frames.append(Frame(t=t, width=640, height=480, hands=hands))
    return frames
