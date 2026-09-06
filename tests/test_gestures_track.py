from __future__ import annotations

import math

from zerohero.config import GestureConfig
from zerohero.gestures.features import HandFeatures
from zerohero.gestures.track import HandTrack


def _feat(palm: tuple[float, float], width: float = 0.15) -> HandFeatures:
    return HandFeatures(palm=palm, width=width, extended=(True,) * 5, closed_score=0.0)


def test_velocity_converges_on_known_linear_motion():
    """Constant palm velocity of 2 hand-widths/sec downward; the EMA should
    settle onto that value once enough samples have flushed the initial 0."""
    cfg = GestureConfig()
    width = 0.15
    speed_hw = 2.0
    dt = 1 / 30
    step = speed_hw * width * dt  # normalised-unit displacement per frame

    track = HandTrack("right", cfg)
    t = 0.0
    y = 0.5
    track.update(_feat((0.5, y), width), t)
    for _ in range(60):  # ~2s, plenty for a 0.35 EMA to converge
        t += dt
        y += step
        track.update(_feat((0.5, y), width), t)

    assert math.isclose(track.speed, speed_hw, rel_tol=0.05)
    assert track.velocity[1] > 0  # moving down (+y)


def test_reset_after_lost_after_prevents_velocity_spike():
    cfg = GestureConfig()
    width = 0.15
    track = HandTrack("right", cfg)
    t = 0.0
    track.update(_feat((0.2, 0.2), width), t)
    t += 1 / 30
    track.update(_feat((0.25, 0.2), width), t)
    assert track.speed > 0

    # hand goes missing for longer than lost_after
    for _ in range(20):
        t += 1 / 30
        track.mark_missing(t)
    assert track.present is False

    # reappears far away -- should not be read as a huge velocity
    t += 1 / 30
    track.update(_feat((0.9, 0.9), width), t)
    assert track.velocity == (0.0, 0.0)
    assert track.present is True


def test_tiny_dt_is_ignored_not_divided_by():
    cfg = GestureConfig()
    width = 0.15
    track = HandTrack("right", cfg)
    track.update(_feat((0.5, 0.5), width), 0.0)
    track.update(_feat((0.5, 0.52), width), 1 / 30)
    v_before = track.velocity

    # a near-duplicate timestamp with a big jump would blow up 1/dt if not clamped
    track.update(_feat((0.5, 0.9), width), 1 / 30 + 0.0002)
    assert track.velocity == v_before


def test_large_dt_resets_instead_of_reporting_a_jump():
    cfg = GestureConfig()
    width = 0.15
    track = HandTrack("right", cfg)
    track.update(_feat((0.5, 0.5), width), 0.0)
    track.update(_feat((0.5, 0.52), width), 1 / 30)
    assert track.speed > 0

    # a huge gap between two calls, without going through mark_missing at all
    track.update(_feat((0.1, 0.1), width), 5.0)
    assert track.velocity == (0.0, 0.0)
