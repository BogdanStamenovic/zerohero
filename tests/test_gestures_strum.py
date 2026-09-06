from __future__ import annotations

import random

from synth_hands import HandSpec, drift_path, hold_path, motion, piecewise, stroke_path

from zerohero.config import GestureConfig
from zerohero.gestures.features import HandFeatures
from zerohero.gestures.strum import StrumDetector
from zerohero.gestures.track import HandTrack

WIDTH = 0.15


def _run(frames, cfg, side="right"):
    track = HandTrack(side, cfg)
    detector = StrumDetector(side, cfg)
    events = []
    for f in frames:
        hand = f.hand(side)
        if hand is None:
            track.mark_missing(f.t)
            continue
        track.update(HandFeatures.from_hand(hand), f.t)
        ev = detector.update(track, f.t)
        if ev is not None:
            events.append(ev)
    return events


def _chain(p0, strokes):
    """strokes: list of (t0, duration, delta). Builds a piecewise path that
    holds at p0 before the first stroke and at each stroke's end afterwards."""
    pieces = []
    p = p0
    for t0, duration, delta in strokes:
        pieces.append((t0, t0 + duration, stroke_path(p, delta, t0, duration)))
        p = (p[0] + delta[0], p[1] + delta[1])
    return piecewise(pieces)


def test_alternating_strokes_fire_once_each_with_correct_direction():
    cfg = GestureConfig()
    dy = 1.5 * WIDTH
    gap = 0.42  # stroke (0.12s) + pause (0.3s), well past unlock + refractory
    strokes = []
    for i in range(8):
        t0 = 0.5 + i * gap
        delta = (0.0, dy if i % 2 == 0 else -dy)
        strokes.append((t0, 0.12, delta))
    path = _chain((0.5, 0.5), strokes)

    total_duration = 0.5 + 8 * gap + 0.3
    spec = HandSpec(side="right", width=WIDTH, palm=path, shape=lambda _t: "open")
    frames = motion([spec], t_start=0.0, duration=total_duration, rng=random.Random(1))

    events = _run(frames, cfg)
    assert len(events) == 8
    expected = ["down", "up"] * 4
    assert [e.direction for e in events] == expected


def test_no_strum_on_slow_drift():
    cfg = GestureConfig()
    # 1 hand-width/sec is well under strum_speed_on (3.5)
    path = drift_path((0.3, 0.3), (0.0, 1.0 * WIDTH), t0=0.0)
    spec = HandSpec(side="right", width=WIDTH, palm=path, shape=lambda _t: "open")
    frames = motion([spec], t_start=0.0, duration=5.0, rng=random.Random(2))
    assert _run(frames, cfg) == []


def test_no_strum_from_jitter_alone():
    cfg = GestureConfig()
    path = hold_path((0.5, 0.5))
    spec = HandSpec(side="right", width=WIDTH, palm=path, shape=lambda _t: "open")
    frames = motion([spec], t_start=0.0, duration=5.0, rng=random.Random(3))
    assert _run(frames, cfg) == []


def test_no_phantom_strum_when_hand_reappears_elsewhere():
    cfg = GestureConfig()

    def shape(t: float) -> str:
        return "absent" if 1.0 <= t < 2.0 else "open"

    def palm(t: float) -> tuple[float, float]:
        return (0.7, 0.7) if t >= 2.0 else (0.2, 0.2)

    spec = HandSpec(side="right", width=WIDTH, palm=palm, shape=shape)
    frames = motion([spec], t_start=0.0, duration=3.0, rng=random.Random(4))
    assert _run(frames, cfg) == []


def test_intensity_orders_fast_stroke_above_slow_stroke():
    cfg = GestureConfig()

    fast_path = _chain((0.5, 0.5), [(0.3, 0.12, (0.0, 1.5 * WIDTH))])
    slow_path = _chain((0.5, 0.5), [(0.3, 0.30, (0.0, 0.6 * WIDTH))])

    fast_spec = HandSpec(side="right", width=WIDTH, palm=fast_path, shape=lambda _t: "open")
    slow_spec = HandSpec(side="right", width=WIDTH, palm=slow_path, shape=lambda _t: "open")

    fast_frames = motion([fast_spec], t_start=0.0, duration=1.0, rng=random.Random(5))
    slow_frames = motion([slow_spec], t_start=0.0, duration=1.0, rng=random.Random(6))

    fast_events = _run(fast_frames, cfg)
    slow_events = _run(slow_frames, cfg)

    assert len(fast_events) == 1
    assert len(slow_events) == 1
    assert fast_events[0].intensity > slow_events[0].intensity


def test_sideways_wave_does_not_fire_a_strum():
    cfg = GestureConfig()
    # Same stroke_path profile drives both axes, so vx/vy stays proportional
    # to dx/dy throughout: 2:1 here trips the 1.5x sideways-rejection ratio
    # while |vy| alone (dy=1.0 hw / 0.15s, ~9.6 hw/s measured) clears strum_speed_on,
    # so this exercises the ratio check rather than just failing the speed gate.
    path = _chain((0.5, 0.5), [(0.3, 0.15, (2.0 * WIDTH, 1.0 * WIDTH))])
    spec = HandSpec(side="right", width=WIDTH, palm=path, shape=lambda _t: "open")
    frames = motion([spec], t_start=0.0, duration=1.0, rng=random.Random(7))
    assert _run(frames, cfg) == []
