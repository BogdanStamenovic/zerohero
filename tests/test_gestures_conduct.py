from __future__ import annotations

import random

from synth_hands import HandSpec, motion, stroke_path

from zerohero.config import GestureConfig
from zerohero.gestures.conduct import BeatDetector
from zerohero.gestures.features import HandFeatures
from zerohero.gestures.track import HandTrack

WIDTH = 0.15


def _run(frames, cfg, side="right", closed_fn=lambda _t: False):
    track = HandTrack(side, cfg)
    detector = BeatDetector(side, cfg)
    events = []
    for f in frames:
        hand = f.hand(side)
        if hand is None:
            track.mark_missing(f.t)
            continue
        track.update(HandFeatures.from_hand(hand), f.t)
        ev = detector.update(track, f.t, closed_fn(f.t))
        if ev is not None:
            events.append(ev)
    return events


def _stroke_frames(delta, duration=0.15, t0=0.3, total=1.0, seed=0):
    path = stroke_path((0.5, 0.5), delta, t0, duration)
    spec = HandSpec(side="right", width=WIDTH, palm=path, shape=lambda _t: "open")
    return motion([spec], t_start=0.0, duration=total, rng=random.Random(seed))


def test_direction_left():
    cfg = GestureConfig()
    frames = _stroke_frames((-1.0 * WIDTH, 0.0), seed=20)
    events = _run(frames, cfg)
    assert len(events) == 1
    assert events[0].direction == "left"


def test_direction_right():
    cfg = GestureConfig()
    frames = _stroke_frames((1.0 * WIDTH, 0.0), seed=21)
    events = _run(frames, cfg)
    assert len(events) == 1
    assert events[0].direction == "right"


def test_direction_up():
    cfg = GestureConfig()
    frames = _stroke_frames((0.0, -1.0 * WIDTH), seed=22)
    events = _run(frames, cfg)
    assert len(events) == 1
    assert events[0].direction == "up"


def test_direction_down():
    cfg = GestureConfig()
    frames = _stroke_frames((0.0, 1.0 * WIDTH), seed=23)
    events = _run(frames, cfg)
    assert len(events) == 1
    assert events[0].direction == "down"


def test_closed_flag_reflects_caller_supplied_state():
    cfg = GestureConfig()
    frames = _stroke_frames((0.0, 1.0 * WIDTH), seed=24)
    events = _run(frames, cfg, closed_fn=lambda _t: True)
    assert len(events) == 1
    assert events[0].closed is True

    frames2 = _stroke_frames((0.0, 1.0 * WIDTH), seed=25)
    events2 = _run(frames2, cfg, closed_fn=lambda _t: False)
    assert len(events2) == 1
    assert events2[0].closed is False
