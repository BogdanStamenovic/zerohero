"""Pianist detectors on synthetic hand motion."""

import random

from synth_hands import HandSpec, hold_path, motion, piecewise, stroke_path

from zerohero.config import Config, PianoConfig
from zerohero.events import GripHit, GripRelease, Passage, SweepStep
from zerohero.gestures.pianist import Pianist
from zerohero.gestures.pipeline import GesturePipeline
from zerohero.music import piano
from zerohero.music.theory import parse_chord

W = 0.12
PCFG = PianoConfig()
LATTICE = piano.lattice(parse_chord("C"), 0.0, PCFG)


def run(frames, side="right"):
    cfg = Config()
    pipeline = GesturePipeline(cfg)
    pianist = Pianist(side, cfg.piano)
    events = []
    for f in frames:
        pipeline.update(f)
        track = pipeline.tracks[side]
        slot = piano.slot_at(track.palm[0], LATTICE, PCFG) if track.present else None
        events.extend(pianist.update(track, f.t, slot))
    return events


def test_grip_hit_then_release():
    # open hand still, then grip, hit downward while gripped, hold, open again
    def shape(t):
        return "closed" if 0.6 <= t < 2.0 else "open"

    path = piecewise([(1.0, 1.14, stroke_path((0.6, 0.4), (0.0, 1.6 * W), 1.0, 0.14))])
    frames = motion([HandSpec("right", W, path, shape=shape)], 0.0, 2.6, rng=random.Random(1))
    events = run(frames)
    hits = [e for e in events if isinstance(e, GripHit)]
    rel = [e for e in events if isinstance(e, GripRelease)]
    assert len(hits) == 1
    assert hits[0].downward is True
    assert 0.5 < hits[0].x < 0.7
    assert hits[0].intensity > 0.3
    assert len(rel) == 1 and rel[0].t > hits[0].t
    assert not [e for e in events if isinstance(e, (Passage, SweepStep))]


def test_sweep_right_fires_ascending_slots_then_left_descending():
    right = stroke_path((0.15, 0.5), (0.7, 0.0), 0.5, 0.9)
    left = stroke_path((0.85, 0.5), (-0.7, 0.0), 2.0, 0.9)
    path = piecewise([(0.5, 1.4, right), (2.0, 2.9, left)])
    frames = motion([HandSpec("right", W, path)], 0.0, 3.5, rng=random.Random(2))
    events = run(frames)
    steps = [e for e in events if isinstance(e, SweepStep)]
    assert len(steps) >= 8
    rights = [e for e in steps if e.direction == "right"]
    lefts = [e for e in steps if e.direction == "left"]
    assert len(rights) >= 4 and len(lefts) >= 4
    assert [e.slot for e in rights] == sorted(e.slot for e in rights)
    assert [e.slot for e in lefts] == sorted((e.slot for e in lefts), reverse=True)
    assert all(e.t < lefts[0].t for e in rights)
    assert not [e for e in events if isinstance(e, GripHit)]


def test_slow_drift_is_not_a_sweep_and_rest_is_silent():
    # ~0.6 hand-widths per second peak plus tracker jitter: below sweep_speed_on
    slow = stroke_path((0.3, 0.5), (0.3 * W * 2, 0.0), 0.5, 2.0)
    frames = motion([HandSpec("right", W, piecewise([(0.5, 2.5, slow)]))], 0.0, 3.0, rng=random.Random(3))
    assert not run(frames)
    frames = motion([HandSpec("right", W, hold_path((0.5, 0.5)))], 0.0, 3.0, rng=random.Random(4))
    assert not run(frames)


def test_open_vertical_hit_is_a_passage_with_direction():
    up = stroke_path((0.5, 0.6), (0.0, -1.6 * W), 0.5, 0.14)
    down = stroke_path((0.5, 0.6 - 1.6 * W), (0.0, 1.6 * W), 1.5, 0.14)
    frames = motion([HandSpec("right", W, piecewise([(0.5, 0.64, up), (1.5, 1.64, down)]))], 0.0, 2.2)
    events = run(frames)
    runs = [e for e in events if isinstance(e, Passage)]
    assert [r.direction for r in runs] == ["up", "down"]
    assert all(not r.erratic for r in runs)
    assert not [e for e in events if isinstance(e, GripHit)]


def test_erratic_wiggle_is_an_erratic_passage():
    # four fast horizontal reversals inside half a second
    pieces = []
    x, t = 0.5, 0.5
    for i in range(4):
        d = 1.2 * W if i % 2 == 0 else -1.2 * W
        pieces.append((t, t + 0.1, stroke_path((x, 0.5), (d, 0.0), t, 0.1)))
        x += d
        t += 0.1
    frames = motion([HandSpec("right", W, piecewise(pieces))], 0.0, 1.5, rng=random.Random(5))
    events = run(frames)
    assert any(isinstance(e, Passage) and e.erratic for e in events)


def test_hand_leaving_frame_releases_grip():
    def shape(t):
        return "absent" if t >= 1.0 else "closed"

    frames = motion([HandSpec("right", W, hold_path((0.5, 0.5)), shape=shape)], 0.0, 1.5)
    events = run(frames)
    assert [type(e) for e in events] == [GripRelease]
