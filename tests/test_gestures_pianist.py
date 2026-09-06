"""Pianist detectors on synthetic hand motion."""

import random

from synth_hands import HandSpec, hold_path, motion, piecewise, stroke_path

from zerohero.config import Config, PianoConfig
from zerohero.events import ChordHit, Frame, Landmark, SweepStep, WiggleNote
from zerohero.gestures.pianist import Pianist
from zerohero.gestures.pipeline import GesturePipeline
from zerohero.music import piano
from zerohero.music.theory import parse_chord

W = 0.12
PCFG = PianoConfig()
LATTICE = piano.lattice(parse_chord("C"), 0.0, PCFG)
ABOVE = PCFG.limiter_y - 0.2
BELOW = PCFG.limiter_y + 0.2


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


def test_open_hand_hit_down_is_a_chord_and_hit_up_is_nothing():
    down = stroke_path((0.6, ABOVE), (0.0, 1.6 * W), 0.5, 0.14)
    up = stroke_path((0.6, ABOVE + 1.6 * W), (0.0, -1.6 * W), 1.5, 0.14)
    frames = motion([HandSpec("right", W, piecewise([(0.5, 0.64, down), (1.5, 1.64, up)]))], 0.0, 2.2)
    events = run(frames)
    hits = [e for e in events if isinstance(e, ChordHit)]
    assert len(hits) == 1
    assert 0.5 < hits[0].x < 0.7
    assert hits[0].intensity > 0.3
    assert not [e for e in events if isinstance(e, (SweepStep, WiggleNote))]


def test_sweep_below_limiter_fires_slots_and_above_does_nothing():
    right = stroke_path((0.15, BELOW), (0.7, 0.0), 0.5, 0.9)
    left = stroke_path((0.85, BELOW), (-0.7, 0.0), 2.0, 0.9)
    frames = motion([HandSpec("right", W, piecewise([(0.5, 1.4, right), (2.0, 2.9, left)]))], 0.0, 3.5)
    steps = [e for e in run(frames) if isinstance(e, SweepStep)]
    rights = [e for e in steps if e.direction == "right"]
    lefts = [e for e in steps if e.direction == "left"]
    assert len(rights) >= 4 and len(lefts) >= 4
    assert [e.slot for e in rights] == sorted(e.slot for e in rights)
    assert [e.slot for e in lefts] == sorted((e.slot for e in lefts), reverse=True)

    above = stroke_path((0.15, ABOVE), (0.7, 0.0), 0.5, 0.9)
    frames = motion([HandSpec("right", W, piecewise([(0.5, 1.4, above)]))], 0.0, 2.0)
    assert not [e for e in run(frames) if isinstance(e, SweepStep)]


def test_slow_drift_and_rest_are_silent():
    slow = stroke_path((0.3, BELOW), (0.3 * W * 2, 0.0), 0.5, 2.0)
    frames = motion([HandSpec("right", W, piecewise([(0.5, 2.5, slow)]))], 0.0, 3.0, rng=random.Random(3))
    assert not run(frames)
    frames = motion([HandSpec("right", W, hold_path((0.5, BELOW)))], 0.0, 3.0, rng=random.Random(4))
    assert not run(frames)


def _wiggle_frames(seconds: float, amplitude: float, hz: float = 6.0) -> list[Frame]:
    """A still open hand whose fingertips oscillate: the wiggle signal without palm motion."""
    import math

    base = motion([HandSpec("right", W, hold_path((0.5, ABOVE)))], 0.0, seconds, rng=random.Random(9))
    out = []
    for f in base:
        h = f.hands[0]
        lm = list(h.landmarks)
        phase = math.sin(2 * math.pi * hz * f.t)
        for i in (4, 8, 12, 16, 20):
            p = lm[i]
            lm[i] = Landmark(p.x + amplitude * W * phase * (1 if i % 8 == 0 else -1), p.y + amplitude * W * phase, p.z)
        h.landmarks = lm
        out.append(f)
    return out


def test_wiggling_fingers_emit_zigzag_notes_and_still_fingers_do_not():
    events = run(_wiggle_frames(2.0, amplitude=0.5))
    notes = [e for e in events if isinstance(e, WiggleNote)]
    assert len(notes) >= 6
    gaps = [b.t - a.t for a, b in zip(notes, notes[1:], strict=False)]
    assert min(gaps) >= PCFG.wiggle_gap_fast * 0.9
    assert all(e.drift == "none" for e in notes)
    assert not [e for e in events if isinstance(e, ChordHit)]
    assert not [e for e in run(_wiggle_frames(2.0, amplitude=0.0)) if isinstance(e, WiggleNote)]
