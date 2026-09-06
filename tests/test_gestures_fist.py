from __future__ import annotations

import random

from synth_hands import HandSpec, hold_path, motion

from zerohero.config import GestureConfig
from zerohero.events import FistClose, FistOpen
from zerohero.gestures.features import HandFeatures
from zerohero.gestures.fist import FistDetector
from zerohero.gestures.track import HandTrack

PALM = (0.3, 0.5)
WIDTH = 0.15


def _run(frames, cfg, side="left"):
    track = HandTrack(side, cfg)
    detector = FistDetector(side, cfg)
    events = []
    for f in frames:
        hand = f.hand(side)
        if hand is None:
            track.mark_missing(f.t)
            detector.update(track, f.t)
            continue
        track.update(HandFeatures.from_hand(hand), f.t)
        ev = detector.update(track, f.t)
        if ev is not None:
            events.append(ev)
    return events, detector


def _windowed_shape(closed_windows):
    def shape(t: float) -> str:
        for a, b in closed_windows:
            if a <= t < b:
                return "fist"
        return "open"

    return shape


def test_close_then_open_each_fire_exactly_once():
    cfg = GestureConfig()
    shape = _windowed_shape([(0.3, 0.9)])
    spec = HandSpec(side="left", width=WIDTH, palm=hold_path(PALM), shape=shape)
    frames = motion([spec], t_start=0.0, duration=1.5, rng=random.Random(10))
    events, _ = _run(frames, cfg)

    closes = [e for e in events if isinstance(e, FistClose)]
    opens = [e for e in events if isinstance(e, FistOpen)]
    # one bootstrap FistOpen when the initial open baseline confirms, one after the fist opens again
    assert len(closes) == 1
    assert len(opens) == 2
    assert opens[0].t < closes[0].t < opens[1].t


def test_flicker_is_rejected_but_a_real_hold_still_fires():
    cfg = GestureConfig()

    def shape(t: float) -> str:
        if t < 0.3:
            return "open"  # stable baseline
        if t < 1.3:
            # toggle every nominal frame (~0.033s), well under fist_min_hold (0.06s)
            return "fist" if int(t / (1 / 30)) % 2 == 0 else "open"
        return "fist"  # settles closed for good

    spec = HandSpec(side="left", width=WIDTH, palm=hold_path(PALM), shape=shape)
    frames = motion([spec], t_start=0.0, duration=1.6, rng=random.Random(11), jitter=0.0)
    events, _ = _run(frames, cfg)

    closes = [e for e in events if isinstance(e, FistClose)]
    # exactly the one real close at the end, none from the flicker in between
    assert len(closes) == 1
    assert closes[0].t > 1.3


def test_refractory_suppresses_a_close_that_follows_too_soon():
    cfg = GestureConfig()
    # open(0-0.15) close(0.15-0.30)->fires  open(0.30-0.40)  close(0.40-0.50)->suppressed
    # open(0.50-0.60)  close(0.60-0.90)->fires (refractory has elapsed)
    shape = _windowed_shape([(0.15, 0.30), (0.40, 0.50), (0.60, 0.90)])
    spec = HandSpec(side="left", width=WIDTH, palm=hold_path(PALM), shape=shape)
    frames = motion([spec], t_start=0.0, duration=1.0, rng=random.Random(12))
    events, _ = _run(frames, cfg)

    closes = [e for e in events if isinstance(e, FistClose)]
    assert len(closes) == 2
    assert closes[1].t - closes[0].t > cfg.fist_refractory


def test_no_spurious_close_when_hand_reappears_already_closed():
    cfg = GestureConfig()

    def shape(t: float) -> str:
        if t < 0.5:
            return "absent"
        return "fist"  # reappears already closed, with no prior open seen

    spec = HandSpec(side="left", width=WIDTH, palm=hold_path(PALM), shape=shape)
    frames = motion([spec], t_start=0.0, duration=1.0, rng=random.Random(13))
    events, detector = _run(frames, cfg)

    assert [e for e in events if isinstance(e, FistClose)] == []
    assert detector.closed is True  # state is known even though no edge fired
