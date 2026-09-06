from __future__ import annotations

import random

from synth_hands import HandSpec, drift_path, hold_path, motion

from zerohero.config import GestureConfig
from zerohero.gestures.features import HandFeatures
from zerohero.gestures.track import HandTrack
from zerohero.gestures.vibe import VibeMeter

WIDTH = 0.15


def _drive(frames, cfg, sides=("left", "right")):
    """Feeds frames through real HandTracks + a VibeMeter, returns the list
    of (t, value) samples over the whole run."""
    tracks = {s: HandTrack(s, cfg) for s in sides}
    vibe = VibeMeter(cfg)
    samples = []
    for f in frames:
        for s in sides:
            hand = f.hand(s)
            if hand is None:
                tracks[s].mark_missing(f.t)
            else:
                tracks[s].update(HandFeatures.from_hand(hand), f.t)
        vibe.update(list(tracks.values()), f.t)
        samples.append((f.t, vibe.value))
    return samples


def test_vibe_is_higher_during_vigorous_motion_than_at_rest():
    cfg = GestureConfig()

    rest_spec = HandSpec(side="right", width=WIDTH, palm=hold_path((0.5, 0.5)), shape=lambda _t: "open")
    rest_frames = motion([rest_spec], t_start=0.0, duration=2.0, rng=random.Random(30))
    rest_values = [v for _, v in _drive(rest_frames, cfg, sides=("right",))]

    # 8 hand-widths/sec is a vigorous, sustained motion
    vig_palm = drift_path((0.1, 0.1), (0.0, 8.0 * WIDTH), t0=0.0)
    vig_spec = HandSpec(side="right", width=WIDTH, palm=vig_palm, shape=lambda _t: "open")
    vig_frames = motion([vig_spec], t_start=0.0, duration=2.0, rng=random.Random(31))
    vig_values = [v for _, v in _drive(vig_frames, cfg, sides=("right",))]

    assert max(vig_values) > max(rest_values)
    assert max(vig_values[-30:]) > 0.5  # steady-state, well after the window fills


def test_vibe_decays_to_zero_within_two_seconds_after_motion_stops():
    cfg = GestureConfig()

    def shape(t: float) -> str:
        return "open" if t < 1.0 else "absent"

    def palm(t: float) -> tuple[float, float]:
        return (0.1 + min(t, 1.0) * 8.0 * WIDTH, 0.3)

    spec = HandSpec(side="right", width=WIDTH, palm=palm, shape=shape)
    # 1s of vigorous motion, then 2s with the hand gone entirely
    frames = motion([spec], t_start=0.0, duration=3.0, rng=random.Random(32))
    samples = _drive(frames, cfg, sides=("right",))

    during = [v for t, v in samples if 0.5 <= t < 1.0]
    assert max(during) > 0.3

    tail = [v for t, v in samples if t >= 3.0 - 0.05]
    assert tail and tail[-1] < 0.05
