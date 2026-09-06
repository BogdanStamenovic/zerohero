from __future__ import annotations

import random
import time

from synth_hands import HandSpec, hold_path, motion, piecewise, stroke_path

from zerohero.config import Config
from zerohero.events import FistClose, Strum
from zerohero.gestures.pipeline import GesturePipeline

WIDTH = 0.15


def _chain(p0, strokes):
    pieces = []
    p = p0
    for t0, duration, delta in strokes:
        pieces.append((t0, t0 + duration, stroke_path(p, delta, t0, duration)))
        p = (p[0] + delta[0], p[1] + delta[1])
    return piecewise(pieces)


def test_end_to_end_combined_scenario_strums_and_fists_in_order():
    cfg = Config()  # strum_hand="right", fist_hand="left" by default
    dy = 1.5 * WIDTH

    right_strokes = [
        (0.3, 0.12, (0.0, dy)),  # down
        (1.2, 0.12, (0.0, -dy)),  # up
        (2.1, 0.12, (0.0, dy)),  # down
        (2.52, 0.12, (0.0, -dy)),  # up
    ]
    right_path = _chain((0.5, 0.5), right_strokes)
    right_spec = HandSpec(side="right", width=WIDTH, palm=right_path, shape=lambda _t: "open")

    closed_windows = [(0.75, 0.90), (1.65, 1.80)]

    def left_shape(t: float) -> str:
        return "fist" if any(a <= t < b for a, b in closed_windows) else "open"

    left_spec = HandSpec(side="left", width=WIDTH, palm=hold_path((0.2, 0.5)), shape=left_shape)

    frames = motion([right_spec, left_spec], t_start=0.0, duration=3.0, rng=random.Random(99))

    pipeline = GesturePipeline(cfg)
    all_events = []
    for f in frames:
        all_events.extend(pipeline.update(f))

    strums = [e for e in all_events if isinstance(e, Strum)]
    closes = [e for e in all_events if isinstance(e, FistClose)]
    assert len(strums) == 4
    assert [s.direction for s in strums] == ["down", "up", "down", "up"]
    assert len(closes) == 2

    # right order: strum, close, strum, close, strum, strum -- fist closes land
    # strictly between the surrounding strums, matching the scenario timeline.
    relevant = [e for e in all_events if isinstance(e, Strum | FistClose)]
    kinds = ["strum" if isinstance(e, Strum) else "close" for e in relevant]
    assert kinds == ["strum", "close", "strum", "close", "strum", "strum"]
    assert all(relevant[i].t <= relevant[i + 1].t for i in range(len(relevant) - 1))


def test_pipeline_update_is_well_under_a_millisecond_on_average():
    cfg = Config()
    dy = 1.5 * WIDTH
    path = _chain((0.5, 0.5), [(0.5, 0.12, (0.0, dy)), (1.0, 0.12, (0.0, -dy))])
    right_spec = HandSpec(side="right", width=WIDTH, palm=path, shape=lambda _t: "open")
    left_spec = HandSpec(
        side="left", width=WIDTH, palm=hold_path((0.2, 0.5)), shape=lambda t: "fist" if 0.6 <= t < 0.8 else "open"
    )
    frames = motion([right_spec, left_spec], t_start=0.0, duration=10.0, rng=random.Random(100))
    frames = frames[:300]
    assert len(frames) == 300

    pipeline = GesturePipeline(cfg)
    start = time.perf_counter()
    for f in frames:
        pipeline.update(f)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / len(frames)) * 1000
    assert avg_ms < 1.0, f"average pipeline.update time {avg_ms:.4f} ms exceeds 1 ms budget"
