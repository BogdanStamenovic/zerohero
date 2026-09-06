from __future__ import annotations

import time
from pathlib import Path

from zerohero.events import Frame, Hand, Landmark
from zerohero.vision.replay import FrameRecorder, ReplaySource


def _hand(side: str, seed: float) -> Hand:
    landmarks = [Landmark(x=seed + i * 0.01, y=seed + i * 0.02, z=seed + i * 0.03) for i in range(21)]
    return Hand(side=side, score=0.95, landmarks=landmarks)


def _frame(t: float, seed: float) -> Frame:
    return Frame(t=t, width=640, height=480, hands=[_hand("left", seed), _hand("right", seed + 1)])


def test_record_replay_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    frames = [_frame(0.0, 0.0), _frame(0.033, 1.0), _frame(0.066, 2.0)]

    recorder = FrameRecorder(path)
    for f in frames:
        recorder.write(f)
    recorder.close()

    source = ReplaySource(path, realtime=False)
    replayed = list(source.frames())
    source.close()

    assert len(replayed) == len(frames)
    for original, back in zip(frames, replayed, strict=True):
        assert back.t == original.t
        assert back.width == original.width
        assert back.height == original.height
        assert back.image is None  # replay never carries pixels
        assert {h.side for h in back.hands} == {h.side for h in original.hands}
        for oh in original.hands:
            bh = back.hand(oh.side)
            assert bh is not None
            assert [(lm.x, lm.y, lm.z) for lm in bh.landmarks] == [(lm.x, lm.y, lm.z) for lm in oh.landmarks]


def test_non_realtime_replay_does_not_sleep(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    recorder = FrameRecorder(path)
    recorder.write(_frame(0.0, 0.0))
    recorder.write(_frame(1.0, 1.0))  # 1s gap: would be very slow if not skipped
    recorder.close()

    source = ReplaySource(path, realtime=False)
    start = time.monotonic()
    replayed = list(source.frames())
    elapsed = time.monotonic() - start

    assert len(replayed) == 2
    assert elapsed < 0.2


def test_realtime_replay_reproduces_intervals(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    recorder = FrameRecorder(path)
    recorder.write(_frame(0.0, 0.0))
    recorder.write(_frame(0.1, 1.0))  # 100ms after the first
    recorder.close()

    source = ReplaySource(path, realtime=True)
    start = time.monotonic()
    replayed = list(source.frames())
    elapsed = time.monotonic() - start

    assert len(replayed) == 2
    assert elapsed >= 0.08
    # t is rewritten to look live: monotonically increasing, near "now".
    assert replayed[1].t >= replayed[0].t
    assert abs(replayed[-1].t - time.monotonic()) < 0.5
