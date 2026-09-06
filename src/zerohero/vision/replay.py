"""JSONL record/replay of Frames, so the pipeline runs without a camera.

See ARCHITECTURE.md "vision": replay yields the same Frames a camera would,
minus `image`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

from zerohero.events import Frame


class FrameRecorder:
    def __init__(self, path: str | Path) -> None:
        self._f = open(path, "w", encoding="utf-8")

    def write(self, frame: Frame) -> None:
        self._f.write(json.dumps(frame.to_json()) + "\n")

    def close(self) -> None:
        self._f.close()


class ReplaySource:
    def __init__(self, path: str | Path, realtime: bool = True) -> None:
        self.path = Path(path)
        self.realtime = realtime
        with self.path.open(encoding="utf-8") as f:
            self._lines = [line for line in f.read().splitlines() if line.strip()]
        self.fps = self._estimate_fps()

    def _estimate_fps(self) -> float:
        timestamps = [json.loads(line)["t"] for line in self._lines]
        gaps = [b - a for a, b in zip(timestamps, timestamps[1:], strict=False) if b > a]
        if not gaps:
            return 0.0
        gaps.sort()
        median_gap = gaps[len(gaps) // 2]
        return 1.0 / median_gap if median_gap > 0 else 0.0

    def frames(self) -> Iterator[Frame]:
        prev_recorded_t: float | None = None
        for line in self._lines:
            frame = Frame.from_json(json.loads(line))
            if self.realtime:
                if prev_recorded_t is not None:
                    gap = frame.t - prev_recorded_t
                    if gap > 0:
                        time.sleep(gap)
                prev_recorded_t = frame.t
                # Rewrite to current time so downstream code (velocity, event
                # timestamps, scheduler deadlines) sees live-looking t, not
                # whatever monotonic clock reading the recording session had.
                frame.t = time.monotonic()
            yield frame

    def close(self) -> None:
        pass
