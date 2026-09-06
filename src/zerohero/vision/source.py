"""FrameSource protocol: unifies live camera and replay-file iteration.

Everything downstream of `open_source()` (engine/app.py's main loop) only
ever depends on this protocol, so it runs identically against a camera or a
recorded session. See ARCHITECTURE.md "Pipeline" and "vision".
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable

from zerohero.config import CameraConfig, Config
from zerohero.events import Frame
from zerohero.vision.camera import Camera
from zerohero.vision.hands import HandTracker
from zerohero.vision.replay import ReplaySource

__all__ = ["FrameSource", "CameraSource", "ReplaySource", "open_source"]


@runtime_checkable
class FrameSource(Protocol):
    def frames(self) -> Iterator[Frame]: ...
    def close(self) -> None: ...
    @property
    def fps(self) -> float: ...


class CameraSource:
    """Camera + HandTracker composed into one Frame-yielding source."""

    def __init__(self, cfg: CameraConfig, model_path: str | Path) -> None:
        self.camera = Camera(cfg)
        self.tracker = HandTracker(model_path, cfg)

    def frames(self) -> Iterator[Frame]:
        self.camera.start()
        self.tracker.warmup()
        while True:
            got = self.camera.wait(timeout=1.0)
            if got is None:
                continue
            t, bgr = got
            yield self.tracker.process(bgr, t)

    def close(self) -> None:
        self.camera.stop()
        self.tracker.close()

    @property
    def fps(self) -> float:
        return self.camera.fps


def open_source(cfg: Config) -> FrameSource:
    from zerohero.paths import hand_model_path

    if cfg.replay is not None:
        return ReplaySource(cfg.replay, realtime=cfg.replay_realtime)
    return CameraSource(cfg.camera, hand_model_path())
