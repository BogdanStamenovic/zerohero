"""Capture thread: reads frames continuously, keeps only the newest one.

See ARCHITECTURE.md "Pipeline" -- frames are dropped, never queued, so
latency does not build up when inference is slower than the camera.
"""

from __future__ import annotations

import sys
import threading
import time

import cv2
import numpy as np

from zerohero.config import CameraConfig

_OPEN_TIMEOUT = 3.0


class CameraError(Exception):
    """The camera device could not be opened or produced no frame in time."""


class Camera:
    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._frame: tuple[float, np.ndarray] | None = None
        self._consumed = True
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._fps_ema = 0.0
        self._last_read_t: float | None = None

    def start(self) -> None:
        # V4L2 is the backend that actually honours FOURCC/size/fps requests
        # on Linux; the auto-picked default backend silently ignores some of
        # them on certain drivers.
        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.cfg.index, backend)
        if not cap.isOpened():
            raise CameraError(
                f"could not open camera index {self.cfg.index}. "
                f"Check that the camera exists (Linux: /dev/video{self.cfg.index}) and no other process holds it."
            )
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # type: ignore[attr-defined]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)

        deadline = time.monotonic() + _OPEN_TIMEOUT
        ok, frame = False, None
        while time.monotonic() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None:
                break
        if not ok or frame is None:
            cap.release()
            raise CameraError(
                f"camera index {self.cfg.index} opened but produced no frame within "
                f"{_OPEN_TIMEOUT:.0f}s. Is the sensor in use by another process?"
            )

        self._cap = cap
        t0 = time.monotonic()
        self._frame = (t0, frame)
        self._consumed = False
        self._last_read_t = t0

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="zerohero-camera", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        assert self._cap is not None
        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            t = time.monotonic()
            if not ok or frame is None:
                continue
            if self._last_read_t is not None:
                dt = t - self._last_read_t
                if dt > 1e-6:
                    inst_fps = 1.0 / dt
                    self._fps_ema = inst_fps if self._fps_ema == 0.0 else 0.9 * self._fps_ema + 0.1 * inst_fps
            self._last_read_t = t
            with self._cond:
                self._frame = (t, frame)
                self._consumed = False
                self._cond.notify_all()

    def latest(self) -> tuple[float, np.ndarray] | None:
        """Newest frame not yet handed out, or None. Never returns the same frame twice."""
        with self._lock:
            if self._frame is None or self._consumed:
                return None
            self._consumed = True
            return self._frame

    def wait(self, timeout: float | None = None) -> tuple[float, np.ndarray] | None:
        """Block until a new frame arrives (or timeout), then hand it out."""
        with self._cond:
            if self._frame is None or self._consumed:
                self._cond.wait(timeout=timeout)
            if self._frame is None or self._consumed:
                return None
            self._consumed = True
            return self._frame

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def fps(self) -> float:
        return self._fps_ema
