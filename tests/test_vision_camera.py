"""Real-hardware smoke test. Skipped by default; needs an actual webcam.

Run with: ZEROHERO_CAMERA_TEST=1 pytest tests/test_vision_camera.py
"""

from __future__ import annotations

import os

import pytest

from zerohero.config import CameraConfig
from zerohero.vision.camera import Camera

pytestmark = pytest.mark.skipif(
    os.environ.get("ZEROHERO_CAMERA_TEST") != "1",
    reason="set ZEROHERO_CAMERA_TEST=1 to run against the real camera",
)


def test_real_camera_delivers_frames_at_a_reasonable_fps() -> None:
    cam = Camera(CameraConfig())
    cam.start()
    try:
        got = 0
        while got < 30:
            frame = cam.wait(timeout=3.0)
            assert frame is not None, "camera stalled"
            got += 1
        assert cam.fps > 10, f"measured fps too low: {cam.fps}"
    finally:
        cam.stop()
