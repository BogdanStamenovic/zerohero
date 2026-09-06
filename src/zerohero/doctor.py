"""`zerohero doctor`: check every dependency; `zerohero calibrate`: live gesture readout."""

from __future__ import annotations

import ctypes.util
import platform
import socket
import sys
import time

from zerohero.config import Config
from zerohero.paths import hand_model_path, soundfont_path


def _ok(label: str, good: bool, detail: str = "") -> bool:
    mark = "ok " if good else "FAIL"
    print(f"[{mark}] {label}" + (f": {detail}" if detail else ""))
    return good


def run_doctor() -> int:
    good = True
    print(f"zerohero doctor on {platform.system()} {platform.release()}, Python {platform.python_version()}")

    good &= _ok("python 3.12", sys.version_info[:2] == (3, 12), platform.python_version())

    try:
        import mediapipe as mp

        good &= _ok("mediapipe", True, mp.__version__)
    except Exception as e:
        good &= _ok("mediapipe", False, str(e))

    try:
        import cv2

        _ok("opencv", True, cv2.__version__)
    except Exception as e:
        good &= _ok("opencv", False, str(e))

    model = hand_model_path()
    good &= _ok("hand model", model.exists(), str(model) if model.exists() else "missing, run `zerohero setup`")

    sf = soundfont_path()
    lib = ctypes.util.find_library("fluidsynth")
    if lib:
        _ok("libfluidsynth", True, lib)
        _ok(
            "soundfont",
            sf.exists(),
            str(sf) if sf.exists() else "missing, run `zerohero setup` (numpy synth will be used)",
        )
    else:
        _ok(
            "libfluidsynth", False, "not found; built-in numpy synth will be used (install fluidsynth for better sound)"
        )

    try:
        import sounddevice as sd

        dev = sd.query_devices(kind="output")
        _ok("audio output", True, f"{dev['name']}")
    except Exception as e:
        _ok("audio output", False, str(e))

    _ok(
        "bluetooth sockets",
        hasattr(socket, "AF_BLUETOOTH"),
        "AF_BLUETOOTH available" if hasattr(socket, "AF_BLUETOOTH") else "not in this Python build; use WiFi/TCP link",
    )

    # Camera last: it takes a second and needs the model.
    try:
        from zerohero.config import CameraConfig
        from zerohero.vision.camera import Camera

        cam = Camera(CameraConfig())
        cam.start()
        t0 = time.monotonic()
        n = 0
        while time.monotonic() - t0 < 1.0:
            if cam.wait(0.2) is not None:
                n += 1
        cam.stop()
        good &= _ok("camera", n > 5, f"{n} frames in 1 s")
    except Exception as e:
        good &= _ok("camera", False, str(e))

    if good and model.exists():
        try:
            import numpy as np

            from zerohero.config import CameraConfig
            from zerohero.vision.hands import HandTracker

            tr = HandTracker(model, CameraConfig())
            black = np.zeros((480, 640, 3), np.uint8)
            tr.process(black, time.monotonic())
            t0 = time.monotonic()
            tr.process(black, time.monotonic())
            _ok("hand tracker", True, f"{(time.monotonic() - t0) * 1000:.0f} ms per frame on a blank image")
            tr.close()
        except Exception as e:
            good &= _ok("hand tracker", False, str(e))

    print("all good" if good else "problems found", file=sys.stderr)
    return 0 if good else 1


def run_calibrate(cfg: Config) -> int:
    """Show the camera with hand readouts and print every gesture event. Raise your right hand: it must say right."""
    from zerohero.events import Beat, FistClose, FistOpen, Strum
    from zerohero.gestures.pipeline import GesturePipeline
    from zerohero.vision.source import open_source

    cfg.progression = cfg.progression or "C"
    source = open_source(cfg)
    pipeline = GesturePipeline(cfg)
    from zerohero.ui.overlay import OverlayState
    from zerohero.ui.view import open_view

    view = open_view(cfg, "zerohero calibrate")
    last_event = ""
    last_t = 0.0
    last_print = 0.0
    print("q to quit. Events print below; hands print once a second.")
    try:
        for frame in source.frames():
            for ev in pipeline.update(frame):
                if isinstance(ev, Strum):
                    last_event = f"STRUM {ev.hand} {ev.direction} {ev.intensity:.2f}"
                elif isinstance(ev, FistClose):
                    last_event = f"FIST {ev.hand}"
                elif isinstance(ev, FistOpen):
                    last_event = f"OPEN {ev.hand}"
                elif isinstance(ev, Beat):
                    last_event = f"BEAT {ev.hand} {ev.direction} {ev.intensity:.2f}{' fist' if ev.closed else ''}"
                last_t = frame.t
                print(f"{frame.t:10.3f} {last_event}")
            if frame.t - last_print > 1.0:
                last_print = frame.t
                parts = []
                for side, tr in pipeline.tracks.items():
                    if tr.present:
                        parts.append(
                            f"{side}: palm=({tr.palm[0]:.2f},{tr.palm[1]:.2f}) "
                            f"v=({tr.velocity[0]:+.1f},{tr.velocity[1]:+.1f}) "
                            f"closed={tr.closed_score:.2f}"
                        )
                print(
                    f"{frame.t:10.3f} hands: " + ("; ".join(parts) if parts else "none") + f" vibe={pipeline.vibe:.2f}"
                )
            if view is not None:
                tracks: dict[str, tuple[float, float, float, float, bool]] = {
                    s: (tr.palm[0], tr.palm[1], tr.velocity[0], tr.velocity[1], pipeline.closed(s))
                    for s, tr in pipeline.tracks.items()
                    if tr.present
                }
                state = OverlayState(
                    mode="calibrate",
                    chords=[],
                    index=0,
                    vibe=pipeline.vibe,
                    tracks=tracks,
                    last_event=last_event,
                    last_event_t=last_t,
                    fps=source.fps,
                    link="",
                )
                view.draw(frame, state)
                if view.poll_key() in ("q", "esc"):
                    break
    finally:
        if view is not None:
            view.close()
        source.close()
    return 0
