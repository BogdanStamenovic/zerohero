"""MediaPipe HandLandmarker wrapper: BGR frame + timestamp -> Frame.

See ARCHITECTURE.md "Coordinate conventions" for the mirror/handedness
contract this implements.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision

from zerohero.config import CameraConfig
from zerohero.events import Frame, Hand, Landmark, Side


def _map_side(label: str, mirror: bool) -> Side:
    """Map a MediaPipe handedness label to the user's actual side.

    MediaPipe's handedness assumes a selfie-style (mirrored) camera. When
    `mirror` is True the frame has already been flipped to match that
    assumption, so the label is correct as-is. When it's False the frame
    fed to MediaPipe is the raw, un-mirrored camera image, which is the
    opposite of what the label assumes, so left/right must be swapped.
    """
    side = label.lower()
    if side not in ("left", "right"):
        side = "right"
    if not mirror:
        side = "left" if side == "right" else "right"
    return side  # type: ignore[return-value]


class HandTracker:
    def __init__(self, model_path: str | Path, cfg: CameraConfig) -> None:
        self.cfg = cfg
        options = mp_vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=cfg.max_hands,
            min_hand_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._last_ms = -1

    def process(self, bgr: np.ndarray, t: float) -> Frame:
        image = cv2.flip(bgr, 1) if self.cfg.mirror else bgr
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # detect_for_video requires a strictly increasing millisecond
        # timestamp and raises if it goes backwards or repeats. `t` is
        # time.monotonic(), which is precise enough in seconds but can
        # collide at millisecond resolution between two fast frames (and
        # replay files can carry duplicate/out-of-order t). Clamp forward.
        ms = max(self._last_ms + 1, int(t * 1000))
        self._last_ms = ms
        result = self._landmarker.detect_for_video(mp_image, ms)

        world_lists = result.hand_world_landmarks or []
        hands: list[Hand] = []
        for i, lm_list in enumerate(result.hand_landmarks):
            category = result.handedness[i][0]
            side = _map_side(category.category_name, self.cfg.mirror)
            landmarks = [Landmark(p.x, p.y, p.z) for p in lm_list]
            world = None
            if i < len(world_lists) and world_lists[i]:
                world = [Landmark(p.x, p.y, p.z) for p in world_lists[i]]
            hands.append(Hand(side=side, score=category.score, landmarks=landmarks, world=world))

        height, width = image.shape[:2]
        return Frame(t=t, width=width, height=height, hands=hands, image=image)

    def warmup(self) -> None:
        """Run one throwaway detection so the first real frame isn't slow."""
        black = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8)
        self.process(black, 0.0)

    def close(self) -> None:
        self._landmarker.close()
