"""MediaPipe GestureRecognizer wrapper: BGR frame + timestamp -> Frame.

The gesture recognizer is the hand landmarker plus a small classifier that
labels each hand (Closed_Fist, Open_Palm, ...). Landmark geometry alone could
not tell a flat hand pointing at the camera from a fist in live recordings,
the trained label can, and it costs nothing extra (same ~40 ms per frame).

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


def _palm_x(hand: Hand) -> float:
    return sum(hand.landmarks[i].x for i in (0, 5, 9, 13, 17)) / 5


class HandTracker:
    def __init__(self, model_path: str | Path, cfg: CameraConfig) -> None:
        self.cfg = cfg
        options = mp_vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=cfg.max_hands,
            min_hand_detection_confidence=cfg.min_detection_confidence,
            min_hand_presence_confidence=cfg.min_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._gamma_lut = None
        if abs(cfg.gamma - 1.0) > 1e-3:
            table = [min(255, int((i / 255.0) ** cfg.gamma * 255 + 0.5)) for i in range(256)]
            self._gamma_lut = np.array(table, np.uint8)
        self._landmarker = mp_vision.GestureRecognizer.create_from_options(options)
        self._last_ms = -1
        self._last_palms: list[tuple[Side, float]] = []

    def _assign_sides(self, hands: list[Hand]) -> None:
        """Sides by position, not by the model's label.

        MediaPipe's Left/Right is the chirality of the 2D projection, so it
        flips when the user shows the back of the hand instead of the palm,
        which happens constantly while strumming or conducting. In the mirror
        view the user's left hand is the one further left, so with two hands
        the smaller x is "left". With one hand, keep whatever side that hand
        had last frame (nearest previous palm), else split at the centre.
        """
        palms = [_palm_x(h) for h in hands]
        if len(hands) >= 2:
            order = sorted(range(len(hands)), key=lambda i: palms[i])
            for rank, i in enumerate(order):
                hands[i].side = "left" if rank == 0 else "right"
        elif len(hands) == 1:
            x = palms[0]
            side: Side | None = None
            if self._last_palms:
                prev_side, prev_x = min(self._last_palms, key=lambda p: abs(p[1] - x))
                if abs(prev_x - x) < 0.25:
                    side = prev_side
            hands[0].side = side or ("left" if x < 0.5 else "right")
        self._last_palms = [(h.side, palms[i]) for i, h in enumerate(hands)]

    def process(self, bgr: np.ndarray, t: float) -> Frame:
        image = cv2.flip(bgr, 1) if self.cfg.mirror else bgr
        if self._gamma_lut is not None:
            image = cv2.LUT(image, self._gamma_lut)  # the overlay shows the brightened frame too
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # detect_for_video requires a strictly increasing millisecond
        # timestamp and raises if it goes backwards or repeats. `t` is
        # time.monotonic(), which is precise enough in seconds but can
        # collide at millisecond resolution between two fast frames (and
        # replay files can carry duplicate/out-of-order t). Clamp forward.
        ms = max(self._last_ms + 1, int(t * 1000))
        self._last_ms = ms
        result = self._landmarker.recognize_for_video(mp_image, ms)

        world_lists = result.hand_world_landmarks or []
        hands: list[Hand] = []
        gestures = result.gestures or []
        for i, lm_list in enumerate(result.hand_landmarks):
            category = result.handedness[i][0]
            landmarks = [Landmark(p.x, p.y, p.z) for p in lm_list]
            world = None
            if i < len(world_lists) and world_lists[i]:
                world = [Landmark(p.x, p.y, p.z) for p in world_lists[i]]
            gesture, gscore = "None", 0.0
            if i < len(gestures) and gestures[i]:
                gesture, gscore = gestures[i][0].category_name, float(gestures[i][0].score)
            hands.append(
                Hand(
                    side="right",
                    score=category.score,
                    landmarks=landmarks,
                    world=world,
                    gesture=gesture,
                    gesture_score=gscore,
                )
            )
        self._assign_sides(hands)

        height, width = image.shape[:2]
        return Frame(t=t, width=width, height=height, hands=hands, image=image)

    def warmup(self) -> None:
        """Run one throwaway detection so the first real frame isn't slow."""
        black = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8)
        self.process(black, 0.0)

    def close(self) -> None:
        self._landmarker.close()
