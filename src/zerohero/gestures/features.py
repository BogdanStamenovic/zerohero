"""Per-hand geometric features derived from a single frame's landmarks.

Everything here is a pure function of one `Hand` -- no history, no time. See
ARCHITECTURE.md "gestures" and "Coordinate conventions" for the contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from zerohero.events import (
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_TIP,
    RING_MCP,
    RING_PIP,
    RING_TIP,
    THUMB_IP,
    THUMB_TIP,
    WRIST,
    Hand,
    Landmark,
)

# Fingers must be this fraction farther from the wrist (tip vs. PIP) to count
# as extended. Without a margin, a nearly-straight-but-relaxed finger flickers
# across the boundary every frame.
_GESTURE_MIN_SCORE = 0.5
_TIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
_EXTEND_MARGIN = 1.05

_NON_THUMB_TIPS_PIPS = (
    (INDEX_TIP, INDEX_PIP),
    (MIDDLE_TIP, MIDDLE_PIP),
    (RING_TIP, RING_PIP),
    (PINKY_TIP, PINKY_PIP),
)


def _dist(a: Landmark, b: Landmark) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


@dataclass(frozen=True, slots=True)
class HandFeatures:
    palm: tuple[float, float]
    width: float
    extended: tuple[bool, bool, bool, bool, bool]  # thumb, index, middle, ring, pinky
    closed_score: float  # 0..1, fraction of the four non-thumb fingers curled
    # Fingertips relative to the palm centre, in hand widths: the shape of the
    # hand independent of where it is. Frame-to-frame change = finger activity.
    tips: tuple[tuple[float, float], ...] = ()

    @classmethod
    def from_hand(cls, hand: Hand) -> HandFeatures:
        lm = hand.landmarks
        wrist = lm[WRIST]

        palm_pts = (lm[WRIST], lm[INDEX_MCP], lm[MIDDLE_MCP], lm[RING_MCP], lm[PINKY_MCP])
        palm = (sum(p.x for p in palm_pts) / len(palm_pts), sum(p.y for p in palm_pts) / len(palm_pts))

        width = _dist(wrist, lm[MIDDLE_MCP])
        # A zero-width hand (degenerate landmarks) would make every ratio
        # below divide by ~0 and blow up; treat it as a non-detection instead.
        safe_width = width if width > 1e-6 else 1e-6

        thumb_extended = _dist(lm[THUMB_TIP], lm[PINKY_MCP]) > _EXTEND_MARGIN * _dist(lm[THUMB_IP], lm[PINKY_MCP])

        curled = 0
        finger_flags: list[bool] = []
        for tip_i, pip_i in _NON_THUMB_TIPS_PIPS:
            tip_dist = _dist(lm[tip_i], wrist)
            pip_dist = _dist(lm[pip_i], wrist)
            is_extended = tip_dist > _EXTEND_MARGIN * pip_dist
            finger_flags.append(is_extended)
            if not is_extended:
                curled += 1

        closed_score = curled / len(_NON_THUMB_TIPS_PIPS)
        # The recognizer's label beats geometry: a flat hand pointing at the
        # camera collapses in projection and reads as curled. "None" (unsure,
        # common mid-motion) maps to 0.5 so hysteresis holds the last state
        # instead of releasing a grip in the middle of a hit.
        if hand.gesture_score >= _GESTURE_MIN_SCORE:
            if hand.gesture == "Closed_Fist":
                closed_score = 1.0
            elif hand.gesture != "None":
                closed_score = 0.0
            else:
                closed_score = 0.5

        index_e, middle_e, ring_e, pinky_e = finger_flags
        extended = (thumb_extended, index_e, middle_e, ring_e, pinky_e)
        tips = tuple(((lm[i].x - palm[0]) / safe_width, (lm[i].y - palm[1]) / safe_width) for i in _TIPS)
        return cls(palm=palm, width=safe_width, extended=extended, closed_score=closed_score, tips=tips)
