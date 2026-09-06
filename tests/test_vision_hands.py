"""Unit tests for the mirror/handedness mapping, factored out of HandTracker
as `_map_side` so it's testable without a real MediaPipe landmarker.
See ARCHITECTURE.md "Coordinate conventions".
"""

from __future__ import annotations

from zerohero.vision.hands import HandTracker


def _hand(x: float):
    from zerohero.events import Hand, Landmark

    return Hand(side="right", score=1.0, landmarks=[Landmark(x, 0.5, 0.0) for _ in range(21)])


def test_sides_assigned_by_position_two_hands():
    tr = HandTracker.__new__(HandTracker)
    tr._last_palms = []
    hands = [_hand(0.8), _hand(0.2)]
    tr._assign_sides(hands)
    assert [h.side for h in hands] == ["right", "left"]


def test_single_hand_keeps_its_side_when_it_crosses_centre():
    tr = HandTracker.__new__(HandTracker)
    tr._last_palms = []
    a = [_hand(0.7)]
    tr._assign_sides(a)
    assert a[0].side == "right"
    b = [_hand(0.45)]  # moved left of centre but close to where it was
    tr._assign_sides(b)
    assert b[0].side == "right"
    c = [_hand(0.1)]  # jumped far: new hand, decide by position
    tr._assign_sides(c)
    assert c[0].side == "left"
