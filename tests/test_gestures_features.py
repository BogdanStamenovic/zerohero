from __future__ import annotations

import math

from synth_hands import fist, open_hand

from zerohero.gestures.features import HandFeatures


def test_open_hand_is_fully_extended_and_not_closed():
    hand = open_hand("right", (0.5, 0.4), 0.15, noise_sigma=0.0)
    f = HandFeatures.from_hand(hand)
    assert f.extended == (True, True, True, True, True)
    assert f.closed_score == 0.0


def test_fist_is_fully_curled_and_closed():
    hand = fist("right", (0.5, 0.4), 0.15, noise_sigma=0.0)
    f = HandFeatures.from_hand(hand)
    assert f.extended == (False, False, False, False, False)
    assert f.closed_score == 1.0


def test_palm_matches_requested_position():
    palm = (0.37, 0.62)
    hand = open_hand("left", palm, 0.12, noise_sigma=0.0)
    f = HandFeatures.from_hand(hand)
    assert math.isclose(f.palm[0], palm[0], abs_tol=1e-9)
    assert math.isclose(f.palm[1], palm[1], abs_tol=1e-9)


def test_width_matches_wrist_to_middle_mcp_distance():
    hand = open_hand("right", (0.5, 0.5), 0.2, noise_sigma=0.0)
    f = HandFeatures.from_hand(hand)
    assert math.isclose(f.width, 0.2, rel_tol=1e-6)
