"""Unit tests for the mirror/handedness mapping, factored out of HandTracker
as `_map_side` so it's testable without a real MediaPipe landmarker.
See ARCHITECTURE.md "Coordinate conventions".
"""

from __future__ import annotations

from zerohero.vision.hands import _map_side


def test_mirror_true_keeps_label_as_is() -> None:
    # Frame was already flipped, matching MediaPipe's selfie-camera assumption.
    assert _map_side("Left", mirror=True) == "left"
    assert _map_side("Right", mirror=True) == "right"


def test_mirror_false_swaps_label() -> None:
    # Raw, un-mirrored frame: MediaPipe's label is backwards relative to the user.
    assert _map_side("Left", mirror=False) == "right"
    assert _map_side("Right", mirror=False) == "left"


def test_unknown_label_defaults_to_right() -> None:
    assert _map_side("Unknown", mirror=True) == "right"
