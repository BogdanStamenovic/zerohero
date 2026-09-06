"""Frame -> gesture state machines. See ARCHITECTURE.md "gestures"."""

from __future__ import annotations

from zerohero.gestures.conduct import BeatDetector
from zerohero.gestures.features import HandFeatures
from zerohero.gestures.fist import FistDetector
from zerohero.gestures.pipeline import GesturePipeline
from zerohero.gestures.strum import StrumDetector
from zerohero.gestures.track import HandTrack
from zerohero.gestures.vibe import VibeMeter

__all__ = [
    "BeatDetector",
    "FistDetector",
    "GesturePipeline",
    "HandFeatures",
    "HandTrack",
    "StrumDetector",
    "VibeMeter",
]
