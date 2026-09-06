"""Frame -> gesture events. The one entry point the rest of the app calls
into. See ARCHITECTURE.md "gestures" and "Pipeline"."""

from __future__ import annotations

from zerohero.config import Config
from zerohero.events import Frame, GestureEvent, Side
from zerohero.gestures.conduct import BeatDetector
from zerohero.gestures.features import HandFeatures
from zerohero.gestures.fist import FistDetector
from zerohero.gestures.strum import StrumDetector
from zerohero.gestures.track import HandTrack
from zerohero.gestures.vibe import VibeMeter

_SIDES: tuple[Side, Side] = ("left", "right")


class GesturePipeline:
    """Owns one `HandTrack` per side and every detector, and runs them in a
    fixed order each frame: fist edges, then the strum, then conducting
    beats. Fist first because `BeatDetector` needs the (possibly just
    updated) closed state for its `closed` field; strum before beat is
    otherwise an arbitrary but stable choice per the contract.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        gcfg = cfg.gesture

        self.tracks: dict[Side, HandTrack] = {side: HandTrack(side, gcfg) for side in _SIDES}
        self._fists: dict[Side, FistDetector] = {side: FistDetector(side, gcfg) for side in _SIDES}
        self._strum = StrumDetector(gcfg.strum_hand, gcfg)  # type: ignore[arg-type]
        self._beats: dict[Side, BeatDetector] = {side: BeatDetector(side, gcfg) for side in _SIDES}
        self.vibe_meter = VibeMeter(gcfg)

    @property
    def vibe(self) -> float:
        return self.vibe_meter.value

    def closed(self, side: Side) -> bool:
        return self._fists[side].closed

    def update(self, frame: Frame) -> list[GestureEvent]:
        events: list[GestureEvent] = []
        gcfg = self.cfg.gesture

        for side in _SIDES:
            track = self.tracks[side]
            hand = frame.hand(side)
            if hand is not None:
                track.update(HandFeatures.from_hand(hand), frame.t)
            else:
                track.mark_missing(frame.t)

        for side in _SIDES:
            edge = self._fists[side].update(self.tracks[side], frame.t)
            if edge is not None:
                events.append(edge)

        strum = self._strum.update(self.tracks[gcfg.strum_hand], frame.t)  # type: ignore[index]
        if strum is not None:
            events.append(strum)

        for side in _SIDES:
            beat = self._beats[side].update(self.tracks[side], frame.t, self._fists[side].closed)
            if beat is not None:
                events.append(beat)

        self.vibe_meter.update(list(self.tracks.values()), frame.t)
        return events
