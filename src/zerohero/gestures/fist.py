"""Fist open/close edge detector, used for chord-advance in guitar mode and
for the piano's per-hand `closed` flag."""

from __future__ import annotations

from zerohero.config import GestureConfig
from zerohero.events import FistClose, FistOpen, Side
from zerohero.gestures.track import HandTrack

GestureEdge = FistClose | FistOpen


_NOT_CLOSED_IS_OPEN = 0.3  # seconds in the dead zone that count as an open hand
_GAP_FORGETS_STATE = 1.0  # seconds without the hand before its prior state is unknown


class FistDetector:
    """Hysteresis state machine on `closed_score`, debounced and edge-triggered.

    `closed_score` crossing `fist_close_above` or dropping below
    `fist_open_below` becomes a *candidate* state; the candidate must hold for
    `fist_min_hold` before it is confirmed and an edge event fires (rejects
    one-frame flicker). `FistClose` additionally has a refractory period.

    A track that just reappeared after being lost has no known prior state:
    firing `FistClose` immediately if it reappears already closed would be a
    guess, not an observation, so the detector requires one confirmed *open*
    state (real or by construction) before it will ever emit a close.
    """

    def __init__(self, side: Side, cfg: GestureConfig) -> None:
        self.side = side
        self.cfg = cfg
        self._confirmed: bool | None = None  # None = unknown, True = closed, False = open
        self._candidate: bool | None = None
        self._candidate_since: float | None = None
        self._last_close_fire_t: float | None = None
        self._seen_open = False
        self._was_present = False
        self._not_closed_since: float | None = None
        self._missing_since: float | None = None

    @property
    def closed(self) -> bool:
        return bool(self._confirmed)

    def update(self, track: HandTrack, t: float) -> GestureEdge | None:
        cfg = self.cfg

        if not track.present:
            # The tracker often loses a hand for a few frames right as it
            # closes into a fist. Keep the open baseline across a short gap;
            # only a real absence makes the prior state unknown.
            if self._was_present:
                self._missing_since = t
            self._was_present = False
            if self._missing_since is not None and (t - self._missing_since) > _GAP_FORGETS_STATE:
                self._reset_unknown()
            return None
        self._was_present = True
        self._missing_since = None

        score = track.closed_score
        if score >= cfg.fist_close_above:
            raw: bool | None = True
            self._not_closed_since = None
        elif score <= cfg.fist_open_below:
            raw = False
        else:
            # Dead zone. The recognizer reports "None" (0.5) for most frames of
            # an ordinary open hand, so a hand that simply is not a fist for a
            # while must count as open, or the close edge never gets a baseline.
            self._not_closed_since = t if self._not_closed_since is None else self._not_closed_since
            raw = False if (t - self._not_closed_since) >= _NOT_CLOSED_IS_OPEN else None

        if raw is not None and raw != self._candidate:
            self._candidate = raw
            self._candidate_since = t

        if (
            self._candidate is None
            or self._candidate_since is None
            or (t - self._candidate_since) < cfg.fist_min_hold
            or self._candidate == self._confirmed
        ):
            return None

        if self._candidate:  # confirming closed
            if not self._seen_open:
                self._confirmed = True  # state known, but no open baseline yet: no event
                return None
            if self._last_close_fire_t is not None and (t - self._last_close_fire_t) < cfg.fist_refractory:
                self._confirmed = True  # state updates even though the event is suppressed
                return None
            self._confirmed = True
            self._last_close_fire_t = t
            return FistClose(t=t, hand=self.side)

        self._confirmed = False
        self._seen_open = True
        return FistOpen(t=t, hand=self.side)

    def _reset_unknown(self) -> None:
        self._confirmed = None
        self._candidate = None
        self._candidate_since = None
        self._seen_open = False
        self._not_closed_since = None
