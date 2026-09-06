"""Guitar: right-hand strum plays the current chord, left fist advances it."""

from __future__ import annotations

import logging
from collections.abc import Callable

from zerohero.config import CH_GUITAR, Config
from zerohero.engine.session import Session
from zerohero.events import Beat, FistClose, FistOpen, GestureEvent, Strum, StrumDirection
from zerohero.music import guitar
from zerohero.synth import Scheduler

log = logging.getLogger(__name__)


class GuitarMode:
    name = "guitar"

    def __init__(self, cfg: Config, session: Session, scheduler: Scheduler, lead=None) -> None:
        self.cfg = cfg
        self.session = session
        self.scheduler = scheduler
        self.lead = lead  # link.Lead or None
        self.on_event: Callable[[str], None] | None = None  # overlay flash text
        self._announce_chord()

    def _announce_chord(self) -> None:
        if self.lead is not None:
            self.lead.send_chord(self.session.index, self.session.current.symbol)

    def handle(self, ev: GestureEvent) -> None:
        if isinstance(ev, Strum) and ev.hand == self.cfg.gesture.strum_hand:
            self.strum(ev.direction, ev.intensity)
        elif isinstance(ev, FistClose) and ev.hand == self.cfg.gesture.fist_hand:
            self.next_chord()
        elif isinstance(ev, (FistOpen, Beat)):
            pass

    def strum(self, direction: StrumDirection, intensity: float) -> None:
        chord = self.session.current
        # A new strum cuts the ringing strings, like a real pick hand damping them.
        self.scheduler.cut(CH_GUITAR)
        self.scheduler.play(guitar.strum(chord, direction, intensity, self.cfg.music))
        if self.lead is not None:
            self.lead.send_strum(direction, intensity)
        if self.on_event:
            self.on_event(f"{'v' if direction == 'down' else '^'} {chord.symbol} {intensity:.2f}")

    def next_chord(self) -> None:
        self.session.next()
        self._announce_chord()
        if self.on_event:
            self.on_event(f"-> {self.session.current.symbol}")

    def prev_chord(self) -> None:
        self.session.prev()
        self._announce_chord()

    def goto(self, index: int) -> None:
        self.session.goto(index)
        self._announce_chord()

    def key(self, k: str) -> None:
        if k == "n":
            self.next_chord()
        elif k == "p":
            self.prev_chord()
        elif k == " ":
            self.strum("down", 0.6)
        elif k == "r":
            self.goto(0)

    def tick(self, t: float) -> None:
        """Called once per frame; nothing periodic in guitar mode."""

    def status(self) -> str:
        if self.lead is None:
            return ""
        return f"lead: {self.lead.followers} follower(s)"
