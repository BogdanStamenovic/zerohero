"""Piano: accompaniment to a guitar session. The hand is the pianist.

The chord comes from the linked guitar (`--follow`) and gestures never change
it. Standalone, the progression given on the command line is held and only
the n/p keys step through it, for testing.

Per hand, every frame (`observe`), the `Pianist` detector turns the track into
chord hits, sweep steps and wiggle notes; this class turns those into notes.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from zerohero.config import Config
from zerohero.engine.session import Session
from zerohero.events import ChordHit, Frame, GestureEvent, Side, SweepStep, WiggleNote
from zerohero.gestures.pianist import Pianist
from zerohero.gestures.pipeline import GesturePipeline
from zerohero.music import piano
from zerohero.synth import Scheduler

log = logging.getLogger(__name__)


class PianoMode:
    name = "piano"

    def __init__(self, cfg: Config, session: Session, scheduler: Scheduler, follower=None) -> None:
        self.cfg = cfg
        self.session = session
        self.scheduler = scheduler
        self.follower = follower  # link.Follower or None
        self.vibe = 0.0
        self.on_event: Callable[[str], None] | None = None
        self.pianists: dict[Side, Pianist] = {s: Pianist(s, cfg.piano) for s in ("left", "right")}
        self.zigzags: dict[Side, piano.Zigzag] = {"left": piano.Zigzag(1), "right": piano.Zigzag(2)}
        self._lattice: list[int] = []
        self._lattice_key: tuple[str, int] | None = None
        if follower is not None:
            follower.on_chord = self._on_remote_chord
            follower.on_connect = lambda: self._flash("linked")
            follower.on_disconnect = lambda: self._flash("link lost")

    # ---- per frame -------------------------------------------------------

    def observe(self, pipeline: GesturePipeline, frame: Frame) -> None:
        self.vibe = pipeline.vibe
        notes = self.lattice()
        for side, pianist in self.pianists.items():
            track = pipeline.tracks[side]
            slot = piano.slot_at(track.palm[0], notes, self.cfg.piano) if track.present else None
            for ev in pianist.update(track, frame.t, slot):
                self.pianist_event(ev)

    def lattice(self) -> list[int]:
        # Density buckets keep the lattice stable while vibe wobbles.
        key = (self.session.current.symbol, int(self.vibe * 3))
        if key != self._lattice_key:
            self._lattice = piano.lattice(self.session.current, self.vibe, self.cfg.piano)
            self._lattice_key = key
        return self._lattice

    def handle(self, ev: GestureEvent) -> None:
        """Generic gesture events (strum, fist, beat) mean nothing on the piano."""

    def pianist_event(self, ev: ChordHit | SweepStep | WiggleNote) -> None:
        cfg = self.cfg
        chord = self.session.current
        if isinstance(ev, ChordHit):
            events = piano.chord_hit(chord, ev.x, ev.intensity, self.vibe, cfg.music, cfg.piano)
            self.scheduler.play(events)
            self._flash(f"{ev.hand[0].upper()} chord {chord.symbol} @{ev.x:.2f} {ev.intensity:.2f}")
        elif isinstance(ev, SweepStep):
            note = piano.sweep_note(self.lattice(), ev.slot, ev.speed, self.vibe, cfg.music, cfg.piano)
            self.scheduler.play([note])
            self._flash(f"{ev.hand[0].upper()} sweep {ev.direction} {note.note}")
        elif isinstance(ev, WiggleNote):
            note = self.zigzags[ev.hand].next(chord, ev.x, ev.drift, ev.activity, self.vibe, cfg.music, cfg.piano)
            self.scheduler.play([note])
            self._flash(f"{ev.hand[0].upper()} zigzag {note.note} {ev.drift}")

    # ---- keys and link ---------------------------------------------------

    def key(self, k: str) -> None:
        if k == "n":
            self.session.next()
            self._flash(f"-> {self.session.current.symbol}")
        elif k == "p":
            self.session.prev()
            self._flash(f"-> {self.session.current.symbol}")
        elif k == " ":
            self.pianist_event(ChordHit(t=time.monotonic(), hand="right", x=0.6, intensity=0.6))
        elif k == "r":
            self.session.reset()

    def _on_remote_chord(self, index: int, symbol: str, t_local: float) -> None:
        self.session.set_symbol(index, symbol)
        self._flash(f"guitar -> {symbol}")

    def tick(self, t: float) -> None:
        """Nothing periodic: the hand decides when to play."""

    def status(self) -> str:
        if self.follower is None:
            return ""
        if not self.follower.connected:
            return "link: connecting"
        tempo = self.follower.tempo
        return f"link: guitar {tempo:.0f} bpm" if tempo else "link: guitar"

    def _flash(self, text: str) -> None:
        if self.on_event:
            self.on_event(text)
