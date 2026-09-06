"""Piano: accompaniment to a guitar session. The hand is the pianist.

The chord comes from the linked guitar (`--follow`) and gestures never change
it. Standalone, the progression given on the command line is held and only
the n/p keys step through it, for testing.

Per hand, every frame (`observe`), the `Pianist` detector turns the track into
grip hits, grip releases, sweep steps and passages; this class turns those
into notes. A gripped chord is held until the grip opens.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from zerohero.config import CH_PIANO, Config
from zerohero.engine.session import Session
from zerohero.events import Frame, GestureEvent, GripHit, GripRelease, Passage, Side, SweepStep
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
        self.held: dict[Side, list[int]] = {"left": [], "right": []}
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

    def pianist_event(self, ev: GripHit | GripRelease | SweepStep | Passage) -> None:
        cfg = self.cfg
        chord = self.session.current
        if isinstance(ev, GripHit):
            self._release(ev.hand)
            events = piano.grip_chord(chord, ev.x, ev.intensity, self.vibe, ev.downward, cfg.music, cfg.piano)
            self.held[ev.hand] = [e.note for e in events]
            self.scheduler.play(events)
            self._flash(f"{ev.hand[0].upper()} chord {chord.symbol} @{ev.x:.2f} {ev.intensity:.2f}")
        elif isinstance(ev, GripRelease):
            self._release(ev.hand)
        elif isinstance(ev, SweepStep):
            note = piano.sweep_note(self.lattice(), ev.slot, ev.speed, self.vibe, cfg.music, cfg.piano)
            self.scheduler.play([note])
            self._flash(f"{ev.hand[0].upper()} sweep {ev.direction} {note.note}")
        elif isinstance(ev, Passage):
            events = piano.passage(chord, ev.x, ev.direction, ev.intensity, self.vibe, ev.erratic, cfg.music, cfg.piano)
            self.scheduler.play(events)
            self._flash(f"{ev.hand[0].upper()} run {ev.direction}{' zigzag' if ev.erratic else ''} {ev.intensity:.2f}")

    def _release(self, hand: Side) -> None:
        if self.held[hand]:
            self.scheduler.release(CH_PIANO, self.held[hand])
            self.held[hand] = []

    # ---- keys and link ---------------------------------------------------

    def key(self, k: str) -> None:
        if k == "n":
            self.session.next()
            self._flash(f"-> {self.session.current.symbol}")
        elif k == "p":
            self.session.prev()
            self._flash(f"-> {self.session.current.symbol}")
        elif k == " ":
            self.pianist_event(GripHit(t=time.monotonic(), hand="right", x=0.6, intensity=0.6, downward=False))
        elif k == "r":
            self.session.reset()
            for side in ("left", "right"):
                self._release(side)

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
