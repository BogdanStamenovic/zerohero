"""Piano: conducting hits become phrases over the current chord.

Chord advance policy (`cfg.advance`):
  fist    left fist steps to the next chord, same gesture as guitar mode
  auto:N  every N beats the chord advances by itself
  follow  the chord comes from a linked guitar device; when the guitar is
          strumming and the user here is idle, a light accompaniment keeps
          time using the lead's tempo
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from zerohero.config import CH_PIANO, Config
from zerohero.engine.session import Session
from zerohero.events import Beat, FistClose, FistOpen, GestureEvent, Strum
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
        self.voicer = piano.PianoVoicer()
        self.vibe = 0.0
        self.on_event: Callable[[str], None] | None = None
        self.beats = 0
        self.auto_every = 0
        if cfg.advance.startswith("auto:"):
            self.auto_every = max(1, int(cfg.advance.split(":", 1)[1]))
        self.last_beat_t = 0.0
        self._next_bar_t: float | None = None
        if follower is not None:
            follower.on_chord = self._on_remote_chord
            follower.on_strum = self._on_remote_strum
            follower.on_connect = lambda: self._flash("linked")
            follower.on_disconnect = lambda: self._flash("link lost")

    # ---- gestures --------------------------------------------------------

    def handle(self, ev: GestureEvent) -> None:
        if isinstance(ev, Beat):
            self.beat(ev)
        elif isinstance(ev, FistClose) and ev.hand == self.cfg.gesture.fist_hand and self.cfg.advance == "fist":
            self.next_chord()
        elif isinstance(ev, (FistOpen, Strum)):
            pass

    def beat(self, ev: Beat) -> None:
        tempo = self.follower.tempo if self.follower is not None else None
        events = piano.phrase(self.session.current, ev, self.vibe, tempo, self.voicer, self.cfg.music)
        # Only the same hand's register is cut, so a left-hand bass can ring under right-hand hits.
        self.scheduler.play(events)
        self.beats += 1
        self.last_beat_t = ev.t
        self._flash(f"{ev.hand[0].upper()} {ev.direction} {ev.intensity:.2f}{' fist' if ev.closed else ''}")
        if self.auto_every and self.beats % self.auto_every == 0:
            self.next_chord()

    def next_chord(self) -> None:
        self.session.next()
        self._flash(f"-> {self.session.current.symbol}")

    def key(self, k: str) -> None:
        if k == "n":
            self.next_chord()
        elif k == "p":
            self.session.prev()
        elif k == " ":
            self.beat(Beat(t=time.monotonic(), hand="right", direction="left", intensity=0.6, closed=False))
        elif k == "r":
            self.session.reset()
            self.voicer = piano.PianoVoicer()

    # ---- link ------------------------------------------------------------

    def _on_remote_chord(self, index: int, symbol: str, t_local: float) -> None:
        self.session.set_symbol(index, symbol)
        self._flash(f"lead -> {symbol}")

    def _on_remote_strum(self, direction: str, intensity: float, t_local: float) -> None:
        pass  # tempo is derived inside the follower; nothing to play per strum

    def tick(self, t: float) -> None:
        """Accompaniment when following: play a bar on the lead's tempo while the user is idle."""
        if self.follower is None or self.cfg.advance != "follow":
            return
        tempo = self.follower.tempo
        if tempo is None or t - self.last_beat_t < 2.0:
            self._next_bar_t = None
            return
        if self._next_bar_t is None:
            self._next_bar_t = t
        if t >= self._next_bar_t:
            bar = piano.accompaniment(self.session.current, tempo, self.vibe, self.voicer, self.cfg.music)
            self.scheduler.cut(CH_PIANO)
            self.scheduler.play(bar, t0=self._next_bar_t)
            self._next_bar_t += 4 * 60.0 / tempo

    def status(self) -> str:
        if self.follower is None:
            return ""
        if not self.follower.connected:
            return "link: connecting"
        tempo = self.follower.tempo
        rtt = self.follower.rtt_ms
        return f"link: {tempo:.0f} bpm" if tempo else "link: ok" + (f" {rtt:.0f}ms" if rtt else "")

    def _flash(self, text: str) -> None:
        if self.on_event:
            self.on_event(text)
