"""Wire the pipeline together and run the main loop."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from zerohero.config import CH_GUITAR, CH_PIANO, Config
from zerohero.engine.guitar_mode import GuitarMode
from zerohero.engine.piano_mode import PianoMode
from zerohero.engine.session import Session
from zerohero.gestures.pipeline import GesturePipeline
from zerohero.synth import Scheduler, open_synth

if TYPE_CHECKING:
    from zerohero.ui.view import View
    from zerohero.vision.replay import FrameRecorder
    from zerohero.vision.source import FrameSource

log = logging.getLogger(__name__)


class App:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.session = Session(cfg.progression)
        self.synth = open_synth(cfg)
        self.scheduler = Scheduler(self.synth)
        self.pipeline = GesturePipeline(cfg)
        self.lead = None
        self.follower = None
        self.recorder: FrameRecorder | None = None
        self.view: View | None = None
        self.last_event = ""
        self.last_event_t = 0.0
        self.source: FrameSource | None = None
        self.mode: GuitarMode | PianoMode

        if cfg.lead:
            from zerohero.link import Lead

            self.lead = Lead(cfg.link, cfg.lead_transport)
            self.lead.start()
        if cfg.follow:
            from zerohero.link import Follower

            self.follower = Follower(cfg.link, cfg.follow, cfg.link.name)

        if cfg.mode == "guitar":
            self.mode = GuitarMode(cfg, self.session, self.scheduler, lead=self.lead)
        else:
            self.mode = PianoMode(cfg, self.session, self.scheduler, follower=self.follower)
        self.mode.on_event = self.flash

        if self.follower is not None:
            self.follower.start()

    def flash(self, text: str) -> None:
        self.last_event = text
        self.last_event_t = time.monotonic()
        log.info("%s", text)

    def run(self) -> None:
        from zerohero.vision.source import open_source

        cfg = self.cfg
        self.source = open_source(cfg)
        if cfg.record:
            from zerohero.vision.replay import FrameRecorder

            self.recorder = FrameRecorder(cfg.record)
        from zerohero.ui.view import open_view

        self.view = open_view(cfg, f"zerohero {cfg.mode}")
        log.info("%s mode, progression %s", cfg.mode, " ".join(self.session.symbols))
        try:
            self._loop()
        finally:
            self.close()

    def _loop(self) -> None:
        assert self.source is not None
        loop_t = time.monotonic()
        fps = 0.0
        for frame in self.source.frames():
            now = time.monotonic()
            dt = now - loop_t
            loop_t = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps

            if self.recorder is not None:
                self.recorder.write(frame)

            events = self.pipeline.update(frame)
            if isinstance(self.mode, PianoMode):
                self.mode.vibe = self.pipeline.vibe
            for ev in events:
                self.mode.handle(ev)
            self.mode.tick(frame.t)

            if self.view is not None:
                key = self._draw(frame, fps)
                if key in ("q", "esc"):
                    break
                if key:
                    self.mode.key(key)

    def _draw(self, frame, fps: float) -> str | None:
        from zerohero.ui.overlay import OverlayState

        assert self.view is not None
        tracks: dict[str, tuple[float, float, float, float, bool]] = {}
        for side, tr in self.pipeline.tracks.items():
            if tr.present:
                tracks[side] = (tr.palm[0], tr.palm[1], tr.velocity[0], tr.velocity[1], self.pipeline.closed(side))
        state = OverlayState(
            mode=self.cfg.mode,
            chords=self.session.symbols,
            index=self.session.index,
            vibe=self.pipeline.vibe,
            tracks=tracks,
            last_event=self.last_event,
            last_event_t=self.last_event_t,
            fps=fps,
            link=self.mode.status(),
        )
        self.view.draw(frame, state)
        return self.view.poll_key()

    def close(self) -> None:
        for ch in (CH_GUITAR, CH_PIANO):
            self.scheduler.cut(ch)
        self.scheduler.stop()
        if self.follower is not None:
            self.follower.stop()
        if self.lead is not None:
            self.lead.stop()
        if self.recorder is not None:
            self.recorder.close()
        if self.view is not None:
            self.view.close()
        if self.source is not None:
            self.source.close()
        self.synth.close()


def run(cfg: Config) -> None:
    App(cfg).run()
