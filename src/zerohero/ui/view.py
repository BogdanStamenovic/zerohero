"""Pick a hand-tracking view: OpenCV window, braille terminal, or nothing.

Both views expose the same four calls so the engine does not care which one
it got: `start()`, `draw(frame, state)`, `poll_key()`, `close()`.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Protocol

from zerohero.config import Config
from zerohero.events import Frame
from zerohero.ui.overlay import OverlayState

log = logging.getLogger(__name__)


class View(Protocol):
    def start(self) -> None: ...
    def draw(self, frame: Frame, state: OverlayState) -> None: ...
    def poll_key(self) -> str | None: ...
    def close(self) -> None: ...


class WindowView:
    def __init__(self, cfg: Config, title: str) -> None:
        from zerohero.ui.overlay import Overlay, Window

        self.overlay = Overlay(cfg)
        self.window = Window(title)

    def start(self) -> None:
        pass

    def draw(self, frame: Frame, state: OverlayState) -> None:
        self.window.show(self.overlay.draw(frame.image, frame, state))

    def poll_key(self) -> str | None:
        return self.window.poll_key()

    def close(self) -> None:
        self.window.close()


def display_available() -> bool:
    """A GUI window can only open where there is a display server, or on Windows/macOS always."""
    if sys.platform in ("win32", "darwin"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def open_view(cfg: Config, title: str) -> View | None:
    choice = cfg.ui
    if choice == "none":
        return None
    if choice == "auto":
        choice = "window" if display_available() else ("terminal" if sys.stdout.isatty() else "none")
        if choice == "none":
            log.info("no display and no terminal; running without a view")
            return None
    if choice == "window":
        try:
            view: View = WindowView(cfg, title)
            view.start()
            return view
        except Exception as e:  # headless X, missing GUI backend in the OpenCV build
            log.warning("window unavailable (%s); falling back to the terminal view", e)
            choice = "terminal"
    if choice == "terminal":
        from zerohero.ui.terminal import TerminalView

        view = TerminalView(cfg)
        view.start()
        return view
    raise ValueError(f"unknown ui {cfg.ui!r}")
