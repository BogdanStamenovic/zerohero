"""Synth protocol. See ARCHITECTURE.md `synth` section."""

from __future__ import annotations

from typing import Protocol


class SynthUnavailable(RuntimeError):
    """Raised by a backend's constructor when it cannot be brought up: missing
    library, missing soundfont, or no audio driver could be started."""


class Synth(Protocol):
    name: str

    def note_on(self, channel: int, note: int, velocity: int) -> None: ...

    def note_off(self, channel: int, note: int) -> None: ...

    def program(self, channel: int, program: int) -> None: ...

    def all_notes_off(self, channel: int) -> None: ...

    def close(self) -> None: ...


class NullSynth:
    """Records calls instead of making sound. Used by tests and `--no-audio`."""

    name = "null"

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def note_on(self, channel: int, note: int, velocity: int) -> None:
        self.calls.append(("note_on", channel, note, velocity))

    def note_off(self, channel: int, note: int) -> None:
        self.calls.append(("note_off", channel, note))

    def program(self, channel: int, program: int) -> None:
        self.calls.append(("program", channel, program))

    def all_notes_off(self, channel: int) -> None:
        self.calls.append(("all_notes_off", channel))

    def close(self) -> None:
        self.calls.append(("close",))
