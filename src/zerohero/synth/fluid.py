"""fluidsynth backend. See ARCHITECTURE.md `synth` section."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

from zerohero.config import SynthConfig
from zerohero.synth.base import SynthUnavailable


def _default_driver() -> str:
    system = platform.system()
    if system == "Darwin":
        return "coreaudio"
    if system == "Windows":
        return "dsound"
    return "pulseaudio"


class FluidSynth:
    name = "fluid"

    def __init__(
        self,
        cfg: SynthConfig,
        soundfont: Path,
        programs: dict[int, int] | None = None,
    ) -> None:
        try:
            import fluidsynth
        except (ImportError, OSError) as e:
            # ImportError: pyfluidsynth not installed. OSError: ctypes could not
            # dlopen libfluidsynth (the Python package is present but the system
            # library is missing).
            raise SynthUnavailable(f"fluidsynth library unavailable: {e}") from e

        if not soundfont.exists():
            raise SynthUnavailable(f"soundfont not found: {soundfont}")

        try:
            self._fs = fluidsynth.Synth(gain=cfg.gain, samplerate=cfg.samplerate)
        except (OSError, RuntimeError) as e:
            raise SynthUnavailable(f"fluidsynth failed to initialise: {e}") from e

        # Best effort: lower audio.period-size/periods for latency. Older
        # pyfluidsynth builds (or libfluidsynth builds without these settings
        # registered) may reject or ignore them, so this must never be fatal.
        for opt, val in (("audio.period-size", 256), ("audio.periods", 3)):
            try:
                self._fs.setting(opt, val)
            except Exception:
                pass

        driver = cfg.audio_driver or _default_driver()
        started = self._start(driver)
        if not started and sys.platform.startswith("linux") and driver != "alsa":
            # pulseaudio (the Linux default) can be unavailable, e.g. inside a
            # container with no PipeWire/Pulse socket. Fall back to raw ALSA
            # before giving up.
            started = self._start("alsa")
        if not started:
            self._fs.delete()
            raise SynthUnavailable(f"fluidsynth could not start an audio driver (tried {driver!r})")

        sfid = self._fs.sfload(str(soundfont))
        if sfid == -1:
            self._fs.delete()
            raise SynthUnavailable(f"fluidsynth could not load soundfont: {soundfont}")
        self._sfid = sfid

        programs = programs or {}
        for channel, program in programs.items():
            self.program(channel, program)

    def _start(self, driver: str) -> bool:
        self._fs.start(driver=driver)
        # pyfluidsynth's start() ignores driver failures: fluid_synth.start()
        # always returns FLUID_OK regardless of whether the underlying audio
        # driver actually came up. The only observable signal of failure is
        # that `audio_driver` (a ctypes pointer) stays None.
        return self._fs.audio_driver is not None

    def note_on(self, channel: int, note: int, velocity: int) -> None:
        self._fs.noteon(channel, note, velocity)

    def note_off(self, channel: int, note: int) -> None:
        self._fs.noteoff(channel, note)

    def program(self, channel: int, program: int) -> None:
        self._fs.program_select(channel, self._sfid, 0, program)

    def all_notes_off(self, channel: int) -> None:
        self._fs.all_notes_off(channel)

    def close(self) -> None:
        self._fs.delete()


__all__ = ["FluidSynth", "SynthUnavailable"]
