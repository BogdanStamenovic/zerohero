"""Synth backend selection. See ARCHITECTURE.md `synth` section.

`open_synth` tries fluidsynth with the configured soundfont, then falls back
to `basic.NumpySynth` with a warning. Programs: guitar 25 (steel), piano 0.
"""

from __future__ import annotations

import logging

from zerohero.config import CH_GUITAR, CH_PIANO, Config
from zerohero.paths import soundfont_path
from zerohero.synth.base import NullSynth, Synth, SynthUnavailable
from zerohero.synth.basic import NumpySynth
from zerohero.synth.fluid import FluidSynth
from zerohero.synth.scheduler import Scheduler

logger = logging.getLogger(__name__)


def open_synth(cfg: Config) -> Synth:
    """Build the Synth described by `cfg.synth.backend`:

    - "none": NullSynth (also used for --no-audio)
    - "fluid": FluidSynth, raising SynthUnavailable if it cannot start
    - "basic": NumpySynth, raising SynthUnavailable if it cannot start
    - "auto" (default): FluidSynth, falling back to NumpySynth with a
      logged warning if fluidsynth is unavailable for any reason
    """
    backend = cfg.synth.backend
    programs = {CH_GUITAR: cfg.music.guitar_program, CH_PIANO: cfg.music.piano_program}

    if backend == "none":
        return NullSynth()
    if backend == "fluid":
        return _open_fluid(cfg, programs)
    if backend == "basic":
        return _open_basic(cfg, programs)
    if backend != "auto":
        raise ValueError(f"unknown synth backend: {backend!r}")

    try:
        return _open_fluid(cfg, programs)
    except SynthUnavailable as e:
        logger.warning("fluidsynth unavailable (%s), falling back to the numpy synth", e)
        return _open_basic(cfg, programs)


def _open_fluid(cfg: Config, programs: dict[int, int]) -> Synth:
    soundfont = cfg.synth.soundfont or soundfont_path()
    return FluidSynth(cfg.synth, soundfont, programs)


def _open_basic(cfg: Config, programs: dict[int, int]) -> Synth:
    synth = NumpySynth(samplerate=cfg.synth.samplerate)
    for channel, program in programs.items():
        synth.program(channel, program)
    return synth


__all__ = ["Synth", "NullSynth", "Scheduler", "SynthUnavailable", "open_synth"]
