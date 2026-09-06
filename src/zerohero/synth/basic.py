"""numpy + sounddevice fallback synth. See ARCHITECTURE.md `synth` section.

Two voice types, chosen by MIDI channel: CH_GUITAR gets a Karplus-Strong
plucked string, CH_PIANO gets additive (fundamental + harmonics). Both are
*pre-rendered in full* on note_on rather than synthesised sample-by-sample
inside the audio callback. This trades a small amount of note-on latency
(rendering a ~2s/96k-float buffer takes well under a millisecond for the
additive voice, and a few hundred vectorised period-blocks for Karplus-Strong)
for a callback that only ever slices and sums numpy arrays -- no per-sample
Python loop ever runs on the audio thread, which is what keeps a 256-frame
block with a dozen voices well under the device's callback deadline.
"""

from __future__ import annotations

import math
import threading

import numpy as np

from zerohero.config import CH_GUITAR
from zerohero.synth.base import SynthUnavailable

GUITAR_SUSTAIN = 2.0  # seconds, pre-rendered buffer length for a plucked string
PIANO_SUSTAIN = 1.5  # seconds, pre-rendered buffer length for a struck note
RELEASE_S = 0.08  # note_off fade-out
MAX_VOICES = 32


def _karplus_strong(freq: float, duration_s: float, samplerate: int, velocity: int) -> np.ndarray:
    """Noise burst through a circular delay line of length `period`, with a
    one-pole lowpass in the feedback path. Because the filter only ever looks
    back exactly `period` samples, the whole delay line's *next* state is a
    single vectorised operation on its *current* state (a roll-by-one plus a
    blend) -- so the per-sample recursion becomes a per-period loop, a few
    hundred iterations even for a 2s low-E buffer, instead of tens of
    thousands of individual Python steps.
    """
    n = max(1, int(duration_s * samplerate))
    period = max(2, int(round(samplerate / freq)))
    rng = np.random.default_rng()
    ring = rng.uniform(-1.0, 1.0, size=period)

    vel = max(1, min(127, velocity)) / 127.0
    # Harder pluck -> less smoothing per pass -> brighter, twangier sustain.
    smoothing = 0.5 - 0.2 * vel
    # Per-period decay so ~duration_s seconds brings any pitch down to ~1% amplitude,
    # independent of how many periods fit in that time.
    periods_total = max(1.0, duration_s * freq)
    per_period_decay = 0.01 ** (1.0 / periods_total)

    out = np.empty(n, dtype=np.float64)
    pos = 0
    while pos < n:
        take = min(period, n - pos)
        out[pos : pos + take] = ring[:take]
        shifted = np.roll(ring, -1)
        ring = ((1.0 - smoothing) * ring + smoothing * shifted) * per_period_decay
        pos += take

    out *= vel
    return out.astype(np.float32)


def _additive_piano(freq: float, duration_s: float, samplerate: int, velocity: int) -> np.ndarray:
    """Fundamental + 3 harmonics, each with its own exponential decay (higher
    partials die faster, as on a real piano) and a slight inharmonicity
    stretch. Fully vectorised over the sample axis; the only Python loop is
    over the 4 partials.
    """
    n = max(1, int(duration_s * samplerate))
    t = np.arange(n, dtype=np.float64) / samplerate
    vel = max(1, min(127, velocity)) / 127.0
    brightness = 0.3 + 0.7 * vel  # louder hits keep more energy in the upper partials

    # -40dB over duration_s for the fundamental; each higher partial decays faster.
    base_rate = math.log(100.0) / duration_s
    inharmonicity = 0.0004

    buf = np.zeros(n, dtype=np.float64)
    rng = np.random.default_rng()
    for i, harmonic in enumerate(range(1, 5)):
        partial_freq = freq * harmonic * math.sqrt(1.0 + inharmonicity * harmonic * harmonic)
        amp = (brightness**i) / harmonic
        decay_rate = base_rate * (1.0 + 0.4 * i)
        phase = rng.uniform(0, 2 * math.pi)
        buf += amp * np.exp(-decay_rate * t) * np.sin(2 * math.pi * partial_freq * t + phase)

    buf *= vel
    peak = np.abs(buf).max()
    if peak > 1.0:
        buf /= peak
    return buf.astype(np.float32)


def _pick_device(sd: object, channels: int) -> int | None:
    """PortAudio's own "default" ALSA device silently fails on at least one
    PipeWire+Pulse-shim setup (the dev laptop): the stream opens, the callback
    runs, PortAudio reports continuous output-underflow, and no stream ever
    appears in PipeWire's graph -- total silence with no exception raised.
    The "pipewire" ALSA plugin device, when present, connects cleanly. Prefer
    it by name; otherwise fall back to whatever PortAudio calls default.
    """
    try:
        devices = sd.query_devices()  # type: ignore[attr-defined]
    except Exception:
        return None
    for i, d in enumerate(devices):
        if d.get("name") == "pipewire" and d.get("max_output_channels", 0) >= channels:
            return i
    return None


class _Voice:
    __slots__ = ("seq", "channel", "note", "buf", "pos", "release_at")

    def __init__(self, seq: int, channel: int, note: int, buf: np.ndarray) -> None:
        self.seq = seq
        self.channel = channel
        self.note = note
        self.buf = buf
        self.pos = 0
        self.release_at: int | None = None  # sample index into buf where note_off happened


class NumpySynth:
    """numpy + sounddevice fallback. Construct with open_stream=False to render
    offline (tests, benchmarking) without touching any audio device.
    """

    name = "basic"

    def __init__(
        self,
        samplerate: int = 48000,
        blocksize: int = 256,
        channels: int = 2,
        open_stream: bool = True,
    ) -> None:
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.channels = channels
        self._voices: list[_Voice] = []
        self._lock = threading.Lock()
        self._seq = 0
        self._stream = None

        if open_stream:
            try:
                import sounddevice as sd
            except ImportError as e:
                raise SynthUnavailable(f"sounddevice unavailable: {e}") from e
            try:
                self._stream = sd.OutputStream(
                    samplerate=samplerate,
                    blocksize=blocksize,
                    channels=channels,
                    dtype="float32",
                    callback=self._callback,
                    device=_pick_device(sd, channels),
                )
                self._stream.start()
            except Exception as e:
                raise SynthUnavailable(f"could not open audio output stream: {e}") from e

    def _callback(self, outdata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        mix = self.render(frames)
        outdata[:] = mix[:, None]  # broadcasts the mono mix to every output channel

    def render(self, nframes: int) -> np.ndarray:
        """Render `nframes` of mono mix as float32 in [-1, 1]. This is what the
        audio callback calls; tests call it directly with open_stream=False.
        """
        mix = np.zeros(nframes, dtype=np.float32)
        release_len = max(1, int(RELEASE_S * self.samplerate))
        with self._lock:
            alive = []
            for v in self._voices:
                start = v.pos
                end = min(start + nframes, len(v.buf))
                take = end - start
                if take > 0:
                    seg = v.buf[start:end]
                    if v.release_at is not None:
                        idx = np.arange(start, end)
                        env = np.clip(1.0 - (idx - v.release_at) / release_len, 0.0, 1.0)
                        seg = seg * env
                    mix[:take] += seg
                v.pos = end
                done = v.pos >= len(v.buf)
                if v.release_at is not None and v.pos >= v.release_at + release_len:
                    done = True
                if not done:
                    alive.append(v)
            self._voices = alive
        return np.tanh(mix)

    def note_on(self, channel: int, note: int, velocity: int) -> None:
        freq = 440.0 * 2.0 ** ((note - 69) / 12.0)
        if channel == CH_GUITAR:
            buf = _karplus_strong(freq, GUITAR_SUSTAIN, self.samplerate, velocity)
        else:
            buf = _additive_piano(freq, PIANO_SUSTAIN, self.samplerate, velocity)
        with self._lock:
            self._seq += 1
            self._voices.append(_Voice(self._seq, channel, note, buf))
            if len(self._voices) > MAX_VOICES:
                self._voices.pop(0)  # steal the oldest (list stays insertion-ordered)

    def note_off(self, channel: int, note: int) -> None:
        with self._lock:
            for v in self._voices:
                if v.channel == channel and v.note == note and v.release_at is None:
                    v.release_at = v.pos

    def program(self, channel: int, program: int) -> None:
        pass  # voice timbre is fixed per channel (guitar/piano); nothing to select

    def all_notes_off(self, channel: int) -> None:
        with self._lock:
            for v in self._voices:
                if v.channel == channel and v.release_at is None:
                    v.release_at = v.pos

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        with self._lock:
            self._voices = []


__all__ = ["NumpySynth", "SynthUnavailable", "MAX_VOICES"]
