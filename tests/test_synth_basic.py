from __future__ import annotations

import os
import time

import numpy as np
import pytest

from zerohero.config import CH_GUITAR, CH_PIANO
from zerohero.synth.basic import MAX_VOICES, NumpySynth


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def test_plucked_guitar_note_produces_decaying_output() -> None:
    synth = NumpySynth(samplerate=48000, open_stream=False)
    synth.note_on(CH_GUITAR, 60, 110)
    early = synth.render(2048)
    # Skip well ahead to compare energy after most of the ~2s decay.
    for _ in range(80):
        synth.render(1024)
    late = synth.render(2048)
    assert _rms(early) > 0.0
    assert _rms(early) > _rms(late) * 3  # clearly decayed


def test_piano_note_produces_decaying_output() -> None:
    synth = NumpySynth(samplerate=48000, open_stream=False)
    synth.note_on(CH_PIANO, 60, 100)
    early = synth.render(2048)
    for _ in range(60):
        synth.render(1024)
    late = synth.render(2048)
    assert _rms(early) > 0.0
    assert _rms(early) > _rms(late) * 3


def test_note_off_fades_out_quickly() -> None:
    synth = NumpySynth(samplerate=48000, open_stream=False)
    synth.note_on(CH_GUITAR, 60, 110)
    synth.render(256)
    synth.note_off(CH_GUITAR, 60)
    # Release is ~80ms; after 200ms the voice should be gone.
    for _ in range(int(0.2 * 48000 / 256) + 2):
        synth.render(256)
    assert synth._voices == []


def test_polyphony_cap_steals_oldest_voice() -> None:
    synth = NumpySynth(samplerate=48000, open_stream=False)
    for i in range(MAX_VOICES + 8):
        synth.note_on(CH_GUITAR, 40 + (i % 30), 100)
    assert len(synth._voices) == MAX_VOICES
    # The oldest voices (lowest seq) should have been evicted.
    seqs = [v.seq for v in synth._voices]
    assert min(seqs) > 8


def test_render_is_fast_enough_for_realtime(benchmark_blocks: int = 100) -> None:
    synth = NumpySynth(samplerate=48000, open_stream=False)
    for i in range(12):
        channel = CH_GUITAR if i % 2 == 0 else CH_PIANO
        synth.note_on(channel, 48 + i, 100)
    assert len(synth._voices) == 12

    start = time.perf_counter()
    for _ in range(benchmark_blocks):
        synth.render(256)
    elapsed = time.perf_counter() - start
    per_block_ms = (elapsed / benchmark_blocks) * 1000
    assert per_block_ms < 2.0, f"render(256) with 12 voices took {per_block_ms:.3f} ms/block"


@pytest.mark.skipif(
    os.environ.get("ZEROHERO_AUDIO_TEST") != "1",
    reason="set ZEROHERO_AUDIO_TEST=1 to exercise a real audio device",
)
def test_real_device_playback() -> None:
    synth = NumpySynth(samplerate=48000, open_stream=True)
    try:
        synth.note_on(CH_GUITAR, 60, 100)
        time.sleep(0.3)
        synth.note_off(CH_GUITAR, 60)
        time.sleep(0.2)
    finally:
        synth.close()
