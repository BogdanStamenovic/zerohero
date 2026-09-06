"""Piano voicings with voice leading, and Beat -> phrase NoteEvents. See ARCHITECTURE.md > music."""

from __future__ import annotations

import random
from typing import Literal

from zerohero.config import CH_PIANO, MusicConfig
from zerohero.events import Beat, NoteEvent
from zerohero.music.theory import Chord, chord_scale, midi

Register = Literal["low", "high"]

_REGISTER_RANGE: dict[Register, tuple[int, int]] = {
    "low": (36, 55),  # roughly octaves 2-3
    "high": (55, 79),  # roughly octaves 4-5
}


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clip(t, 0.0, 1.0)


class PianoVoicer:
    """Picks inversions that minimise movement from the previous voicing, per register."""

    def __init__(self) -> None:
        self._memory: dict[Register, list[int]] = {}

    def voicing(self, chord: Chord, register: Register, density: float) -> list[int]:
        lo, hi = _REGISTER_RANGE[register]
        tones = {(chord.root + iv) % 12 for iv in chord.intervals}
        if chord.bass is not None:
            tones.add(chord.bass)

        if density > 0.35:
            seventh = 11 if chord.quality in ("maj", "maj7", "maj9", "6") else 10
            tones.add((chord.root + seventh) % 12)
        if density > 0.65:
            tones.add((chord.root + 2) % 12)  # 9th

        prev = self._memory.get(register)
        # Same target for every tone: cheap approximation of minimal total
        # movement that works well because chords sharing tones (or close on
        # the circle of fifths) end up choosing nearby octaves for all of them.
        target = (sum(prev) / len(prev)) if prev else (lo + hi) / 2

        notes: list[int] = []
        for pc in sorted(tones):
            candidates = [n for n in range(lo, hi + 1) if n % 12 == pc]
            if candidates:
                notes.append(min(candidates, key=lambda n: abs(n - target)))

        if density > 0.85:
            root_notes = [n for n in notes if n % 12 == chord.root]
            if root_notes:
                r = root_notes[0]
                if r - 12 >= lo and r - 12 not in notes:
                    notes.append(r - 12)
                elif r + 12 <= hi and r + 12 not in notes:
                    notes.append(r + 12)

        notes = sorted(set(notes))
        self._memory[register] = notes
        return notes


def _velocity(cfg: MusicConfig, intensity: float, vibe: float) -> float:
    mix = 0.6 * _clip(intensity, 0.0, 1.0) + 0.4 * _clip(vibe, 0.0, 1.0)
    return _clip(cfg.velocity_min + (cfg.velocity_max - cfg.velocity_min) * mix, cfg.velocity_min, cfg.velocity_max)


def phrase(
    chord: Chord,
    beat: Beat,
    vibe: float,
    tempo: float | None,
    voicer: PianoVoicer,
    cfg: MusicConfig,
) -> list[NoteEvent]:
    register: Register = "low" if beat.hand == "left" else "high"
    duration = cfg.piano_staccato if beat.closed else cfg.piano_sustain
    notes = voicer.voicing(chord, register, vibe)
    velocity = round(_velocity(cfg, beat.intensity, vibe))

    events: list[NoteEvent] = []

    if beat.direction == "left":
        for n in notes:
            jitter = random.uniform(0, 0.008)  # tiny per-note humanisation
            events.append(NoteEvent(offset=jitter, note=n, velocity=velocity, duration=duration, channel=CH_PIANO))

    elif beat.direction == "right":
        spacing = (60 / tempo / 4) if tempo else _lerp(0.14, 0.05, beat.intensity)
        for i, n in enumerate(notes):
            events.append(NoteEvent(offset=i * spacing, note=n, velocity=velocity, duration=duration, channel=CH_PIANO))

    elif beat.direction == "down":
        heavy_velocity = round(_clip(velocity + 15, cfg.velocity_min, cfg.velocity_max))
        bass = [midi(chord.root, 2), midi(chord.root, 3)]
        for n in sorted(set(bass) | set(notes)):
            events.append(
                NoteEvent(offset=0.0, note=n, velocity=heavy_velocity, duration=duration, channel=CH_PIANO)
            )

    else:  # "up": a run up the chord scale, glissando feel
        scale = set(chord_scale(chord))
        start = min(notes) if notes else midi(chord.root, 3)
        span = 18  # ~1.5 octaves
        run = [n for n in range(start, start + span + 1) if n % 12 in scale]
        spacing = _lerp(0.06, 0.03, beat.intensity)
        for i, n in enumerate(run):
            rising = velocity + (cfg.velocity_max - velocity) * (i / max(1, len(run) - 1))
            events.append(
                NoteEvent(
                    offset=i * spacing,
                    note=n,
                    velocity=round(_clip(rising, cfg.velocity_min, cfg.velocity_max)),
                    duration=duration,
                    channel=CH_PIANO,
                )
            )

    return events


def accompaniment(chord: Chord, tempo: float, vibe: float, voicer: PianoVoicer, cfg: MusicConfig) -> list[NoteEvent]:
    """One bar (4 beats) of comping: bass on 1 & 3, chord on 2 & 4, denser when vibe is high."""
    beat_dur = 60.0 / tempo
    velocity_range = cfg.velocity_max - cfg.velocity_min
    velocity = round(_clip(cfg.velocity_min + velocity_range * vibe, cfg.velocity_min, cfg.velocity_max))
    ring = beat_dur * 0.9

    bass_note = midi(chord.root, 2)
    chord_notes = voicer.voicing(chord, "high", vibe)

    events = [
        NoteEvent(offset=0.0, note=bass_note, velocity=velocity, duration=ring, channel=CH_PIANO),
        NoteEvent(offset=2 * beat_dur, note=bass_note, velocity=velocity, duration=ring, channel=CH_PIANO),
    ]
    for n in chord_notes:
        events.append(NoteEvent(offset=beat_dur, note=n, velocity=velocity, duration=ring, channel=CH_PIANO))
        events.append(NoteEvent(offset=3 * beat_dur, note=n, velocity=velocity, duration=ring, channel=CH_PIANO))

    if vibe > 0.6:
        # Denser comping: an extra hit on the "and" of beat 4.
        off_velocity = round(velocity * 0.8)
        for n in chord_notes:
            events.append(
                NoteEvent(offset=3.5 * beat_dur, note=n, velocity=off_velocity, duration=ring / 2, channel=CH_PIANO)
            )

    return events
