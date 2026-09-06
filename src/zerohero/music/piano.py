"""Piano: the hand is the pianist. See ARCHITECTURE.md "piano".

Three ways to play, all over the chord the guitar is on:

- chord hit  -> open hand hit down: block chord voiced at the hand's position
- sweep      -> below the limiter, the chord tones tiled across the keyboard;
                the hand crossing a slot plays that tone, so speed is tempo
- zigzag     -> fingers wiggling: random scale steps around the hand, drifting
                the way the hand moves
"""

from __future__ import annotations

import random

from zerohero.config import CH_PIANO, MusicConfig, PianoConfig
from zerohero.events import NoteEvent
from zerohero.music.theory import Chord, chord_scale


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clip(t, 0.0, 1.0)


def velocity(cfg: MusicConfig, intensity: float, vibe: float) -> int:
    mix = 0.6 * _clip(intensity, 0.0, 1.0) + 0.4 * _clip(vibe, 0.0, 1.0)
    return int(round(_clip(cfg.velocity_min + (cfg.velocity_max - cfg.velocity_min) * mix, 1, 127)))


def key_at(x: float, pcfg: PianoConfig) -> int:
    """Frame x in 0..1 -> MIDI note on the configured keyboard span (left is low)."""
    return int(round(_lerp(pcfg.key_low, pcfg.key_high, x)))


def chord_tones(chord: Chord, density: float) -> list[int]:
    """Pitch classes of the chord, thickened with 7th and 9th as density rises."""
    tones = [(chord.root + iv) % 12 for iv in chord.intervals]
    if density > 0.35:
        seventh = 11 if chord.quality in ("maj", "maj7", "maj9", "6") else 10
        pc = (chord.root + seventh) % 12
        if pc not in tones:
            tones.append(pc)
    if density > 0.65:
        pc = (chord.root + 2) % 12
        if pc not in tones:
            tones.append(pc)
    return tones


def voicing_at(chord: Chord, center: int, density: float) -> list[int]:
    """Closed voicing of the chord whose lowest note sits just below `center`.

    Root at the bottom, other tones stacked upward within an octave, so the
    shape a real hand would grab at that spot on the keyboard.
    """
    tones = chord_tones(chord, density)
    root = chord.bass if chord.bass is not None else chord.root
    low = center - ((center - root) % 12)  # highest note <= center with the root's pitch class
    notes = [low]
    for pc in tones:
        if pc == root % 12:
            continue
        n = low + ((pc - low) % 12)
        notes.append(n)
    return sorted(set(notes))


def chord_hit(
    chord: Chord, x: float, intensity: float, vibe: float, cfg: MusicConfig, pcfg: PianoConfig
) -> list[NoteEvent]:
    """Open hand hit down: a block chord at the hand's position, rolled slightly."""
    center = key_at(x, pcfg)
    notes = voicing_at(chord, center, vibe)
    if vibe > 0.85 or intensity > 0.8:
        notes = [notes[0] - 12, *notes]  # weight in the bass when leaning into it
    notes = [n for n in notes if 21 <= n <= 108]
    vel = velocity(cfg, intensity, vibe)
    roll = _lerp(0.012, 0.003, intensity)
    return [
        NoteEvent(offset=i * roll, note=n, velocity=vel, duration=cfg.piano_sustain, channel=CH_PIANO)
        for i, n in enumerate(notes)
    ]


def lattice(chord: Chord, vibe: float, pcfg: PianoConfig) -> list[int]:
    """Chord tones tiled across the keyboard span, ascending. Slot i is the i-th of these."""
    tones = sorted(chord_tones(chord, vibe))
    notes = [n for n in range(pcfg.key_low, pcfg.key_high + 1) if n % 12 in tones]
    return notes


def slot_at(x: float, notes: list[int], pcfg: PianoConfig) -> int:
    """Which lattice slot the hand is over. The keyboard span is shared with key_at."""
    key = key_at(x, pcfg)
    best = 0
    for i, n in enumerate(notes):
        if n <= key:
            best = i
    return best


def sweep_note(
    notes: list[int], slot: int, speed: float, vibe: float, cfg: MusicConfig, pcfg: PianoConfig
) -> NoteEvent:
    slot = max(0, min(len(notes) - 1, slot))
    # Speed is the tempo; it also sets how hard the key is struck.
    intensity = _clip((speed - pcfg.sweep_speed_on) / (pcfg.hit_speed_full - pcfg.sweep_speed_on), 0.0, 1.0)
    vel = velocity(cfg, 0.3 + 0.7 * intensity, vibe)
    duration = _lerp(0.9, 0.25, intensity)  # fast sweeps ring short, like a glissando
    return NoteEvent(offset=0.0, note=notes[slot], velocity=vel, duration=duration, channel=CH_PIANO)


class Zigzag:
    """Random zigzag notes around the hand, drifting where the hand is heading.

    Stateful per hand so consecutive notes make a line, not a lottery: each
    note steps 1-3 scale degrees, flipping direction most of the time (the
    zigzag), with the hand's drift pulling the centre of gravity along.
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._last: int | None = None
        self._dir = 1

    def next(
        self, chord: Chord, x: float, drift: str, activity: float, vibe: float, cfg: MusicConfig, pcfg: PianoConfig
    ) -> NoteEvent:
        scale = chord_scale(chord)
        centre = key_at(x, pcfg)
        # Stay within an octave of the hand; the hand moving pulls the pattern with it.
        if self._last is None or abs(self._last - centre) > 12:
            self._last = _nearest_scale_note(centre, scale)
        if self._rng.random() < 0.7:
            self._dir = -self._dir
        if drift == "right" and self._rng.random() < 0.6:
            self._dir = 1
        elif drift == "left" and self._rng.random() < 0.6:
            self._dir = -1
        note = _scale_step(self._last, scale, self._dir * self._rng.choice((1, 1, 2, 3)))
        if abs(note - centre) > 12:  # bounce off the octave walls around the hand
            self._dir = -self._dir
            note = _scale_step(self._last, scale, self._dir * 2)
        note = max(21, min(108, note))
        self._last = note
        vel = velocity(cfg, 0.3 + 0.7 * _clip(activity, 0, 1), vibe)
        vel = int(_clip(vel * self._rng.uniform(0.85, 1.0), 1, 127))
        return NoteEvent(offset=0.0, note=note, velocity=vel, duration=0.35, channel=CH_PIANO)


def _nearest_scale_note(midi: int, scale: tuple[int, ...]) -> int:
    for d in range(12):
        for cand in (midi - d, midi + d):
            if cand % 12 in scale:
                return cand
    return midi


def _scale_step(midi: int, scale: tuple[int, ...], steps: int) -> int:
    n = midi
    direction = 1 if steps > 0 else -1
    for _ in range(abs(steps)):
        n += direction
        guard = 0
        while n % 12 not in scale and guard < 12:
            n += direction
            guard += 1
    return n
