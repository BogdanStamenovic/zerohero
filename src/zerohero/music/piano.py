"""Piano: the hand is the pianist. See ARCHITECTURE.md "piano".

Three ways to play, all over the chord the guitar is on:

- grip hit      -> block chord voiced around the keyboard position of the hand
- sweep         -> the chord tones tiled across the keyboard; the hand crossing
                   a slot plays that tone, so hand speed is the tempo
- passage       -> a run through the chord's scale from the hand's position
"""

from __future__ import annotations

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


def grip_chord(
    chord: Chord, x: float, intensity: float, vibe: float, downward: bool, cfg: MusicConfig, pcfg: PianoConfig
) -> list[NoteEvent]:
    """Block chord at the hand's position. Held by the mode until the grip opens."""
    center = key_at(x, pcfg)
    notes = voicing_at(chord, center, vibe)
    if downward or vibe > 0.85:
        notes = [notes[0] - 12, *notes]  # weight in the bass, like leaning into it
    notes = [n for n in notes if 21 <= n <= 108]
    vel = velocity(cfg, intensity, vibe)
    if downward:
        vel = min(127, vel + 10)
    # Tiny roll so the chord sounds like fingers, not a sequencer.
    roll = _lerp(0.012, 0.003, intensity)
    return [
        NoteEvent(offset=i * roll, note=n, velocity=vel, duration=cfg.piano_sustain * 3, channel=CH_PIANO)
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


def passage(
    chord: Chord,
    x: float,
    direction: str,
    intensity: float,
    vibe: float,
    erratic: bool,
    cfg: MusicConfig,
    pcfg: PianoConfig,
) -> list[NoteEvent]:
    """A run through the chord scale from the hand's position, up or down.

    Erratic motion gives a zigzag instead of a straight run.
    """
    scale = chord_scale(chord)
    start = key_at(x, pcfg)
    steps = 4 + int(round(5 * _clip(intensity, 0, 1)) + round(2 * _clip(vibe, 0, 1)))
    spacing = _lerp(0.12, 0.045, intensity)
    step = 1 if direction == "up" else -1
    run: list[int] = []
    n = start
    # Walk semitone by semitone, keeping scale tones, until we have enough.
    guard = 0
    while len(run) < steps and guard < 60:
        guard += 1
        n += step
        if n % 12 in scale and 21 <= n <= 108:
            run.append(n)
    if erratic and len(run) >= 4:
        # zigzag: 0 2 1 3 2 4 ... keeps the direction but wobbles
        zig = []
        for i in range(len(run)):
            j = i + 1 if i % 2 == 0 and i + 1 < len(run) else i - 1 if i % 2 == 1 else i
            zig.append(run[max(0, min(len(run) - 1, j))])
        run = zig
    base_vel = velocity(cfg, intensity, vibe)
    events = []
    for i, note in enumerate(run):
        # crescendo into the top of the run
        v = int(_clip(base_vel * (0.75 + 0.25 * i / max(1, len(run) - 1)), 1, 127))
        events.append(NoteEvent(offset=i * spacing, note=note, velocity=v, duration=spacing * 2.2, channel=CH_PIANO))
    return events
