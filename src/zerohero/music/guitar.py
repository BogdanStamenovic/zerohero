"""Guitar voicings and strums. See ARCHITECTURE.md > music."""

from __future__ import annotations

from zerohero.config import CH_GUITAR, MusicConfig
from zerohero.events import NoteEvent, StrumDirection
from zerohero.music.theory import Chord

STANDARD_TUNING = (40, 45, 50, 55, 59, 64)  # E2 A2 D3 G3 B3 E4, low to high

# Open-position shapes, fret per string (None = muted), keyed by (root pc, quality).
# These are standard fingerings; picked because they reproduce the real chord
# tones exactly (verified against Chord.pitch_classes()) rather than guessed.
_OPEN_SHAPES: dict[tuple[int, str], tuple[int | None, ...]] = {
    (0, "maj"): (None, 3, 2, 0, 1, 0),  # C
    (2, "maj"): (None, None, 0, 2, 3, 2),  # D
    (2, "m"): (None, None, 0, 2, 3, 1),  # Dm
    (2, "7"): (None, None, 0, 2, 1, 2),  # D7
    (4, "maj"): (0, 2, 2, 1, 0, 0),  # E
    (4, "m"): (0, 2, 2, 0, 0, 0),  # Em
    (4, "7"): (0, 2, 0, 1, 0, 0),  # E7
    (5, "maj"): (1, 3, 3, 2, 1, 1),  # F (full barre)
    (7, "maj"): (3, 2, 0, 0, 0, 3),  # G
    (7, "7"): (3, 2, 0, 0, 0, 1),  # G7
    (9, "maj"): (None, 0, 2, 2, 2, 0),  # A
    (9, "m"): (None, 0, 2, 2, 1, 0),  # Am
    (9, "7"): (None, 0, 2, 0, 2, 0),  # A7
    (11, "7"): (None, 2, 1, 2, 0, 2),  # B7
    (11, "m"): (None, 2, 4, 4, 3, 2),  # Bm
    (0, "maj7"): (None, 3, 2, 0, 0, 0),  # Cmaj7
    (2, "maj7"): (None, None, 0, 2, 2, 2),  # Dmaj7
    (9, "maj7"): (None, 0, 2, 1, 2, 0),  # Amaj7
    (4, "m7"): (0, 2, 0, 0, 0, 0),  # Em7
    (9, "m7"): (None, 0, 2, 0, 1, 0),  # Am7
    (2, "sus2"): (None, None, 0, 2, 3, 0),  # Dsus2
    (2, "sus4"): (None, None, 0, 2, 3, 3),  # Dsus4
    (9, "sus2"): (None, 0, 2, 2, 0, 0),  # Asus2
    (9, "sus4"): (None, 0, 2, 2, 3, 0),  # Asus4
    (4, "sus4"): (0, 0, 2, 2, 0, 0),  # Esus4
    (0, "add9"): (None, 3, 2, 0, 3, 0),  # Cadd9
    (5, "maj7"): (None, None, 3, 2, 1, 0),  # Fmaj7
}

# Movable barre fallback, offsets relative to the barre fret, root on string 0.
_E_SHAPE_OFFSETS: dict[str, tuple[int | None, ...]] = {
    "maj": (0, 2, 2, 1, 0, 0),
    "m": (0, 2, 2, 0, 0, 0),
    "7": (0, 2, 0, 1, 0, 0),
    "m7": (0, 2, 0, 0, 0, 0),
}
# Root on string 1; string 0 is muted in the A shape.
_A_SHAPE_OFFSETS: dict[str, tuple[int | None, ...]] = {
    "maj": (None, 0, 2, 2, 2, 0),
    "m": (None, 0, 2, 2, 1, 0),
    "7": (None, 0, 2, 0, 2, 0),
    "m7": (None, 0, 2, 0, 1, 0),
}
_BARRE_QUALITIES = frozenset(_E_SHAPE_OFFSETS)


def _barre_frets(chord: Chord) -> list[int | None]:
    barre_e = (chord.root - STANDARD_TUNING[0]) % 12
    barre_a = (chord.root - STANDARD_TUNING[1]) % 12
    offsets, fret = (_E_SHAPE_OFFSETS, barre_e) if barre_e <= barre_a else (_A_SHAPE_OFFSETS, barre_a)
    return [None if off is None else fret + off for off in offsets[chord.quality]]


def _generic_frets(chord: Chord) -> list[int | None]:
    """Chord tones across the neck for qualities with no fixed shape.

    Lowest fret <= 5 per string that lands on a chord tone; mutes strings that
    can't reach one that close. Prefers the root on the lowest string, and
    widens the search on muted strings if that leaves fewer than 3 notes.
    """
    pcs = set(chord.pitch_classes())
    frets: list[int | None] = [None] * 6
    for i, open_note in enumerate(STANDARD_TUNING):
        for f in range(6):
            if (open_note + f) % 12 in pcs:
                frets[i] = f
                break

    if frets[0] is None or (STANDARD_TUNING[0] + frets[0]) % 12 != chord.root:
        for f in range(6):
            if (STANDARD_TUNING[0] + f) % 12 == chord.root:
                frets[0] = f
                break

    if sum(f is not None for f in frets) < 3:
        for i, open_note in enumerate(STANDARD_TUNING):
            if frets[i] is not None:
                continue
            for f in range(12):
                if (open_note + f) % 12 in pcs:
                    frets[i] = f
                    break
            if sum(f is not None for f in frets) >= 3:
                break
    return frets


def _apply_bass(chord: Chord, frets: list[int | None]) -> list[int | None]:
    """Slash bass: put the bass pitch class on the lowest (low-E) string."""
    if chord.bass is None:
        return frets
    frets = list(frets)
    frets[0] = (chord.bass - STANDARD_TUNING[0]) % 12
    return frets


def _frets(chord: Chord) -> list[int | None]:
    key = (chord.root, chord.quality)
    if key in _OPEN_SHAPES:
        frets = list(_OPEN_SHAPES[key])
    elif chord.quality in _BARRE_QUALITIES:
        frets = _barre_frets(chord)
    else:
        frets = _generic_frets(chord)
    return _apply_bass(chord, frets)


def voicing(chord: Chord) -> list[int | None]:
    """Six MIDI notes low E to high e, None where muted. Always >= 3 sounding."""
    frets = _frets(chord)
    notes: list[int | None] = [None if f is None else STANDARD_TUNING[i] + f for i, f in enumerate(frets)]
    if sum(n is not None for n in notes) < 3:
        # Defensive: only the generic path could undershoot, and it already
        # guards this, but never hand back an unplayable voicing.
        for i, (n, open_note) in enumerate(zip(notes, STANDARD_TUNING, strict=True)):
            if n is not None:
                continue
            notes[i] = open_note
            if sum(x is not None for x in notes) >= 3:
                break
    return notes


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clip(t, 0.0, 1.0)


def strum(chord: Chord, direction: StrumDirection, intensity: float, cfg: MusicConfig) -> list[NoteEvent]:
    notes = voicing(chord)
    sounding = [(i, n) for i, n in enumerate(notes) if n is not None]  # low to high

    order = sounding if direction == "down" else list(reversed(sounding))
    if direction == "up" and intensity < 0.5 and len(order) > 3:
        # Real up-strums are pick-hand flicks that mostly catch the treble
        # strings; skip 1-2 of the (now-trailing, since we reversed) low ones.
        skip = 2 if intensity < 0.25 else 1
        order = order[: len(order) - skip]

    spacing = _lerp(cfg.strum_spread_slow, cfg.strum_spread_fast, intensity)
    base_velocity = cfg.velocity_min + (cfg.velocity_max - cfg.velocity_min) * _clip(intensity, 0.0, 1.0)

    events = []
    for idx, (_string, note) in enumerate(order):
        accent = 10 if idx == 0 else 0
        velocity = round(_clip(base_velocity + accent, 1, 127))
        events.append(
            NoteEvent(
                offset=idx * spacing,
                note=note,
                velocity=velocity,
                duration=cfg.guitar_sustain,
                channel=CH_GUITAR,
            )
        )
    return events


def describe(chord: Chord) -> str:
    frets = _frets(chord)
    fret_str = "".join("x" if f is None else str(f) for f in frets)
    return f"{chord.name}  {fret_str}"
