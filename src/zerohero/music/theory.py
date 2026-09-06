"""Chord symbol parsing and pitch-class theory. See ARCHITECTURE.md > music."""

from __future__ import annotations

import re
from dataclasses import dataclass

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# Natural-letter pitch classes; sharp/flat accidentals shift by +-1 semitone.
_ROOT_PITCH = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# Root letters are uppercase A-G only; lowercase "b" after a letter is the flat
# accidental, never a root itself ("Bb" is B-flat, not two roots).
_ROOT_RE = re.compile(r"^([A-G])(#|b)?")

# Every accepted quality suffix, keyed exactly (no prefix matching) so "m",
# "m7" and "m7b5" cannot be confused with each other.
_QUALITY_ALIASES = {
    "": "maj",
    "maj": "maj",
    "M": "maj",
    "maj7": "maj7",
    "M7": "maj7",
    "Δ7": "maj7",
    "Δ": "maj7",
    "maj9": "maj9",
    "m": "m",
    "min": "m",
    "-": "m",
    "m7": "m7",
    "m9": "m9",
    "m6": "m6",
    "dim": "dim",
    "°": "dim",
    "dim7": "dim7",
    "m7b5": "m7b5",
    "ø": "m7b5",
    "aug": "aug",
    "+": "aug",
    "sus2": "sus2",
    "sus4": "sus4",
    "add9": "add9",
    "6": "6",
    "9": "9",
    "7": "7",
    "5": "5",
}

# Semitone offsets from the root for each canonical quality.
_QUALITY_INTERVALS: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "maj7": (0, 4, 7, 11),
    "maj9": (0, 4, 7, 11, 14),
    "m": (0, 3, 7),
    "m7": (0, 3, 7, 10),
    "m9": (0, 3, 7, 10, 14),
    "m6": (0, 3, 7, 9),
    "dim": (0, 3, 6),
    "dim7": (0, 3, 6, 9),
    "m7b5": (0, 3, 6, 10),
    "aug": (0, 4, 8),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "add9": (0, 4, 7, 14),
    "6": (0, 4, 7, 9),
    "9": (0, 4, 7, 10, 14),
    "7": (0, 4, 7, 10),
    "5": (0, 7),
}

# How each canonical quality renders in a canonical symbol, e.g. "Amin" -> "Am".
_QUALITY_SUFFIX: dict[str, str] = {
    "maj": "",
    "maj7": "maj7",
    "maj9": "maj9",
    "m": "m",
    "m7": "m7",
    "m9": "m9",
    "m6": "m6",
    "dim": "dim",
    "dim7": "dim7",
    "m7b5": "m7b5",
    "aug": "aug",
    "sus2": "sus2",
    "sus4": "sus4",
    "add9": "add9",
    "6": "6",
    "9": "9",
    "7": "7",
    "5": "5",
}

# Scale (semitone offsets from root) used for runs/glissandi per quality.
_SCALE_BY_QUALITY: dict[str, tuple[int, ...]] = {
    "maj": (0, 2, 4, 5, 7, 9, 11),  # ionian
    "maj7": (0, 2, 4, 5, 7, 9, 11),
    "maj9": (0, 2, 4, 5, 7, 9, 11),
    "6": (0, 2, 4, 5, 7, 9, 11),
    "add9": (0, 2, 4, 5, 7, 9, 11),
    "sus2": (0, 2, 4, 5, 7, 9, 11),
    "sus4": (0, 2, 4, 5, 7, 9, 11),
    "5": (0, 2, 4, 5, 7, 9, 11),
    "m": (0, 2, 3, 5, 7, 8, 10),  # aeolian
    "m7": (0, 2, 3, 5, 7, 8, 10),
    "m9": (0, 2, 3, 5, 7, 8, 10),
    "m6": (0, 2, 3, 5, 7, 9, 10),  # dorian (natural 6th)
    "7": (0, 2, 4, 5, 7, 9, 10),  # mixolydian
    "9": (0, 2, 4, 5, 7, 9, 10),
    "dim": (0, 2, 3, 5, 6, 8, 9, 11),  # whole-half
    "dim7": (0, 2, 3, 5, 6, 8, 9, 11),
    "m7b5": (0, 1, 3, 5, 6, 8, 10),  # locrian
    "aug": (0, 2, 4, 6, 8, 10),  # whole tone
}


class ChordError(ValueError):
    """Raised when a chord symbol can't be parsed. Message names the bad token."""


@dataclass(frozen=True, slots=True)
class Chord:
    root: int  # pitch class 0..11
    quality: str  # canonical quality key, see _QUALITY_INTERVALS
    bass: int | None  # slash-bass pitch class, or None
    symbol: str  # canonical rendering, e.g. "Am7", "C/G"
    intervals: tuple[int, ...]  # semitone offsets from root

    def pitch_classes(self) -> tuple[int, ...]:
        """Chord tones, bass pitch class first if it isn't already one of them."""
        pcs = tuple((self.root + iv) % 12 for iv in self.intervals)
        if self.bass is not None and self.bass not in pcs:
            pcs = (self.bass, *pcs)
        return pcs

    @property
    def name(self) -> str:
        return self.symbol

    def __str__(self) -> str:
        return self.symbol


def _parse_root(text: str, whole_symbol: str) -> tuple[int, str]:
    """Returns (pitch_class, matched_text) or raises ChordError naming whole_symbol."""
    m = _ROOT_RE.match(text)
    if not m:
        raise ChordError(f"unknown root {text!r} in chord {whole_symbol!r}")
    letter, accidental = m.group(1), m.group(2) or ""
    pc = _ROOT_PITCH[letter]
    if accidental == "#":
        pc = (pc + 1) % 12
    elif accidental == "b":
        pc = (pc - 1) % 12
    return pc, letter + accidental


def parse_chord(symbol: str) -> Chord:
    raw = symbol.strip()
    if not raw:
        raise ChordError("empty chord symbol")

    if "/" in raw:
        main_text, bass_text = raw.split("/", 1)
    else:
        main_text, bass_text = raw, None

    root_pc, root_text = _parse_root(main_text, raw)
    remainder = main_text[len(root_text) :]
    if remainder not in _QUALITY_ALIASES:
        raise ChordError(f"unknown chord quality {remainder!r} in chord {raw!r}")
    quality = _QUALITY_ALIASES[remainder]

    bass_pc = None
    bass_text_render = None
    if bass_text is not None:
        bm = _ROOT_RE.match(bass_text)
        if not bm or bm.group(0) != bass_text:
            raise ChordError(f"unknown bass note {bass_text!r} in chord {raw!r}")
        bass_pc, bass_text_render = _parse_root(bass_text, raw)

    intervals = _QUALITY_INTERVALS[quality]
    canonical = root_text + _QUALITY_SUFFIX[quality]
    if bass_text_render is not None:
        canonical += f"/{bass_text_render}"

    return Chord(root=root_pc, quality=quality, bass=bass_pc, symbol=canonical, intervals=intervals)


def parse_progression(text: str) -> list[Chord]:
    stripped = text.strip()
    if not stripped:
        raise ChordError("empty progression")

    tokens = [t for t in re.split(r"[\s,|]+", stripped) if t]
    if not tokens:
        raise ChordError("empty progression")

    chords: list[Chord] = []
    for tok in tokens:
        # A "-" is a minor suffix directly on a root (e.g. "A-") unless it also
        # works as a separator between two whole chords (e.g. "Am-G"); try the
        # latter first since it only succeeds when both halves are real chords.
        parts = tok.split("-")
        if len(parts) > 1 and all(parts):
            try:
                chords.extend(parse_chord(p) for p in parts)
                continue
            except ChordError:
                pass
        chords.append(parse_chord(tok))
    return chords


def midi(pc: int, octave: int) -> int:
    """C4 = 60."""
    return (octave + 1) * 12 + pc


def note_name(midi: int) -> str:
    octave = midi // 12 - 1
    return f"{NOTE_NAMES[midi % 12]}{octave}"


def chord_scale(chord: Chord) -> tuple[int, ...]:
    """Pitch classes of a scale fitting the chord, for runs/glissandi."""
    scale = _SCALE_BY_QUALITY.get(chord.quality, _SCALE_BY_QUALITY["maj"])
    return tuple((chord.root + iv) % 12 for iv in scale)
