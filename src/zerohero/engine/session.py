"""Progression state: which chord is current, stepping through it."""

from __future__ import annotations

from zerohero.music.theory import Chord, parse_progression


class Session:
    def __init__(self, progression: list[Chord] | str) -> None:
        self.chords = parse_progression(progression) if isinstance(progression, str) else list(progression)
        if not self.chords:
            raise ValueError("empty progression")
        self.index = 0
        self.version = 0  # bumps on every change so listeners can notice

    @property
    def current(self) -> Chord:
        return self.chords[self.index]

    @property
    def symbols(self) -> list[str]:
        return [c.symbol for c in self.chords]

    def goto(self, index: int) -> Chord:
        self.index = index % len(self.chords)
        self.version += 1
        return self.current

    def next(self) -> Chord:
        return self.goto(self.index + 1)

    def prev(self) -> Chord:
        return self.goto(self.index - 1)

    def reset(self) -> Chord:
        return self.goto(0)

    def set_symbol(self, index: int, symbol: str) -> Chord:
        """Follower path: the lead's chord may not be in our list."""
        from zerohero.music.theory import parse_chord

        chord = parse_chord(symbol)
        if 0 <= index < len(self.chords) and self.chords[index].symbol == chord.symbol:
            return self.goto(index)
        if index >= len(self.chords):
            self.chords.extend([chord] * (index + 1 - len(self.chords)))
        self.chords[index] = chord
        return self.goto(index)
