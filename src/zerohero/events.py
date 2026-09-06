"""Shared data types crossing module boundaries. See ARCHITECTURE.md."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Side = Literal["left", "right"]
StrumDirection = Literal["down", "up"]
BeatDirection = Literal["left", "right", "up", "down"]

# MediaPipe hand landmark indices.
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


@dataclass(frozen=True, slots=True)
class Landmark:
    x: float
    y: float
    z: float


@dataclass(slots=True)
class Hand:
    """One detected hand. `side` is the USER's hand after mirror correction."""

    side: Side
    score: float
    landmarks: list[Landmark]  # 21, normalised image coords, y down
    world: list[Landmark] | None = None  # metres, hand-centred, optional

    def to_json(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "score": self.score,
            "lm": [[p.x, p.y, p.z] for p in self.landmarks],
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Hand:
        return cls(
            side=d["side"],
            score=float(d.get("score", 1.0)),
            landmarks=[Landmark(*p) for p in d["lm"]],
        )


@dataclass(slots=True)
class Frame:
    t: float  # time.monotonic() seconds
    width: int
    height: int
    hands: list[Hand] = field(default_factory=list)
    image: Any = None  # BGR numpy array for the overlay, None in replay/headless

    def hand(self, side: Side) -> Hand | None:
        for h in self.hands:
            if h.side == side:
                return h
        return None

    def to_json(self) -> dict[str, Any]:
        return {"t": self.t, "w": self.width, "h": self.height, "hands": [h.to_json() for h in self.hands]}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Frame:
        return cls(
            t=float(d["t"]), width=int(d["w"]), height=int(d["h"]), hands=[Hand.from_json(h) for h in d["hands"]]
        )


# ---- gesture events ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Strum:
    t: float
    hand: Side
    direction: StrumDirection
    intensity: float  # 0..1


@dataclass(frozen=True, slots=True)
class FistClose:
    t: float
    hand: Side


@dataclass(frozen=True, slots=True)
class FistOpen:
    t: float
    hand: Side


@dataclass(frozen=True, slots=True)
class Beat:
    """A conducting hit: a sharp movement onset in one direction."""

    t: float
    hand: Side
    direction: BeatDirection
    intensity: float  # 0..1
    closed: bool  # fist at the moment of the hit


GestureEvent = Strum | FistClose | FistOpen | Beat


# ---- music ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NoteEvent:
    offset: float  # seconds relative to phrase start
    note: int  # MIDI 0..127
    velocity: int  # 1..127
    duration: float  # seconds until note-off
    channel: int
