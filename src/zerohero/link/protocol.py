"""Wire format: one JSON object per line, UTF-8. See ARCHITECTURE.md "link".

TCP and Bluetooth RFCOMM share this codec since both are stream sockets and
the framing (newline-delimited JSON) doesn't care which.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


class ProtocolError(Exception):
    """Malformed line or unknown message type."""


@dataclass(frozen=True, slots=True)
class Hello:
    role: str  # "lead" | "follow"
    name: str
    version: int = 1


@dataclass(frozen=True, slots=True)
class ChordMsg:
    index: int
    symbol: str
    t: float  # lead's time.monotonic()


@dataclass(frozen=True, slots=True)
class StrumMsg:
    direction: str  # "down" | "up"
    intensity: float
    t: float  # lead's time.monotonic()


@dataclass(frozen=True, slots=True)
class Ping:
    t: float


@dataclass(frozen=True, slots=True)
class Pong:
    t: float  # echoed from the Ping
    rt: float  # responder's monotonic time when it answered


@dataclass(frozen=True, slots=True)
class Bye:
    pass


Message = Hello | ChordMsg | StrumMsg | Ping | Pong | Bye

_TYPE_NAMES = {
    Hello: "hello",
    ChordMsg: "chord",
    StrumMsg: "strum",
    Ping: "ping",
    Pong: "pong",
    Bye: "bye",
}
_NAME_TYPES = {v: k for k, v in _TYPE_NAMES.items()}


def encode(msg: Message) -> bytes:
    d = asdict(msg)
    d["type"] = _TYPE_NAMES[type(msg)]
    return (json.dumps(d, separators=(",", ":")) + "\n").encode("utf-8")


def decode(line: bytes) -> Message:
    try:
        d = json.loads(line.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ProtocolError(f"invalid JSON: {e}") from e
    if not isinstance(d, dict) or "type" not in d:
        raise ProtocolError("missing 'type' field")
    cls = _NAME_TYPES.get(d["type"])
    if cls is None:
        raise ProtocolError(f"unknown message type: {d['type']!r}")
    fields = {k: v for k, v in d.items() if k != "type"}
    try:
        return cls(**fields)
    except TypeError as e:
        raise ProtocolError(f"malformed {d['type']} message: {e}") from e
