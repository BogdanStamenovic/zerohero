"""All tunables in one place. CLI flags and ~/.config/zerohero/config.toml override defaults."""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from zerohero.paths import config_file, data_dir

CH_GUITAR = 0
CH_PIANO = 1
PROGRAM_GUITAR = 25  # GM steel-string acoustic
PROGRAM_PIANO = 0  # GM acoustic grand


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    mirror: bool = True
    max_hands: int = 2
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class GestureConfig:
    # Velocity smoothing: EMA factor per frame (0 = none, 1 = frozen).
    velocity_smoothing: float = 0.35
    # Seconds without a detection before a hand's track is dropped.
    lost_after: float = 0.4
    # A palm jump of more than this many hand widths in one frame is a tracker
    # glitch (re-acquired somewhere else), not motion: restart the track.
    teleport_widths: float = 2.5
    # Strum (pick hand, vertical speed, hand widths per second).
    strum_hand: str = "right"
    strum_speed_on: float = 3.5
    strum_speed_off: float = 1.5
    strum_speed_full: float = 12.0
    strum_refractory: float = 0.12
    # Fist (chord advance).
    fist_hand: str = "left"
    fist_close_above: float = 0.7
    fist_open_below: float = 0.4
    fist_min_hold: float = 0.06
    fist_refractory: float = 0.25
    # Conducting hits (piano), 2D speed.
    beat_speed_on: float = 3.0
    beat_speed_off: float = 1.2
    beat_speed_full: float = 11.0
    beat_refractory: float = 0.15
    # Vibe meter.
    vibe_window: float = 1.5
    vibe_speed_full: float = 5.0


@dataclass
class PianoConfig:
    """Piano mode: the hand is the pianist. Speeds in hand widths per second."""

    # The limiter is the keyboard surface: a horizontal line at this fraction
    # of the frame height (y grows downward). Above it the hand only travels;
    # below it, sideways movement sweeps the keys.
    limiter_y: float = 0.55
    # An open hand hitting downward is a chord. Measured live: deliberate hits
    # peak at 6-20 hand widths/s, casual moves at 1-3.
    hit_speed_on: float = 4.0
    hit_speed_off: float = 1.5
    hit_speed_full: float = 14.0
    hit_refractory: float = 0.2
    # Sweep: below the limiter, sideways speed above this fires a note per
    # lattice slot crossed.
    sweep_speed_on: float = 1.5
    sweep_min_gap: float = 0.03
    # Wiggle: fingertips moving relative to the palm (hand widths/s, EMA).
    # Tracker jitter alone reads as ~2; deliberate wiggling as 6 and up.
    wiggle_on: float = 4.0
    wiggle_off: float = 2.0
    wiggle_min_hold: float = 0.12  # activity must stay up this long: rejects single-frame jitter spikes
    wiggle_gap_slow: float = 0.16  # seconds between zigzag notes at wiggle_on
    wiggle_gap_fast: float = 0.07  # ... at wiggle_full
    wiggle_full: float = 12.0
    # Keyboard span the frame width maps onto (MIDI): C2 .. C6.
    key_low: int = 36
    key_high: int = 84


@dataclass
class MusicConfig:
    guitar_program: int = PROGRAM_GUITAR
    piano_program: int = PROGRAM_PIANO
    # Strum spread in seconds between adjacent strings at intensity 0 and 1.
    strum_spread_slow: float = 0.045
    strum_spread_fast: float = 0.012
    guitar_sustain: float = 2.5
    piano_sustain: float = 1.8
    piano_staccato: float = 0.25
    velocity_min: int = 40
    velocity_max: int = 127


@dataclass
class SynthConfig:
    backend: str = "auto"  # auto | fluid | basic
    soundfont: Path | None = None
    audio_driver: str | None = None  # fluidsynth driver: pulseaudio, alsa, coreaudio, dsound
    samplerate: int = 48000
    gain: float = 0.8


@dataclass
class LinkConfig:
    port: int = 47475
    discovery_port: int = 47474
    bt_channel: int = 3
    name: str = ""
    discover_timeout: float = 5.0


@dataclass
class Config:
    mode: str = "guitar"  # guitar | piano
    progression: str = ""
    ui: str = "auto"  # auto | window | terminal | none
    record: Path | None = None
    replay: Path | None = None
    replay_realtime: bool = True
    lead: bool = False
    lead_transport: str = "tcp"  # tcp | bt
    follow: str | None = None  # auto | host[:port] | bt:ADDR
    verbose: bool = False
    camera: CameraConfig = field(default_factory=CameraConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    music: MusicConfig = field(default_factory=MusicConfig)
    piano: PianoConfig = field(default_factory=PianoConfig)
    synth: SynthConfig = field(default_factory=SynthConfig)
    link: LinkConfig = field(default_factory=LinkConfig)

    @property
    def data_dir(self) -> Path:
        return data_dir()


def _apply(obj: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if not hasattr(obj, key):
            raise ValueError(f"unknown config key: {key}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _apply(current, value)
        elif isinstance(current, Path) or (current is None and key in ("soundfont", "record", "replay")):
            setattr(obj, key, Path(value).expanduser() if value else None)
        else:
            setattr(obj, key, value)


def load_config(path: Path | None = None) -> Config:
    """Defaults, then the TOML file if it exists. CLI flags are applied by cli.py afterwards."""
    cfg = Config()
    path = path or config_file()
    if path.exists():
        with path.open("rb") as f:
            _apply(cfg, tomllib.load(f))
    return cfg
