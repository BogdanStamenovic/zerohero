"""zerohero command line.

zerohero guitar "Am G C F"          strum with the right hand, left fist = next chord
zerohero piano  "Am G C F"          conduct with either hand
zerohero guitar "Am G" --lead        broadcast chords/strums to followers
zerohero piano  "Am G" --follow auto follow a guitar device on the LAN
zerohero chords "Am G"               print voicings and play them, no camera
zerohero setup / doctor / calibrate
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from zerohero import __version__
from zerohero.config import Config, load_config

log = logging.getLogger("zerohero")


def _add_play_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("progression", help="chords in order, e.g. 'Am G C F' or 'C | G | Am | F'")
    p.add_argument("--camera", type=int, help="camera index (default 0)")
    p.add_argument("--no-mirror", action="store_true", help="camera already mirrors; swap handedness instead")
    p.add_argument(
        "--ui",
        choices=["auto", "window", "terminal", "none"],
        default="auto",
        help="hand-tracking view: OpenCV window, braille view in the terminal, or none "
        "(auto: window if a display exists, else terminal)",
    )
    p.add_argument("--no-window", action="store_true", help="same as --ui terminal")
    p.add_argument("--record", type=Path, metavar="FILE", help="write hand landmarks as JSONL for replay")
    p.add_argument("--replay", type=Path, metavar="FILE", help="run from a recorded JSONL instead of the camera")
    p.add_argument("--synth", choices=["auto", "fluid", "basic", "none"], help="audio backend")
    p.add_argument("--soundfont", type=Path, help="path to an .sf2/.sf3 for fluidsynth")
    p.add_argument("--audio-driver", help="fluidsynth audio driver (pulseaudio, alsa, coreaudio, dsound)")
    p.add_argument("--left-handed", action="store_true", help="strum with the left hand, fist with the right")
    p.add_argument("--name", help="device name shown to linked peers")
    p.add_argument("--port", type=int, help="link TCP port (default 47475)")
    p.add_argument("-v", "--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="zerohero", description="air guitar and air piano from a webcam")
    p.add_argument("--version", action="version", version=f"zerohero {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("guitar", help="air guitar")
    _add_play_args(g)
    g.add_argument("--lead", action="store_true", help="broadcast chords and strums to piano followers")
    g.add_argument("--transport", choices=["tcp", "bt"], default="tcp", help="lead transport")

    pi = sub.add_parser("piano", help="air piano, accompanying a guitar session")
    _add_play_args(pi)
    pi.add_argument("--follow", metavar="TARGET", help="guitar to follow: auto | host[:port] | bt:AA:BB:CC:DD:EE:FF")

    c = sub.add_parser("chords", help="print voicings and play the progression, no camera")
    c.add_argument("progression")
    c.add_argument("--synth", choices=["auto", "fluid", "basic", "none"])
    c.add_argument("--silent", action="store_true")

    s = sub.add_parser("setup", help="download the hand model and soundfont")
    s.add_argument("--force", action="store_true")

    sub.add_parser("doctor", help="check camera, model, soundfont, audio, bluetooth")

    cal = sub.add_parser("calibrate", help="show the camera with gesture readouts, print events")
    cal.add_argument("--camera", type=int)
    cal.add_argument("--no-mirror", action="store_true")
    cal.add_argument("--ui", choices=["auto", "window", "terminal", "none"], default="auto")
    cal.add_argument("--no-window", action="store_true", help="same as --ui terminal")
    cal.add_argument("-v", "--verbose", action="store_true")
    return p


def _config_from_args(a: argparse.Namespace, mode: str) -> Config:
    cfg = load_config()
    cfg.mode = mode
    cfg.progression = a.progression
    cfg.verbose = a.verbose
    if a.camera is not None:
        cfg.camera.index = a.camera
    if a.no_mirror:
        cfg.camera.mirror = False
    cfg.ui = "terminal" if a.no_window else a.ui
    cfg.record = a.record
    cfg.replay = a.replay
    if a.synth:
        cfg.synth.backend = a.synth
    if a.soundfont:
        cfg.synth.soundfont = a.soundfont
    if a.audio_driver:
        cfg.synth.audio_driver = a.audio_driver
    if a.left_handed:
        cfg.gesture.strum_hand, cfg.gesture.fist_hand = "left", "right"
    if a.name:
        cfg.link.name = a.name
    if a.port:
        cfg.link.port = a.port
    if mode == "guitar":
        cfg.lead = a.lead
        cfg.lead_transport = a.transport
    else:
        cfg.follow = a.follow
    return cfg


def cmd_play(a: argparse.Namespace, mode: str) -> int:
    cfg = _config_from_args(a, mode)
    from zerohero.engine.app import run

    run(cfg)
    return 0


def cmd_chords(a: argparse.Namespace) -> int:
    import time

    from zerohero.config import CH_GUITAR, CH_PIANO
    from zerohero.music import guitar, piano, theory
    from zerohero.synth import Scheduler, open_synth

    cfg = load_config()
    cfg.progression = a.progression
    if a.silent:
        cfg.synth.backend = "none"
    elif a.synth:
        cfg.synth.backend = a.synth
    chords = theory.parse_progression(a.progression)
    for c in chords:
        print(guitar.describe(c))
    if a.silent:
        return 0
    synth = open_synth(cfg)
    sched = Scheduler(synth)
    try:
        for c in chords:
            sched.cut(CH_GUITAR)
            sched.play(guitar.strum(c, "down", 0.5, cfg.music))
            time.sleep(0.9)
            sched.play(piano.chord_hit(c, 0.55, 0.5, 0.4, cfg.music, cfg.piano))
            time.sleep(1.1)
        time.sleep(1.0)
    finally:
        sched.cut(CH_GUITAR)
        sched.cut(CH_PIANO)
        sched.stop()
        synth.close()
    return 0


def cmd_setup(a: argparse.Namespace) -> int:
    from zerohero.assets import ensure_assets

    try:
        fetched = ensure_assets(force=a.force)
    except RuntimeError as e:
        print(f"setup failed: {e}", file=sys.stderr)
        return 1
    print("assets ready" + (f" ({len(fetched)} downloaded)" if fetched else " (nothing to download)"))
    return 0


def cmd_doctor(_: argparse.Namespace) -> int:
    from zerohero.doctor import run_doctor

    return run_doctor()


def cmd_calibrate(a: argparse.Namespace) -> int:
    from zerohero.doctor import run_calibrate

    cfg = load_config()
    if a.camera is not None:
        cfg.camera.index = a.camera
    if a.no_mirror:
        cfg.camera.mirror = False
    cfg.ui = "terminal" if a.no_window else a.ui
    cfg.verbose = a.verbose
    return run_calibrate(cfg)


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(a, "verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    try:
        if a.cmd in ("guitar", "piano"):
            return cmd_play(a, a.cmd)
        if a.cmd == "chords":
            return cmd_chords(a)
        if a.cmd == "setup":
            return cmd_setup(a)
        if a.cmd == "doctor":
            return cmd_doctor(a)
        if a.cmd == "calibrate":
            return cmd_calibrate(a)
    except KeyboardInterrupt:
        return 130
    except Exception as e:  # user-facing boundary: one clear line, details with -v
        if getattr(a, "verbose", False):
            raise
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
