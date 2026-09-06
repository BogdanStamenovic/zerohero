#!/usr/bin/env python3
"""Set up zerohero in a local .venv and fetch its assets. Run by ownbox, or by hand.

Python 3.12 is required by MediaPipe 0.10 (1.0.x on 3.14 crashes on this
hardware, see ARCHITECTURE.md). Resolution order for an interpreter:
`uv` (downloads 3.12 if needed), then `python3.12` on PATH.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
WINDOWS = os.name == "nt"
PY = VENV / ("Scripts/python.exe" if WINDOWS else "bin/python")


def run(*cmd: str | Path) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT)


def create_venv() -> None:
    if PY.exists():
        out = subprocess.run([str(PY), "-c", "import sys;print(sys.version_info[:2])"], capture_output=True, text=True)
        if "(3, 12)" in out.stdout:
            print(f"reusing {VENV}")
            return
        print(f"{VENV} is not Python 3.12, recreating")
        shutil.rmtree(VENV)
    if uv := shutil.which("uv"):
        run(uv, "venv", "-p", "3.12", VENV)
        return
    py312 = shutil.which("python3.12") or shutil.which("python3.12.exe")
    if not py312:
        sys.exit(
            "need Python 3.12: install `uv` (https://docs.astral.sh/uv/) so it can be provisioned "
            "automatically, or install python3.12 from your package manager"
        )
    run(py312, "-m", "venv", VENV)


def install_package() -> None:
    if uv := shutil.which("uv"):
        run(uv, "pip", "install", "-q", "-p", PY, "-e", ROOT)
    else:
        run(PY, "-m", "pip", "install", "-q", "-e", ROOT)


def check_fluidsynth() -> None:
    probe = "import ctypes.util,sys; sys.exit(0 if ctypes.util.find_library('fluidsynth') else 1)"
    if subprocess.run([str(PY), "-c", probe]).returncode != 0:
        hint = {
            "linux": "sudo pacman -S fluidsynth   (Debian: apt install libfluidsynth3)",
            "darwin": "brew install fluid-synth",
            "win32": "download fluidsynth from https://github.com/FluidSynth/fluidsynth/releases, put the DLL on PATH",
        }.get(sys.platform, "install libfluidsynth")
        print(f"warning: libfluidsynth not found; the built-in numpy synth will be used.\n  better sound: {hint}")


def main() -> None:
    create_venv()
    install_package()
    check_fluidsynth()
    run(PY, "-m", "zerohero", "setup")
    print("\nzerohero installed. Try:\n  zerohero doctor\n  zerohero guitar 'Am G C F'")


if __name__ == "__main__":
    main()
