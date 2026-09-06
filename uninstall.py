#!/usr/bin/env python3
"""Remove the venv and downloaded assets. ownbox removes the checkout itself."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def data_dir() -> Path:
    if env := os.environ.get("ZEROHERO_DATA"):
        return Path(env).expanduser()
    base = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(base) / "zerohero"


def main() -> None:
    targets = [ROOT / ".venv", data_dir()]
    if "--yes" not in sys.argv:
        print("will remove:")
        for t in targets:
            print(f"  {t}")
        if input("continue? [y/N] ").strip().lower() != "y":
            sys.exit("aborted")
    for t in targets:
        if t.exists():
            shutil.rmtree(t)
            print(f"removed {t}")


if __name__ == "__main__":
    main()
