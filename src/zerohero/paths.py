"""Where things live on disk."""

from __future__ import annotations

import os
from pathlib import Path

HAND_MODEL_NAME = "hand_landmarker.task"
SOUNDFONT_NAME = "GeneralUser-GS.sf2"


def data_dir() -> Path:
    if env := os.environ.get("ZEROHERO_DATA"):
        return Path(env).expanduser()
    base = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(base) / "zerohero"


def config_file() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "zerohero" / "config.toml"


def hand_model_path() -> Path:
    return data_dir() / HAND_MODEL_NAME


def soundfont_path() -> Path:
    return data_dir() / SOUNDFONT_NAME
