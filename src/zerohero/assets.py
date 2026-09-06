"""Download and verify the hand model and the soundfont into the data dir."""

from __future__ import annotations

import hashlib
import logging
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from zerohero.paths import data_dir, hand_model_path, soundfont_path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Asset:
    name: str
    path: Path
    urls: tuple[str, ...]
    min_bytes: int  # sanity check: a 404 HTML page is never this large
    sha256: str | None = None


def assets() -> list[Asset]:
    return [
        Asset(
            name="hand gesture recognizer model",
            path=hand_model_path(),
            urls=(
                "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task",
                "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
            ),
            min_bytes=5_000_000,
        ),
        Asset(
            name="GeneralUser GS soundfont",
            path=soundfont_path(),
            urls=(
                "https://github.com/mrbumpy409/GeneralUser-GS/raw/main/GeneralUser-GS.sf2",
                "https://raw.githubusercontent.com/mrbumpy409/GeneralUser-GS/main/GeneralUser-GS.sf2",
            ),
            min_bytes=20_000_000,
        ),
    ]


def _download(url: str, dest: Path, min_bytes: int) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "zerohero-setup"})
    with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
        total = resp.headers.get("Content-Length")
        total_i = int(total) if total else None
        done = 0
        while chunk := resp.read(1 << 16):
            out.write(chunk)
            done += len(chunk)
            if sys.stderr.isatty():
                pct = f"{done * 100 // total_i:3d}%" if total_i else f"{done >> 20} MiB"
                print(f"\r  {dest.name}: {pct}", end="", file=sys.stderr, flush=True)
    if sys.stderr.isatty():
        print(file=sys.stderr)
    if tmp.stat().st_size < min_bytes:
        tmp.unlink()
        raise OSError(f"{url}: file too small ({tmp.stat().st_size if tmp.exists() else 0} bytes), not a real asset")
    tmp.replace(dest)


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def present(asset: Asset) -> bool:
    return asset.path.exists() and asset.path.stat().st_size >= asset.min_bytes


def ensure_assets(force: bool = False) -> list[Asset]:
    """Download whatever is missing. Returns the assets that were fetched."""
    data_dir().mkdir(parents=True, exist_ok=True)
    fetched: list[Asset] = []
    for asset in assets():
        if present(asset) and not force:
            log.info("%s: present at %s", asset.name, asset.path)
            continue
        errors: list[str] = []
        for url in asset.urls:
            try:
                print(f"downloading {asset.name} from {url}", file=sys.stderr)
                _download(url, asset.path, asset.min_bytes)
                fetched.append(asset)
                break
            except OSError as e:  # urllib errors subclass OSError
                errors.append(f"{url}: {e}")
        else:
            raise RuntimeError(f"could not fetch {asset.name}:\n  " + "\n  ".join(errors))
    return fetched


def remove_assets() -> None:
    d = data_dir()
    if d.exists():
        shutil.rmtree(d)
