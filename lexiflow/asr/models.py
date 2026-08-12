"""Fetch and track the ggml Whisper weights the native backends expect."""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..config import DEFAULT_STATE_DIR

GGML_BASE_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
DOWNLOAD_CHUNK = 1 << 20


@dataclass(frozen=True)
class ModelSpec:
    """One downloadable ggml model and the trade-off it represents."""

    name: str
    filename: str
    megabytes: int
    note: str

    @property
    def url(self) -> str:
        return f"{GGML_BASE_URL}/{self.filename}"


CATALOGUE: Dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        ModelSpec("tiny.en", "ggml-tiny.en.bin", 75, "fastest, English only, noticeably lossy"),
        ModelSpec("tiny", "ggml-tiny.bin", 75, "fastest, multilingual"),
        ModelSpec("base.en", "ggml-base.en.bin", 142, "good default for live English on a laptop"),
        ModelSpec("base", "ggml-base.bin", 142, "good default, multilingual"),
        ModelSpec("small.en", "ggml-small.en.bin", 466, "clearly better English, needs ~4 cores"),
        ModelSpec("small", "ggml-small.bin", 466, "clearly better, multilingual"),
        ModelSpec("medium.en", "ggml-medium.en.bin", 1500, "strong English, borderline realtime"),
        ModelSpec("large-v3", "ggml-large-v3.bin", 3100, "best quality, not realtime on CPU"),
        ModelSpec(
            "large-v3-turbo",
            "ggml-large-v3-turbo.bin",
            1600,
            "large quality at roughly small speed",
        ),
    )
}


def models_directory() -> Path:
    return Path(os.environ.get("LEXIFLOW_MODELS", DEFAULT_STATE_DIR / "models"))


def local_path(name: str) -> Path:
    spec = CATALOGUE.get(name)
    filename = spec.filename if spec else name
    return models_directory() / filename


def is_installed(name: str) -> bool:
    path = local_path(name)
    return path.is_file() and path.stat().st_size > 0


def installed_models() -> List[Dict[str, object]]:
    directory = models_directory()
    if not directory.is_dir():
        return []
    rows = []
    for path in sorted(directory.glob("ggml-*.bin")):
        rows.append(
            {
                "filename": path.name,
                "path": str(path),
                "megabytes": round(path.stat().st_size / (1 << 20), 1),
            }
        )
    return rows


def resolve(name_or_path: Optional[str]) -> Optional[str]:
    """Accept a catalogue name, a bare filename or an explicit path."""
    if not name_or_path:
        return None
    candidate = Path(name_or_path)
    if candidate.is_file():
        return str(candidate)
    if is_installed(name_or_path):
        return str(local_path(name_or_path))
    return None


def sha256(path: Path, chunk: int = DOWNLOAD_CHUNK) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def download(
    name: str,
    destination: Optional[Path] = None,
    force: bool = False,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """Stream a ggml model to disk, resuming into a temp file then renaming."""
    spec = CATALOGUE.get(name)
    if spec is None:
        raise ValueError(f"unknown model '{name}', choose from {', '.join(sorted(CATALOGUE))}")

    target = Path(destination) if destination else local_path(name)
    if target.is_file() and target.stat().st_size > 0 and not force:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")

    request = urllib.request.Request(spec.url, headers={"User-Agent": "lexiflow"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, open(partial, "wb") as sink:
            total = int(response.headers.get("Content-Length") or 0)
            written = 0
            while True:
                block = response.read(DOWNLOAD_CHUNK)
                if not block:
                    break
                sink.write(block)
                written += len(block)
                if progress is not None:
                    progress(written, total)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"could not download {spec.filename}: {exc}") from exc

    shutil.move(str(partial), str(target))
    return target


def describe_catalogue() -> List[Dict[str, object]]:
    return [
        {
            "name": spec.name,
            "filename": spec.filename,
            "megabytes": spec.megabytes,
            "note": spec.note,
            "installed": is_installed(spec.name),
        }
        for spec in CATALOGUE.values()
    ]
