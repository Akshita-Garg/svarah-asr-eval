"""Hashing and atomic-write helpers.

Used by caching (cache keys include model-file and audio hashes) and by the run
manifest (local model/executable hashes). Atomic writes prevent a partial JSON
file from surviving an interruption (DESIGN: "Cache files are written atomically").
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

_CHUNK = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in chunks (models can be large)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_hash(text: str, length: int = 16) -> str:
    return sha256_text(text)[:length]


def write_json_atomic(path: Path, obj: Any, *, indent: int = 2) -> None:
    """Write JSON to ``path`` atomically.

    Writes to a temp file in the same directory (so ``os.replace`` is atomic on
    the same filesystem, including Windows) and then replaces the target. If the
    process dies mid-write, the original file is untouched and no partial file
    is left at the target path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=indent, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        # Clean up the temp file on any failure so we never leak partial files.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
