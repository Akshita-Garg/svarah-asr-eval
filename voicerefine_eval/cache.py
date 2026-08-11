"""Transcript cache keyed by backend config + model + utterance + audio.

DESIGN.md "Caching": successful RAW transcriptions are cached before
normalization and scoring. The cache key includes the backend id + versioned
config, the model-file hash (local backends), the utterance id, and the
prepared-audio hash — so changing a model, config, or audio file automatically
invalidates the old entry. Files are written atomically. Failures are never
cached as successful results. ``--no-cache`` forces fresh transcription.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import REPO_ROOT
from .hashing import short_hash, write_json_atomic

CACHE_DIR = REPO_ROOT / "cache"


def _canonical(obj: Any) -> str:
    """Stable JSON string for hashing (sorted keys, no whitespace noise)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class CacheKey:
    backend_id: str
    key_hash: str


class TranscriptCache:
    def __init__(self, *, enabled: bool = True, root: Path = CACHE_DIR):
        self.enabled = enabled
        self.root = Path(root)

    def _path(self, key: CacheKey) -> Path:
        return self.root / key.backend_id / f"{key.key_hash}.json"

    def make_key(
        self,
        *,
        signature: dict[str, Any],
        eval_id: str,
        audio_sha256: str,
    ) -> CacheKey:
        material = _canonical(
            {"signature": signature, "eval_id": eval_id, "audio": audio_sha256}
        )
        return CacheKey(backend_id=signature["backend_id"], key_hash=short_hash(material, 32))

    def get(self, key: CacheKey) -> str | None:
        if not self.enabled:
            return None
        p = self._path(key)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data["text"]
        except (json.JSONDecodeError, KeyError, OSError):
            # A corrupt/partial cache entry is treated as a miss, not a crash.
            return None

    def put(
        self,
        key: CacheKey,
        *,
        text: str,
        eval_id: str,
        audio_sha256: str,
        signature: dict[str, Any],
    ) -> None:
        if not self.enabled:
            return
        record = {
            "backend_id": key.backend_id,
            "eval_id": eval_id,
            "text": text,
            "audio_sha256": audio_sha256,
            "signature": signature,
            "created_at": time.time(),
        }
        write_json_atomic(self._path(key), record)
