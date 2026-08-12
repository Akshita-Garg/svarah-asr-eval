"""The backend contract shared by every ASR system under test.

DESIGN.md "Backend Contract":

    class ASRBackend:
        name: str
        def start(self) -> None: ...
        def transcribe(self, audio_path: Path) -> str: ...
        def close(self) -> None: ...

``start()`` loads a model / starts a server / validates API config.
``transcribe()`` returns raw transcript text for one prepared WAV.
``close()`` releases owned resources even when an error occurs.

Beyond the contract we add:
- ``is_available()`` for graceful degradation (skip, don't crash the run).
- ``cache_signature()`` — the versioned identity that goes into the cache key so
  that changing a model, binary, or config invalidates cached transcripts.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any

from ..config import BackendConfig


class BackendUnavailableError(RuntimeError):
    """Raised by ``start()`` when a required model/binary/key is missing."""


class TranscriptionError(RuntimeError):
    """Raised by ``transcribe()`` for a per-utterance failure.

    Carries a coarse ``category`` used in failure reporting (DESIGN: "a failure
    records the backend, utterance, exception category, message, attempt count").
    """

    def __init__(self, message: str, *, category: str = "transcription_error", attempts: int = 1):
        super().__init__(message)
        self.category = category
        self.attempts = attempts


class ASRBackend(abc.ABC):
    def __init__(self, cfg: BackendConfig):
        self.cfg = cfg
        self.name = cfg.backend_id
        self.settings = cfg.settings
        self.last_attempts = 1

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend can run on this machine right now."""

    @abc.abstractmethod
    def start(self) -> None:
        ...

    @abc.abstractmethod
    def transcribe(self, audio_path: Path) -> str:
        ...

    def prepare_request(self) -> None:
        """Perform untimed benchmark coordination immediately before a call.

        Normal backends do nothing. A rate-limited cloud backend may sleep here
        to keep a sequential batch within its workspace limit without counting
        artificial benchmark pacing as user-facing request latency.
        """
        return None

    def close(self) -> None:  # default: nothing to release
        return None

    @abc.abstractmethod
    def cache_signature(self) -> dict[str, Any]:
        """Versioned identity of this backend for cache-key construction.

        Must include everything whose change should invalidate a cached
        transcript: backend id, backend type, relevant runtime settings, and a
        hash of the local model/binary when one is used.
        """

    # Context-manager sugar so runners can guarantee close() on error.
    def __enter__(self) -> "ASRBackend":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
