"""ElevenLabs Scribe v2 batch Speech-to-Text (the cloud reference backend).

DESIGN.md: English, no diarization or audio-event tags. Timing measured by the
runner is API end-to-end latency (upload + network + service + download), not
server compute time.

Retry policy (DESIGN "Failure And Comparison Rules"): HTTP 429 and 5xx are
retried with bounded exponential backoff + jitter; other 4xx are not retried.

The API key is read from ``ELEVENLABS_API_KEY`` (env/.env). It is never written
to any artifact.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any

import requests

from ..config import BackendConfig
from .base import ASRBackend, BackendUnavailableError, TranscriptionError


class ElevenLabsBackend(ASRBackend):
    def __init__(self, cfg: BackendConfig):
        super().__init__(cfg)
        self.model_id = self.settings.get("model_id", "scribe_v2")
        self.language_code = self.settings.get("language_code", "eng")
        self.diarize = bool(self.settings.get("diarize", False))
        self.tag_audio_events = bool(self.settings.get("tag_audio_events", False))
        self.base_url = self.settings.get(
            "base_url", "https://api.elevenlabs.io/v1/speech-to-text"
        )
        self.max_retries = int(self.settings.get("max_retries", 5))
        self.backoff_base = float(self.settings.get("backoff_base_seconds", 1.0))
        self.backoff_max = float(self.settings.get("backoff_max_seconds", 30.0))
        self.request_timeout = float(self.settings.get("request_timeout_seconds", 300))
        self._api_key: str | None = None

    def is_available(self) -> bool:
        return bool(os.environ.get("ELEVENLABS_API_KEY"))

    def start(self) -> None:
        key = os.environ.get("ELEVENLABS_API_KEY")
        if not key:
            raise BackendUnavailableError(
                "ELEVENLABS_API_KEY is not set (put it in .env). Skipping cloud backend."
            )
        self._api_key = key

    def _backoff_seconds(self, attempt: int) -> float:
        # Exponential with full jitter, capped. attempt starts at 1.
        raw = min(self.backoff_max, self.backoff_base * (2 ** (attempt - 1)))
        return random.uniform(0, raw)

    def transcribe(self, audio_path: Path) -> str:
        if self._api_key is None:
            raise TranscriptionError("Backend not started", category="not_started")

        wav_bytes = Path(audio_path).read_bytes()
        headers = {"xi-api-key": self._api_key}
        data = {
            "model_id": self.model_id,
            "language_code": self.language_code,
            # Booleans as lowercase strings for multipart form fields.
            "diarize": str(self.diarize).lower(),
            "tag_audio_events": str(self.tag_audio_events).lower(),
        }

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    self.base_url,
                    headers=headers,
                    files={"file": ("recording.wav", wav_bytes, "audio/wav")},
                    data=data,
                    timeout=self.request_timeout,
                )
            except requests.RequestException as e:
                # Network errors are transient: retry within the budget.
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(self._backoff_seconds(attempt))
                    continue
                raise TranscriptionError(
                    str(e), category="network_error", attempts=attempt
                ) from e

            if resp.ok:
                try:
                    return (resp.json().get("text") or "").strip()
                except ValueError as e:
                    raise TranscriptionError(
                        f"Non-JSON success body: {resp.text[:200]}",
                        category="bad_response",
                        attempts=attempt,
                    ) from e

            # Retry only on 429 and 5xx; other 4xx are terminal.
            retriable = resp.status_code == 429 or 500 <= resp.status_code < 600
            last_error = TranscriptionError(
                f"ElevenLabs {resp.status_code}: {resp.text[:200]}",
                category=f"http_{resp.status_code}",
                attempts=attempt,
            )
            if retriable and attempt < self.max_retries:
                # Honor Retry-After when the server provides it.
                retry_after = resp.headers.get("Retry-After")
                delay = float(retry_after) if (retry_after or "").isdigit() else self._backoff_seconds(attempt)
                time.sleep(delay)
                continue
            raise last_error

        # Exhausted retries.
        if isinstance(last_error, TranscriptionError):
            raise last_error
        raise TranscriptionError(
            str(last_error) if last_error else "ElevenLabs transcription failed",
            category="exhausted_retries",
            attempts=self.max_retries,
        )

    def cache_signature(self) -> dict[str, Any]:
        return {
            "backend_id": self.name,
            "type": "elevenlabs",
            "model_id": self.model_id,
            "language_code": self.language_code,
            "diarize": self.diarize,
            "tag_audio_events": self.tag_audio_events,
        }
