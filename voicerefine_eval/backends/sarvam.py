"""Sarvam Saaras batch-style REST transcription for Indian English."""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any

import requests

from ..config import BackendConfig
from .base import ASRBackend, BackendUnavailableError, TranscriptionError


class SarvamBackend(ASRBackend):
    def __init__(self, cfg: BackendConfig):
        super().__init__(cfg)
        self.model = self.settings.get("model", "saaras:v4")
        self.language_code = self.settings.get("language_code", "en-IN")
        self.base_url = self.settings.get(
            "base_url", "https://api.sarvam.ai/speech-to-text"
        )
        self.max_retries = int(self.settings.get("max_retries", 5))
        self.backoff_base = float(self.settings.get("backoff_base_seconds", 1.0))
        self.backoff_max = float(self.settings.get("backoff_max_seconds", 30.0))
        self.request_timeout = float(self.settings.get("request_timeout_seconds", 300))
        self._api_key: str | None = None

    def is_available(self) -> bool:
        return bool(os.environ.get("SARVAM_API_KEY"))

    def start(self) -> None:
        key = os.environ.get("SARVAM_API_KEY")
        if not key:
            raise BackendUnavailableError(
                "SARVAM_API_KEY is not set (put it in .env). Skipping cloud backend."
            )
        self._api_key = key

    def _backoff_seconds(self, attempt: int) -> float:
        maximum = min(self.backoff_max, self.backoff_base * (2 ** (attempt - 1)))
        return random.uniform(0, maximum)

    def transcribe(self, audio_path: Path) -> str:
        if self._api_key is None:
            raise TranscriptionError("Backend not started", category="not_started")

        wav_bytes = Path(audio_path).read_bytes()
        headers = {"api-subscription-key": self._api_key}
        data = {
            "model": self.model,
            "language_code": self.language_code,
            "with_timestamps": "false",
        }
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    files={"file": ("recording.wav", wav_bytes, "audio/wav")},
                    data=data,
                    timeout=self.request_timeout,
                )
            except requests.RequestException as error:
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(self._backoff_seconds(attempt))
                    continue
                raise TranscriptionError(
                    str(error), category="network_error", attempts=attempt
                ) from error

            if response.ok:
                try:
                    transcript = response.json().get("transcript")
                except ValueError as error:
                    raise TranscriptionError(
                        f"Non-JSON success body: {response.text[:200]}",
                        category="bad_response",
                        attempts=attempt,
                    ) from error
                if not isinstance(transcript, str):
                    raise TranscriptionError(
                        f"Success response is missing transcript: {response.text[:200]}",
                        category="bad_response",
                        attempts=attempt,
                    )
                return transcript.strip()

            retriable = response.status_code == 429 or 500 <= response.status_code < 600
            last_error = TranscriptionError(
                f"Sarvam {response.status_code}: {response.text[:200]}",
                category=f"http_{response.status_code}",
                attempts=attempt,
            )
            if retriable and attempt < self.max_retries:
                retry_after = response.headers.get("Retry-After")
                delay = (
                    float(retry_after)
                    if retry_after and retry_after.replace(".", "", 1).isdigit()
                    else self._backoff_seconds(attempt)
                )
                time.sleep(delay)
                continue
            raise last_error

        if isinstance(last_error, TranscriptionError):
            raise last_error
        raise TranscriptionError(
            str(last_error) if last_error else "Sarvam transcription failed",
            category="exhausted_retries",
            attempts=self.max_retries,
        )

    def cache_signature(self) -> dict[str, Any]:
        return {
            "backend_id": self.name,
            "type": "sarvam",
            "model": self.model,
            "language_code": self.language_code,
            "base_url": self.base_url,
            "with_timestamps": False,
        }
