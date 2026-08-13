"""Smallest.ai Pulse-family pre-recorded English transcription."""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any

import requests

from ..config import BackendConfig
from .base import ASRBackend, BackendUnavailableError, TranscriptionError


def _api_key() -> str | None:
    # Prefer the provider's documented spelling while accepting the name
    # already used in this project's local .env file.
    return os.environ.get("SMALLEST_API_KEY") or os.environ.get("SMALLESTAI_API_KEY")


class SmallestBackend(ASRBackend):
    def __init__(self, cfg: BackendConfig):
        super().__init__(cfg)
        self.model = self.settings.get("model", "pulse-pro")
        self.language = self.settings.get("language", "en")
        self.base_url = self.settings.get(
            "base_url", "https://api.smallest.ai/waves/v1/stt/"
        )
        self.min_request_interval = float(
            self.settings.get("min_request_interval_seconds", 0)
        )
        self.max_retries = int(self.settings.get("max_retries", 5))
        self.backoff_base = float(self.settings.get("backoff_base_seconds", 1.0))
        self.backoff_max = float(self.settings.get("backoff_max_seconds", 30.0))
        self.request_timeout = float(self.settings.get("request_timeout_seconds", 300))
        self._api_key: str | None = None
        self._last_request_started: float | None = None

    def is_available(self) -> bool:
        return bool(_api_key())

    def start(self) -> None:
        key = _api_key()
        if not key:
            raise BackendUnavailableError(
                "SMALLEST_API_KEY is not set (put it in .env). Skipping cloud backend."
            )
        self._api_key = key

    def _backoff_seconds(self, attempt: int) -> float:
        maximum = min(self.backoff_max, self.backoff_base * (2 ** (attempt - 1)))
        return random.uniform(0, maximum)

    def prepare_request(self) -> None:
        if self.min_request_interval <= 0:
            return
        now = time.monotonic()
        if self._last_request_started is not None:
            remaining = self.min_request_interval - (now - self._last_request_started)
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_started = time.monotonic()

    def transcribe(self, audio_path: Path) -> str:
        if self._api_key is None:
            raise TranscriptionError("Backend not started", category="not_started")

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/octet-stream",
        }
        params = {"model": self.model, "language": self.language}
        wav_bytes = Path(audio_path).read_bytes()
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            self.last_attempts = attempt
            try:
                response = requests.post(
                    self.base_url,
                    params=params,
                    headers=headers,
                    data=wav_bytes,
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
                    transcript = response.json().get("transcription")
                except ValueError as error:
                    raise TranscriptionError(
                        f"Non-JSON success body: {response.text[:200]}",
                        category="bad_response",
                        attempts=attempt,
                    ) from error
                if not isinstance(transcript, str):
                    raise TranscriptionError(
                        f"Success response is missing transcription: {response.text[:200]}",
                        category="bad_response",
                        attempts=attempt,
                    )
                return transcript.strip()

            retriable = response.status_code == 429 or 500 <= response.status_code < 600
            last_error = TranscriptionError(
                f"Smallest.ai {response.status_code}: {response.text[:200]}",
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
            str(last_error) if last_error else "Smallest.ai transcription failed",
            category="exhausted_retries",
            attempts=self.max_retries,
        )

    def cache_signature(self) -> dict[str, Any]:
        return {
            "backend_id": self.name,
            "type": "smallest",
            "model": self.model,
            "language": self.language,
            "base_url": self.base_url,
            "input": "raw_wav_bytes",
            "min_request_interval_seconds": self.min_request_interval,
        }
