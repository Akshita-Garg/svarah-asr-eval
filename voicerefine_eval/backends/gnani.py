"""Gnani Prisma v2.5 speech-to-text (Vachana REST endpoint).

The endpoint takes no model-selection parameter: the served model follows from
the API key, which is issued against a specific model. ``model_label`` is
recorded for provenance and is never sent as a request parameter.
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

# The documented REST contract caps pre-recorded audio at 60 seconds.
MAX_AUDIO_SECONDS = 60


class GnaniBackend(ASRBackend):
    def __init__(self, cfg: BackendConfig):
        super().__init__(cfg)
        self.language_code = self.settings.get("language_code", "en-IN")
        # `verbatim` returns speech as spoken; `transcribe` applies Gnani's own
        # formatting. Verbatim keeps the Whisper normalizer as the only
        # formatting pass, matching every other backend in this evaluation.
        self.output_format = self.settings.get("output_format", "verbatim")
        self.base_url = self.settings.get("base_url", "https://api.vachana.ai/stt/v3")
        self.model_label = self.settings.get("model_label", "prisma-v2.5")
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
        return bool(os.environ.get("GNANI_API_KEY"))

    def start(self) -> None:
        key = os.environ.get("GNANI_API_KEY")
        if not key:
            raise BackendUnavailableError(
                "GNANI_API_KEY is not set (put it in .env). Skipping cloud backend."
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

        wav_bytes = Path(audio_path).read_bytes()
        headers = {"X-API-Key-ID": self._api_key}
        data = {
            "language_code": self.language_code,
            "format": self.output_format,
        }
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            self.last_attempts = attempt
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    files={"audio_file": ("recording.wav", wav_bytes, "audio/wav")},
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
                    body = response.json()
                except ValueError as error:
                    raise TranscriptionError(
                        f"Non-JSON success body: {response.text[:200]}",
                        category="bad_response",
                        attempts=attempt,
                    ) from error
                transcript = body.get("transcript")
                if not isinstance(transcript, str):
                    raise TranscriptionError(
                        f"Success response is missing transcript: {response.text[:200]}",
                        category="bad_response",
                        attempts=attempt,
                    )
                return transcript.strip()

            retriable = response.status_code == 429 or 500 <= response.status_code < 600
            last_error = TranscriptionError(
                f"Gnani {response.status_code}: {response.text[:200]}",
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
            str(last_error) if last_error else "Gnani transcription failed",
            category="exhausted_retries",
            attempts=self.max_retries,
        )

    def cache_signature(self) -> dict[str, Any]:
        return {
            "backend_id": self.name,
            "type": "gnani",
            # Not a request parameter: the model follows from the API key.
            # Recorded so a change of stated model invalidates cached transcripts.
            "model_label": self.model_label,
            "language_code": self.language_code,
            "format": self.output_format,
            "base_url": self.base_url,
            "input": "multipart_audio_file",
        }
