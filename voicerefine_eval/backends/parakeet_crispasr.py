"""Local ASR models through one persistent CrispASR server protocol.

Mirrors ``startCrispAsrServer`` / ``postToCrispServer`` / ``waitForPort`` /
``stopCrispAsrServer`` in ``voicerefine-desktop/src/main/asr.js``:

- Launch once: ``crispasr.exe --server --backend parakeet --model <gguf>
  --language en --threads 8 --host 127.0.0.1 --port 51234 --no-prints
  --no-timestamps``.
- Readiness: poll a TCP connect to the port (not an HTTP probe) until it accepts.
- Transcribe: multipart POST to ``/v1/audio/transcriptions`` (fallback
  ``/inference``) with fields ``file`` (WAV) and ``language=en``.
- Response: JSON ``{text|transcription|result}`` else cleaned plaintext.
- Shutdown: terminate the child (escalate to kill) so a later run doesn't race
  the dying process for the port.
"""

from __future__ import annotations

import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import requests

from ..config import BackendConfig
from ..hashing import sha256_file
from .base import ASRBackend, BackendUnavailableError, TranscriptionError


class CrispAsrServerBackend(ASRBackend):
    def __init__(self, cfg: BackendConfig):
        super().__init__(cfg)
        self.bin_path = Path(self.settings["bin"])
        self.model_path = Path(self.settings["model"])
        self.backend = self.settings.get("backend", "parakeet")
        self.language = self.settings.get("language", "en")
        self.threads = int(self.settings.get("threads", 8))
        self.gpu_backend = self.settings.get("gpu_backend", "cpu")
        self.quantization = self.settings.get("quantization")
        self.no_punctuation = bool(self.settings.get("no_punctuation", True))
        self.host = self.settings.get("host", "127.0.0.1")
        self.port = int(self.settings.get("port", 51234))
        self.start_timeout_ms = int(self.settings.get("start_timeout_ms", 15000))
        self.request_timeout_ms = int(self.settings.get("request_timeout_ms", 180000))
        self._proc: subprocess.Popen | None = None
        self._hashes: dict[str, str] | None = None

    def is_available(self) -> bool:
        return self.bin_path.exists() and self.model_path.exists()

    # --- server lifecycle ----------------------------------------------------
    def start(self) -> None:
        if not self.bin_path.exists():
            raise BackendUnavailableError(f"CrispASR binary missing: {self.bin_path}")
        if not self.model_path.exists():
            raise BackendUnavailableError(f"CrispASR model missing: {self.model_path}")

        args = [
            str(self.bin_path),
            "--server",
            "--backend", self.backend,
            "--model", str(self.model_path),
            "--language", self.language,
            "--threads", str(self.threads),
            "--gpu-backend", self.gpu_backend,
            "--host", self.host,
            "--port", str(self.port),
            "--no-prints",
            "--no-timestamps",
        ]
        if self.no_punctuation:
            args.append("--no-punctuation")
        # windowsHide equivalent: no new console window on Windows.
        creationflags = 0
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            creationflags = subprocess.CREATE_NO_WINDOW

        self._proc = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        self._wait_for_port(self.start_timeout_ms / 1000.0)

    def _wait_for_port(self, timeout_s: float) -> None:
        started = time.perf_counter()
        while True:
            # If the process died while we were waiting, fail fast with its code.
            if self._proc is not None and self._proc.poll() is not None:
                raise BackendUnavailableError(
                    f"CrispASR server exited before ready: code={self._proc.returncode}"
                )
            try:
                with socket.create_connection((self.host, self.port), timeout=1.0):
                    return
            except OSError:
                if time.perf_counter() - started > timeout_s:
                    raise BackendUnavailableError(
                        f"CrispASR server did not become ready on {self.host}:{self.port} "
                        f"within {timeout_s:.0f}s"
                    )
                time.sleep(0.25)

    # --- transcription -------------------------------------------------------
    def transcribe(self, audio_path: Path) -> str:
        if self._proc is None:
            raise TranscriptionError("Server not started", category="not_started")

        wav_bytes = Path(audio_path).read_bytes()
        endpoints = ["/v1/audio/transcriptions", "/inference"]
        timeout_s = self.request_timeout_ms / 1000.0
        last_error: Exception | None = None

        for endpoint in endpoints:
            url = f"http://{self.host}:{self.port}{endpoint}"
            try:
                resp = requests.post(
                    url,
                    files={"file": ("recording.wav", wav_bytes, "audio/wav")},
                    data={"language": self.language},
                    timeout=timeout_s,
                )
            except requests.RequestException as e:
                last_error = e
                continue

            if not resp.ok:
                last_error = TranscriptionError(
                    f"CrispASR {endpoint} failed: {resp.status_code} {resp.text[:200]}",
                    category=f"http_{resp.status_code}",
                )
                continue

            return _parse_response(resp.text, resp.headers.get("content-type", ""))

        if isinstance(last_error, TranscriptionError):
            raise last_error
        raise TranscriptionError(
            str(last_error) if last_error else "CrispASR transcription failed",
            category=type(last_error).__name__ if last_error else "transcription_error",
        )

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                pass

    def _hash_map(self) -> dict[str, str]:
        if self._hashes is None:
            self._hashes = {}
            if self.model_path.exists():
                self._hashes["model"] = sha256_file(self.model_path)
            if self.bin_path.exists():
                self._hashes["bin"] = sha256_file(self.bin_path)
        return self._hashes

    def cache_signature(self) -> dict[str, Any]:
        return {
            "backend_id": self.name,
            "type": "crispasr_server",
            "runtime": "persistent_server",
            "backend": self.backend,
            "language": self.language,
            "threads": self.threads,
            "gpu_backend": self.gpu_backend,
            "quantization": self.quantization,
            "no_punctuation": self.no_punctuation,
            "hashes": self._hash_map(),
        }


# Compatibility for imports written when this adapter only supported Parakeet.
ParakeetCrispAsrBackend = CrispAsrServerBackend


def _parse_response(body: str, content_type: str) -> str:
    import json

    text = body.strip()
    if not text:
        return ""
    if "application/json" in content_type or text.startswith("{"):
        try:
            data = json.loads(text)
            parsed = data.get("text") or data.get("transcription") or data.get("result")
            # Don't fall back to the raw JSON body: an error object must not be
            # pasted as if it were the transcript.
            return parsed.strip() if isinstance(parsed, str) else ""
        except json.JSONDecodeError:
            return text
    # Plaintext: collapse to trimmed non-empty lines.
    return "\n".join(line.strip() for line in text.splitlines() if line.strip()).strip()
