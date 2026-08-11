"""Whisper Tiny English INT8 via sherpa-onnx (VoiceRefine's lightweight local option).

Mirrors ``createWhisperTinyEnglishConfig`` in
``voicerefine-desktop/src/main/asr.js``: 16 kHz feature input, int8 encoder/decoder,
``tiny.en-tokens.txt``, language ``en``, task ``transcribe``, ``tail_paddings=-1``,
4 threads, CPU provider. The Python ``sherpa-onnx`` package is the same engine as
the desktop's ``sherpa-onnx-node`` binding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..audio import load_wav_mono_f32
from ..config import BackendConfig
from ..hashing import sha256_file
from .base import ASRBackend, BackendUnavailableError, TranscriptionError


class WhisperTinyBackend(ASRBackend):
    def __init__(self, cfg: BackendConfig):
        super().__init__(cfg)
        self.model_dir = Path(self.settings["model_dir"])
        self.precision = self.settings.get("precision", "int8")
        self.num_threads = int(self.settings.get("num_threads", 4))
        self._recognizer = None
        self._model_hashes: dict[str, str] | None = None

    # --- file resolution -----------------------------------------------------
    def _suffix(self) -> str:
        return ".int8" if self.precision != "fp32" else ""

    def _encoder(self) -> Path:
        return self.model_dir / f"tiny.en-encoder{self._suffix()}.onnx"

    def _decoder(self) -> Path:
        return self.model_dir / f"tiny.en-decoder{self._suffix()}.onnx"

    def _tokens(self) -> Path:
        return self.model_dir / "tiny.en-tokens.txt"

    def _required_files(self) -> list[Path]:
        return [self._encoder(), self._decoder(), self._tokens()]

    # --- contract ------------------------------------------------------------
    def is_available(self) -> bool:
        try:
            import sherpa_onnx  # noqa: F401
        except ImportError:
            return False
        return all(p.exists() for p in self._required_files())

    def start(self) -> None:
        try:
            import sherpa_onnx
        except ImportError as e:
            raise BackendUnavailableError(f"sherpa-onnx not installed: {e}") from e

        for p in self._required_files():
            if not p.exists():
                raise BackendUnavailableError(f"Whisper model file missing: {p}")

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(self._encoder()),
            decoder=str(self._decoder()),
            tokens=str(self._tokens()),
            language=self.settings.get("language", "en"),
            task=self.settings.get("task", "transcribe"),
            num_threads=self.num_threads,
            decoding_method=self.settings.get("decoding_method", "greedy_search"),
            provider=self.settings.get("provider", "cpu"),
            tail_paddings=int(self.settings.get("tail_paddings", -1)),
        )

    def transcribe(self, audio_path: Path) -> str:
        if self._recognizer is None:
            raise TranscriptionError("Backend not started", category="not_started")
        try:
            samples, sr = load_wav_mono_f32(Path(audio_path))
            stream = self._recognizer.create_stream()
            stream.accept_waveform(sr, samples)
            self._recognizer.decode_stream(stream)
            return (stream.result.text or "").strip()
        except Exception as e:  # noqa: BLE001 - normalize to a categorized error
            raise TranscriptionError(str(e), category=type(e).__name__) from e

    def close(self) -> None:
        self._recognizer = None

    def _model_hash_map(self) -> dict[str, str]:
        if self._model_hashes is None:
            self._model_hashes = {
                p.name: sha256_file(p) for p in self._required_files() if p.exists()
            }
        return self._model_hashes

    def cache_signature(self) -> dict[str, Any]:
        return {
            "backend_id": self.name,
            "type": "whisper_sherpa",
            "precision": self.precision,
            "num_threads": self.num_threads,
            "language": self.settings.get("language", "en"),
            "task": self.settings.get("task", "transcribe"),
            "decoding_method": self.settings.get("decoding_method", "greedy_search"),
            "tail_paddings": int(self.settings.get("tail_paddings", -1)),
            "provider": self.settings.get("provider", "cpu"),
            "model_hashes": self._model_hash_map(),
        }
