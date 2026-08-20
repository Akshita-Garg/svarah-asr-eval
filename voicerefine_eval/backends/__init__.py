"""ASR backend registry and construction.

A backend that is unavailable (missing model, binary, or key) is constructed but
reports ``is_available() == False`` and is skipped by the runner — one backend's
missing dependency must not block the others (DESIGN: graceful degradation).
"""

from __future__ import annotations

from ..config import BackendConfig
from .base import ASRBackend


def build_backend(cfg: BackendConfig) -> ASRBackend:
    if cfg.type == "whisper_sherpa":
        from .whisper_tiny import WhisperTinyBackend

        return WhisperTinyBackend(cfg)
    if cfg.type == "crispasr_server":
        from .parakeet_crispasr import CrispAsrServerBackend

        return CrispAsrServerBackend(cfg)
    if cfg.type == "elevenlabs":
        from .elevenlabs import ElevenLabsBackend

        return ElevenLabsBackend(cfg)
    if cfg.type == "sarvam":
        from .sarvam import SarvamBackend

        return SarvamBackend(cfg)
    if cfg.type == "smallest":
        from .smallest import SmallestBackend

        return SmallestBackend(cfg)
    if cfg.type == "deepgram":
        from .deepgram import DeepgramBackend

        return DeepgramBackend(cfg)
    if cfg.type == "gnani":
        from .gnani import GnaniBackend

        return GnaniBackend(cfg)
    raise ValueError(f"Unknown backend type: {cfg.type!r} for {cfg.backend_id}")


__all__ = ["ASRBackend", "build_backend"]
