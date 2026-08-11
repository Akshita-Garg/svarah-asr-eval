"""Shared result records for one (backend, utterance) attempt."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UtteranceResult:
    backend_id: str
    eval_id: str
    status: str  # "success" | "failure"
    from_cache: bool = False

    # timing / speed (present on success)
    inference_seconds: float | None = None
    audio_seconds: float | None = None
    rtf: float | None = None

    # text (raw always retained; normalized derived) — DESIGN: never replace source
    raw_reference: str = ""
    raw_hypothesis: str = ""
    normalized_reference: str = ""
    normalized_hypothesis: str = ""

    # edit counts / WER (present on success)
    hits: int | None = None
    substitutions: int | None = None
    deletions: int | None = None
    insertions: int | None = None
    reference_words: int | None = None
    wer: float | None = None

    # failure detail
    error_category: str | None = None
    error_message: str | None = None
    attempts: int = 1

    @property
    def ok(self) -> bool:
        return self.status == "success"
