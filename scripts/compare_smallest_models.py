"""Compare Smallest.ai Pulse and Pulse Pro on one prepared Svarah recording.

Edit only ``AUDIO_PATH`` below, then run:

    uv run python scripts/compare_smallest_models.py

The script uses the same Smallest.ai backend implementation and configuration
as the evaluation harness. It sends the exact same WAV bytes to both models and
prints their raw API transcripts without normalization or post-processing.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from voicerefine_eval.backends.smallest import SmallestBackend
from voicerefine_eval.config import load_config


# EDIT THIS LINE ONLY to choose another prepared Svarah recording.
AUDIO_PATH = REPO_ROOT / "data" / "prepared" / "svarah_test_0048.wav"

BACKEND_IDS = ("smallest_pulse", "smallest_pulse_pro")
SUBSET_MANIFEST = REPO_ROOT / "data" / "subset_manifest.json"


def contains_devanagari(text: str) -> bool:
    return any("\u0900" <= character <= "\u097f" for character in text)


def reference_for(audio_path: Path) -> str | None:
    if not SUBSET_MANIFEST.exists():
        return None

    manifest = json.loads(SUBSET_MANIFEST.read_text(encoding="utf-8"))
    for entry in manifest.get("entries", []):
        if entry.get("eval_id") == audio_path.stem:
            return entry.get("reference")
    return None


def transcribe(backend: SmallestBackend, audio_path: Path) -> tuple[str, float]:
    backend.start()
    try:
        backend.prepare_request()
        started_at = time.perf_counter()
        text = backend.transcribe(audio_path)
        return text, time.perf_counter() - started_at
    finally:
        backend.close()


def main() -> None:
    audio_path = AUDIO_PATH.resolve()
    if not audio_path.is_file():
        raise SystemExit(
            f"Audio file not found: {audio_path}\n"
            "Edit AUDIO_PATH near the top of this script to select a prepared WAV."
        )

    config = load_config()
    missing_backends = [backend_id for backend_id in BACKEND_IDS if backend_id not in config.backends]
    if missing_backends:
        raise SystemExit(f"Missing backend configuration: {', '.join(missing_backends)}")

    audio_bytes = audio_path.read_bytes()
    reference = reference_for(audio_path)

    print("=" * 72)
    print("Smallest.ai one-recording comparison")
    print("=" * 72)
    print(f"Audio:     {audio_path}")
    print(f"Bytes:     {len(audio_bytes):,}")
    print(f"SHA-256:   {hashlib.sha256(audio_bytes).hexdigest()}")
    print(f"Reference: {reference or '[not found in subset manifest]'}")

    for backend_id in BACKEND_IDS:
        backend = SmallestBackend(config.backends[backend_id])
        print(f"\n{'-' * 72}")
        print(f"{backend_id} (API model: {backend.model})")
        print("-" * 72)
        try:
            text, elapsed_seconds = transcribe(backend, audio_path)
        except Exception as error:
            print(f"ERROR: {type(error).__name__}: {error}")
            continue

        print(f"Latency:             {elapsed_seconds:.3f} seconds")
        print(f"Contains Devanagari: {'YES' if contains_devanagari(text) else 'no'}")
        print("Raw transcript:")
        print(text)


if __name__ == "__main__":
    main()
