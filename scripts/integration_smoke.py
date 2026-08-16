"""Integration smoke test WITHOUT the gated dataset.

Stages a tiny subset manifest + prepared WAVs (copied from the desktop app's own
test_wavs) so the full orchestrator — cache, scoring, outputs, both local
backends — can be exercised end-to-end before the HF token is available.

Run:  uv run python scripts/integration_smoke.py
It cleans up the staged manifest/prepared files at the end (results/ is left for
inspection). This is a throwaway harness, not part of the pipeline.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# A loose script under scripts/ puts scripts/ on sys.path, not the repo root;
# add the repo root so `voicerefine_eval` imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voicerefine_eval import run as run_mod
from voicerefine_eval.audio import PREPARED_DIR, prepared_path_for
from voicerefine_eval.dataset import MANIFEST_PATH
from voicerefine_eval.hashing import write_json_atomic

# Sample WAVs ship with the desktop app's Whisper Tiny model directory. The
# default assumes the sibling checkout layout; override for any other location.
DEFAULT_TEST_WAVS = (
    Path(__file__).resolve().parents[2]
    / "voicerefine-desktop"
    / "resources"
    / "models"
    / "sherpa-onnx-whisper-tiny.en"
    / "test_wavs"
)
DESKTOP = Path(os.environ.get("VOICEREFINE_TEST_WAVS_DIR", DEFAULT_TEST_WAVS))

# References ~ ground truth for these LibriSpeech-style clips (used only to make
# the metrics exercise real numbers in the smoke test).
STAGE = [
    ("svarah_test_0000", "0.wav",
     "After early nightfall, the yellow lamps would light up here and there "
     "the squalid quarter of the brothels."),
    ("svarah_test_0001", "1.wav",
     "God, as a direct consequence of the sin which man thus punished, had "
     "given her a lovely child, whose place was on that same dishonored bosom "
     "to connect her parent forever with the race and descent of mortals, and "
     "to be finally a blessed soul in heaven."),
]


def stage() -> None:
    if not DESKTOP.is_dir():
        sys.exit(
            f"Sample WAV directory not found: {DESKTOP}\n"
            "Set VOICEREFINE_TEST_WAVS_DIR to a directory containing 0.wav and 1.wav."
        )
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for seq, (eval_id, wav, ref) in enumerate(STAGE):
        shutil.copyfile(DESKTOP / wav, prepared_path_for(eval_id))
        entries.append({"eval_id": eval_id, "row_index": seq, "reference": ref,
                        "source_id": wav, "duration_seconds": None})
    manifest = {
        "dataset": "SMOKE/staged", "split": "test", "revision": "staged",
        "seed": 42, "subset_size": len(entries), "total_rows": len(entries),
        "schema": {"audio_field": "audio", "transcript_field": "text",
                   "id_field": "source_id", "duration_field": None,
                   "column_names": ["audio", "text"]},
        "entries": entries,
    }
    write_json_atomic(MANIFEST_PATH, manifest)


def cleanup() -> None:
    MANIFEST_PATH.unlink(missing_ok=True)
    for eval_id, _, _ in STAGE:
        prepared_path_for(eval_id).unlink(missing_ok=True)


if __name__ == "__main__":
    stage()
    try:
        print("\n########## RUN 1 (fresh) ##########")
        run_mod.main(["--debug", "--backends",
                      "voicerefine_whisper_tiny_int8,voicerefine_parakeet_q4"])
        print("\n########## RUN 2 (should be cache hits) ##########")
        run_mod.main(["--debug", "--backends",
                      "voicerefine_whisper_tiny_int8,voicerefine_parakeet_q4"])
    finally:
        cleanup()
        print("\n[cleanup] removed staged manifest + prepared WAVs")
