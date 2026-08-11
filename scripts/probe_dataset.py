"""One-off probe: inspect the live Svarah dataset schema and decode one example.

DESIGN.md Acceptance Criteria #1: "The dataset schema and one decoded example are
inspected before backend work." This script satisfies that gate. It is a probe,
not part of the pipeline — it prints; it does not write artifacts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make the package importable and load .env so HF_TOKEN reaches the HF client.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from voicerefine_eval.config import load_dotenv  # noqa: E402

load_dotenv()

from datasets import load_dataset  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402

DATASET = "ai4bharat/Svarah"
SPLIT = "test"


def main() -> None:
    # Resolve the current revision hash of the dataset repo so we can pin it.
    api = HfApi()
    info = api.dataset_info(DATASET)
    print("=== REVISION ===")
    print("sha:", info.sha)
    print("last_modified:", info.lastModified)

    # Stream so we do not download the whole split just to read the schema.
    ds = load_dataset(DATASET, split=SPLIT, streaming=True)

    print("\n=== FEATURES (streaming) ===")
    print(ds.features)

    # Find the Audio column and DISABLE decoding — the pipeline decodes the raw
    # bytes with soundfile (no torchcodec dependency).
    from datasets import Audio

    audio_field = next(n for n, f in ds.features.items() if isinstance(f, Audio))
    ds = ds.cast_column(audio_field, Audio(decode=False))

    first = next(iter(ds))
    print("\n=== FIRST ROW KEYS ===")
    print(list(first.keys()))

    printable = {k: v for k, v in first.items() if k != audio_field}
    print("\n=== FIRST ROW (non-audio fields) ===")
    print(json.dumps(printable, indent=2, default=str)[:1500])

    audio = first[audio_field]
    print(f"\n=== AUDIO FIELD '{audio_field}' (decode=False) ===")
    print("struct keys:", list(audio.keys()))
    print("has bytes:", audio.get("bytes") is not None,
          "| path:", audio.get("path"))

    # Decode this ONE example exactly the way the pipeline does, to prove the
    # torch-free path works on real Svarah audio (the design's decoded-example gate).
    sys.path  # noqa: B018
    from voicerefine_eval.audio import _decode_audio_struct  # noqa: E402

    samples, sr = _decode_audio_struct(audio)
    dur = (len(samples) / sr) if sr else 0.0
    print(f"\n=== DECODED ONE EXAMPLE (via soundfile) ===")
    print(f"sample_rate={sr}  samples={len(samples)}  duration_s={dur:.2f}")
    print("transcript:", repr(first.get("text"))[:200])


if __name__ == "__main__":
    main()
