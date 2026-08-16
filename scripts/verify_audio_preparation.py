"""Check that audio preparation did not alter the evaluated recordings.

Every backend is sent a prepared 16 kHz mono PCM_16 WAV rather than the dataset's
own audio, so a fair question is whether that conversion changed the signal.

This decodes each utterance's native audio straight from the dataset and
compares it with the prepared WAV, reporting the native sample rate, how many
utterances actually needed resampling, and the largest per-sample difference.

Svarah ships at 16 kHz mono, so the expected result is zero resampled utterances
and a residual at the scale of one 16-bit quantization step (1/32767 = 3.05e-05),
which is the float-to-PCM_16 rounding done when writing the WAV.

Run:  uv run python scripts/verify_audio_preparation.py
Requires HF_TOKEN and accepted Svarah terms (or a populated local dataset cache).
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voicerefine_eval.audio import TARGET_SR, prepared_path_for
from voicerefine_eval.dataset import MANIFEST_PATH

# Rows whose Pulse Pro output contained Devanagari; reported separately so any
# difference between them and the rest would be visible.
AFFECTED = {
    "svarah_test_0008", "svarah_test_0018", "svarah_test_0029", "svarah_test_0034",
    "svarah_test_0048", "svarah_test_0061", "svarah_test_0068", "svarah_test_0070",
    "svarah_test_0074", "svarah_test_0130", "svarah_test_0140", "svarah_test_0141",
    "svarah_test_0159", "svarah_test_0185", "svarah_test_0190", "svarah_test_0191",
}


def _decode_native(entry: dict) -> tuple[np.ndarray, int]:
    """Decode one row's native audio via the same raw-bytes path the harness uses."""
    if entry.get("bytes") is not None:
        data, sr = sf.read(io.BytesIO(entry["bytes"]), dtype="float32", always_2d=False)
    else:
        data, sr = sf.read(entry["path"], dtype="float32", always_2d=False)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data, int(sr)


def main() -> int:
    from datasets import Audio, load_dataset

    manifest = json.loads(Path(MANIFEST_PATH).read_text(encoding="utf-8"))
    audio_col = manifest["schema"]["audio_field"]

    ds = load_dataset(manifest["dataset"], split=manifest["split"])
    ds = ds.cast_column(audio_col, Audio(decode=False))

    rates: dict[int, int] = {}
    resampled: list[str] = []
    diffs: list[tuple[str, float]] = []
    missing = 0

    for entry in manifest["entries"]:
        eval_id = entry["eval_id"]
        prepared = prepared_path_for(eval_id)
        if not prepared.exists():
            missing += 1
            continue

        native, sr = _decode_native(ds[entry["row_index"]][audio_col])
        rates[sr] = rates.get(sr, 0) + 1
        if sr != TARGET_SR:
            resampled.append(eval_id)
            continue

        prep, _ = sf.read(str(prepared), dtype="float32", always_2d=False)
        if len(native) != len(prep):
            resampled.append(eval_id)
            continue

        # Quantize the source the same way soundfile writes PCM_16, so the only
        # residual left is that rounding rather than a format mismatch.
        quantized = np.round(np.clip(native, -1.0, 1.0) * 32767).astype(np.int16)
        diffs.append((eval_id, float(np.max(np.abs(quantized.astype(np.float32) / 32767 - prep)))))

    if missing:
        print(f"WARNING: {missing} prepared WAV(s) absent; run the evaluation first.")
    if not diffs and not resampled:
        print("No utterances compared.")
        return 1

    print(f"native sample rates: {rates}")
    print(f"required resampling: {len(resampled)} / {len(diffs) + len(resampled)}")

    if diffs:
        worst_id, worst = max(diffs, key=lambda d: d[1])
        aff = [d for eid, d in diffs if eid in AFFECTED]
        una = [d for eid, d in diffs if eid not in AFFECTED]
        print(f"max per-sample difference: {worst:.2e}  ({worst_id})")
        print(f"one 16-bit quantization step: {1 / 32767:.2e}")
        if aff:
            print(f"  affected rows   (n={len(aff)}): max {max(aff):.2e}")
        if una:
            print(f"  unaffected rows (n={len(una)}): max {max(una):.2e}")

    ok = not resampled and all(d <= 1 / 32767 for _, d in diffs)
    print("\nRESULT:", "no resampling; residual is PCM_16 rounding only" if ok
          else "differences exceed quantization rounding — inspect above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
