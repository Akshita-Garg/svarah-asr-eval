"""Audio preparation: convert each utterance once to 16 kHz mono signed-16 WAV.

DESIGN.md "Audio Preparation": every selected utterance is converted once to
16000 Hz, mono, signed 16-bit PCM WAV. All backends receive the SAME prepared
file so backend-specific decoding/resampling cannot contaminate the comparison.

We decode the dataset's raw encoded bytes with ``soundfile`` and resample with
``soxr`` — a torch-free path that does not depend on the ``datasets`` library's
audio-decoding backend (which in 5.x can pull in torchcodec/torch).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from .config import REPO_ROOT
from .hashing import sha256_file

PREPARED_DIR = REPO_ROOT / "data" / "prepared"
TARGET_SR = 16000


@dataclass(frozen=True)
class PreparedAudio:
    path: Path
    duration_seconds: float
    sha256: str
    samples: int


def _decode_audio_struct(audio: dict) -> tuple[np.ndarray, int]:
    """Return (float32 samples [N] or [N, C], sample_rate) from an Audio struct.

    Handles both the decode=False form ({'bytes', 'path'}) and an already-decoded
    form ({'array', 'sampling_rate'}).
    """
    if audio is None:
        raise ValueError("audio struct is None")
    if audio.get("array") is not None and audio.get("sampling_rate"):
        return np.asarray(audio["array"], dtype=np.float32), int(audio["sampling_rate"])
    if audio.get("bytes") is not None:
        data, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
        return data, int(sr)
    if audio.get("path"):
        data, sr = sf.read(audio["path"], dtype="float32", always_2d=False)
        return data, int(sr)
    raise ValueError(f"Unusable audio struct with keys {list(audio.keys())}")


def prepare_audio(audio: dict, out_path: Path) -> PreparedAudio:
    """Decode -> mono -> 16 kHz -> signed-16 WAV, written to ``out_path``.

    Idempotent-ish: if the target exists we still rewrite (cheap, and guarantees
    the file matches the current code); callers can skip based on existence if
    desired.
    """
    data, sr = _decode_audio_struct(audio)

    # Downmix to mono by averaging channels.
    if data.ndim == 2:
        data = data.mean(axis=1)
    data = np.ascontiguousarray(data, dtype=np.float32)

    if sr != TARGET_SR:
        data = soxr.resample(data, sr, TARGET_SR)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # PCM_16 = signed 16-bit. soundfile clips floats to [-1, 1] range on write.
    sf.write(out_path, data, TARGET_SR, subtype="PCM_16", format="WAV")

    n = int(data.shape[0])
    return PreparedAudio(
        path=out_path,
        duration_seconds=n / TARGET_SR,
        sha256=sha256_file(out_path),
        samples=n,
    )


def prepared_path_for(eval_id: str) -> Path:
    return PREPARED_DIR / f"{eval_id}.wav"


def describe_prepared(path: Path) -> PreparedAudio:
    """Return metadata for an already-prepared WAV without decoding all samples.

    Lets re-runs reuse prepared audio (duration + hash for RTF and cache keys)
    without re-downloading the dataset.
    """
    path = Path(path)
    info = sf.info(str(path))
    return PreparedAudio(
        path=path,
        duration_seconds=info.frames / info.samplerate,
        sha256=sha256_file(path),
        samples=info.frames,
    )


def load_wav_mono_f32(path: Path) -> tuple[np.ndarray, int]:
    """Read a prepared WAV as float32 mono samples for local backends."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), int(sr)
