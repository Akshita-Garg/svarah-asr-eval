# VoiceRefine Svarah ASR Evaluation

Measures the **accuracy** (Word Error Rate) and **repeated-inference speed**
(Real-Time Factor) of the transcription systems VoiceRefine can use, on a fixed,
reproducible subset of the [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah)
Indian-English dataset.

> ⚠️ Svarah measures **Indian English accent robustness**. It is **not** a
> laptop-dictation dataset, so these numbers do not represent VoiceRefine's full
> dictation experience.

See `DESIGN.md` for the methodology and `BUILD_LOG.md` for a step-by-step account
of how this was built (including the manual steps and pitfalls).

## Systems under test

| Backend ID | System | Runtime |
| --- | --- | --- |
| `voicerefine_whisper_tiny_int8` | Whisper Tiny English INT8 ONNX | sherpa-onnx, CPU, 4 threads |
| `voicerefine_parakeet_q4` | Parakeet TDT 0.6B v3 Q4 GGUF | persistent CrispASR server, CPU, 8 threads |
| `elevenlabs_scribe_v2` | ElevenLabs Scribe v2 | batch Speech-to-Text API |

The two local backends use the **same model artifacts and runtime configuration
as VoiceRefine Desktop** (see `config/eval.toml`).

## Prerequisites

1. **Python 3.12** and [uv](https://docs.astral.sh/uv/). Python 3.13/3.14 do
   **not** work — the ASR runtime (`sherpa-onnx-core`) has no wheels for them yet
   (see Troubleshooting).
2. **VoiceRefine Desktop artifacts.** By default the config points at a sibling
   checkout `../voicerefine-desktop/resources/...` for the Parakeet GGUF, the
   CrispASR binary, and the Whisper Tiny ONNX model. If yours live elsewhere,
   set the `VOICEREFINE_*` variables in `.env` (see `.env.example`).
3. **HuggingFace access** to the gated Svarah dataset (token + accepted terms).
4. **ElevenLabs API key** (optional — that backend skips cleanly without one).

## Setup

```bash
# From the repo root:
uv sync                     # creates the Python 3.12 venv from the lockfile
cp .env.example .env        # then edit .env and add HF_TOKEN (+ ELEVENLABS_API_KEY)
```

To use the gated dataset you must first accept its terms while logged in at
<https://huggingface.co/datasets/ai4bharat/Svarah>, then put a read token in
`.env` as `HF_TOKEN=...`.

## Running

```bash
# Fast sanity check on the 20-utterance debug subset:
uv run python -m voicerefine_eval.run --debug

# Full 200-utterance evaluation:
uv run python -m voicerefine_eval.run

# Run only specific backends:
uv run python -m voicerefine_eval.run --debug --backends voicerefine_whisper_tiny_int8

# Force fresh transcription (ignore the cache):
uv run python -m voicerefine_eval.run --no-cache
```

The first run downloads the dataset and prepares 16 kHz mono WAVs under
`data/prepared/` (gitignored). Later runs reuse the prepared audio and the
transcript cache, so re-scoring is fast.

### Outputs (written to `results/`)

- `per_utterance.csv` — one row per backend × utterance (raw + normalized text,
  edit counts, WER, timing, RTF, failures).
- `summary.md` — standalone methodology + comparison report.
- `run_manifest.json` — exact provenance (git commit, dataset revision, Python
  and dependency versions, hardware, model/binary hashes, timing). No secrets.

The console shows per-utterance progress, failures, and the ten worst
successful utterances per backend.

## Tests

```bash
uv run pytest
```

Covers text normalization (locked against Whisper's `EnglishTextNormalizer`),
WER edit counts / corpus-vs-mean aggregation, and cache-key invalidation.

## Reproducibility

The subset is frozen in `data/subset_manifest.json` (committed): dataset revision
`ebbf7777…`, seed 42. Later runs read this manifest rather than re-sampling. To
deliberately re-sample, run with `--resample-subset`.

## Troubleshooting

- **`The given version [27] is not supported, only version 1 to 10 is supported`**
  — You're on Python 3.13/3.14. `sherpa-onnx` installs but its native
  `sherpa-onnx-core` has no wheel there, so it fails at runtime. Use Python 3.12
  (`uv python pin 3.12 && rm -rf .venv uv.lock && uv sync`) and ensure
  `sherpa-onnx-core` is installed.
- **`DatasetNotFoundError: ... gated dataset`** — accept the dataset terms on the
  Hub and set `HF_TOKEN` in `.env`.
- **ElevenLabs backend skipped** — `ELEVENLABS_API_KEY` is not set, or verify the
  `model_id`/endpoint in `config/eval.toml` against current ElevenLabs docs.
- **Windows symlink warning from huggingface_hub** — harmless; set
  `HF_HUB_DISABLE_SYMLINKS_WARNING=1` or enable Developer Mode.
