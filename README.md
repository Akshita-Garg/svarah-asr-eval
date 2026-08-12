# VoiceRefine Svarah ASR Evaluation

Measures the **accuracy** (Word Error Rate) and **repeated-inference speed**
(Real-Time Factor) of the transcription systems VoiceRefine can use, on a fixed,
reproducible subset of the [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah)
Indian-English dataset.

> ⚠️ Svarah measures **Indian English accent robustness**. It is **not** a
> laptop-dictation dataset, so these numbers do not represent VoiceRefine's full
> dictation experience.

See `DESIGN.md` for the methodology and `BUILD_LOG.md` for a step-by-step account
of how this was built (including the manual steps and pitfalls). The current
six-system results (the original five plus Smallest.ai) are in
[`results/comparisons/v0823-six-system-smallest/summary.md`](results/comparisons/v0823-six-system-smallest/summary.md),
with conclusions in
[`interpretation.md`](results/comparisons/v0823-six-system-smallest/interpretation.md).
For an intuitive explanation of the complete pipeline, metrics, timing boundary,
runtime architecture, and learning resources, read
[`UNDERSTANDING_THE_EVALUATION.md`](UNDERSTANDING_THE_EVALUATION.md).

## Systems under test

| Backend ID | System | Runtime |
| --- | --- | --- |
| `crisp_v0823_whisper_base_en_q4k` | Whisper Base English Q4_K | CrispASR 0.8.23 server, CPU, 8 threads |
| `crisp_v0823_parakeet_q4k` | Parakeet TDT 0.6B v3 Q4_K | CrispASR 0.8.23 server, CPU, 8 threads |
| `crisp_v0823_cohere_q4k` | Cohere Transcribe Q4_K | CrispASR 0.8.23 server, CPU, 8 threads |
| `elevenlabs_scribe_v2` | ElevenLabs Scribe v2 | Speech-to-Text API; preserved baseline results |
| `sarvam_saaras_v4` | Sarvam Saaras v4 | Speech-to-Text API, `en-IN` |
| `smallest_pulse_pro` | Smallest.ai Pulse Pro | Speech-to-Text API, English |

The three controlled local backends use the same runtime version, server mode,
thread count, CPU backend, prepared WAVs, warm-up rule, and timing boundary.
Whisper Base replaces Tiny because Tiny's width cannot be represented by the
legacy Whisper Q4_K format accepted by CrispASR. See `BUILD_LOG.md`, Phase 17.
Whisper Medium was also evaluated, but is retained as a supplementary result
rather than part of this current comparison.

## Prerequisites

1. **Python 3.12** and [uv](https://docs.astral.sh/uv/). Python 3.13/3.14 do
   **not** work — the ASR runtime (`sherpa-onnx-core`) has no wheels for them yet
   (see Troubleshooting).
2. **Local runtime and model artifacts.** The config points to the gitignored
   CrispASR 0.8.23 runtime and Whisper models under `models/`, plus Parakeet and
   Cohere artifacts in the sibling VoiceRefine Desktop checkout. Paths can be
   overridden with the `VOICEREFINE_*` variables in `.env`.
3. **HuggingFace access** to the gated Svarah dataset (token + accepted terms).
4. **Cloud API keys** for new Sarvam or Smallest.ai runs: `SARVAM_API_KEY` and
   `SMALLEST_API_KEY` (the harness also accepts `SMALLESTAI_API_KEY`). The final
   ElevenLabs rows are reused and do not require another paid API call.

## Setup

```bash
# From the repo root:
uv sync                     # creates the Python 3.12 venv from the lockfile
cp .env.example .env        # then add HF_TOKEN and any cloud API keys
```

To use the gated dataset you must first accept its terms while logged in at
<https://huggingface.co/datasets/ai4bharat/Svarah>, then put a read token in
`.env` as `HF_TOKEN=...`.

## Running

```bash
# Fast sanity check on the 20-utterance debug subset:
uv run python -m voicerefine_eval.run --debug

# Full 200-utterance controlled local evaluation:
uv run python -m voicerefine_eval.run

# Run only specific backends:
uv run python -m voicerefine_eval.run --debug --backends crisp_v0823_parakeet_q4k

# Run Smallest.ai only in a fresh output directory:
uv run python -m voicerefine_eval.run --backends smallest_pulse_pro --no-cache --output-dir results/runs/smallest-pulse-pro

# Force fresh transcription (ignore the cache):
uv run python -m voicerefine_eval.run --no-cache
```

Use `--output-dir results/runs/<name>` for an isolated run. The first run
downloads the dataset and prepares 16 kHz mono WAVs under
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

Covers text normalization, WER aggregation, cache invalidation, backend request
contracts, controlled CrispASR flags/provenance, cloud request handling,
retry/resume behavior, and safe merging of immutable run artifacts.

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
- **Smallest.ai backend skipped** - set `SMALLEST_API_KEY` or the accepted
  `SMALLESTAI_API_KEY` alias in `.env`.
