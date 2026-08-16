# VoiceRefine Svarah ASR Evaluation

A controlled, reproducible benchmark of eight speech-to-text systems — four
local CPU models and four hosted APIs — on a frozen 200-utterance subset of
[Svarah](https://huggingface.co/datasets/ai4bharat/Svarah), the AI4Bharat
Indian-English accent dataset.

Every system receives byte-identical 16 kHz mono WAV files, is scored with the
same normalizer, and is timed at the same boundary. Every published number is
backed by a committed per-utterance CSV and a provenance manifest recording the
git commit, dataset revision, dependency versions, hardware, and artifact
hashes.

> **Scope.** Svarah measures **Indian-English accent robustness**. It is not a
> laptop-dictation dataset, so these numbers do not represent VoiceRefine's full
> dictation experience. The intended follow-up is a product-level measurement
> from recording-stop through final text insertion.

---

## Headline results

All eight systems transcribed the same 200 recordings with **zero failures**.
Corpus WER (aggregate edits ÷ aggregate reference words) is the primary accuracy
metric; lower is better. Aggregate RTF is total measured transcription time ÷
total audio duration; below 1.0 is faster than real time.

| System | Location | Corpus WER | Aggregate RTF | Startup |
| --- | --- | ---: | ---: | ---: |
| Sarvam Saaras v4 | Cloud API | **0.0386** | 0.122 | 0.00 s |
| Cohere Transcribe Q4_K | Local CPU | 0.0723 | 1.837 | 13.89 s |
| Whisper Medium English Q4_K | Local CPU | 0.0728 | 2.219 | 5.14 s |
| Smallest.ai Pulse | Cloud API | 0.0752 | **0.114** | 0.00 s |
| ElevenLabs Scribe v2 | Cloud API | 0.0752 | 0.254 | 0.00 s |
| Parakeet TDT 0.6B v3 Q4_K | Local CPU | 0.0829 | 0.439 | 8.84 s |
| Smallest.ai Pulse Pro | Cloud API | 0.1013 | 0.118 | 0.00 s |
| Whisper Base English Q4_K | Local CPU | 0.1143 | 0.298 | 0.53 s |

Full metrics: [`results/comparisons/v0823-eight-system/summary.md`](results/comparisons/v0823-eight-system/summary.md).
Analysis: [`interpretation.md`](results/comparisons/v0823-eight-system/interpretation.md).

**What the numbers say.** Sarvam Saaras v4 is the most accurate system tested,
by a wide margin. Among local models, Parakeet is the strongest practical
default: less accurate than Cohere or Whisper Medium, but the only one of the
three comfortably faster than real time — both of the more accurate local models
run slower than real time on this CPU and are poor defaults for an interaction
that should feel immediate. Whisper Base is faster still but has the weakest
local accuracy.

---

## Key finding: Pulse Pro returns Devanagari for English-only audio

Smallest.ai documents Pulse Pro as English-only — its `language` query parameter
accepts the single enum value `en`. On Indian-accented English audio it
nevertheless returns **Devanagari transliterations of ordinary English words** on
16 of 200 utterances (8.0%): `volume level` → `वॉल्यूम लेवल`, `Right` → `राइट`,
`London, Singapore, New York, Bangkok, Dubai` → `लंदन, सिंगापुर, न्यूयॉर्क, बैंकॉक, दुबई`.

The same audio bytes sent to standard `pulse` return clean Latin English. Three
controls rule out a client-side mistake — most decisively, requesting
`language=hi` returns **HTTP 400** (`"Invalid enum value. Expected 'en',
received 'hi'"`), proving the parameter is validated rather than ignored. There
is no request a client could send that would select a non-English mode.

The measured impact:

| System | WER, all 200 | WER, excluding the 16 affected rows |
| --- | ---: | ---: |
| Smallest.ai Pulse Pro | **0.1013** | **0.0627** |
| Smallest.ai Pulse | 0.0752 | 0.0681 |

Those 16 rows carry **2.4% of the reference words but 39.5% of Pulse Pro's total
error mass**. Excluding them, Pulse Pro (0.0627) is *more* accurate than standard
Pulse (0.0681) — the expected ordering. **The output-script behavior alone
inverts the ranking of Smallest.ai's two models.**

The behavior is deterministic: three complete runs plus a separate live
reproduction flagged exactly the same rows. It is also distinct from ordinary
code-switching — ElevenLabs Scribe v2 emitted Devanagari on 7 rows, but only for
genuinely Indic proper nouns, leaving surrounding English in Latin. Pulse Pro
transliterates English function words (`ऑफ़` = "of", `ऑल` = "all", `इन` = "in").

📄 **Full write-up with reproduction steps, all 16 rows, and a scoring caveat:**
[`results/artifact-reports/smallest-pulse-pro-script-issue.md`](results/artifact-reports/smallest-pulse-pro-script-issue.md)

```bash
uv run python scripts/compare_smallest_models.py   # live reproduction, 4 API calls
```

---

## Methodology

- **Dataset:** `ai4bharat/Svarah`, revision `ebbf7777…`, split `test`.
- **Subset:** 200 utterances, seed 42, frozen in `data/subset_manifest.json`.
- **Audio:** every backend receives identical 16 kHz mono signed-16 WAVs,
  decoded with `soundfile` + `soxr` (deliberately torch-free and deterministic).
- **Normalization:** Whisper's `EnglishTextNormalizer`, applied identically to
  every system — including the ones it disadvantages. No provider-specific
  post-processing.
- **Scoring:** `jiwer`. Corpus WER is primary; mean/median per-utterance WER are
  reported alongside because a few catastrophic rows distort the mean.
- **Timing:** the measured window wraps only the backend call. Model startup is
  measured separately and reported in its own column. Cloud timing is API
  end-to-end latency (upload + network + service + download).
- **Coverage is reported next to accuracy** so a backend cannot look better by
  failing on hard samples. All eight reached 100%.

### Comparability caveats

Recorded honestly rather than smoothed over:

- **Cloud systems were measured at different times.** Small latency differences
  between hosted APIs are observations, not permanent speed rankings.
- **Local latency is contention-sensitive.** Whisper Medium's first run was
  distorted by concurrent system load: aggregate RTF 3.751 vs **2.219** on a
  lower-contention rerun — a 40.8% swing — while all 200 transcripts stayed
  byte-for-byte identical, so corpus WER was unchanged at 0.0728. Both runs are
  retained in `results/runs/`. The comparison uses the quiet rerun.
- **Whisper Base replaces Whisper Tiny.** Tiny's width cannot be represented by
  the legacy Whisper Q4_K format CrispASR accepts; the Q4_K conversion succeeds
  but fails to load. Documented in BUILD_LOG Phases 15–17.
- **Quantized local artifacts** are Q4_K conversions, not the providers'
  original full-precision checkpoints, and should not be read as measurements of
  those checkpoints.

### Systems under test

| Backend ID | System | Runtime |
| --- | --- | --- |
| `crisp_v0823_whisper_base_en_q4k` | Whisper Base English Q4_K | CrispASR 0.8.23 server, CPU, 8 threads |
| `crisp_v0823_whisper_medium_en_q4k` | Whisper Medium English Q4_K | CrispASR 0.8.23 server, CPU, 8 threads |
| `crisp_v0823_parakeet_q4k` | Parakeet TDT 0.6B v3 Q4_K | CrispASR 0.8.23 server, CPU, 8 threads |
| `crisp_v0823_cohere_q4k` | Cohere Transcribe Q4_K | CrispASR 0.8.23 server, CPU, 8 threads |
| `elevenlabs_scribe_v2` | ElevenLabs Scribe v2 | Speech-to-Text API |
| `sarvam_saaras_v4` | Sarvam Saaras v4 | Speech-to-Text API, `en-IN` |
| `smallest_pulse_pro` | Smallest.ai Pulse Pro | Speech-to-Text API, English |
| `smallest_pulse` | Smallest.ai Pulse | Speech-to-Text API, English |

All four local backends share the same runtime version, server mode, thread
count, CPU backend, prepared WAVs, warm-up rule, and timing boundary.

### Model provenance

Checked against official model cards and provider documentation on
**2026-08-13**. "Artifact size" is the exact quantized file evaluated locally,
not peak RAM. Where a field is undisclosed, no estimate was substituted.

| System | Access / license | Parameters | Evaluated artifact | Published training background |
| --- | --- | ---: | ---: | --- |
| Whisper Base English Q4_K | Public weights; MIT | 74M | 46.5 MB | Whisper family: 680,000 h of internet audio; `.en` is English-only |
| Whisper Medium English Q4_K | Public weights; MIT | 769M | 444.5 MB | Same family and English-only checkpoint lineage |
| Parakeet TDT 0.6B v3 Q4_K | Public weights; CC BY 4.0 | 600M | 488.7 MB | Granary-based multilingual pretraining; final stage ~7,500 h human-transcribed |
| Cohere Transcribe Q4_K | Open weights; Apache 2.0 | 2B | 1.510 GB | 500,000 h curated audio-transcript pairs plus synthetic data |
| ElevenLabs Scribe v2 | Proprietary hosted API | Not disclosed | Not available | Not publicly disclosed |
| Smallest.ai Pulse | Proprietary hosted API | Not disclosed | Not available | Not publicly disclosed |
| Smallest.ai Pulse Pro | Proprietary hosted API | Not disclosed | Not available | Not publicly disclosed |
| Sarvam Saaras v4 | Proprietary hosted API | Not disclosed | Not available | No v4-specific disclosure found |

"Open" in *Open ASR Leaderboard* refers to the public benchmark and evaluation
framework, not to every listed model — its own metadata labels ElevenLabs Scribe
v2 and Smallest.ai Pulse as proprietary. Sources and full reasoning are in
[`interpretation.md`](results/comparisons/v0823-eight-system/interpretation.md).

---

## Repository layout

```
voicerefine_eval/      evaluation harness (dataset, audio, backends, metrics, reporting)
  backends/            one module per ASR system, behind a common ASRBackend contract
config/eval.toml       experiment definition: backends, runtime flags, scoring rules
data/subset_manifest.json   the frozen 200-utterance subset (committed)
scripts/               dataset probe, integration smoke test, Smallest.ai comparison
tests/                 40 unit tests over normalization, metrics, cache, backends, merge
results/
  runs/                individual per-backend runs (immutable)
  gates/               5-recording integration gates run before each full evaluation
  comparisons/         merged multi-system reports
  artifact-reports/    the Pulse Pro output-script finding
  archive/             superseded baseline, retained for provenance
DESIGN.md              methodology spec — the source of truth the build follows
BUILD_LOG.md           phase-by-phase build record, including dead ends and fixes
```

Results artifacts are treated as immutable evidence: a new run writes a new
directory rather than overwriting an existing one, and merged comparisons record
the SHA-256 of every source run they were assembled from.

---

## Reproducing

### Prerequisites

1. **Python 3.12** and [uv](https://docs.astral.sh/uv/). Python 3.13/3.14 do
   **not** work — the ASR runtime (`sherpa-onnx-core`) has no wheels for them
   yet (see Troubleshooting).
2. **Local runtime and model artifacts** for the four CrispASR backends. These
   are gitignored; paths are overridable via the `VOICEREFINE_*` variables.
3. **HuggingFace access** to the gated Svarah dataset (token + accepted terms).
4. **Cloud API keys** for new Sarvam or Smallest.ai runs: `SARVAM_API_KEY` and
   `SMALLEST_API_KEY` (`SMALLESTAI_API_KEY` is also accepted).

The harness degrades gracefully: a missing model or key skips that backend
cleanly rather than failing the run.

### Setup

```bash
uv sync                     # creates the Python 3.12 venv from the lockfile
cp .env.example .env        # then add HF_TOKEN and any cloud API keys
```

To use the gated dataset, accept its terms while logged in at
<https://huggingface.co/datasets/ai4bharat/Svarah>, then put a read token in
`.env` as `HF_TOKEN=...`.

### Running

```bash
# Fast sanity check on the 20-utterance debug subset:
uv run python -m voicerefine_eval.run --debug

# Full 200-utterance controlled evaluation:
uv run python -m voicerefine_eval.run

# Run only specific backends, into an isolated output directory:
uv run python -m voicerefine_eval.run --backends smallest_pulse_pro \
    --no-cache --output-dir results/runs/smallest-pulse-pro

# Force fresh transcription (ignore the cache):
uv run python -m voicerefine_eval.run --no-cache
```

The first run downloads the dataset and prepares 16 kHz mono WAVs under
`data/prepared/` (gitignored). Later runs reuse the prepared audio and the
transcript cache, so re-scoring is fast.

### Outputs

Each run directory contains:

- `per_utterance.csv` — one row per backend × utterance: raw and normalized
  text, edit counts, WER, timing, RTF, failure category, attempt count.
- `summary.md` — standalone methodology + comparison report.
- `run_manifest.json` — provenance: git commit, dataset revision, Python and
  dependency versions, hardware, model/binary hashes, timing. Contains no
  secrets.

### Tests

```bash
uv run pytest
```

40 tests covering text normalization, WER aggregation, cache invalidation,
backend request contracts, CrispASR flags and provenance, cloud retry/resume
behavior, and safe merging of immutable run artifacts.

### Reproducibility

The subset is frozen in `data/subset_manifest.json` (committed): dataset
revision `ebbf7777…`, seed 42. Later runs read this manifest rather than
re-sampling. To deliberately re-sample, run with `--resample-subset`.

### Verifying the evidence chain

Every merged comparison records the SHA-256 of each source run it was assembled
from, so the published tables can be traced back to the runs that produced them
without re-running anything:

```bash
python -c "
import json, hashlib, pathlib
m = json.load(open('results/comparisons/v0823-eight-system/comparison_manifest.json'))
for s in m['source_runs']:
    p = pathlib.Path(s['path'])
    for f, k in [('run_manifest.json', 'run_manifest_sha256'),
                 ('per_utterance.csv', 'per_utterance_sha256')]:
        got = hashlib.sha256((p / f).read_bytes()).hexdigest()
        print('OK ' if got == s[k] else 'MISMATCH ', p / f)
"
```

On the committed artifacts this verifies 16 hashes across 8 source runs with
zero mismatches (8 backends × 200 utterances = 1,600 scored rows).

---

## Troubleshooting

- **`The given version [27] is not supported, only version 1 to 10 is supported`**
  — You're on Python 3.13/3.14. `sherpa-onnx` installs but its native
  `sherpa-onnx-core` has no wheel there, so it fails at runtime. The model is
  not the problem. Use Python 3.12
  (`uv python pin 3.12 && rm -rf .venv uv.lock && uv sync`) and ensure
  `sherpa-onnx-core` is installed explicitly — it is not pulled in as a
  dependency of `sherpa-onnx`.
- **`DatasetNotFoundError: ... gated dataset`** — accept the dataset terms on
  the Hub and set `HF_TOKEN` in `.env`.
- **ElevenLabs backend skipped** — `ELEVENLABS_API_KEY` is not set, or verify
  the `model_id`/endpoint in `config/eval.toml` against current docs.
- **Smallest.ai backend skipped** — set `SMALLEST_API_KEY` or the accepted
  `SMALLESTAI_API_KEY` alias in `.env`.
- **HTTP 429 from Smallest.ai on a full run** — the harness paces request starts
  (`min_request_interval_seconds`). The wait happens *before* the timed call, so
  it changes batch wall-clock but does not make individual requests look faster.
- **Windows symlink warning from `huggingface_hub`** — harmless; set
  `HF_HUB_DISABLE_SYMLINKS_WARNING=1` or enable Developer Mode.

---

## Attribution

This repository redistributes reference transcripts from the **Svarah** dataset
in its `per_utterance.csv` artifacts. Svarah is released by AI4Bharat under
**CC BY 4.0**; the audio itself is not redistributed here.

```bibtex
@inproceedings{DBLP:conf/interspeech/JavedJNSNRBKK23,
  author       = {Tahir Javed and Sakshi Joshi and Vignesh Nagarajan and
                  Sai Sundaresan and Janki Nawale and Abhigyan Raman and
                  Kaushal Santosh Bhogale and Pratyush Kumar and
                  Mitesh M. Khapra},
  title        = {Svarah: Evaluating English {ASR} Systems on Indian Accents},
  booktitle    = {{INTERSPEECH}},
  pages        = {5087--5091},
  publisher    = {{ISCA}},
  year         = {2023}
}
```

Model artifacts remain under their respective licenses (Whisper: MIT; Parakeet
TDT 0.6B v3: CC BY 4.0; Cohere Transcribe: Apache 2.0). Hosted APIs were
accessed under their providers' normal terms of service.
