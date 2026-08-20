# VoiceRefine Svarah ASR Evaluation

A controlled, reproducible benchmark of ten speech-to-text systems — four
local CPU models and six hosted APIs — on a frozen 200-utterance subset of
[Svarah](https://huggingface.co/datasets/ai4bharat/Svarah), the AI4Bharat
Indian-English accent dataset.

Every system receives byte-identical 16 kHz mono WAV files, is scored with the
same normalizer, and is timed at the same boundary. Every published number is
backed by a committed per-utterance CSV and a provenance manifest recording the
git commit, dataset revision, dependency versions, hardware, and artifact
hashes.

> **Scope.** Svarah measures **Indian-English accent robustness** across read,
> spontaneous, and task-oriented speech. It is not a dictation corpus, so these
> numbers do not represent VoiceRefine's dictation experience. Read
> [Limitations](#limitations) before carrying any number here further.

---

## Headline results

All ten systems transcribed the same 200 recordings with **zero failures**.
Corpus WER (aggregate edits ÷ aggregate reference words) is the primary accuracy
metric; lower is better. Aggregate RTF is total measured transcription time ÷
total audio duration; below 1.0 is faster than real time.

| System | Location | Corpus WER | Aggregate RTF | Startup |
| --- | --- | ---: | ---: | ---: |
| Sarvam Saaras v4 | Cloud API | **0.0386** | 0.122 | 0.00 s |
| Cohere Transcribe Q4_K | Local CPU | 0.0723 | 1.837 | 13.89 s |
| Whisper Medium English Q4_K | Local CPU | 0.0728 | 2.219 | 5.14 s |
| Gnani Prisma v2.5 | Cloud API | 0.0733 | 0.147 | 0.00 s |
| Smallest.ai Pulse | Cloud API | 0.0752 | **0.114** | 0.00 s |
| ElevenLabs Scribe v2 | Cloud API | 0.0752 | 0.254 | 0.00 s |
| Deepgram Nova-3 | Cloud API | 0.0762 | 0.441 | 0.00 s |
| Parakeet TDT 0.6B v3 Q4_K | Local CPU | 0.0829 | 0.439 | 8.84 s |
| Smallest.ai Pulse Pro | Cloud API | 0.1013 | 0.118 | 0.00 s |
| Whisper Base English Q4_K | Local CPU | 0.1143 | 0.298 | 0.53 s |

Full metrics: [`results/comparisons/v0823-ten-system/summary.md`](results/comparisons/v0823-ten-system/summary.md).
Analysis: [`interpretation.md`](results/comparisons/v0823-ten-system/interpretation.md).

**What the numbers say.** Sarvam Saaras v4 is the most accurate system tested,
by a wide margin. Among local models, Parakeet is the strongest practical
default: less accurate than Cohere or Whisper Medium, but the only one of the
three comfortably faster than real time — both of the more accurate local models
run slower than real time on this CPU and are poor defaults for an interaction
that should feel immediate. Whisper Base is faster still but has the weakest
local accuracy.

**One result needs a footnote.** On 16 of the 200 utterances, Pulse Pro returned
Devanagari transliterations of English words while standard Pulse returned Latin
English for the same audio. Those rows carry 2.4% of the reference words but
39.5% of Pulse Pro's error mass; excluding them it scores 0.0627, ahead of
standard Pulse's 0.0681. The headline table above leaves them in, because
excluding rows for one provider only would not be a like-for-like comparison.
Details, controls, and a scoring caveat:
[`smallest-pulse-pro-script-issue.md`](results/artifact-reports/smallest-pulse-pro-script-issue.md).

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
  failing on hard samples. All ten reached 100%.

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
| `deepgram_nova3` | Deepgram Nova-3 | Speech-to-Text API, English |
| `gnani_prisma_v25` | Gnani Prisma v2.5 | Speech-to-Text API, `en-IN` |

All four local backends share the same runtime version, server mode, thread
count, CPU backend, prepared WAVs, warm-up rule, and timing boundary.

### Model provenance

Checked against official model cards and provider documentation on
**2026-08-13**, **2026-08-19** for Deepgram and Sarvam, and **2026-08-20**
for Gnani. "Artifact size" is
the exact quantized file evaluated locally, not peak RAM. Where a field is
undisclosed, no estimate was substituted.

| System | Access / license | Parameters | Evaluated artifact | Published training background |
| --- | --- | ---: | ---: | --- |
| Whisper Base English Q4_K | Public weights; MIT | 74M | 46.5 MB | Whisper family: 680,000 h of internet audio; `.en` is English-only |
| Whisper Medium English Q4_K | Public weights; MIT | 769M | 444.5 MB | Same family and English-only checkpoint lineage |
| Parakeet TDT 0.6B v3 Q4_K | Public weights; CC BY 4.0 | 600M | 488.7 MB | Granary-based multilingual pretraining; final stage ~7,500 h human-transcribed |
| Cohere Transcribe Q4_K | Open weights; Apache 2.0 | 2B | 1.510 GB | 500,000 h curated audio-transcript pairs plus synthetic data |
| ElevenLabs Scribe v2 | Proprietary hosted API | Not disclosed | Not available | Not publicly disclosed |
| Smallest.ai Pulse | Proprietary hosted API | Not disclosed | Not available | Not publicly disclosed |
| Smallest.ai Pulse Pro | Proprietary hosted API | Not disclosed | Not available | Not publicly disclosed |
| Sarvam Saaras v4 | Proprietary hosted API | Not disclosed | Not available | Language coverage documented; corpus, hours and architecture not disclosed |
| Deepgram Nova-3 | Proprietary hosted API | Not disclosed | Not available | Not publicly disclosed |
| Gnani Prisma v2.5 | Proprietary hosted API | Not disclosed | Not available | Language coverage documented; corpus, hours and architecture not disclosed |

Sources for each row are in
[`interpretation.md`](results/comparisons/v0823-ten-system/interpretation.md).

---

## Limitations

What this benchmark does **not** establish. These are stated up front because
each one bounds how far the numbers above can be carried.

**On the task being measured**

- **Svarah is a mixed-register accent benchmark, not a dictation corpus.** It
  combines read speech with spontaneous conversational speech, spanning domains
  such as history, culture, tourism, government and sports, alongside real-world
  task utterances (ordering groceries, digital payments, checking a pension
  claim or passport status). The 200-utterance subset reflects that mix: it
  ranges from multi-clause extempore passages to bare voice-assistant commands
  like `Up`, `Backward`, and `volume level`. That spread is a strength for
  accent coverage, but it is not the same distribution as sustained
  laptop dictation, and short command utterances make per-utterance WER
  volatile — a single wrong word on a one-word reference scores 1.0.
- **Batch transcription only.** Every system received a complete WAV and
  returned a final transcript. Streaming and partial-hypothesis decoding were
  not measured, and that is the mode a live dictation product would use.
- **No end-to-end product measurement.** The timed window covers the backend
  call only. Audio capture, conversion, cleanup, optional transformation, and
  text insertion are all excluded.
- **200 utterances, one dataset, one accent family.** Small enough that a
  handful of hard recordings move corpus WER meaningfully, and narrow enough
  that nothing here generalises to other languages or to non-Indian English.

**On the local models**

- **Q4_K artifacts, not original checkpoints.** All four local systems were
  evaluated as 4-bit quantized conversions. Results should not be read as
  measurements of the providers' released models.
- **No full-precision control.** No FP16/FP32 run was performed alongside the
  quantized ones, so the accuracy cost of quantization is **not isolated** by
  this experiment. Where a local model trails a hosted API, this data cannot
  say how much of the gap is quantization versus the model itself.
- **One machine, one CPU, no GPU.** All local figures come from a single
  8-thread Intel CPU. RTF is hardware-specific and will not transfer.
- **Whisper Base substitutes for Whisper Tiny.** Tiny's width cannot be
  represented by the legacy Whisper Q4_K format CrispASR accepts — the
  conversion succeeds but the load fails (BUILD_LOG Phases 15–17).

**On timing**

- **Cloud systems were measured at different times**, against live production
  endpoints under unknown load. Small latency differences between hosted APIs
  are observations, not durable speed rankings.
- **Local latency is contention-sensitive.** Whisper Medium's first run was
  distorted by concurrent system load: aggregate RTF 3.751 versus **2.219** on a
  lower-contention rerun — a 40.8% swing — while all 200 transcripts stayed
  byte-for-byte identical, leaving corpus WER unchanged at 0.0728. Both runs are
  retained in `results/runs/`; the comparison uses the quiet rerun. Any RTF here
  should be read with that margin in mind.
- **Network latency is inside the cloud numbers.** Cloud RTF includes upload and
  download from a single location, so it measures the service as consumed from
  here, not the model's inference speed.

**On scoring**

- **One normalizer defines "correct".** Whisper's `EnglishTextNormalizer` is
  applied identically to every system, but it encodes choices — it strips
  Devanagari combining marks, which inflates hypothesis token counts on
  non-Latin output. The affected rows are quantified in the finding report.
- **Reference transcripts are not infallible.** Svarah's references are human
  transcriptions and contain occasional inconsistencies that count against every
  system equally.

---

## Repository layout

```
voicerefine_eval/      evaluation harness (dataset, audio, backends, metrics, reporting)
  backends/            one module per ASR system, behind a common ASRBackend contract
config/eval.toml       experiment definition: backends, runtime flags, scoring rules
data/subset_manifest.json   the frozen 200-utterance subset (committed)
scripts/               dataset probe, integration smoke test, audio-preparation check
tests/                 40 unit tests over normalization, metrics, cache, backends, merge
results/
  runs/                individual per-backend runs (immutable)
  gates/               5-recording integration gates run before each full evaluation
  comparisons/         merged multi-system reports
  artifact-reports/    write-ups of individual findings
  archive/             superseded baseline, retained for provenance
DESIGN.md              methodology spec — the source of truth the build follows
BUILD_LOG.md           phase-by-phase build record, including dead ends and fixes
```

Results artifacts are treated as immutable evidence: a new run writes a new
directory rather than overwriting an existing one, and merged comparisons record
the SHA-256 of every source run they were assembled from.

---

## Reproducing

```bash
uv sync                                    # 1. install (Python 3.12, from lockfile)
cp .env.example .env                       # 2. add your own HF_TOKEN + API keys
uv run python -m voicerefine_eval.run      # 3. run the 200-utterance evaluation
```

No credentials are bundled. Backends whose key or model artifact is missing skip
cleanly, so the harness runs with whatever subset you have access to — the four
CrispASR backends additionally need local model artifacts, which are not
redistributed here.

### Prerequisites

1. **[uv](https://docs.astral.sh/uv/)**, which handles the interpreter for you.
   `.python-version` and `requires-python = ">=3.12,<3.13"` pin the project to
   **Python 3.12**, so `uv sync` selects it — fetching it if it is not already
   installed — rather than using a newer one. The pin exists because
   `sherpa-onnx-core` has no wheels past 3.12. That is a dependency retained
   from the superseded sherpa-based Whisper Tiny backend and is not used by any
   system in the results: the four local backends run on the standalone CrispASR
   binary and are driven over HTTP, so they are indifferent to the Python
   version. BUILD_LOG Phase 3.5 records how the pin was diagnosed.
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
m = json.load(open('results/comparisons/v0823-ten-system/comparison_manifest.json'))
for s in m['source_runs']:
    p = pathlib.Path(s['path'])
    for f, k in [('run_manifest.json', 'run_manifest_sha256'),
                 ('per_utterance.csv', 'per_utterance_sha256')]:
        got = hashlib.sha256((p / f).read_bytes()).hexdigest()
        print('OK ' if got == s[k] else 'MISMATCH ', p / f)
"
```

On the committed artifacts this verifies 20 hashes across 10 source runs with
zero mismatches (10 backends × 200 utterances = 2,000 scored rows).

---

## License and attribution

The evaluation harness, scripts, tests, and documentation in this repository are
released under the [MIT License](LICENSE). Third-party terms covering the
dataset and the evaluated models are set out in [NOTICE](NOTICE).

The benchmark is built on **[Svarah](https://huggingface.co/datasets/ai4bharat/Svarah)**,
created and released by **[AI4Bharat](https://ai4bharat.iitm.ac.in/)** under
**[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**. This repository
redistributes Svarah reference transcripts inside its `per_utterance.csv`
artifacts; the audio itself is **not** redistributed. Paper:
[*Svarah: Evaluating English ASR Systems on Indian Accents*](https://arxiv.org/abs/2305.15760)
(Javed et al., INTERSPEECH 2023).

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
