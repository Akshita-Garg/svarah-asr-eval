# VoiceRefine Svarah ASR Evaluation Design

## Objective

Measure the accuracy and repeated-inference speed of the transcription systems
available to VoiceRefine on a fixed subset of the Svarah Indian English dataset.

The evaluation answers two product questions:

1. Which on-device transcription option should VoiceRefine use by default?
2. How large is the accuracy and latency gap between the on-device options and
   a cloud speech-to-text service?

This experiment measures accent robustness on Svarah. It does not measure
VoiceRefine's complete dictation experience, because Svarah is not a dedicated
laptop-dictation dataset.

## Systems Under Test

The local backends must use the same model artifacts and relevant runtime
configuration as VoiceRefine Desktop.

| Backend ID | System | Runtime |
| --- | --- | --- |
| `voicerefine_parakeet_q4` | Parakeet TDT 0.6B v3 Q4 GGUF | Persistent CrispASR server, CPU, English, 8 threads |
| `voicerefine_whisper_tiny_int8` | Whisper Tiny English INT8 ONNX | `sherpa-onnx`, CPU, English transcription, 4 threads |
| `elevenlabs_scribe_v2` | ElevenLabs Scribe v2 | Batch Speech-to-Text API, English, no diarization or audio-event tags |

NVIDIA NeMo Parakeet is intentionally excluded. It is not the quantized model
or runtime shipped by VoiceRefine, so benchmarking it would not answer the
product-default question.

Cohere Q4 is also excluded from this first evaluation to keep the comparison
limited to VoiceRefine's default local model, its lightweight local alternative,
and one cloud reference. It can be added later through the same backend contract.

## Dataset And Subset

- Dataset: `ai4bharat/Svarah`
- Split: `test`
- Final subset size: 200 utterances
- Debug subset size: 20 utterances
- Selection seed: 42
- Dataset revision: recorded before final sampling

The loader must inspect and validate the live dataset schema before depending on
field names. The selected row indexes and stable evaluation IDs are written to
`data/subset_manifest.json`. Later runs read this manifest rather than sampling
again.

The manifest is part of the experiment definition and is committed to Git. It
records enough metadata to recover the exact subset from the pinned dataset
revision.

## Audio Preparation

Every selected utterance is converted once to:

- 16,000 samples per second
- One channel (mono)
- Signed 16-bit PCM WAV

Prepared files are stored under `data/prepared/` and are reproducible from the
dataset manifest, so they are ignored by Git.

All backends receive the same prepared WAV file. This prevents differences in
backend-specific decoding or resampling from contaminating the ASR comparison.

## Backend Contract

Each backend implements the same lifecycle:

```python
class ASRBackend:
    name: str

    def start(self) -> None:
        ...

    def transcribe(self, audio_path: Path) -> str:
        ...

    def close(self) -> None:
        ...
```

`start()` loads a local model, starts a server, or validates API configuration.
`transcribe()` returns raw transcript text for one prepared audio file. `close()`
releases owned resources even when an error occurs.

Backends are selected through configuration. A missing optional dependency or
model for one backend must not prevent the other selected backends from running.

## Timing Rules

One-time startup is measured and reported separately. It is not included in an
utterance's inference time.

Before timed local inference, each local backend processes one unscored warm-up
utterance. For every scored utterance, timing starts immediately before the
backend request and stops when final transcript text is available.

For ElevenLabs, this elapsed time includes file upload, network travel, service
processing, and response download. It is therefore API end-to-end latency, not a
direct measurement of server compute time.

Real-time factor is calculated as:

```text
RTF = inference seconds / audio duration seconds
```

An RTF below 1 means transcription is faster than the audio's duration.

## Caching

Successful raw transcriptions are cached before normalization and scoring. A
cache key includes:

- Backend ID and versioned configuration
- Model-file hash when a local model is used
- Utterance ID
- Prepared-audio hash

Changing a model, backend configuration, or audio file therefore invalidates the
old entry automatically. Cache files are written atomically to avoid preserving
partial JSON after interruption.

Failures are logged but not cached as successful results. `--no-cache` forces
fresh transcription.

## Text Normalization

Reference and hypothesis text are normalized with the same English text
normalizer before scoring. The selected implementation must be tested against
Whisper's `EnglishTextNormalizer` behavior for punctuation, case, numbers, and
contractions.

Raw references and hypotheses are always retained. Normalization must never
replace the source evidence.

## Accuracy Metrics

`jiwer` supplies word alignment and edit counts. We do not implement edit
distance ourselves.

For every successful utterance and backend, record:

- Hits
- Substitutions
- Deletions
- Insertions
- Reference word count
- Word error rate

Per-utterance WER is:

```text
(substitutions + deletions + insertions) / reference words
```

For each backend, report:

- Mean per-utterance WER
- Median per-utterance WER
- Corpus-level WER from aggregate edit counts
- Mean per-utterance RTF
- Aggregate RTF from total inference time divided by total audio duration
- Success count, failure count, and coverage

Mean WER gives every utterance equal weight. Corpus WER gives longer references
more weight and is the primary benchmark figure.

## Failure And Comparison Rules

Every selected backend attempts every selected utterance. A failure records the
backend, utterance, exception category, message, and attempt count.

Results include two comparison views:

1. Each backend's metrics over all of its successful utterances.
2. Metrics over the shared subset successfully transcribed by every active
   backend.

The shared-success view is the primary direct comparison. Coverage and failure
counts must appear beside WER so that a backend cannot look better by failing on
difficult samples.

ElevenLabs retries HTTP 429 and 5xx responses using bounded exponential backoff
with jitter. Other 4xx responses are not retried.

## Reproducibility Record

Each completed evaluation writes `results/run_manifest.json` containing:

- Git commit
- Dataset name, revision, split, subset size, and seed
- Python and dependency versions
- Operating system and hardware summary
- Active backend configurations
- Local model and executable hashes
- Start and completion times
- Cache policy

No API key, access token, username, or machine-specific secret is written to a
result artifact.

## Outputs

- `results/per_utterance.csv`: long-form row per backend and utterance
- `results/run_manifest.json`: exact run provenance
- `results/summary.md`: standalone methodology and comparison report
- Console output: progress, failures, and ten worst successful utterances per
  backend

The summary must state that Svarah measures Indian English accent robustness and
is not a dictation-specific benchmark.

## Out Of Scope

- Gemma or refinement-quality evaluation
- Synthetic speech
- Self-recorded dictation data
- Training or fine-tuning models
- Memory profiling
- Streaming latency or time to first partial transcript
- Changes to VoiceRefine Desktop

## Acceptance Criteria

- The dataset schema and one decoded example are inspected before backend work.
- A committed manifest identifies a reproducible subset.
- All backends receive identical 16 kHz mono WAV files.
- A 20-utterance run succeeds with both local backends.
- Normalization and metric unit tests pass.
- A second run demonstrates cache hits without retranscription.
- Failures are visible and included in coverage reporting.
- The final report compares all three backends on 200 utterances.
- A new user can reproduce the run from the README without editing source code.
