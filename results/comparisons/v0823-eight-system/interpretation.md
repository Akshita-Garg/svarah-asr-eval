# Eight-System Svarah Evaluation - Interpretation

## Result table

All eight systems successfully transcribed the same 200 frozen Svarah
recordings. Corpus WER is the primary accuracy metric; lower is better.
Aggregate RTF is total measured transcription time divided by total audio
duration; lower is faster, and values below 1 are faster than real time.

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

The generated metrics are in `summary.md`, raw transcripts and per-file timings
are in `per_utterance.csv`, and source hashes and backend provenance are in
`comparison_manifest.json`.

## Model size, access, and openness

These terms answer different questions:

- **Parameter count** is the number of learned values in the original model.
- **Artifact size** is the exact quantized file used by this evaluation. It is
  affected by quantization, tensor types, architecture, and file metadata.
- **Runtime memory** is the RAM or VRAM needed while the model is running. It is
  usually larger than the file and was not formally benchmarked here.
- **Open weights** means the checkpoint can be downloaded under a stated
  license. It does not mean the complete training dataset and training pipeline
  are public.

| System | Access classification | Parameters | Exact evaluated artifact |
| --- | --- | ---: | ---: |
| Whisper Base English Q4_K | Downloadable weights; MIT-licensed repository | 74M | 46,484,531 bytes (46.5 MB) |
| Whisper Medium English Q4_K | Downloadable weights; MIT-licensed repository | 769M | 444,506,557 bytes (444.5 MB) |
| Parakeet TDT 0.6B v3 Q4_K | Downloadable weights; CC BY 4.0 | 600M | 488,674,176 bytes (488.7 MB) |
| Cohere Transcribe Q4_K | Open weights; Apache 2.0 | 2B | 1,510,362,752 bytes (1.510 GB) |
| ElevenLabs Scribe v2 | Proprietary hosted API | Not publicly disclosed | No downloadable artifact |
| Smallest.ai Pulse | Proprietary hosted API | Not publicly disclosed | No downloadable artifact |
| Smallest.ai Pulse Pro | Proprietary hosted API | Not publicly disclosed | No downloadable artifact |
| Sarvam Saaras v4 | Proprietary hosted API in this evaluation | Not publicly disclosed | No downloadable artifact found |

The local sizes above were measured from the actual files whose hashes are
captured by this evaluation. They should not be compared directly with provider
GPU recommendations: for example, a recommendation for a 24 GB GPU describes a
deployment environment, not a 24 GB checkpoint.

### Why the Open ASR Leaderboard contains closed models

"Open ASR Leaderboard" describes an open evaluation project built on public
benchmarks. It is not an open-source-only catalog. The leaderboard UI has a
**Show proprietary (API) models** option, and its published metadata identifies
ElevenLabs Scribe v2 and Smallest.ai Pulse as `Proprietary` with no public model
size. Their presence on that leaderboard therefore does not make their weights,
architecture, or training data open.

## Architecture and training background

### Whisper Base English and Medium English

OpenAI reports **74M parameters** for Base and **769M** for Medium. Whisper is a
sequence-to-sequence Transformer family trained with large-scale weak
supervision. Its model card reports **680,000 hours** of internet audio and
transcripts: 438,000 hours of English audio with English transcripts, 126,000
hours of non-English audio translated to English, and 117,000 hours of
non-English transcription. This evaluation uses the English-only `.en`
checkpoints, not the multilingual checkpoints. The evaluated Q4_K files are
local quantized conversions rather than OpenAI's original full-precision files.

Sources: [OpenAI Whisper model card](https://github.com/openai/whisper/blob/main/model-card.md),
[Whisper repository and MIT license](https://github.com/openai/whisper).

### NVIDIA Parakeet TDT 0.6B v3

NVIDIA describes Parakeet as a **600M-parameter** multilingual ASR model with a
FastConformer encoder and token-and-duration transducer decoder. It supports 25
European languages and is released under **CC BY 4.0**. Its card says it was
initialized from a multilingual CTC checkpoint pretrained on Granary, trained
for 150,000 steps, then fine-tuned for 5,000 steps using about **7,500 hours**
of higher-quality human-transcribed data. The listed data sources include
human-transcribed NeMo ASR Set 3.0 and large pseudo-labelled Granary sources.

Source: [NVIDIA Parakeet TDT 0.6B v3 model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).

### Cohere Transcribe 03-2026

Cohere describes its release as a **2B-parameter** encoder-decoder ASR model
trained from scratch for 14 languages. It uses a Fast-Conformer encoder, a
lightweight decoder, and cross-attention, with more than 90% of parameters in
the encoder. Cohere reports **500,000 hours** of curated audio-transcript pairs,
augmented with synthetic data and filtered through an internal pipeline. It
also states that its data-mix balancing methods are proprietary. The original
model is released under **Apache 2.0**; the Q4_K file evaluated here is a local
GGUF conversion and is much smaller than the original checkpoint.

Source: [Cohere Transcribe release and training description](https://huggingface.co/blog/CohereLabs/cohere-transcribe-03-2026-release).

### ElevenLabs Scribe v2

ElevenLabs documents Scribe v2 as a hosted batch transcription model for more
than 90 languages, with word timestamps, diarization, language detection,
audio-event tags, keyterm prompting, and entity detection. Its launch material
says it is optimized for long, complex recordings and diverse speakers and
accents. Official material found for this audit does **not** disclose parameter
count, checkpoint size, architecture, named training corpora, or training
hours. The Open ASR Leaderboard marks it proprietary.

Sources: [ElevenLabs model documentation](https://elevenlabs.io/docs/overview/models),
[Scribe v2 launch](https://elevenlabs.io/blog/introducing-scribe-v2), and
[Open ASR Leaderboard results metadata](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard-results/viewer).

### Smallest.ai Pulse and Pulse Pro

Smallest.ai documents Pulse as a multilingual streaming and batch model, while
Pulse Pro is an English-only batch model. These evaluations used their hosted
HTTP APIs. The official model cards publish supported languages, features,
benchmarks, latency claims, and deployment guidance, but do **not** publish a
downloadable checkpoint, parameter count, architecture, named training corpus,
or training hours. The Open ASR Leaderboard marks the Smallest.ai entry
proprietary. Consequently, neither a recommended GPU nor hosted throughput can
be converted into a defensible model-size estimate.

Sources: [Pulse model card](https://docs.smallest.ai/models/model-cards/speech-to-text/pulse),
[Pulse Pro model card](https://docs.smallest.ai/waves/model-cards/speech-to-text/pulse-pro),
and [Open ASR Leaderboard results metadata](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard-results/viewer).

### Sarvam Saaras v4

This evaluation successfully called the hosted `saaras:v4` model, but no
official V4 model card disclosing parameters, architecture, checkpoint access,
license, or V4-specific training corpus was found during this audit. It should
therefore be treated as a proprietary API model here. For lineage context only,
Sarvam states that the preceding Saaras v3 was trained on more than **one
million hours** of curated multilingual audio covering Indian languages,
accents, and acoustic conditions. That v3 statement must **not** be assumed to
describe v4.

Sources: [Sarvam speech-to-text API documentation](https://docs.sarvam.ai/api-reference/speech-to-text/transcribe)
and [Saaras v3 training overview](https://www.sarvam.ai/blogs/asr).

Research status: checked **2026-08-13** against official developer/model pages
and the leaderboard's own metadata. Where a field says "not publicly
disclosed," no numerical estimate was substituted.

## Whisper Medium quiet-system rerun

The accepted Medium row comes from a fresh no-cache run using the same frozen
audio, model file, CrispASR 0.8.23 executable, persistent server, CPU backend,
eight threads, Q4_K quantization, warm-up rule, and timed HTTP boundary as the
earlier run. The changed experimental condition was lower concurrent system
load.

All 200 requests succeeded. The new run took about 37 minutes wall-clock and
2,207.75 measured inference seconds, compared with about 62 minutes and
3,731.76 measured seconds previously. Aggregate RTF improved from **3.751 to
2.219**, a **40.8% reduction** in measured inference time.

The latency distribution also tightened:

| Statistic | Busy run | Lower-contention run |
| --- | ---: | ---: |
| Mean request | 18.66 s | 11.04 s |
| Median request | 15.60 s | 10.75 s |
| P90 request | 28.88 s | 13.26 s |
| P95 request | 36.78 s | 13.74 s |
| Maximum request | 57.49 s | 27.38 s |

The lower-contention run was faster on 183 of 200 recordings. All 200 raw
transcripts and all 200 normalized transcripts were byte-for-byte identical
between runs, so corpus WER remained exactly **0.0728**. This isolates the
difference to execution latency rather than model quality.

Live observations during the rerun showed CrispASR using approximately 7.8 to
8 logical CPU cores and about 1.0 GB working memory. These are useful context,
not a formal peak-memory or power benchmark.

## Local architecture interpretation

Whisper Medium and Cohere are essentially tied in corpus accuracy: Medium made
151 errors and Cohere made 150 across 2,074 reference words. Medium is still
about 21% slower by aggregate RTF on this CPU, even in the lower-contention run.
Both are slower than real time and are poor defaults for an interaction that
should feel immediate.

Parakeet remains the strongest practical local default among these tested
configurations. It is less accurate than Medium and Cohere but comfortably
faster than real time. Whisper Base is faster still but has the weakest local
accuracy. The Medium rerun shows that background workload can distort local
latency substantially, so latency tests should record system conditions and be
run without competing CPU-heavy work.

## Cloud and Pulse interpretation

Sarvam remains the most accurate system in this benchmark. Standard Pulse tied
ElevenLabs' aggregate WER, stayed script-consistent, and had the lowest observed
cloud aggregate RTF. Cloud providers were measured at different times, so small
latency differences are observations rather than permanent speed rankings.

Pulse Pro produced 16 Devanagari rows despite `language=en`; standard Pulse
produced none. The separate controlled reproduction and the seven-system
interpretation retain the full diagnostic details.

This benchmark measures Indian-English accent robustness and backend-call
throughput, not complete VoiceRefine behavior. The next product evaluation
should measure recording stop through final text insertion on realistic
dictation.
