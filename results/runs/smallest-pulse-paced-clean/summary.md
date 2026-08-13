# VoiceRefine Svarah ASR Evaluation — Results

This experiment measures **Indian English accent robustness** on the Svarah dataset. Svarah is not a dedicated laptop-dictation dataset, so these numbers do not measure VoiceRefine's complete dictation experience.

## Methodology

- Dataset: `ai4bharat/Svarah` (revision `ebbf7777fe771490696a3f7b007097606fa8c924`), split `test`.
- Subset: 200 utterances, seed 42.
- All backends receive identical 16 kHz mono signed-16 WAV files.
- Text normalized with Whisper's `EnglishTextNormalizer` before scoring.
- WER/edit counts from `jiwer`. Corpus WER (aggregate edits) is the primary figure.
- Per-utterance timing wraps only the backend call; startup is measured separately.
- Cloud timing is API end-to-end latency (upload + network + service + download).

## Per-backend results (each backend over its own successes)

RTF is measured only over utterances freshly transcribed this run; cache hits carry no timing and are excluded (the **RTF n** column is that sample size, which can be smaller than the success count on a cached run).

| Backend | Avail | Success | Fail | Coverage | Corpus WER | Mean WER | Median WER | Mean RTF | Agg RTF | RTF n | Startup (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smallest_pulse | yes | 200 | 0 | 100% | 0.0752 | 0.1845 | 0.0000 | 0.378 | 0.114 | 200 | 0.00 |

## Primary comparison — shared subset transcribed by every active backend

Shared successful utterances: **200**. This is the primary direct comparison; coverage above shows that a backend cannot look better by failing on hard samples.

| Backend | Corpus WER | Mean WER | Median WER | Mean RTF | Agg RTF | RTF n |
| --- | --- | --- | --- | --- | --- | --- |
| smallest_pulse | 0.0752 | 0.1845 | 0.0000 | 0.378 | 0.114 | 200 |

## Failures by category

- None.
