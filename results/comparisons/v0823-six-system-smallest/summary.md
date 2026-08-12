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
| elevenlabs_scribe_v2 | yes | 200 | 0 | 100% | 0.0752 | 0.1809 | 0.0000 | 0.651 | 0.254 | 200 | 0.00 |
| crisp_v0823_whisper_base_en_q4k | yes | 200 | 0 | 100% | 0.1143 | 0.1850 | 0.0438 | 1.017 | 0.298 | 200 | 0.53 |
| crisp_v0823_parakeet_q4k | yes | 200 | 0 | 100% | 0.0829 | 0.1884 | 0.0000 | 0.488 | 0.439 | 200 | 8.84 |
| crisp_v0823_cohere_q4k | yes | 200 | 0 | 100% | 0.0723 | 0.1766 | 0.0000 | 1.921 | 1.837 | 200 | 13.89 |
| sarvam_saaras_v4 | yes | 200 | 0 | 100% | 0.0386 | 0.0705 | 0.0000 | 0.365 | 0.122 | 200 | 0.00 |
| smallest_pulse_pro | yes | 200 | 0 | 100% | 0.1013 | 0.2254 | 0.0000 | 0.353 | 0.118 | 200 | 0.00 |

## Primary comparison — shared subset transcribed by every active backend

Shared successful utterances: **200**. This is the primary direct comparison; coverage above shows that a backend cannot look better by failing on hard samples.

| Backend | Corpus WER | Mean WER | Median WER | Mean RTF | Agg RTF | RTF n |
| --- | --- | --- | --- | --- | --- | --- |
| elevenlabs_scribe_v2 | 0.0752 | 0.1809 | 0.0000 | 0.651 | 0.254 | 200 |
| crisp_v0823_whisper_base_en_q4k | 0.1143 | 0.1850 | 0.0438 | 1.017 | 0.298 | 200 |
| crisp_v0823_parakeet_q4k | 0.0829 | 0.1884 | 0.0000 | 0.488 | 0.439 | 200 |
| crisp_v0823_cohere_q4k | 0.0723 | 0.1766 | 0.0000 | 1.921 | 1.837 | 200 |
| sarvam_saaras_v4 | 0.0386 | 0.0705 | 0.0000 | 0.365 | 0.122 | 200 |
| smallest_pulse_pro | 0.1013 | 0.2254 | 0.0000 | 0.353 | 0.118 | 200 |

## Failures by category

- None.
