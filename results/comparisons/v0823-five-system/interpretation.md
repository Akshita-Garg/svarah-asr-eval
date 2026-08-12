# Five-System Svarah Evaluation - Interpretation

## Result table

All five systems successfully transcribed the same 200 frozen Svarah recordings.
Corpus WER is the primary accuracy metric; lower is better. Aggregate RTF is
total measured transcription time divided by total audio duration; lower is
faster, and values below 1 are faster than real time.

| System | Location | Corpus WER | Aggregate RTF | Startup |
| --- | --- | ---: | ---: | ---: |
| Sarvam Saaras v4 | Cloud API | **0.0386** | **0.122** | 0.00 s |
| Cohere Transcribe Q4_K | Local CPU | 0.0723 | 1.837 | 13.89 s |
| ElevenLabs Scribe v2 | Cloud API | 0.0752 | 0.254 | 0.00 s |
| Parakeet TDT 0.6B v3 Q4_K | Local CPU | 0.0829 | 0.439 | 8.84 s |
| Whisper Base English Q4_K | Local CPU | 0.1143 | 0.298 | 0.53 s |

The complete generated metrics, including mean/median WER, mean RTF, coverage,
and failure counts, are in `summary.md`. Raw transcripts and per-file timings
are in `per_utterance.csv`; source hashes and runtime provenance are in
`comparison_manifest.json`.

## What the controlled local comparison says

- **Parakeet is the best local default candidate.** It is more accurate than
  Whisper Base while staying comfortably faster than real time on this CPU.
- **Cohere is the most accurate local candidate, but not the best interactive
  default on this machine.** Its WER is about 13% lower than Parakeet's, but its
  aggregate inference time is about 4.2 times higher and exceeds real time.
- **Whisper Base is the fastest local candidate.** Its aggregate RTF is about
  32% lower than Parakeet's, but its corpus WER is about 38% higher.
- The three local latency figures are directly comparable as tested: all used
  CrispASR v0.8.23's Windows CPU build, a persistent server, eight threads,
  Q4_K artifacts, identical prepared audio, one unscored warm-up, and the same
  timed HTTP request boundary.

## What the cloud comparison says

- **Sarvam produced the best accuracy and observed latency in this run.** Its
  WER is about 47% lower than the best local model, Cohere.
- Sarvam and ElevenLabs RTF include upload, network, provider-side inference,
  and response download. They measure the wait observed during these runs, not
  the providers' pure model compute.
- ElevenLabs was not called again. Its stored rows come from the original run
  over the exact same subset and prepared audio. This avoids spending credits,
  but its latency was measured at a different time and the provider may have
  changed infrastructure since then.
- Both cloud options send audio off-device and therefore do not satisfy
  VoiceRefine's local-only privacy and offline positioning.

## Recommended VoiceRefine direction

Keep Parakeet as the default local transcription path for this Windows CPU
architecture. It offers the strongest practical balance in this experiment.
Keep Cohere as an optional accuracy-first local choice only if the UI clearly
communicates the latency cost, or revisit it when a materially faster native
GPU backend is available. Sarvam can be offered as an explicit opt-in cloud
mode for users who prioritize accuracy and speed over local-only processing.

Do not use this benchmark alone as the final product decision. Svarah measures
Indian English accent robustness and includes many very short clips and proper
nouns; it is not a desktop dictation workflow. The next useful experiment is a
small product-latency evaluation that times recording stop through final text
insertion on realistic VoiceRefine utterances.
