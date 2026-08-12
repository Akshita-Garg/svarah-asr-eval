# Six-System Svarah Evaluation - Interpretation

## Result table

All six systems successfully transcribed the same 200 frozen Svarah recordings.
Corpus WER is the primary accuracy metric; lower is better. Aggregate RTF is
total measured transcription time divided by total audio duration; lower is
faster, and values below 1 are faster than real time.

| System | Location | Corpus WER | Aggregate RTF | Startup |
| --- | --- | ---: | ---: | ---: |
| Sarvam Saaras v4 | Cloud API | **0.0386** | **0.122** | 0.00 s |
| Cohere Transcribe Q4_K | Local CPU | 0.0723 | 1.837 | 13.89 s |
| Whisper Medium English Q4_K | Local CPU | 0.0728 | 3.751 | 1.78 s |
| ElevenLabs Scribe v2 | Cloud API | 0.0752 | 0.254 | 0.00 s |
| Parakeet TDT 0.6B v3 Q4_K | Local CPU | 0.0829 | 0.439 | 8.84 s |
| Whisper Base English Q4_K | Local CPU | 0.1143 | 0.298 | 0.53 s |

The generated metrics are in `summary.md`. Raw transcripts and timings are in
`per_utterance.csv`; source hashes and runtime provenance are in
`comparison_manifest.json`.

## What Whisper Medium adds

Medium answers whether the earlier Whisper result was weak mainly because Base
was much smaller than Parakeet and Cohere. Scaling Whisper from Base to Medium
reduced corpus WER from 0.1143 to 0.0728, an improvement of about 36%. The model
capacity therefore mattered substantially.

The compute cost was much larger than the accuracy gain. Medium used about 12.6
times Base's total inference time, 8.5 times Parakeet's, and 2.0 times Cohere's.
Its median request took 15.6 seconds. The ten slowest requests accounted for
only 11.9% of total inference time, so its slowness was broad rather than caused
by one isolated outlier.

Medium and Cohere should be treated as effectively tied on this sample, not as
a meaningful ranking. Cohere made 150 total word errors and Medium made 151
across 2,074 reference words. At the utterance level, Medium had lower WER on
35 files, Cohere on 32, and they tied on 133. No confidence interval or
significance test was run.

## What the controlled local comparison says

- **Parakeet remains the best local default candidate.** It stays comfortably
  faster than real time and gives a much better accuracy/latency balance than
  either accuracy-first model on this CPU.
- **Cohere remains the better accuracy-first local option of the tested
  configurations.** It provides essentially Medium's corpus accuracy at about
  half the inference time, although its aggregate RTF of 1.837 is still too slow
  for a natural interactive default.
- **Whisper Medium is informative but not practical in this CPU runtime.** It
  proves that a larger Whisper closes the accuracy gap, but its aggregate RTF of
  3.751 is the slowest result in the comparison.
- **Whisper Base remains the fastest local model.** Its aggregate RTF is 0.298,
  but its corpus WER is materially worse than the other local options.

All four local latency figures are directly comparable as tested: each used
CrispASR v0.8.23's Windows CPU build, a persistent server, eight threads, Q4_K
weights, identical prepared audio, one unscored warm-up, and the same timed HTTP
request boundary. The models retain their inherent architectures and decoding
algorithms; controlling execution does not make those algorithms identical.

## Cloud context

Sarvam produced the best accuracy and observed latency in this run. Sarvam and
ElevenLabs timings include upload, network, provider-side inference, and response
download, so they measure observed wait rather than controlled model compute.
ElevenLabs was not called again; its exact stored rows over the frozen subset
were reused to avoid spending credits.

Both cloud systems send audio off-device and therefore do not satisfy
VoiceRefine's local-only privacy and offline positioning.

## Recommended VoiceRefine direction

Keep Parakeet as the default local transcription path for the current Windows
CPU architecture. Do not add Whisper Medium to the product in this runtime: its
accuracy is strong, but the latency is unsuitable for interactive dictation.
Keep Cohere only as an optional accuracy-first local mode if its wait is clearly
communicated, or revisit both larger models when a materially faster GPU or
platform-native backend is available.

Do not use Svarah alone as the final product decision. It measures Indian
English accent robustness and includes many short clips and proper nouns; it is
not a desktop dictation workflow. The next useful experiment remains a small
product-latency evaluation that times recording stop through final text
insertion on realistic VoiceRefine utterances.
