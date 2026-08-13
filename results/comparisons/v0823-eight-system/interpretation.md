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
