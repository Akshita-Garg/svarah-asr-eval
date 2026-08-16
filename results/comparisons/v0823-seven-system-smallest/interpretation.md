# Seven-System Svarah Evaluation - Interpretation

## Result table

All seven systems successfully transcribed the same 200 frozen Svarah
recordings. Corpus WER is the primary accuracy metric; lower is better.
Aggregate RTF is total measured transcription time divided by total audio
duration; lower is faster, and values below 1 are faster than real time.

| System | Location | Corpus WER | Aggregate RTF | Startup |
| --- | --- | ---: | ---: | ---: |
| Sarvam Saaras v4 | Cloud API | **0.0386** | 0.122 | 0.00 s |
| Cohere Transcribe Q4_K | Local CPU | 0.0723 | 1.837 | 13.89 s |
| Smallest.ai Pulse | Cloud API | 0.0752 | **0.114** | 0.00 s |
| ElevenLabs Scribe v2 | Cloud API | 0.0752 | 0.254 | 0.00 s |
| Parakeet TDT 0.6B v3 Q4_K | Local CPU | 0.0829 | 0.439 | 8.84 s |
| Smallest.ai Pulse Pro | Cloud API | 0.1013 | 0.118 | 0.00 s |
| Whisper Base English Q4_K | Local CPU | 0.1143 | 0.298 | 0.53 s |

The generated metrics are in `summary.md`, raw transcripts and per-file timings
are in `per_utterance.csv`, and source hashes and backend provenance are in
`comparison_manifest.json`.

## Standard Pulse result

The standard Pulse run was a fresh, fully paced 200-recording run. All 200
requests succeeded on their first attempt. A four-second interval between
request starts ran before the stopwatch, preventing sustained-rate failures
without changing the reported request latency.

Standard Pulse made 156 word errors across 2,074 reference words, for **0.0752
corpus WER**. This is exactly the same aggregate error count as the preserved
ElevenLabs run, although the two systems did not necessarily make the same
errors. It was close to local Cohere's 150 errors and materially behind Sarvam's
80 errors.

Its aggregate RTF was **0.114**, and its median request latency was 0.557
seconds. Pulse Pro's corresponding figures were 0.118 and 0.553 seconds. These
runs show similar observed latency; their small difference is not evidence of
a permanent provider speed ranking.

## Pulse versus Pulse Pro script behavior

Standard Pulse produced **zero Devanagari rows and zero other non-Latin rows**.
Pulse Pro produced 16 Devanagari rows and 17 non-Latin rows in total despite
both configurations requesting `language=en` through the same unified endpoint.

Those 16 Devanagari rows account for 83 of Pulse Pro's 210 total errors across
49 reference words. Standard Pulse made 18 errors on those exact same rows.
Across all 200 recordings, standard Pulse had lower utterance WER on 39 rows,
equal WER on 128, and higher WER on 33.

The controlled reproduction for `svarah_test_0048` further ruled out local
audio conversion: the original dataset WAV and prepared WAV were byte-for-byte
identical. Pulse Pro returned Devanagari with both generic-binary and WAV content
types, while standard Pulse returned correct Latin-script English through both
the unified and legacy Pulse endpoints. The strongest current explanation is a
Pulse Pro model or API-routing behavior, not evaluation normalization.

## Product interpretation

Standard Pulse is a credible positive result on this Indian-English accent
benchmark: it matched ElevenLabs' aggregate WER, was close to Cohere, had full
coverage, stayed comfortably faster than real time, and remained script
consistent. Sarvam was still substantially more accurate.

The Pulse Pro script behavior is scoped to what was measured: the same audio and
the same English request produced script switching only when the selected model
was Pulse Pro. It does not follow that Pulse Pro always fails, and this benchmark
does not measure every product scenario.

This is an accent-robustness benchmark, not a complete VoiceRefine product test.
Cloud latency includes upload, network, provider processing, and response time,
and the providers were measured at different times. The next product evaluation
should measure recording stop through final text insertion on realistic
dictation.
