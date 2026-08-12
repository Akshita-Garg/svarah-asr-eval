# Six-System Svarah Evaluation - Interpretation

## Result table

All six systems successfully transcribed the same 200 frozen Svarah recordings.
Corpus WER is the primary accuracy metric; lower is better. Aggregate RTF is
total measured transcription time divided by total audio duration; lower is
faster, and values below 1 are faster than real time.

| System | Location | Corpus WER | Aggregate RTF | Startup |
| --- | --- | ---: | ---: | ---: |
| Sarvam Saaras v4 | Cloud API | **0.0386** | 0.122 | 0.00 s |
| Cohere Transcribe Q4_K | Local CPU | 0.0723 | 1.837 | 13.89 s |
| ElevenLabs Scribe v2 | Cloud API | 0.0752 | 0.254 | 0.00 s |
| Parakeet TDT 0.6B v3 Q4_K | Local CPU | 0.0829 | 0.439 | 8.84 s |
| Smallest.ai Pulse Pro | Cloud API | 0.1013 | **0.118** | 0.00 s |
| Whisper Base English Q4_K | Local CPU | 0.1143 | 0.298 | 0.53 s |

The generated metrics are in `summary.md`, raw transcripts and per-file timings
are in `per_utterance.csv`, and source hashes and backend provenance are in
`comparison_manifest.json`.

## Smallest.ai result

The final Pulse Pro run was a fresh, fully paced 200-recording run. All 200
requests succeeded on their first attempt. A four-second interval between
request starts prevented rate limiting and ran before the stopwatch, so it
increased total batch duration without changing the reported API-call latency.

Pulse Pro was the fastest cloud API observed in these stored runs by a narrow
margin: aggregate RTF 0.118 versus Sarvam's 0.122. These providers were measured
at different times, so the difference is not evidence that Smallest.ai will
always be faster.

The important weakness was script consistency. Although the request explicitly
used `language=en`, 16 outputs contained Devanagari characters. Those 16 rows
produced 83 of Pulse Pro's 210 total word errors. The other 184 rows had 127
errors across 2,025 reference words, but **0.1013 remains the official corpus
WER**. Removing or transliterating only Pulse Pro's difficult rows would no
longer be an apples-to-apples comparison.

## Controlled local comparison

- Parakeet remains the best local default candidate. It is more accurate than
  Whisper Base while staying comfortably faster than real time on this CPU.
- Cohere is the most accurate local model, but its 1.837 aggregate RTF makes it
  too slow for a natural interactive default on this machine.
- Whisper Base is the fastest local candidate, but has the highest corpus WER.
- All three local figures are directly comparable as tested: CrispASR 0.8.23,
  persistent server mode, CPU, eight threads, Q4_K artifacts, identical WAVs,
  one unscored warm-up, and the same timed local HTTP request boundary.

## Cloud comparison and recommendation

Sarvam remains the strongest cloud candidate in this experiment. It has much
better corpus WER than Pulse Pro while showing nearly identical observed
aggregate latency. ElevenLabs is also more accurate than Pulse Pro, although
its preserved latency was measured during an earlier run.

Do not choose Pulse Pro over Sarvam for VoiceRefine based on this evaluation.
Its speed is attractive, but the English-to-Devanagari switching would create
surprising text in an English desktop-dictation product. It may be worth
retesting if Smallest.ai adds a documented Latin-script or transliteration
control. Every cloud option is opt-in by nature because audio leaves the device
and offline use is unavailable.

This is an accent-robustness benchmark, not a complete VoiceRefine product test.
The next experiment should measure recording stop through final text insertion
on realistic dictation, including audio conversion, ASR, cleanup, optional
transformation, and insertion latency.
