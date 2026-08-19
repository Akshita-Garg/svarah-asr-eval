# Nine-System Svarah Evaluation - Interpretation

Extends the eight-system comparison with Deepgram Nova-3. All other rows are
carried over unchanged from their original source runs; nothing was re-run.

## Result table

All nine systems successfully transcribed the same 200 frozen Svarah
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
| Deepgram Nova-3 | Cloud API | 0.0762 | 0.441 | 0.00 s |
| Parakeet TDT 0.6B v3 Q4_K | Local CPU | 0.0829 | 0.439 | 8.84 s |
| Smallest.ai Pulse Pro | Cloud API | 0.1013 | 0.118 | 0.00 s |
| Whisper Base English Q4_K | Local CPU | 0.1143 | 0.298 | 0.53 s |

Generated metrics are in `summary.md`, raw transcripts and per-file timings in
`per_utterance.csv`, and source hashes and backend provenance in
`comparison_manifest.json`.

## Deepgram Nova-3

Nova-3 completed all 200 recordings with zero failures on the first attempt and
required no request pacing. It scored a corpus WER of **0.0762**, placing it
sixth of nine and within 0.001 of ElevenLabs Scribe v2 and Smallest.ai Pulse
(both 0.0752). On this subset those three should be read as tied rather than
ranked: a difference of 0.001 corpus WER is two word errors across 2,074
reference words.

Its mean per-utterance WER of **0.1349** is the second lowest of all nine
systems, behind only Sarvam (0.0705) and ahead of Whisper Medium (0.1419). The
gap between its mean rank and its corpus rank indicates errors distributed
fairly evenly across utterances rather than concentrated in a few catastrophic
rows, which is the opposite of the Smallest.ai Pulse Pro pattern.

Aggregate RTF was 0.441, comfortably faster than real time but roughly 3.6x
slower per request than Sarvam (0.122) in these measurements. As with every
cloud row here, that number is API end-to-end latency measured from one location
at one point in time, and is an observation rather than a durable ranking.

Output was Latin-script throughout, with no non-English script on any row.

### Configuration

`model=nova-3`, `language=en`, `punctuate=true`, `smart_format=false`, raw
16 kHz mono WAV bytes with `Content-Type: audio/wav`.

`smart_format` rewrites numbers, currency and dates. It was left off so that
Whisper's `EnglishTextNormalizer` remains the only formatting pass applied,
matching how every other system in this comparison is scored. Enabling it would
have put Deepgram through two formatting stages where the others receive one.

### Disclosure

Deepgram documents Nova-3 as a proprietary hosted API. Parameter count,
architecture, training corpus, and training hours are **not publicly
disclosed**, and no downloadable checkpoint is offered. Deepgram publishes
comparative WER-reduction claims against unnamed competitors; those are vendor
figures measured on their own evaluation sets and are not comparable with the
corpus WER reported here.

Source: [Deepgram models overview](https://developers.deepgram.com/docs/models-languages-overview).

## Results by utterance duration

Corpus WER within each duration band (band edits / band reference words).
The subset spans 0.14 s to 26.61 s, median 4.13 s, so a single aggregate
figure averages over very different speech.

| Band | Utterances | Reference words | Share of corpus |
| --- | ---: | ---: | ---: |
| <1 s | 50 | 61 | 2.9% |
| 1-3 s | 26 | 133 | 6.4% |
| 3-6 s | 60 | 632 | 30.5% |
| >6 s | 64 | 1248 | 60.2% |

Note the imbalance: sub-second utterances are a quarter of the row count but
under 3% of the reference words, so they barely move corpus WER while
dominating any per-utterance mean.

| System | <1 s | 1-3 s | 3-6 s | >6 s | Shorter half | Longer half |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sarvam Saaras v4 | 0.1311 | 0.0752 | 0.0364 | 0.0312 | 0.0543 | 0.0348 |
| Cohere Transcribe Q4_K | 0.3279 | 0.1429 | 0.0585 | 0.0593 | 0.1210 | 0.0605 |
| Whisper Medium English Q4_K | 0.2459 | 0.1203 | 0.0649 | 0.0633 | 0.0963 | 0.0671 |
| ElevenLabs Scribe v2 | 0.3443 | 0.1203 | 0.0759 | 0.0569 | 0.1136 | 0.0659 |
| Smallest.ai Pulse | 0.4098 | 0.0977 | 0.0680 | 0.0601 | 0.1136 | 0.0659 |
| Deepgram Nova-3 | 0.2295 | 0.1278 | 0.0791 | 0.0617 | 0.1086 | 0.0683 |
| Parakeet TDT 0.6B v3 Q4_K | 0.3607 | 0.1805 | 0.0665 | 0.0673 | 0.1358 | 0.0701 |
| Smallest.ai Pulse Pro | 0.3607 | 0.2105 | 0.1266 | 0.0641 | 0.1383 | 0.0923 |
| Whisper Base English Q4_K | 0.2787 | 0.1429 | 0.1203 | 0.1002 | 0.1383 | 0.1084 |

Halves split at the median duration (4.13 s): 405 reference words in
the shorter half, 1669 in the longer. Rows are ordered by overall corpus WER.

### What the bands show

**The three-way tie at 0.075 is not a tie.** ElevenLabs Scribe v2, Smallest.ai
Pulse and Deepgram Nova-3 sit within 0.001 corpus WER overall, but they are not
the same system. On sub-second utterances Deepgram scores 0.2295 against
ElevenLabs' 0.3443 and Pulse's 0.4098. Above six seconds the order reverses and
Deepgram is the weakest of the three (0.0617 against 0.0569 and 0.0601). A
single aggregate number averages that difference away entirely.

For a product decision this is the part that matters: these three are
interchangeable on paper and clearly not interchangeable in use. Short-command
input favours Deepgram; sustained dictation favours ElevenLabs or Pulse.

**Every system degrades sharply on short audio.** Corpus WER in the sub-second
band is between 2.8x and 6.8x the same system's figure above six seconds -
without exception, local and hosted, small and large. Smallest.ai Pulse shows
the widest gap (6.8x) and Whisper Base the narrowest (2.8x), the latter only
because its long-form accuracy is already the weakest in the comparison.

Short utterances offer no surrounding context to recover from, and a single wrong word on a one-word
reference scores 1.0. Any product whose real traffic is dominated by short
commands should expect materially worse accuracy than an aggregate benchmark
figure implies.

**Sarvam's lead holds in every band.** It is the most accurate system in all
four duration bands and in both halves, so its overall margin is not an artifact
of one length regime. This is the strongest single piece of evidence in the
comparison.

**Whisper Base is the clearest capacity story.** It is mid-pack on short audio
(0.2787, fourth) but last by a wide margin above six seconds (0.1002). Its
weakness is sustained transcription rather than short-utterance robustness.

### Caveats on these bands

- **The sub-second band is small.** 50 utterances but only **61 reference
  words**, 2.9% of the corpus. A handful of errors moves that column by several
  points, so it is directional rather than precise. The 3-6 s (632 words) and
  >6 s (1,248 words) bands carry the weight and are the reliable columns.
- **Pulse Pro's short-band figure is confounded by its output-script issue.**
  Eight of the sixteen Devanagari rows are under one second, and they account
  for **55% of Pulse Pro's sub-second errors**. Its 0.3607 in that column is
  substantially the script defect reappearing, not an independent finding about
  short-audio handling.
- **Band membership is fixed by the audio, not the system**, so every system is
  scored on exactly the same utterances within each band. The bands are
  comparable across systems even where they are small.
- These are the same runs as the headline table, re-aggregated. No system was
  re-run and no transcript changed.

## What the ninth system changes

Nothing about the existing ordering. Deepgram slots into the dense band between
0.0723 and 0.0829 that already contained five systems, and it does not displace
Sarvam at the top or Whisper Base at the bottom.

The more useful observation is what the band itself now shows. Six of the nine
systems — two local Whisper variants, Cohere, Parakeet, ElevenLabs, Smallest.ai
Pulse and Deepgram — land within roughly one percentage point of each other on
Indian-accented English, while Sarvam sits nearly twice as accurate as any of
them. Deepgram is a large general-purpose ASR provider and performs here in the
same range as a quantized 600M local model. On this dataset, provider scale is
not what separates the field; targeted work on Indian-English acoustics is.

That reading is bounded by what this benchmark measures. It is 200 utterances
from one accent family, scored with an English normalizer, in batch mode. It
does not establish that Sarvam leads on other accents, other languages, or in
streaming use, and the limitations stated in the repository README apply to this
comparison unchanged.

## Timing comparability

Deepgram was measured on 2026-08-19, later than every other cloud system here.
Cloud providers were measured at different times against live production
endpoints under unknown load, so small latency differences between hosted APIs
are observations rather than permanent speed rankings. Accuracy figures are
unaffected by measurement time.
