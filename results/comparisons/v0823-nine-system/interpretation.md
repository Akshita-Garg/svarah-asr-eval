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
