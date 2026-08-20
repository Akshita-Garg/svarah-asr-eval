# Ten-System Svarah Evaluation - Interpretation

Extends the nine-system comparison with Gnani Prisma v2.5. All other rows are
carried over unchanged from their original source runs; nothing was re-run.

## Result table

All ten systems successfully transcribed the same 200 frozen Svarah recordings.
Corpus WER is the primary accuracy metric; lower is better. Aggregate RTF is
total measured transcription time divided by total audio duration; lower is
faster, and values below 1 are faster than real time.

| System | Location | Corpus WER | Aggregate RTF | Startup |
| --- | --- | ---: | ---: | ---: |
| Sarvam Saaras v4 | Cloud API | 0.0386 | 0.122 | 0.00 s |
| Cohere Transcribe Q4_K | Local CPU | 0.0723 | 1.837 | 13.89 s |
| Whisper Medium English Q4_K | Local CPU | 0.0728 | 2.219 | 5.14 s |
| Gnani Prisma v2.5 | Cloud API | 0.0733 | 0.147 | 0.00 s |
| ElevenLabs Scribe v2 | Cloud API | 0.0752 | 0.254 | 0.00 s |
| Smallest.ai Pulse | Cloud API | 0.0752 | 0.114 | 0.00 s |
| Deepgram Nova-3 | Cloud API | 0.0762 | 0.441 | 0.00 s |
| Parakeet TDT 0.6B v3 Q4_K | Local CPU | 0.0829 | 0.439 | 8.84 s |
| Smallest.ai Pulse Pro | Cloud API | 0.1013 | 0.118 | 0.00 s |
| Whisper Base English Q4_K | Local CPU | 0.1143 | 0.298 | 0.53 s |

Generated metrics are in `summary.md`, raw transcripts and per-file timings in
`per_utterance.csv`, and source hashes and backend provenance in
`comparison_manifest.json`.

## Gnani Prisma v2.5

Gnani completed all 200 recordings with zero failures on the first attempt and
required no request pacing. Corpus WER **0.0733**, aggregate RTF **0.147**.

That places it **fourth of ten overall and the second most accurate hosted API,
behind Sarvam**, ahead of ElevenLabs Scribe v2, both Smallest.ai models and
Deepgram Nova-3. Its aggregate RTF of 0.147 is fourth fastest.

Sarvam is not displaced: it beats Gnani on both axes at once, being both more
accurate (0.0386 against 0.0733) and slightly faster (RTF 0.122 against 0.147).

What Gnani does change is the rest of the field. The only other two systems
more accurate than it are Cohere Transcribe and Whisper Medium, local models
that run **12.5x and 15.1x slower** and need a model load before the first
request. Every remaining hosted API is less accurate than Gnani.

Output was Latin-script on all 200 rows, with no non-English script anywhere.

### Configuration

`POST https://api.vachana.ai/stt/v3` with `language_code=en-IN`,
`format=verbatim`, audio posted as multipart `audio_file`, authenticated with
the `X-API-Key-ID` header.

The endpoint takes no model-selection parameter: the served model follows from
the API key, which is issued against Prisma v2.5. `model_label` is recorded in
the run manifest and cache signature for provenance and is not sent as a
request parameter.

`format` accepts `verbatim` (as spoken) or `transcribe` (Gnani's own
formatting, with optional native-numeral ITN). Verbatim was used so that
Whisper's `EnglishTextNormalizer` remains the only formatting pass, matching
the `smart_format=false` decision for Deepgram and the treatment of every other
system here.

### On Gnani's published accuracy claims

Gnani's Prisma v2.5 launch material states lower WER than Sarvam - reported as
roughly 15% lower on rural Hindi dialects and 18% lower in noisy Dravidian
conditions, with a Gramvaani figure of 22.0% against Sarvam's 23.4%.

**Those claims and this result are not in conflict, because they measure
different things.** The differences are substantial enough that neither speaks
to the other:

| | Gnani's published comparison | This evaluation |
| --- | --- | --- |
| Sarvam version | Saaras **v3** | Saaras **v4** |
| Language | Hindi, Dravidian languages | Indian-accented **English** |
| Audio | Telephony, noisy, 8 kHz | 16 kHz, read and spontaneous |
| Dataset | Gramvaani | Svarah |

Prisma v2.5 is positioned for noisy real-world and telephony conditions, which
is a different regime from the one measured here. Nothing in this benchmark
tests rural Hindi, Dravidian languages, 8 kHz telephony, or Saaras v3, so it
neither supports nor refutes Gnani's figures. It measures one axis those claims
do not cover: Indian-accented English at 16 kHz on Svarah, where Sarvam Saaras
v4 leads by a wide margin.

Sources: [Gnani Prisma v2.5 launch coverage](https://www.businesstoday.in/technology/artificial-intelligence/story/gnani-ai-launches-prisma-v2-5-claims-better-accuracy-than-sarvam-elevenlabs-on-indian-speech-538008-2026-06-19),
[Gnani model page](https://huggingface.co/gnani-ai/gnani).

### Disclosure

Gnani documents Prisma v2.5 as a proprietary hosted service. No model weights,
tokenizers or processor files are distributed, and parameter count,
architecture, training corpus and training hours are **not publicly disclosed**.
Supported languages are documented.

## Results by utterance duration

Corpus WER within each duration band, recomputed across all ten systems. Bands
and methodology are unchanged from the nine-system comparison: the subset spans
0.14 s to 26.61 s (median 4.13 s), and sub-second utterances are a quarter of
the rows but only 61 reference words, 2.9% of the corpus.

| System | <1 s | 1-3 s | 3-6 s | >6 s | Shorter half | Longer half |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sarvam Saaras v4 | 0.1311 | 0.0752 | 0.0364 | 0.0312 | 0.0543 | 0.0348 |
| Cohere Transcribe Q4_K | 0.3279 | 0.1429 | 0.0585 | 0.0593 | 0.1210 | 0.0605 |
| Whisper Medium English Q4_K | 0.2459 | 0.1203 | 0.0649 | 0.0633 | 0.0963 | 0.0671 |
| Gnani Prisma v2.5 | 0.3607 | 0.0902 | 0.0633 | 0.0625 | 0.1037 | 0.0659 |
| ElevenLabs Scribe v2 | 0.3443 | 0.1203 | 0.0759 | 0.0569 | 0.1136 | 0.0659 |
| Smallest.ai Pulse | 0.4098 | 0.0977 | 0.0680 | 0.0601 | 0.1136 | 0.0659 |
| Deepgram Nova-3 | 0.2295 | 0.1278 | 0.0791 | 0.0617 | 0.1086 | 0.0683 |
| Parakeet TDT 0.6B v3 Q4_K | 0.3607 | 0.1805 | 0.0665 | 0.0673 | 0.1358 | 0.0701 |
| Smallest.ai Pulse Pro | 0.3607 | 0.2105 | 0.1266 | 0.0641 | 0.1383 | 0.0923 |
| Whisper Base English Q4_K | 0.2787 | 0.1429 | 0.1203 | 0.1002 | 0.1383 | 0.1084 |

### Where Gnani sits in the bands

Gnani has the most lopsided profile in the comparison. It is **among the
weakest on sub-second audio** (0.3607, ninth of ten and tied with Parakeet and
Pulse Pro; only Smallest.ai Pulse is worse) yet **second best on 1-3 s
utterances** (0.0902, behind only Sarvam), and remains strong from three
seconds up.

Its overall fourth place is therefore earned almost entirely on utterances
longer than a second. For sustained speech it is close to the best hosted API
available here; for one-word commands it is among the weakest. That is close to
the mirror image of Deepgram, which is the strongest non-Sarvam system on
sub-second audio and the weakest of the cloud group on long-form.

The earlier caveats still apply: the sub-second band holds only 61 reference
words and is directional rather than precise, and 55% of Pulse Pro's sub-second
errors come from its output-script rows.

## What the tenth system changes

Gnani is the first system added that materially changes the picture, because it
breaks the assumption that the field divides into "Sarvam, then everyone else at
roughly 0.075". At 0.0733 it sits between the two slow local models and the
hosted cluster, and it does so at a latency none of the accurate systems match.

Sarvam still leads decisively - 0.0386 against 0.0733 is close to a factor of
two - and still leads in every duration band. But the practical shortlist for an
Indian-English product is now two hosted systems rather than one, separated by
accuracy on one side and by very little on the other.

This remains a 200-utterance, single-accent-family, English-only, batch-mode
benchmark. The limitations stated in the repository README apply to this
comparison unchanged.

## Timing comparability

Gnani was measured on 2026-08-20, later than every other system here. Cloud
providers were measured at different times against live production endpoints
under unknown load, so small latency differences between hosted APIs are
observations rather than permanent speed rankings. Accuracy figures are
unaffected by measurement time.
