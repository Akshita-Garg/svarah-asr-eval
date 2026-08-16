# Smallest.ai Pulse Pro returns Devanagari script for English audio

**Status:** confirmed and independently reproduced live.
**Scope:** 16 of 200 utterances (8.0%) in a frozen Svarah subset.
**Affected model:** `pulse-pro` only. Standard `pulse` is unaffected (0 of 200).

## Summary

Smallest.ai documents Pulse Pro as English-only — its `language` query parameter
accepts the single enum value `en`. On Indian-accented English audio, Pulse Pro
nevertheless returns **Devanagari transliterations of ordinary English words**
on a reproducible subset of recordings: `volume level` → `वॉल्यूम लेवल`,
`Right` → `राइट`, `Backward` → `बैकवर्ड`, `London, Singapore, New York, Bangkok,
Dubai` → `लंदन, सिंगापुर, न्यूयॉर्क, बैंकॉक, दुबई`.

This is not translation and it is not Indic code-switching in the source audio.
The same audio bytes sent to standard `pulse` return clean Latin English.

Because no Devanagari token can ever match a Latin reference token, each affected
row scores WER ≥ 1.0. Those 16 rows carry **2.4% of the reference words but 39.5%
of Pulse Pro's total error mass**, and they are the sole reason Pulse Pro scores
worse than standard Pulse in this benchmark.

## Evidence: the API accepted English and confirmed it

Live reproduction (same request contract the evaluation harness used; full
response bodies, secrets omitted):

```
POST https://api.smallest.ai/waves/v1/stt/?model=pulse-pro&language=en
Content-Type: application/octet-stream    (raw 16 kHz mono WAV bytes)

HTTP 200
{
  "status": "success",
  "transcription": "शो ए लिस्ट ऑफ़ ऑल अवेलेबल वैक्सीनेशन सेंटर इन स्टेट महाराष्ट्र।",
  "words": [],
  "language": "en",
  "metadata": { "duration": 4.151, "processing_time_ms": 376.79, "rtfx": 11, "num_chunks": 1 },
  "totalBytes": 132890,
  "request_id": "5e91c787-c84e-4ae4-a579-856d382fe79a"
}
```

The response **echoes `"language": "en"`**. English was requested, accepted, and
acknowledged — and the transcript is still Devanagari.

Three controls rule out a client-side mistake:

| Control | Request | Result |
| --- | --- | --- |
| Is `language` being silently ignored? | `model=pulse-pro`, parameter omitted | HTTP 200, byte-identical Devanagari output |
| Is `language` actually validated? | `model=pulse-pro&language=hi` | **HTTP 400** — `"Invalid enum value. Expected 'en', received 'hi'"`, `"options": ["en"]` |
| Is the audio itself Hindi? | `model=pulse&language=en`, same bytes | HTTP 200, `"Show a list of all available vaccination center in state Maharashtra."` |

The 400 is the decisive control: the server's own validator states Pulse Pro
accepts only `en`. There is no request the client could have sent that would
select a non-English mode, so the Devanagari output cannot be attributed to a
mis-specified language.

Reproduce with:

```bash
uv run python scripts/compare_smallest_models.py    # edit AUDIO_FILE to pick the recording
```

## Evidence: this harness did not alter the audio

A natural objection is that the evaluation's own audio preparation degraded the
signal and pushed a language-identification path toward Indic. It did not: **no
resampling took place at all.**

Svarah's test split is distributed as 16 kHz mono audio, which is already the
harness's target format, so the resampling branch never executes. Verified
across the whole frozen subset by decoding the dataset's native audio and
comparing it against the prepared WAV actually sent to every backend:

| Check | Result |
| --- | --- |
| Native sample rate, all 200 utterances | 16000 Hz, mono |
| Utterances that required resampling | **0 of 200** |
| Max per-sample difference, source vs prepared | **1.53e-05** |

That residual is half of one 16-bit quantization step (1/32767 ≈ 3.05e-05) — the
float-to-PCM_16 rounding done when writing the WAV, not a filtering artifact.
The figure is the same for the 16 affected rows as for the 184 unaffected ones,
so nothing about the affected audio is distinguishable.

Reproduce with:

```bash
uv run python scripts/verify_audio_preparation.py
```

Two further controls point the same way:

- **Same bytes, different model.** Standard `pulse` receives the byte-identical
  prepared WAV and returns clean Latin English. Any explanation resting on audio
  degradation would have to degrade Pulse Pro while leaving Pulse untouched.
- **Same bytes, four other systems.** Whisper Base, Whisper Medium, Parakeet and
  Cohere consume the same prepared files and all return Latin-script English.

## Evidence: 16 affected utterances

Same audio, same request contract, only `model` differs.

> **On the identifiers.** `svarah_test_NNNN` is assigned by this harness, not by
> Svarah. It maps to a row of the dataset's `test` split at revision
> `ebbf7777…`; the mapping to Svarah's own `audio_filepath` and row index is
> committed in [`data/subset_manifest.json`](../../data/subset_manifest.json)
> and repeated below the table so this report stands alone.

| Utterance | Svarah reference | Pulse Pro (`language=en`) | Pulse (`language=en`) |
| --- | --- | --- | --- |
| `svarah_test_0008` | So, starting with Maghi Chakrati, on Maghi Chakrati which | के सो स्टार्टिंग विद मागे सक्राती ऑन मागे सक्राती आह विच | Okay, so starting with Maghesakrati, on Maghesakrati, which |
| `svarah_test_0018` | Manas | मानव | Manash |
| `svarah_test_0029` | Chandrashekhar Jha | चंद्र सेकर जह | Chandrasekhar Ja |
| `svarah_test_0034` | Padma | पद्म | But no. |
| `svarah_test_0048` | Show a list of all available vaccination center in State Maharashtra | शो ए लिस्ट ऑफ़ ऑल अवेलेबल वैक्सीनेशन सेंटर इन स्टेट महाराष्ट्र। | Show a list of all available vaccination center in state Maharashtra. |
| `svarah_test_0061` | volume level | वॉल्यूम लेवल | Volume level. |
| `svarah_test_0068` | Up | आप | Up. |
| `svarah_test_0070` | Backward | बैकवर्ड | Backward. |
| `svarah_test_0074` | Subarnapur | सुबोर्नपुर | Subarnapur |
| `svarah_test_0130` | London, Singapore, New York, Bangkok, Dubai. | लंदन, सिंगापुर, न्यूयॉर्क, बैंकॉक, दुबई | London, Singapore, New York, Bangkok, Dubai |
| `svarah_test_0140` | Kamrup metro Nalbari, Nagaon, Sonitpur, Lakhimpur. | काम्रूप मेट्रो, नलबारी, नगांव, सुनिदपुर, लखीमपुर | Kamruk Metro, Nalwari, Nagaon, Sunitpur, Lockhur |
| `svarah_test_0141` | Katak | कटक | Cutter |
| `svarah_test_0159` | Three | त्रिग | Three. |
| `svarah_test_0185` | Sunitpur Dhubri | सोनीपुर, धुपरी | Sonitpur Dhopri. |
| `svarah_test_0190` | Right | राइट | Right? |
| `svarah_test_0191` | Add Rs 500 | एड रिबीस फाइव हंड्रेड | Add rupees five hundred. |

### Mapping to Svarah's own identifiers

Dataset `ai4bharat/Svarah`, split `test`, revision `ebbf7777…`. `row` is the
index into that split; `audio_filepath` is Svarah's own filename.

| Harness ID | Row | Svarah `audio_filepath` |
| --- | ---: | --- |
| `svarah_test_0008` | 355 | `281474976888873_f2272_chunk_4.wav` |
| `svarah_test_0018` | 712 | `281474976897459_f2145_chunk_8.wav` |
| `svarah_test_0029` | 913 | `281474976888145_f1900_chunk_4.wav` |
| `svarah_test_0034` | 1170 | `281474976893988_f2006_chunk_3.wav` |
| `svarah_test_0048` | 1728 | `281474976893960_f2523_chunk_0.wav` |
| `svarah_test_0061` | 2006 | `281474976899703_f2791_chunk_0.wav` |
| `svarah_test_0068` | 2185 | `281474976898593_f650_chunk_0.wav` |
| `svarah_test_0070` | 2193 | `281474976902158_f164_chunk_0.wav` |
| `svarah_test_0074` | 2277 | `281474976897387_f2253_chunk_4.wav` |
| `svarah_test_0130` | 4532 | `281474976886918_f1857_chunk_0.wav` |
| `svarah_test_0140` | 4881 | `281474976934169_f3288_chunk_0.wav` |
| `svarah_test_0141` | 4885 | `281474976898115_f1790_chunk_2.wav` |
| `svarah_test_0159` | 5369 | `281474976897091_f2875_chunk_0.wav` |
| `svarah_test_0185` | 6151 | `281474976893839_f3333_chunk_3.wav` |
| `svarah_test_0190` | 6254 | `281474976895799_f3036_chunk_0.wav` |
| `svarah_test_0191` | 6294 | `281474976934140_f2456_chunk_0.wav` |

Rows `0048`, `0061`, `0068`, `0070`, `0130`, `0190` are the clearest exhibits:
the reference contains no Indic proper noun at all, only common English words and
international place names, and Pulse Pro still transliterates the whole utterance.

## This is distinct from ordinary code-switching

ElevenLabs Scribe v2 also emitted Devanagari on this subset, on 7 rows — but only
for genuinely Indic proper nouns, leaving the surrounding English in Latin:

> `Show a list of all available vaccination center in state महाराष्ट्र`

That is normal code-switching behavior. Pulse Pro transliterates the entire
utterance including English function words (`ऑफ़` = "of", `ऑल` = "all",
`इन` = "in", `ए` = "a"), which no code-switching account explains. Only 5 of the
16 Pulse Pro rows overlap with ElevenLabs' 7.

## Reproducibility

Behavior is deterministic, not a transient glitch:

- An earlier unpaced run flagged 14 rows. The 2 missing rows (`0048`, `0159`)
  had failed that run with HTTP 429 and produced no transcript at all. The
  14 are a strict subset of the 16; every completed run agrees exactly.
- Two subsequent complete 200/200 runs each flagged the same 16 rows.
- The live reproduction above, run separately and later, reproduces the same
  Devanagari strings for `0048` and `0061`.

## Accuracy impact

Corpus WER on 200 Svarah utterances (Whisper `EnglishTextNormalizer`, applied
identically to every system):

| System | WER, all 200 | WER, excluding the 16 affected rows |
| --- | --- | --- |
| Smallest.ai Pulse Pro | **0.1013** | **0.0627** |
| Smallest.ai Pulse | 0.0752 | 0.0681 |
| Sarvam Saaras v4 | 0.0386 | 0.0346 |
| ElevenLabs Scribe v2 | 0.0752 | 0.0538 |
| Cohere Transcribe Q4_K | 0.0723 | 0.0632 |
| Whisper Medium.en Q4_K | 0.0728 | 0.0612 |
| Parakeet TDT 0.6B v3 Q4_K | 0.0829 | 0.0721 |
| Whisper Base.en Q4_K | 0.1143 | 0.1042 |

The 16 rows contribute 83 of Pulse Pro's 210 total errors against just 49 of
2,074 reference words. **Excluding them, Pulse Pro (0.0627) is more accurate
than standard Pulse (0.0681)** — the expected ordering. The script issue alone
inverts the ranking of Smallest.ai's two models.

### Caveat on the per-row WER figures

The 83 errors on those rows break down as 49 substitutions and 34 insertions,
with zero deletions. The 49 substitutions equal the reference word count exactly:
under any scoring, every word on these rows is wrong, so WER ≥ 1.0 is a genuine
measurement. The 34 insertions are partly a **scoring artifact**: the Whisper
English normalizer strips Devanagari combining vowel marks, which splits single
words into several tokens (`लिस्ट` → `ल सट`) and inflates the hypothesis word
count.

Scored as pure substitution — the most charitable possible treatment — Pulse Pro's
corpus WER would be 0.0849 rather than 0.1013. Both figures are above the 0.0627
it achieves on the unaffected rows, so the conclusion is unchanged, but **quote
"WER ≥ 1.0 on 16 of 200 rows" and the 0.1013 → 0.0627 contrast rather than the
individual per-row WERs of 1.36–2.33**, which are not clean measurements.

No post-processing was applied: transliterating or dropping the affected rows
would mean special-casing one provider, so the published headline figure stays
0.1013.

## Environment

- Endpoint `https://api.smallest.ai/waves/v1/stt/`, raw WAV bytes,
  `Content-Type: application/octet-stream`, Bearer auth — matching the documented
  pre-recorded contract (language and options as query parameters).
- Audio: Svarah (`ai4bharat/Svarah`) test split, revision `ebbf7777…`, seed 42,
  200-utterance frozen subset, decoded to 16 kHz mono PCM WAV.
- Eligible source run: `results/runs/smallest-pulse-pro-paced-clean/`
  (200/200 successes, zero failures, every row on first attempt).
- Comparison artifacts: `results/comparisons/v0823-eight-system/`.

## What would resolve it

No documented Pulse Pro option to force Latin-script output or disable
transliteration was found. Either an option to pin output script, or Pulse Pro
matching standard Pulse's Latin-script behavior when `language=en`, would resolve
this.
