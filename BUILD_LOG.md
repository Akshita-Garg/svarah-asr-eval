# Build Log — VoiceRefine Svarah ASR Evaluation

This document is the companion to the code. It exists so the evaluation can be
**rebuilt from scratch**, including the parts that a finished repository
normally hides. Every step carries one of two tags:

- **[SCRIPTED]** — captured in code, configuration, or a recorded command.
  Reproducing it means running the command or reading the file.
- **[MANUAL]** — requires human action that *cannot* be scripted away:
  obtaining API keys, downloading proprietary model artifacts, making judgement
  calls, or deciding trade-offs. These are the steps that vanish from a normal
  "finished code" handoff — they are captured here on purpose.

Each entry records **what**, **why**, and (where relevant) **what went wrong**.
Dead ends and corrected mistakes are kept rather than tidied away: several of
them (the Python 3.14 ABI failure, the Whisper Q4_K format incompatibility, the
CPU-contention latency distortion) are the most transferable findings here.

---

## Phase 0 — Orientation & environment probing

### 0.1 Read the spec, inventoried the repo  `[SCRIPTED]`

Starting state of the repo (from `git ls-files`):

- `DESIGN.md` — the full spec (this is the source of truth; the build follows it,
  it does not invent architecture).
- `.gitignore` — already ignores `.venv/`, `.env`, `cache/`, `data/prepared/`.
- `.python-version` → `3.14.3`
- `.venv/` — a pre-created virtual environment.

> **[MANUAL] equivalent:** read `DESIGN.md` end to end and treat the *Acceptance
> Criteria* section as both your build order and your definition of done. Don't
> start coding backends first — the spec mandates dataset inspection first.

### 0.2 Decided documentation is a first-class deliverable  `[SCRIPTED]`

Created this file up front and append to it as work happens, rather than
reconstructing at the end (reconstruction loses exactly the manual steps and
dead ends that matter most).

### 0.3 Probed Python 3.14 dependency compatibility  `[SCRIPTED]` — **key decision**

**Why this first:** the venv is on Python **3.14.3**, which is bleeding-edge.
The tightest technical risk in the whole project was whether the ASR runtimes
(`onnxruntime`, `sherpa-onnx`) even publish wheels for cp314. If they didn't,
we'd be *forced* to recreate the venv on Python 3.12. This is empirically
checkable, so it was checked rather than assumed.

**What went wrong (minor):** first `uv add` failed —
`error: No pyproject.toml found`. `uv add` needs project metadata to exist. Fixed
with `uv init --no-workspace --bare`, which created `pyproject.toml`.

**Commands run:**

```bash
uv init --no-workspace --bare
uv add datasets jiwer soundfile numpy
uv add onnxruntime sherpa-onnx
```

**Result — the good outcome:** everything installed with native cp314 wheels.
No Python downgrade needed.

| Package | Version |
| --- | --- |
| onnxruntime | 1.28.0 |
| sherpa-onnx | 1.13.4 |
| datasets | 5.0.1 |
| jiwer | 4.0.0 |
| soundfile | 0.14.0 |
| numpy | 2.5.2 |
| pandas | 3.0.5 (pulled in by datasets) |

> **Decision recorded:** stay on Python 3.14.3. Had a wheel been missing, the
> [MANUAL] step would have been: `uv python pin 3.12 && rm -rf .venv && uv sync`.

### 0.4 The one external dependency: model artifacts & keys  `[MANUAL]`

Three backends are specified. What gets *built* is independent of credentials;
what can actually *run today* is not. The design explicitly requires graceful
degradation ("a missing dependency or model for one backend must not prevent the
others"), so all three get built; unavailable ones skip cleanly.

| Backend | External requirement | Runnable without it? |
| --- | --- | --- |
| `voicerefine_whisper_tiny_int8` | nothing — model is a public sherpa-onnx download | **Yes** (the verifiable spine) |
| `voicerefine_parakeet_q4` | the internal Parakeet Q4 GGUF + CrispASR server binary | Only if those files exist on this machine |
| `elevenlabs_scribe_v2` | an ElevenLabs API key with Scribe v2 access, placed in `.env` | Only with a real key |

**Update after locating artifacts:** Parakeet Q4 and Whisper Tiny were confirmed
to run on-device in the desktop app, and all three local artifacts were located
in the sibling `voicerefine-desktop/` project:

| Artifact | Path (relative to `track-5-voicerefine/`) |
| --- | --- |
| Parakeet Q4 GGUF | `voicerefine-desktop/resources/models/parakeet-tdt-0.6b-v3-GGUF/parakeet-tdt-0.6b-v3-q4_k.gguf` |
| CrispASR server | `voicerefine-desktop/resources/bin/crispasr-windows-x86_64-cpu/crispasr-windows-x86_64-cpu/crispasr.exe` |
| Whisper Tiny INT8 | `voicerefine-desktop/resources/models/sherpa-onnx-whisper-tiny.en/` (encoder/decoder `.int8.onnx` + `tiny.en-tokens.txt`) |

> **[MANUAL] to locate these independently:** `find` the desktop project for `*.gguf`,
> `*.onnx`, and `crispasr*`. They live under `voicerefine-desktop/resources/`.

### 0.5 Extracted the desktop app's EXACT runtime config  `[SCRIPTED]` — **critical**

The design requires the local backends use "the same model artifacts and runtime
configuration as VoiceRefine Desktop." Guessing the config would silently make
the benchmark measure a *different* system than what ships. So I read the desktop
source of truth: `voicerefine-desktop/src/main/asr.js`. Extracted:

**Parakeet Q4 → CrispASR server** (`startCrispAsrServer`, `postToCrispServer`):
```
crispasr.exe --server --backend parakeet --model <gguf> --language en \
  --threads 8 --host 127.0.0.1 --port 51234 --no-prints --no-timestamps
```
- Readiness: poll a TCP connect to `127.0.0.1:51234` every 250 ms, 15 s timeout.
- Transcribe: multipart POST to `/v1/audio/transcriptions` (fallback `/inference`),
  form fields `file` (WAV) and `language=en`.
- Response: JSON `{ text | transcription | result }`, else plaintext.
- Env overrides used by the app: `VOICEREFINE_CRISPASR_BIN`,
  `VOICEREFINE_CRISPASR_PARAKEET_MODEL`, `VOICEREFINE_CRISPASR_PORT` (51234),
  `VOICEREFINE_CRISPASR_THREADS` (8).

**Whisper Tiny → sherpa-onnx** (`createWhisperTinyEnglishConfig`):
- `sampleRate 16000`, `featureDim 80`, encoder/decoder `*.int8.onnx`,
  `tokens tiny.en-tokens.txt`, `language en`, `task transcribe`,
  `tailPaddings -1`, `numThreads 4`, `provider cpu`.

Both line up exactly with the design's Systems-Under-Test table (8 threads /
4 threads, English, CPU). This is now replicated in the Python backends rather
than reinvented.

> **[MANUAL] equivalent:** open `voicerefine-desktop/src/main/asr.js` and read
> `startCrispAsrServer`, `postToCrispServer`, and `createWhisperTinyEnglishConfig`.
> The desktop app is the spec for "same runtime config."

### 0.6 Directory decision: keep the eval self-contained  `[SCRIPTED]` — **decision**

You asked whether to run this inside the parent `track-5-voicerefine` instead.
**Decision: no — keep `voicerefine-svarah-eval` as its own repo/project and
*reference* the desktop artifacts by configurable path** (defaulting to
`../voicerefine-desktop/resources/...`, overridable by the same env vars the app
uses). Rationale: the design mandates standalone reproducibility (own manifests,
own git provenance, "reproduce from README"); the parent repo is full of
`node_modules/` and packaged builds that would pollute that. The only thing the
parent had that I needed — how the app invokes the models — is already extracted
in 0.5. Reversible if you later disagree.

### 0.7 Live dataset probe: got the revision, hit the gate  `[SCRIPTED]` + `[MANUAL]`

Ran `scripts/probe_dataset.py`. Two results:

1. **Pinned revision resolved (unauthenticated metadata works):**
   `ai4bharat/Svarah` sha = `ebbf7777fe771490696a3f7b007097606fa8c924`
   (last modified 2025-03-10). This is the exact revision the manifest will pin.
2. **The dataset is GATED.** `load_dataset(...)` raised
   `DatasetNotFoundError: ... gated dataset ... You must be authenticated`.

> **[MANUAL] — blocks any live data run (two manual steps, cannot be scripted):**
> 1. Visit https://huggingface.co/datasets/ai4bharat/Svarah while logged in and
>    accept the access terms (gated datasets require a click-through).
> 2. Create a HF access token (Settings → Access Tokens) and put it in `.env` as
>    `HF_TOKEN=...`. The loader reads it; it is never written to any artifact.
>
> Also observed (non-blocking): a Windows symlink warning from `huggingface_hub`.
> Harmless (degraded cache, more disk). Silence with
> `HF_HUB_DISABLE_SYMLINKS_WARNING=1` or enable Windows Developer Mode.

### 0.8 Locked library choices, verified their APIs  `[SCRIPTED]`

Rather than reimplement normalization or edit distance (the design forbids both),
I verified the libraries the design points to actually behave as needed:

- **`whisper-normalizer`** ships Whisper's real `EnglishTextNormalizer`
  (`from whisper_normalizer.english import EnglishTextNormalizer`) — far lighter
  than installing `openai-whisper` (which drags in torch). Tests will lock its
  behavior.
- **`sherpa-onnx`** exposes `OfflineRecognizer.from_whisper(encoder, decoder,
  tokens, language, task, num_threads, decoding_method, provider, tail_paddings)`
  — a direct match to the desktop's `createWhisperTinyEnglishConfig`.
- **`jiwer` 4.0** for alignment/edit counts.

**Secrets summary — what YOU must place in `.env` (never committed):**

| Var | For | How to get it |
| --- | --- | --- |
| `HF_TOKEN` | load the gated Svarah dataset | HF Settings → Access Tokens (after accepting dataset terms) |
| `ELEVENLABS_API_KEY` | the cloud reference backend | ElevenLabs dashboard → API key |

---

## Phase 1 — Foundations (normalize, metrics, hashing, config)  `[SCRIPTED]`

Built the low-level, model-independent modules first because they are the
verifiable core and need no artifacts or dataset:

- `voicerefine_eval/normalize.py` — thin wrapper over the real Whisper
  `EnglishTextNormalizer`. Not reimplemented (design forbids it).
- `voicerefine_eval/metrics.py` — `jiwer`-based per-utterance edit counts +
  aggregation into mean WER, **corpus WER** (from aggregate edits — the primary
  figure), and RTF (mean and aggregate).
- `voicerefine_eval/hashing.py` — file/text SHA-256 + **atomic JSON writes**
  (temp file + `os.replace`) so an interrupted run never leaves partial JSON.
- `voicerefine_eval/config.py` — loads `config/eval.toml`, applies the same env
  overrides the desktop app uses, resolves artifact paths, and a tiny `.env`
  loader (no extra dependency).

**Unit tests (`tests/`) — acceptance criterion met.** 13 tests pass.

> **[MANUAL] pitfall #1 — pytest can't import the package.** With a flat,
> non-installed package, pytest doesn't put the repo root on `sys.path`. Fix:
> `[tool.pytest.ini_options] pythonpath = ["."]` in `pyproject.toml`.
>
> **[MANUAL] pitfall #2 — trust the real normalizer, not the assumption about
> it.** An early test asserted `"twenty percent" → "20 percent"`. The real
> Whisper normalizer produces `"20%"`. The test *should* lock actual behavior,
> so the assertion was corrected rather than the code. (This is exactly why the
> design says to test against Whisper's normalizer.)

## Phase 2 — Audio prep: a torch-free decode path  `[SCRIPTED]` — **decision**

`voicerefine_eval/audio.py` converts each utterance once to 16 kHz / mono /
signed-16 WAV. **Decision: decode the dataset's raw bytes with `soundfile` and
resample with `soxr`, rather than relying on the `datasets` library's audio
decoding.** Why: `datasets` 5.x can route audio decoding through
`torchcodec`/`torch`, a heavy and (on new Pythons) fragile dependency. Decoding
the bytes ourselves is small, deterministic, and backend-independent. Verified
the round-trip (read → mono → resample → PCM_16 WAV → re-read) on a real WAV.

## Phase 3 — Backend contract + Whisper Tiny  `[SCRIPTED]`

- `backends/base.py` — the `ASRBackend` ABC exactly as in the design, plus
  `is_available()` (graceful degradation) and `cache_signature()` (feeds the
  cache key).
- `backends/whisper_tiny.py` — sherpa-onnx `OfflineRecognizer.from_whisper`,
  configured identically to the desktop's `createWhisperTinyEnglishConfig`.

**Smoke-tested end-to-end on the desktop's own `test_wavs`** (no HF token
needed) — real transcripts, RTF < 1. The verifiable spine works.

## Phase 3.5 — The Python 3.14 → 3.12 downgrade  `[SCRIPTED]` — **major, forced**

This was the single most expensive diagnosis in the build.

**Symptom:** the Whisper backend crashed with
`The given version [27] is not supported, only version 1 to 10 is supported`.

**Diagnosis (the non-obvious part):**
1. Inspected the models with `onnx`: they are **IR version 7, opset 13** — old
   and universally supported. So the model is NOT the problem; "[27]" is a
   *garbage* value — the hallmark of an ABI mismatch, not a real version.
2. Ran the identical load in an **ephemeral Python 3.12** env
   (`uv run --no-project --python 3.12 --with sherpa-onnx ...`): **it worked.**
3. The tell: the 3.12 install pulled a separate native package
   **`sherpa-onnx-core` (15.7 MiB)**, which has **no Python 3.14 (cp314) wheel**.
   On 3.14 the pure-Python `sherpa-onnx` wheel installed *without* its native
   core, leaving a broken binary interface → the garbage version read.

**Conclusion:** `onnxruntime`/`sherpa-onnx` *install* on 3.14 but do not *run*.
Python 3.14 is too new for this ASR stack today. **Pinned the project to 3.12.**

**Commands run:**
```bash
# pyproject: requires-python = ">=3.12,<3.13"
uv python pin 3.12
rm -rf .venv uv.lock
uv sync
uv add sherpa-onnx-core     # NOT auto-pulled as a dependency; add it explicitly
```

> **[MANUAL] pitfall #3 (the expensive one):** the ASR wheels install cleanly on
> Python 3.14 but fail at *runtime* with a fake "version [27]" error. Don't chase
> the model — the model is fine (IR 7). The fix is Python **3.12**, and you must
> add **`sherpa-onnx-core`** explicitly (it is not declared as a dependency of
> `sherpa-onnx`, so a normal resolve silently omits the native library). After
> re-pinning, delete `uv.lock` so the resolution is redone for 3.12.

Re-verified after the downgrade: **13 tests pass**, Whisper backend transcribes
correctly.

## Phase 4 — Parakeet CrispASR-server backend  `[SCRIPTED]`

`backends/parakeet_crispasr.py` reproduces the desktop server lifecycle in
Python: spawn `crispasr.exe --server ...`, poll the TCP port for readiness,
multipart-POST each WAV to `/v1/audio/transcriptions` (fallback `/inference`),
parse the JSON transcript, and terminate the child on close. **Smoke-tested
against the real binary + GGUF** — server started in ~3 s, transcribed both
desktop test WAVs correctly, shut down cleanly.

## Phase 5 — ElevenLabs Scribe v2 backend  `[SCRIPTED]` (not yet run here)

`backends/elevenlabs.py` implements the batch STT call with the required retry
policy: 429 and 5xx retried with bounded exponential backoff + full jitter
(honoring `Retry-After`); other 4xx terminal. Key read from `ELEVENLABS_API_KEY`.

> **[MANUAL]:** put your key in `.env`, then verify the `model_id` (`scribe_v2`)
> and endpoint in `config/eval.toml` against current ElevenLabs docs — the exact
> Scribe model id/URL can change and I could not call the API without a key.

**Update (key provided):** verified with a 1-utterance call — the configured
`model_id="scribe_v2"`, `language_code="eng"`, and endpoint all work as-is (WER
0.033, API RTF 0.13). No config change needed. Full 3-backend run followed —
see Phase 13.

## Phase 6 — Caching  `[SCRIPTED]`

`cache.py` keys each entry on `{backend cache_signature + eval_id + audio hash}`.
The signature includes local model/binary hashes, so swapping a model or audio
file auto-invalidates. Writes are atomic; failures are never cached as success;
`--no-cache` bypasses. (Verified live — see Phase 8.)

## Phase 7 — Orchestrator + outputs  `[SCRIPTED]`

`run.py` (CLI: `--debug`, `--no-cache`, `--backends`, `--limit`,
`--resample-subset`) wires it together: build/read the frozen manifest → prepare
identical WAVs (loads the dataset only if some are missing) → for each available
backend: time startup separately, run one unscored warm-up (local), then
transcribe→cache→normalize→score every utterance → aggregate. `report.py` writes
`per_utterance.csv` + `summary.md` and prints the worst-10; `manifest.py` writes
`run_manifest.json`. Unavailable backends skip cleanly.

## Phase 8 — End-to-end integration test (no dataset needed)  `[SCRIPTED]`

`scripts/integration_smoke.py` stages a tiny manifest + prepared WAVs from the
desktop `test_wavs`, then runs the orchestrator **twice** with both local
backends. Result:

- **Run 1 (fresh):** both backends transcribed, WER/RTF computed, all three
  outputs written, worst-10 printed, shared-success = 2.
- **Run 2 (cache):** every utterance was a **cache hit — no retranscription**.

Inspected the outputs: `run_manifest.json` carries full provenance and **no
secrets**; `summary.md` leads with the Svarah accent caveat and both comparison
views; `per_utterance.csv` retains raw and normalized text. Staged files were
then removed.

> **[MANUAL] pitfall #4:** a loose script under `scripts/` puts `scripts/` on
> `sys.path`, not the repo root, so `import voicerefine_eval` fails. Fix in the
> script: `sys.path.insert(0, <repo root>)`.

## Phase 9 — Repro docs  `[SCRIPTED]`

`README.md` (reproduce-from-scratch guide + troubleshooting), `.env.example`
(the two secrets + optional overrides), and this log. Full suite: **19 tests
pass**.

---

## Mid-build status: complete vs. externally blocked

> ⚠️ **Mid-build snapshot, SUPERSEDED by Phases 10–12 below.** The `HF_TOKEN` is
> now provided, the schema probe has run, and the real 20-utterance debug run is
> done. The authoritative status is the acceptance-criteria table at the very end.

**Done and verified on this machine (no secrets needed):**
- Both local backends (`whisper_tiny_int8`, `parakeet_q4`) transcribe real audio.
- Normalization, metrics, caching — unit-tested (19 tests) and exercised live.
- Full pipeline runs end-to-end; second run hits cache; outputs are correct and
  secret-free.

**Blocked on external credentials and access (cannot be scripted):**
1. **HuggingFace:** accept Svarah's terms + put `HF_TOKEN` in `.env`. Without it
   the dataset (hence the real 200-utterance run) cannot load.
2. **ElevenLabs:** put `ELEVENLABS_API_KEY` in `.env` and confirm the Scribe v2
   `model_id`, endpoint, **and `language_code`** in `config/eval.toml` against
   current ElevenLabs docs (the code format may be `"en"` rather than `"eng"`).
   Until then that backend skips (by design).
3. **Run the schema probe FIRST** once `HF_TOKEN` is set:
   `uv run python scripts/probe_dataset.py`. This is the design's first gate and
   it has NOT actually run yet — the earlier attempt printed only the revision
   before hitting the gated-access error, so it never showed the feature names.
   The transcript/id/duration field names in `dataset.py` are currently
   *defensive guesses* from a candidate list, validated at runtime. If Svarah's
   transcript column isn't in `_TRANSCRIPT_CANDIDATES`, `build_manifest` will
   raise on the first real run — that's a one-line fix to the candidate list, not
   a mystery. Confirm the real column with the probe before the full run.
4. **The real runs:** `uv run python -m voicerefine_eval.run --debug` (20 utt,
   satisfies the "both local backends" criterion on real Svarah data), then the
   full `uv run python -m voicerefine_eval.run` (200 utt, all three backends).

## Phase 10 — Review pass: two real bugs the tests hadn't caught  `[SCRIPTED]`

A dedicated review pass over the finished code found two issues that green tests
had missed:

1. **RTF was corrupted by cache hits.** A cache hit has no timing
   (`inference_seconds = None`), and the aggregator was coercing that to `0.0` —
   counting cached utterances as *infinitely fast* and pulling RTF toward zero.
   Because the README says to run `--debug` first, the full 200-run would
   cache-hit those 20 and report RTF diluted by 20 zeros. **Fix:** exclude
   `None`-timed utterances from RTF (they still count for WER) and report the RTF
   sample size (`RTF n`) so a cached run honestly shows RTF over fewer utterances.
2. **The cache ignored its own `root`.** `CacheKey.path()` hardcoded the global
   `CACHE_DIR`, so `TranscriptCache(root=...)` didn't actually relocate the cache
   (tests weren't isolated; they wrote into the real `cache/`). **Fix:** path
   construction moved into `TranscriptCache._path()` using `self.root`.

Added a regression test for each. **19 tests pass.**

> **[MANUAL] lesson:** passing tests ≠ correct — they only check what you thought
> to assert. An adversarial review of the *aggregation* (not the per-row output,
> which was already correct) is what surfaced the RTF dilution.

## Phase 11 — Schema gate satisfied with a real token  `[SCRIPTED]` + `[MANUAL]`

You created `HF_TOKEN` and accepted the dataset terms; I re-ran the probe (after
making it load `.env` and decode via the pipeline's own path). The design's first
gate is now genuinely met — schema **and** one decoded example:

**Live Svarah `test` schema (revision `ebbf7777…`):**

| Column | Type | Role |
| --- | --- | --- |
| `audio_filepath` | `Audio` | audio (detected by type, not name) |
| `text` | string | **transcript / reference** |
| `duration` | float64 | audio length |
| `gender`, `age-group`, `primary_language`, `native_place_state`, `native_place_district`, `highest_qualification`, `job_category`, `occupation_domain` | string | speaker/accent metadata |

My defensive field guesses matched: `text` was already the first transcript
candidate, `duration` a duration candidate, `audio_filepath` an id candidate. No
code change needed.

**One decoded example (proves the torch-free path on real audio):** transcript
_"Our district Hasana was the land of the great Hoysala dynasty…"_ (speaker: L1
Kannada, Karnataka) decoded to 16 kHz, 140152 samples, 8.76 s — matching the
`duration` field. Audio ships as **embedded bytes** (already 16 kHz), so
`soundfile` reads it directly.

> **[MANUAL] pitfall #5 — torchcodec.** `datasets` 5.x refuses to *decode* audio
> without `torchcodec` installed (`ImportError: please install 'torchcodec'`).
> The pipeline avoids this by casting the audio column to `Audio(decode=False)`
> and decoding the raw bytes with `soundfile`. If you ever iterate the dataset
> with default decoding you'll hit this error — don't install torchcodec, use
> `decode=False`.

**Acceptance-criteria status:**

| Criterion | Status |
| --- | --- |
| Schema + one example inspected before backend work | ✅ ran `scripts/probe_dataset.py` with the token — see Phase 11 |
| Committed manifest → reproducible subset | ⏳ loader written; `data/subset_manifest.json` is created on the first tokened run, then commit it |
| All backends get identical 16 kHz mono WAV | ✅ |
| 20-utterance run with both local backends | ✅ proven on staged audio; rerun on real data after HF token |
| Normalization + metric unit tests pass | ✅ (19 tests) |
| Second run shows cache hits | ✅ (Phase 8) |
| Failures visible in coverage | ✅ |
| Final 3-backend report on 200 utterances | ⏳ needs HF token + ElevenLabs key |
| Reproduce from README without editing source | ✅ |

## Phase 12 — Real 20-utterance debug run on Svarah  `[SCRIPTED]`

Ran `uv run python -m voicerefine_eval.run --debug` on the actual (tokened)
dataset. It downloaded the split (~1.1 GB, cached once), built the committed
`data/subset_manifest.json` (200-utt subset from **6,656** rows, seed 42),
prepared 20 WAVs, warmed up, and ran both local backends. **Both 100% coverage.**

| Backend | Corpus WER | Mean WER | Agg RTF | Success |
| --- | --- | --- | --- | --- |
| `voicerefine_whisper_tiny_int8` | 0.1148 | 0.1298 | 0.197 (20 timed) | 20/20 |
| `voicerefine_parakeet_q4` | **0.0813** | 0.1614 | 0.516 (20 timed) | 20/20 |

Real product signal: Parakeet is **more accurate** (corpus WER 0.081 vs 0.115)
but **~2.6× slower** (RTF 0.52 vs 0.20). Worst utterances are genuine Indian-
English accent cases ("magy shakarati", "tulisi", esic→exic) — exactly what
Svarah is meant to probe. RTF reads "over 20 timed" (all fresh — the Phase 10 fix
holds).

### Bug found while verifying the manifest  `[SCRIPTED]`

`data/subset_manifest.json` came out **136 MB**. Cause: the id-field detector
resolved to `audio_filepath`, which (with `decode=False`) is the audio STRUCT
`{bytes, path}` — so `str(row[id_field])` embedded the raw audio bytes of all 200
rows into the manifest. **Fix:** `_clean_source_id()` now uses only the struct's
`path` (the filename) when the id value is an audio struct. Manifest dropped
**136 MB → 51 KB**; `source_id` is now e.g. `281474976887664_f1766_chunk_0.wav`.

> **[MANUAL] pitfall #6:** a Hugging Face `Audio` column read with `decode=False`
> is a `{bytes, path}` dict, not a filename string. Never stringify it into an
> artifact — pull `.path`. (Surfaced only by checking the manifest's file size;
> the run itself succeeded and hid it.)

---

## AUTHORITATIVE STATUS (supersedes all earlier snapshots)

| Acceptance criterion | Status |
| --- | --- |
| Schema + one decoded example inspected before backend work | ✅ Phase 11 |
| Committed manifest → reproducible subset | ✅ `data/subset_manifest.json` (200 utt, 51 KB) — commit it |
| All backends get identical 16 kHz mono WAV | ✅ |
| 20-utterance run with both local backends | ✅ **on real Svarah data** (Phase 12) |
| Normalization + metric unit tests pass | ✅ 19 tests |
| Second run shows cache hits | ✅ Phase 8 |
| Failures visible in coverage | ✅ |
| Final 3-backend report on 200 utterances | ✅ **complete** (Phase 13) |
| Reproduce from README without editing source | ✅ |

**All acceptance criteria met.** The remaining action is yours and optional:
commit `data/subset_manifest.json` (the frozen subset) and, if you want the
report in Git, `results/`. Everything else — code, tests, docs, and a full
verified run — is done.

## Phase 13 — Final 3-backend run on 200 utterances  `[SCRIPTED]`

You added `ELEVENLABS_API_KEY`. I verified it on one utterance (config correct as
shipped: `scribe_v2` / `language_code=eng` / endpoint), then ran all three
backends over the full 200-utterance subset. **200/200 success each, 0 failures,
RTF freshly timed over all 200.**

### The crash that had to be fixed first  `[SCRIPTED]`

The first full run **crashed** — but not where you'd guess. All 600 transcriptions
(200 × 3) *succeeded and cached*; the process then died in
`print_worst_utterances` with `UnicodeEncodeError: 'charmap' codec can't encode…`.
The Windows console is **cp1252**, and Svarah transcripts contain Indian-language
characters it can't encode. The crash happened *before* the output files were
written, so the whole run's results were lost to a printing bug.

**Fix:** `_force_utf8_console()` reconfigures stdout/stderr to UTF-8
(`errors="replace"`) at startup. Validated on a 2-utterance all-backend run, then
re-ran the full 200 fresh (cleared cache so RTF is measured over all 200, not
diluted by cache hits — the Phase 10 concern).

> **[MANUAL] pitfall #7 — Windows console encoding.** On Windows, printing
> non-Latin text (any Indian-language transcript) raises `UnicodeEncodeError` and
> can kill a long run *after* all the expensive work is done. Force UTF-8 on
> stdout/stderr. (File writes were already UTF-8, so only console prints were at
> risk.) Also note: a backgrounded `cmd | grep` reports **grep's** exit code — the
> first run showed "exit 0" while Python had actually crashed; check the output,
> not just the code.

### Final results (corpus WER is the primary figure)

| Backend | Corpus WER | Mean WER | Median WER | Agg RTF | Coverage |
| --- | --- | --- | --- | --- | --- |
| `voicerefine_whisper_tiny_int8` | 0.1639 | 0.2775 | 0.1026 | 0.166 | 200/200 |
| `voicerefine_parakeet_q4` | 0.0834 | 0.1932 | 0.0000 | 0.511 | 200/200 |
| `elevenlabs_scribe_v2` | 0.0752 | 0.1809 | 0.0000 | 0.254 | 200/200 |

**What this answers (the two product questions in DESIGN.md):**
1. **On-device default:** **Parakeet Q4** — corpus WER 0.083, essentially matching
   the cloud (0.075) and half the error of Whisper Tiny (0.164), while still
   comfortably real-time (aggregate RTF 0.51). Whisper Tiny is ~3× faster
   (RTF 0.17) but noticeably less accurate — a fallback for latency-critical use.
2. **Gap to cloud:** small on accuracy. On-device Parakeet (0.083) is within ~0.008
   corpus WER of ElevenLabs Scribe v2 (0.075). (ElevenLabs RTF 0.25 is API
   end-to-end latency, not compute time, and is network-dependent.)

Note mean ≫ corpus WER for every backend: short utterances (e.g. a 1-word "time"
→ "high" = WER 1.0) inflate the mean, which is exactly why the design designates
**corpus WER** the primary benchmark. All three get a **median WER at/near 0** —
half the utterances are transcribed essentially perfectly.

Outputs: `results/summary.md`, `results/per_utterance.csv` (200×3 rows),
`results/run_manifest.json` (all 3 backends, model/binary hashes, **no secrets** —
scanned). **The evaluation is complete.**

---

# Controlled Model Evaluation Extension

This section begins the next part of the evaluation. The original Phase 13
results are preserved unchanged under `results/runs/baseline-3-models/` and in
Git commit `7936ca8`. ElevenLabs will not be called again.

## Phase 14 - Experiment redesign and artifact preservation  `[SCRIPTED]`

### Question this extension answers

> How do the three on-device ASR model architectures compare under controlled
> execution, before VoiceRefine's application-specific runtime choices are
> reconsidered?

The controlled local protocol is:

- CrispASR v0.6.11, CPU-only build, for Whisper, Parakeet, and Cohere.
- Persistent HTTP server for every local model.
- Eight CPU threads for every local model.
- Q4_K quantization for every local model, subject to Whisper compatibility
  validation before the full run.
- Identical prepared 16 kHz mono PCM16 WAV files.
- One unscored warm-up request after model loading.
- Startup measured separately; each scored timing wraps the same WAV-to-text
  backend call.
- No VAD, timestamps, punctuation model, or application cleanup.

The final five-system report will contain:

1. New controlled Whisper Tiny Q4_K + CrispASR results.
2. Existing Parakeet Q4_K + CrispASR results from Phase 13.
3. New controlled Cohere Transcribe Q4_K + CrispASR results.
4. Existing ElevenLabs Scribe v2 results from Phase 13.
5. New Sarvam Saaras v4 API results (`en-IN`).

The old Sherpa Whisper INT8 rows remain preserved as a product-runtime baseline,
but they are not part of the final controlled five-system table.

### Why eight threads

The evaluation machine reports eight logical CPUs. The original code used four
threads for Sherpa and eight for CrispASR, but Git history contains no benchmark
that justified either default. This extension fixes all three local backends at
eight threads so CPU concurrency is controlled. Thread scaling (for example,
four versus eight) is deliberately deferred to a separate experiment.

### Safe run storage

The old harness overwrote `results/per_utterance.csv`, `summary.md`, and
`run_manifest.json` on each invocation. Commit `34bab36` added:

- `--output-dir` so every run has immutable artifacts.
- `voicerefine_eval.merge`, which validates dataset identity, the exact 200
  utterance IDs, references, and duplicate keys before combining stored rows.
- A comparison manifest containing hashes of every source artifact.

This lets the three new systems be evaluated without invoking Whisper-Sherpa,
Parakeet, or ElevenLabs again.

### Backends prepared so far

- Generalized the persistent CrispASR adapter so the same server code can run
  `whisper`, `parakeet`, or `cohere`.
- Added explicit provenance for runtime mode, threads, CPU backend,
  quantization, model hash, and executable hash.
- Added a Sarvam adapter that reads `SARVAM_API_KEY`, sends the shared WAV,
  retries network/429/5xx failures, and never stores the key.
- Added mocked backend and merge tests. Current status: **28 tests pass**.

### Whisper source artifact

Downloaded the official Whisper.cpp `ggml-tiny.en.bin` source into the
gitignored evaluation `models/` directory:

- Size: 77,704,715 bytes.
- SHA-256: `921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f`.
- Source has not yet been quantized.
- No new ASR evaluation or Sarvam API request has run yet.

The next gate is intentionally small: attempt Q4_K conversion with the bundled
`crispasr-quantize.exe`, verify that CrispASR loads the output, then transcribe
five recordings before authorizing the 200-recording run.

## Phase 15 - Whisper Q4_K compatibility gate  `[SCRIPTED]`

The bundled quantizer was invoked with the official Whisper.cpp source and
`q4_k`. It failed immediately with:

```text
gguf_init_from_file_ptr: invalid magic characters: 'lmgg', expected 'GGUF'
failed to quantize model
```

This confirms that `crispasr-quantize.exe` accepts true GGUF containers, while
the official Whisper.cpp `ggml-tiny.en.bin` uses Whisper's older custom GGML
container. No Q4_K output was created, no transcription ran, and no API credit
was used. The experiment is paused at this decision gate rather than silently
switching to an unofficial model or a different quantization.

## Phase 16 - Official Whisper Q4 compatibility test  `[SCRIPTED]`

### Tooling decision

Instead of using an unofficial pre-quantized model, the official Whisper.cpp
Windows tools were pinned at release `v1.9.2` and downloaded from its GitHub
release. The `whisper-bin-x64.zip` archive matched GitHub's published SHA-256:

```text
49dcc16de826f20bd53d44f947a1ae49dfa81f86cad67a64d80820cb192d674a
```

The release's `whisper-quantize.exe` understands Whisper's legacy GGML model
container and supports both Q4_K and Q4_0. Both test artifacts were generated
from the verified `ggml-tiny.en.bin` source; no third-party model weights were
used.

### Q4_K result: conversion succeeds, CrispASR load fails

Whisper.cpp successfully generated:

```text
ggml-tiny.en-q4_k.bin
size: 25,335,371 bytes
sha256: a4e2d60026277fb6eb5fb77964bf74bd3141b966d47a55a53a482b97df64d2b3
```

CrispASR 0.6.11 read the model metadata but rejected a quantized tensor with:

```text
tensor 'decoder.token_embedding.weight' has wrong size in model file
```

The five-file gate therefore skipped before inference. Q4_K is not a usable
Whisper format with the shipping CrispASR binary and is excluded from the
controlled run.

### Q4_0 result: compatible

The same official quantizer generated:

```text
ggml-tiny.en-q4_0.bin
size: 25,335,371 bytes
sha256: 3653b98189ab4dab967110b355a863722213dfac63ca76691b149f79d7b33831
```

CrispASR loaded and transcribed with this artifact. During the diagnostic run,
it also revealed that Whisper automatically enables an external punctuation
restoration model. The controlled protocol explicitly excludes post-processing,
so the shared server adapter now sends `--no-punctuation` and records
`no_punctuation: true` in its cache signature and run manifest.

The corrected five-recording gate used a persistent CrispASR server, CPU-only,
eight threads, one unscored warm-up, and fresh inference:

- Startup: 0.522 seconds.
- Successes: 5 of 5.
- Corpus WER: 0.0645.
- Aggregate RTF: 0.133.

These numbers only establish compatibility and end-to-end instrumentation; the
sample is too small for model comparison. The controlled Whisper configuration
is now Q4_0, while Parakeet and Cohere remain Q4_K. All 28 harness tests pass.

## Phase 17 - Latest-runtime Q4_K protocol correction  `[SCRIPTED]`

The controlled protocol was updated at the user's direction: all three local
systems must use the latest CrispASR release and Q4_K quantization. The official
Windows x86-64 CPU build of **CrispASR v0.8.23** (git `7d22deec`, release date
2026-07-26) was downloaded and its published archive checksum was verified:

```text
archive sha256: 43451e16c7ba3617beb41747bd857bebd0c4ed6c2af918da98c8280daf167d8e
crispasr.exe sha256: 7276a38caab4d8440263d4e808b2adcc785596ffb2a5998cef9e28ec22fb389e
```

### Why Whisper Tiny was replaced by Whisper Base English

The Tiny Q4_K failure is not specific to CrispASR v0.6.11. The same official
Tiny artifact fails in v0.8.23 because Tiny's transformer width is 384, while
Q4_K requires quantized rows divisible by its 256-value super-block. CrispASR's
legacy Whisper loader stores one global tensor type, so it cannot mix Q4_K with
a fallback type for the incompatible tensors. A public Tiny Q4_K artifact was
also byte-identical to our official conversion and failed in the same place.

Whisper Base has width 512 and is therefore the smallest standard Whisper model
that can be represented validly as Q4_K in this loader. The official
`ggml-base.en.bin` source was pinned from Whisper.cpp revision
`5359861c739e955e79d9a303bcbc70fb988958b1`, then quantized with the official
Whisper.cpp v1.9.2 tool:

```text
source size: 147,964,211 bytes
source sha256: a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002
Q4_K size: 46,484,531 bytes
Q4_K sha256: e2be598b0a063bc356c122fdd1a55ab97225c9918f98693d336ef3a690342911
```

CrispASR v0.8.23 successfully loaded the resulting model and transcribed a
14.6-second prepared Svarah recording in 2.80 seconds using the CPU build and
eight threads. The final local comparison is consequently:

1. Whisper **Base English** Q4_K, not Tiny.
2. Parakeet TDT 0.6B v3 Q4_K.
3. Cohere Transcribe Q4_K.

All three use the same CrispASR v0.8.23 executable, persistent server protocol,
CPU backend, eight threads, prepared audio, warm-up rule, and timing boundary.
Whisper uses its native punctuation path in v0.8.23; Cohere's optional external
punctuation restoration is disabled. Evaluation normalization still removes
punctuation and casing before WER scoring, so punctuation style cannot change
the accuracy comparison.

The merger now supports selecting backend IDs from source runs. This is needed
to reuse only the original ElevenLabs rows from the three-backend baseline while
excluding its obsolete local rows. No ElevenLabs request will be repeated.

## Phase 18 - Five-recording integration gates  `[SCRIPTED]`

Every candidate was run through the complete harness on the same first five
frozen subset recordings. Each local gate used CrispASR v0.8.23, CPU, eight
threads, a persistent server, one unscored warm-up, and fresh timed inference.

| Backend | Success | Corpus WER | Aggregate RTF | Startup |
| --- | ---: | ---: | ---: | ---: |
| Whisper Base English Q4_K | 5/5 | 0.0430 | 0.396 | 0.53 s |
| Parakeet Q4_K | 5/5 | 0.0323 | 0.573 | 6.33 s |
| Cohere Q4_K | 5/5 | 0.0538 | 2.043 | 11.85 s |
| Sarvam Saaras v4 | 5/5 | 0.0215 | 0.071 | 0.00 s |

These five-item values are compatibility checks, not model conclusions. In
particular, a one-word file makes per-file means unstable; the full 200-item
aggregate and shared-subset results remain the decision data.

### Sarvam configuration defect found by the gate

The first Sarvam gate returned five HTTP 400 responses. The configuration
loader had treated every setting named `model` as a local path, changing the
API identifier `saaras:v4` into an absolute Windows path. Path resolution is now
backend-type-aware: `model` is resolved only for CrispASR, while cloud model IDs
remain literal. A regression test covers this distinction. The corrected gate
passed 5/5; no failed gate row will be used in the final report.

Harness status after the fix: **31 tests pass**.

## Phase 19 - Full controlled local runs  `[SCRIPTED]`

### Whisper Base English Q4_K

The fresh 200-recording run completed with 200 unique utterance rows, no cache
hits, and no failures:

- Corpus WER: **0.1143**.
- Mean WER: **0.1850**.
- Aggregate RTF: **0.298** over all 200 timed calls.
- Startup: **0.53 seconds**.
- Runtime/model hashes match the Phase 17 provenance.

Output: `results/runs/v0823-whisper-base-en-q4k/`.

### Parakeet Q4_K

The fresh 200-recording run completed with 200 unique utterance rows, no cache
hits, and no failures:

- Corpus WER: **0.0829**.
- Mean WER: **0.1884**.
- Aggregate RTF: **0.439** over all 200 timed calls.
- Startup: **8.84 seconds**.

Output: `results/runs/v0823-parakeet-q4k/`.

### Cohere Q4_K

The fresh 200-recording run completed with 200 unique utterance rows, no cache
hits, and no failures:

- Corpus WER: **0.0723**.
- Mean WER: **0.1766**.
- Aggregate RTF: **1.837** over all 200 timed calls.
- Startup: **13.89 seconds**.

Output: `results/runs/v0823-cohere-q4k/`.

This is the best local corpus WER in the controlled run, but an aggregate RTF
above 1 means inference took longer than the input audio on this CPU. That makes
the result an accuracy/latency tradeoff rather than an unqualified winner.

## Phase 20 - Full Sarvam run  `[SCRIPTED]`

Sarvam Saaras v4 completed all 200 fresh API calls with no cache hits and no
failures:

- Corpus WER: **0.0386**.
- Mean WER: **0.0705**.
- Aggregate end-to-end API RTF: **0.122** over all 200 calls.
- Startup: effectively **0 seconds** (no local model load).

Output: `results/runs/sarvam-saaras-v4/`.

Sarvam's RTF measures upload, network, hosted inference, and response download.
It represents observed user wait time on this run, but it is not a controlled
measure of the provider's model compute and should not be interpreted as such.

## Phase 21 - Archived baseline and final five-system comparison  `[SCRIPTED]`

The original three-system artifacts were copied unchanged to
`results/archive/pre-v0823-baseline-3-models/`. Their hashes remain identical to
the committed baseline. The final merger selected only the preserved
`elevenlabs_scribe_v2` rows from that archive, excluding its obsolete local
rows, and combined them with the four new runs.

Integrity checks passed:

- Five backend IDs and **1,000 total rows**.
- Exactly **200 unique utterances per backend**.
- The same dataset revision, split, seed, subset size, utterance IDs, and raw
  references in every source run.
- **200 shared successful utterances**, 100% coverage, and zero failures for
  every backend.
- Source artifact SHA-256 hashes recorded with repository-relative paths in
  `comparison_manifest.json`.
- No ElevenLabs API call was made.

Final artifacts:

- `results/comparisons/v0823-five-system/summary.md` - generated metric tables.
- `results/comparisons/v0823-five-system/per_utterance.csv` - all 1,000 rows.
- `results/comparisons/v0823-five-system/comparison_manifest.json` - provenance.
- `results/comparisons/v0823-five-system/interpretation.md` - conclusions and
  product implications.

## Phase 22 - Whisper Medium and six-system comparison  `[SCRIPTED]`

Whisper Medium English was added to answer a specific fairness question: was
Whisper Base's weaker accuracy mainly a consequence of comparing a 74M-parameter
model with substantially larger local models? Medium has about 769M parameters,
making it a closer capacity comparison for Parakeet while retaining Whisper's
English-only model family.

### Controlled configuration

The new backend changes model capacity while preserving the existing local
protocol:

- Whisper Medium English, Q4_K.
- CrispASR v0.8.23 (`7d22deec`), persistent server mode.
- Windows x86-64 CPU build, eight threads, CPU backend.
- Native Whisper punctuation path, matching the Base run.
- Identical prepared WAVs, one unscored warm-up, and the same timed HTTP request
  boundary.

The official `ggml-medium.en.bin` source was downloaded from the same pinned
Whisper.cpp model revision used for Base and verified before conversion:

```text
source size: 1,533,774,781 bytes
source sha256: cc37e93478338ec7700281a7ac30a10128929eb8f427dda2e865faa8f6da4356
```

The pinned Whisper.cpp v1.9.2 quantizer produced:

```text
Q4_K size: 444,506,557 bytes
Q4_K sha256: b742cebba0cf9a21b69e7fb99ccb7cc175003123f9f0f468295d8dd6d7e86bbc
quantization time: 202.49 seconds
```

Download and quantization are one-time artifact preparation and are excluded
from startup and RTF.

### Compatibility gate

The five-recording gate passed with 5/5 successes, corpus WER 0.0323, aggregate
RTF 1.933, and 1.90-second startup. This established compatibility only; it was
not used for the final model comparison.

### Full 200-recording run

The fresh run completed with no cache hits and no failures:

- Corpus WER: **0.0728** (151 errors across 2,074 reference words).
- Mean WER: **0.1419**.
- Aggregate RTF: **3.751**.
- Total measured inference: **3,731.76 seconds** over 994.86 seconds of audio.
- Median request latency: **15.60 seconds**.
- Startup: **1.78 seconds**.

An observed live working-set snapshot was about 1.07 GB. This is context, not a
formal peak-RAM benchmark.

### Six-system merge and interpretation

The final merge rebuilt the comparison from six immutable source runs. Only the
stored ElevenLabs rows were selected from the archived baseline; no cloud API
was called.

Integrity checks passed:

- Six backend IDs and **1,200 total rows**.
- Exactly 200 unique utterances per backend.
- Identical dataset revision, subset, utterance IDs, and raw references.
- 200 shared successful utterances, 100% coverage, and zero failures.
- All 200 Medium rows were fresh rather than cache hits.

Medium reduced Base's corpus WER by about 36%, confirming that Whisper capacity
was an important variable. It was about 12.6 times slower than Base, 8.5 times
slower than Parakeet, and 2.0 times slower than Cohere. Medium and Cohere were
effectively tied on corpus accuracy: Cohere made 150 total word errors and
Medium made 151. Medium had lower utterance WER on 35 files, Cohere on 32, and
133 tied; no statistical significance test was run.

Final artifacts:

- `results/runs/v0823-whisper-medium-en-q4k/` - standalone full run.
- `results/gates/whisper-medium-en-q4k-5/` - compatibility gate.
- `results/comparisons/v0823-six-system/summary.md` - generated metrics.
- `results/comparisons/v0823-six-system/per_utterance.csv` - all 1,200 rows.
- `results/comparisons/v0823-six-system/comparison_manifest.json` - provenance.
- `results/comparisons/v0823-six-system/interpretation.md` - conclusions.

## Phase 23 - Smallest.ai Pulse Pro evaluation  `[SCRIPTED]`

At the user's direction, Whisper Medium was set aside and Smallest.ai Pulse Pro
was added to the original five-system comparison. Pulse Pro was selected because
Smallest.ai documents it as the accuracy-oriented English model for prerecorded
HTTP transcription. The adapter follows the documented request contract:

- `POST https://api.smallest.ai/waves/v1/stt/`.
- Query parameters `model=pulse-pro` and `language=en`.
- Raw prepared WAV bytes with `Content-Type: application/octet-stream`.
- Bearer authentication from `SMALLEST_API_KEY`; the user's existing
  `SMALLESTAI_API_KEY` spelling is also accepted.
- Network, HTTP 429, and HTTP 5xx failures are retryable; authentication and
  other permanent HTTP failures are not.

No API key value is included in cache signatures, manifests, CSV files, or
logs. Request-shape, key-alias, key-precedence, retry, pacing, and attempt-count
behavior were covered with mocked tests.

### Integration gate

The five-recording gate in `results/gates/smallest-pulse-pro-5/` completed with
5/5 successes, corpus WER 0.0323, and aggregate RTF 0.071. This established API
compatibility only and is not used as the model result.

### Rate-limit diagnosis

The first unpaced 200-recording attempt in
`results/runs/smallest-pulse-pro/` produced 163 successes and 37 failures: 35
HTTP 429 responses and two network failures. This artifact is diagnostic and
is not eligible for the final comparison.

A resume facility was added so a validated prior run can retain successful rows
and retry only failed utterances. It verifies the dataset identity, selected
utterance IDs, references, backend signature, and source hashes. The completion
attempt in `results/runs/smallest-pulse-pro-complete/` reached 200/200, but one
request spent 1,089.85 seconds in network/retry waiting. Its aggregate RTF of
1.293 is contaminated and is also excluded from the final comparison.

The final strategy spaces request starts by four seconds. This coordination
wait occurs before the timed backend call, so it changes total batch wall-clock
duration but does not make individual API requests look faster. This matches
the experiment's existing definition of RTF as request-to-response latency.

### Clean final run

`results/runs/smallest-pulse-pro-paced-clean/` is the eligible source run:

- 200 successes, zero failures, and 100% coverage.
- Every row succeeded on attempt one; no retry/backoff wait is hidden in RTF.
- Corpus WER: **0.1013**.
- Mean WER: **0.2254**.
- Aggregate API-call RTF: **0.118**.
- Mean RTF: **0.353**.
- Total batch wall time: about 800 seconds, including untimed pacing.

Despite `language=en`, 16 outputs contained Devanagari characters. Those rows
produced 83 errors against 49 reference words, while the other 184 rows produced
127 errors against 2,025 reference words. The official result remains 0.1013;
selectively transliterating or removing those rows would unfairly post-process
one provider. No documented Pulse Pro option to force Latin script or disable
transliteration was found.

Of those 83 errors, 49 are substitutions (exactly the reference word count, so
every word on these rows is genuinely wrong) and 34 are insertions. The
insertions are partly a scoring artifact: the Whisper English normalizer strips
Devanagari combining vowel marks, splitting one word into several tokens
(`लिस्ट` -> `ल सट`). Scored as pure substitution, corpus WER would be 0.0849
instead of 0.1013 - still well above the 0.0627 Pulse Pro reaches on the other
184 rows. Cite the 0.1013 -> 0.0627 contrast rather than the per-row WERs.

The finding was later reproduced live and written up for the provider in
`results/artifact-reports/smallest-pulse-pro-script-issue.md`. Three controls
rule out a client-side cause: omitting `language` returns byte-identical
Devanagari; `language=hi` is rejected with HTTP 400 and
`"options": ["en"]`, proving the parameter is validated rather than ignored;
and the same bytes sent to `model=pulse` return clean Latin English. Successful
Pulse Pro responses echo `"language": "en"`.

### Current comparison

The merger combined the clean Smallest.ai run with the five previously accepted
systems. Integrity checks passed:

- Six backend IDs and **1,200 total rows**.
- Exactly 200 unique utterances per backend.
- Identical dataset revision, subset, utterance IDs, and raw references.
- 200 shared successes, 100% coverage, and zero failures.
- Source artifact hashes recorded in the comparison manifest.
- No ElevenLabs API call was made.

Current artifacts are under
`results/comparisons/v0823-six-system-smallest/`. Sarvam retains the best corpus
WER (0.0386). Smallest.ai has the lowest observed aggregate RTF (0.118), narrowly
ahead of Sarvam (0.122), but cloud runs happened at different times and that
latency ordering is not a controlled provider benchmark. Pulse Pro is not the
recommended VoiceRefine cloud option because Sarvam was much more accurate at
nearly the same observed latency and Pulse Pro showed inconsistent output
script.

Whisper Medium artifacts remain available as supplementary history under
`results/comparisons/v0823-six-system/`, but Medium is not part of the current
comparison while its latency is awaiting a quieter-system rerun.

## Phase 24 - Standard Smallest.ai Pulse evaluation  `[SCRIPTED]`

The user reproduced correct Latin-script English in Smallest.ai's playground
for a recording that Pulse Pro had returned in Devanagari through the API. A
controlled API diagnostic used `svarah_test_0048` and established:

- The original Svarah WAV and prepared evaluation WAV were byte-for-byte
  identical: 132,890 bytes with SHA-256
  `dde7874834c8f337ed865690c0cc37b1f485e23b8bb66a00595a304322ff8437`.
- Unified `model=pulse-pro` returned Devanagari with both
  `application/octet-stream` and `audio/wav` content types.
- Unified `model=pulse` returned correct Latin-script English.
- The legacy standard-Pulse endpoint also returned correct Latin-script English.

This ruled out local resampling, MIME labeling, shared normalization, and the
unified endpoint itself as sufficient explanations. The changed model selection
was the variable associated with the script change.

### Implementation and gate

A separate `smallest_pulse` config was added. It reuses the Smallest.ai adapter
and differs from Pulse Pro only in `model=pulse`; language, endpoint, request
bytes, pacing, retries, timeout, frozen subset, normalization, scoring, and
timing boundary remain the same. A config contract test protects the model ID.
All 40 tests passed.

The five-recording gate completed 5/5 with corpus WER 0.0323 and aggregate RTF
0.065. It established compatibility only and is not used as the final result.

### Clean 200-recording run

`results/runs/smallest-pulse-paced-clean/` completed in about 800 seconds:

- 200 successes, zero failures, 100% coverage, and no cache hits.
- Every request succeeded on attempt one.
- Corpus WER: **0.0752** (156 errors / 2,074 reference words).
- Mean WER: **0.1845**.
- Aggregate API-call RTF: **0.114**.
- Mean RTF: **0.378**.
- Median request latency: **0.557 seconds**.
- Zero Devanagari rows and zero other non-Latin rows.

The new merger output at
`results/comparisons/v0823-seven-system-smallest/` contains seven backends and
1,400 rows. It verified the same 200 utterance IDs and references for every
backend, full shared coverage, and zero failures. No existing backend was rerun.

Standard Pulse tied ElevenLabs' exact aggregate count of 156 word errors, came
within six errors of local Cohere, and was behind Sarvam's 80 errors. On the 16
rows where Pulse Pro used Devanagari, Pulse Pro made 83 errors while standard
Pulse made 18. Across all rows, standard Pulse had lower utterance WER on 39,
equal WER on 128, and higher WER on 33. This supports a positive standard Pulse
result alongside a narrowly framed, reproducible Pulse Pro script observation.

## Phase 25 - Whisper Medium lower-contention rerun  `[SCRIPTED]`

The first Whisper Medium run happened while the user was actively working on
the machine. At the user's direction, Medium was rerun when no other heavy
processes were expected. The old run was preserved unchanged, and the fresh run
was written to `results/runs/v0823-whisper-medium-en-q4k-quiet-rerun/`.

No execution controls changed: CrispASR 0.8.23, the same Medium English Q4_K
artifact, CPU backend, eight threads, persistent server mode, native Whisper
punctuation, one unscored warm-up, the same prepared WAVs, and the same timed
local HTTP request boundary. Caching was disabled.

### Result and repeatability

The lower-contention run completed 200/200 with no failures:

- Corpus WER: **0.0728** (151 errors / 2,074 reference words), unchanged.
- Aggregate RTF: **2.219**, down from 3.751.
- Total measured inference: **2,207.75 seconds**, down from 3,731.76 seconds.
- Wall-clock duration: about **37 minutes**, down from about 62 minutes.
- Median request latency: **10.75 seconds**, down from 15.60 seconds.
- P90: **13.26 seconds**, down from 28.88 seconds.
- P95: **13.74 seconds**, down from 36.78 seconds.
- Maximum: **27.38 seconds**, down from 57.49 seconds.
- 183 of 200 rows were faster in the lower-contention run.

All 200 raw transcripts and normalized transcripts were byte-for-byte identical
between the two runs. The 40.8% reduction in total measured inference therefore
reflects execution conditions rather than a quality change. Live samples showed
approximately 7.8-8 logical cores in use and a working set around 1.0 GB; these
are observational context rather than formal resource benchmarks.

### Eight-system merge

`results/comparisons/v0823-eight-system/` combines the lower-contention Medium
run with the accepted seven systems. The merger verified eight backend IDs,
1,600 rows, exactly 200 matching utterances per backend, 200 shared successes,
and zero failures. No other model or API was rerun.

Medium's 151 errors are effectively tied with Cohere's 150, but Medium remains
about 21% slower by aggregate RTF (2.219 versus 1.837). Both remain slower than
real time on this CPU. Parakeet therefore remains the strongest local default
balance among the controlled candidates.

## Phase 26 - Model provenance, size, and openness audit  `[SCRIPTED]`

The final summary and interpretation were extended with a source-backed model
profile for all eight evaluated systems. This audit deliberately separates
three concepts that are often all called "model size":

- Original parameter count.
- Exact evaluated checkpoint/artifact size.
- Runtime RAM or VRAM, which requires a separate resource benchmark.

Exact local artifact sizes were measured from the files used in the runs:

| Model | Parameters | Evaluated Q4_K artifact |
| --- | ---: | ---: |
| Whisper Base English | 74M | 46,484,531 bytes |
| Whisper Medium English | 769M | 444,506,557 bytes |
| Parakeet TDT 0.6B v3 | 600M | 488,674,176 bytes |
| Cohere Transcribe | 2B | 1,510,362,752 bytes |

Cloud checkpoints are not present on the evaluation machine, so no file size
or runtime-memory value was inferred for them. Official provider pages did not
disclose parameter counts or training corpora for ElevenLabs Scribe v2,
Smallest.ai Pulse, Smallest.ai Pulse Pro, or Sarvam Saaras v4; those fields are
recorded as unknown rather than filled with estimates.

The audit also resolved a naming misconception: the **Open ASR Leaderboard** is
an open benchmark, not a list containing only open models. Its interface has a
toggle for proprietary API systems, and its metadata labels ElevenLabs Scribe
v2 and Smallest.ai Pulse as proprietary. Whisper, Parakeet, and Cohere have
downloadable weights under MIT-repository, CC BY 4.0, and Apache 2.0 terms
respectively. The evaluated cloud services are treated as proprietary/API-only.

Training descriptions were taken only from official model cards or provider
material. Sarvam's published one-million-hour statement applies to Saaras v3;
it is retained only as predecessor context and is not attributed to the
evaluated v4 model. Research was checked on 2026-08-13, and source links are
kept next to the detailed claims in the final interpretation.


---

## Phase 27 - Publication pass  `[SCRIPTED]`

### Known debt: sherpa-onnx is now vestigial

`sherpa-onnx` and `sherpa-onnx-core` remain hard dependencies in
`pyproject.toml`, and `sherpa-onnx-core` has no wheels past Python 3.12. That is
the sole reason the project pins `requires-python = ">=3.12,<3.13"`.

Nothing in the published results uses them. The only sherpa-backed backend is
`voicerefine_whisper_tiny_int8`, which is not in the active set — Phase 17
replaced it with Whisper Base running under CrispASR. The four local backends in
the comparison are `crispasr_server` type: they drive a standalone binary over
HTTP and never import sherpa. The Python pin is therefore inherited from a
retained dependency, not a requirement of any system being measured.

**Why it was not removed.** Dropping the dependency would regenerate `uv.lock`,
after which the environment would no longer match the one recorded in every
committed `run_manifest.json`. Those manifests pin exact dependency versions and
are a large part of what makes the results checkable. Keeping the locked
environment that actually produced the numbers was judged more valuable than
lifting an interpreter ceiling that uv satisfies automatically.

The right moment to remove it is the next full re-run, when the manifests are
being regenerated anyway.

### Audio preparation ruled out as a cause of the Pulse Pro script output

A fair objection to the Pulse Pro finding is that this harness's own audio
conversion degraded the signal. `scripts/verify_audio_preparation.py` decodes
each utterance's native dataset audio and compares it with the prepared WAV that
was actually sent to the backends.

Svarah's test split is already 16 kHz mono, so the resampling branch never
executes: **0 of 200 utterances were resampled**. The largest per-sample
difference is 1.53e-05, half of one 16-bit quantization step (1/32767), i.e. the
float-to-PCM_16 rounding done when writing the file. That value is identical for
the 16 affected rows and the 184 unaffected ones, so nothing distinguishes the
affected audio. Standard Pulse returns clean Latin English from the
byte-identical file.
