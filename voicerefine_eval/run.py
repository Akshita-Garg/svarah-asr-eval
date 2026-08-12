"""Evaluation orchestrator and CLI entry point.

Ties the pieces together per DESIGN.md: build/read the subset manifest, prepare
identical WAVs, run each active backend over every utterance (with warm-up,
caching, and timing), score with the shared normalizer + jiwer, and write the
CSV / run manifest / summary outputs plus console progress and worst-10.

Usage:
    uv run python -m voicerefine_eval.run [--debug] [--no-cache]
        [--backends id1,id2] [--limit N] [--resample-subset]
        [--output-dir results/runs/my-run]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

from .audio import describe_prepared, prepare_audio, prepared_path_for
from .backends import build_backend
from .backends.base import ASRBackend, BackendUnavailableError, TranscriptionError
from .cache import TranscriptCache
from .config import REPO_ROOT, EvalConfig, load_config
from .dataset import (
    SubsetEntry,
    build_manifest,
    read_manifest,
    select_entries,
    validate_schema,
)
from .hashing import sha256_file, write_json_atomic
from .manifest import build_run_manifest
from .metrics import score_utterance
from .normalize import normalize_text
from .report import (
    print_worst_utterances,
    read_per_utterance_csv,
    shared_success_metrics,
    summarize_backend,
    write_per_utterance_csv,
    write_summary_md,
)
from .results import UtteranceResult


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _force_utf8_console() -> None:
    """Make stdout/stderr UTF-8 so printing non-Latin transcripts can't crash the run.

    Svarah transcripts contain Indian-language characters; the default Windows
    console encoding (cp1252) raises UnicodeEncodeError on them. ``errors="replace"``
    is a belt-and-suspenders fallback for any glyph even UTF-8 can't render.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VoiceRefine Svarah ASR evaluation")
    p.add_argument("--debug", action="store_true", help="Use the small debug subset.")
    p.add_argument("--no-cache", action="store_true", help="Force fresh transcription.")
    p.add_argument("--backends", type=str, default=None,
                   help="Comma-separated backend ids to run (default: all active).")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the number of utterances (after subset selection).")
    p.add_argument("--resample-subset", action="store_true",
                   help="Rebuild data/subset_manifest.json from scratch (re-sample).")
    p.add_argument("--config", type=str, default=None, help="Path to eval.toml.")
    p.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for this run's CSV, summary, and manifest (relative to repo root).",
    )
    p.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Prior run directory: retain its successes and retry only failures.",
    )
    return p.parse_args(argv)


# --- audio preparation ------------------------------------------------------


def ensure_prepared_audio(cfg: EvalConfig, entries: list[SubsetEntry]) -> dict[str, "PreparedInfo"]:
    """Ensure a prepared WAV exists for each entry; return per-entry audio info.

    Loads the dataset ONLY if some prepared files are missing (so cached-audio
    re-runs stay offline). Returns eval_id -> (path, duration_seconds, sha256).
    """
    prepared: dict[str, PreparedInfo] = {}
    missing = [e for e in entries if not prepared_path_for(e.eval_id).exists()]

    if not missing:
        for e in entries:
            info = describe_prepared(prepared_path_for(e.eval_id))
            prepared[e.eval_id] = PreparedInfo(info.path, info.duration_seconds, info.sha256)
        return prepared

    print(f"Preparing audio for {len(missing)} utterance(s) (loading dataset)...")
    from .dataset import load_svarah

    ds = load_svarah(cfg.dataset)
    schema = validate_schema(ds)
    audio_field = schema.audio_field

    for e in entries:
        out = prepared_path_for(e.eval_id)
        if out.exists():
            info = describe_prepared(out)
        else:
            row = ds[e.row_index]
            info = prepare_audio(row[audio_field], out)
        prepared[e.eval_id] = PreparedInfo(info.path, info.duration_seconds, info.sha256)
    return prepared


class PreparedInfo:
    __slots__ = ("path", "duration_seconds", "sha256")

    def __init__(self, path: Path, duration_seconds: float, sha256: str):
        self.path = path
        self.duration_seconds = duration_seconds
        self.sha256 = sha256


# --- per-backend execution --------------------------------------------------


def run_backend(
    backend: ASRBackend,
    entries: list[SubsetEntry],
    prepared: dict[str, PreparedInfo],
    cache: TranscriptCache,
    resumed_successes: dict[str, UtteranceResult] | None = None,
) -> list[UtteranceResult]:
    """Warm up (local), then transcribe/score every utterance for one backend."""
    signature = backend.cache_signature()
    results: list[UtteranceResult] = []

    is_local = backend.cfg.type in ("whisper_sherpa", "crispasr_server")
    if is_local and entries:
        # DESIGN: one unscored warm-up utterance before timed local inference.
        warm = prepared[entries[0].eval_id]
        try:
            backend.transcribe(warm.path)
            print(f"  [{backend.name}] warm-up done")
        except Exception as e:  # noqa: BLE001
            print(f"  [{backend.name}] warm-up failed (continuing): {e}")

    for i, e in enumerate(entries, 1):
        info = prepared[e.eval_id]
        resumed = (resumed_successes or {}).get(e.eval_id)
        if resumed is not None:
            results.append(resumed)
            print(f"  [{backend.name}] {i}/{len(entries)} {e.eval_id} (resumed success)")
            continue

        key = cache.make_key(signature=signature, eval_id=e.eval_id, audio_sha256=info.sha256)

        cached_text = cache.get(key)
        if cached_text is not None:
            results.append(
                _score_result(backend.name, e, cached_text, info, from_cache=True,
                              inference_seconds=None)
            )
            print(f"  [{backend.name}] {i}/{len(entries)} {e.eval_id} (cache hit)")
            continue

        try:
            backend.prepare_request()
            t0 = time.perf_counter()
            raw = backend.transcribe(info.path)
            inference_seconds = time.perf_counter() - t0
        except TranscriptionError as ex:
            results.append(_failure_result(backend.name, e, ex.category, str(ex), ex.attempts))
            print(f"  [{backend.name}] {i}/{len(entries)} {e.eval_id} FAILED ({ex.category})")
            continue
        except Exception as ex:  # noqa: BLE001 - unexpected, still recorded as failure
            results.append(_failure_result(backend.name, e, type(ex).__name__, str(ex), 1))
            print(f"  [{backend.name}] {i}/{len(entries)} {e.eval_id} FAILED ({type(ex).__name__})")
            continue

        # Cache the RAW transcript before normalization/scoring (DESIGN).
        cache.put(key, text=raw, eval_id=e.eval_id, audio_sha256=info.sha256, signature=signature)
        results.append(_score_result(backend.name, e, raw, info, from_cache=False,
                                      inference_seconds=inference_seconds,
                                      attempts=backend.last_attempts))
        print(f"  [{backend.name}] {i}/{len(entries)} {e.eval_id} wer="
              f"{results[-1].wer:.3f} rtf={results[-1].rtf:.2f}")

    return results


def _load_resume(
    resume_dir: Path,
    *,
    backend_id: str,
    signature: dict,
    entries: list[SubsetEntry],
    cfg: EvalConfig,
    debug: bool,
) -> tuple[dict[str, UtteranceResult], dict]:
    """Validate a prior run and return only successful rows to carry forward."""
    manifest_path = resume_dir / "run_manifest.json"
    csv_path = resume_dir / "per_utterance.csv"
    if not manifest_path.exists() or not csv_path.exists():
        raise ValueError(f"Resume directory is missing run artifacts: {resume_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = manifest.get("dataset", {})
    expected_dataset = {
        "name": cfg.dataset.name,
        "revision": cfg.dataset.revision,
        "split": cfg.dataset.split,
        "seed": cfg.dataset.seed,
        "subset_size": len(entries),
        "debug": debug,
    }
    for key, expected in expected_dataset.items():
        if dataset.get(key) != expected:
            raise ValueError(
                f"Resume dataset mismatch for {key}: {dataset.get(key)!r} != {expected!r}"
            )

    previous_signature = manifest.get("backends", {}).get(backend_id)
    if not isinstance(previous_signature, dict):
        raise ValueError(f"Resume run does not contain backend {backend_id}")
    # A resumed run may add execution-only pacing. Every setting recorded by the
    # source must still match, while new settings are allowed and re-recorded.
    for key, previous in previous_signature.items():
        if signature.get(key) != previous:
            raise ValueError(f"Resume backend mismatch for {key}")

    rows = read_per_utterance_csv(csv_path)
    expected = {entry.eval_id: entry for entry in entries}
    if len(rows) != len(entries) or {row.eval_id for row in rows} != set(expected):
        raise ValueError("Resume run does not contain exactly the selected utterance set")
    for row in rows:
        if row.backend_id != backend_id:
            raise ValueError(f"Unexpected backend in resume rows: {row.backend_id}")
        if row.raw_reference != expected[row.eval_id].reference:
            raise ValueError(f"Reference mismatch in resume row {row.eval_id}")

    successes = {row.eval_id: row for row in rows if row.ok}
    try:
        source_path = resume_dir.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        source_path = str(resume_dir.resolve())
    provenance = {
        "source_path": source_path,
        "run_manifest_sha256": sha256_file(manifest_path),
        "per_utterance_sha256": sha256_file(csv_path),
        "retained_successes": len(successes),
        "retried_failures": len(rows) - len(successes),
    }
    return successes, provenance


def _score_result(
    backend_id: str,
    entry: SubsetEntry,
    raw_hypothesis: str,
    info: PreparedInfo,
    *,
    from_cache: bool,
    inference_seconds: float | None,
    attempts: int = 1,
) -> UtteranceResult:
    norm_ref = normalize_text(entry.reference)
    norm_hyp = normalize_text(raw_hypothesis)
    s = score_utterance(norm_ref, norm_hyp)
    rtf = (inference_seconds / info.duration_seconds
           if inference_seconds is not None and info.duration_seconds > 0 else None)
    return UtteranceResult(
        backend_id=backend_id,
        eval_id=entry.eval_id,
        status="success",
        from_cache=from_cache,
        inference_seconds=inference_seconds,
        audio_seconds=info.duration_seconds,
        rtf=rtf,
        raw_reference=entry.reference,
        raw_hypothesis=raw_hypothesis,
        normalized_reference=norm_ref,
        normalized_hypothesis=norm_hyp,
        hits=s.hits,
        substitutions=s.substitutions,
        deletions=s.deletions,
        insertions=s.insertions,
        reference_words=s.reference_words,
        wer=s.wer,
        attempts=attempts,
    )


def _failure_result(backend_id, entry, category, message, attempts) -> UtteranceResult:
    return UtteranceResult(
        backend_id=backend_id,
        eval_id=entry.eval_id,
        status="failure",
        raw_reference=entry.reference,
        error_category=category,
        error_message=message,
        attempts=attempts,
    )


# --- main -------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _force_utf8_console()
    cfg = load_config(Path(args.config) if args.config else None)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    per_utterance_csv = output_dir / "per_utterance.csv"
    run_manifest_path = output_dir / "run_manifest.json"
    summary_md = output_dir / "summary.md"
    started_at = _utcnow()

    # 1) Subset manifest (committed, frozen).
    build_manifest(cfg.dataset, force=args.resample_subset)
    entries = select_entries(read_manifest(), debug=args.debug, cfg=cfg.dataset)
    if args.limit is not None:
        entries = entries[: args.limit]
    print(f"Subset: {len(entries)} utterances"
          f"{' (debug)' if args.debug else ''}.")

    # 2) Identical prepared audio for all backends.
    prepared = ensure_prepared_audio(cfg, entries)

    # 3) Select backends.
    active_ids = (
        [b.strip() for b in args.backends.split(",")] if args.backends
        else list(cfg.active_backend_ids)
    )
    cache = TranscriptCache(enabled=not args.no_cache)
    resume_dir = None
    if args.resume_from:
        resume_dir = Path(args.resume_from)
        if not resume_dir.is_absolute():
            resume_dir = (REPO_ROOT / resume_dir).resolve()
        if len(active_ids) != 1:
            raise ValueError("--resume-from requires exactly one selected backend")

    results_by_backend: dict[str, list[UtteranceResult]] = {}
    summaries = {}
    backend_signatures: dict[str, dict] = {}
    resume_provenance: dict | None = None

    for backend_id in active_ids:
        if backend_id not in cfg.backends:
            print(f"[skip] Unknown backend id in config: {backend_id}")
            continue
        backend = build_backend(cfg.backends[backend_id])
        resumed_successes: dict[str, UtteranceResult] = {}
        if resume_dir is not None:
            resumed_successes, resume_provenance = _load_resume(
                resume_dir,
                backend_id=backend_id,
                signature=backend.cache_signature(),
                entries=entries,
                cfg=cfg,
                debug=args.debug,
            )
            print(
                f"Resume: retaining {len(resumed_successes)} successes and retrying "
                f"{len(entries) - len(resumed_successes)} failures."
            )

        if not backend.is_available():
            print(f"[skip] Backend '{backend_id}' is unavailable "
                  f"(missing model/binary/key). Continuing.")
            summaries[backend_id] = summarize_backend(
                backend_id, [], available=False, startup_seconds=None,
                total_attempted=len(entries))
            continue

        print(f"\n=== Backend: {backend_id} ===")
        try:
            t0 = time.perf_counter()
            backend.start()  # startup timed separately (DESIGN)
            startup_seconds = time.perf_counter() - t0
            print(f"  startup: {startup_seconds:.2f}s")
        except BackendUnavailableError as ex:
            print(f"[skip] Backend '{backend_id}' failed to start: {ex}. Continuing.")
            summaries[backend_id] = summarize_backend(
                backend_id, [], available=False, startup_seconds=None,
                total_attempted=len(entries))
            continue

        try:
            results = run_backend(
                backend,
                entries,
                prepared,
                cache,
                resumed_successes=resumed_successes,
            )
        finally:
            backend.close()

        results_by_backend[backend_id] = results
        backend_signatures[backend_id] = backend.cache_signature()
        summaries[backend_id] = summarize_backend(
            backend_id, results, available=True, startup_seconds=startup_seconds,
            total_attempted=len(entries))
        print_worst_utterances(backend_id, results)

    # 4) Aggregate + write outputs.
    all_results = [r for rs in results_by_backend.values() for r in rs]
    write_per_utterance_csv(all_results, per_utterance_csv)

    shared_ids, shared_metrics = shared_success_metrics(results_by_backend)

    completed_at = _utcnow()
    run_manifest = build_run_manifest(
        cfg,
        backend_signatures=backend_signatures,
        backend_summaries={
            bid: {
                "available": summary.available,
                "startup_seconds": summary.startup_seconds,
                "success_count": summary.success_count,
                "failure_count": summary.failure_count,
            }
            for bid, summary in summaries.items()
        },
        started_at=started_at,
        completed_at=completed_at,
        cache_enabled=not args.no_cache,
        subset_size=len(entries),
        debug=args.debug,
    )
    if resume_provenance is not None:
        run_manifest["resume"] = resume_provenance
    write_json_atomic(run_manifest_path, run_manifest)

    write_summary_md(
        summaries=summaries,
        shared_ids=shared_ids,
        shared_metrics=shared_metrics,
        dataset_name=cfg.dataset.name,
        revision=cfg.dataset.revision,
        split=cfg.dataset.split,
        seed=cfg.dataset.seed,
        subset_size=len(entries),
        debug=args.debug,
        path=summary_md,
    )

    _print_final_summary(summaries, shared_ids)
    print(f"\nWrote:\n  {per_utterance_csv}\n  {run_manifest_path}\n  {summary_md}")
    return 0


def _print_final_summary(summaries, shared_ids) -> None:
    print("\n" + "=" * 60)
    print("SUMMARY (each backend over its own successes)")
    print("=" * 60)
    for bid, s in summaries.items():
        if not s.available:
            print(f"  {bid}: UNAVAILABLE (skipped)")
            continue
        m = s.own_metrics
        print(f"  {bid}: corpus_wer={m.corpus_wer:.4f} mean_wer={m.mean_wer:.4f} "
              f"agg_rtf={m.aggregate_rtf:.3f} (over {m.rtf_sample_size} timed) "
              f"success={s.success_count} fail={s.failure_count} coverage={s.coverage:.0%}")
    print(f"\n  Shared successful utterances across active backends: {len(shared_ids)}")


if __name__ == "__main__":
    sys.exit(main())
