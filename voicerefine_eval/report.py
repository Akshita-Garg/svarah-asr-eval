"""Output writers: per_utterance.csv, summary.md, and console reporting.

DESIGN.md "Outputs": a long-form CSV row per backend+utterance, a standalone
methodology + comparison report, and console output of progress, failures, and
the ten worst successful utterances per backend. The summary must state that
Svarah measures Indian English accent robustness and is not a dictation
benchmark.

DESIGN.md "Failure And Comparison Rules": two comparison views — each backend
over its own successes, and metrics over the SHARED subset every active backend
transcribed. The shared-success view is the primary comparison; coverage and
failure counts appear beside WER.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .config import REPO_ROOT
from .metrics import BackendMetrics, UtteranceScore, aggregate_metrics
from .results import UtteranceResult

RESULTS_DIR = REPO_ROOT / "results"
PER_UTTERANCE_CSV = RESULTS_DIR / "per_utterance.csv"
SUMMARY_MD = RESULTS_DIR / "summary.md"

_CSV_FIELDS = [
    "backend_id",
    "eval_id",
    "status",
    "from_cache",
    "wer",
    "hits",
    "substitutions",
    "deletions",
    "insertions",
    "reference_words",
    "inference_seconds",
    "audio_seconds",
    "rtf",
    "error_category",
    "error_message",
    "attempts",
    "raw_reference",
    "raw_hypothesis",
    "normalized_reference",
    "normalized_hypothesis",
]


def write_per_utterance_csv(results: list[UtteranceResult], path: Path = PER_UTTERANCE_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "backend_id": r.backend_id,
                    "eval_id": r.eval_id,
                    "status": r.status,
                    "from_cache": int(r.from_cache),
                    "wer": _fmt(r.wer),
                    "hits": _blank(r.hits),
                    "substitutions": _blank(r.substitutions),
                    "deletions": _blank(r.deletions),
                    "insertions": _blank(r.insertions),
                    "reference_words": _blank(r.reference_words),
                    "inference_seconds": _fmt(r.inference_seconds),
                    "audio_seconds": _fmt(r.audio_seconds),
                    "rtf": _fmt(r.rtf),
                    "error_category": r.error_category or "",
                    "error_message": (r.error_message or "")[:500],
                    "attempts": r.attempts,
                    "raw_reference": r.raw_reference,
                    "raw_hypothesis": r.raw_hypothesis,
                    "normalized_reference": r.normalized_reference,
                    "normalized_hypothesis": r.normalized_hypothesis,
                }
            )


def _fmt(x: float | None) -> str:
    return "" if x is None else f"{x:.6f}"


def _blank(x: int | None) -> str | int:
    return "" if x is None else x


# --- aggregation helpers ----------------------------------------------------


@dataclass
class BackendSummary:
    backend_id: str
    available: bool
    success_count: int
    failure_count: int
    total_attempted: int
    startup_seconds: float | None
    own_metrics: BackendMetrics
    failure_categories: dict[str, int]

    @property
    def coverage(self) -> float:
        return self.success_count / self.total_attempted if self.total_attempted else 0.0


def _scores_and_timings(results: list[UtteranceResult]):
    scores, inf, aud = [], [], []
    for r in results:
        if not r.ok:
            continue
        scores.append(
            UtteranceScore(
                hits=r.hits or 0,
                substitutions=r.substitutions or 0,
                deletions=r.deletions or 0,
                insertions=r.insertions or 0,
                reference_words=r.reference_words or 0,
                wer=r.wer or 0.0,
            )
        )
        # Keep None (cache hit) as None so RTF excludes it; WER still counts it.
        inf.append(r.inference_seconds)
        aud.append(r.audio_seconds or 0.0)
    return scores, inf, aud


def summarize_backend(
    backend_id: str,
    results: list[UtteranceResult],
    *,
    available: bool,
    startup_seconds: float | None,
    total_attempted: int,
) -> BackendSummary:
    scores, inf, aud = _scores_and_timings(results)
    failures = [r for r in results if not r.ok]
    categories: dict[str, int] = {}
    for r in failures:
        categories[r.error_category or "unknown"] = categories.get(r.error_category or "unknown", 0) + 1
    return BackendSummary(
        backend_id=backend_id,
        available=available,
        success_count=len(scores),
        failure_count=len(failures),
        total_attempted=total_attempted,
        startup_seconds=startup_seconds,
        own_metrics=aggregate_metrics(scores, inf, aud),
        failure_categories=categories,
    )


def shared_success_metrics(
    results_by_backend: dict[str, list[UtteranceResult]],
) -> tuple[list[str], dict[str, BackendMetrics]]:
    """Metrics over utterances every active backend transcribed successfully.

    Returns (shared_eval_ids, {backend_id: metrics_over_shared}).
    """
    if not results_by_backend:
        return [], {}

    success_sets = []
    for results in results_by_backend.values():
        success_sets.append({r.eval_id for r in results if r.ok})
    shared = set.intersection(*success_sets) if success_sets else set()
    shared_ids = sorted(shared)

    out: dict[str, BackendMetrics] = {}
    for backend_id, results in results_by_backend.items():
        by_id = {r.eval_id: r for r in results if r.ok and r.eval_id in shared}
        subset = [by_id[i] for i in shared_ids if i in by_id]
        scores, inf, aud = _scores_and_timings(subset)
        out[backend_id] = aggregate_metrics(scores, inf, aud)
    return shared_ids, out


# --- console ----------------------------------------------------------------


def print_worst_utterances(backend_id: str, results: list[UtteranceResult], n: int = 10) -> None:
    successes = [r for r in results if r.ok]
    worst = sorted(successes, key=lambda r: (r.wer or 0.0), reverse=True)[:n]
    print(f"\n  Ten worst successful utterances for {backend_id}:")
    if not worst:
        print("    (no successful utterances)")
        return
    for r in worst:
        print(f"    {r.eval_id}  WER={r.wer:.3f}  ref_words={r.reference_words}")
        print(f"      REF: {r.normalized_reference[:120]}")
        print(f"      HYP: {r.normalized_hypothesis[:120]}")


# --- summary.md -------------------------------------------------------------

_SVARAH_CAVEAT = (
    "This experiment measures **Indian English accent robustness** on the Svarah "
    "dataset. Svarah is not a dedicated laptop-dictation dataset, so these numbers "
    "do not measure VoiceRefine's complete dictation experience."
)


def write_summary_md(
    *,
    summaries: dict[str, BackendSummary],
    shared_ids: list[str],
    shared_metrics: dict[str, BackendMetrics],
    dataset_name: str,
    revision: str,
    split: str,
    seed: int,
    subset_size: int,
    debug: bool,
    path: Path = SUMMARY_MD,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    a = lines.append

    a("# VoiceRefine Svarah ASR Evaluation — Results\n")
    a(_SVARAH_CAVEAT + "\n")

    a("## Methodology\n")
    a(f"- Dataset: `{dataset_name}` (revision `{revision}`), split `{split}`.")
    a(f"- Subset: {subset_size} utterances, seed {seed}{' (DEBUG run)' if debug else ''}.")
    a("- All backends receive identical 16 kHz mono signed-16 WAV files.")
    a("- Text normalized with Whisper's `EnglishTextNormalizer` before scoring.")
    a("- WER/edit counts from `jiwer`. Corpus WER (aggregate edits) is the primary figure.")
    a("- Per-utterance timing wraps only the backend call; startup is measured separately.")
    a("- ElevenLabs timing is API end-to-end latency (upload + network + service + download).\n")

    a("## Per-backend results (each backend over its own successes)\n")
    a("RTF is measured only over utterances freshly transcribed this run; "
      "cache hits carry no timing and are excluded (the **RTF n** column is that "
      "sample size, which can be smaller than the success count on a cached run).\n")
    a("| Backend | Avail | Success | Fail | Coverage | Corpus WER | Mean WER | Median WER | Mean RTF | Agg RTF | RTF n | Startup (s) |")
    a("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for bid, s in summaries.items():
        m = s.own_metrics
        a(
            f"| {bid} | {'yes' if s.available else 'no'} | {s.success_count} | "
            f"{s.failure_count} | {s.coverage:.0%} | {m.corpus_wer:.4f} | {m.mean_wer:.4f} | "
            f"{m.median_wer:.4f} | {m.mean_rtf:.3f} | {m.aggregate_rtf:.3f} | {m.rtf_sample_size} | "
            f"{'' if s.startup_seconds is None else f'{s.startup_seconds:.2f}'} |"
        )
    a("")

    a("## Primary comparison — shared subset transcribed by every active backend\n")
    a(f"Shared successful utterances: **{len(shared_ids)}**. This is the primary "
      "direct comparison; coverage above shows that a backend cannot look better "
      "by failing on hard samples.\n")
    if shared_metrics:
        a("| Backend | Corpus WER | Mean WER | Median WER | Mean RTF | Agg RTF | RTF n |")
        a("| --- | --- | --- | --- | --- | --- | --- |")
        for bid, m in shared_metrics.items():
            a(
                f"| {bid} | {m.corpus_wer:.4f} | {m.mean_wer:.4f} | {m.median_wer:.4f} | "
                f"{m.mean_rtf:.3f} | {m.aggregate_rtf:.3f} | {m.rtf_sample_size} |"
            )
        a("")

    a("## Failures by category\n")
    any_fail = False
    for bid, s in summaries.items():
        if s.failure_categories:
            any_fail = True
            cats = ", ".join(f"{k}={v}" for k, v in sorted(s.failure_categories.items()))
            a(f"- **{bid}**: {cats}")
    if not any_fail:
        a("- None.")
    a("")

    path.write_text("\n".join(lines), encoding="utf-8")
