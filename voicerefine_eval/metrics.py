"""Accuracy metrics built on ``jiwer`` word alignment.

DESIGN.md "Accuracy Metrics": ``jiwer`` supplies word alignment and edit counts;
we do not implement edit distance ourselves. For every successful utterance we
record hits / substitutions / deletions / insertions / reference-word-count / WER.

Per-utterance WER = (S + D + I) / reference_words.

Corpus WER is computed from AGGREGATE edit counts (sum of S, D, I over all
utterances divided by the sum of reference words) rather than by averaging
per-utterance WERs. Mean WER weights every utterance equally; corpus WER weights
longer references more and is the primary benchmark figure (DESIGN).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import jiwer


@dataclass(frozen=True)
class UtteranceScore:
    """Edit counts and WER for a single (reference, hypothesis) pair."""

    hits: int
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int
    wer: float


def score_utterance(reference_norm: str, hypothesis_norm: str) -> UtteranceScore:
    """Align one normalized reference/hypothesis pair and return its edit counts.

    Inputs must already be normalized. An empty reference is handled explicitly:
    ``jiwer`` cannot divide by zero reference words, so WER is defined as 0.0 when
    there are also no insertions and 1.0-scaled by insertions otherwise. We keep
    it simple and mirror jiwer's own convention by guarding the zero case.
    """
    ref_words = len(reference_norm.split())
    out = jiwer.process_words(reference_norm, hypothesis_norm)

    if ref_words == 0:
        # No reference words: every hypothesis word is an insertion. Report the
        # raw insertion count; WER is undefined/None-like, so we surface 0.0 when
        # the hypothesis is also empty and leave the edit counts truthful.
        insertions = len(hypothesis_norm.split())
        wer = 0.0 if insertions == 0 else float(insertions)
        return UtteranceScore(0, 0, 0, insertions, 0, wer)

    edits = out.substitutions + out.deletions + out.insertions
    return UtteranceScore(
        hits=out.hits,
        substitutions=out.substitutions,
        deletions=out.deletions,
        insertions=out.insertions,
        reference_words=ref_words,
        wer=edits / ref_words,
    )


@dataclass(frozen=True)
class BackendMetrics:
    """Aggregate accuracy + speed metrics for one backend over a set of scores."""

    success_count: int
    mean_wer: float
    median_wer: float
    corpus_wer: float
    total_substitutions: int
    total_deletions: int
    total_insertions: int
    total_reference_words: int
    mean_rtf: float
    aggregate_rtf: float
    total_inference_seconds: float
    total_audio_seconds: float
    # Number of utterances with a fresh timing (RTF is measured over these only;
    # cache hits carry no timing and are excluded so they can't distort speed).
    rtf_sample_size: int


def aggregate_metrics(
    scores: list[UtteranceScore],
    inference_seconds: list[float | None],
    audio_seconds: list[float],
) -> BackendMetrics:
    """Combine per-utterance scores and timings into backend-level metrics.

    ``scores``, ``inference_seconds`` and ``audio_seconds`` are parallel lists over
    the SAME successful utterances (same order, same length). An
    ``inference_seconds`` entry of ``None`` means the transcript came from cache
    (no fresh timing); such utterances still count for WER but are EXCLUDED from
    RTF — otherwise a cache hit would masquerade as infinitely fast and pull RTF
    toward zero.
    """
    n = len(scores)
    if n == 0:
        return BackendMetrics(0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0)

    wers = [s.wer for s in scores]
    total_s = sum(s.substitutions for s in scores)
    total_d = sum(s.deletions for s in scores)
    total_i = sum(s.insertions for s in scores)
    total_ref = sum(s.reference_words for s in scores)

    corpus_wer = (total_s + total_d + total_i) / total_ref if total_ref else 0.0

    # RTF only over utterances that were freshly transcribed this run.
    timed = [
        (inf, aud)
        for inf, aud in zip(inference_seconds, audio_seconds)
        if inf is not None and aud > 0
    ]
    rtfs = [inf / aud for inf, aud in timed]
    total_inf = sum(inf for inf, _ in timed)
    total_aud = sum(aud for _, aud in timed)

    return BackendMetrics(
        success_count=n,
        mean_wer=statistics.fmean(wers),
        median_wer=statistics.median(wers),
        corpus_wer=corpus_wer,
        total_substitutions=total_s,
        total_deletions=total_d,
        total_insertions=total_i,
        total_reference_words=total_ref,
        mean_rtf=statistics.fmean(rtfs) if rtfs else 0.0,
        aggregate_rtf=(total_inf / total_aud) if total_aud > 0 else 0.0,
        total_inference_seconds=total_inf,
        total_audio_seconds=total_aud,
        rtf_sample_size=len(rtfs),
    )
