"""Unit tests for WER edit counts and corpus/mean aggregation.

DESIGN.md "Accuracy Metrics": per-utterance WER = (S+D+I)/ref_words; corpus WER
comes from aggregate edit counts, and mean WER weights utterances equally. These
tests verify each edit type and the mean-vs-corpus distinction.
"""

import math

from voicerefine_eval.metrics import (
    aggregate_metrics,
    score_utterance,
)


def test_perfect_match():
    s = score_utterance("the quick brown fox", "the quick brown fox")
    assert (s.substitutions, s.deletions, s.insertions) == (0, 0, 0)
    assert s.hits == 4
    assert s.reference_words == 4
    assert s.wer == 0.0


def test_one_substitution():
    s = score_utterance("the quick brown fox", "the quick red fox")
    assert s.substitutions == 1
    assert s.deletions == 0
    assert s.insertions == 0
    assert s.reference_words == 4
    assert math.isclose(s.wer, 0.25)


def test_one_deletion():
    s = score_utterance("the quick brown fox", "the quick fox")
    assert s.deletions == 1
    assert math.isclose(s.wer, 0.25)


def test_one_insertion():
    s = score_utterance("the quick fox", "the quick brown fox")
    assert s.insertions == 1
    assert s.reference_words == 3
    assert math.isclose(s.wer, 1 / 3)


def test_empty_reference_counts_insertions():
    s = score_utterance("", "hello world")
    assert s.reference_words == 0
    assert s.insertions == 2


def test_corpus_vs_mean_wer_differ_by_length():
    # Short utterance: 1 error over 1 word -> WER 1.0
    # Long utterance: 1 error over 9 words -> WER ~0.111
    short = score_utterance("yes", "no")
    long = score_utterance(
        "the meeting is scheduled for tomorrow at noon sharp",
        "the meeting is scheduled for tomorrow at noon sharpe",
    )
    m = aggregate_metrics([short, long], [0.1, 0.9], [1.0, 5.0])

    # Mean weights each utterance equally: (1.0 + ~0.111) / 2
    assert math.isclose(m.mean_wer, (short.wer + long.wer) / 2)
    # Corpus weights by reference words: total edits / total ref words
    total_edits = (
        short.substitutions + short.deletions + short.insertions
        + long.substitutions + long.deletions + long.insertions
    )
    total_ref = short.reference_words + long.reference_words
    assert math.isclose(m.corpus_wer, total_edits / total_ref)
    # The two must differ here (the whole point of reporting both).
    assert not math.isclose(m.mean_wer, m.corpus_wer)


def test_rtf_aggregation():
    s = score_utterance("a b c", "a b c")
    # inference 2s over 4s audio, and 1s over 1s audio
    m = aggregate_metrics([s, s], [2.0, 1.0], [4.0, 1.0])
    # aggregate RTF = total inf / total audio = 3 / 5 = 0.6
    assert math.isclose(m.aggregate_rtf, 0.6)
    # mean RTF = mean(0.5, 1.0) = 0.75
    assert math.isclose(m.mean_rtf, 0.75)


def test_empty_aggregate():
    m = aggregate_metrics([], [], [])
    assert m.success_count == 0
    assert m.corpus_wer == 0.0
    assert m.rtf_sample_size == 0


def test_cache_hits_excluded_from_rtf_but_counted_for_wer():
    # Three successful utterances; the middle one is a cache hit (inference=None).
    s = score_utterance("a b c", "a b x")  # one substitution over 3 words
    scores = [s, s, s]
    inference = [2.0, None, 1.0]  # middle came from cache
    audio = [4.0, 4.0, 1.0]
    m = aggregate_metrics(scores, inference, audio)

    # WER counts all three utterances.
    assert m.success_count == 3
    # RTF only over the two timed ones: total 3s inf / 5s audio = 0.6
    assert m.rtf_sample_size == 2
    assert math.isclose(m.aggregate_rtf, 0.6)
    # mean RTF = mean(0.5, 1.0) = 0.75 — the None does NOT enter as a 0.
    assert math.isclose(m.mean_rtf, 0.75)
