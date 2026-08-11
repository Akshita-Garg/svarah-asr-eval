"""Text normalization for reference and hypothesis transcripts.

DESIGN.md "Text Normalization": reference and hypothesis are normalized with the
SAME English normalizer before scoring, and the implementation must match
Whisper's ``EnglishTextNormalizer`` for punctuation, case, numbers, and
contractions.

We do not reimplement this. We use the real ``EnglishTextNormalizer`` shipped in
``whisper-normalizer`` (the same algorithm as OpenAI Whisper), so behavior is
identical by construction. The unit tests in ``tests/test_normalize.py`` lock the
specific cases the design calls out.

Raw text is never mutated in place; callers keep the raw strings and store the
normalized copy alongside (DESIGN: "Normalization must never replace the source
evidence").
"""

from __future__ import annotations

from functools import lru_cache

from whisper_normalizer.english import EnglishTextNormalizer


@lru_cache(maxsize=1)
def _normalizer() -> EnglishTextNormalizer:
    # The normalizer loads a small mapping table on construction; build it once.
    return EnglishTextNormalizer()


def normalize_text(text: str) -> str:
    """Return the Whisper-English-normalized form of ``text``.

    The normalizer lowercases, expands contractions, standardizes numbers and
    currency, and strips punctuation. An empty or whitespace-only input maps to
    an empty string.
    """
    if text is None:
        return ""
    return _normalizer()(text)
