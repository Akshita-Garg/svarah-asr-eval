"""Lock the Whisper English normalizer behavior the design calls out.

DESIGN.md "Text Normalization": the normalizer must handle punctuation, case,
numbers, and contractions like Whisper's ``EnglishTextNormalizer``. These tests
assert the specific transformations so a future dependency bump that changes
behavior is caught.
"""

from voicerefine_eval.normalize import normalize_text


def test_lowercases_and_strips_punctuation():
    assert normalize_text("Hello, World!") == "hello world"


def test_expands_contractions():
    # Whisper's normalizer expands common contractions.
    assert normalize_text("I don't know") == "i do not know"
    assert normalize_text("we're here") == "we are here"


def test_numbers_and_currency():
    # Reference and hypothesis both go through this, so consistency matters more
    # than any single "correct" spelling; we lock the actual behavior.
    assert normalize_text("It costs $5") == "it costs $5"
    # The Whisper normalizer collapses the word "percent" to the "%" symbol and
    # spells numbers as digits. We lock the actual behavior, not an assumption.
    assert normalize_text("twenty percent") == "20%"
    assert normalize_text("five dollars") == "$5"


def test_empty_and_none():
    assert normalize_text("") == ""
    assert normalize_text("   ") == ""
    assert normalize_text(None) == ""


def test_idempotent_on_already_normalized():
    once = normalize_text("The Dog's Bone.")
    twice = normalize_text(once)
    assert once == twice
