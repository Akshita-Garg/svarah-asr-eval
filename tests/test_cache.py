"""Cache-key behavior: stability and correct invalidation.

DESIGN.md "Caching": changing a model, backend config, or audio file must
invalidate the old entry; identical inputs must hit.
"""

from voicerefine_eval.cache import TranscriptCache

SIG = {"backend_id": "b1", "type": "whisper_sherpa", "model_hashes": {"m": "aaa"}}


def _cache(tmp_path):
    return TranscriptCache(enabled=True, root=tmp_path)


def test_same_inputs_same_key():
    c = TranscriptCache()
    k1 = c.make_key(signature=SIG, eval_id="u1", audio_sha256="h1")
    k2 = c.make_key(signature=SIG, eval_id="u1", audio_sha256="h1")
    assert k1.key_hash == k2.key_hash
    assert k1.backend_id == "b1"


def test_audio_change_invalidates():
    c = TranscriptCache()
    k1 = c.make_key(signature=SIG, eval_id="u1", audio_sha256="h1")
    k2 = c.make_key(signature=SIG, eval_id="u1", audio_sha256="h2")
    assert k1.key_hash != k2.key_hash


def test_model_hash_change_invalidates():
    c = TranscriptCache()
    k1 = c.make_key(signature=SIG, eval_id="u1", audio_sha256="h1")
    sig2 = {**SIG, "model_hashes": {"m": "bbb"}}
    k2 = c.make_key(signature=sig2, eval_id="u1", audio_sha256="h1")
    assert k1.key_hash != k2.key_hash


def test_roundtrip_put_get(tmp_path):
    c = _cache(tmp_path)
    k = c.make_key(signature=SIG, eval_id="u1", audio_sha256="h1")
    assert c.get(k) is None
    c.put(k, text="hello world", eval_id="u1", audio_sha256="h1", signature=SIG)
    assert c.get(k) == "hello world"


def test_disabled_cache_never_hits(tmp_path):
    c = TranscriptCache(enabled=False, root=tmp_path)
    k = c.make_key(signature=SIG, eval_id="u1", audio_sha256="h1")
    c.put(k, text="x", eval_id="u1", audio_sha256="h1", signature=SIG)
    assert c.get(k) is None
