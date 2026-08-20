from pathlib import Path

import pytest
import requests

from voicerefine_eval.backends import build_backend
from voicerefine_eval.backends.base import TranscriptionError
from voicerefine_eval.backends.parakeet_crispasr import CrispAsrServerBackend
from voicerefine_eval.backends.sarvam import SarvamBackend
from voicerefine_eval.backends.deepgram import DeepgramBackend
from voicerefine_eval.backends.gnani import GnaniBackend
from voicerefine_eval.backends.smallest import SmallestBackend
from voicerefine_eval.config import BackendConfig, load_config


class _FakeProcess:
    returncode = None

    def poll(self):
        return None

    def terminate(self):
        return None

    def wait(self, timeout=None):
        return 0


class _Response:
    def __init__(self, status_code=200, payload=None, text="", headers=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.headers = headers or {}

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _crisp_config(tmp_path: Path) -> BackendConfig:
    binary = tmp_path / "crispasr.exe"
    model = tmp_path / "model.gguf"
    binary.write_bytes(b"binary")
    model.write_bytes(b"model")
    return BackendConfig(
        "controlled_model",
        "crispasr_server",
        {
            "bin": str(binary),
            "model": str(model),
            "backend": "whisper",
            "language": "en",
            "threads": 8,
            "gpu_backend": "cpu",
            "quantization": "q4_k",
            "no_punctuation": True,
            "runtime_version": "0.8.23",
            "runtime_git_sha": "7d22deec",
            "host": "127.0.0.1",
            "port": 51234,
        },
    )


def test_crispasr_server_uses_controlled_execution_flags(tmp_path, monkeypatch):
    captured = {}

    def fake_popen(args, **kwargs):
        captured["args"] = args
        return _FakeProcess()

    monkeypatch.setattr("voicerefine_eval.backends.parakeet_crispasr.subprocess.Popen", fake_popen)
    monkeypatch.setattr(CrispAsrServerBackend, "_wait_for_port", lambda self, timeout: None)
    backend = CrispAsrServerBackend(_crisp_config(tmp_path))
    backend.start()
    backend.close()

    args = captured["args"]
    assert args[args.index("--backend") + 1] == "whisper"
    assert args[args.index("--threads") + 1] == "8"
    assert args[args.index("--gpu-backend") + 1] == "cpu"
    assert "--server" in args
    assert "--no-punctuation" in args


def test_crispasr_signature_records_resource_controls(tmp_path):
    backend = build_backend(_crisp_config(tmp_path))
    signature = backend.cache_signature()
    assert signature["runtime"] == "persistent_server"
    assert signature["threads"] == 8
    assert signature["gpu_backend"] == "cpu"
    assert signature["quantization"] == "q4_k"
    assert signature["no_punctuation"] is True
    assert signature["runtime_version"] == "0.8.23"
    assert signature["runtime_git_sha"] == "7d22deec"
    assert set(signature["hashes"]) == {"model", "bin"}


def _sarvam_config() -> BackendConfig:
    return BackendConfig(
        "sarvam_saaras_v4",
        "sarvam",
        {
            "model": "saaras:v4",
            "language_code": "en-IN",
            "base_url": "https://api.sarvam.ai/speech-to-text",
            "max_retries": 2,
            "backoff_base_seconds": 0,
        },
    )


def test_sarvam_sends_expected_request_without_leaking_key(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")
    monkeypatch.setenv("SARVAM_API_KEY", "secret-key")
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return _Response(payload={"transcript": "  hello world  "})

    monkeypatch.setattr("voicerefine_eval.backends.sarvam.requests.post", fake_post)
    backend = SarvamBackend(_sarvam_config())
    backend.start()

    assert backend.transcribe(audio) == "hello world"
    assert captured["headers"]["api-subscription-key"] == "secret-key"
    assert captured["data"]["model"] == "saaras:v4"
    assert captured["data"]["language_code"] == "en-IN"
    assert "secret-key" not in str(backend.cache_signature())


def test_sarvam_retries_transient_network_failure(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")
    monkeypatch.setenv("SARVAM_API_KEY", "secret-key")
    responses = iter([
        requests.ConnectionError("temporary"),
        _Response(payload={"transcript": "recovered"}),
    ])

    def fake_post(*args, **kwargs):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr("voicerefine_eval.backends.sarvam.requests.post", fake_post)
    monkeypatch.setattr("voicerefine_eval.backends.sarvam.time.sleep", lambda seconds: None)
    backend = SarvamBackend(_sarvam_config())
    backend.start()
    assert backend.transcribe(audio) == "recovered"


def test_sarvam_does_not_retry_terminal_client_error(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")
    monkeypatch.setenv("SARVAM_API_KEY", "secret-key")
    calls = 0

    def fake_post(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _Response(status_code=401, text="unauthorized")

    monkeypatch.setattr("voicerefine_eval.backends.sarvam.requests.post", fake_post)
    backend = SarvamBackend(_sarvam_config())
    backend.start()
    with pytest.raises(TranscriptionError, match="Sarvam 401"):
        backend.transcribe(audio)
    assert calls == 1


def test_cloud_model_id_is_not_resolved_as_a_file_path():
    cfg = load_config()
    assert cfg.backends["sarvam_saaras_v4"].settings["model"] == "saaras:v4"
    assert Path(cfg.backends["crisp_v0823_parakeet_q4k"].settings["model"]).is_absolute()


def test_whisper_medium_matches_controlled_local_protocol():
    cfg = load_config()
    medium = cfg.backends["crisp_v0823_whisper_medium_en_q4k"]

    assert medium.type == "crispasr_server"
    assert medium.settings["backend"] == "whisper"
    assert medium.settings["language"] == "en"
    assert medium.settings["threads"] == 8
    assert medium.settings["gpu_backend"] == "cpu"
    assert medium.settings["quantization"] == "q4_k"
    assert medium.settings["no_punctuation"] is False
    assert medium.settings["runtime_version"] == "0.8.23"
    assert medium.settings["runtime_git_sha"] == "7d22deec"
    assert Path(medium.settings["model"]).name == "ggml-medium.en-q4_k.bin"


def _smallest_config() -> BackendConfig:
    return BackendConfig(
        "smallest_pulse_pro",
        "smallest",
        {
            "model": "pulse-pro",
            "language": "en",
            "base_url": "https://api.smallest.ai/waves/v1/stt/",
            "max_retries": 2,
            "backoff_base_seconds": 0,
        },
    )


def test_standard_pulse_config_uses_unified_endpoint(tmp_path, monkeypatch):
    cfg = load_config().backends["smallest_pulse"]

    assert cfg.type == "smallest"
    assert cfg.settings["model"] == "pulse"
    assert cfg.settings["language"] == "en"
    assert cfg.settings["base_url"] == "https://api.smallest.ai/waves/v1/stt/"
    assert cfg.settings["min_request_interval_seconds"] == 4.0

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav-bytes")
    monkeypatch.setenv("SMALLEST_API_KEY", "secret-key")
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return _Response(payload={"transcription": "hello world"})

    monkeypatch.setattr("voicerefine_eval.backends.smallest.requests.post", fake_post)
    backend = SmallestBackend(cfg)
    backend.start()

    assert backend.transcribe(audio) == "hello world"
    assert captured["params"] == {"model": "pulse", "language": "en"}


def test_smallest_sends_raw_audio_without_leaking_key(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav-bytes")
    monkeypatch.delenv("SMALLEST_API_KEY", raising=False)
    monkeypatch.setenv("SMALLESTAI_API_KEY", "secret-key")
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return _Response(payload={"transcription": "  hello world  "})

    monkeypatch.setattr("voicerefine_eval.backends.smallest.requests.post", fake_post)
    backend = SmallestBackend(_smallest_config())
    backend.start()

    assert backend.transcribe(audio) == "hello world"
    assert captured["url"] == "https://api.smallest.ai/waves/v1/stt/"
    assert captured["params"] == {"model": "pulse-pro", "language": "en"}
    assert captured["headers"]["Authorization"] == "Bearer secret-key"
    assert captured["headers"]["Content-Type"] == "application/octet-stream"
    assert captured["data"] == b"wav-bytes"
    assert "secret-key" not in str(backend.cache_signature())


def test_smallest_prefers_official_key_name(monkeypatch):
    monkeypatch.setenv("SMALLEST_API_KEY", "official-key")
    monkeypatch.setenv("SMALLESTAI_API_KEY", "alias-key")
    backend = SmallestBackend(_smallest_config())
    backend.start()
    assert backend._api_key == "official-key"


def test_smallest_retries_rate_limit(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")
    monkeypatch.setenv("SMALLEST_API_KEY", "secret-key")
    responses = iter([
        _Response(status_code=429, text="slow down", headers={"Retry-After": "0"}),
        _Response(payload={"transcription": "recovered"}),
    ])

    monkeypatch.setattr(
        "voicerefine_eval.backends.smallest.requests.post",
        lambda *args, **kwargs: next(responses),
    )
    monkeypatch.setattr("voicerefine_eval.backends.smallest.time.sleep", lambda seconds: None)
    backend = SmallestBackend(_smallest_config())
    backend.start()
    assert backend.transcribe(audio) == "recovered"
    assert backend.last_attempts == 2


def test_smallest_does_not_retry_terminal_client_error(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")
    monkeypatch.setenv("SMALLEST_API_KEY", "secret-key")
    calls = 0

    def fake_post(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _Response(status_code=401, text="unauthorized")

    monkeypatch.setattr("voicerefine_eval.backends.smallest.requests.post", fake_post)
    backend = SmallestBackend(_smallest_config())
    backend.start()
    with pytest.raises(TranscriptionError, match="Smallest.ai 401"):
        backend.transcribe(audio)
    assert calls == 1


def test_smallest_pacing_uses_untimed_prepare_hook(monkeypatch):
    cfg = _smallest_config()
    cfg.settings["min_request_interval_seconds"] = 3.5
    backend = SmallestBackend(cfg)
    monotonic = iter([10.0, 10.0, 11.0, 13.5])
    sleeps = []
    monkeypatch.setattr("voicerefine_eval.backends.smallest.time.monotonic", lambda: next(monotonic))
    monkeypatch.setattr("voicerefine_eval.backends.smallest.time.sleep", sleeps.append)

    backend.prepare_request()
    backend.prepare_request()

    assert sleeps == [2.5]
    assert backend.cache_signature()["min_request_interval_seconds"] == 3.5


def _deepgram_config(**overrides) -> BackendConfig:
    settings = {
        "model": "nova-3",
        "language": "en",
        "base_url": "https://api.deepgram.com/v1/listen",
        "punctuate": True,
        "smart_format": False,
        "max_retries": 2,
        "backoff_base_seconds": 0,
    }
    settings.update(overrides)
    return BackendConfig("deepgram_nova3", "deepgram", settings)


def _deepgram_payload(transcript: str) -> dict:
    return {"results": {"channels": [{"alternatives": [{"transcript": transcript}]}]}}


def test_deepgram_sends_expected_request_without_leaking_key(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "secret-key")
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return _Response(payload=_deepgram_payload("  hello world  "))

    monkeypatch.setattr("voicerefine_eval.backends.deepgram.requests.post", fake_post)
    backend = DeepgramBackend(_deepgram_config())
    backend.start()

    assert backend.transcribe(audio) == "hello world"
    assert captured["url"] == "https://api.deepgram.com/v1/listen"
    assert captured["headers"]["Authorization"] == "Token secret-key"
    assert captured["headers"]["Content-Type"] == "audio/wav"
    assert captured["params"]["model"] == "nova-3"
    assert captured["params"]["language"] == "en"
    # Query params must be explicit strings, not Python bools.
    assert captured["params"]["punctuate"] == "true"
    assert captured["params"]["smart_format"] == "false"
    assert captured["data"] == b"wav"
    assert "secret-key" not in str(backend.cache_signature())


def test_deepgram_smart_format_stays_off_by_default():
    # The Whisper normalizer must remain the only formatting pass applied.
    assert DeepgramBackend(_deepgram_config()).smart_format is False
    assert DeepgramBackend(_deepgram_config(smart_format=True)).smart_format is True


def test_deepgram_rejects_unexpected_response_shape(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "secret-key")

    monkeypatch.setattr(
        "voicerefine_eval.backends.deepgram.requests.post",
        lambda *a, **k: _Response(payload={"results": {"channels": []}}),
    )
    backend = DeepgramBackend(_deepgram_config())
    backend.start()

    with pytest.raises(TranscriptionError) as excinfo:
        backend.transcribe(audio)
    assert excinfo.value.category == "bad_response"


def test_deepgram_retries_transient_server_error(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "secret-key")
    responses = iter([
        _Response(status_code=503, text="unavailable"),
        _Response(payload=_deepgram_payload("recovered")),
    ])

    monkeypatch.setattr(
        "voicerefine_eval.backends.deepgram.requests.post",
        lambda *a, **k: next(responses),
    )
    backend = DeepgramBackend(_deepgram_config())
    backend.start()

    assert backend.transcribe(audio) == "recovered"
    assert backend.last_attempts == 2


def test_deepgram_does_not_retry_terminal_auth_failure(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "secret-key")
    calls = {"n": 0}

    def fake_post(*a, **k):
        calls["n"] += 1
        return _Response(status_code=401, text="unauthorized")

    monkeypatch.setattr("voicerefine_eval.backends.deepgram.requests.post", fake_post)
    backend = DeepgramBackend(_deepgram_config())
    backend.start()

    with pytest.raises(TranscriptionError) as excinfo:
        backend.transcribe(audio)
    assert excinfo.value.category == "http_401"
    assert calls["n"] == 1


def test_deepgram_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    assert DeepgramBackend(_deepgram_config()).is_available() is False


def test_build_backend_constructs_deepgram():
    assert isinstance(build_backend(_deepgram_config()), DeepgramBackend)


def _gnani_config(**overrides) -> BackendConfig:
    settings = {
        "model_label": "prisma-v2.5",
        "language_code": "en-IN",
        "base_url": "https://api.vachana.ai/stt/v3",
        "output_format": "verbatim",
        "max_retries": 2,
        "backoff_base_seconds": 0,
    }
    settings.update(overrides)
    return BackendConfig("gnani_prisma_v25", "gnani", settings)


def test_gnani_sends_expected_request_without_leaking_key(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")
    monkeypatch.setenv("GNANI_API_KEY", "secret-key")
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return _Response(payload={"success": True, "transcript": "  hello world  "})

    monkeypatch.setattr("voicerefine_eval.backends.gnani.requests.post", fake_post)
    backend = GnaniBackend(_gnani_config())
    backend.start()

    assert backend.transcribe(audio) == "hello world"
    assert captured["url"] == "https://api.vachana.ai/stt/v3"
    assert captured["headers"]["X-API-Key-ID"] == "secret-key"
    assert captured["data"]["language_code"] == "en-IN"
    assert captured["data"]["format"] == "verbatim"
    # Audio goes as multipart under the documented field name.
    assert captured["files"]["audio_file"][0] == "recording.wav"
    assert captured["files"]["audio_file"][1] == b"wav"
    assert "secret-key" not in str(backend.cache_signature())


def test_gnani_never_sends_model_label_as_a_parameter(tmp_path, monkeypatch):
    """The endpoint has no model selector; the label is provenance only."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")
    monkeypatch.setenv("GNANI_API_KEY", "secret-key")
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(**kwargs)
        return _Response(payload={"transcript": "ok"})

    monkeypatch.setattr("voicerefine_eval.backends.gnani.requests.post", fake_post)
    backend = GnaniBackend(_gnani_config())
    backend.start()
    backend.transcribe(audio)

    assert "model" not in captured["data"]
    assert "model_label" not in captured["data"]
    # ...but it must stay in the cache signature so a relabel invalidates cache.
    assert backend.cache_signature()["model_label"] == "prisma-v2.5"


def test_gnani_rejects_missing_transcript(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")
    monkeypatch.setenv("GNANI_API_KEY", "secret-key")
    monkeypatch.setattr(
        "voicerefine_eval.backends.gnani.requests.post",
        lambda *a, **k: _Response(payload={"success": True}),
    )
    backend = GnaniBackend(_gnani_config())
    backend.start()

    with pytest.raises(TranscriptionError) as excinfo:
        backend.transcribe(audio)
    assert excinfo.value.category == "bad_response"


def test_gnani_retries_transient_server_error(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")
    monkeypatch.setenv("GNANI_API_KEY", "secret-key")
    responses = iter([
        _Response(status_code=502, text="bad gateway"),
        _Response(payload={"transcript": "recovered"}),
    ])
    monkeypatch.setattr(
        "voicerefine_eval.backends.gnani.requests.post",
        lambda *a, **k: next(responses),
    )
    backend = GnaniBackend(_gnani_config())
    backend.start()

    assert backend.transcribe(audio) == "recovered"
    assert backend.last_attempts == 2


def test_gnani_does_not_retry_terminal_auth_failure(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")
    monkeypatch.setenv("GNANI_API_KEY", "secret-key")
    calls = {"n": 0}

    def fake_post(*a, **k):
        calls["n"] += 1
        return _Response(status_code=403, text="forbidden")

    monkeypatch.setattr("voicerefine_eval.backends.gnani.requests.post", fake_post)
    backend = GnaniBackend(_gnani_config())
    backend.start()

    with pytest.raises(TranscriptionError) as excinfo:
        backend.transcribe(audio)
    assert excinfo.value.category == "http_403"
    assert calls["n"] == 1


def test_gnani_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("GNANI_API_KEY", raising=False)
    assert GnaniBackend(_gnani_config()).is_available() is False


def test_build_backend_constructs_gnani():
    assert isinstance(build_backend(_gnani_config()), GnaniBackend)
