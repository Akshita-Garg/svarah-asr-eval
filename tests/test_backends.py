from pathlib import Path

import pytest
import requests

from voicerefine_eval.backends import build_backend
from voicerefine_eval.backends.base import TranscriptionError
from voicerefine_eval.backends.parakeet_crispasr import CrispAsrServerBackend
from voicerefine_eval.backends.sarvam import SarvamBackend
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
