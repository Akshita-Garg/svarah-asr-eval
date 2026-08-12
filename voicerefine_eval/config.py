"""Configuration loading: TOML file + environment overrides + .env.

DESIGN.md: "Backends are selected through configuration." The committed
``config/eval.toml`` holds the experiment definition (pinned dataset revision,
per-backend runtime settings). Per-machine secrets and path overrides come from
environment variables / a gitignored ``.env`` file — never from the committed
config or any result artifact.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Repo root = parent of this package directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "eval.toml"


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (avoids a python-dotenv dependency).

    Parses ``KEY=VALUE`` lines, ignores blanks and ``#`` comments, strips
    surrounding quotes, and does NOT overwrite variables already set in the real
    environment (so an exported var wins over the file).
    """
    path = path or (REPO_ROOT / ".env")
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _resolve(path_str: str) -> Path:
    """Resolve a config path relative to the repo root unless already absolute."""
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    split: str
    revision: str
    seed: int
    subset_size: int
    debug_subset_size: int


@dataclass(frozen=True)
class BackendConfig:
    """A backend's identity plus its type-specific settings.

    ``settings`` is the raw table from the TOML with paths already resolved and
    env overrides applied. Backend classes read what they need from it. Keeping
    it as a dict (rather than N dataclasses) keeps the cache key — a hash of this
    versioned config — trivially serializable.
    """

    backend_id: str
    type: str
    settings: dict[str, Any]


@dataclass(frozen=True)
class EvalConfig:
    dataset: DatasetConfig
    active_backend_ids: list[str]
    backends: dict[str, BackendConfig] = field(default_factory=dict)
    config_path: Path = DEFAULT_CONFIG_PATH


def _env(name: str) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else None


def _apply_env_overrides(backend_id: str, type_: str, settings: dict[str, Any]) -> dict[str, Any]:
    """Overlay the same environment variables VoiceRefine Desktop honors."""
    s = dict(settings)
    if type_ == "whisper_sherpa":
        if (v := _env("VOICEREFINE_SHERPA_MODEL_DIR")):
            s["model_dir"] = v
        if (v := _env("VOICEREFINE_SHERPA_PRECISION")):
            s["precision"] = v
        if (v := _env("VOICEREFINE_SHERPA_THREADS")):
            s["num_threads"] = int(v)
    elif type_ == "crispasr_server":
        if (v := _env("VOICEREFINE_CRISPASR_BIN")):
            s["bin"] = v
        model_env = {
            "parakeet": "VOICEREFINE_CRISPASR_PARAKEET_MODEL",
            "cohere": "VOICEREFINE_CRISPASR_COHERE_MODEL",
            "whisper": "VOICEREFINE_CRISPASR_WHISPER_MODEL",
        }.get(s.get("backend"))
        if model_env and (v := _env(model_env)):
            s["model"] = v
        if (v := _env("VOICEREFINE_CRISPASR_PORT")):
            s["port"] = int(v)
        if (v := _env("VOICEREFINE_CRISPASR_THREADS")):
            s["threads"] = int(v)
        if (v := _env("VOICEREFINE_CRISPASR_START_TIMEOUT_MS")):
            s["start_timeout_ms"] = int(v)
        if (v := _env("VOICEREFINE_CRISPASR_TIMEOUT_MS")):
            s["request_timeout_ms"] = int(v)
        if (v := _env("VOICEREFINE_CRISPASR_GPU_BACKEND")):
            s["gpu_backend"] = v
    return s


def _path_keys_for_backend(type_: str) -> set[str]:
    """Return only settings that are filesystem paths for this backend type.

    Cloud backends use ``model`` as an API identifier (for example,
    ``saaras:v4``), while CrispASR uses it as an actual model file path.
    """
    if type_ == "whisper_sherpa":
        return {"model_dir"}
    if type_ == "crispasr_server":
        return {"bin", "model"}
    return set()


def load_config(config_path: Path | None = None) -> EvalConfig:
    """Load ``config/eval.toml``, apply env overrides, resolve paths."""
    load_dotenv()
    config_path = config_path or DEFAULT_CONFIG_PATH
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    d = raw["dataset"]
    dataset = DatasetConfig(
        name=d["name"],
        split=d["split"],
        revision=d["revision"],
        seed=int(d["seed"]),
        subset_size=int(d["subset_size"]),
        debug_subset_size=int(d["debug_subset_size"]),
    )

    backends_table = raw["backends"]
    active = list(backends_table["active"])

    backends: dict[str, BackendConfig] = {}
    for backend_id, table in backends_table.items():
        if backend_id == "active" or not isinstance(table, dict):
            continue
        type_ = table["type"]
        settings = {k: v for k, v in table.items() if k != "type"}
        settings = _apply_env_overrides(backend_id, type_, settings)
        for key in _path_keys_for_backend(type_):
            if key in settings and isinstance(settings[key], str):
                settings[key] = str(_resolve(settings[key]))
        backends[backend_id] = BackendConfig(backend_id, type_, settings)

    return EvalConfig(
        dataset=dataset,
        active_backend_ids=active,
        backends=backends,
        config_path=config_path,
    )
