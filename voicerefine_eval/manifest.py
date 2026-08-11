"""Run provenance manifest.

DESIGN.md "Reproducibility Record": each completed run writes
``results/run_manifest.json`` with git commit, dataset identity, Python and
dependency versions, OS/hardware summary, active backend configs, local model and
executable hashes, start/completion times, and cache policy — and NO secrets.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from .config import REPO_ROOT, EvalConfig

RUN_MANIFEST_PATH = REPO_ROOT / "results" / "run_manifest.json"

# Dependencies whose versions materially affect results.
_TRACKED_PACKAGES = [
    "datasets",
    "jiwer",
    "numpy",
    "sherpa-onnx",
    "sherpa-onnx-core",
    "soundfile",
    "soxr",
    "whisper-normalizer",
    "requests",
]


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return bool(out.stdout.strip()) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_run_manifest(
    cfg: EvalConfig,
    *,
    backend_signatures: dict[str, dict[str, Any]],
    started_at: str,
    completed_at: str,
    cache_enabled: bool,
    subset_size: int,
    debug: bool,
) -> dict[str, Any]:
    """Assemble the provenance record. ``backend_signatures`` are the per-backend
    ``cache_signature()`` dicts (already secret-free: hashes + config, no keys)."""
    return {
        "git": {"commit": _git_commit(), "dirty": _git_dirty()},
        "dataset": {
            "name": cfg.dataset.name,
            "revision": cfg.dataset.revision,
            "split": cfg.dataset.split,
            "seed": cfg.dataset.seed,
            "subset_size": subset_size,
            "debug": debug,
        },
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
        },
        "dependencies": _package_versions(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": _safe_cpu_count(),
        },
        "backends": backend_signatures,
        "timing": {"started_at": started_at, "completed_at": completed_at},
        "cache": {"enabled": cache_enabled},
    }


def _safe_cpu_count() -> int | None:
    import os

    return os.cpu_count()
