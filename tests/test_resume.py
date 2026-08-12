import json

import pytest

from voicerefine_eval.config import load_config
from voicerefine_eval.dataset import SubsetEntry
from voicerefine_eval.hashing import write_json_atomic
from voicerefine_eval.report import write_per_utterance_csv
from voicerefine_eval.results import UtteranceResult
from voicerefine_eval.run import _load_resume


def _entry(eval_id: str, reference: str) -> SubsetEntry:
    return SubsetEntry(eval_id=eval_id, row_index=0, reference=reference)


def _row(eval_id: str, reference: str, status: str) -> UtteranceResult:
    return UtteranceResult(
        backend_id="smallest_pulse_pro",
        eval_id=eval_id,
        status=status,
        raw_reference=reference,
        raw_hypothesis=reference if status == "success" else "",
        normalized_reference=reference if status == "success" else "",
        normalized_hypothesis=reference if status == "success" else "",
        hits=1 if status == "success" else None,
        substitutions=0 if status == "success" else None,
        deletions=0 if status == "success" else None,
        insertions=0 if status == "success" else None,
        reference_words=1 if status == "success" else None,
        wer=0.0 if status == "success" else None,
    )


def test_resume_retains_only_validated_successes(tmp_path):
    cfg = load_config()
    entries = [_entry("u1", "one"), _entry("u2", "two")]
    signature = {
        "backend_id": "smallest_pulse_pro",
        "type": "smallest",
        "model": "pulse-pro",
        "min_request_interval_seconds": 3.5,
    }
    write_per_utterance_csv(
        [_row("u1", "one", "success"), _row("u2", "two", "failure")],
        tmp_path / "per_utterance.csv",
    )
    manifest = {
        "dataset": {
            "name": cfg.dataset.name,
            "revision": cfg.dataset.revision,
            "split": cfg.dataset.split,
            "seed": cfg.dataset.seed,
            "subset_size": 2,
            "debug": False,
        },
        "backends": {
            "smallest_pulse_pro": {
                "backend_id": "smallest_pulse_pro",
                "type": "smallest",
                "model": "pulse-pro",
            }
        },
    }
    write_json_atomic(tmp_path / "run_manifest.json", manifest)

    successes, provenance = _load_resume(
        tmp_path,
        backend_id="smallest_pulse_pro",
        signature=signature,
        entries=entries,
        cfg=cfg,
        debug=False,
    )

    assert set(successes) == {"u1"}
    assert provenance["retained_successes"] == 1
    assert provenance["retried_failures"] == 1


def test_resume_rejects_reference_mismatch(tmp_path):
    cfg = load_config()
    entries = [_entry("u1", "expected")]
    write_per_utterance_csv([_row("u1", "different", "success")], tmp_path / "per_utterance.csv")
    write_json_atomic(
        tmp_path / "run_manifest.json",
        {
            "dataset": {
                "name": cfg.dataset.name,
                "revision": cfg.dataset.revision,
                "split": cfg.dataset.split,
                "seed": cfg.dataset.seed,
                "subset_size": 1,
                "debug": False,
            },
            "backends": {"smallest_pulse_pro": {"backend_id": "smallest_pulse_pro"}},
        },
    )

    with pytest.raises(ValueError, match="Reference mismatch"):
        _load_resume(
            tmp_path,
            backend_id="smallest_pulse_pro",
            signature={"backend_id": "smallest_pulse_pro"},
            entries=entries,
            cfg=cfg,
            debug=False,
        )
