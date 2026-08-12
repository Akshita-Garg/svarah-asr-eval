import json

import pytest

from voicerefine_eval.hashing import write_json_atomic
from voicerefine_eval.merge import merge_runs
from voicerefine_eval.report import read_per_utterance_csv, write_per_utterance_csv
from voicerefine_eval.results import UtteranceResult


DATASET = {
    "name": "ai4bharat/Svarah",
    "revision": "revision-1",
    "split": "test",
    "seed": 42,
    "subset_size": 2,
    "debug": False,
}


def _result(backend_id: str, eval_id: str, reference: str) -> UtteranceResult:
    return UtteranceResult(
        backend_id=backend_id,
        eval_id=eval_id,
        status="success",
        inference_seconds=0.2,
        audio_seconds=1.0,
        rtf=0.2,
        raw_reference=reference,
        raw_hypothesis=reference,
        normalized_reference=reference,
        normalized_hypothesis=reference,
        hits=1,
        substitutions=0,
        deletions=0,
        insertions=0,
        reference_words=1,
        wer=0.0,
    )


def _run(tmp_path, name: str, backend_id: str, *, revision: str = "revision-1"):
    run_dir = tmp_path / name
    rows = [_result(backend_id, "u1", "one"), _result(backend_id, "u2", "two")]
    write_per_utterance_csv(rows, run_dir / "per_utterance.csv")
    manifest = {
        "dataset": {**DATASET, "revision": revision},
        "backends": {backend_id: {"backend_id": backend_id, "type": "test"}},
        "backend_summaries": {backend_id: {"startup_seconds": 0.5}},
    }
    write_json_atomic(run_dir / "run_manifest.json", manifest)
    return run_dir


def test_csv_round_trip_preserves_typed_results(tmp_path):
    path = tmp_path / "results.csv"
    original = [_result("backend-a", "u1", "hello")]
    write_per_utterance_csv(original, path)
    loaded = read_per_utterance_csv(path)
    assert loaded == original


def test_merge_combines_runs_without_rerunning_backends(tmp_path, monkeypatch):
    monkeypatch.setattr("voicerefine_eval.merge.REPO_ROOT", tmp_path)
    first = _run(tmp_path, "first", "backend-a")
    second = _run(tmp_path, "second", "backend-b")
    output = tmp_path / "combined"

    manifest = merge_runs([first, second], output)

    rows = read_per_utterance_csv(output / "per_utterance.csv")
    assert len(rows) == 4
    assert {row.backend_id for row in rows} == {"backend-a", "backend-b"}
    assert manifest["backend_count"] == 2
    assert manifest["shared_successful_utterances"] == 2
    assert json.loads((output / "comparison_manifest.json").read_text())["row_count"] == 4
    assert {source["path"] for source in manifest["source_runs"]} == {"first", "second"}


def test_merge_rejects_dataset_mismatch(tmp_path):
    first = _run(tmp_path, "first", "backend-a")
    second = _run(tmp_path, "second", "backend-b", revision="different")

    with pytest.raises(ValueError, match="Dataset mismatch"):
        merge_runs([first, second], tmp_path / "combined")


def test_merge_rejects_duplicate_backend_rows(tmp_path):
    first = _run(tmp_path, "first", "backend-a")
    second = _run(tmp_path, "second", "backend-a")

    with pytest.raises(ValueError, match="Duplicate result row"):
        merge_runs([first, second], tmp_path / "combined")


def test_merge_can_select_backends_from_multi_backend_source(tmp_path):
    first = _run(tmp_path, "first", "backend-a")
    second = _run(tmp_path, "second", "backend-b")
    output = tmp_path / "combined"

    manifest = merge_runs(
        [first, second],
        output,
        backend_ids={"backend-b"},
    )

    rows = read_per_utterance_csv(output / "per_utterance.csv")
    assert {row.backend_id for row in rows} == {"backend-b"}
    assert manifest["selected_backends"] == ["backend-b"]


def test_merge_rejects_missing_selected_backend(tmp_path):
    first = _run(tmp_path, "first", "backend-a")
    second = _run(tmp_path, "second", "backend-b")

    with pytest.raises(ValueError, match="Requested backends were not found"):
        merge_runs(
            [first, second],
            tmp_path / "combined",
            backend_ids={"backend-c"},
        )
