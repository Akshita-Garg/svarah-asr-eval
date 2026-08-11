"""Svarah dataset loading, schema validation, and the committed subset manifest.

DESIGN.md "Dataset And Subset":
- The loader must inspect and VALIDATE the live schema before depending on field
  names (Svarah's exact column names are confirmed at load time, not assumed).
- A seeded sample of row indexes + stable eval IDs is written to
  ``data/subset_manifest.json`` and committed. Later runs read the manifest
  rather than resampling.

Loading the dataset requires HF auth for this gated repo (HF_TOKEN in .env).
Reading an existing manifest does NOT require the dataset, so downstream steps
can run without re-downloading once the manifest and prepared audio exist.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import REPO_ROOT, DatasetConfig
from .hashing import write_json_atomic

MANIFEST_PATH = REPO_ROOT / "data" / "subset_manifest.json"

# Candidate field names; the actual one is confirmed against the live schema.
_TRANSCRIPT_CANDIDATES = ("text", "transcript", "transcription", "sentence", "normalized_text")
_ID_CANDIDATES = ("id", "utt_id", "utterance_id", "audio_filepath", "file", "filename", "path")
_DURATION_CANDIDATES = ("duration", "duration_seconds", "length", "audio_duration")


@dataclass(frozen=True)
class SubsetEntry:
    """One utterance in the frozen subset."""

    eval_id: str
    row_index: int
    reference: str
    source_id: str | None = None
    duration_seconds: float | None = None


@dataclass(frozen=True)
class SchemaInfo:
    audio_field: str
    transcript_field: str
    id_field: str | None
    duration_field: str | None
    column_names: list[str]


def _detect_field(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def validate_schema(dataset) -> SchemaInfo:
    """Confirm the live dataset exposes an audio column and a transcript column.

    Raises a clear error listing the actual columns if either is missing, so a
    dataset revision that renames fields fails loudly instead of silently
    scoring against the wrong column.
    """
    from datasets import Audio

    features = dataset.features
    columns = list(features.keys())

    audio_field = None
    for name, feat in features.items():
        if isinstance(feat, Audio):
            audio_field = name
            break
    if audio_field is None:
        raise ValueError(
            f"No Audio feature found in Svarah schema. Columns: {columns}"
        )

    transcript_field = _detect_field(columns, _TRANSCRIPT_CANDIDATES)
    if transcript_field is None:
        raise ValueError(
            "Could not find a transcript column among "
            f"{_TRANSCRIPT_CANDIDATES}. Actual columns: {columns}"
        )

    return SchemaInfo(
        audio_field=audio_field,
        transcript_field=transcript_field,
        id_field=_detect_field(columns, _ID_CANDIDATES),
        duration_field=_detect_field(columns, _DURATION_CANDIDATES),
        column_names=columns,
    )


def load_svarah(cfg: DatasetConfig):
    """Load the pinned Svarah split (non-streaming, so rows are index-addressable).

    Audio decoding is disabled here (``decode=False``) so building the manifest
    doesn't decode every waveform; audio preparation decodes on demand.
    """
    from datasets import Audio, load_dataset

    ds = load_dataset(cfg.name, split=cfg.split, revision=cfg.revision)
    # Keep the audio column but avoid eager decoding while sampling metadata.
    if any(_is_audio_feature(f) for f in ds.features.values()):
        audio_col = next(n for n, f in ds.features.items() if _is_audio_feature(f))
        ds = ds.cast_column(audio_col, Audio(decode=False))
    return ds


def _is_audio_feature(feat) -> bool:
    from datasets import Audio

    return isinstance(feat, Audio)


def build_manifest(cfg: DatasetConfig, *, force: bool = False) -> list[SubsetEntry]:
    """Sample the subset (seed-stable) and write the committed manifest.

    If a manifest already exists and ``force`` is False, it is returned unchanged
    so the subset stays frozen across runs.
    """
    if MANIFEST_PATH.exists() and not force:
        return read_manifest()

    ds = load_svarah(cfg)
    schema = validate_schema(ds)

    n_rows = ds.num_rows
    rng = random.Random(cfg.seed)
    count = min(cfg.subset_size, n_rows)
    indexes = sorted(rng.sample(range(n_rows), count))

    entries: list[SubsetEntry] = []
    for seq, row_index in enumerate(indexes):
        row = ds[row_index]
        reference = str(row[schema.transcript_field])
        source_id = _clean_source_id(row, schema)
        duration = (
            float(row[schema.duration_field])
            if schema.duration_field and row.get(schema.duration_field) is not None
            else None
        )
        entries.append(
            SubsetEntry(
                eval_id=f"svarah_{cfg.split}_{seq:04d}",
                row_index=row_index,
                reference=reference,
                source_id=source_id,
                duration_seconds=duration,
            )
        )

    manifest = {
        "dataset": cfg.name,
        "split": cfg.split,
        "revision": cfg.revision,
        "seed": cfg.seed,
        "subset_size": count,
        "total_rows": n_rows,
        "schema": asdict_schema(schema),
        "entries": [asdict(e) for e in entries],
    }
    write_json_atomic(MANIFEST_PATH, manifest)
    return entries


def _clean_source_id(row, schema: SchemaInfo) -> str | None:
    """A short, human-readable id for an entry.

    The id candidate can resolve to the audio column itself (Svarah's id field is
    ``audio_filepath``, which is the Audio feature). With ``decode=False`` that
    value is the audio STRUCT ``{bytes, path}`` — stringifying it would embed the
    raw audio bytes in the manifest (it ballooned to 136 MB before this fix). So
    if the id value is an audio struct, use only its ``path`` (the filename).
    """
    if not schema.id_field:
        return None
    value = row[schema.id_field]
    if isinstance(value, dict):
        return value.get("path")
    return str(value)


def asdict_schema(schema: SchemaInfo) -> dict:
    return {
        "audio_field": schema.audio_field,
        "transcript_field": schema.transcript_field,
        "id_field": schema.id_field,
        "duration_field": schema.duration_field,
        "column_names": schema.column_names,
    }


def read_manifest(path: Path = MANIFEST_PATH) -> list[SubsetEntry]:
    import json

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        SubsetEntry(
            eval_id=e["eval_id"],
            row_index=e["row_index"],
            reference=e["reference"],
            source_id=e.get("source_id"),
            duration_seconds=e.get("duration_seconds"),
        )
        for e in data["entries"]
    ]


def select_entries(entries: list[SubsetEntry], *, debug: bool, cfg: DatasetConfig) -> list[SubsetEntry]:
    """Debug mode uses the first ``debug_subset_size`` entries of the frozen subset.

    Making the debug set a strict prefix of the full subset keeps it fully
    deterministic and a genuine subset (no separate sampling to drift out of sync).
    """
    if debug:
        return entries[: cfg.debug_subset_size]
    return entries
