"""Manifest loading and validation for the provisional eligibility manifests.

This module is read-only with respect to the audit artifacts: it never edits
the CSV files, labels, eligibility decisions, or image files. It resolves
manifest image paths against a verified image root and reports problems
without silently rewriting paths.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

# Columns that the primary/secondary eligibility manifests are expected to
# provide. We do not assume these names up front; load_manifest() validates
# them dynamically.
PRIMARY_REQUIRED_COLUMNS = {
    "manifest_status",
    "manifest_version",
    "approval_reference",
    "image_path",
    "original_label",
    "task_label",
    "task_label_basis",
    "source_family",
    "source_status",
    "patient_group_key",
    "split_group_id",
    "duplicate_group_id",
    "quarantine_label_conflict",
    "quarantine_quality_flag",
    "eligibility",
    "notes",
}

SECONDARY_REQUIRED_COLUMNS = PRIMARY_REQUIRED_COLUMNS


class ManifestValidationError(ValueError):
    """Raised when a manifest is missing required columns or has structural issues."""


class ImagePathDoesNotExistError(ValueError):
    """Raised when a manifest references an image that is not present on disk."""


class DuplicateManifestRowError(ValueError):
    """Raised when the same image_path appears more than once in a manifest."""


Record = Dict[str, str]


def _read_csv(path: Path) -> List[Record]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ManifestValidationError(f"No header found in {path}")
        rows = list(reader)
    return rows


def _norm_flag(value: str | None) -> str:
    if value is None:
        return "no"
    v = str(value).strip().lower()
    if v in {"", "none", "n/a", "na", "null"}:
        return "no"
    return "yes" if v == "yes" else "no"


def _make_record(raw: Record) -> Record:
    """Normalize a CSV row into the manifest record schema used by this package.

    The audit manifests are authoritative. We normalize recognized
    boolean-ish and sentinel fields into stable string values so downstream
    code has a predictable API. We never rewrite labels, eligibility decisions,
    or other clinical/audit content.
    """
    row = dict(raw)
    row["quarantine_quality_flag"] = _norm_flag(row.get("quarantine_quality_flag"))
    row["quarantine_label_conflict"] = _norm_flag(row.get("quarantine_label_conflict"))
    return row


def validate_columns(path: Path, required: Sequence[str], task: str) -> None:
    rows = _read_csv(path)
    if not rows:
        raise ManifestValidationError(f"{task} manifest has no data rows: {path}")
    present = set(rows[0].keys())
    missing = sorted(set(required) - present)
    if missing:
        raise ManifestValidationError(
            f"{task} manifest is missing required columns: {missing}"
        )


def load_manifest(
    path: Path,
    task: str,
    image_root: Path | None = None,
    *,
    require_files_exist: bool = False,
) -> List[Record]:
    """Load and validate an eligibility manifest.

    Parameters
    ----------
    path:
        CSV path for the manifest.
    task:
        One-line description used in error messages, e.g. "primary" or "secondary".
    image_root:
        Optional root used to resolve relative ``image_path`` values. If provided,
        resolved paths are not checked by default unless ``require_files_exist`` is
        True.
    require_files_exist:
        If True, raise ``ImagePathDoesNotExistError`` for any missing image file.
    """
    required = PRIMARY_REQUIRED_COLUMNS if task.lower().startswith("prim") else SECONDARY_REQUIRED_COLUMNS
    validate_columns(path, required, task)

    rows = _read_csv(path)
    rows = [_make_record(row) for row in rows]

    # Structural checks
    counts: Dict[str, int] = Counter()
    for row in rows:
        p = row.get("image_path") or ""
        counts[p] += 1

    dups = sorted(p for p, n in counts.items() if n > 1)
    if dups:
        raise DuplicateManifestRowError(
            f"{task} manifest has duplicate image_path rows: {dups[:5]}"
        )

    # Path-existence checks
    missing_files: List[str] = []
    for row in rows:
        p = row.get("image_path") or ""
        resolved = _resolve_path(p, image_root)
        row["_resolved"] = str(resolved)
        if require_files_exist and not resolved.exists():
            missing_files.append(p)

    if missing_files:
        raise ImagePathDoesNotExistError(
            f"{task} manifest references missing files: {missing_files[:10]}"
        )

    return rows


def _resolve_path(path_value: str, image_root: Path | None) -> Path:
    """Resolve a manifest path against an image root without rewriting the source.

    The manifest value is preserved verbatim in the record. This function only
    computes a candidate absolute path for checks/lookups.
    """
    if image_root is None:
        return Path(path_value)
    return image_root / Path(path_value)


def summarize_rows(rows: List[Record]) -> Dict:
    """Return a lightweight summary for reporting/inspection.

    This is intentionally small and stable across manifest versions.
    """
    eligibility: Dict[str, int] = {}
    original_labels: Dict[str, int] = {}
    source_families: Dict[str, int] = {}
    quarantine: Dict[str, Dict[str, int]] = {
        "quality": {"yes": 0, "no": 0},
        "conflict": {"yes": 0, "no": 0},
    }
    for row in rows:
        eligibility[row.get("eligibility", "")] = eligibility.get(row.get("eligibility", ""), 0) + 1
        original_labels[row.get("original_label", "")] = original_labels.get(row.get("original_label", ""), 0) + 1
        source_families[row.get("source_family", "")] = source_families.get(row.get("source_family", ""), 0) + 1
        qf = row.get("quarantine_quality_flag", "no")
        cf = row.get("quarantine_label_conflict", "no")
        quarantine["quality"][qf] = quarantine["quality"].get(qf, 0) + 1
        quarantine["conflict"][cf] = quarantine["conflict"].get(cf, 0) + 1

    return {
        "row_count": len(rows),
        "eligibility": eligibility,
        "original_labels": original_labels,
        "source_families": source_families,
        "quarantine": quarantine,
    }
