"""Tests for manifest loading and validation."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest
from dataset.manifest import (
    DuplicateManifestRowError,
    ImagePathDoesNotExistError,
    ManifestValidationError,
    load_manifest,
    summarize_rows,
)

AUDIT_DIR = Path(__file__).resolve().parents[1] / "audit_output" / "pretraining_audit"
PRIMARY_CSV = AUDIT_DIR / "PROVISIONAL_ELIGIBILITY_PRIMARY.csv"
SECONDARY_CSV = AUDIT_DIR / "PROVISIONAL_ELIGIBILITY_SECONDARY.csv"
IMAGE_ROOT = Path(r"C:\Users\anous\Downloads\Diabetic Retinopathy and Retinal Diseases Fundus I")

_PRIMARY_REQUIRED_FIELDS = {
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


def _default_row(overrides=None):
    row = {
        "manifest_status": "x",
        "manifest_version": "x",
        "approval_reference": "x",
        "image_path": "a.jpg",
        "original_label": "x",
        "task_label": "x",
        "task_label_basis": "x",
        "source_family": "x",
        "source_status": "x",
        "patient_group_key": "x",
        "split_group_id": "x",
        "duplicate_group_id": "",
        "quarantine_label_conflict": "no",
        "quarantine_quality_flag": "no",
        "eligibility": "x",
        "notes": "x",
    }
    if overrides:
        row.update(overrides)
    return row


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_primary_manifest_loads():
    rows = load_manifest(PRIMARY_CSV, "primary", image_root=IMAGE_ROOT)
    assert isinstance(rows, list)
    assert len(rows) == 22453


def test_secondary_manifest_loads():
    rows = load_manifest(SECONDARY_CSV, "secondary", image_root=IMAGE_ROOT)
    assert len(rows) == 22453


def test_primary_has_required_columns():
    rows = load_manifest(PRIMARY_CSV, "primary", image_root=IMAGE_ROOT)
    present = set(rows[0].keys())
    expected = _PRIMARY_REQUIRED_FIELDS | {"_resolved"}
    assert expected.issubset(present)


def test_primary_missing_files_integration():
    rows = load_manifest(PRIMARY_CSV, "primary", image_root=IMAGE_ROOT)
    missing = [row["image_path"] for row in rows if not Path(row["_resolved"]).exists()]
    assert not missing, f"Primary manifest references missing files: {missing[:5]}"


def test_secondary_missing_files_integration():
    rows = load_manifest(SECONDARY_CSV, "secondary", image_root=IMAGE_ROOT)
    missing = [row["image_path"] for row in rows if not Path(row["_resolved"]).exists()]
    assert not missing, f"Secondary manifest references missing files: {missing[:5]}"


def test_primary_no_duplicate_image_paths(tmp_path):
    csv_path = tmp_path / "primary_dup.csv"
    _write_csv(
        csv_path,
        [
            _default_row({"image_path": "a.jpg", "eligibility": "x"}),
            _default_row({"image_path": "b.jpg", "eligibility": "x"}),
            _default_row({"image_path": "a.jpg", "eligibility": "x"}),
        ],
    )
    with pytest.raises(DuplicateManifestRowError):
        load_manifest(csv_path, "primary", image_root=tmp_path)


def test_missing_required_column(tmp_path):
    csv_path = tmp_path / "bad.csv"
    _write_csv(csv_path, [{"image_path": "a.jpg"}])
    with pytest.raises(ManifestValidationError):
        load_manifest(csv_path, "primary", image_root=tmp_path)


def test_missing_file_raises_when_require_files_exist(tmp_path):
    csv_path = tmp_path / "primary.csv"
    _write_csv(csv_path, [_default_row({"image_path": "a.jpg", "eligibility": "x"})])
    with pytest.raises(ImagePathDoesNotExistError):
        load_manifest(csv_path, "primary", image_root=tmp_path, require_files_exist=True)


def test_summarize_primary():
    rows = load_manifest(PRIMARY_CSV, "primary", image_root=IMAGE_ROOT)
    s = summarize_rows(rows)
    assert s["row_count"] == 22453
    assert s["eligibility"]["eligible_provisional"] == 14201


def test_summarize_secondary():
    rows = load_manifest(SECONDARY_CSV, "secondary", image_root=IMAGE_ROOT)
    s = summarize_rows(rows)
    assert s["row_count"] == 22453
    assert s["eligibility"]["eligible_provisional"] == 22127
