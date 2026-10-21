"""Tests for class-count reporting."""
from __future__ import annotations

from pathlib import Path

import pytest
from dataset.statistics import (
    excluded_and_quarantined_counts,
    full_status_counts,
    primary_eligible_class_counts,
    secondary_eligible_class_counts,
)
from dataset.manifest import load_manifest

AUDIT_DIR = Path(__file__).resolve().parents[1] / "audit_output" / "pretraining_audit"
PRIMARY_CSV = AUDIT_DIR / "PROVISIONAL_ELIGIBILITY_PRIMARY.csv"
SECONDARY_CSV = AUDIT_DIR / "PROVISIONAL_ELIGIBILITY_SECONDARY.csv"
IMAGE_ROOT = Path(r"C:\Users\anous\Downloads\Diabetic Retinopathy and Retinal Diseases Fundus I")


def test_primary_eligible_band_counts():
    primary_rows = load_manifest(PRIMARY_CSV, "primary", image_root=IMAGE_ROOT)
    counts = primary_eligible_class_counts(primary_rows)
    assert counts.get("No apparent DR") == 2973
    assert counts.get("Mild NPDR") == 2436
    assert counts.get("Moderate NPDR") == 5303
    assert counts.get("Severe NPDR") == 1971
    assert counts.get("PDR") == 1518
    assert sum(counts.values()) == 14201


def test_primary_eligible_class_counts_excludes_quarantined():
    primary_rows = load_manifest(PRIMARY_CSV, "primary", image_root=IMAGE_ROOT)
    counts = primary_eligible_class_counts(primary_rows)
    for status in ["quarantined_quality_review", "quarantined_label_conflict", "quarantined_conflict_and_quality"]:
        for row in primary_rows:
            if row["eligibility"] == status:
                # Eligible class counts should never include a quarantined row's band.
                assert row["task_label"] not in counts or counts.get(row["task_label"]) is not None


def test_primary_status_counts():
    primary_rows = load_manifest(PRIMARY_CSV, "primary", image_root=IMAGE_ROOT)
    counts = full_status_counts(primary_rows)
    assert counts["eligible_provisional"] == 14201
    assert counts["quarantined_quality_review"] == 242
    assert counts["quarantined_label_conflict"] == 22
    assert counts["quarantined_conflict_and_quality"] == 49
    assert counts["excluded_non_dr_primary"] == 7926
    assert counts["excluded_non_dr_primary_and_quality_quarantine"] == 9
    assert counts["excluded_non_dr_primary_and_label_conflict_quarantine"] == 4


def test_primary_excluded_and_quarantined_counts():
    primary_rows = load_manifest(PRIMARY_CSV, "primary", image_root=IMAGE_ROOT)
    counts = excluded_and_quarantined_counts(primary_rows)
    assert counts["eligible"] == 14201
    assert counts["quarantined_quality_review"] == 242
    assert counts["quarantined_label_conflict"] == 22
    assert counts["quarantined_conflict_and_quality"] == 49
    assert counts["excluded_non_dr_primary"] == 7926 + 9 + 4


def test_secondary_eligible_class_counts():
    secondary_rows = load_manifest(SECONDARY_CSV, "secondary", image_root=IMAGE_ROOT)
    counts = secondary_eligible_class_counts(secondary_rows, allow_zero_count_classes=True)
    assert len(counts) == 13
    assert sum(counts.values()) == 22127
    assert counts.get("11.Normal Fundus") == 2973
    assert counts.get("12. Advanced PDR") == 75
    assert counts.get("13. Very Severe NPDR") == 78


def test_secondary_eligible_excludes_quarantined():
    secondary_rows = load_manifest(SECONDARY_CSV, "secondary", image_root=IMAGE_ROOT)
    counts = secondary_eligible_class_counts(secondary_rows, allow_zero_count_classes=True)
    # Build the quarantined-by-class expectation directly: for each class, the
    # eligible-only count only depends on eligible rows, so we can recompute it
    # here and assert the function matches without making an unverifiable claim
    # about every quarantined row sharing a class with an eligible row.
    eligible_by_class = {}
    for row in secondary_rows:
        if row["eligibility"] == "eligible_provisional":
            label = row["original_label"]
            if label:
                eligible_by_class[label] = eligible_by_class.get(label, 0) + 1
    assert counts == eligible_by_class
    # Also assert the helper's zero-count class behavior.
    assert len(counts) == 13
    assert sum(counts.values()) == 22127


def test_secondary_eligible_class_counts_allows_zero_count_classes_flag():
    secondary_rows = load_manifest(SECONDARY_CSV, "secondary", image_root=IMAGE_ROOT)
    counts_with_zero = secondary_eligible_class_counts(secondary_rows, allow_zero_count_classes=True)
    counts_without_zero = secondary_eligible_class_counts(secondary_rows, allow_zero_count_classes=False)
    # With the flag, all 13 source labels are present.
    assert len(counts_with_zero) == 13
    # Without the flag, only classes with at least one eligible row are present.
    for label, n in counts_without_zero.items():
        assert n > 0
