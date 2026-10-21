"""Tests for split-assignment validation."""
from __future__ import annotations

import pytest
from dataset.split_validation import SplitValidationError, SplitValidator


def _row(image_path, eligibility="eligible_provisional", duplicate_group_id="", *, quality="no", conflict="no", source_family="plain", patient_group_key=""):
    return {
        "image_path": image_path,
        "eligibility": eligibility,
        "duplicate_group_id": duplicate_group_id,
        "source_family": source_family,
        "patient_group_key": patient_group_key,
        "quarantine_quality_flag": quality,
        "quarantine_label_conflict": conflict,
    }


def _validator(*rows, **kwargs):
    return SplitValidator(list(rows), allowed_partitions=["train", "validation", "test"], **kwargs)


def _row(image_path, eligibility="eligible_provisional", duplicate_group_id="", *, quality="no", conflict="no", source_family="plain", patient_group_key=""):
    return {
        "image_path": image_path,
        "eligibility": eligibility,
        "duplicate_group_id": duplicate_group_id,
        "source_family": source_family,
        "patient_group_key": patient_group_key,
        "quarantine_quality_flag": quality,
        "quarantine_label_conflict": conflict,
    }


def _validator(*rows, **kwargs):
    return SplitValidator(list(rows), allowed_partitions=["train", "validation", "test"], **kwargs)


def _assignment(**mapping):
    # Pass duplicates through the validator as a sequence so duplicate-key
    # detection can inspect the raw input before dict-key collapse.
    return [(k, v) for k, v in mapping.items()]


def test_valid_assignment():
    v = _validator(_row("a.jpg"), _row("b.jpg"))
    report = v.validate(_assignment(**{"a.jpg": "train", "b.jpg": "validation"}))
    assert report["valid"]
    assert report["errors"] == []


def test_unknown_path_rejected():
    v = _validator(_row("a.jpg"))
    with pytest.raises(SplitValidationError, match="Unknown image_path"):
        v.validate(_assignment(**{"does-not-exist.jpg": "train"}))


def test_double_assignment_rejected():
    v = _validator(_row("a.jpg"))
    with pytest.raises(SplitValidationError, match="assigned to more than one partition"):
        v.validate([("a.jpg", "train"), ("a.jpg", "validation")])


def test_invalid_partition_rejected():
    v = _validator(_row("a.jpg"))
    with pytest.raises(SplitValidationError, match="Invalid partition"):
        v.validate(_assignment(**{"a.jpg": "eval"}))


def test_quarantined_assigned_rejected():
    v = _validator(_row("a.jpg", eligibility="quarantined_quality_review", quality="yes"))
    with pytest.raises(SplitValidationError, match="Quarantined image assigned"):
        v.validate(_assignment(**{"a.jpg": "train"}))


def test_conflict_quarantined_assigned_rejected():
    v = _validator(_row("b.jpg", eligibility="quarantined_label_conflict", conflict="yes"))
    with pytest.raises(SplitValidationError, match="Quarantined image assigned"):
        v.validate(_assignment(**{"b.jpg": "test"}))


def test_both_quarantined_assigned_rejected():
    v = _validator(_row("c.jpg", eligibility="quarantined_conflict_and_quality", quality="yes", conflict="yes"))
    with pytest.raises(SplitValidationError, match="Quarantined image assigned"):
        v.validate(_assignment(**{"c.jpg": "validation"}))


def test_excluded_non_dr_assigned_rejected():
    v = _validator(_row("d.jpg", eligibility="excluded_non_dr_primary"))
    with pytest.raises(SplitValidationError, match="Excluded non-DR image assigned"):
        v.validate(_assignment(**{"d.jpg": "train"}))


def test_duplicate_group_must_stay_in_one_partition():
    v = _validator(_row("g1.jpg", duplicate_group_id="DUP-G001"), _row("g2.jpg", duplicate_group_id="DUP-G001"))
    with pytest.raises(SplitValidationError, match="Duplicate-connected group"):
        v.validate(_assignment(**{"g1.jpg": "train", "g2.jpg": "test"}))


def test_duplicate_group_in_same_partition_ok():
    v = _validator(_row("g1.jpg", duplicate_group_id="DUP-G002"), _row("g2.jpg", duplicate_group_id="DUP-G002"))
    report = v.validate(_assignment(**{"g1.jpg": "train", "g2.jpg": "train"}))
    assert report["valid"]


def test_report_partition_counts():
    v = _validator(_row("a.jpg"), _row("b.jpg"), _row("c.jpg"))
    report = v.validate(_assignment(**{"a.jpg": "train", "b.jpg": "train", "c.jpg": "validation"}))
    assert report["partition_counts"]["train"] == 2
    assert report["partition_counts"]["validation"] == 1
    assert report["partition_counts"]["test"] == 0


def test_group_integrity_only_applies_to_assigned_members():
    v = _validator(_row("g1.jpg", duplicate_group_id="DUP-G010"), _row("g2.jpg", duplicate_group_id="DUP-G010"))
    report = v.validate(_assignment(**{"g1.jpg": "train"}))
    assert report["valid"]
