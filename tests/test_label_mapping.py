"""Tests for label mapping."""
from __future__ import annotations

import pytest

from dataset.labels import (
    NON_DR_LABELS,
    PRIMARY_MAPPING,
    SOURCE_LABELS,
    is_non_dr_source,
    is_primary_eligible_source,
    primary_band_counts_from_rows,
    primary_task_label,
    secondary_task_label,
    validate_source_labels,
)


def test_source_label_set_has_13_members():
    assert len(SOURCE_LABELS) == 13


def test_primary_mapping_keys_are_known_source_labels():
    assert set(PRIMARY_MAPPING) == {
        "11.Normal Fundus",
        "3.Mild DR",
        "4.Moderate DR",
        "5.Severe DR",
        "13. Very Severe NPDR",
        "6.Proliferate DR",
        "12. Advanced PDR",
    }


def test_primary_mapping_values():
    assert primary_task_label("11.Normal Fundus") == "No apparent DR"
    assert primary_task_label("3.Mild DR") == "Mild NPDR"
    assert primary_task_label("4.Moderate DR") == "Moderate NPDR"
    assert primary_task_label("5.Severe DR") == "Severe NPDR"
    assert primary_task_label("13. Very Severe NPDR") == "Severe NPDR"
    assert primary_task_label("6.Proliferate DR") == "PDR"
    assert primary_task_label("12. Advanced PDR") == "PDR"


def test_secondary_mapping_is_verbatim():
    for label in SOURCE_LABELS:
        assert secondary_task_label(label) == label


def test_unknown_source_label_raises():
    with pytest.raises(ValueError, match="Unknown source label"):
        secondary_task_label("Unknown Class")


def test_unknown_or_non_primary_label_raises():
    with pytest.raises(ValueError, match="Unknown or non-primary source label"):
        primary_task_label("7.Cataract")

    with pytest.raises(ValueError, match="Unknown or non-primary source label"):
        primary_task_label("Unknown Class")


def test_non_dr_labels():
    expected_non_dr = {
        "1.Dry AMD",
        "2.Wet AMD",
        "7.Cataract",
        "8.Hypertensive Retinopathy",
        "9.Pathological Myopia",
        "10.Glaucoma",
    }
    assert NON_DR_LABELS == expected_non_dr
    for label in expected_non_dr:
        assert is_non_dr_source(label)
    assert not is_non_dr_source("11.Normal Fundus")


def test_primary_eligible_source_labels():
    expected = {
        "11.Normal Fundus",
        "3.Mild DR",
        "4.Moderate DR",
        "5.Severe DR",
        "13. Very Severe NPDR",
        "6.Proliferate DR",
        "12. Advanced PDR",
    }
    assert is_primary_eligible_source("11.Normal Fundus")
    assert is_primary_eligible_source("5.Severe DR")
    assert not is_primary_eligible_source("7.Cataract")
    assert set(PRIMARY_MAPPING.keys()) == expected


def test_validate_source_labels_passes():
    validate_source_labels(SOURCE_LABELS)


def test_validate_source_labels_fails():
    with pytest.raises(ValueError, match="Unknown source label"):
        validate_source_labels(["11.Normal Fundus", "Bogus Label"])


def test_primary_band_counts_from_rows():
    rows = [
        {"task_label": "No apparent DR"},
        {"task_label": "Mild NPDR"},
        {"task_label": "No apparent DR"},
        {"task_label": ""},
        {"task_label": None},
    ]
    counts = primary_band_counts_from_rows(rows)
    assert counts.get("No apparent DR") == 2
    assert counts.get("Mild NPDR") == 1
    assert "Bogus" not in counts
