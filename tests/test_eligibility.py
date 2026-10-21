"""Tests for eligibility and quarantine enforcement."""
from __future__ import annotations

from pathlib import Path

from dataset.eligibility import (
    EXCLUDED_NON_DR_VALUES,
    QUARANTINE_VALUES,
    filter_primary_eligible,
    filter_secondary_eligible,
    is_any_quarantined,
    is_primary_eligible,
    is_primary_excluded_non_dr,
    traceable_excluded_rows,
)
from dataset.manifest import load_manifest

AUDIT_DIR = Path(__file__).resolve().parents[1] / "audit_output" / "pretraining_audit"
PRIMARY_CSV = AUDIT_DIR / "PROVISIONAL_ELIGIBILITY_PRIMARY.csv"
SECONDARY_CSV = AUDIT_DIR / "PROVISIONAL_ELIGIBILITY_SECONDARY.csv"
IMAGE_ROOT = Path(r"C:\Users\anous\Downloads\Diabetic Retinopathy and Retinal Diseases Fundus I")


def test_primary_eligible_count():
    primary_rows = load_manifest(PRIMARY_CSV, "primary", image_root=IMAGE_ROOT)
    eligible = [r for r in primary_rows if is_primary_eligible(r)]
    assert len(eligible) == 14201


def test_primary_eligible_are_only_eligible_provisional():
    primary_rows = load_manifest(PRIMARY_CSV, "primary", image_root=IMAGE_ROOT)
    for r in primary_rows:
        if is_primary_eligible(r):
            assert r["eligibility"] == "eligible_provisional"


def test_primary_excludes_non_dr():
    primary_rows = load_manifest(PRIMARY_CSV, "primary", image_root=IMAGE_ROOT)
    non_dr = [r for r in primary_rows if is_primary_excluded_non_dr(r)]
    assert len(non_dr) == 7926 + 9 + 4


def test_non_dr_excluded_values():
    primary_rows = load_manifest(PRIMARY_CSV, "primary", image_root=IMAGE_ROOT)
    values = {r["eligibility"] for r in primary_rows if is_primary_excluded_non_dr(r)}
    assert values.issubset(EXCLUDED_NON_DR_VALUES)


def test_primary_quality_quarantine_count():
    primary_rows = load_manifest(PRIMARY_CSV, "primary", image_root=IMAGE_ROOT)
    q = [r for r in primary_rows if r["eligibility"] == QUARANTINE_VALUES["quality"]]
    assert len(q) == 242


def test_primary_conflict_quarantine_count():
    primary_rows = load_manifest(PRIMARY_CSV, "primary", image_root=IMAGE_ROOT)
    c = [r for r in primary_rows if r["eligibility"] == QUARANTINE_VALUES["conflict"]]
    assert len(c) == 22


def test_primary_both_quarantine_count():
    primary_rows = load_manifest(PRIMARY_CSV, "primary", image_root=IMAGE_ROOT)
    b = [r for r in primary_rows if r["eligibility"] == QUARANTINE_VALUES["both"]]
    assert len(b) == 49


def test_primary_quarantine_rows_are_excluded_from_eligible():
    rows = [
        {"eligibility": "quarantined_quality_review", "image_path": "a.jpg"},
        {"eligibility": "quarantined_label_conflict", "image_path": "b.jpg"},
        {"eligibility": "quarantined_conflict_and_quality", "image_path": "c.jpg"},
        {"eligibility": "eligible_provisional", "image_path": "d.jpg"},
    ]
    eligible = filter_primary_eligible(rows)
    paths = {r["image_path"] for r in eligible}
    assert paths == {"d.jpg"}


def test_primary_excluded_rows_remain_traceable():
    primary_rows = load_manifest(PRIMARY_CSV, "primary", image_root=IMAGE_ROOT)
    excluded = traceable_excluded_rows(primary_rows)
    assert len(excluded) == 22453 - 14201


def test_is_any_quarantined_flag_logic():
    assert is_any_quarantined({"quarantine_quality_flag": "yes", "quarantine_label_conflict": "no"})
    assert is_any_quarantined({"quarantine_quality_flag": "no", "quarantine_label_conflict": "yes"})
    assert is_any_quarantined({"quarantine_quality_flag": "yes", "quarantine_label_conflict": "yes"})
    assert not is_any_quarantined({"quarantine_quality_flag": "no", "quarantine_label_conflict": "no"})
    assert not is_any_quarantined({})


def test_secondary_eligible_count():
    secondary_rows = load_manifest(SECONDARY_CSV, "secondary", image_root=IMAGE_ROOT)
    eligible = filter_secondary_eligible(secondary_rows)
    assert len(eligible) == 22127


def test_secondary_excludes_quarantined():
    secondary_rows = load_manifest(SECONDARY_CSV, "secondary", image_root=IMAGE_ROOT)
    q = [r for r in secondary_rows if r["eligibility"] == "quarantined_quality_review"]
    assert len(q) == 251
    c = [r for r in secondary_rows if r["eligibility"] == "quarantined_label_conflict"]
    assert len(c) == 26
    b = [r for r in secondary_rows if r["eligibility"] == "quarantined_conflict_and_quality"]
    assert len(b) == 49


def test_secondary_eligible_are_only_eligible_provisional():
    secondary_rows = load_manifest(SECONDARY_CSV, "secondary", image_root=IMAGE_ROOT)
    for r in secondary_rows:
        if r["eligibility"] == "eligible_provisional":
            continue
        assert not is_primary_eligible(r)
