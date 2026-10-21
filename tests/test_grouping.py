"""Tests for grouping metadata."""
from __future__ import annotations

from pathlib import Path

import pytest
from dataset.grouping import (
    group_membership_map,
    has_patient_group_key,
    known_group_ids,
    rows_by_duplicate_group_id,
    rows_by_group_id,
    summarize_grouping,
    zenodo_unlinked_rows,
)
from dataset.manifest import load_manifest

AUDIT_DIR = Path(__file__).resolve().parents[1] / "audit_output" / "pretraining_audit"
PRIMARY_CSV = AUDIT_DIR / "PROVISIONAL_ELIGIBILITY_PRIMARY.csv"
IMAGE_ROOT = Path(r"C:\Users\anous\Downloads\Diabetic Retinopathy and Retinal Diseases Fundus I")


def test_rows_by_group_id():
    primary_rows = load_manifest(PRIMARY_CSV, "primary", image_root=IMAGE_ROOT)
    grouped = rows_by_group_id(primary_rows)
    assert len(grouped) > 0
    total = sum(len(v) for v in grouped.values())
    assert total == len(primary_rows)


def test_rows_by_duplicate_group_id():
    primary_rows = load_manifest(PRIMARY_CSV, "primary", image_root=IMAGE_ROOT)
    grouped = rows_by_duplicate_group_id(primary_rows)
    for gid, rows in grouped.items():
        assert gid
        assert len(rows) >= 1


def test_known_group_ids():
    primary_rows = load_manifest(PRIMARY_CSV, "primary", image_root=IMAGE_ROOT)
    ids = known_group_ids(primary_rows)
    assert isinstance(ids, set)
    for gid in ids:
        assert gid


def test_group_membership_map():
    primary_rows = load_manifest(PRIMARY_CSV, "primary", image_root=IMAGE_ROOT)
    m = group_membership_map(primary_rows)
    for gid, members in m.items():
        assert gid
        assert len(members) >= 1


def test_group_membership_map_is_deterministic():
    primary_rows = load_manifest(PRIMARY_CSV, "primary", image_root=IMAGE_ROOT)
    m1 = group_membership_map(primary_rows)
    m2 = group_membership_map(primary_rows)
    assert m1 == m2


def test_has_patient_group_key():
    assert has_patient_group_key({"patient_group_key": "prefixed:1.Dry AMD|ID=102"})
    assert has_patient_group_key({"patient_group_key": "NONE"}) is False
    assert has_patient_group_key({"patient_group_key": ""}) is False
    assert has_patient_group_key({"patient_group_key": "   "}) is False
    assert has_patient_group_key({}) is False


def test_zenodo_unlinked_rows():
    primary_rows = load_manifest(PRIMARY_CSV, "primary", image_root=IMAGE_ROOT)
    zen = zenodo_unlinked_rows(primary_rows)
    assert len(zen) == 757
    for row in zen:
        assert (row.get("source_family") or "").lower() == "zenodo_v02"
        assert not has_patient_group_key(row)


def test_summarize_grouping():
    primary_rows = load_manifest(PRIMARY_CSV, "primary", image_root=IMAGE_ROOT)
    s = summarize_grouping(primary_rows)
    assert s["split_group_count"] > 0
    assert s["duplicate_group_count"] > 0
    assert s["zenodo_unlinked_rows"] == 757


def test_grouping_summary_matches_manual_counts():
    primary_rows = load_manifest(PRIMARY_CSV, "primary", image_root=IMAGE_ROOT)
    s = summarize_grouping(primary_rows)
    assert s["split_group_count"] == len(rows_by_group_id(primary_rows))
    assert s["duplicate_group_count"] == len(rows_by_duplicate_group_id(primary_rows))
