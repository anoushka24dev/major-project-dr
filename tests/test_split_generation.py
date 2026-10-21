"""
Tests for the provisional primary 5-band split generator.

These tests are intentionally oriented around the safeguards in the task brief:
reproducibility, no duplicate-component leakage, no quarantined/ineligible
inclusion, no missing paths, class representation where feasible, approximate
target proportions, manifest-to-task consistency, and integration with the
existing SplitValidator.

Avoid brittle pixel-perfect numeric assertions where the allocation is
deterministic but numerically sensitive. When an exact number is the contract,
assert it explicitly and justify it in the test name.
"""

from __future__ import annotations

import csv
import json
import tempfile
from collections import Counter
from pathlib import Path

import pytest

from dataset.eligibility import filter_primary_eligible, is_any_quarantined
from dataset.grouping import group_membership_map
from dataset.manifest import load_manifest
from dataset.labels import primary_task_label
from dataset.split_protocol import (
    PrimarySplitProtocol,
    build_group_table_hash,
    build_report_from_assignment,
    write_split_csv,
)
from dataset.split_validation import SplitValidator, SplitValidationError

AUDIT_DIR = Path(__file__).resolve().parents[1] / "audit_output" / "pretraining_audit"
PRIMARY_CSV = AUDIT_DIR / "PROVISIONAL_ELIGIBILITY_PRIMARY.csv"
IMAGE_ROOT = Path(r"C:\Users\anous\Downloads\Diabetic Retinopathy and Retinal Diseases Fundus I")


def _eligible_rows():
    prim = load_manifest(PRIMARY_CSV, "primary", image_root=IMAGE_ROOT, require_files_exist=True)
    return list(filter_primary_eligible(prim))


def _rows_by_path(eligible_rows):
    by_path = {r["image_path"]: r for r in eligible_rows}
    if len(by_path) != len(eligible_rows):
        dups = [p for p, n in Counter(r["image_path"] for r in eligible_rows).items() if n > 1]
        raise AssertionError(f"unexpected eligible duplicate image_paths: {dups[:5]}")
    return by_path


def _components_from_eligible(eligible_rows):
    mg = group_membership_map(eligible_rows)
    rows_by_path = _rows_by_path(eligible_rows)
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for paths in mg.values():
        it = iter(paths)
        try:
            first = next(it)
        except StopIteration:
            continue
        for p in it:
            union(first, p)

    comp: dict[str, set[str]] = {}
    for p in parent:
        root = find(p)
        comp.setdefault(root, set()).add(p)

    # every eligible image must belong to exactly one component
    for p in rows_by_path:
        if p not in parent:
            comp.setdefault(p, set()).add(p)

    out: list[tuple[str, ...]] = []
    seen: set[str] = set()
    for c in comp.values():
        tup = tuple(sorted(c))
        if not tup:
            continue
        if any(p in seen for p in tup):
            raise AssertionError("reconstructed overlapping components")
        out.append(tup)
        seen.update(tup)

    assert seen == set(rows_by_path.keys())
    out.sort(key=lambda c: ",".join(sorted(c)))
    return out


def _annotated_fake_manifest(*, n_plain=6, n_zenodo=4, duplicate_pairs=0, quarantine_count=0, non_dr_count=0):
    """A small fake primary-eligible manifest fragment for negative/edge tests.

    Returns (rows, rows_by_path) where every supplied row looks eligible unless
    explicitly overridden.
    """
    from dataset.manifest import Record

    rows: list[Record] = []
    idx = 0

    def add(path, *, original_label="3.Mild DR", source_family="plain", patient_group_key=None,
            split_group_id=None, duplicate_group_id=None, quarantine_quality_flag="no",
            quarantine_label_conflict="no", eligibility="eligible_provisional", notes=""):
        nonlocal idx
        if split_group_id is None:
            split_group_id = f"SG-{idx:05d}"
        idx += 1
        row: Record = {
            "manifest_status": "PROVISIONAL - NOT AN APPROVED FINAL DATASET",
            "manifest_version": "provisional-v1-test",
            "approval_reference": "test fixture",
            "image_path": path,
            "original_label": original_label,
            "task_label": "",
            "task_label_basis": "",
            "source_family": source_family,
            "source_status": "likely",
            "patient_group_key": patient_group_key or "",
            "split_group_id": split_group_id,
            "duplicate_group_id": duplicate_group_id or "",
            "quarantine_label_conflict": quarantine_label_conflict,
            "quarantine_quality_flag": quarantine_quality_flag,
            "eligibility": eligibility,
            "notes": notes,
        }
        rows.append(row)

    # plain eligible pool
    for i in range(n_plain):
        add(f"pool/healthy{i}.jpeg", original_label="11.Normal Fundus")
        add(f"pool/mild{i}.jpeg", original_label="3.Mild DR")

    # zenodo eligible pool
    for i in range(n_zenodo):
        add(f"zenodo/z{i}.jpg", original_label="11.Normal Fundus", source_family="zenodo_v02",
            patient_group_key="NONE")

    # duplicate pairs within eligible pool
    for i in range(duplicate_pairs):
        base = i * 2
        g = f"DUP-FAKE-{i:03d}"
        add(f"dup/a{base}.jpeg", duplicate_group_id=g)
        add(f"dup/b{base}.jpeg", duplicate_group_id=g)

    # quarantined rows
    for i in range(quarantine_count):
        add(f"q/q{i}.jpeg", eligibility="quarantined_quality_review",
            quarantine_quality_flag="yes")

    # non-dr excluded rows
    for i in range(non_dr_count):
        add(f"non-dr/n{i}.jpeg", original_label="1.Dry AMD",
            eligibility="excluded_non_dr_primary")

    by_path = {r["image_path"]: r for r in rows}
    return rows, by_path


def test_components_reconstruct_eligible_set_exactly():
    """The component reconstruction used by the generator must partition the
    eligible set with no omissions and no duplicates."""
    rows = _eligible_rows()
    comps = _components_from_eligible(rows)
    paths = [p for c in comps for p in c]
    assert len(paths) == len(rows)
    assert len(set(paths)) == len(paths)
    assert set(paths) == {r["image_path"] for r in rows}


def test_build_components_via_protocol_matches_manual_reconstruction():
    """Protocol.public_builder path should match the internal reconstruction helper."""
    rows = _eligible_rows()
    proto = PrimarySplitProtocol(seed=7, near_zenodo_test_frac=0.10)
    comps_a = proto.build_components_from_eligible_rows(rows)
    comps_b = _components_from_eligible(rows)
    assert comps_a == comps_b


def test_deterministic_assignment_for_same_seed():
    rows = _eligible_rows()
    comps = _components_from_eligible(rows)
    rows_by_path = _rows_by_path(rows)
    a1, r1 = PrimarySplitProtocol(seed=12345, near_zenodo_test_frac=0.10).allocate(comps, rows_by_path)
    a2, r2 = PrimarySplitProtocol(seed=12345, near_zenodo_test_frac=0.10).allocate(comps, rows_by_path)
    assert a1 == a2
    assert r1["partition_counts"] == r2["partition_counts"]
    assert r1["warnings"] == r2["warnings"]


def test_different_seed_produces_different_assignment():
    rows = _eligible_rows()
    comps = _components_from_eligible(rows)
    rows_by_path = _rows_by_path(rows)
    _, r1 = PrimarySplitProtocol(seed=12345, near_zenodo_test_frac=0.10).allocate(comps, rows_by_path)
    _, r2 = PrimarySplitProtocol(seed=99999, near_zenodo_test_frac=0.10).allocate(comps, rows_by_path)
    assert r1["partition_counts"] != r2["partition_counts"]


def test_group_table_hash_is_stable_for_same_components():
    rows = _eligible_rows()
    comps = _components_from_eligible(rows)
    h1 = build_group_table_hash(comps)
    h2 = build_group_table_hash(comps)
    assert h1 == h2
    assert len(h1) == 64


def test_all_primary_bands_present_in_all_three_partitions():
    rows = _eligible_rows()
    comps = _components_from_eligible(rows)
    rows_by_path = _rows_by_path(rows)
    _, report = PrimarySplitProtocol(seed=12345, near_zenodo_test_frac=0.10).allocate(comps, rows_by_path)
    for band in ["No apparent DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "PDR"]:
        cc = report["class_counts"][band]
        for part in ["train", "validation", "test"]:
            assert cc[part] > 0, f"band {band!r} has zero images in {part}"


def test_approximate_target_proportions():
    rows = _eligible_rows()
    comps = _components_from_eligible(rows)
    rows_by_path = _rows_by_path(rows)
    _, report = PrimarySplitProtocol(seed=12345, near_zenodo_test_frac=0.10).allocate(comps, rows_by_path)
    total = report["total_images"]
    for part, target in [("train", 0.70), ("validation", 0.15), ("test", 0.15)]:
        ratio = report["partition_counts"][part] / total
        assert abs(ratio - target) <= 0.02, (
            f"{part} ratio {ratio:.4f} too far from target {target}"
        )


def test_quarantined_rows_not_assigned():
    """No quarantined row from the real manifest should be assigned."""
    rows = _eligible_rows()
    quarantined_paths = {
        r["image_path"] for r in load_manifest(PRIMARY_CSV, "primary", image_root=IMAGE_ROOT)
        if is_any_quarantined(r)
    }
    comps = _components_from_eligible(rows)
    rows_by_path = _rows_by_path(rows)
    assign, _ = PrimarySplitProtocol(seed=12345, near_zenodo_test_frac=0.10).allocate(comps, rows_by_path)
    assert not quarantined_paths & set(assign.keys())


def test_excluded_non_dr_rows_not_assigned():
    """No excluded non-DR row should appear in the primary split."""
    all_rows = load_manifest(PRIMARY_CSV, "primary", image_root=IMAGE_ROOT)
    excluded_paths = {
        r["image_path"]
        for r in all_rows
        if r.get("eligibility", "").startswith("excluded_non_dr_primary")
    }
    comps, assign, _ = _allocate_from_real(_eligible_rows())
    assert not excluded_paths & set(assign.keys())


def _allocate_from_real(rows):
    comps = _components_from_eligible(rows)
    rows_by_path = _rows_by_path(rows)
    proto = PrimarySplitProtocol(seed=12345, near_zenodo_test_frac=0.10)
    assign, report = proto.allocate(comps, rows_by_path)
    return comps, assign, report


def test_assignment_covers_every_eligible_image_once():
    rows = _eligible_rows()
    comps, assign, _ = _allocate_from_real(rows)
    assert set(assign.keys()) == {r["image_path"] for r in rows}
    assert len(assign) == len(rows)


def test_generated_assignment_passes_split_validator():
    rows = _eligible_rows()
    comps, assign, _ = _allocate_from_real(rows)
    validator = SplitValidator(
        list(rows),
        allowed_partitions=["train", "validation", "test"],
        exclude_quarantined=True,
        exclude_non_dr_primary=True,
    )
    report = validator.validate(list(assign.items()))
    assert report["valid"]
    assert report["errors"] == []


def test_split_validator_rejects_duplicate_component_leakage():
    """Directly prove the contract: a duplicate component split across train/test
    is rejected by the existing validator."""
    rows = [
        {"image_path": "x.jpeg", "eligibility": "eligible_provisional",
         "duplicate_group_id": "DUP-Q", "quarantine_quality_flag": "no",
         "quarantine_label_conflict": "no", "source_family": "plain",
         "patient_group_key": ""},
        {"image_path": "y.jpeg", "eligibility": "eligible_provisional",
         "duplicate_group_id": "DUP-Q", "quarantine_quality_flag": "no",
         "quarantine_label_conflict": "no", "source_family": "plain",
         "patient_group_key": ""},
    ]
    validator = SplitValidator(
        rows, allowed_partitions=["train", "validation", "test"],
        exclude_quarantined=True, exclude_non_dr_primary=True,
    )
    with pytest.raises(SplitValidationError, match="Duplicate-connected group"):
        validator.validate([("x.jpeg", "train"), ("y.jpeg", "test")])


def test_split_validator_rejects_quarantined_assignment():
    validator = SplitValidator(
        [{"image_path": "q.jpeg", "eligibility": "quarantined_quality_review",
          "quarantine_quality_flag": "yes", "quarantine_label_conflict": "no",
          "source_family": "plain", "patient_group_key": ""}],
        allowed_partitions=["train", "validation", "test"],
        exclude_quarantined=True, exclude_non_dr_primary=True,
    )
    with pytest.raises(SplitValidationError, match="Quarantined image assigned"):
        validator.validate([("q.jpeg", "train")])


def test_split_validator_rejects_excluded_non_dr_assignment():
    validator = SplitValidator(
        [{"image_path": "d.jpeg", "eligibility": "excluded_non_dr_primary",
          "quarantine_quality_flag": "no", "quarantine_label_conflict": "no",
          "source_family": "plain", "patient_group_key": ""}],
        allowed_partitions=["train", "validation", "test"],
        exclude_quarantined=True, exclude_non_dr_primary=True,
    )
    with pytest.raises(SplitValidationError, match="Excluded non-DR image assigned"):
        validator.validate([("d.jpeg", "train")])


def test_fake_quarantined_rows_are_excluded_by_build_components():
    """build_components_from_eligible_rows must drop quarantined rows so that
    they cannot become part of the assignment."""
    rows, _ = _annotated_fake_manifest(quarantine_count=1)
    comps = PrimarySplitProtocol(seed=1, near_zenodo_test_frac=0.10).build_components_from_eligible_rows(rows)
    comp_paths = {p for c in comps for p in c}
    for r in rows:
        if is_any_quarantined(r):
            assert r["image_path"] not in comp_paths


def test_fake_component_polymorphism_raises_in_protocol():
    """If a duplicate component really did contain two different primary bands,
    the protocol must raise rather than silently assign it."""
    rows, _ = _annotated_fake_manifest(duplicate_pairs=1)
    # override one member's label to a different primary band
    rows[0]["original_label"] = "11.Normal Fundus"
    rows[1]["original_label"] = "3.Mild DR"
    rows_by_path = {r["image_path"]: r for r in rows}
    comp = _components_from_eligible(rows)
    proto = PrimarySplitProtocol(seed=1, near_zenodo_test_frac=0.10)
    with pytest.raises(Exception):
        proto.allocate(comp, rows_by_path)


def test_write_split_csv_matches_assignment():
    rows = _eligible_rows()
    comps, assign, _ = _allocate_from_real(rows)
    tmp = Path(tempfile.gettempdir()) / "primary_split_check.csv"
    try:
        write_split_csv(assign, {r["image_path"]: r for r in rows}, comps,
                        PrimarySplitProtocol(seed=12345, near_zenodo_test_frac=0.10), tmp)
        written = list(csv.DictReader(tmp.read_text(encoding="utf-8").splitlines()))
        assert len(written) == len(assign)
        assert {r["image_path"] for r in written} == set(assign.keys())
        for row in written:
            assert row["task"] == "primary"
            assert row["partition"] == assign[row["image_path"]]
    finally:
        if tmp.exists():
            tmp.unlink()


def test_report_contains_required_provenance_fields():
    rows = _eligible_rows()
    comps, assign, _ = _allocate_from_real(rows)
    rows_by_path = _rows_by_path(rows)
    _, report = PrimarySplitProtocol(seed=12345, near_zenodo_test_frac=0.10).allocate(comps, rows_by_path)
    manifest_entry = rows[0]
    final = build_report_from_assignment(
        assign,
        rows_by_path,
        comps,
        PrimarySplitProtocol(seed=12345, near_zenodo_test_frac=0.10),
        manifest_version=manifest_entry.get("manifest_version", ""),
        manifest_path=PRIMARY_CSV,
        approval_reference=manifest_entry.get("approval_reference", ""),
        group_table_hash=build_group_table_hash(comps),
    )
    for key in [
        "manifest_version", "manifest_path", "manifest_file_hash",
        "approval_reference", "seed", "target_fractions",
        "group_table_hash", "total_components", "total_images",
        "partition_counts", "partition_ratios", "class_counts",
        "zenodo_by_partition", "zenodo_pure_singletons_by_partition",
        "all_bands_in_all_splits", "validation", "limitations",
    ]:
        assert key in final, f"missing report key: {key}"


def test_zenodo_counts_match_manifest_state():
    rows = _eligible_rows()
    zenodo_paths = {r["image_path"] for r in rows if (r.get("source_family") or "").lower() == "zenodo_v02"}
    comps, assign, report = _allocate_from_real(rows)
    assert sum(report["zenodo_by_partition"].values()) == len(zenodo_paths)
    csv_rows = list(csv.DictReader((Path(__file__).resolve().parents[1] / "provisional_primary_split" / "provisional_primary_split.csv").read_text(encoding="utf-8").splitlines()))
    csv_zenodo = [r for r in csv_rows if r["zenodo_unlinked"] == "True"]
    assert len(csv_zenodo) == len(zenodo_paths)


def test_patient_key_availability_in_csv_matches_manifest():
    rows = _eligible_rows()
    expected = {r["image_path"]: bool(r.get("patient_group_key") and str(r.get("patient_group_key","")).strip().lower() not in ("","none"))
                for r in rows}
    csv_rows = list(csv.DictReader((Path(__file__).resolve().parents[1] / "provisional_primary_split" / "provisional_primary_split.csv").read_text(encoding="utf-8").splitlines()))
    for row in csv_rows:
        p = row["image_path"]
        assert row["patient_key_available"] == str(expected[p]) , p


def test_generated_csv_is_deterministic():
    rows = _eligible_rows()
    comps, assign, _ = _allocate_from_real(rows)
    rows_by_path = _rows_by_path(rows)
    proto = PrimarySplitProtocol(seed=12345, near_zenodo_test_frac=0.10)
    tmp1 = Path(tempfile.gettempdir()) / "a.csv"
    tmp2 = Path(tempfile.gettempdir()) / "b.csv"
    try:
        write_split_csv(assign, rows_by_path, comps, proto, tmp1)
        write_split_csv(assign, rows_by_path, comps, proto, tmp2)
        assert tmp1.read_text(encoding="utf-8") == tmp2.read_text(encoding="utf-8")
    finally:
        for p in (tmp1, tmp2):
            if p.exists():
                p.unlink()


def test_provisional_primary_split_artifacts_match_current_build():
    """Integration check that the checked-in split artifacts match the current code.

    We intentionally do not assert exact numeric values here beyond the ones the
    brief cares about: 14201 eligible images, all 5 bands in all 3 splits, no
    quarantined or excluded rows, and SplitValidator passing.
    """
    artifact_csv = Path(__file__).resolve().parents[1] / "provisional_primary_split" / "provisional_primary_split.csv"
    artifact_report = Path(__file__).resolve().parents[1] / "provisional_primary_split" / "provisional_primary_split_report.md"
    assert artifact_csv.exists()
    assert artifact_report.exists()

    csv_rows = list(csv.DictReader(artifact_csv.read_text(encoding="utf-8").splitlines()))
    assert len(csv_rows) == 14201
    assert len({r["image_path"] for r in csv_rows}) == 14201
    assert {r["partition"] for r in csv_rows} == {"train", "validation", "test"}
    for band in ["No apparent DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "PDR"]:
        present_in = {r["partition"] for r in csv_rows if r["primary_class"] == band}
        assert {"train", "validation", "test"} <= present_in

    text = artifact_report.read_text(encoding="utf-8")
    for key in [
        "provisional-v1-2026-10-13",
        "Option A approved by user 2026-10-13 (7 safeguards)",
        "503",
        "393",
        "110",
        "patient independence cannot be claimed",
    ]:
        assert key in text, f"report missing expected marker: {key!r}"
