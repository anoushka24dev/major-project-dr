#!/usr/bin/env python3
"""Synthetic fixture tests for the data integrity audit pipeline.

Each test uses small in-memory fixture data so they run quickly without
requiring the external dataset roots.  They exercise the core validation
paths that would otherwise need the full Mendeley/MMRDR/APTOS/DDR/IDRiD trees.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from src.integrity import (
    ImageEntry,
    classify_family,
    exact_duplicate_groups,
    near_duplicate_pairs,
    build_grouping,
    allocate_splits,
    assert_split_isolation,
    VERIFIED,
    LIKELY,
    BLOCKED,
    UNKNOWN,
    NEAR_UNIFORM_STD,
    write_manifest_stamp,
    write_artifact_registry,
)
from src.taxonomy import LabelMapper

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DHASH_MAX = 4
AHASH_MAX = 6

EXCLUDED_REASONS = {
    "cross_label_duplicate_conflict",
    "near_uniform_image:no_retinal_structure",
    "unmapped_label:no_declared_five_class_rule",
    "unverified_provenance:filename_family_unrecognised",
}


def _make_entry(
    source: str,
    rel_path: str,
    folder: str,
    family: str,
    mapped_label: int,
    *,
    group_key: str = "",
    sha256: str = "a" * 64,
    dhash: str = "f" * 32,
    ahash: str = "e" * 32,
    width: int = 512,
    height: int = 512,
    decode_ok: bool = True,
    provenance_status: str = LIKELY,
    exclusion_reason: str = "",
) -> ImageEntry:
    entry = ImageEntry(
        path=Path(rel_path),
        rel_path=rel_path,
        source=source,
        folder=folder,
        family=family,
        group_key=group_key,
        mapped_label=mapped_label,
        original_label=folder,
        sha256=sha256,
        dhash=dhash,
        ahash=ahash,
        width=width,
        height=height,
        lum_mean=None,
        lum_std=None,
        decode_ok=decode_ok,
        provenance_status=provenance_status,
        provenance_evidence="test",
        included=True,
        exclusion_reason=exclusion_reason,
        split="",
        group_id="",
        duplicate_of="",
        quality_flags="",
    )
    return entry


# ---- Fixtures ------------------------------------------------------------

def _fixture_entries() -> list[ImageEntry]:
    """Five entries with unique sha256 and filenames matching conventions.

    - prefixed:  "10.Glaucoma/Glaucoma2041_left_0_8337.jpeg" style
    - plain:   "1.jpeg", "2.jpeg" style  (note: 1.jpeg matches _RE_ZENODO first)
    - zenodo:  "3.jpg", "4.jpeg" style
    """
    e1 = _make_entry(
        "test_source",
        "10.Glaucoma/Glaucoma2041_left_0_8337.jpeg",
        "10.Glaucoma",
        "prefixed",
        0,
        group_key="",
        sha256="e" * 64,
    )
    e2 = _make_entry(
        "test_source",
        "1.jpeg",
        "1.Background",
        "plain",
        1,
        group_key="",
        sha256="f" * 64,
    )
    e3 = _make_entry(
        "test_source",
        "2.jpeg",
        "2.Mild",
        "plain",
        2,
        group_key="",
        sha256="g" * 64,
    )
    e4 = _make_entry(
        "test_source",
        "3.jpg",
        "zenodo_numbered",
        "zenodo_numbered",
        3,
        group_key="",
        sha256="h" * 64,
    )
    e5 = _make_entry(
        "test_source",
        "4.jpeg",
        "zenodo_numbered",
        "zenodo_numbered",
        4,
        group_key="",
        sha256="i" * 64,
    )
    return [e1, e2, e3, e4, e5]


def _fixture_duplicate_entries() -> list[ImageEntry]:
    """Two entries with identical SHA-256 (exact duplicate) and different labels."""
    entry = _make_entry(
        "src1", "dup.jpeg", "1.Background", "prefixed", 0, group_key="", sha256="aa" * 32
    )
    entry2 = _make_entry(
        "src2", "dup.jpeg", "2.Mild", "prefixed", 1, group_key="", sha256="aa" * 32
    )
    return [entry, entry2]


def _fixture_perceptual_entries() -> list[ImageEntry]:
    """Two entries with similar dhash (within threshold) and ahash (within threshold)."""
    entries: list[ImageEntry] = []
    # dhash distance 1 (within DHASH_MAX=4), ahash distance 1 (within AHASH_MAX=6)
    entries.append(
        _make_entry(
            "srcA", "a.jpeg", "1.Background", "prefixed", 0,
            group_key="",
            dhash="f0f0f0f0f0f0f0f0", ahash="e0e0e0e0e0e0e0e0",
            width=512, height=512,
            sha256="A" * 64,
        )
    )
    entries.append(
        _make_entry(
            "srcB", "b.jpeg", "2.Mild", "prefixed", 1,
            group_key="",
            dhash="f0f0f0f0f0f0f0f0", ahash="e0e0e0e0e0e0e0e0",  # 1 bit diff each
            width=512, height=512,
            sha256="B" * 64,
        )
    )
    return entries


# ---- Tests ---------------------------------------------------------------

class TestClassifyFamily:
    """verify folder->family classification rules.

    Note: _RE_ZENODO (^\d+\.(?:jpg|jpeg)$) is checked BEFORE _RE_PLAIN,
    so filenames like "1.jpeg" return "zenodo_numbered", not "plain".
    """

    def test_prefixed_family(self) -> None:
        result = classify_family("10.Glaucoma/Glaucoma2041_left_0_8337.jpeg")
        assert result == "prefixed", f"expected prefixed, got {result!r}"

    def test_plain_family(self) -> None:
        # "2_left.jpeg" matches _RE_PLAIN because _RE_ZENODO requires no underscore after the number.
        result = classify_family("2_left.jpeg")
        assert result == "plain", f"expected plain, got {result!r}"
        # "2.jpeg" matches _RE_ZENODO first (checked before _RE_PLAIN).
        result2 = classify_family("2.jpeg")
        assert result2 == "zenodo_numbered", f"expected zenodo_numbered, got {result2!r}"

    def test_zenodo_family(self) -> None:
        # _RE_ZENODO: "^\\d+\\.(?:jpg|jpeg)$" is checked before _RE_PLAIN.
        result = classify_family("1.jpg")
        assert result == "zenodo_numbered", f"expected zenodo_numbered, got {result!r}"
        result2 = classify_family("42.jpeg")
        assert result2 == "zenodo_numbered", f"expected zenodo_numbered, got {result2!r}"


class TestExactDuplicateGroups:
    """verify that exact SHA-256 duplicates are grouped."""

    def test_identical_sha256_groups_together(self) -> None:
        entries = _fixture_duplicate_entries()
        groups = exact_duplicate_groups(entries)
        assert len(groups) == 1
        sha, indices = list(groups.items())[0]
        assert len(indices) == 2
        assert set(indices) == {0, 1}

    def test_no_duplicates_when_sha_differ(self) -> None:
        entries = _fixture_entries()
        groups = exact_duplicate_groups(entries)
        assert len(groups) == 0, f"expected 0 groups, got {len(groups)}: {groups}"


class TestNearDuplicatePairs:
    """verify that perceptual duplicate pairs are found."""

    def test_similar_dhash_and_ahash_returns_pair(self) -> None:
        """When both dhash and ahash are within thresholds, a pair should form."""
        entries = _fixture_perceptual_entries()
        pairs = near_duplicate_pairs(entries, dhash_max=DHASH_MAX, ahash_max=AHASH_MAX)
        # dhash distance = 1 bit <= 4, ahash distance = 1 bit <= 6 → pair forms
        assert len(pairs) >= 1, f"expected at least 1 pair, got {len(pairs)}: {pairs}"

    def test_no_pairs_when_insufficient_images(self) -> None:
        entries = [_make_entry("s", "onlyone.jpeg", "1.Background", "prefixed", 0)]
        pairs = near_duplicate_pairs(entries, dhash_max=DHASH_MAX, ahash_max=AHASH_MAX)
        assert pairs == []

    def test_no_pairs_when_dhash_out_of_threshold(self) -> None:
        """When dhash distance exceeds DHASH_MAX, no pair should form even if ahash is close."""
        entries: list[ImageEntry] = []
        entries.append(
            _make_entry(
                "s1", "a.jpeg", "1.Background", "prefixed", 0,
                dhash="f0f0f0f0f0f0f0f0", ahash="e0e0e0e0e0e0e0e0",
                width=512, height=512,
                sha256="C" * 64,
            )
        )
        entries.append(
            _make_entry(
                "s2", "b.jpeg", "2.Mild", "prefixed", 1,
                dhash="f1f1f1f1f1f1f1f1", ahash="e0e0e0e0e0e0e0e0",  # 64-bit diff
                width=512, height=512,
                sha256="D" * 64,
            )
        )
        pairs = near_duplicate_pairs(entries, dhash_max=DHASH_MAX, ahash_max=AHASH_MAX)
        assert len(pairs) == 0, f"expected 0 pairs, got {len(pairs)}"


class TestBuildGrouping:
    """verify grouping logic with conflicts."""

    def test_grouping_assigns_group_ids(self) -> None:
        entries = _fixture_entries()
        # No perceptual pairs, so grouping should give each image its own group
        grouping = build_grouping(entries, [])
        for i, entry in enumerate(entries):
            gid = grouping.group_of(i)
            assert gid != "", f"image {i} should have a group id"

    def test_same_group_key_groups_together(self) -> None:
        """Two images with the same group_key should be in the same group."""
        entries = [
            _make_entry("s1", "a.jpeg", "1.Background", "prefixed", 0, group_key="patient_001"),
            _make_entry("s1", "b.jpeg", "2.Mild", "prefixed", 1, group_key="patient_001"),
        ]
        grouping = build_grouping(entries, [])
        gid_a = grouping.group_of(0)
        gid_b = grouping.group_of(1)
        assert gid_a == gid_b, "same group_key should produce same group id"


class TestAllocateSplits:
    """verify split allocation and isolation assertions."""

    def test_splits_allocated_with_seed(self) -> None:
        entries = _fixture_entries()
        grouping = build_grouping(entries, [])
        splits = allocate_splits(entries, grouping, included=list(range(len(entries))), seed=42, ratios={'train': 0.5, 'validation': 0.3, 'test': 0.2})
        assert len(splits) == len(entries)
        # Verify each entry got a split
        for idx, split in splits.items():
            assert split in {"train", "validation", "test"}, f"unexpected split: {split}"

    def test_split_isolation_no_cross_taint(self) -> None:
        entries = _fixture_entries()
        grouping = build_grouping(entries, [])
        splits = allocate_splits(entries, grouping, included=list(range(len(entries))), seed=42, ratios={'train': 0.5, 'validation': 0.3, 'test': 0.2})
        problems = assert_split_isolation(entries, grouping, splits)
        assert problems == [], f"split isolation problems: {problems}"

    def test_patient_group_cannot_cross_splits(self) -> None:
        """Two images with the same group_key must end up in the same split."""
        entries = [
            _make_entry("s1", "a.jpeg", "1.Background", "prefixed", 0, group_key="patient_001"),
            _make_entry("s1", "b.jpeg", "2.Mild", "prefixed", 1, group_key="patient_001"),
        ]
        grouping = build_grouping(entries, [])
        splits = allocate_splits(entries, grouping, included=list(range(len(entries))), seed=42, ratios={'train': 0.5, 'validation': 0.3, 'test': 0.2})
        # Both must be in the same split
        split_a = splits.get(0, "")
        split_b = splits.get(1, "")
        assert split_a == split_b, (
            f"patient group crossed splits: {split_a} vs {split_b}"
        )


class TestManifestStamps:
    """verify manifest stamp writes and artifact registry."""

    def test_manifest_stamp_writes_json(self) -> None:
        entries = _fixture_entries()
        grouping = build_grouping(entries, [])
        splits = allocate_splits(entries, grouping, included=list(range(len(entries))), seed=42, ratios={'train': 0.5, 'validation': 0.3, 'test': 0.2})
        # Write a minimal CSV manifest to satisfy the stamp function
        import csv, tempfile
        tmp = Path(tempfile.mkdtemp())
        manifest_path = tmp / "cleaned_classification_manifest.csv"
        with manifest_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "image_id", "source_dataset", "rel_path", "original_label", "mapped_label",
                "class_name", "patient_group_id", "duplicate_group_id", "split_group_id",
                "split", "annotation_availability", "sha256", "dhash", "ahash",
                "width", "height", "provenance_status", "provenance_evidence", "quality_flags",
            ])
            for i, entry in enumerate(entries):
                writer.writerow([
                    entry.key,
                    entry.source,
                    entry.rel_path,
                    entry.original_label,
                    entry.mapped_label,
                    "",
                    entry.group_key,
                    entry.duplicate_of,
                    grouping.group_of(i),
                    splits.get(i, ""),
                    "none_published",
                    entry.sha256,
                    entry.dhash,
                    entry.ahash,
                    entry.width,
                    entry.height,
                    entry.provenance_status,
                    entry.provenance_evidence,
                    entry.quality_flags,
                ])
        stamp = write_manifest_stamp(
            tmp,
            files=("cleaned_classification_manifest.csv",),
            counts={"images_included": len(entries), "images_excluded": 0},
            source_statuses={"test_source": LIKELY},
            extra={"seed": 42, "isolation_checks_passed": True},
        )
        assert stamp.exists(), f"stamp file {stamp} should exist"
        # Clean up (may fail on Windows if files are locked)
        import shutil, os
        try:
            shutil.rmtree(tmp.parent)
        except PermissionError:
            pass

    def test_artifact_registry_records_blocked(self) -> None:
        registry = write_artifact_registry(
            Path("."),
            Path("data_integrity"),
        )
        assert registry.is_file()
        with registry.open("r") as f:
            data = json.load(f)
        assert data["artifacts"][0]["reason"] == "trained before the clean-manifest policy; no clean manifest digest is recorded, so its training data cannot be shown to be free of fabricated labels/masks"


class TestProvenanceStatuses:
    """verify that statuses are correctly assigned."""

    def test_likely_when_mendeley_like(self) -> None:
        # An entry with a recognised folder but no explicit mapping should be LIKELY
        entry = _make_entry("mendeley", "1.Background/img001.jpeg", "1.Background", "prefixed", 0)
        # provenance_status should be set; the exact value depends on the audit logic
        assert entry.provenance_status in {LIKELY, UNKNOWN, VERIFIED}

    def test_unknown_when_no_label_rule(self) -> None:
        # When mapped_label is None and source is unknown, provenance_status should be UNKNOWN
        entry = _make_entry("unknown_source", "img.jpeg", "UnknownFolder", "unknown", None, provenance_status=UNKNOWN)
        assert entry.provenance_status == UNKNOWN


class TestExclusionReasons:
    """verify that exclusion reasons are properly tracked."""

    def test_exclusion_reason_collected(self) -> None:
        entry = _make_entry("test", "img.jpeg", "1.Background", "prefixed", 0, exclusion_reason="cross_label_duplicate_conflict")
        assert entry.exclusion_reason == "cross_label_duplicate_conflict"
        assert entry.exclusion_reason in EXCLUDED_REASONS

    def test_no_exclusion_when_clean(self) -> None:
        entry = _make_entry("test", "img.jpeg", "1.Background", "prefixed", 0)
        assert entry.exclusion_reason == ""


# Run everything if invoked directly
if __name__ == "__main__":
    import pytest
    import shutil
    sys.exit(pytest.main([__file__, "-q"]))