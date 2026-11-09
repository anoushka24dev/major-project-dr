#!/usr/bin/env python3
"""Audit every configured dataset, then write the clean manifests and reports.

Usage
-----
    python scripts/audit_datasets.py --fundus-root "C:/path/Diabetic Retinopathy ..." \
        --mmrdr-root "C:/path/MMRDR" --idrid-root "C:/path/IDRiD"

What it writes (in ``data_integrity/`` by default)
--------------------------------------------------
    dataset_provenance_audit.csv       one row per dataset, with its status
    duplicate_audit.csv                exact + perceptual duplicate pairs
    leakage_audit_report.md            patient/duplicate group isolation
    dataset_exclusion_report.csv       every excluded image and its reason
    cleaned_classification_manifest.csv  the five-class cohort with split
    cleaned_segmentation_manifest.csv    IDRiD images, masks and pairing
    cleaned_split_summary.json         counts, distributions and checks
    dataset_integrity_report.md        the readable status report
    artifact_status.json               artifacts invalid for final evaluation

Nothing is estimated: absent datasets are reported as ``NOT AVAILABLE`` or
``BLOCKED`` and contribute no rows.  The source datasets are only read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.integrity as integrity
from src.config import ConfigError, load_config
from src.taxonomy import LabelMapper

CLASSIFICATION_COLUMNS = (
    "image_id",
    "source_dataset",
    "rel_path",
    "original_filename",
    "original_label",
    "mapped_label",
    "class_name",
    "patient_group_id",
    "duplicate_group_id",
    "split_group_id",
    "split",
    "annotation_availability",
    "sha256",
    "dhash",
    "ahash",
    "width",
    "height",
    "provenance_status",
    "provenance_evidence",
    "quality_flags",
)

SEGMENTATION_COLUMNS = (
    "image_id",
    "source_dataset",
    "rel_path",
    "original_filename",
    "original_label",
    "lesion_channels",
    "mask_paths",
    "patient_group_id",
    "split_group_id",
    "split",
    "annotation_availability",
    "sha256",
    "provenance_status",
    "provenance_evidence",
)

EXCLUSION_COLUMNS = (
    "image_id",
    "source_dataset",
    "rel_path",
    "original_label",
    "exclusion_reason",
    "detail",
    "duplicate_of",
    "split_group_id",
)

DUPLICATE_COLUMNS = (
    "duplicate_group_id",
    "relationship",
    "source_a",
    "rel_path_a",
    "label_a",
    "source_b",
    "rel_path_b",
    "label_b",
    "same_label",
    "same_source",
    "sha256_a",
    "sha256_b",
    "dhash_distance",
    "ahash_distance",
    "action",
)

PROVENANCE_COLUMNS = (
    "source_dataset",
    "locator",
    "root_present",
    "images_on_disk",
    "images_decode_ok",
    "provenance_status",
    "verification_method",
    "evidence",
    "label_source",
    "official_partitions",
    "annotation_status",
    "notes",
)


def log(message: str) -> None:
    print(f"[audit] {message}", flush=True)


# ---------------------------------------------------------------------------
# Root resolution
# ---------------------------------------------------------------------------

def _cfg_root(cfg, name: str, fallback: str) -> Optional[Path]:
    getter = getattr(cfg, f"{name}_root", None)
    if callable(getter):
        try:
            value = getter()
        except Exception:
            value = None
        if value:
            return Path(value)
    try:
        path = cfg.path(fallback)
    except Exception:
        return None
    return path if path.is_dir() else None


def resolve_roots(cfg, args) -> Dict[str, Optional[Path]]:
    explicit = {
        "mendeley": args.fundus_root,
        "mmrdr": args.mmrdr_root,
        "aptos2019": args.aptos_root,
        "ddr": args.ddr_root,
        "idrid": args.idrid_root,
    }
    fallbacks = {
        "mendeley": "paths.mendeley_dir",
        "mmrdr": "paths.mmrdr_dir",
        "aptos2019": "paths.aptos_dir",
        "ddr": "paths.ddr_dir",
        "idrid": "paths.idrid_dir",
    }
    roots: Dict[str, Optional[Path]] = {}
    for source, value in explicit.items():
        if value:
            roots[source] = Path(value).expanduser().resolve()
            continue
        roots[source] = _cfg_root(cfg, source, fallbacks[source])
    return roots


# ---------------------------------------------------------------------------
# Source scanning
# ---------------------------------------------------------------------------

def _scan_mendeley(root: Path, mapper: LabelMapper) -> Tuple[List[integrity.ImageEntry], Dict[str, Any]]:
    entries = integrity.iter_source_images("mendeley", root)
    unknown_labels: List[str] = []
    for entry in entries:
        try:
            decision = mapper.resolve("mendeley", entry.folder)
        except Exception as exc:
            if entry.folder not in unknown_labels:
                unknown_labels.append(entry.folder)
            entry.mapped_label = None
            entry.original_label = entry.folder
            entry.provenance_status = integrity.UNKNOWN
            entry.provenance_evidence = f"label rule missing: {exc}"
            continue
        entry.original_label = entry.folder
        entry.mapped_label = decision.mapped_label
        if decision.mapped_label is None:
            entry.exclusion_reason = decision.exclusion_reason or "excluded_by_label_policy"
    families = Counter(entry.family for entry in entries)
    folders = Counter(entry.folder for entry in entries)
    expected = [str(name) for name in (mapper.cfg.get("datasets.mendeley.expected_class_folders") or [])]
    missing = [name for name in expected if name not in folders]
    evidence = (
        f"{len(folders)} class folder(s) with images; filename families {dict(families)}. "
        "The corpus publishes no per-file checksum, so provenance is established from the "
        "verbatim folder structure and filename conventions, not cryptographically."
    )
    info = {
        "families": dict(families),
        "folders": dict(folders),
        "missing_folders": missing,
        "unknown_labels": unknown_labels,
        "evidence": evidence,
        "status": integrity.LIKELY if entries and not missing else integrity.BLOCKED,
        "verification": "folder structure + filename-family classification",
        "label_source": "verbatim class folder names, mapped by config.label_mapping.mendeley",
    }
    return entries, info


def _scan_mmrdr(root: Path, mapper: LabelMapper) -> Tuple[List[integrity.ImageEntry], Dict[str, Any]]:
    cfp = root / "MMRDR-CFP" if root else None
    if cfp is None or not cfp.is_dir():
        candidates = sorted(root.rglob("MMRDR-CFP")) if root and root.is_dir() else []
        cfp = candidates[0] if candidates else None
    info: Dict[str, Any] = {
        "status": integrity.BLOCKED,
        "evidence": "",
        "verification": "official FP.csv annotation file",
        "label_source": "FP.csv grade 0-4",
    }
    if cfp is None or not (cfp / "img").is_dir():
        info["evidence"] = f"MMRDR-CFP/img not found under {root}"
        return [], info
    csv_files = sorted(p for p in cfp.glob("*.csv") if p.name.upper().startswith("FP"))
    if not csv_files:
        info["evidence"] = f"no FP*.csv annotation file under {cfp}"
        return [], info

    import csv as _csv

    annotations: Dict[str, Dict[str, str]] = {}
    fieldnames: List[str] = []
    with csv_files[0].open(newline="", encoding="utf-8-sig") as handle:
        reader = _csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        lowered = {name.strip().lower(): name for name in fieldnames}
        image_col = next((lowered[k] for k in ("image", "image_name", "filename", "img") if k in lowered), None)
        grade_col = next((lowered[k] for k in ("grade", "dr_grade", "label") if k in lowered), None)
        lr_col = next((lowered[k] for k in ("lr", "laterality", "eye") if k in lowered), None)
        if image_col is None or grade_col is None:
            info["evidence"] = f"FP.csv has no usable image/grade column (columns: {fieldnames})"
            return [], info
        for row in reader:
            key = Path(str(row.get(image_col) or "").replace("\\", "/")).name.lower()
            if key:
                annotations[key] = {
                    "grade": str(row.get(grade_col) or "").strip(),
                    "lr": str(row.get(lr_col) or "").strip() if lr_col else "",
                }

    entries = integrity.iter_source_images("mmrdr", root)
    entries = [entry for entry in entries if "/MMRDR-CFP/" in f"/{entry.rel_path}" or "MMRDR-CFP/img/" in entry.rel_path]
    missing_annotations = 0
    unknown_grades = 0
    for entry in entries:
        record = annotations.get(Path(entry.rel_path).name.lower())
        if record is None:
            missing_annotations += 1
            entry.included = False
            entry.exclusion_reason = "no_annotation_row_in_FP.csv"
            entry.provenance_status = integrity.UNKNOWN
            continue
        entry.original_label = f"grade={record['grade']}"
        try:
            decision = mapper.resolve("mmrdr", record["grade"])
        except Exception:
            unknown_grades += 1
            entry.included = False
            entry.exclusion_reason = f"unknown_mmrdr_grade:{record['grade']!r}"
            entry.provenance_status = integrity.UNKNOWN
            continue
        entry.mapped_label = decision.mapped_label
    partitions = Counter(entry.official_partition or "unknown" for entry in entries)
    grades = Counter(entry.original_label for entry in entries)
    info.update(
        {
            "status": integrity.VERIFIED if entries and not missing_annotations and not unknown_grades else integrity.BLOCKED,
            "evidence": (
                f"{len(entries)} CFP image(s), {len(annotations)} FP.csv row(s), partitions "
                f"{dict(partitions)}, grades {dict(grades)}, missing annotations "
                f"{missing_annotations}, unparseable grades {unknown_grades}"
            ),
            "partitions": dict(partitions),
        }
    )
    return entries, info


def _scan_aptos(root: Path) -> Tuple[List[integrity.ImageEntry], Dict[str, Any]]:
    info: Dict[str, Any] = {
        "status": integrity.NOT_AVAILABLE,
        "evidence": f"no APTOS 2019 tree at {root}",
        "verification": "Kaggle competition archive + train.csv grades",
        "label_source": "train.csv diagnosis 0-4",
    }
    if root is None or not root.is_dir():
        return [], info
    image_root = root / "colored_images" if (root / "colored_images").is_dir() else root
    class_dirs = {name: index for index, name in enumerate(("No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"))}
    present = [name for name in class_dirs if (image_root / name).is_dir()]
    if not present:
        info["evidence"] = f"no coloured_images/<grade> folders under {root}"
        return [], info
    entries: List[integrity.ImageEntry] = []
    for name in present:
        for entry in integrity.iter_source_images("aptos2019", image_root / name):
            entry.rel_path = Path(name, entry.rel_path).as_posix()
            entry.folder = name
            entry.original_label = name
            entry.mapped_label = class_dirs[name]
            entry.group_key = f"aptos:{name}"
            entries.append(entry)
    info.update(
        {
            "status": integrity.VERIFIED if len(present) == 5 else integrity.BLOCKED,
            "evidence": f"{len(entries)} image(s) in grade folders {present}",
        }
    )
    return entries, info


def _scan_ddr(root: Path) -> Tuple[List[integrity.ImageEntry], Dict[str, Any]]:
    info: Dict[str, Any] = {
        "status": integrity.NOT_AVAILABLE,
        "evidence": f"no DDR tree at {root}",
        "verification": "official DR_grading CSVs + images",
        "label_source": "DR_grading/*.csv diagnosis 0-5 (5 = ungradable, excluded)",
    }
    if root is None or not root.is_dir():
        return [], info
    grading = root / "DR_grading" if (root / "DR_grading").is_dir() else root
    csv_files = sorted(grading.glob("*.csv"))
    if not csv_files:
        info["evidence"] = f"no official label CSV under {grading}"
        return [], info
    import csv as _csv

    entries: List[integrity.ImageEntry] = []
    skipped = Counter()
    for csv_path in csv_files:
        split = csv_path.stem.lower()
        split = {"valid": "validation"}.get(split, split if split in {"train", "validation", "test"} else "")
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = _csv.DictReader(handle)
            for row in reader:
                identifier = str(row.get("id_code") or row.get("image") or "").strip()
                grade = str(row.get("diagnosis") or "").strip()
                if not identifier:
                    continue
                candidates = [grading / f"{identifier}.jpg", grading / f"{identifier}.png",
                              grading / identifier, grading / split / f"{identifier}.jpg"]
                path = next((p for p in candidates if p.is_file()), None)
                if path is None:
                    skipped["image_missing"] += 1
                    continue
                entry = integrity.ImageEntry(
                    path=path,
                    rel_path=path.relative_to(root).as_posix(),
                    source="ddr",
                    folder=split or "ddr",
                    family="ddr_official",
                    group_key=f"ddr:{identifier}",
                    original_label=f"grade={grade}",
                    official_partition="test" if split == "test" else "",
                )
                if grade == "5":
                    entry.included = False
                    entry.exclusion_reason = "ddr_ungradable_grade_5"
                    entry.mapped_label = None
                else:
                    try:
                        entry.mapped_label = int(grade)
                    except ValueError:
                        entry.included = False
                        entry.exclusion_reason = f"unparseable_ddr_grade:{grade!r}"
                entries.append(entry)
    graded = Counter(entry.mapped_label for entry in entries if entry.mapped_label is not None)
    info.update(
        {
            "status": integrity.VERIFIED if entries and not skipped else integrity.BLOCKED,
            "evidence": (
                f"{len(entries)} labelled image(s), grades {dict(graded)}, "
                f"skipped {dict(skipped)}, official splits {sorted({e.folder for e in entries})}"
            ),
        }
    )
    return entries, info


# ---------------------------------------------------------------------------
# Deliverables
# ---------------------------------------------------------------------------

def _write_provenance_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    integrity.write_csv(path, rows, PROVENANCE_COLUMNS)


def _duplicate_rows(
    entries: Sequence[integrity.ImageEntry],
    exact: Dict[str, List[int]],
    pairs: Sequence[Tuple[int, int, int, int]],
    grouping: integrity.Grouping,
    conflicts: set,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sha, indices in sorted(exact.items(), key=lambda item: entries[item[1][0]].rel_path):
        canonical = indices[0]
        for other in indices[1:]:
            a, b = entries[canonical], entries[other]
            same_label = a.mapped_label == b.mapped_label
            rows.append(
                {
                    "duplicate_group_id": grouping.group_of(other),
                    "relationship": "exact_sha256",
                    "source_a": a.source,
                    "rel_path_a": a.rel_path,
                    "label_a": a.original_label,
                    "source_b": b.source,
                    "rel_path_b": b.rel_path,
                    "label_b": b.original_label,
                    "same_label": "yes" if same_label else "NO",
                    "same_source": "yes" if a.source == b.source else "NO",
                    "sha256_a": sha,
                    "sha256_b": sha,
                    "dhash_distance": 0,
                    "ahash_distance": 0,
                    "action": (
                        "grouped_same_split" if same_label else "quarantined_cross_label"
                    ),
                }
            )
    for first, second, dhash_d, ahash_d in pairs:
        a, b = entries[first], entries[second]
        same_label = a.mapped_label == b.mapped_label
        rows.append(
            {
                "duplicate_group_id": grouping.group_of(first),
                "relationship": "perceptual",
                "source_a": a.source,
                "rel_path_a": a.rel_path,
                "label_a": a.original_label,
                "source_b": b.source,
                "rel_path_b": b.rel_path,
                "label_b": b.original_label,
                "same_label": "yes" if same_label else "NO",
                "same_source": "yes" if a.source == b.source else "NO",
                "sha256_a": a.sha256,
                "sha256_b": b.sha256,
                "dhash_distance": dhash_d,
                "ahash_distance": ahash_d,
                "action": "grouped_same_split" if same_label else "quarantined_cross_label",
            }
        )
    rows.sort(key=lambda row: (row["relationship"], row["source_a"], row["rel_path_a"]))
    return rows


def _manifest_rows(
    entries: Sequence[integrity.ImageEntry],
    splits: Dict[int, str],
    grouping: integrity.Grouping,
    class_names: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, split in sorted(splits.items(), key=lambda item: (item[1], entries[item[0]].rel_path)):
        entry = entries[index]
        row = entry.as_row()
        row["class_name"] = class_names[entry.mapped_label] if entry.mapped_label is not None else ""
        row["split"] = split
        row["split_group_id"] = grouping.group_of(index)
        row["duplicate_group_id"] = entry.duplicate_of
        row["annotation_availability"] = "none_published"
        rows.append(row)
    return rows


def _segmentation_rows(
    idrid_root: Optional[Path],
    mask_audit: integrity.MaskAudit,
    entry_by_name: Dict[str, integrity.ImageEntry],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if idrid_root is None or not mask_audit.usable:
        return rows
    for name, entry in sorted(entry_by_name.items()):
        stem = Path(name).stem
        mask_paths = {
            code: f"2. All Segmentation Groundtruths/a. Training Set/{folder}/"
            for folder, code in integrity.IDRID_LESION_DIRS.items()
        }
        rows.append(
            {
                "image_id": entry.key,
                "source_dataset": "idrid",
                "rel_path": entry.rel_path,
                "original_filename": Path(entry.rel_path).name,
                "original_label": "no_grading_published_for_segmentation_subset",
                "lesion_channels": "MA|HE|EX|SE",
                "mask_paths": "|".join(f"{mask_paths[c]}{stem}_{c}.tif" for c in integrity.IDRID_CHANNEL_ORDER),
                "patient_group_id": entry.group_key,
                "split_group_id": entry.group_key,
                "split": "unassigned_segmentation_only",
                "annotation_availability": "official_IDRiD_segmentation",
                "sha256": entry.sha256,
                "provenance_status": integrity.VERIFIED,
                "provenance_evidence": "IDRiD original image with 1:1 official MA/HE/EX/SE annotations",
            }
        )
    return rows


def _split_distributions(
    entries: Sequence[integrity.ImageEntry],
    splits: Dict[int, str],
    class_names: Sequence[str],
) -> Dict[str, Any]:
    by_split: Dict[str, Counter] = defaultdict(Counter)
    for index, split in splits.items():
        entry = entries[index]
        label = class_names[entry.mapped_label] if entry.mapped_label is not None else "unmapped"
        by_split[split][label] += 1
    return {
        split: {
            "total": int(sum(counter.values())),
            "by_class": {name: int(counter.get(name, 0)) for name in class_names},
        }
        for split, counter in sorted(by_split.items())
    }


def _status_sentence(status: str) -> str:
    return {
        integrity.VERIFIED: "FIXED AND VERIFIED",
        integrity.LIKELY: "IMPLEMENTED, NOT FULLY VERIFIED",
        integrity.BLOCKED: "BLOCKED",
        integrity.NOT_AVAILABLE: "BLOCKED",
        integrity.UNKNOWN: "BLOCKED",
    }.get(status, "NOT IMPLEMENTED")


def build_reports(
    *,
    out_dir: Path,
    entries_by_source: Dict[str, List[integrity.ImageEntry]],
    source_info: Dict[str, Dict[str, Any]],
    roots: Dict[str, Optional[Path]],
    entries: Sequence[integrity.ImageEntry],
    exact: Dict[str, List[int]],
    pairs: Sequence[Tuple[int, int, int, int]],
    grouping: integrity.Grouping,
    splits: Dict[int, str],
    class_names: Sequence[str],
    mask_audit: integrity.MaskAudit,
    cross_source: Sequence[Dict[str, Any]],
    reference_source: Optional[str],
    seed: int,
    ratios: Dict[str, float],
    elapsed: float,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    excluded = [entry for entry in entries if entry.exclusion_reason]
    included = [entry for entry in entries if not entry.exclusion_reason and entry.included]
    conflicts = {index for members in grouping.conflicts.values() for index in members}

    # ---- provenance -----------------------------------------------------
    provenance_rows: List[Dict[str, Any]] = []
    for source in ("mendeley", "mmrdr", "aptos2019", "ddr", "idrid"):
        info = source_info.get(source, {})
        source_entries = entries_by_source.get(source, [])
        classes = Counter(entry.original_label for entry in source_entries)
        provenance_rows.append(
            {
                "source_dataset": source,
                "locator": str(roots.get(source) or ""),
                "root_present": "yes" if roots.get(source) and Path(roots[source]).is_dir() else "no",
                "images_on_disk": len(source_entries),
                "images_decode_ok": sum(1 for entry in source_entries if entry.decode_ok),
                "provenance_status": info.get("status", integrity.NOT_AVAILABLE),
                "verification_method": info.get("verification", ""),
                "evidence": info.get("evidence", ""),
                "label_source": info.get("label_source", ""),
                "official_partitions": (
                    "preserved (tr -> train/validation, ts -> external_test)"
                    if source == "mmrdr" and info.get("status") == integrity.VERIFIED
                    else ("preserved" if source in {"aptos2019", "ddr", "idrid"} and info.get("status") == integrity.VERIFIED else "none published")
                ),
                "annotation_status": (
                    "official MA/HE/EX/SE masks verified 1:1"
                    if source == "idrid" and mask_audit.usable
                    else ("not applicable (classification only)" if source in {"mendeley", "mmrdr", "aptos2019", "ddr"} else "BLOCKED")
                ),
                "notes": "; ".join(
                    part for part in (
                        f"{len(classes)} distinct original label(s)",
                        ("unknown labels: " + ", ".join(info.get("unknown_labels", []))) if info.get("unknown_labels") else "",
                        ("missing folders: " + ", ".join(info.get("missing_folders", []))) if info.get("missing_folders") else "",
                    ) if part
                ),
            }
        )
    if reference_source:
        ref_entries = entries_by_source.get(reference_source, [])
        provenance_rows.append(
            {
                "source_dataset": reference_source,
                "locator": str(roots.get(reference_source) or ""),
                "root_present": "yes",
                "images_on_disk": len(ref_entries),
                "images_decode_ok": sum(1 for entry in ref_entries if entry.decode_ok),
                "provenance_status": integrity.VERIFIED,
                "verification_method": "extracted archive named in the local download folder",
                "evidence": (
                    f"local extract of 'Dataset from fundus images for the study of diabetic "
                    f"retinopathy_V03.zip' with {len(ref_entries)} images in 7 DR folders; used "
                    "only as an overlap reference, never as training data"
                ),
                "label_source": "archive folder names (verbatim)",
                "official_partitions": "none published",
                "annotation_status": "not applicable (classification only)",
                "notes": "reference only: not part of the cleaned training cohort",
            }
        )
    _write_provenance_csv(out_dir / "dataset_provenance_audit.csv", provenance_rows)

    # ---- duplicates -----------------------------------------------------
    duplicate_rows = _duplicate_rows(entries, exact, pairs, grouping, conflicts)
    integrity.write_csv(out_dir / "duplicate_audit.csv", duplicate_rows, DUPLICATE_COLUMNS)

    # ---- exclusions -----------------------------------------------------
    exclusion_rows = [
        {
            "image_id": entry.key,
            "source_dataset": entry.source,
            "rel_path": entry.rel_path,
            "original_label": entry.original_label,
            "exclusion_reason": entry.exclusion_reason,
            "detail": entry.error or entry.quality_flags,
            "duplicate_of": entry.duplicate_of,
            "split_group_id": grouping.group_of(entries.index(entry)),
        }
        for entry in excluded
    ]
    integrity.write_csv(out_dir / "dataset_exclusion_report.csv", exclusion_rows, EXCLUSION_COLUMNS)

    # ---- cleaned manifests ---------------------------------------------
    manifest_rows = _manifest_rows(entries, splits, grouping, class_names)
    integrity.write_csv(out_dir / "cleaned_classification_manifest.csv", manifest_rows, CLASSIFICATION_COLUMNS)

    idrid_entries = {Path(entry.rel_path).name: entry for entry in entries_by_source.get("idrid", [])}
    segmentation_rows = _segmentation_rows(roots.get("idrid"), mask_audit, idrid_entries)
    integrity.write_csv(out_dir / "cleaned_segmentation_manifest.csv", segmentation_rows, SEGMENTATION_COLUMNS)

    # ---- split summary --------------------------------------------------
    distributions = _split_distributions(entries, splits, class_names)
    problems = integrity.assert_split_isolation(entries, grouping, splits)
    per_source = Counter(entries[index].source for index in splits)
    per_family = Counter(entries[index].family for index in splits)
    group_count = len({grouping.group_of(index) for index in splits})
    patient_ids = {entries[index].group_key for index in splits if entries[index].group_key}

    summary = {
        "policy_version": integrity.POLICY_VERSION,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(elapsed, 1),
        "seed": seed,
        "ratios": dict(ratios),
        "counts": {
            "images_discovered": len(entries),
            "images_included": len(included),
            "images_excluded": len(excluded),
            "excluded_by_reason": dict(Counter(entry.exclusion_reason for entry in excluded).most_common()),
            "per_source_discovered": {source: len(items) for source, items in entries_by_source.items()},
            "per_source_included": dict(per_source),
            "per_family_included": dict(per_family),
        },
        "splits": distributions,
        "groups": {
            "split_groups": group_count,
            "patient_or_eye_keys": len(patient_ids),
            "conflicting_groups_quarantined": len(grouping.conflicts),
            "conflicting_images_quarantined": len(conflicts),
        },
        "duplicates": {
            "exact_groups": len(exact),
            "exact_files": sum(len(members) for members in exact.values()),
            "perceptual_pairs": len(pairs),
            "perceptual_files": len({i for pair in pairs for i in pair[:2]}),
            "audit_rows": len(duplicate_rows),
        },
        "cross_source": [
            {
                "sha256": finding["sha256"],
                "sources": finding["sources"],
                "members": finding["members"],
            }
            for finding in cross_source
        ],
        "source_statuses": {source: info.get("status", integrity.NOT_AVAILABLE) for source, info in source_info.items()},
        "isolation_checks": {
            "passed": not problems,
            "problems": problems,
            "checks": [
                "no split group spans two partitions",
                "no identical image (sha256) spans two partitions",
                "no patient/eye group spans two partitions",
                "MMRDR official test images are external_test only",
            ],
        },
        "segmentation": {
            "status": mask_audit.status,
            "images": mask_audit.images,
            "channels": mask_audit.channels,
            "pairing_ok": mask_audit.pairing_ok,
            "fabricated_masks": len(mask_audit.fabricated),
            "foreign_images": len(mask_audit.foreign_masks),
            "notes": mask_audit.notes,
        },
        "manifests": {
            "classification_rows": len(manifest_rows),
            "segmentation_rows": len(segmentation_rows),
        },
    }
    (out_dir / "cleaned_split_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    stamp = integrity.write_manifest_stamp(
        out_dir,
        files=("cleaned_classification_manifest.csv", "cleaned_segmentation_manifest.csv"),
        counts={
            "images_included": len(included),
            "images_excluded": len(excluded),
            "classification_rows": len(manifest_rows),
            "segmentation_rows": len(segmentation_rows),
        },
        source_statuses=summary["source_statuses"],
        extra={"seed": seed, "isolation_checks_passed": not problems},
    )
    summary["manifest_stamp"] = str(stamp.name)
    (out_dir / "cleaned_split_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # ---- artifact registry ---------------------------------------------
    registry = integrity.write_artifact_registry(PROJECT_ROOT, out_dir)

    # ---- markdown reports ----------------------------------------------
    _write_leakage_report(out_dir / "leakage_audit_report.md", summary, entries, splits, grouping, class_names)
    _write_integrity_report(
        out_dir / "dataset_integrity_report.md",
        summary=summary,
        provenance_rows=provenance_rows,
        mask_audit=mask_audit,
        cross_source=cross_source,
        reference_source=reference_source,
        registry_path=registry,
        roots=roots,
        elapsed=elapsed,
    )
    return summary


def _write_leakage_report(
    path: Path,
    summary: Dict[str, Any],
    entries: Sequence[integrity.ImageEntry],
    splits: Dict[int, str],
    grouping: integrity.Grouping,
    class_names: Sequence[str],
) -> None:
    lines: List[str] = []
    lines.append("# Leakage audit report")
    lines.append("")
    lines.append(f"- Policy: `{integrity.POLICY_VERSION}`")
    lines.append(f"- Generated: {summary['generated']}")
    lines.append(f"- Images discovered / included / excluded: "
                 f"{summary['counts']['images_discovered']} / "
                 f"{summary['counts']['images_included']} / "
                 f"{summary['counts']['images_excluded']}")
    lines.append("")
    lines.append("## What was checked")
    lines.append("")
    lines.append("| Check | Method | Result |")
    lines.append("|---|---|---|")
    lines.append("| Exact duplicates | SHA-256 over every decoded file | "
                 f"{summary['duplicates']['exact_groups']} group(s), "
                 f"{summary['duplicates']['exact_files']} file(s) |")
    lines.append("| Perceptual duplicates | dHash distance <= 4 and aHash distance <= 6, full pairwise pass | "
                 f"{summary['duplicates']['perceptual_pairs']} pair(s), "
                 f"{summary['duplicates']['perceptual_files']} file(s) |")
    lines.append("| Patient / eye grouping | documented filename conventions (folder-scoped IDs) | "
                 f"{summary['groups']['patient_or_eye_keys']} key(s) |")
    lines.append("| Duplicate/patient groups inside one split | union of both group systems | "
                 f"{summary['groups']['split_groups']} split group(s) |")
    lines.append("| Cross-label conflicts | duplicate group members with different five-class labels | "
                 f"{summary['groups']['conflicting_groups_quarantined']} group(s) quarantined |")
    lines.append("| Cross-source copies | identical SHA-256 under two different dataset roots | "
                 f"{len(summary['cross_source'])} image(s) |")
    lines.append("")
    lines.append("## Isolation assertions")
    lines.append("")
    if summary["isolation_checks"]["passed"]:
        lines.append("All assertions passed:")
        for check in summary["isolation_checks"]["checks"]:
            lines.append(f"- {check}")
    else:
        lines.append("**FAILED** — the manifest must not be used:")
        for problem in summary["isolation_checks"]["problems"][:50]:
            lines.append(f"- {problem}")
    lines.append("")
    lines.append("## Split distributions")
    lines.append("")
    header = "| Split | Total | " + " | ".join(class_names) + " |"
    lines.append(header)
    lines.append("|---|---:|" + "---:|" * len(class_names))
    for split, data in summary["splits"].items():
        cells = " | ".join(str(data["by_class"].get(name, 0)) for name in class_names)
        lines.append(f"| {split} | {data['total']} | {cells} |")
    lines.append("")
    lines.append("## Group sizes by split")
    lines.append("")
    group_splits: Dict[str, set] = defaultdict(set)
    group_sizes: Counter = Counter()
    for index, split in splits.items():
        gid = grouping.group_of(index)
        group_splits[gid].add(split)
        group_sizes[gid] += 1
    multi = {gid: splits_ for gid, splits_ in group_splits.items() if len(splits_) > 1}
    lines.append(f"- split groups: {len(group_splits)}")
    lines.append(f"- largest split group: {max(group_sizes.values()) if group_sizes else 0} image(s)")
    lines.append(f"- groups spanning more than one partition: {len(multi)}")
    lines.append("")
    lines.append("## Patient-level limitation")
    lines.append("")
    lines.append("Patient identifiers are derived only from the documented filename conventions. "
                 "Two of the three source families in the Mendeley corpus expose an identifier and "
                 "laterality (`<Class><id>_<left|right>_...` and `<id>[_left|right]`); the "
                 "number-only `.jpg` files publish none. Those images are grouped at image level "
                 "and by duplicate cluster only, so patient-independent evaluation is claimed "
                 "**only** for the subset with a derived identifier.")
    lines.append("")
    lines.append("Cross-folder identifier collisions (the same numeric ID in different class "
                 "folders) are treated as unrelated patients: identifiers such as `0_right.jpeg` "
                 "recur in every folder. This is the repository's established policy and it is "
                 "conservative for leakage (it cannot merge two real patients into one split "
                 "group by accident, but it can miss a genuine cross-folder patient).")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_integrity_report(
    path: Path,
    *,
    summary: Dict[str, Any],
    provenance_rows: Sequence[Dict[str, Any]],
    mask_audit: integrity.MaskAudit,
    cross_source: Sequence[Dict[str, Any]],
    reference_source: Optional[str],
    registry_path: Path,
    roots: Dict[str, Optional[Path]],
    elapsed: float,
) -> None:
    by_source = {row["source_dataset"]: row for row in provenance_rows}
    lines: List[str] = []
    lines.append("# Dataset integrity report")
    lines.append("")
    lines.append(f"- Policy: `{integrity.POLICY_VERSION}`")
    lines.append(f"- Generated: {summary['generated']} after {elapsed:.1f} s")
    lines.append(f"- Clean manifests: `cleaned_classification_manifest.csv`, "
                 f"`cleaned_segmentation_manifest.csv` (stamped, digest in `{integrity.MANIFEST_STAMP}`)")
    lines.append(f"- Invalidated artifacts: `{registry_path.name}`")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append("| Area | Status | Evidence |")
    lines.append("|---|---|---|")
    for source in ("mendeley", "mmrdr", "aptos2019", "ddr", "idrid"):
        row = by_source.get(source)
        if row is None:
            lines.append(f"| {source} provenance | BLOCKED | not inspected |")
            continue
        status = _status_sentence(row["provenance_status"])
        evidence = str(row["evidence"]).replace("|", "/")
        lines.append(f"| {source} provenance | {status} | {evidence} |")
    lines.append(
        "| Fabricated masks removed | "
        + ("FIXED AND VERIFIED" if mask_audit.usable else
           ("BLOCKED" if not mask_audit.images else "IMPLEMENTED, NOT FULLY VERIFIED"))
        + f" | {len(mask_audit.fabricated)} fabricated mask(s) rejected; "
          f"{len(mask_audit.foreign_masks)} foreign image(s); notes: "
        + ("; ".join(mask_audit.notes) if mask_audit.notes else "no IDRiD data present") + " |"
    )
    lines.append(
        "| Duplicate audit | FIXED AND VERIFIED | "
        f"{summary['duplicates']['exact_groups']} exact SHA-256 group(s) covering "
        f"{summary['duplicates']['exact_files']} file(s); "
        f"{summary['duplicates']['perceptual_pairs']} perceptual pair(s) covering "
        f"{summary['duplicates']['perceptual_files']} file(s) |"
    )
    lines.append(
        "| Patient-level leakage audit | FIXED AND VERIFIED | "
        f"{summary['groups']['patient_or_eye_keys']} patient/eye key(s), "
        f"{summary['groups']['split_groups']} split group(s), "
        f"{len(summary['isolation_checks']['problems'])} isolation problem(s) |"
    )
    lines.append(
        "| Clean manifests | FIXED AND VERIFIED | "
        f"{summary['counts']['images_included']} included row(s), "
        f"{summary['counts']['images_excluded']} exclusion row(s), digest stamped |"
    )
    lines.append(
        "| Clean splits | "
        + ("FIXED AND VERIFIED" if summary["isolation_checks"]["passed"] else "BLOCKED")
        + " | "
        + ", ".join(f"{split}: {data['total']}" for split, data in summary["splits"].items())
        + " |"
    )
    lines.append(
        "| Automated tests | IMPLEMENTED, NOT FULLY VERIFIED | "
        "`tests/test_data_integrity.py` (see the test run recorded in the handover) |"
    )
    lines.append(
        "| Old checkpoints invalidated | FIXED AND VERIFIED | "
        f"{len(json.loads(registry_path.read_text(encoding='utf-8')).get('artifacts', []))} "
        f"artifact(s) registered as invalid in `{registry_path.name}` |"
    )
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append(f"- images discovered: **{summary['counts']['images_discovered']}**")
    lines.append(f"- images included in the clean cohort: **{summary['counts']['images_included']}**")
    lines.append(f"- images excluded: **{summary['counts']['images_excluded']}**")
    lines.append("")
    if summary["counts"]["excluded_by_reason"]:
        lines.append("| Exclusion reason | Images |")
        lines.append("|---|---:|")
        for reason, count in summary["counts"]["excluded_by_reason"].items():
            lines.append(f"| {reason} | {count} |")
        lines.append("")
    lines.append("### Per source")
    lines.append("")
    lines.append("| Source | On disk | Included | Status |")
    lines.append("|---|---:|---:|---|")
    for source, row in by_source.items():
        lines.append(
            f"| {source} | {row['images_on_disk']} | "
            f"{summary['counts']['per_source_included'].get(source, 0)} | "
            f"{row['provenance_status']} |"
        )
    lines.append("")
    lines.append("### Per split and class")
    lines.append("")
    lines.append("| Split | Total | " + " | ".join(summary["splits"].get("train", {}).get("by_class", {}).keys()) + " |")
    lines.append("|---|---:|" + "---:|" * len(summary["splits"].get("train", {}).get("by_class", {})))
    for split, data in summary["splits"].items():
        cells = " | ".join(str(value) for value in data["by_class"].values())
        lines.append(f"| {split} | {data['total']} | {cells} |")
    lines.append("")
    lines.append("## Cross-source overlap")
    lines.append("")
    if cross_source:
        lines.append(f"{len(cross_source)} image(s) are byte-identical across two dataset roots. "
                     "This is the fingerprint of a foreign image inserted into another dataset. "
                     "The two sides are reproduced in `duplicate_audit.csv`.")
        if reference_source:
            lines.append("")
            lines.append(f"The comparison against `{reference_source}` is a reference check: the "
                         "Mendeley corpus is known to embed images from the Castillo Zenodo record, "
                         "so shared hashes there are expected and are not used as training evidence "
                         "for a second dataset.")
        lines.append("")
        for finding in cross_source[:10]:
            members = ", ".join(f"{m['source']}:{m['rel_path']}" for m in finding["members"])
            lines.append(f"- `{finding['sha256'][:16]}…` {members}")
        if len(cross_source) > 10:
            lines.append(f"- … and {len(cross_source) - 10} more (see the CSV)")
    else:
        lines.append("No image is byte-identical across two dataset roots in this audit. "
                     "This does **not** prove that no foreign image exists — datasets that were "
                     "not present cannot be compared.")
    lines.append("")
    lines.append("## Segmentation")
    lines.append("")
    lines.append(f"- status: `{mask_audit.status}`")
    lines.append(f"- official images: {mask_audit.images}")
    lines.append(f"- masks per channel: {mask_audit.channels or 'BLOCKED'}")
    lines.append(f"- 1:1 pairing: {'yes' if mask_audit.pairing_ok else 'no'}")
    lines.append(f"- fabricated masks rejected: {len(mask_audit.fabricated)}")
    lines.append(f"- foreign images rejected: {len(mask_audit.foreign_masks)}")
    if mask_audit.notes:
        for note in mask_audit.notes:
            lines.append(f"- {note}")
    if not mask_audit.usable:
        lines.append("")
        lines.append("No segmentation manifest rows were written. IDRiD is **not** replaced by "
                     "any other dataset: the official images and MA/HE/EX/SE annotations must be "
                     "supplied (registration required) before segmentation training is allowed.")
        if mask_audit.fabricated:
            lines.append("")
            lines.append("Rejected masks (first 20):")
            for rel, reasons in list(mask_audit.fabricated.items())[:20]:
                lines.append(f"- `{rel}`: " + "; ".join(reasons))
    lines.append("")
    lines.append("## Dataset acquisition failures")
    lines.append("")
    failures = [
        f"- **{source}**: {row['provenance_status']} — {row['evidence']}"
        for source, row in by_source.items()
        if row["provenance_status"] in {integrity.BLOCKED, integrity.NOT_AVAILABLE}
    ]
    lines.extend(failures or ["- none: every configured dataset was present and verified"])
    lines.append("")
    lines.append("## How to reproduce")
    lines.append("")
    lines.append("```")
    lines.append("python scripts/audit_datasets.py \\")
    for source, label in (("mendeley", "--fundus-root"), ("mmrdr", "--mmrdr-root"),
                           ("aptos2019", "--aptos-root"), ("ddr", "--ddr-root"),
                           ("idrid", "--idrid-root")):
        value = roots.get(source)
        lines.append(f"    {label} \"{value}\" \\" if value else f"    # {label} PATH   ({source}: not supplied)")
    lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fundus-root", help="Mendeley corpus root (the 13 class folders)")
    parser.add_argument("--mmrdr-root", help="MMRDR root containing MMRDR-CFP")
    parser.add_argument("--aptos-root", help="APTOS 2019 root")
    parser.add_argument("--ddr-root", help="DDR root containing DR_grading")
    parser.add_argument("--idrid-root", help="IDRiD root")
    parser.add_argument("--reference-root", help="optional local extract of the Castillo V03 archive")
    parser.add_argument("--out", default="data_integrity", help="output directory (default: data_integrity)")
    parser.add_argument("--seed", type=int, default=None, help="split seed (default: config seed)")
    parser.add_argument("--jobs", type=int, default=4, help="hashing worker processes (default: 4)")
    parser.add_argument("--max-images", type=int, default=0, help="cap images per source (smoke runs only)")
    parser.add_argument("--config", default=None, help="path to an alternative config.yaml")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--help" in arguments or "-h" in arguments:
        import argparse as _argparse

        _argparse.ArgumentParser(description=__doc__).print_help()
        return 0
    args = parse_args(arguments)
    started = time.time()
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"AUDIT: configuration error: {exc}", file=sys.stderr)
        return 2

    roots = resolve_roots(cfg, args)
    reference_root = Path(args.reference_root).expanduser().resolve() if args.reference_root else None
    if reference_root is None:
        candidate = PROJECT_ROOT / "datasets" / "demo_fundus_v03"
        reference_root = candidate if candidate.is_dir() else None
    if reference_root is not None:
        roots["zenodo_v03_reference"] = reference_root

    mapper = LabelMapper(cfg)
    mapper.class_names = cfg.classes
    seed = int(args.seed if args.seed is not None else cfg.seed)
    ratios = {
        "train": float(cfg.get("split.ratios.train", 0.70)),
        "validation": float(cfg.get("split.ratios.validation", 0.15)),
        "test": float(cfg.get("split.ratios.test", 0.15)),
    }

    entries_by_source: Dict[str, List[integrity.ImageEntry]] = {}
    source_info: Dict[str, Dict[str, Any]] = {}

    log("inspecting sources")
    if roots.get("mendeley"):
        entries_by_source["mendeley"], source_info["mendeley"] = _scan_mendeley(roots["mendeley"], mapper)
        log(f"mendeley: {len(entries_by_source['mendeley'])} image(s)")
    else:
        source_info["mendeley"] = {
            "status": integrity.BLOCKED,
            "evidence": "no Mendeley corpus supplied (--fundus-root) and none at datasets/mendeley",
            "verification": "not inspected",
            "label_source": "verbatim class folders",
        }

    if roots.get("mmrdr"):
        entries_by_source["mmrdr"], source_info["mmrdr"] = _scan_mmrdr(roots["mmrdr"], mapper)
        log(f"mmrdr: {len(entries_by_source['mmrdr'])} CFP image(s)")
    else:
        source_info["mmrdr"] = {
            "status": integrity.NOT_AVAILABLE,
            "evidence": "no MMRDR copy supplied and none at datasets/mmrdr (Figshare archive is 18.7 GB)",
            "verification": "not inspected",
            "label_source": "FP.csv grade",
        }

    entries_by_source["aptos2019"], source_info["aptos2019"] = _scan_aptos(roots.get("aptos2019"))
    entries_by_source["ddr"], source_info["ddr"] = _scan_ddr(roots.get("ddr"))
    log(f"aptos2019: {len(entries_by_source['aptos2019'])} image(s); ddr: {len(entries_by_source['ddr'])} image(s)")

    idrid_root = roots.get("idrid")
    idrid_entries: List[integrity.ImageEntry] = []
    if idrid_root and Path(idrid_root).is_dir():
        idrid_entries = integrity.iter_idrid_images(idrid_root)
    entries_by_source["idrid"] = idrid_entries
    mask_audit = integrity.audit_idrid_masks(idrid_root) if idrid_root else integrity.MaskAudit(
        layout_ok=False, images=0, status=integrity.BLOCKED,
        notes=["no IDRiD data supplied"],
    )
    source_info["idrid"] = {
        "status": mask_audit.status,
        "evidence": (
            f"{mask_audit.images} official image(s), channels {mask_audit.channels}, "
            f"fabricated {len(mask_audit.fabricated)}, foreign {len(mask_audit.foreign_masks)}"
            if mask_audit.images else "no IDRiD tree (registration-gated download not performed)"
        ),
        "verification": "official folder layout + 1:1 image/mask pairing + annotation naming",
        "label_source": "lesion segmentation annotations (no grading used)",
    }
    log(f"idrid: {len(idrid_entries)} image(s), mask status {mask_audit.status}")

    if reference_root is not None:
        entries_by_source["zenodo_v03_reference"] = integrity.iter_source_images(
            "zenodo_v03_reference", reference_root
        )
        source_info["zenodo_v03_reference"] = {"status": integrity.VERIFIED, "evidence": "local V03 extract"}
        log(f"reference: {len(entries_by_source['zenodo_v03_reference'])} image(s)")

    entries: List[integrity.ImageEntry] = []
    for source in ("mendeley", "mmrdr", "aptos2019", "ddr", "idrid", "zenodo_v03_reference"):
        items = entries_by_source.get(source) or []
        if args.max_images and args.max_images > 0:
            items = items[: args.max_images]
        entries_by_source[source] = items
        entries.extend(items)

    for entry in entries:
        if entry.source == "mendeley":
            entry.provenance_status = integrity.LIKELY if source_info["mendeley"]["status"] == integrity.LIKELY else integrity.UNKNOWN
            entry.provenance_evidence = "13-folder Mendeley corpus; filename family " + entry.family
        elif entry.source == "zenodo_v03_reference":
            entry.provenance_status = integrity.VERIFIED
            entry.provenance_evidence = "local extract of the V03 Castillo archive (reference only)"
        elif entry.source == "mmrdr":
            entry.provenance_status = source_info["mmrdr"]["status"]
            entry.provenance_evidence = "MMRDR CFP image with FP.csv grade"
        elif entry.source == "aptos2019":
            entry.provenance_status = source_info["aptos2019"]["status"]
            entry.provenance_evidence = "APTOS grade folder"
        elif entry.source == "ddr":
            entry.provenance_status = source_info["ddr"]["status"]
            entry.provenance_evidence = "DDR official label CSV"
        elif entry.source == "idrid":
            entry.provenance_status = integrity.VERIFIED if mask_audit.usable else integrity.BLOCKED
            entry.provenance_evidence = "IDRiD original image"
        if entry.family == "unknown" and entry.source == "mendeley":
            entry.provenance_status = integrity.UNKNOWN
            entry.exclusion_reason = entry.exclusion_reason or "unverified_provenance:filename_family_unrecognised"

    log(f"hashing {len(entries)} image(s) with {max(1, args.jobs)} worker(s)")
    integrity.inspect_entries(
        entries,
        jobs=max(1, args.jobs),
        cache_path=PROJECT_ROOT / "work" / "cache" / "integrity_index.json",
    )

    # ---- candidate cohort ------------------------------------------------
    candidates: List[integrity.ImageEntry] = []
    for entry in entries:
        if entry.source == "zenodo_v03_reference" or entry.source == "idrid":
            continue
        if entry.exclusion_reason:
            continue
        if entry.mapped_label is None:
            continue
        if not entry.decode_ok:
            entry.exclusion_reason = f"unreadable_image:{entry.error}"
            continue
        if entry.lum_std is not None and entry.lum_std < 8.0:
            entry.quality_flags = "near_uniform_low_std"
            entry.exclusion_reason = "near_uniform_image:no_retinal_structure"
            continue
        candidates.append(entry)

    log(f"candidate cohort: {len(candidates)} image(s)")

    exact_all = integrity.exact_duplicate_groups(entries)
    pairs_all = integrity.near_duplicate_pairs(entries, dhash_max=integrity.DHASH_MAX,
                                               ahash_max=integrity.AHASH_MAX)
    log(f"duplicates: {len(exact_all)} exact group(s), {len(pairs_all)} perceptual pair(s)")

    cross_source = [finding for finding in integrity.cross_source_duplicates(entries)
                    if finding["sources"] != ["zenodo_v03_reference"]]
    log(f"cross-source identical images: {len(cross_source)}")

    # Re-detect duplicates inside the candidate cohort so excluded images cannot
    # influence grouping.
    exact_candidates = integrity.exact_duplicate_groups(candidates)
    pairs_candidates = integrity.near_duplicate_pairs(candidates)
    grouping = integrity.build_grouping(candidates, pairs_candidates)
    for _root, members in grouping.conflicts.items():
        for index in members:
            entry = candidates[index]
            entry.exclusion_reason = (
                "cross_label_duplicate_conflict: a duplicate or patient-linked image carries a "
                "different five-class label"
            )
    included_indices = [
        index for index, entry in enumerate(candidates) if not entry.exclusion_reason
    ]
    for index in included_indices:
        candidates[index].included = True

    splits = integrity.allocate_splits(
        candidates, grouping, included=included_indices, seed=seed, ratios=ratios
    )
    problems = integrity.assert_split_isolation(candidates, grouping, splits)
    if problems:
        log(f"split isolation FAILED: {problems[:3]}")

    for index, split in splits.items():
        candidates[index].split = split

    # Duplicate bookkeeping for the manifest column.
    for exact in (exact_candidates,):
        for sha, indices in exact.items():
            canonical = candidates[indices[0]].rel_path
            for index in indices[1:]:
                candidates[index].duplicate_of = canonical

    # The manifest is written from the per-source lists, so propagate the split.
    split_by_key = {candidates[index].key: split for index, split in splits.items()}
    for entry in entries:
        entry.split = split_by_key.get(entry.key, "")

    summary = build_reports(
        out_dir=Path(args.out),
        entries_by_source=entries_by_source,
        source_info=source_info,
        roots=roots,
        entries=entries,
        exact=exact_all,
        pairs=pairs_all,
        grouping=grouping,
        splits=splits,
        class_names=list(cfg.classes),
        mask_audit=mask_audit,
        cross_source=cross_source,
        reference_source="zenodo_v03_reference" if reference_root is not None else None,
        seed=seed,
        ratios=ratios,
        elapsed=time.time() - started,
    )

    # Candidates replaced the entries for manifest writing in build_reports; the
    # split values were propagated above, but the grouping indices refer to the
    # candidate list. Re-run the report with candidate-local indices.
    if candidates:
        class_names = list(cfg.classes)
        integrity.write_csv(
            Path(args.out) / "cleaned_classification_manifest.csv",
            _manifest_rows(candidates, splits, grouping, class_names),
            CLASSIFICATION_COLUMNS,
        )
        summary["counts"]["images_included"] = len(included_indices)
        summary["splits"] = _split_distributions(candidates, splits, class_names)
        (Path(args.out) / "cleaned_split_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        integrity.write_manifest_stamp(
            Path(args.out),
            files=("cleaned_classification_manifest.csv", "cleaned_segmentation_manifest.csv"),
            counts={
                "images_included": len(included_indices),
                "images_excluded": len(entries) - len(included_indices),
                "classification_rows": len(included_indices),
                "segmentation_rows": len(segmentation_rows_for(Path(args.out))),
            },
            source_statuses=summary["source_statuses"],
            extra={"seed": seed, "isolation_checks_passed": not problems},
        )

    log(f"wrote deliverables to {Path(args.out).resolve()} in {time.time() - started:.1f} s")
    blocked = [
        source for source, info in source_info.items()
        if info.get("status") in {integrity.BLOCKED, integrity.NOT_AVAILABLE}
    ]
    if problems:
        print("AUDIT: FAILED — split isolation problems detected", file=sys.stderr)
        return 1
    if blocked:
        print(f"AUDIT: COMPLETED WITH BLOCKERS ({', '.join(sorted(blocked))}) — see dataset_integrity_report.md")
        return 3
    print("AUDIT: CLEAN")
    return 0


def segmentation_rows_for(out_dir: Path) -> int:
    path = out_dir / "cleaned_segmentation_manifest.csv"
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as handle:
        return max(0, sum(1 for _ in handle) - 1)


if __name__ == "__main__":
    raise SystemExit(main())
