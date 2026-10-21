"""Setup helpers for ``colab/train_on_colab_fixed.ipynb``.

Pure, CPU-cheap helpers the fixed notebook imports after it has located the
project. This module:

- **never** installs packages itself (it *plans* repairs; the notebook executes
  the plan and verifies the result afterwards),
- **never** downloads pretrained weights,
- **never** trains or evaluates a model,
- **never** writes anything inside the audited project: the only file it writes
  is a clearly derived canary split CSV under the user-visible output folder
  (:func:`build_canary_split`).

The pinned split (``provisional_primary_split/provisional_primary_split.csv``),
the provisional label policy, and every audit artifact are only ever *read*.

Provisional status: the five-band mapping is provisional and unsigned; see
``audit_output/pretraining_audit/LABEL_POLICY_DRAFT.md``. Nothing produced
through these helpers is a clinical result.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Pinned facts about the existing project artifacts
# ---------------------------------------------------------------------------

#: Files that must all exist for a folder to be considered the project root.
#: The notebook keeps a copy of this tuple and asserts the two agree.
PROJECT_MARKERS = (
    "dataset/labels.py",
    "colab/train.py",
    "colab/setup_helpers.py",
    "provisional_primary_split/provisional_primary_split.csv",
)

#: One marker is enough to *find* candidate folders cheaply before the full
#: marker validation runs (the notebook's bootstrap cell searches for this).
BOOTSTRAP_MARKER = "colab/setup_helpers.py"

PROJECT_ENV_VAR = "DR_PROJECT_ROOT"
DATASET_ENV_VAR = "DR_FUNDUS_ROOT"
SPLIT_ENV_VAR = "DR_SPLIT_DIR"
OUTPUT_ENV_VAR = "DR_OUTPUT_DIR"

SPLIT_DIR_NAME = "provisional_primary_split"
SPLIT_CSV_NAME = "provisional_primary_split.csv"
DATASET_DIR_NAME = "Diabetic Retinopathy and Retinal Diseases Fundus I"

ELIGIBILITY_MANIFEST_RELPATH = Path("audit_output/pretraining_audit/PROVISIONAL_ELIGIBILITY_PRIMARY.csv")
LABEL_POLICY_RELPATH = Path("audit_output/pretraining_audit/LABEL_POLICY_DRAFT.md")

#: The audited split, pinned exactly. A different artifact must fail loudly
#: rather than "adjust" itself to pass.
EXPECTED_ROWS = 14201
EXPECTED_PARTITIONS = ("train", "validation", "test")
EXPECTED_PARTITION_COUNTS = {"train": 9924, "validation": 2133, "test": 2144}
EXPECTED_CLASSES = [
    "No apparent DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "PDR",
]
REQUIRED_SPLIT_COLUMNS = ("image_path", "task", "partition", "primary_class", "group_id")

#: Minimum versions the current code genuinely requires. These gate *repair*
#: decisions only; a working, newer stack is never downgraded to match them.
#: - torchvision >= 0.13 is the first release with the ``weights=`` enum API.
#: - torch >= 1.13 is the first release with ``torch.load(weights_only=...)``.
MINIMUM_VERSIONS = {"torch": (1, 13), "torchvision": (0, 13), "numpy": (1, 22), "Pillow": (9, 0)}

#: (distribution name, import name) pairs inspected by the runtime report.
INSPECTED_PACKAGES = (
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("numpy", "numpy"),
    ("Pillow", "PIL"),
    ("pytest", "pytest"),
)

PROVISIONAL_DISCLOSURE = (
    "PROVISIONAL LABEL POLICY - read before trusting any number this notebook prints.\n"
    "The primary task uses a five-band mapping (No apparent DR / Mild NPDR / Moderate NPDR /\n"
    "Severe NPDR / PDR). '13. Very Severe NPDR' merges into Severe NPDR and '12. Advanced PDR'\n"
    "merges into PDR. Both merges are PROVISIONAL and unsigned: no retinal specialist has\n"
    "confirmed the mapping, and no per-image clinical adjudication has ever occurred.\n"
    "503 Zenodo-derived images lack verified patient linkage, so patient independence is not\n"
    "claimed. Folder names are source-specific weak supervision, not ground truth.\n"
    "Nothing produced by this notebook is a clinical result."
)


# ---------------------------------------------------------------------------
# Project discovery support
# ---------------------------------------------------------------------------

def missing_markers(root: Path | str, markers: Sequence[str] | None = None) -> List[str]:
    """Return the marker paths (relative) that do not exist under ``root``."""
    root = Path(root)
    markers = PROJECT_MARKERS if markers is None else markers
    return [m for m in markers if not (root / m).is_file()]


def bounded_scan(
    top: Path | str,
    *,
    marker: str = BOOTSTRAP_MARKER,
    max_depth: int = 4,
    max_visit_dirs: int = 3000,
) -> Dict[str, Any]:
    """Breadth-first search for ``marker`` under ``top``, with hard bounds.

    Bounds that keep this from ever becoming an unbounded Drive crawl:

    - ``max_depth``: directories deeper than this are never entered.
    - ``max_visit_dirs``: at most this many directories are opened in total;
      ``truncated`` reports when the budget ran out first.
    - hidden directories (``.git`` etc.) and ``__pycache__`` /
      ``node_modules`` / ``site-packages`` are skipped.
    - a directory that contains ``marker`` is recorded and *not* descended
      into (a project does not contain a second copy of itself).

    Returns ``{"matches": [Path, ...], "visited": int, "truncated": bool}``.
    """
    top = Path(top)
    if not top.is_dir():
        return {"matches": [], "visited": 0, "truncated": False}
    matches: List[Path] = []
    queue: List[Path] = [top]
    visited = 0
    truncated = False
    skip_names = {"__pycache__", "node_modules", "site-packages"}
    while queue:
        if visited >= max_visit_dirs:
            truncated = True
            break
        current = queue.pop(0)
        visited += 1
        if (current / marker).is_file():
            matches.append(current)
            continue
        try:
            children = sorted(
                (
                    d
                    for d in current.iterdir()
                    if d.is_dir()
                    and not d.name.startswith(".")
                    and d.name not in skip_names
                ),
                key=lambda p: p.name,
            )
        except OSError:
            continue
        depth = len(current.relative_to(top).parts)
        if depth < max_depth:
            queue.extend(children)
    return {"matches": matches, "visited": visited, "truncated": truncated}


def resolve_output_root(
    project_root: Path | str,
    *,
    explicit: str | Path | None = None,
    env: Dict[str, str] | None = None,
    drive_mount: Path | str = "/content/drive/MyDrive",
    default_name: str = "dr_baseline_outputs",
) -> Tuple[Path, bool, List[str]]:
    """Choose where runs are written, refusing unsafe paths.

    Resolution order: explicit ``OUTPUT_PATH`` -> ``DR_OUTPUT_DIR`` env ->
    ``<drive>/dr_baseline_outputs`` when Drive is mounted -> the project folder
    (with ``persistent=False`` because a Colab runtime is ephemeral).

    Safety rule: an *explicit* output path may live inside the project (the
    default local case) but may never be the project root itself, one of
    ``{dataset, colab, tests, audit_output, provisional_primary_split}``, or an
    ancestor of the project root - because ``overwrite=True`` on an output
    directory deletes its previous run recursively.

    Returns ``(output_dir, persistent, checked)`` where ``checked`` lists every
    candidate that was considered, with why it was chosen or skipped.
    """
    env = os.environ if env is None else env
    project_root = Path(project_root)
    checked: List[str] = []

    candidates: List[Tuple[str, Path]] = []
    explicit = str(explicit).strip() if explicit not in (None, "") else ""
    if explicit:
        candidates.append(("config OUTPUT_PATH", Path(explicit).expanduser()))
    env_out = (env.get(OUTPUT_ENV_VAR) or "").strip()
    if env_out and not explicit:
        candidates.append((f"env {OUTPUT_ENV_VAR}", Path(env_out).expanduser()))

    drive = Path(drive_mount)
    if not candidates:
        if drive.is_dir():
            candidates.append(("Drive default", drive / default_name))
        else:
            checked.append(f"Drive default: {drive / default_name} (Drive not mounted)")
            candidates.append(("project folder default", project_root / default_name))

    source, chosen = candidates[0]
    # Safety: explicit user/env choices cannot point at the project itself,
    # its audited subfolders, or anywhere above the project.
    if source in ("config OUTPUT_PATH", f"env {OUTPUT_ENV_VAR}"):
        protected = [project_root] + [project_root / d for d in
                                      ("dataset", "colab", "tests", "audit_output", SPLIT_DIR_NAME)]
        resolved = chosen.resolve() if chosen.exists() else chosen
        if resolved in {p.resolve() if p.exists() else p for p in protected}:
            raise ValueError(
                f"Refusing OUTPUT_DIR={chosen}: it equals {project_root} or one of its protected "
                f"folders (dataset, colab, tests, audit_output, provisional_primary_split). "
                "An output directory with overwrite=True would delete these. "
                "Pick a fresh folder such as <project>/dr_baseline_outputs_runs or a Drive path."
            )
        root_resolved = project_root.resolve()
        if root_resolved == resolved or root_resolved in resolved.parents:
            raise ValueError(
                f"Refusing OUTPUT_DIR={chosen}: {project_root} lives inside it, so a destructive "
                "overwrite would remove the project itself. Choose a folder that does not "
                "contain the project (inside the project is fine)."
            )
    checked.append(f"output dir: {source} -> {chosen}")
    persistent = str(chosen).startswith(str(drive))
    if not persistent:
        checked.append("note: output is NOT on Drive; a Colab runtime discards it when it ends.")
    return chosen, persistent, checked


def _looks_like_dataset_root(path: Path) -> bool:
    """True when ``path`` holds the fundus corpus (at least one source folder)."""
    if not path.is_dir():
        return False
    try:
        from dataset.labels import PRIMARY_MAPPING

        keys: Sequence[str] = tuple(PRIMARY_MAPPING)
    except Exception:  # pragma: no cover - dataset package absent from sys.path
        keys = ()
    try:
        if keys:
            return any((path / k).is_dir() for k in keys)
        return any(child.is_dir() for child in path.iterdir())
    except OSError:
        return False


def resolve_data_root(
    project_root: Path | str,
    *,
    explicit: str | Path | None = None,
    env: Dict[str, str] | None = None,
    dataset_dir_name: str = DATASET_DIR_NAME,
) -> Tuple[Optional[Path], List[str]]:
    """Locate the fundus image corpus. Returns ``(root_or_None, checked)``.

    Candidates, in order: explicit ``DATASET_PATH`` -> ``DR_FUNDUS_ROOT`` env
    -> the dataset folder next to the project -> inside the project -> the
    Drive root -> ``/content``. ``checked`` names every candidate so a failure
    can report *exactly* which paths were tested. Nothing is created or copied.
    """
    env = os.environ if env is None else env
    project_root = Path(project_root)
    checked: List[str] = []
    candidates: List[Tuple[str, Path]] = []

    def _push(source: str, value: Any) -> None:
        if value in (None, ""):
            return
        try:
            p = Path(str(value)).expanduser()
        except Exception:  # pragma: no cover - defensive
            return
        if p not in [c for _, c in candidates]:
            candidates.append((source, p))

    explicit = str(explicit).strip() if explicit not in (None, "") else ""
    _push("config DATASET_PATH", explicit)
    _push(f"env {DATASET_ENV_VAR}", (env.get(DATASET_ENV_VAR) or "").strip())
    _push("dataset folder next to the project", project_root.parent / dataset_dir_name)
    _push("dataset folder inside the project", project_root / dataset_dir_name)
    _push("Drive MyDrive root", Path("/content/drive/MyDrive") / dataset_dir_name)
    _push("/content root", Path("/content") / dataset_dir_name)

    for source, p in candidates:
        if _looks_like_dataset_root(p):
            checked.append(f"{source}: {p} (OK)")
            return p, checked
        if p.is_dir():
            reason = "no source-label subfolder (e.g. '11.Normal Fundus/') inside"
        else:
            reason = "path does not exist"
        checked.append(f"{source}: {p} ({reason})")
    return None, checked


# ---------------------------------------------------------------------------
# Split / artifact validation
# ---------------------------------------------------------------------------

def _stratified_sample(
    rows: Sequence[Dict[str, str]],
    per_stratum: int,
) -> List[Dict[str, str]]:
    """First ``per_stratum`` rows of every (partition, class) stratum - up to 45."""
    quota: Dict[Tuple[str, str], int] = {}
    out: List[Dict[str, str]] = []
    for row in rows:
        key = (row.get("partition", ""), row.get("primary_class", ""))
        seen = quota.get(key, 0)
        if seen < per_stratum and per_stratum > 0:
            quota[key] = seen + 1
            out.append(row)
    return out


def _primary_mapping() -> Optional[Dict[str, str]]:
    try:
        from dataset.labels import PRIMARY_MAPPING

        return dict(PRIMARY_MAPPING)
    except Exception:  # pragma: no cover - dataset package absent
        return None


def validate_split_artifact(
    split_dir: Path | str,
    *,
    project_root: Path | str | None = None,
    image_root: Path | str | None = None,
    expected_rows: int = EXPECTED_ROWS,
    expected_partition_counts: Dict[str, int] | None = None,
    expected_classes: Sequence[str] | None = None,
    path_check: str = "sample",
    path_sample_per_stratum: int = 4,
    decode_sample_per_stratum: int = 2,
    run_protocol_validation: bool = True,
) -> Dict[str, Any]:
    """Validate the pinned split CSV without modifying anything.

    Checks (each reported individually with PASS/FAIL/SKIP + detail):

    1. split CSV exists at the expected path and is readable with the
       required columns;
    2. exactly ``expected_rows`` rows (14,201 for the real artifact);
    3. every row belongs to the primary task;
    4. partitions are exactly train/validation/test with the audited counts
       9,924 / 2,133 / 2,144;
    5. classes match ``EXPECTED_CLASSES`` **and**
       ``dataset.labels.PRIMARY_MAPPING`` (provisional merges intact);
    6. no duplicated ``image_path`` (nothing silently remapped);
    7. every class present in every partition;
    8. each row's path prefix maps through ``PRIMARY_MAPPING`` to its
       ``primary_class`` (no silent label/path remapping);
    9. no group leakage: every ``group_id`` sits in exactly one partition;
    10. full re-validation against the eligibility manifest with the project's
        own ``SplitValidator`` (the existing split protocol);
    11. image files decode through ``dataset.decoding.load_rgb`` on a
        stratified sample (existence checked on a stratified sample, or on
        every row with ``path_check="all"``).

    ``path_check`` is one of ``"sample"`` (default), ``"all"``, ``"off"``.
    Decode/path checks SKIP (never silently pass) when ``image_root`` is
    unresolved; the notebook gates that case separately.

    Returns ``{"ok", "checks", "rows", "partition_counts", "skipped"}``.
    """
    expected_partition_counts = dict(
        EXPECTED_PARTITION_COUNTS if expected_partition_counts is None else expected_partition_counts
    )
    expected_classes = list(EXPECTED_CLASSES if expected_classes is None else expected_classes)
    split_dir = Path(split_dir)
    csv_path = split_dir / SPLIT_CSV_NAME
    checks: List[Dict[str, Any]] = []

    def _add(name: str, ok: bool, detail: str, *, status: Optional[str] = None) -> None:
        status = status or ("pass" if ok else "fail")
        checks.append(
            {
                "name": name,
                "status": status,
                "ok": status in ("pass", "skip"),
                "detail": detail,
            }
        )

    def _finish(rows: List[Dict[str, str]]) -> Dict[str, Any]:
        partition_counts: Dict[str, int] = {}
        for r in rows:
            key = r.get("partition") or "<missing>"
            partition_counts[key] = partition_counts.get(key, 0) + 1
        skipped = [c["name"] for c in checks if c["status"] == "skip"]
        return {
            "ok": all(c["ok"] for c in checks) and bool(checks),
            "checks": checks,
            "rows": len(rows),
            "partition_counts": partition_counts,
            "skipped": skipped,
            "split_csv": str(csv_path),
        }

    image_root_path: Optional[Path] = Path(image_root) if image_root else None

    # 1. existence -------------------------------------------------------
    if not csv_path.is_file():
        _add("split_csv_exists", False, f"missing: {csv_path}")
        return _finish([])
    _add("split_csv_exists", True, str(csv_path))

    # 2. readability + columns ------------------------------------------
    try:
        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            header = list(reader.fieldnames or [])
            rows = [dict(r) for r in reader]
    except Exception as exc:
        _add("split_csv_readable", False, f"{type(exc).__name__}: {exc}")
        return _finish([])
    missing_cols = [c for c in REQUIRED_SPLIT_COLUMNS if c not in header]
    if missing_cols:
        _add("split_csv_readable", False, f"missing columns {missing_cols}; header is {header}")
        return _finish([])
    _add("split_csv_readable", True, f"{len(rows)} data rows, columns {header}")

    # 3. row count -------------------------------------------------------
    ok = len(rows) == expected_rows
    _add(
        "row_count",
        ok,
        f"expected exactly {expected_rows} rows, found {len(rows)}"
        if not ok
        else f"{len(rows)} rows (matches the audited artifact)",
    )

    # 4. primary task membership ----------------------------------------
    tasks = sorted({r.get("task", "") for r in rows})
    _add(
        "all_primary_task",
        tasks == ["primary"],
        "every row has task='primary'" if tasks == ["primary"] else f"unexpected task values: {tasks}",
    )

    # 5. partitions + audited counts -------------------------------------
    parts = sorted({r.get("partition", "") for r in rows})
    expected_parts = sorted(EXPECTED_PARTITIONS)
    _add(
        "partitions_exact",
        parts == expected_parts,
        "partitions are exactly train/validation/test"
        if parts == expected_parts
        else f"expected {expected_parts}, found {parts}",
    )
    actual_counts: Dict[str, int] = {}
    for r in rows:
        actual_counts[r.get("partition") or "<missing>"] = actual_counts.get(
            r.get("partition") or "<missing>", 0
        ) + 1
    ok = actual_counts == expected_partition_counts
    _add(
        "partition_counts_match_audited_split",
        ok,
        "train 9,924 / validation 2,133 / test 2,144"
        if ok
        else f"expected {expected_partition_counts}, found {actual_counts}",
    )

    # 6. class vocabulary + label mapping agreement -----------------------
    classes = sorted({r.get("primary_class", "") for r in rows})
    expected_sorted = sorted(expected_classes)
    _add(
        "classes_are_the_five_provisional_bands",
        classes == expected_sorted,
        "exactly the five expected bands" if classes == expected_sorted
        else f"expected {expected_sorted}, found {classes}",
    )
    mapping = _primary_mapping()
    if mapping is None:
        _add("labels_module_agreement", False,
             "dataset.labels could not be imported - is the repo root on sys.path?")
    else:
        map_values = sorted(set(mapping.values()))
        _add(
            "labels_module_agreement",
            map_values == expected_sorted,
            "dataset.labels.PRIMARY_MAPPING values match EXPECTED_CLASSES"
            if map_values == expected_sorted
            else f"PRIMARY_MAPPING values {map_values} != expected {expected_sorted}",
        )
        merges_ok = (
            mapping.get("13. Very Severe NPDR") == "Severe NPDR"
            and mapping.get("12. Advanced PDR") == "PDR"
        )
        _add(
            "provisional_merges_intact",
            merges_ok,
            "13. Very Severe NPDR->Severe NPDR and 12. Advanced PDR->PDR (PROVISIONAL)"
            if merges_ok
            else "the two provisional merges are missing from dataset.labels",
        )
    try:
        import colab_helpers as H  # type: ignore

        band_ok = list(H.BAND_ORDER) == list(expected_classes)
        _add(
            "band_order_matches_colab_helpers",
            band_ok,
            "colab_helpers.BAND_ORDER agrees" if band_ok
            else f"colab_helpers.BAND_ORDER {list(H.BAND_ORDER)} != {list(expected_classes)}",
        )
    except Exception as exc:  # pragma: no cover - colab dir absent from sys.path
        _add("band_order_matches_colab_helpers", False,
             f"colab_helpers unavailable: {type(exc).__name__}: {exc}")

    # 7. duplicates --------------------------------------------------------
    seen: Dict[str, int] = {}
    dups: List[str] = []
    for r in rows:
        p = r.get("image_path", "")
        if p in seen:
            dups.append(p)
        seen[p] = 1
    _add(
        "image_paths_unique",
        not dups,
        "no duplicated image_path" if not dups
        else f"{len(dups)} duplicated paths, first: {dups[:3]}",
    )

    # 8. every band present in every partition -----------------------------
    absent = [
        f"{pt}/{cls}"
        for pt in EXPECTED_PARTITIONS
        for cls in expected_classes
        if not any(r.get("partition") == pt and r.get("primary_class") == cls for r in rows)
    ]
    _add(
        "every_band_in_every_partition",
        not absent,
        "all five bands present in all three partitions" if not absent
        else f"missing strata: {absent[:6]}",
    )

    # 9. path <-> label consistency -----------------------------------------
    if mapping is None:
        _add("path_prefix_matches_label", False, "dataset.labels unavailable; cannot verify")
    else:
        unknown: List[str] = []
        mismatched: List[str] = []
        for r in rows:
            prefix = r.get("image_path", "").split("/")[0]
            expected_label = mapping.get(prefix)
            if expected_label is None:
                unknown.append(prefix)
            elif r.get("primary_class") != expected_label:
                mismatched.append(f"{r.get('image_path')} -> {r.get('primary_class')}")
        _add(
            "path_prefix_matches_label",
            not unknown and not mismatched,
            "every path's source folder maps through PRIMARY_MAPPING to its primary_class"
            if not unknown and not mismatched
            else f"unknown prefixes {sorted(set(unknown))[:5]}, mismatches {mismatched[:5]}",
        )

    # 10. group leakage -------------------------------------------------------
    group_parts: Dict[str, set] = {}
    for r in rows:
        group_parts.setdefault(r.get("group_id", ""), set()).add(r.get("partition", ""))
    leaks = sorted(
        g for g, parts_set in group_parts.items() if len(parts_set) > 1
    )
    _add(
        "no_group_leakage_across_partitions",
        not leaks,
        f"{len(group_parts)} groups, each confined to one partition"
        if not leaks
        else f"{len(leaks)} groups span partitions, first: {[(g, sorted(group_parts[g])) for g in leaks[:3]]}",
    )

    # 11. full protocol re-validation ------------------------------------------
    if not run_protocol_validation:
        _add("split_protocol_validation", True,
             "disabled (run_protocol_validation=False)", status="skip")
    elif project_root is None:
        _add("split_protocol_validation", True,
             "skipped: project_root not provided", status="skip")
    else:
        manifest_path = Path(project_root) / ELIGIBILITY_MANIFEST_RELPATH
        if not manifest_path.is_file():
            _add(
                "split_protocol_validation",
                False,
                f"missing eligibility manifest: {manifest_path} "
                "(upload the full project including audit_output/)",
            )
        else:
            try:
                from dataset.manifest import load_manifest
                from dataset.split_validation import SplitValidator

                manifest_rows = load_manifest(manifest_path, "primary")
                assignment = {r["image_path"]: r["partition"] for r in rows}
                validator = SplitValidator(manifest_rows, allowed_partitions=list(EXPECTED_PARTITIONS))
                report = validator.validate(assignment)
                _add(
                    "split_protocol_validation",
                    bool(report.get("valid")),
                    f"SplitValidator OK: {report.get('total_assigned')} assigned, "
                    f"warnings={len(report.get('warnings') or [])}",
                )
            except Exception as exc:
                _add(
                    "split_protocol_validation",
                    False,
                    f"{type(exc).__name__}: {exc}",
                )

    # 12. image files exist ---------------------------------------------------
    if image_root_path is None:
        _add("image_files_exist", True,
             "skipped: dataset image root unresolved (reported by SECTION 2)", status="skip")
    elif path_check == "off":
        _add("image_files_exist", True, "off (path_check='off')", status="skip")
    elif path_check == "all":
        missing = [
            str(image_root_path / r["image_path"])
            for r in rows
            if not (image_root_path / r["image_path"]).is_file()
        ]
        _add(
            "image_files_exist",
            not missing,
            f"all {len(rows)} image paths exist under {image_root_path}"
            if not missing
            else f"{len(missing)} missing files, first: {missing[:5]}",
        )
    else:  # sample
        sample = _stratified_sample(rows, path_sample_per_stratum)
        missing = [
            str(image_root_path / r["image_path"])
            for r in sample
            if not (image_root_path / r["image_path"]).is_file()
        ]
        _add(
            "image_files_exist",
            not missing,
            f"{len(sample)}/{len(rows)} stratified sample paths exist under {image_root_path}"
            if not missing
            else f"{len(missing)} of {len(sample)} sampled paths missing, first: {missing[:5]}",
        )

    # 13. decoding ---------------------------------------------------------------
    if image_root_path is None:
        _add("sample_images_decode", True,
             "skipped: dataset image root unresolved", status="skip")
    else:
        decode_sample = _stratified_sample(rows, decode_sample_per_stratum)
        failures: List[str] = []
        sizes: List[Tuple[int, int]] = []
        from dataset.decoding import load_rgb  # existing, already-tested decoder

        for r in decode_sample:
            p = image_root_path / r["image_path"]
            try:
                im = load_rgb(p)
                if im.mode != "RGB":
                    failures.append(f"{p}: mode {im.mode}")
                else:
                    sizes.append(im.size)
            except Exception as exc:
                failures.append(f"{p}: {type(exc).__name__}: {exc}")
        _add(
            "sample_images_decode",
            not failures and bool(decode_sample),
            f"{len(decode_sample)} stratified images decoded via dataset.decoding.load_rgb "
            f"(size range {min(sizes) if sizes else '-'}..{max(sizes) if sizes else '-'})"
            if not failures and decode_sample
            else (f"decoding failures: {failures[:3]}" if failures
                  else "no rows available to decode"),
        )

    return _finish(rows)


# ---------------------------------------------------------------------------
# Runtime inspection + dependency repair planning
# ---------------------------------------------------------------------------

def parse_version_tuple(version: str) -> Tuple[int, ...]:
    """``'2.11.0+cu130'`` -> ``(2, 11, 0)``. Non-numeric segments stop parsing."""
    base = str(version).split("+", 1)[0]
    parts: List[int] = []
    for chunk in base.split("."):
        m = re.match(r"\d+", chunk)
        if not m:
            break
        parts.append(int(m.group(0)))
    return tuple(parts)


def meets_minimum(version: str, minimum: Tuple[int, ...]) -> bool:
    v = parse_version_tuple(version)
    return v >= minimum


def _pytorch_wheel_index(torch_version: str) -> Optional[str]:
    """Derive the wheel index that matches the installed torch build tag.

    ``2.11.0+cu130`` -> ``https://download.pytorch.org/whl/cu130``; a plain
    PyPI build (no local tag) -> ``None`` (plain PyPI is correct already).
    """
    if "+" not in torch_version:
        return None
    tag = torch_version.split("+", 1)[1].lower()
    if re.fullmatch(r"cu\d{2,3}", tag) or tag.startswith(("cpu", "rocm", "xpu")):
        return f"https://download.pytorch.org/whl/{tag}"
    return None


def inspect_runtime_environment() -> Dict[str, Any]:
    """Report the runtime's real state. Read-only; imports never install."""
    import importlib

    info: Dict[str, Any] = {
        "python_version": ".".join(str(v) for v in sys.version_info[:3]),
        "executable": sys.executable,
        "packages": {},
        "loaded_modules": [m for m in ("torch", "torchvision", "numpy") if m in sys.modules],
    }
    for dist, import_name in INSPECTED_PACKAGES:
        entry: Dict[str, Any] = {"present": False, "version": None, "error": None}
        try:
            mod = importlib.import_module(import_name)
            entry["present"] = True
            entry["version"] = str(getattr(mod, "__version__", "unknown"))
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        info["packages"][dist] = entry
    info["cuda_available"] = False
    info["cuda_device_name"] = None
    if info["packages"]["torch"]["present"]:
        try:
            import torch

            info["cuda_available"] = bool(torch.cuda.is_available())
            if info["cuda_available"]:
                info["cuda_device_name"] = torch.cuda.get_device_name(0)
        except Exception as exc:  # pragma: no cover - driver-dependent
            info["cuda_error"] = f"{type(exc).__name__}: {exc}"
    return info


def plan_dependency_repair(
    report: Dict[str, Any],
    *,
    requirements_path: str | Path | None = None,
    min_versions: Dict[str, Tuple[int, ...]] | None = None,
    python_executable: str | None = None,
) -> Dict[str, Any]:
    """Inspect a runtime report and decide the *minimal* safe repair.

    Never plans a downgrade: a healthy, working stack is left untouched.
    Possible actions:

    - ``"none"`` - everything present and compatible; install nothing.
    - ``"install_stack"`` - torch itself is missing or too old, so there is no
      healthy CUDA stack to protect: the pinned requirements file is correct.
    - ``"install_torchvision"`` - torch is healthy; fetch only torchvision from
      the index matching torch's build tag. pip selects the torchvision release
      whose exact ``torch==`` pin already matches the installed torch (PEP 440
      local versions such as ``+cu130`` satisfy ``==2.11.0``), so torch itself
      is not replaced.
    - ``"install_additive"`` - only genuinely missing, stack-independent
      packages (numpy / Pillow / pytest) are absent.

    Returns a dict with ``action``, ``reason``, ``command`` (argv list or
    None), ``installing``, ``restart_required``, ``verify_after`` and
    ``failure_fallback_action`` (the single recommended manual action if the
    automatic repair fails).
    """
    min_versions = dict(MINIMUM_VERSIONS if min_versions is None else min_versions)
    python_executable = python_executable or sys.executable
    req_path = str(requirements_path or (Path("colab") / "requirements-colab.txt"))
    pkgs: Dict[str, Any] = report.get("packages", {})
    loaded = set(report.get("loaded_modules", []))

    def _pkg(name: str) -> Dict[str, Any]:
        return pkgs.get(name, {"present": False, "version": None, "error": "not inspected"})

    plan: Dict[str, Any] = {
        "action": "none",
        "reason": "",
        "command": None,
        "installing": [],
        "restart_required": False,
        "verify_after": True,
        "notes": [],
        "failure_fallback_action": (
            "Runtime > Restart runtime, then re-run SECTION 0 through SECTION 3; if the problem "
            f"persists install manually: {python_executable} -m pip install -r {req_path}"
        ),
    }

    torch_p = _pkg("torch")
    tv_p = _pkg("torchvision")

    if not torch_p["present"]:
        plan["action"] = "install_stack"
        plan["reason"] = (
            f"torch is not importable ({torch_p.get('error')}); there is no working stack to "
            "preserve, so the pinned environment in colab/requirements-colab.txt is safe to install."
        )
        plan["command"] = [python_executable, "-m", "pip", "install", "-r", req_path]
        plan["installing"] = ["torch", "torchvision", "numpy", "Pillow"]
        plan["restart_required"] = bool(loaded & {"torch", "torchvision"})
        plan["notes"].append("Only chosen when torch is absent or unusable; a healthy CUDA build is never replaced.")
        plan["failure_fallback_action"] = (
            "Restart the runtime (Runtime > Restart runtime) and re-run SECTION 0-4; install "
            f"manually only if it still fails: {python_executable} -m pip install -r {req_path}"
        )
        return plan

    torch_version = str(torch_p.get("version") or "")
    torch_min = min_versions.get("torch", (0,))
    if not meets_minimum(torch_version, tuple(torch_min)):
        plan["action"] = "install_stack"
        plan["reason"] = (
            f"torch {torch_version} is older than the minimum "
            f"{'.'.join(str(x) for x in torch_min)} this pipeline requires, so it cannot run the "
            "training code as-is; the pinned requirements file is the supported replacement."
        )
        plan["command"] = [python_executable, "-m", "pip", "install", "-r", req_path]
        plan["installing"] = ["torch", "torchvision"]
        plan["restart_required"] = bool(loaded & {"torch", "torchvision"})
        plan["notes"].append("This REPLACES the old torch build; a runtime restart is required if it is already imported.")
        plan["failure_fallback_action"] = (
            f"Install manually ({python_executable} -m pip install -r {req_path}), then "
            "Runtime > Restart runtime and re-run SECTION 0-4."
        )
        return plan

    torch_min_txt = ".".join(str(x) for x in torch_min)
    if not tv_p["present"]:
        index = _pytorch_wheel_index(torch_version)
        cmd = [python_executable, "-m", "pip", "install", "torchvision"]
        if index:
            cmd += ["--index-url", index]
        plan["action"] = "install_torchvision"
        plan["reason"] = (
            f"torch {torch_version} is present and compatible (>= {torch_min_txt}), but torchvision "
            "is missing - it is required to build ResNet-50."
        )
        plan["command"] = cmd
        plan["installing"] = ["torchvision"]
        plan["restart_required"] = False
        plan["notes"].append(
            f"Index {index or 'PyPI'} matches the installed torch build tag; pip only adds "
            "torchvision and leaves the working CUDA torch stack untouched."
        )
        plan["failure_fallback_action"] = (
            f"Run the install manually ({' '.join(cmd)}); if it still fails, do NOT upgrade torch - "
            "report the exact pip error instead."
        )
        return plan

    tv_version = str(tv_p.get("version") or "")
    tv_min = min_versions.get("torchvision", (0,))
    if not meets_minimum(tv_version, tuple(tv_min)):
        index = _pytorch_wheel_index(torch_version)
        cmd = [python_executable, "-m", "pip", "install", "--upgrade", "torchvision"]
        if index:
            cmd += ["--index-url", index]
        plan["action"] = "install_torchvision"
        plan["reason"] = (
            f"torchvision {tv_version} predates the weights= API (needs >= "
            f"{'.'.join(str(x) for x in tv_min)}); torch {torch_version} itself is fine."
        )
        plan["command"] = cmd
        plan["installing"] = ["torchvision"]
        plan["restart_required"] = "torchvision" in loaded
        plan["notes"].append(
            "Only torchvision is touched; torch stays at its working version."
        )
        plan["failure_fallback_action"] = (
            f"Run the install manually ({' '.join(cmd)}) then Runtime > Restart runtime."
        )
        return plan

    additive: List[str] = []
    for name in ("numpy", "Pillow", "pytest"):
        entry = _pkg(name)
        if not entry["present"]:
            additive.append(name)
    if additive:
        plan["action"] = "install_additive"
        plan["reason"] = (
            "torch/torchvision are healthy; only stack-independent packages are missing: "
            + ", ".join(additive)
            + "."
        )
        plan["command"] = [python_executable, "-m", "pip", "install", *additive]
        plan["installing"] = additive
        plan["restart_required"] = False
        plan["notes"].append("Additive install; the GPU stack is not involved.")
        plan["failure_fallback_action"] = (
            f"Run manually ({python_executable} -m pip install {' '.join(additive)}) and re-run SECTION 3."
        )
        return plan

    plan["reason"] = (
        f"torch {torch_version} and torchvision {tv_version} both import and meet their minimums "
        f"(cuda_available={bool(report.get('cuda_available'))}); nothing will be installed."
    )
    return plan


#: Executed in a *subprocess* by :func:`verify_stack` so it proves the on-disk
#: stack works regardless of what this kernel imported earlier. Uses
#: ``weights=None`` - it never downloads anything.
_VERIFY_STACK_CODE = r"""
import json, sys
try:
    import torch
    import torchvision
    from torchvision.models import resnet50
    net = resnet50(weights=None)
    net.eval()
    with torch.no_grad():
        out = net(torch.zeros(1, 3, 224, 224))
    print(json.dumps({"ok": True, "torch": torch.__version__,
                      "torchvision": torchvision.__version__,
                      "cuda_available": bool(torch.cuda.is_available()),
                      "forward_shape": list(out.shape)}))
except Exception as exc:
    print(json.dumps({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}))
"""


def verify_stack(
    python_executable: str | None = None,
    *,
    timeout: int = 300,
) -> Dict[str, Any]:
    """Build ResNet-50 (weights=None) in a clean subprocess. Downloads nothing.

    Returns ``{"ok", "returncode", "detail", "stderr_tail"}`` where ``detail``
    holds the parsed JSON (or an error string).
    """
    python_executable = python_executable or sys.executable
    try:
        proc = subprocess.run(
            [python_executable, "-c", _VERIFY_STACK_CODE],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": None, "detail": f"stack self-check timed out after {timeout}s",
                "stderr_tail": ""}
    parsed: Any = None
    last = ""
    for line in reversed((proc.stdout or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            last = line
            break
    if last:
        try:
            parsed = json.loads(last)
        except json.JSONDecodeError:
            parsed = None
    detail = parsed if parsed is not None else (
        (proc.stdout or "").strip()[-500:] or (proc.stderr or "").strip()[-500:] or "no output"
    )
    return {
        "ok": bool(parsed and parsed.get("ok")),
        "returncode": proc.returncode,
        "detail": detail,
        "stderr_tail": (proc.stderr or "").strip()[-500:],
    }


# ---------------------------------------------------------------------------
# Smoke test execution + gate logic
# ---------------------------------------------------------------------------

def parse_pytest_summary(output: str) -> Dict[str, Any]:
    """Extract counters and the first FAILED test from pytest output."""
    tail = output[-6000:]
    summary: Dict[str, Any] = {"raw": None}
    for line in reversed(tail.splitlines()):
        if re.search(r"\d+ (passed|failed|skipped|error)", line):
            summary["raw"] = line.strip()
            break
    for key in ("passed", "failed", "skipped", "errors"):
        word = "error" if key == "errors" else key
        m = re.search(rf"(\d+) {word}(?:s)?\b", tail)
        summary[key] = int(m.group(1)) if m else None
    m = re.search(r"FAILED (\S+)", tail)
    summary["first_failed"] = m.group(1) if m else None
    return summary


def classify_failure(text: str) -> Dict[str, str]:
    """Heuristic family for a smoke-test failure: environment | data | code | unknown.

    Returns ``{"category", "evidence"}``. The evidence is the first line that
    matched, so the user can judge the classification themselves.
    """
    rules = (
        ("environment", (
            r"No module named ['\"]",
            r"ModuleNotFoundError",
            r"\bImportError\b",
            r"torch\.cuda",
            r"CUDA error",
            r"out of memory",
        )),
        ("data", (
            r"FileNotFoundError",
            r"does not exist",
            r"DecodeError",
            r"verification failed",
            r"decode/convert failed",
            r"No such file or directory",
        )),
        ("code", (
            r"AssertionError",
            r"\bSyntaxError\b",
            r"\bNameError\b",
            r"\bAttributeError\b",
            r"\bTypeError\b",
        )),
    )
    lines = text.splitlines()
    for category, patterns in rules:
        for pat in patterns:
            for line in lines:
                if re.search(pat, line):
                    return {"category": category, "evidence": line.strip()[:400]}
    failures = [l.strip() for l in lines if l.strip().startswith("FAILED ")]
    evidence = failures[0] if failures else "\n".join(
        [l for l in lines if l.strip()][-2:]
    )[:400]
    return {"category": "unknown", "evidence": evidence}


def run_smoke_test(
    project_root: Path | str,
    *,
    env_overrides: Dict[str, Any] | None = None,
    timeout: int = 1800,
    python_executable: str | None = None,
) -> Dict[str, Any]:
    """Run ``python -m pytest colab/smoke_test.py -q`` and capture everything.

    Status values:

    - ``"failed"``            - non-zero exit (with a failure_family heuristic);
    - ``"passed_with_skips"`` - exit 0 but some checks SKIPPED (the notebook
      treats these as *not* fully verified unless the user opts in);
    - ``"passed"``            - exit 0, zero skips - real evidence.
    """
    python_executable = python_executable or sys.executable
    project_root = Path(project_root)
    cmd = [python_executable, "-m", "pytest", "colab/smoke_test.py", "-q"]
    env = dict(os.environ)
    for key, value in (env_overrides or {}).items():
        if value:
            env[str(key)] = str(value)
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(project_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "command": cmd,
            "exit_code": None,
            "status": "failed",
            "summary": {},
            "failure_class": {"category": "environment",
                              "evidence": f"smoke test timed out after {timeout}s"},
            "tail": f"timeout after {timeout}s",
            "duration_seconds": round(time.time() - started, 2),
        }
    except OSError as exc:  # interpreter unusable, pytest unavailable, etc.
        return {
            "command": cmd,
            "exit_code": None,
            "status": "failed",
            "summary": {},
            "failure_class": {"category": "environment", "evidence": f"{type(exc).__name__}: {exc}"},
            "tail": str(exc),
            "duration_seconds": round(time.time() - started, 2),
        }

    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    summary = parse_pytest_summary(combined)
    if proc.returncode != 0:
        status = "failed"
        failure_class = classify_failure(combined)
    elif (summary.get("failed") or 0) or (summary.get("errors") or 0):
        status = "failed"
        failure_class = classify_failure(combined)
    elif (summary.get("skipped") or 0) > 0:
        status = "passed_with_skips"
        failure_class = {"category": "none", "evidence": ""}
    else:
        status = "passed"
        failure_class = {"category": "none", "evidence": ""}
    tail_lines = [l for l in combined.strip().splitlines() if l.strip()]
    return {
        "command": cmd,
        "exit_code": proc.returncode,
        "status": status,
        "summary": summary,
        "failure_class": failure_class,
        "tail": "\n".join(tail_lines[-40:]),
        "duration_seconds": round(time.time() - started, 2),
    }


def training_preconditions(
    *,
    smoke_status: Optional[str] = None,
    allow_skipped_smoke: bool = False,
    artifacts_ok: bool = False,
    data_root_ok: bool = False,
    model_constructed: bool = False,
    weights_ok: bool = False,
) -> Tuple[bool, List[str]]:
    """Compute the gate for any expensive run; returns ``(allowed, reasons)``.

    ``weights_ok`` should already be ``cached or allow_weight_download`` by the
    caller - this function never downloads anything itself.
    """
    reasons: List[str] = []
    if not data_root_ok:
        reasons.append(
            "Dataset image root unresolved - SECTION 2 lists the exact paths it checked."
        )
    if not artifacts_ok:
        reasons.append("Artifact validation did not pass (SECTION 4).")
    allowed_smoke = {"passed"} | ({"passed_with_skips"} if allow_skipped_smoke else set())
    if smoke_status not in allowed_smoke:
        if smoke_status is None:
            reasons.append("Smoke test has not run yet (SECTION 5).")
        elif smoke_status == "failed":
            reasons.append("Smoke test failed - fix the reported cause in SECTION 5.")
        elif smoke_status == "passed_with_skips":
            reasons.append(
                "Smoke test passed but SKIPPED image checks, so the corpus was not verified; "
                "fix the paths, or deliberately set ALLOW_SKIPPED_SMOKE=True in SECTION 0."
            )
        else:
            reasons.append(f"Unknown smoke status {smoke_status!r}.")
    if not model_constructed:
        reasons.append("Model construction check has not passed (SECTION 6).")
    if not weights_ok:
        reasons.append(
            "ImageNet ResNet-50 weights are not cached and ALLOW_WEIGHT_DOWNLOAD=False; "
            "the run would refuse instead of downloading. Either keep it that way and "
            "pre-seed the cache, or deliberately set ALLOW_WEIGHT_DOWNLOAD=True in SECTION 0."
        )
    return (not reasons, reasons)


# ---------------------------------------------------------------------------
# Canary split (derived, clearly labelled, never part of the audited artifact)
# ---------------------------------------------------------------------------

def build_canary_split(
    source_csv: Path | str,
    out_csv: Path | str,
    *,
    train_per_class: int = 8,
    val_per_class: int = 4,
    test_per_class: int = 0,
) -> Dict[str, Any]:
    """Write a tiny, deterministic subset of the pinned split for a canary run.

    Rules that keep this safe:

    - rows keep their original partition and label; nothing is relabelled,
      shuffled across partitions, or written anywhere near the audited split;
    - selection is the *first* N rows per (partition, class) in file order, so
      the result is byte-for-byte reproducible;
    - ``test_per_class=0`` (default and the notebook's choice) means the
      canary never even reads a test row;
    - group integrity cannot regress: a subset of a leakage-free assignment is
      still leakage-free.

    The output lives in a separate folder (``..._canary_split``) chosen by the
    notebook, never inside ``provisional_primary_split/``.
    Returns {"out_path", "rows", "counts", "train_classes"}.
    """
    source_csv, out_csv = Path(source_csv), Path(out_csv)
    with source_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    if not rows:
        raise ValueError(f"source split CSV has no rows: {source_csv}")

    limits = {"train": max(0, int(train_per_class)),
              "validation": max(0, int(val_per_class)),
              "test": max(0, int(test_per_class))}
    quota: Dict[Tuple[str, str], int] = {}
    keep: List[Dict[str, str]] = []
    for row in rows:
        part = row.get("partition", "")
        if part not in limits:
            continue
        key = (part, row.get("primary_class", ""))
        used = quota.get(key, 0)
        if used < limits[part]:
            quota[key] = used + 1
            keep.append(row)

    train_rows = [r for r in keep if r.get("partition") == "train"]
    train_classes = sorted({r.get("primary_class", "") for r in train_rows})
    if train_classes != sorted(EXPECTED_CLASSES):
        raise ValueError(
            "canary split would leave classes out of the training partition: "
            f"expected {sorted(EXPECTED_CLASSES)}, got {train_classes}. "
            "Raise CANARY_TRAIN_PER_CLASS (or check the source split)."
        )
    if not any(r.get("partition") == "validation" for r in keep):
        raise ValueError("canary split has no validation rows; lower the sample or check the split.")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in keep:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    counts: Dict[str, int] = {}
    for r in keep:
        counts[r.get("partition", "")] = counts.get(r.get("partition", ""), 0) + 1
    return {
        "out_path": str(out_csv),
        "rows": len(keep),
        "counts": counts,
        "train_classes": train_classes,
        "source_csv": str(source_csv),
        "note": "Derived, deterministic subset of the pinned split; not an audited artifact.",
    }


def validate_canary_split(
    split_dir: Path | str,
    *,
    dataset_root: Path | str | None = None,
    expected_train: Optional[int] = 50,
    expected_validation: Optional[int] = 25,
    expected_test: Optional[int] = 0,
    train_per_class: Optional[int] = None,
    val_per_class: Optional[int] = None,
    expected_classes: Optional[Sequence[str]] = None,
    decode_per_class: int = 1,
) -> Dict[str, Any]:
    """Validate a *derived canary* split. Separate from :func:`validate_split_artifact`.

    ``validate_split_artifact`` enforces the audited full split: exactly 14,201
    rows, and all five bands present in **all three** partitions including the
    held-out test partition. A canary deliberately has ``test_per_class=0``, so
    the full validator fails on it for reasons that are correct there but
    irrelevant here. This function is therefore a *separate* contract, not a
    relaxation: the full validator is untouched and still rejects a canary that
    is passed off as the real split (pinned by the tests).

    What it checks, each reported individually:

    1. the canary CSV is exactly ``split_dir / "provisional_primary_split.csv"``
       - the one path both this module and ``colab_helpers.load_split_csv``
       (used by ``run_training``) read. A canary written anywhere else would
       validate an empty or unrelated folder, so this fails loudly instead;
    2. the CSV is readable and carries the required split columns;
    3. exactly ``expected_train`` / ``expected_validation`` / ``expected_test``
       rows per partition, and no unknown partition values;
    4. exactly the five provisional bands, present in **both** the training and
       the validation partition;
    5. when ``train_per_class`` / ``val_per_class`` are given, every band has
       exactly that many rows in that partition (deterministic canary shape);
    6. no duplicated ``image_path``;
    7. each row's source folder maps through ``dataset.labels.PRIMARY_MAPPING``
       to its ``primary_class`` (no silent relabelling);
    8. no group leakage: every ``group_id`` sits in exactly one partition;
    9. every selected image path exists under ``dataset_root`` (all rows - the
       canary is tiny, so there is no sampling excuse); and
    10. ``decode_per_class`` images per band actually decode to RGB through
       ``dataset.decoding.load_rgb``.

    Image checks **fail closed**: an unresolved ``dataset_root`` is a failure,
    never a silent skip, because a canary whose images cannot be read proves
    nothing.

    Returns ``{"ok", "checks", "rows", "partition_counts", "split_csv",
    "dataset_root"}``.
    """
    checks: List[Dict[str, Any]] = []
    split_dir = Path(split_dir)
    csv_path = split_dir / SPLIT_CSV_NAME
    expected_classes = list(EXPECTED_CLASSES if expected_classes is None else expected_classes)

    def _add(name: str, ok: bool, detail: str, *, status: Optional[str] = None) -> None:
        status = status or ("pass" if ok else "fail")
        checks.append({"name": name, "status": status, "ok": status in ("pass", "skip"), "detail": detail})

    def _finish(rows: List[Dict[str, str]]) -> Dict[str, Any]:
        partition_counts: Dict[str, int] = {}
        for r in rows:
            key = r.get("partition") or "<missing>"
            partition_counts[key] = partition_counts.get(key, 0) + 1
        return {
            "ok": bool(checks) and all(c["ok"] for c in checks),
            "checks": checks,
            "rows": len(rows),
            "partition_counts": partition_counts,
            "split_csv": str(csv_path),
            "dataset_root": str(dataset_root) if dataset_root is not None else None,
        }

    # 1. exact path -------------------------------------------------------
    if not csv_path.is_file():
        _add(
            "canary_csv_at_expected_path",
            False,
            f"missing {csv_path}. The canary CSV must be <split_dir>/{SPLIT_CSV_NAME}, "
            "because both validate_split_artifact() and colab_helpers.load_split_csv() "
            "read that exact path.",
        )
        return _finish([])
    _add("canary_csv_at_expected_path", True, str(csv_path))

    # 2. readable ---------------------------------------------------------
    try:
        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            header = list(reader.fieldnames or [])
            rows = [dict(r) for r in reader]
    except Exception as exc:
        _add("canary_csv_readable", False, f"{type(exc).__name__}: {exc}")
        return _finish([])
    missing_cols = [c for c in REQUIRED_SPLIT_COLUMNS if c not in header]
    if missing_cols:
        _add("canary_csv_readable", False, f"missing columns {missing_cols}; header is {header}")
        return _finish([])
    _add("canary_csv_readable", True, f"{len(rows)} data rows, columns {header}")

    # 3. partition counts -------------------------------------------------
    expected_counts = {
        "train": expected_train,
        "validation": expected_validation,
        "test": expected_test,
    }
    actual_counts: Dict[str, int] = {}
    for r in rows:
        key = r.get("partition") or "<missing>"
        actual_counts[key] = actual_counts.get(key, 0) + 1

    unknown_partitions = sorted(set(actual_counts) - set(expected_counts))
    _add(
        "canary_partitions_known",
        not unknown_partitions,
        "only train/validation/test rows" if not unknown_partitions
        else f"unexpected partition values: {unknown_partitions}",
    )

    expected_total = sum(v for v in expected_counts.values() if v is not None)
    _add(
        "canary_total_rows",
        len(rows) == expected_total,
        f"{len(rows)} rows (expected {expected_total})",
    )

    for part, expected_n in expected_counts.items():
        if expected_n is None:
            continue
        got = actual_counts.get(part, 0)
        _add(f"canary_{part}_rows", got == expected_n, f"{part}: {got} rows (expected {expected_n})")

    # 4. five bands in train and validation -------------------------------
    classes = sorted({r.get("primary_class", "") for r in rows})
    expected_sorted = sorted(expected_classes)
    _add(
        "canary_classes_are_the_five_bands",
        classes == expected_sorted,
        "exactly the five provisional bands" if classes == expected_sorted
        else f"expected {expected_sorted}, found {classes}",
    )
    for part in ("train", "validation"):
        present = sorted({r.get("primary_class", "") for r in rows if r.get("partition") == part})
        _add(
            f"canary_all_bands_in_{part}",
            present == expected_sorted,
            f"{part} covers all five bands" if present == expected_sorted
            else f"{part} bands {present} != {expected_sorted}",
        )

    # 5. per-class shape ---------------------------------------------------
    for part, per_class in (("train", train_per_class), ("validation", val_per_class)):
        if per_class is None:
            continue
        counts = {
            cls: sum(1 for r in rows if r.get("partition") == part and r.get("primary_class") == cls)
            for cls in expected_sorted
        }
        ok = all(v == per_class for v in counts.values())
        _add(
            f"canary_{part}_per_class",
            ok,
            f"every band has exactly {per_class} {part} rows" if ok
            else f"{part} per-class counts: {counts}",
        )

    # 6. duplicates --------------------------------------------------------
    seen: Dict[str, int] = {}
    dups: List[str] = []
    for r in rows:
        p = r.get("image_path", "")
        if p in seen:
            dups.append(p)
        seen[p] = 1
    _add(
        "canary_image_paths_unique",
        not dups,
        "no duplicated image_path" if not dups else f"{len(dups)} duplicated paths, first: {dups[:3]}",
    )

    # 7. path prefix <-> label --------------------------------------------
    mapping = _primary_mapping()
    if mapping is None:
        _add(
            "canary_path_prefix_matches_label",
            False,
            "dataset.labels unavailable (repo root missing from sys.path); cannot verify labels",
        )
    else:
        unknown: List[str] = []
        mismatched: List[str] = []
        for r in rows:
            prefix = r.get("image_path", "").split("/")[0]
            expected_label = mapping.get(prefix)
            if expected_label is None:
                unknown.append(prefix)
            elif r.get("primary_class") != expected_label:
                mismatched.append(f"{r.get('image_path')} -> {r.get('primary_class')}")
        _add(
            "canary_path_prefix_matches_label",
            not unknown and not mismatched,
            "every canary path's source folder maps to its primary_class"
            if not unknown and not mismatched
            else f"unknown prefixes {sorted(set(unknown))[:5]}, mismatches {mismatched[:5]}",
        )

    # 8. group leakage -----------------------------------------------------
    group_parts: Dict[str, set] = {}
    for r in rows:
        group_parts.setdefault(r.get("group_id", ""), set()).add(r.get("partition", ""))
    leaks = sorted(g for g, parts in group_parts.items() if len(parts) > 1)
    _add(
        "canary_no_group_leakage",
        not leaks,
        f"{len(group_parts)} groups, each confined to one partition" if not leaks
        else f"{len(leaks)} groups span partitions, first: "
             f"{[(g, sorted(group_parts[g])) for g in leaks[:3]]}",
    )

    # 9. image paths -------------------------------------------------------
    if dataset_root is None:
        _add(
            "canary_image_paths_exist",
            False,
            "no dataset_root supplied, so the canary image paths cannot be verified "
            "(failing closed instead of skipping)",
        )
        _add(
            "canary_images_decode",
            False,
            "no dataset_root supplied, so no canary image can be decoded (failing closed)",
        )
        return _finish(rows)

    root_path = Path(dataset_root)
    if not root_path.is_dir():
        _add("canary_image_paths_exist", False, f"dataset_root is not a directory: {root_path}")
        _add("canary_images_decode", False, f"dataset_root is not a directory: {root_path}")
        return _finish(rows)

    missing = [
        str(root_path / r["image_path"]) for r in rows if not (root_path / r["image_path"]).is_file()
    ]
    _add(
        "canary_image_paths_exist",
        not missing,
        f"all {len(rows)} canary image paths exist under {root_path}" if not missing
        else f"{len(missing)} missing under {root_path}, first: {missing[:5]}",
    )

    # 10. decode one sample per band ---------------------------------------
    try:
        from dataset.decoding import load_rgb  # existing, already-tested decoder
    except Exception as exc:
        _add("canary_images_decode", False, f"dataset.decoding unavailable: {type(exc).__name__}: {exc}")
        return _finish(rows)

    failures: List[str] = []
    decoded: List[str] = []
    for cls in expected_sorted:
        picks = [r for r in rows if r.get("primary_class") == cls and r.get("partition") == "validation"]
        picks += [r for r in rows if r.get("primary_class") == cls and r.get("partition") == "train"]
        for row in picks[: max(0, int(decode_per_class))]:
            p = root_path / row["image_path"]
            try:
                im = load_rgb(p)
                if im.mode != "RGB":
                    failures.append(f"{p}: mode {im.mode}")
                else:
                    decoded.append(f"{cls} <- {row['image_path']} {im.size}")
            except Exception as exc:
                failures.append(f"{p}: {type(exc).__name__}: {exc}")

    want = max(0, int(decode_per_class)) * len(expected_sorted)
    _add(
        "canary_images_decode",
        not failures and len(decoded) == want,
        f"decoded {len(decoded)} canary image(s) across the five bands via "
        "dataset.decoding.load_rgb"
        if not failures and len(decoded) == want
        else (f"decode failures: {failures[:3]}" if failures
              else f"decoded {len(decoded)} of {want} expected sample(s)"),
    )

    return _finish(rows)


# ---------------------------------------------------------------------------
# Results verification + policy disclosure
# ---------------------------------------------------------------------------

def verify_output_run(
    output_dir: Path | str,
    *,
    expect_final_evaluation: bool = False,
) -> Dict[str, Any]:
    """Check that an output directory holds a *completed* run, file by file.

    Items marked ``required=False`` (e.g. the final evaluation, when it was not
    requested) do not affect ``ok``.
    """
    output_dir = Path(output_dir)
    manifest_path = output_dir / "run_manifest.json"
    items: List[Dict[str, Any]] = []

    def _add(name: str, ok: bool, detail: str, *, required: bool = True) -> None:
        items.append({"name": name, "ok": ok, "detail": detail, "required": required})

    if not output_dir.is_dir():
        _add("output_dir_exists", False, f"directory does not exist yet: {output_dir}")
        return {"ok": False, "items": items, "output_dir": str(output_dir)}

    _add("output_dir_exists", True, str(output_dir))
    _add("run_meta.json", (output_dir / "run_meta.json").is_file(),
         "configuration snapshot present" if (output_dir / "run_meta.json").is_file()
         else "absent (training has not started here)")

    status = None
    if manifest_path.is_file():
        try:
            status = json.loads(manifest_path.read_text(encoding="utf-8")).get("status")
        except Exception as exc:
            status = f"unreadable ({type(exc).__name__}: {exc})"
    _add(
        "run_manifest_status_completed",
        status == "completed",
        f"status={status!r}" + ("" if status == "completed" else
                                " - an incomplete or in_progress run is not a result"),
    )
    _add("run_summary.json", (output_dir / "run_summary.json").is_file(),
         "per-epoch history present" if (output_dir / "run_summary.json").is_file()
         else "absent (written only when a run completes)")

    best = output_dir / "best_checkpoint" / "model.pt"
    _add("best_checkpoint/model.pt", best.is_file(),
         "validation-selected checkpoint present" if best.is_file() else "absent")

    ckpt_root = output_dir / "checkpoints"
    epochs = sorted(p.name for p in ckpt_root.glob("epoch_*")) if ckpt_root.is_dir() else []
    _add("per_epoch_checkpoints", bool(epochs),
         f"{len(epochs)} epoch folder(s)" if epochs else "none written")

    marker = output_dir / "TRAINING_INCOMPLETE.txt"
    _add(
        "no_TRAINING_INCOMPLETE_marker",
        not marker.exists(),
        "no incomplete marker" if not marker.exists()
        else "TRAINING_INCOMPLETE.txt present - the run was interrupted or failed; "
             "resume is not implemented, restart in a fresh output directory",
    )

    eval_json = output_dir / "final_test_evaluation" / "final_test_metrics.json"
    _add(
        "final_test_metrics.json",
        eval_json.is_file(),
        "present" if eval_json.is_file()
        else ("absent - required for this check" if expect_final_evaluation
              else "not written yet (SECTION 9 not run or not requested)"),
        required=expect_final_evaluation,
    )

    ok = all(i["ok"] for i in items if i["required"])
    return {"ok": ok, "items": items, "output_dir": str(output_dir)}


def load_label_policy_excerpt(project_root: Path | str, *, max_chars: int = 1200) -> str:
    """Return the built-in provisional disclosure plus a file excerpt.

    Never raises: if the policy document is missing from the upload, the
    built-in disclosure still shows and the exact missing path is named.
    """
    policy_path = Path(project_root) / LABEL_POLICY_RELPATH
    if policy_path.is_file():
        try:
            body = policy_path.read_text(encoding="utf-8", errors="replace")[:max_chars]
        except OSError as exc:
            body = f"(could not read {policy_path}: {exc})"
        return f"{PROVISIONAL_DISCLOSURE}\n\n{body}"
    return (
        f"{PROVISIONAL_DISCLOSURE}\n\n"
        f"(NOTE: expected policy file not found at {policy_path}; upload the full project "
        "including audit_output/ to read the source document.)"
    )