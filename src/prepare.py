"""Build the unified five-class manifest from Mendeley + MMRDR + APTOS + DDR.

This is the stage that turns independent datasets into **one** classification
dataset for **one** ResNet-50, while keeping every source traceable.

What it does
------------
1. Scans both classification sources and reads their real labels.
2. Maps each original label to the five-class taxonomy through the audited,
   configuration-declared rules in :mod:`src.taxonomy`. Unknown labels stop the
   pipeline; nothing is guessed.
3. Verifies every image is readable and records dimensions/channels.
4. Computes an exact SHA-256 and two perceptual hashes (dHash, aHash) in a single
   pass, cached on disk so re-runs are cheap.
5. Builds atomic split groups from exact duplicates, near duplicates and patient
   keys, then produces a group-aware stratified train/validation/test split.
6. Refuses to split a group across partitions and quarantines groups whose members
   carry different labels (a genuine label conflict, mirroring the project's
   existing conflict policy).
7. Writes the manifests plus the label-mapping, duplicate and split audits.

Honesty guarantees
------------------
* ``all_images.csv`` keeps *every* discovered image, including excluded ones, with
  ``included`` and ``exclusion_reason`` columns. Nothing is silently dropped.
* Patient IDs are recorded only when they can be derived from a documented
  filename convention; otherwise the column is empty and the limitation is
  reported in ``dataset_summary.json``.
* If a source is unavailable the manifest is built from the sources that are and
  marked ``partial`` when too few remain — the classifier is then *not* described
  as a combined model.

Dataset-composition modes
-------------------------
``combined``  every validated source feeds the internal train/validation/test
             split, except partitions the source authors held out themselves
             (MMRDR ``ts``, DDR official ``test``), which stay external.
``external``  only ``run.development_sources`` (initially Mendeley + MMRDR) feed
             the internal split; APTOS and DDR are kept entirely as the external
             test partition and can never influence training or model selection.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .config import DATASET_MODES, Config
from .dataset_downloaders.base import MANIFEST_FILENAME, DownloadManifest, find_dirs_named
from .dataset_downloaders.ddr import locate_grading_dir, read_label_csv
from .taxonomy import LabelMapper, MappingDecision, cross_check_mendeley_policy

#: Classification sources this stage knows how to read. Each has its own loader
#: below; there is deliberately no generic "copy everything into one folder" path.
CLASSIFICATION_SOURCES = ("mendeley", "mmrdr", "aptos", "ddr")

MANIFEST_COLUMNS = (
    "image_path",
    "rel_path",
    "source_dataset",
    "original_label",
    "mapped_label",
    "class_name",
    "patient_id",
    "group_id",
    "image_hash",
    "dhash",
    "ahash",
    "width",
    "height",
    "mode",
    "split",
    "manifest_version",
    "split_seed",
    "provenance",
    "included",
    "exclusion_reason",
    "duplicate_of",
)

SPLIT_FILES = {
    "train": "train_manifest.csv",
    "validation": "validation_manifest.csv",
    "test": "test_manifest.csv",
    "external_test": "external_test_manifest.csv",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

#: Exactly the conventions used by the pre-existing project audit
#: (``audit_output/pretraining_audit/build_registers.py``), reused rather than
#: reinvented so patient grouping stays consistent with earlier work.
_RE_PREFIXED = re.compile(r"^(.+?)(\d+)_(left|right)_(\d+)_(\d+)\.(?:jpeg|jpg|png)$", re.IGNORECASE)
_RE_PLAIN = re.compile(r"^(\d+)(?:_(left|right))?\.(?:jpeg|jpg|png)$", re.IGNORECASE)


class PrepareError(RuntimeError):
    """Raised when the manifest cannot be built honestly."""


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class ImageRecord:
    image_path: Path
    rel_path: str
    source_dataset: str
    original_label: str
    mapped_label: Optional[int]
    class_name: str
    patient_id: str = ""
    group_key: str = ""
    sha256: str = ""
    dhash: str = ""
    ahash: str = ""
    width: int = 0
    height: int = 0
    mode: str = ""
    split: str = ""
    included: bool = True
    exclusion_reason: str = ""
    duplicate_of: str = ""
    provenance: str = ""
    official_partition: str = ""

    def as_row(self, manifest_version: str, seed: int) -> Dict[str, Any]:
        return {
            "image_path": str(self.image_path),
            "rel_path": self.rel_path,
            "source_dataset": self.source_dataset,
            "original_label": self.original_label,
            "mapped_label": "" if self.mapped_label is None else int(self.mapped_label),
            "class_name": self.class_name,
            "patient_id": self.patient_id,
            "group_id": self.group_key,
            "image_hash": self.sha256,
            "dhash": self.dhash,
            "ahash": self.ahash,
            "width": self.width,
            "height": self.height,
            "mode": self.mode,
            "split": self.split,
            "manifest_version": manifest_version,
            "split_seed": seed,
            "provenance": self.provenance,
            "included": "yes" if self.included else "no",
            "exclusion_reason": self.exclusion_reason,
            "duplicate_of": self.duplicate_of,
        }


# ---------------------------------------------------------------------------
# Image inspection + hashing (single pass, cached)
# ---------------------------------------------------------------------------

def _dhash(gray_small: Any, size: int = 8) -> str:
    import numpy as np

    bits = gray_small[:, 1:] > gray_small[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bool(bit))
    return f"{value:0{size * size // 4}x}"


def _ahash(gray_small: Any, size: int = 8) -> str:
    import numpy as np

    mean = gray_small.mean()
    bits = gray_small > mean
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bool(bit))
    return f"{value:0{size * size // 4}x}"


def _hamming(a: str, b: str) -> int:
    if not a or not b:
        return 64
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except ValueError:
        return 64


class ImageIndex:
    """Computes and caches per-image hashes and header facts."""

    def __init__(self, cache_path: Path, logger=None) -> None:
        self.cache_path = Path(cache_path)
        self.logger = logger
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._dirty = False
        if self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {}

    def flush(self) -> None:
        if self._dirty:
            self.cache_path.write_text(json.dumps(self._cache), encoding="utf-8")
            self._dirty = False

    def inspect(self, path: Path) -> Dict[str, Any]:
        """Return ``{ok, sha256, dhash, ahash, width, height, mode, error}``."""
        import numpy as np
        from PIL import Image

        path = Path(path)
        try:
            stat = path.stat()
            key = f"{path}|{stat.st_size}|{int(stat.st_mtime)}"
        except OSError as exc:
            return {"ok": False, "error": f"stat failed: {exc}"}

        cached = self._cache.get(key)
        if cached is not None:
            return cached

        info: Dict[str, Any] = {"ok": False, "error": ""}
        try:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            info["sha256"] = digest.hexdigest()

            with Image.open(path) as image:
                info["width"] = int(image.width)
                info["height"] = int(image.height)
                info["mode"] = str(image.mode)
                gray = image.convert("L").resize((9, 8), Image.BILINEAR)
                small = np.asarray(gray, dtype=np.int16)
                info["dhash"] = _dhash(small)
                small8 = np.asarray(
                    image.convert("L").resize((8, 8), Image.BILINEAR), dtype=np.int16
                )
                info["ahash"] = _ahash(small8)
                # Force a full decode so truncated files are not treated as valid.
                image.load()
            info["ok"] = True
        except Exception as exc:
            info["ok"] = False
            info["error"] = f"{type(exc).__name__}: {exc}"

        self._cache[key] = info
        self._dirty = True
        return info


# ---------------------------------------------------------------------------
# Patient / group keys
# ---------------------------------------------------------------------------

def mendeley_patient_id(filename: str, folder: str = "") -> Tuple[str, str]:
    """Return ``(patient_id, group_key)`` using the project's documented rules.

    The group key is **folder-scoped** (``prefixed:{folder}|ID={n}``), exactly as
    in ``audit_output/pretraining_audit/build_registers.py``. The folder has to be
    part of the key: identifiers such as ``0_right.jpeg`` recur in every class
    folder, so an unscoped key would merge unrelated patients across all 13
    classes and quarantine the whole corpus.

    ``("", "")`` means no patient linkage could be derived; the image is then
    grouped at image level only, never assumed to be patient-independent.
    """
    name = Path(filename).name
    match = _RE_PREFIXED.match(name)
    if match:
        pid = match.group(2)
        return pid, f"prefixed:{folder}|ID={pid}"
    match = _RE_PLAIN.match(name)
    if match:
        pid = match.group(1)
        return pid, f"plain:{folder}|ID={pid}"
    return "", ""


def mmrdr_official_partition(filename: str) -> str:
    """Return ``train``/``test`` from MMRDR's official filename prefix."""
    stem = Path(filename).stem.lower()
    if stem.startswith("tr"):
        return "train"
    if stem.startswith("ts"):
        return "test"
    return ""


# ---------------------------------------------------------------------------
# Source scanning
# ---------------------------------------------------------------------------

def _iter_images(directory: Path) -> Iterable[Path]:
    for path in sorted(Path(directory).rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def scan_mendeley(
    root: Path,
    mapper: LabelMapper,
    index: ImageIndex,
    cfg: Config,
    logger,
    *,
    provenance: str,
) -> Tuple[List[ImageRecord], List[Dict[str, Any]]]:
    """Scan the 13 class folders of the Mendeley corpus."""
    expected = [str(x) for x in (cfg.get("datasets.mendeley.expected_class_folders") or [])]
    records: List[ImageRecord] = []
    excluded: List[Dict[str, Any]] = []
    unknown: List[str] = []

    present = [name for name in expected if (root / name).is_dir()]
    extra_dirs = [
        child.name
        for child in sorted(root.iterdir())
        if child.is_dir() and child.name not in expected
    ] if root.is_dir() else []
    if extra_dirs:
        logger.warning("Mendeley root contains unexpected folders: %s", extra_dirs[:10])

    for folder in present:
        folder_path = root / folder
        try:
            decision = mapper.resolve("mendeley", folder)
        except Exception as exc:
            unknown.append(f"{folder}: {exc}")
            continue

        for image_path in _iter_images(folder_path):
            rel = str(image_path.relative_to(root)).replace("\\", "/")
            info = index.inspect(image_path)
            patient_id, group = mendeley_patient_id(image_path.name, folder)
            record = ImageRecord(
                image_path=image_path.resolve(),
                rel_path=rel,
                source_dataset="mendeley",
                original_label=folder,
                mapped_label=decision.mapped_label,
                class_name="" if decision.mapped_label is None else mapper.class_names[decision.mapped_label],
                patient_id=patient_id,
                group_key=group,
                sha256=info.get("sha256", ""),
                dhash=info.get("dhash", ""),
                ahash=info.get("ahash", ""),
                width=int(info.get("width") or 0),
                height=int(info.get("height") or 0),
                mode=str(info.get("mode") or ""),
                provenance=provenance,
            )
            mapper.record(decision)
            if not info.get("ok"):
                record.included = False
                record.exclusion_reason = f"unreadable_image: {info.get('error', 'unknown')}"
                excluded.append({"image_path": rel, "reason": record.exclusion_reason})
            elif decision.mapped_label is None:
                record.included = False
                record.exclusion_reason = decision.exclusion_reason or "excluded_by_label_policy"
                excluded.append({"image_path": rel, "reason": record.exclusion_reason})
            records.append(record)

    missing = sorted(set(expected) - set(present))
    if unknown:
        raise PrepareError(
            "These Mendeley class folders have no declared label rule or exclusion:\n  "
            + "\n  ".join(unknown)
            + "\nAdd them to config/config.yaml under label_mapping.mendeley. "
            "The pipeline will not guess a DR grade."
        )
    if missing:
        logger.warning("Mendeley is missing %d expected class folder(s): %s", len(missing), missing)
    return records, excluded


def _pick_column(fieldnames: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lowered = {str(name).strip().lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def read_mmrdr_annotations(csv_path: Path, logger) -> Dict[str, Dict[str, str]]:
    """Parse ``FP.csv`` into ``{image_basename: {grade, lr, ...}}``.

    The column names are matched case-insensitively and the image column may hold
    either a bare filename or a relative path. Anything that cannot be resolved to
    an actual image name is reported, never guessed.
    """
    annotations: Dict[str, Dict[str, str]] = {}
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise PrepareError(f"MMRDR annotation file has no header: {csv_path}")
        fields = list(reader.fieldnames)
        image_col = _pick_column(
            fields, ["image", "image_name", "filename", "file", "img", "case", "name", "id"]
        )
        grade_col = _pick_column(fields, ["grade", "dr_grade", "drgrade", "label", "severity"])
        lr_col = _pick_column(fields, ["lr", "laterality", "eye", "side"])
        if image_col is None:
            image_col = fields[0]
        if grade_col is None:
            raise PrepareError(
                f"MMRDR annotation file has no grade column. Columns found: {fields}. "
                "The pipeline will not infer DR grades."
            )
        logger.info("MMRDR annotations: image column=%r, grade column=%r", image_col, grade_col)
        for row in reader:
            raw_image = str(row.get(image_col, "") or "").strip()
            if not raw_image:
                continue
            key = Path(raw_image.replace("\\", "/")).name
            if not Path(key).suffix:
                key = f"{key}.jpg"
            annotations[key.lower()] = {
                "grade": str(row.get(grade_col, "") or "").strip(),
                "lr": str(row.get(lr_col, "") or "").strip() if lr_col else "",
                "raw": raw_image,
            }
    return annotations


def scan_mmrdr(
    cfg: Config,
    mapper: LabelMapper,
    index: ImageIndex,
    logger,
) -> Tuple[List[ImageRecord], Dict[str, Any]]:
    """Scan the MMRDR CFP subset only."""
    mmrdr_dir = cfg.path("paths.mmrdr_dir")
    info: Dict[str, Any] = {"available": False}

    cfp_root = mmrdr_dir / "MMRDR-CFP"
    if not cfp_root.is_dir():
        found = find_dirs_named(mmrdr_dir, "MMRDR-CFP") if mmrdr_dir.is_dir() else []
        cfp_root = found[0] if found else None
    if cfp_root is None:
        info["reason"] = f"MMRDR-CFP directory not found under {mmrdr_dir}"
        return [], info

    img_dir = cfp_root / "img"
    csv_candidates = sorted(p for p in cfp_root.glob("*.csv") if p.name.upper().startswith("FP"))
    if not img_dir.is_dir():
        info["reason"] = f"MMRDR CFP image directory not found: {img_dir}"
        return [], info
    if not csv_candidates:
        info["reason"] = f"MMRDR CFP annotation CSV (FP.csv) not found under {cfp_root}"
        return [], info

    annotations = read_mmrdr_annotations(csv_candidates[0], logger)
    logger.info("MMRDR: %d annotation rows, %d CFP images on disk",
                len(annotations), sum(1 for _ in _iter_images(img_dir)))

    records: List[ImageRecord] = []
    missing_annotation = 0
    unknown_grade = 0
    for image_path in _iter_images(img_dir):
        rel = str(image_path.relative_to(mmrdr_dir)).replace("\\", "/")
        entry = annotations.get(image_path.name.lower())
        preface = mmrdr_official_partition(image_path.name)
        rec = ImageRecord(
            image_path=image_path.resolve(),
            rel_path=rel,
            source_dataset="mmrdr",
            original_label="",
            mapped_label=None,
            class_name="",
            provenance="figshare article 29423747 (MMRDR-CFP)",
            official_partition=preface,
        )
        if entry is None:
            missing_annotation += 1
            rec.included = False
            rec.exclusion_reason = "no_annotation_row_in_FP.csv"
            records.append(rec)
            continue
        grade = entry.get("grade", "")
        rec.original_label = f"grade={grade}"
        try:
            decision = mapper.resolve("mmrdr", grade)
        except Exception:
            unknown_grade += 1
            rec.included = False
            rec.exclusion_reason = f"unknown_mmrdr_grade: {grade!r}"
            records.append(rec)
            continue
        rec.mapped_label = decision.mapped_label
        rec.class_name = "" if decision.mapped_label is None else mapper.class_names[decision.mapped_label]
        mapper.record(decision)

        observed = index.inspect(image_path)
        rec.sha256 = observed.get("sha256", "")
        rec.dhash = observed.get("dhash", "")
        rec.ahash = observed.get("ahash", "")
        rec.width = int(observed.get("width") or 0)
        rec.height = int(observed.get("height") or 0)
        rec.mode = str(observed.get("mode") or "")
        if not observed.get("ok"):
            rec.included = False
            rec.exclusion_reason = f"unreadable_image: {observed.get('error', 'unknown')}"
        elif decision.mapped_label is None:
            rec.included = False
            rec.exclusion_reason = decision.exclusion_reason or "excluded_by_label_policy"

        # No patient identifier is published for MMRDR-CFP: group at image level
        # and rely on duplicate detection for leakage control.
        rec.patient_id = ""
        rec.group_key = f"mmrdr:{Path(image_path.name).stem.lower()}"
        records.append(rec)

    if missing_annotation:
        logger.warning("%d MMRDR CFP images have no annotation row and were excluded", missing_annotation)
    if unknown_grade:
        logger.warning("%d MMRDR CFP images had an unparseable grade and were excluded", unknown_grade)

    info.update(
        {
            "available": True,
            "cfp_root": str(cfp_root),
            "annotation_file": str(csv_candidates[0]),
            "annotation_rows": len(annotations),
            "images_on_disk": len(records),
            "missing_annotation_rows": missing_annotation,
            "unknown_grade_rows": unknown_grade,
            "patient_ids_available": False,
            "note": (
                "MMRDR-CFP publishes no patient identifier; images are grouped at image "
                "level plus duplicate unions, so patient-independent evaluation is not claimed."
            ),
        }
    )
    return records, info


def _annotation_csv_in(directory: Path, candidates: Sequence[str]) -> Optional[Path]:
    """Return the first existing annotation CSV from a list of documented names."""
    for name in candidates:
        candidate = Path(directory) / name
        if candidate.is_file():
            return candidate
    return None


def read_label_csv_table(path: Path, logger, *, source: str) -> Dict[str, str]:
    """Return ``{image_id_stem_lower: original_label}`` for a simple label CSV.

    Column names are matched against the documented ones; an unreadable file raises
    instead of being guessed at.
    """
    table: Dict[str, str] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise PrepareError(f"{source} annotation file has no header: {path}")
        fields = list(reader.fieldnames)
        image_col = _pick_column(
            fields, ["id_code", "image", "image_name", "filename", "file", "img", "name", "id"]
        )
        label_col = _pick_column(fields, ["diagnosis", "grade", "label", "severity"])
        if image_col is None or label_col is None:
            raise PrepareError(
                f"{source} annotation file {path.name} must contain an image-id column and a "
                f"diagnosis/grade column. Columns found: {fields}. The pipeline will not "
                "infer DR grades."
            )
        logger.info(
            "%s annotations: image column=%r, label column=%r", source.upper(), image_col, label_col
        )
        for row in reader:
            raw_id = str(row.get(image_col, "") or "").strip()
            if not raw_id:
                continue
            key = Path(raw_id.replace("\\", "/")).stem.lower()
            table[key] = str(row.get(label_col, "") or "").strip()
    return table


def _image_index_on_disk(root: Path) -> Dict[str, Path]:
    """Map ``stem.lower()`` -> path for every image under ``root``.

    A duplicate stem would make the image/label correspondence ambiguous, so the
    first (sorted) path wins and the caller is told how many collisions exist.
    """
    found: Dict[str, Path] = {}
    collisions = 0
    for path in _iter_images(root):
        key = path.stem.lower()
        if key in found:
            collisions += 1
            continue
        found[key] = path
    if collisions:
        logger.warning(
            "%d image filename collision(s) under %s; the first file for each name was used",
            collisions,
            root,
        )
    return found


def scan_aptos(
    cfg: Config,
    mapper: LabelMapper,
    index: ImageIndex,
    logger,
) -> Tuple[List[ImageRecord], Dict[str, Any]]:
    """Scan the APTOS 2019 labelled training images.

    APTOS publishes no official split with public labels, so no image is treated as
    an official test image here. Whether these images train the classifier or are
    held out depends entirely on the dataset-composition mode recorded downstream.
    """
    root = cfg.aptos_root()
    info: Dict[str, Any] = {"available": False, "root": str(root)}
    if not root.is_dir():
        info["reason"] = f"APTOS root not found: {root}"
        return [], info

    csv_candidates = [str(x) for x in (cfg.get("datasets.aptos.label_files") or [])]
    label_csv = _annotation_csv_in(root, csv_candidates)
    if label_csv is None:
        info["reason"] = (
            f"no APTOS label CSV ({', '.join(csv_candidates)}) found under {root}"
        )
        return [], info

    images_by_stem = _image_index_on_disk(root)
    if not images_by_stem:
        info["reason"] = f"no APTOS images found under {root}"
        return [], info

    labels = read_label_csv_table(label_csv, logger, source="aptos")
    records: List[ImageRecord] = []
    missing_labels = 0
    for stem, image_path in sorted(images_by_stem.items()):
        rel = str(image_path.relative_to(root)).replace("\\", "/")
        rec = ImageRecord(
            image_path=image_path.resolve(),
            rel_path=rel,
            source_dataset="aptos",
            original_label="",
            mapped_label=None,
            class_name="",
            provenance="APTOS 2019 Blindness Detection (Kaggle) training labels",
            official_partition="",
        )
        label = labels.get(stem)
        if label in (None, ""):
            missing_labels += 1
            rec.included = False
            rec.exclusion_reason = "no_label_row_in_APTOS_csv"
            records.append(rec)
            continue
        rec.original_label = f"diagnosis={label}"
        try:
            decision = mapper.resolve("aptos", label)
        except Exception:
            rec.included = False
            rec.exclusion_reason = f"unknown_aptos_label: {label!r}"
            records.append(rec)
            continue
        rec.mapped_label = decision.mapped_label
        rec.class_name = (
            "" if decision.mapped_label is None else mapper.class_names[decision.mapped_label]
        )
        mapper.record(decision)

        observed = index.inspect(image_path)
        rec.sha256 = observed.get("sha256", "")
        rec.dhash = observed.get("dhash", "")
        rec.ahash = observed.get("ahash", "")
        rec.width = int(observed.get("width") or 0)
        rec.height = int(observed.get("height") or 0)
        rec.mode = str(observed.get("mode") or "")
        if not observed.get("ok"):
            rec.included = False
            rec.exclusion_reason = f"unreadable_image: {observed.get('error', 'unknown')}"
        elif decision.mapped_label is None:
            rec.included = False
            rec.exclusion_reason = decision.exclusion_reason or "excluded_by_label_policy"

        # APTOS id_code identifies an image, not a patient: group at image level and
        # rely on exact/perceptual duplicate detection for leakage control.
        rec.patient_id = ""
        rec.group_key = f"aptos:{stem}"
        records.append(rec)

    info.update(
        {
            "available": True,
            "annotation_file": str(label_csv),
            "annotation_rows": len(labels),
            "images_on_disk": len(images_by_stem),
            "missing_annotation_rows": missing_labels,
            "patient_ids_available": False,
            "official_partitions": False,
            "note": (
                "APTOS publishes only a labelled training file; its competition test labels "
                "are withheld. No APTOS image is claimed to be an official test image."
            ),
        }
    )
    if missing_labels:
        logger.warning("%d APTOS image(s) had no label row and were excluded", missing_labels)
    logger.info("APTOS: %d image(s) with official labels", len(records) - missing_labels)
    return records, info


def scan_ddr(
    cfg: Config,
    mapper: LabelMapper,
    index: ImageIndex,
    logger,
) -> Tuple[List[ImageRecord], Dict[str, Any]]:
    """Scan DDR using its official ``train``/``valid``/``test`` label files.

    The official partition of every image is preserved in ``official_partition``.
    DDR grade 5 (ungradable) is excluded through the configured exclusion rule and
    is never mapped to No DR.
    """
    root = cfg.ddr_root()
    info: Dict[str, Any] = {"available": False, "root": str(root)}
    if not root.is_dir():
        info["reason"] = f"DDR root not found: {root}"
        return [], info

    grading = locate_grading_dir(root)
    if grading is None:
        info["reason"] = f"no DDR DR_grading directory found under {root}"
        return [], info

    images_by_stem = _image_index_on_disk(root)
    if not images_by_stem:
        info["reason"] = f"no DDR images found under {root}"
        return [], info

    label_tables: Dict[str, Dict[str, str]] = {}
    for partition in ("train", "valid", "test"):
        candidate = grading / f"{partition}.csv"
        if not candidate.is_file():
            info["reason"] = f"missing official DDR label file: {candidate}"
            return [], info
        try:
            label_tables[partition] = read_label_csv(candidate)
        except Exception as exc:
            info["reason"] = f"unreadable DDR label file {candidate}: {exc}"
            return [], info

    # id -> (partition, grade). Official files should be disjoint; a collision is
    # reported rather than resolved silently.
    assignments: Dict[str, Tuple[str, str]] = {}
    duplicates = 0
    for partition in ("train", "valid", "test"):
        for identifier, grade in label_tables[partition].items():
            key = Path(str(identifier).replace("\\", "/")).stem.lower()
            if key in assignments and assignments[key][0] != partition:
                duplicates += 1
                continue
            assignments[key] = (partition, str(grade).strip())
    if duplicates:
        logger.warning(
            "%d DDR id(s) appear in more than one official label file; the first was used",
            duplicates,
        )

    ungradable_grade = str(cfg.get("datasets.ddr.ungradable_grade", "5")).strip()
    records: List[ImageRecord] = []
    missing_labels = 0
    ungradable = 0
    for stem, image_path in sorted(images_by_stem.items()):
        rel = str(image_path.relative_to(root)).replace("\\", "/")
        entry = assignments.get(stem)
        rec = ImageRecord(
            image_path=image_path.resolve(),
            rel_path=rel,
            source_dataset="ddr",
            original_label="",
            mapped_label=None,
            class_name="",
            provenance="DDR (OIA-DDR) official grading labels",
            official_partition="" if entry is None else entry[0],
        )
        if entry is None:
            missing_labels += 1
            rec.included = False
            rec.exclusion_reason = "no_label_row_in_DDR_csv"
            records.append(rec)
            continue
        label = entry[1]
        rec.original_label = f"diagnosis={label}"
        try:
            decision = mapper.resolve("ddr", label)
        except Exception:
            rec.included = False
            rec.exclusion_reason = f"unknown_ddr_label: {label!r}"
            records.append(rec)
            continue
        rec.mapped_label = decision.mapped_label
        rec.class_name = (
            "" if decision.mapped_label is None else mapper.class_names[decision.mapped_label]
        )
        mapper.record(decision)

        observed = index.inspect(image_path)
        rec.sha256 = observed.get("sha256", "")
        rec.dhash = observed.get("dhash", "")
        rec.ahash = observed.get("ahash", "")
        rec.width = int(observed.get("width") or 0)
        rec.height = int(observed.get("height") or 0)
        rec.mode = str(observed.get("mode") or "")
        if not observed.get("ok"):
            rec.included = False
            rec.exclusion_reason = f"unreadable_image: {observed.get('error', 'unknown')}"
        elif decision.mapped_label is None:
            rec.included = False
            rec.exclusion_reason = decision.exclusion_reason or "excluded_by_label_policy"
            if label == ungradable_grade:
                ungradable += 1

        rec.patient_id = ""
        rec.group_key = f"ddr:{stem}"
        records.append(rec)

    info.update(
        {
            "available": True,
            "grading_dir": str(grading),
            "label_files": {name: str(grading / f"{name}.csv") for name in label_tables},
            "images_on_disk": len(images_by_stem),
            "labelled_images": len(assignments),
            "missing_annotation_rows": missing_labels,
            "ungradable_images": ungradable,
            "official_partitions": True,
            "patient_ids_available": False,
            "note": (
                "DDR publishes no patient identifier in its grading files. Official split "
                "membership is preserved, and grade 5 (ungradable) images are excluded."
            ),
        }
    )
    if missing_labels:
        logger.warning("%d DDR image(s) had no label row and were excluded", missing_labels)
    logger.info("DDR: %d labelled image(s) across %s", len(assignments), grading)
    return records, info


# ---------------------------------------------------------------------------
# Duplicate + group construction
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra > rb:
                ra, rb = rb, ra
            self.parent[rb] = ra


def build_duplicate_audit(
    records: Sequence[ImageRecord],
    cfg: Config,
    logger,
) -> Tuple[Dict[str, str], List[Dict[str, Any]], Dict[str, int], List[Tuple[str, str]]]:
    """Find exact and near duplicates.

    Returns ``(duplicate_of, audit_rows, stats, cross_label_pairs)``.

    ``duplicate_of`` only contains **same-label** links, because those are the
    links that must never straddle a partition boundary. A near-duplicate pair
    whose two members carry *different* grades is a genuine label conflict; it is
    reported separately and both members are quarantined individually.

    Deliberately not doing this would let transitive near-duplicate chaining merge
    many classes into one connected component and quarantine whole classes.
    """
    by_sha: Dict[str, List[ImageRecord]] = defaultdict(list)
    for record in records:
        if record.sha256:
            by_sha[record.sha256].append(record)

    duplicate_of: Dict[str, str] = {}
    audit_rows: List[Dict[str, Any]] = []
    cross_label_pairs: List[Tuple[str, str]] = []
    exact_groups = 0
    exact_extra_files = 0

    for sha, group in sorted(by_sha.items()):
        if len(group) < 2:
            continue
        exact_groups += 1
        exact_extra_files += len(group) - 1
        ordered = sorted(group, key=lambda r: str(r.image_path))
        canonical = ordered[0]
        for other in ordered[1:]:
            same_label = canonical.mapped_label == other.mapped_label
            duplicate_of[str(other.image_path)] = str(canonical.image_path)
            audit_rows.append(
                {
                    "kind": "exact",
                    "distance": 0,
                    "canonical": str(canonical.image_path),
                    "duplicate": str(other.image_path),
                    "canonical_source": canonical.source_dataset,
                    "duplicate_source": other.source_dataset,
                    "canonical_label": canonical.original_label,
                    "duplicate_label": other.original_label,
                    "same_label": same_label,
                }
            )
            if not same_label:
                cross_label_pairs.append((str(canonical.image_path), str(other.image_path)))

    dhash_max = int(cfg.get("split.duplicate.dhash_max_distance", 4))
    ahash_max = int(cfg.get("split.duplicate.ahash_max_distance", 6))
    near_same_label = 0
    near_cross_label = 0

    if bool(cfg.get("split.duplicate.enable_perceptual", True)):
        candidates = [r for r in records if r.dhash and r.ahash]
        # Bucket on the top byte of dHash so the comparison stays near-linear.
        buckets: Dict[str, List[ImageRecord]] = defaultdict(list)
        for record in candidates:
            buckets[record.dhash[:2]].append(record)
        seen_pairs: set[Tuple[str, str]] = set()
        for bucket_key, bucket in buckets.items():
            neighbour_keys = [bucket_key]
            for other_key in buckets:
                if other_key != bucket_key and sum(
                    1 for a, b in zip(other_key, bucket_key) if a != b
                ) <= 1:
                    neighbour_keys.append(other_key)
            pool = [r for key in neighbour_keys for r in buckets.get(key, [])]
            for first in bucket:
                for second in pool:
                    if first is second:
                        continue
                    pair = tuple(sorted((str(first.image_path), str(second.image_path))))
                    if pair in seen_pairs:
                        continue
                    if _hamming(first.dhash, second.dhash) > dhash_max:
                        continue
                    if _hamming(first.ahash, second.ahash) > ahash_max:
                        continue
                    # Cheap specificity guard: wildly different frame sizes are not
                    # the same acquisition.
                    if first.width and second.width:
                        ratios = (first.width / second.width, first.height / second.height)
                        if max(ratios) > 2.0 or min(ratios) < 0.5:
                            continue
                    seen_pairs.add(pair)
                    same_label = first.mapped_label == second.mapped_label
                    audit_rows.append(
                        {
                            "kind": "near" if same_label else "near_cross_label",
                            "distance": max(
                                _hamming(first.dhash, second.dhash),
                                _hamming(first.ahash, second.ahash),
                            ),
                            "canonical": pair[0],
                            "duplicate": pair[1],
                            "canonical_source": first.source_dataset,
                            "duplicate_source": second.source_dataset,
                            "canonical_label": first.original_label,
                            "duplicate_label": second.original_label,
                            "same_label": same_label,
                        }
                    )
                    if same_label:
                        near_same_label += 1
                        duplicate_of.setdefault(pair[1], pair[0])
                    else:
                        near_cross_label += 1
                        cross_label_pairs.append(pair)

    stats = {
        "exact_duplicate_groups": exact_groups,
        "exact_duplicate_extra_files": exact_extra_files,
        "near_duplicate_pairs_same_label": near_same_label,
        "near_duplicate_pairs_cross_label": near_cross_label,
        "cross_label_conflict_pairs": len(cross_label_pairs),
        "total_duplicate_pairs": len(audit_rows),
    }
    logger.info(
        "duplicates: %d exact group(s) (%d extra files), %d same-label near pair(s), "
        "%d cross-label conflict pair(s)",
        exact_groups, exact_extra_files, near_same_label, near_cross_label,
    )
    return duplicate_of, audit_rows, stats, cross_label_pairs


def assign_groups(
    records: Sequence[ImageRecord],
    duplicate_of: Dict[str, str],
    logger,
) -> Dict[str, Any]:
    """Assign every included record to an atomic group.

    Group membership unions: derived patient keys, exact duplicates and near
    duplicates. A group therefore can never be split across partitions.
    """
    uf = UnionFind()
    for record in records:
        key = str(record.image_path)
        uf.find(key)
        if record.group_key:
            uf.union(key, f"GROUP::{record.group_key}")
    for duplicate, canonical in duplicate_of.items():
        uf.union(duplicate, canonical)

    groups: Dict[str, List[ImageRecord]] = defaultdict(list)
    for record in records:
        groups[uf.find(str(record.image_path))].append(record)

    # A group whose members disagree on the mapped class is a label conflict:
    # the same image (or the same patient) is graded differently by two sources.
    conflicted: List[str] = []
    for root, members in groups.items():
        labels = {m.mapped_label for m in members}
        if len(labels) > 1:
            conflicted.append(root)

    return {
        "roots": {str(record.image_path): uf.find(str(record.image_path)) for record in records},
        "groups": groups,
        "conflicted_roots": set(conflicted),
        "group_count": len(groups),
    }


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def _allocate_groups(
    groups: Sequence[Tuple[str, int, Optional[int]]],
    targets: Dict[str, float],
    *,
    seed: int,
) -> Dict[str, str]:
    """Assign groups to partitions, balancing per class.

    ``groups`` is ``(root, size, mapped_label)``. Deterministic for a given seed.
    """
    import random

    rng = random.Random(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda g: -g[1])  # largest groups first: better balance

    partitions = list(targets)
    total = sum(size for _, size, _ in groups) or 1
    assigned_counts: Dict[str, Dict[Optional[int], int]] = {
        p: defaultdict(int) for p in partitions
    }
    class_totals: Dict[Optional[int], int] = defaultdict(int)
    for _, size, label in groups:
        class_totals[label] += size

    assignment: Dict[str, str] = {}
    for root, size, label in shuffled:
        best_partition = partitions[0]
        best_deficit = None
        for partition in partitions:
            target_for_class = targets[partition] * class_totals[label]
            deficit = target_for_class - assigned_counts[partition][label]
            if best_deficit is None or deficit > best_deficit:
                best_deficit = deficit
                best_partition = partition
        assignment[root] = best_partition
        assigned_counts[best_partition][label] += size
    return assignment


def _split_summary(records: Sequence[ImageRecord]) -> Dict[str, Any]:
    by_partition: Dict[str, Counter] = defaultdict(Counter)
    for record in records:
        if record.included and record.split:
            by_partition[record.split][record.class_name] += 1
    return {
        partition: {
            "total": int(sum(counter.values())),
            "by_class": {name: int(counter.get(name, 0)) for name in sorted(counter)},
        }
        for partition, counter in sorted(by_partition.items())
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})
    return target


def resolve_image_path(row: Dict[str, str], roots: Dict[str, Path]) -> Optional[Path]:
    """Resolve a manifest row to an on-disk path, preferring the stored absolute path."""
    stored = row.get("image_path") or ""
    candidate = Path(stored)
    if candidate.is_file():
        return candidate
    rel = row.get("rel_path") or ""
    root = roots.get(str(row.get("source_dataset", "")))
    if root and rel:
        alternative = Path(root) / rel
        if alternative.is_file():
            return alternative
    return None


def apply_quality_exclusions(cfg: Config, records: Sequence[ImageRecord], logger) -> int:
    """Apply a previous quality report when ``quality.exclude_flagged`` is enabled.

    Returns the number of images newly excluded.
    """
    if not bool(cfg.get("quality.exclude_flagged", False)):
        return 0
    report_path = cfg.outputs_dir("manifests") / "image_quality.csv"
    if not report_path.is_file():
        logger.info(
            "quality.exclude_flagged is enabled but no quality report exists yet; "
            "run the quality stage and re-run prepare to apply it"
        )
        return 0

    with report_path.open(newline="", encoding="utf-8-sig") as handle:
        quality = {row["image_path"]: row for row in csv.DictReader(handle)}

    excluded = 0
    for record in records:
        info = quality.get(str(record.image_path))
        if not info or not record.included:
            continue
        flags = (info.get("quality_flags") or "").strip()
        if flags:
            record.included = False
            record.exclusion_reason = f"quality_flagged: {flags}"
            excluded += 1
    return excluded


def _resolve_dataset_mode(cfg: Config, dataset_mode: Optional[str]) -> str:
    """Return the effective dataset-composition mode (``combined`` or ``external``)."""
    requested = (dataset_mode or cfg.dataset_mode or "combined").strip().lower()
    if requested not in DATASET_MODES:
        raise PrepareError(
            f"Unknown dataset mode {requested!r}; expected one of {list(DATASET_MODES)}."
        )
    return requested


def _is_external_partition(
    record: ImageRecord,
    dataset_mode: str,
    *,
    use_official: bool,
) -> bool:
    """True when this image must stay in the external-test partition.

    Two independent reasons exist, and both are enforced here so no caller can
    forget one:

    * the dataset authors held the image out themselves (MMRDR ``ts``, DDR
      official ``test``), and
    * the run is in external-validation mode, where every non-development source
      is an external test set by definition.
    """
    if (
        use_official
        and record.source_dataset in {"mmrdr", "ddr"}
        and record.official_partition == "test"
    ):
        return True
    if dataset_mode == "external" and record.source_dataset not in {"mendeley", "mmrdr"}:
        return True
    return False


def run_prepare(
    cfg: Config,
    logger,
    events=None,
    *,
    dataset_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the unified manifests and audits. Returns a summary dict."""
    output_dir = cfg.outputs_dir("manifests")
    cache_dir = cfg.path("paths.cache_dir", create=True)
    manifest_version = cfg.manifest_version
    seed = cfg.seed

    dataset_mode = _resolve_dataset_mode(cfg, dataset_mode)
    logger.info(
        "Manifest version %s | seed %d | dataset mode %s",
        manifest_version, seed, dataset_mode,
    )

    mapper = LabelMapper(cfg)
    mapper.class_names = cfg.classes
    index = ImageIndex(cache_dir / "image_index.json", logger=logger)

    policy_check = cross_check_mendeley_policy(cfg)
    if not policy_check["consistent"]:
        raise PrepareError(
            "The configured Mendeley mapping disagrees with the repository's existing "
            f"provisional policy (dataset/labels.py): {policy_check}"
        )
    logger.info("Mendeley mapping matches dataset/labels.py")

    fundus_root = cfg.fundus_root()
    if fundus_root is None:
        candidate = cfg.path("paths.mendeley_dir")
        fundus_root = candidate if candidate.is_dir() else None

    mendeley_records: List[ImageRecord] = []
    mendeley_provenance = ""
    mendeley_source_info: Dict[str, Any] = {"available": False}
    if fundus_root is not None and fundus_root.is_dir():
        download_manifest_path = output_dir / MANIFEST_FILENAME
        provenance = "local corpus"
        if download_manifest_path.exists():
            try:
                entries = json.loads(download_manifest_path.read_text(encoding="utf-8")).get("entries", [])
                for entry in reversed(entries):
                    if entry.get("source") == "mendeley" and entry.get("operation") == "local_copy_used":
                        provenance = "local pre-existing copy (Mendeley API published no files)"
                        break
                    if entry.get("source") == "mendeley" and entry.get("operation") == "download":
                        provenance = "downloaded via Mendeley public API"
                        break
            except Exception:
                pass
        mendeley_provenance = provenance
        mendeley_records, _excluded = scan_mendeley(
            fundus_root, mapper, index, cfg, logger, provenance=provenance
        )
        mendeley_source_info = {
            "available": True,
            "root": str(fundus_root),
            "images_scanned": len(mendeley_records),
            "provenance": provenance,
        }
        logger.info("Mendeley: %d images scanned from %s", len(mendeley_records), fundus_root)
    else:
        mendeley_source_info["reason"] = (
            "no Mendeley corpus found: neither datasets/mendeley nor DR_FUNDUS_ROOT contains "
            "the 13 class folders"
        )
        logger.warning(mendeley_source_info["reason"])

    mmrdr_records, mmrdr_info = scan_mmrdr(cfg, mapper, index, logger)
    if mmrdr_info.get("available"):
        logger.info("MMRDR: %d CFP images scanned", len(mmrdr_records))
    else:
        logger.warning("MMRDR unavailable: %s", mmrdr_info.get("reason"))

    aptos_records, aptos_info = scan_aptos(cfg, mapper, index, logger)
    if not aptos_info.get("available"):
        logger.warning("APTOS unavailable: %s", aptos_info.get("reason"))

    ddr_records, ddr_info = scan_ddr(cfg, mapper, index, logger)
    if not ddr_info.get("available"):
        logger.warning("DDR unavailable: %s", ddr_info.get("reason"))

    index.flush()

    all_records = mendeley_records + mmrdr_records + aptos_records + ddr_records
    if not all_records:
        raise PrepareError(
            "No images were found in any classification source. "
            "Nothing was trained because there is no data.\n"
            "Provide at least one source: Mendeley (DR_FUNDUS_ROOT or datasets/mendeley), "
            "MMRDR (datasets/mmrdr), APTOS (datasets/aptos) or DDR (datasets/ddr), then "
            "re-run."
        )

    # Quality flags recorded by a previous `quality` run can be applied here, so
    # exclusion is always an explicit, recorded and reversible decision.
    quality_excluded = apply_quality_exclusions(cfg, all_records, logger)
    if quality_excluded:
        logger.warning(
            "%d image(s) excluded because quality.exclude_flagged is enabled", quality_excluded
        )

    included = [r for r in all_records if r.included]
    if not included:
        raise PrepareError(
            "Images were found but none passed validation/label mapping. "
            f"Exclusions: {Counter(r.exclusion_reason for r in all_records).most_common(5)}"
        )

    duplicate_of, duplicate_rows, duplicate_stats, cross_label_pairs = build_duplicate_audit(
        included, cfg, logger
    )
    for record in included:
        record.duplicate_of = duplicate_of.get(str(record.image_path), "")

    # A duplicate pair whose two members carry different DR grades is a real label
    # conflict. Both members are quarantined individually (recorded, not deleted).
    conflict_paths = {path for pair in cross_label_pairs for path in pair}
    if conflict_paths:
        quarantined = 0
        for record in included:
            if str(record.image_path) in conflict_paths:
                record.included = False
                record.exclusion_reason = (
                    "cross_label_duplicate_conflict: this image is a duplicate of an image "
                    "carrying a different DR grade, so it cannot be split or graded safely"
                )
                quarantined += 1
        logger.warning(
            "%d image(s) quarantined because they duplicate an image with a different grade",
            quarantined,
        )
        included = [r for r in all_records if r.included]
        duplicate_of = {
            k: v for k, v in duplicate_of.items()
            if k not in conflict_paths and v not in conflict_paths
        }

    grouping = assign_groups(included, duplicate_of, logger)

    # Defensive: a patient/duplicate component must still never span two grades.
    conflicted = grouping["conflicted_roots"]
    if conflicted:
        quarantined = 0
        for root in conflicted:
            for member in grouping["groups"][root]:
                member.included = False
                member.exclusion_reason = (
                    "cross_label_linked_group: patient-linked images carry conflicting DR "
                    "grades and cannot be split safely"
                )
                quarantined += 1
        logger.warning("%d image(s) quarantined for conflicting labels within a group", quarantined)
        included = [r for r in all_records if r.included]
        grouping = assign_groups(included, duplicate_of, logger)

    # Deterministic split.
    use_official = bool(cfg.get("split.use_official_test_partitions", True))
    ratios = cfg.section("split")["ratios"]
    internal_ratios = {
        "train": float(ratios["train"]),
        "validation": float(ratios["validation"]),
        "test": float(ratios["test"]),
    }

    external_records = [
        r for r in included
        if _is_external_partition(r, dataset_mode, use_official=use_official)
    ]
    external_roots = {grouping["roots"][str(r.image_path)] for r in external_records}
    internal_records = [
        r for r in included if grouping["roots"][str(r.image_path)] not in external_roots
    ]
    if not internal_records:
        raise PrepareError(
            "No image is available for training.\n"
            f"Dataset mode: {dataset_mode}. External (held-out) sources found: "
            f"{sorted({r.source_dataset for r in included}) or 'none'}.\n"
            + (
                "External-validation mode trains only on "
                + ", ".join(cfg.development_sources)
                + ". Supply at least one of those genuine sources, or run "
                "--mode combined if the held-out sources are meant to train the classifier."
                if dataset_mode == "external"
                else "Supply at least one genuine classification source with usable images."
            )
        )

    per_source_groups: Dict[str, List[Tuple[str, int, Optional[int]]]] = defaultdict(list)
    seen_roots: set[str] = set()
    for record in internal_records:
        root = grouping["roots"][str(record.image_path)]
        if root in seen_roots:
            continue
        seen_roots.add(root)
        members = grouping["groups"][root]
        per_source_groups[record.source_dataset].append(
            (root, len(members), record.mapped_label)
        )

    for source, groups in per_source_groups.items():
        assignment = _allocate_groups(groups, internal_ratios, seed=seed)
        for root, partition in assignment.items():
            for member in grouping["groups"][root]:
                member.split = partition
        logger.info("split assigned for %s: %d groups", source, len(groups))

    for record in external_records:
        record.split = "external_test"

    final_included = [r for r in all_records if r.included and r.split]

    # ---- leakage assertions -------------------------------------------
    leakage_problems: List[str] = []
    partition_by_group: Dict[str, set] = defaultdict(set)
    partitions_by_hash: Dict[str, set] = defaultdict(set)
    for record in final_included:
        partition_by_group[record.group_key or str(record.image_path)].add(record.split)
        if record.sha256:
            partitions_by_hash[record.sha256].add(record.split)
    for key, partitions in partition_by_group.items():
        if len(partitions) > 1:
            leakage_problems.append(f"patient/duplicate group {key!r} spans partitions {sorted(partitions)}")
    for sha, partitions in partitions_by_hash.items():
        if len(partitions) > 1:
            leakage_problems.append(f"identical image {sha[:12]} appears in partitions {sorted(partitions)}")
    if leakage_problems:
        raise PrepareError(
            "LEAKAGE DETECTED — refusing to write manifests:\n  " + "\n  ".join(leakage_problems[:10])
        )

    # ---- write outputs -------------------------------------------------
    rows_all = [r.as_row(manifest_version, seed) for r in all_records]
    write_csv(output_dir / "all_images.csv", rows_all, MANIFEST_COLUMNS)

    split_paths: Dict[str, str] = {}
    for partition, filename in SPLIT_FILES.items():
        subset = [r for r in final_included if r.split == partition]
        if not subset:
            stale = output_dir / filename
            if stale.exists():
                stale.unlink()
            continue
        path = write_csv(
            output_dir / filename,
            [r.as_row(manifest_version, seed) for r in subset],
            MANIFEST_COLUMNS,
        )
        split_paths[partition] = str(path)
        logger.info("%-14s %6d images -> %s", partition, len(subset), path.name)

    audit_path = mapper.write_audit(output_dir / "label_mapping_audit.csv")
    duplicate_path = write_csv(
        output_dir / "duplicate_audit.csv",
        duplicate_rows,
        ("kind", "distance", "canonical", "duplicate", "canonical_source", "duplicate_source",
         "canonical_label", "duplicate_label", "same_label"),
    )

    # Cross-folder patient identifiers are NOT resolved into one group: the
    # project's established policy is folder-scoped keys, and identifiers such as
    # `0_right.jpeg` recur in every class folder. The collisions are counted and
    # reported so the limitation is visible rather than hidden.
    folders_by_pid: Dict[str, set] = defaultdict(set)
    for record in final_included:
        if record.source_dataset == "mendeley" and record.patient_id:
            folders_by_pid[record.patient_id].add(record.original_label)
    cross_folder_ids = {pid: sorted(folders) for pid, folders in folders_by_pid.items() if len(folders) > 1}

    split_summary = _split_summary(final_included)
    class_totals = {name: 0 for name in cfg.classes}
    for record in final_included:
        if record.class_name in class_totals:
            class_totals[record.class_name] += 1

    sources_present = sorted({r.source_dataset for r in final_included if r.split != "external_test"})
    external_sources_present = sorted(
        {r.source_dataset for r in final_included if r.split == "external_test"}
    )
    min_sources = int(cfg.get("run.min_combined_sources", 2) or 2)
    if dataset_mode == "external":
        report_mode = "external"
        combined_note = (
            "External-validation mode: the classifier trains only on "
            + ", ".join(cfg.development_sources)
            + "; these sources are held out entirely for external evaluation: "
            + (", ".join(external_sources_present) if external_sources_present else "none")
            + ". No external test image influences training or model selection."
        )
    elif len(sources_present) >= min_sources:
        report_mode = "combined"
        combined_note = (
            "One five-class dataset built from: " + ", ".join(sources_present) + "."
        )
    else:
        report_mode = "partial"
        combined_note = (
            "Only these sources contributed usable images: "
            + (", ".join(sources_present) if sources_present else "none")
            + ". The classifier trained on this manifest is NOT a combined model "
            "covering every configured source."
        )

    external_composition = dict(Counter(
        r.source_dataset for r in final_included if r.split == "external_test"
    ))

    summary = {
        "manifest_version": manifest_version,
        "generated": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
        "seed": seed,
        "dataset_mode": dataset_mode,
        "mode": report_mode,
        "combined_note": combined_note,
        "sources_present": sources_present,
        "external_sources_present": external_sources_present,
        "external_partition_composition": external_composition,
        "taxonomy": {"index_to_class": {i: name for i, name in enumerate(cfg.classes)},
                     "lesion_channels": list(cfg.lesion_channels)},
        "sources": {
            "mendeley": mendeley_source_info,
            "mmrdr": mmrdr_info,
            "aptos": aptos_info,
            "ddr": ddr_info,
        },
        "counts": {
            "images_discovered": len(all_records),
            "images_included": len(final_included),
            "images_excluded": len(all_records) - len(final_included),
            "excluded_by_reason": dict(Counter(
                r.exclusion_reason for r in all_records if not r.included and r.exclusion_reason
            ).most_common()),
            "excluded_by_source": dict(Counter(
                r.source_dataset for r in all_records if not r.included
            )),
            "included_by_source": dict(Counter(r.source_dataset for r in final_included)),
            "included_by_class": class_totals,
        },
        "duplicates": duplicate_stats,
        "grouping": {
            "group_count": grouping["group_count"],
            "records_with_patient_id": sum(1 for r in final_included if r.patient_id),
            "records_without_patient_id": sum(1 for r in final_included if not r.patient_id),
            "cross_folder_patient_id_collisions": len(cross_folder_ids),
            "cross_folder_patient_id_examples": dict(list(cross_folder_ids.items())[:10]),
            "limitation": (
                "Patient identifiers are derived only from documented filename conventions "
                "(Mendeley) and are unavailable for MMRDR-CFP. Grouping therefore unions "
                "patient keys where available with exact and near-duplicate clusters. "
                "Patient-independent evaluation is only claimed for the subset that has a "
                "derived patient identifier."
            ),
        },
        "splits": split_summary,
        "split_files": split_paths,
        "leakage_checks": {
            "group_spans_partitions": False,
            "identical_image_spans_partitions": False,
            "checked_images": len(final_included),
        },
        "label_mapping_policy": {
            "provisional": bool(cfg.get("project.provisional_label_policy", True)),
            "policy_matches_repository": policy_check["consistent"],
            "audit_csv": str(audit_path),
        },
        "duplicate_audit_csv": str(duplicate_path),
        "all_images_csv": str(output_dir / "all_images.csv"),
    }

    summary_path = output_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    if events is not None:
        events.write(
            "prepare_done",
            mode=summary["mode"],
            included=len(final_included),
            excluded=len(all_records) - len(final_included),
            splits={k: v["total"] for k, v in split_summary.items()},
        )

    logger.info(
        "manifests written: %d included (%s), %d excluded",
        len(final_included), summary["mode"], len(all_records) - len(final_included),
    )
    return summary
