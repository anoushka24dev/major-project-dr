"""Dataset provenance, contamination and split-integrity enforcement.

This is the single place where the project decides whether an image, a lesion
mask, a manifest or a checkpoint may be used.  It contains no training code and
no fallback that can invent data.

What it enforces
----------------
* A lesion mask is only accepted when it sits in the official IDRiD layout, has
  the official annotation name, and has a matching official source image.  Masks
  produced by RGB thresholding are rejected: they either carry a non-official
  name, a "generated"/"synthetic"/"threshold" marker, or no source image.
* A segmentation image whose bytes are identical to an image from a *different*
  registered dataset (for example an MMRDR fundus image placed in the IDRiD
  directory) is rejected as a foreign-source image.
* Labels are mapped only through the per-source rules in ``config/config.yaml``.
  A Mendeley folder is never mapped using MMRDR's grade rules, so an arbitrary
  remap (grade 0 -> "Dry AMD") cannot happen.
* Exact duplicates (SHA-256), perceptual duplicates (dHash + aHash) and derived
  patient/eye keys are unioned into one split group, so related images can never
  straddle train/validation/test.
* Manifests carry a content digest.  A manifest set that is not stamped by the
  current policy, or was edited after stamping, is refused.
* Artifacts produced before the clean policy (checkpoints, reports, provisional
  splits) are registered as invalid for final evaluation and refused when loaded.

Every number this module reports is measured from the files on disk.  When a
dataset is absent the status is ``NOT AVAILABLE`` or ``BLOCKED``; nothing is
estimated.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

POLICY_VERSION = "data-integrity-v1-2026-11-09"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
MASK_SUFFIXES = {".tif", ".tiff", ".png", ".bmp"}

VERIFIED = "VERIFIED"
LIKELY = "LIKELY"
UNKNOWN = "UNKNOWN"
NOT_AVAILABLE = "NOT AVAILABLE"
BLOCKED = "BLOCKED"

DHASH_MAX = 4
AHASH_MAX = 6

MANIFEST_STAMP = ".clean_manifest.json"
REGISTRY_FILENAME = "artifact_status.json"

#: Filename conventions, reused from ``audit_output/pretraining_audit`` so the
#: grouping stays consistent with the project's earlier audit.
_RE_PREFIXED = re.compile(r"^(.+?)(\d+)_(left|right)_(\d+)_(\d+)\.(?:jpeg|jpg|png)$", re.IGNORECASE)
_RE_PLAIN = re.compile(r"^(\d+)(?:_(left|right))?\.(?:jpeg|jpg|png)$", re.IGNORECASE)
_RE_ZENODO = re.compile(r"^\d+\.(?:jpg|jpeg)$", re.IGNORECASE)
_RE_MMRDR = re.compile(r"^(tr|ts)[^/\\]*\.(?:jpg|jpeg|png)$", re.IGNORECASE)
_RE_IDRID_IMAGE = re.compile(r"^IDRiD_(\d{2})\.(?:jpg|jpeg|png|tif|tiff)$", re.IGNORECASE)
_RE_IDRID_MASK = re.compile(r"^IDRiD_(\d{2})_(MA|HE|EX|SE)\.(?:tif|tiff|png|bmp)$", re.IGNORECASE)
_RE_SUSPICIOUS = re.compile(
    r"(synthetic|generated|fabricat|pseudo|threshold|rgb[_\- ]?mask|fake)",
    re.IGNORECASE,
)

#: Official IDRiD lesion folder -> channel code, in the fixed channel order.
IDRID_LESION_DIRS = {
    "1. Microaneurysms": "MA",
    "2. Haemorrhages": "HE",
    "3. Hard Exudates": "EX",
    "4. Soft Exudates": "SE",
}
IDRID_CHANNEL_ORDER = ("MA", "HE", "EX", "SE")


class IntegrityError(RuntimeError):
    """Raised when data, a manifest or an artifact may not be used."""


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dhash(small: Any) -> str:
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bool(bit))
    return f"{value:016x}"


def _ahash(small: Any) -> str:
    bits = small > small.mean()
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bool(bit))
    return f"{value:016x}"


def hamming(a: str, b: str) -> int:
    """Bit distance between two hex hashes; 64 (maximum) when either is empty."""
    if not a or not b:
        return 64
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except ValueError:
        return 64


def _inspect_worker(path_str: str) -> Dict[str, Any]:
    """Module-level worker so Windows ``spawn`` can pickle it."""
    import numpy as np
    from PIL import Image

    path = Path(path_str)
    info: Dict[str, Any] = {"ok": False, "error": "", "sha256": "", "dhash": "", "ahash": "",
                            "width": 0, "height": 0, "mode": "", "lum_mean": None,
                            "lum_std": None}
    try:
        info["sha256"] = sha256_file(path)
        with Image.open(path) as image:
            info["width"] = int(image.width)
            info["height"] = int(image.height)
            info["mode"] = str(image.mode)
            gray = image.convert("L")
            small9 = np.asarray(gray.resize((9, 8), Image.BILINEAR), dtype=np.int16)
            info["dhash"] = _dhash(small9)
            small8 = np.asarray(gray.resize((8, 8), Image.BILINEAR), dtype=np.int16)
            info["ahash"] = _ahash(small8)
            stats = np.asarray(gray.resize((32, 32), Image.BILINEAR), dtype=np.float32)
            info["lum_mean"] = float(stats.mean())
            info["lum_std"] = float(stats.std())
            image.load()
        info["ok"] = True
    except Exception as exc:  # unreadable / truncated / not an image
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


@dataclass
class ImageEntry:
    """One image on disk, with everything the audit needs to remember about it."""

    path: Path
    rel_path: str
    source: str
    folder: str
    family: str
    patient_id: str = ""
    group_key: str = ""
    official_partition: str = ""
    mapped_label: Optional[int] = None
    original_label: str = ""
    sha256: str = ""
    dhash: str = ""
    ahash: str = ""
    width: int = 0
    height: int = 0
    mode: str = ""
    lum_mean: Optional[float] = None
    lum_std: Optional[float] = None
    decode_ok: bool = True
    error: str = ""
    provenance_status: str = UNKNOWN
    provenance_evidence: str = ""
    included: bool = True
    exclusion_reason: str = ""
    split: str = ""
    group_id: str = ""
    duplicate_of: str = ""
    quality_flags: str = ""

    @property
    def key(self) -> str:
        return f"{self.source}:{self.rel_path}"

    def as_row(self) -> Dict[str, Any]:
        return {
            "image_id": self.key,
            "source_dataset": self.source,
            "rel_path": self.rel_path,
            "original_filename": Path(self.rel_path).name,
            "original_label": self.original_label,
            "mapped_label": "" if self.mapped_label is None else int(self.mapped_label),
            "class_name": "",
            "patient_group_id": self.group_key,
            "duplicate_group_id": self.duplicate_of,
            "split_group_id": self.group_id,
            "split": self.split,
            "annotation_availability": "none_published",
            "sha256": self.sha256,
            "dhash": self.dhash,
            "ahash": self.ahash,
            "width": self.width,
            "height": self.height,
            "provenance_status": self.provenance_status,
            "provenance_evidence": self.provenance_evidence,
            "quality_flags": self.quality_flags,
            "included": "yes" if self.included else "no",
            "exclusion_reason": self.exclusion_reason,
        }


def inspect_entries(
    entries: Sequence[ImageEntry],
    *,
    jobs: int = 1,
    cache_path: Optional[Path] = None,
    logger=None,
) -> None:
    """Fill in hashes and decode facts for every entry, with an on-disk cache."""
    cache: Dict[str, Dict[str, Any]] = {}
    cache_file = Path(cache_path) if cache_path else None
    if cache_file is not None and cache_file.is_file():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    pending: List[Tuple[int, str, str]] = []
    for index, entry in enumerate(entries):
        try:
            stat = Path(entry.path).stat()
            key = f"{entry.path}|{stat.st_size}|{int(stat.st_mtime)}"
        except OSError as exc:
            entry.decode_ok = False
            entry.error = f"stat failed: {exc}"
            continue
        hit = cache.get(key)
        if hit is not None:
            _apply_inspection(entry, hit)
        else:
            pending.append((index, str(entry.path), key))

    if logger is not None:
        logger.info("integrity: %d image(s), %d cached, %d to hash",
                    len(entries), len(entries) - len(pending), len(pending))

    if pending:
        results: List[Dict[str, Any]] = []
        if jobs and jobs > 1 and len(pending) > 32:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=int(jobs)) as pool:
                results = list(pool.map(_inspect_worker, [item[1] for item in pending], chunksize=16))
        else:
            results = [_inspect_worker(item[1]) for item in pending]
        for (index, _path, key), info in zip(pending, results):
            cache[key] = info
            _apply_inspection(entries[index], info)
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cache), encoding="utf-8")


def _apply_inspection(entry: ImageEntry, info: Dict[str, Any]) -> None:
    entry.sha256 = str(info.get("sha256") or "")
    entry.dhash = str(info.get("dhash") or "")
    entry.ahash = str(info.get("ahash") or "")
    entry.width = int(info.get("width") or 0)
    entry.height = int(info.get("height") or 0)
    entry.mode = str(info.get("mode") or "")
    entry.lum_mean = info.get("lum_mean")
    entry.lum_std = info.get("lum_std")
    entry.decode_ok = bool(info.get("ok"))
    entry.error = str(info.get("error") or "")


# ---------------------------------------------------------------------------
# Filename families, patient keys and scanning
# ---------------------------------------------------------------------------

def classify_family(name: str) -> str:
    """Classify a filename by the documented conventions of the sources."""
    base = Path(name).name
    if _RE_MMRDR.match(base) and base.lower()[:2] in {"tr", "ts"}:
        return "mmrdr_official"
    if _RE_IDRID_MASK.match(base):
        return "idrid_mask"
    if _RE_IDRID_IMAGE.match(base):
        return "idrid_image"
    if _RE_ZENODO.match(base):
        return "zenodo_numbered"
    if _RE_PREFIXED.match(base):
        return "prefixed"
    if _RE_PLAIN.match(base):
        return "plain"
    return "unknown"


def mendeley_patient(folder: str, name: str) -> Tuple[str, str]:
    """Return ``(patient_id, group_key)`` for a Mendeley-corpus filename.

    The key is folder-scoped because identifiers such as ``0_right.jpeg`` recur
    in every class folder; an unscoped key would merge unrelated patients.
    """
    base = Path(name).name
    match = _RE_PREFIXED.match(base)
    if match:
        pid = match.group(2)
        return pid, f"prefixed:{folder}|ID={pid}"
    match = _RE_PLAIN.match(base)
    if match:
        pid = match.group(1)
        return pid, f"plain:{folder}|ID={pid}"
    return "", ""


def mmrdr_partition(name: str) -> str:
    """Official MMRDR partition from the ``tr``/``ts`` filename prefix."""
    stem = Path(name).stem.lower()
    if stem.startswith("tr"):
        return "train"
    if stem.startswith("ts"):
        return "test"
    return ""


def iter_source_images(source: str, root: Path) -> List[ImageEntry]:
    """List every image under ``root`` as an :class:`ImageEntry` (no hashing)."""
    root = Path(root)
    entries: List[ImageEntry] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        rel = path.relative_to(root).as_posix()
        folder = rel.split("/")[0] if "/" in rel else ""
        family = classify_family(path.name)
        patient_id, group_key = "", ""
        if source == "mendeley":
            patient_id, group_key = mendeley_patient(folder, path.name)
        elif source == "mmrdr":
            group_key = f"mmrdr:{Path(rel).stem.lower()}"
        entries.append(
            ImageEntry(
                path=path,
                rel_path=rel,
                source=source,
                folder=folder,
                family=family,
                patient_id=patient_id,
                group_key=group_key,
                official_partition=mmrdr_partition(path.name) if source == "mmrdr" else "",
            )
        )
    return entries


def _find_dir(root: Path, relative: str) -> Optional[Path]:
    direct = Path(root) / relative
    if direct.is_dir():
        return direct
    if not Path(root).is_dir():
        return None
    for child in sorted(Path(root).iterdir()):
        if child.is_dir() and (child / relative).is_dir():
            return child / relative
    return None


def idrid_layout(root: Path) -> Dict[str, Any]:
    """Inspect the IDRiD tree; never guess a folder that is not published."""
    root = Path(root)
    report: Dict[str, Any] = {
        "root": str(root),
        "root_exists": root.is_dir(),
        "images_dir": None,
        "lesion_dirs": {},
        "missing": [],
    }
    if not report["root_exists"]:
        report["missing"].append(f"IDRiD root does not exist: {root}")
        report["lesion_dirs"] = {code: None for code in IDRID_CHANNEL_ORDER}
        return report

    images_dir = _find_dir(root, "1. Original Images/a. Training Set")
    if images_dir is None:
        for candidate in sorted(root.rglob("a. Training Set")):
            if candidate.is_dir() and any(
                p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES for p in candidate.iterdir()
            ):
                images_dir = candidate
                break
    report["images_dir"] = str(images_dir) if images_dir else None
    if images_dir is None:
        report["missing"].append("missing '1. Original Images/a. Training Set'")

    for folder, code in IDRID_LESION_DIRS.items():
        found = None
        for candidate in root.rglob(folder):
            if candidate.is_dir() and any(p.is_file() for p in candidate.iterdir()):
                found = candidate
                break
        report["lesion_dirs"][code] = str(found) if found else None
        if found is None:
            report["missing"].append(f"missing official lesion folder {folder!r} ({code})")
    return report


def iter_idrid_images(root: Path) -> List[ImageEntry]:
    layout = idrid_layout(root)
    images_dir = Path(layout["images_dir"]) if layout.get("images_dir") else None
    if images_dir is None:
        return []
    entries: List[ImageEntry] = []
    for path in sorted(images_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        match = _RE_IDRID_IMAGE.match(path.name)
        entries.append(
            ImageEntry(
                path=path,
                rel_path=path.relative_to(Path(root)).as_posix(),
                source="idrid",
                folder="1. Original Images",
                family="idrid_image",
                group_key=f"idrid:{match.group(1)}" if match else f"idrid:{path.stem}",
            )
        )
    return entries


@dataclass
class MaskAudit:
    layout_ok: bool
    images: int
    channels: Dict[str, int] = field(default_factory=dict)
    pairing_ok: bool = False
    missing_masks: Dict[str, int] = field(default_factory=dict)
    orphan_masks: Dict[str, int] = field(default_factory=dict)
    fabricated: Dict[str, List[str]] = field(default_factory=dict)
    foreign_masks: Dict[str, str] = field(default_factory=dict)
    status: str = BLOCKED
    notes: List[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return (
            self.layout_ok
            and self.images > 0
            and self.pairing_ok
            and not self.fabricated
            and not self.foreign_masks
        )


def audit_idrid_masks(root: Path, *, foreign_shas: Optional[Set[str]] = None) -> MaskAudit:
    """Validate IDRiD masks against the official layout, names and images.

    A mask is rejected as fabricated when any of these holds:

    * its filename is not an official ``IDRiD_<nn>_<CHANNEL>`` annotation name;
    * it does not sit under ``2. All Segmentation Groundtruths``;
    * the channel in its filename disagrees with the official folder it is in;
    * there is no official source image with that stem;
    * its name or directory carries a generated/synthetic/threshold marker;
    * a ``*.provenance.json`` sidecar states that it was generated.

    The last two exist because a thresholded mask can otherwise look like a
    perfectly valid binary TIFF.
    """
    root = Path(root)
    layout = idrid_layout(root)
    audit = MaskAudit(layout_ok=not layout["missing"], images=0)
    if layout["images_dir"] is None:
        audit.notes.append("no official IDRiD training images were found")
        return audit

    images_dir = Path(layout["images_dir"])
    image_stems: Set[str] = set()
    for path in images_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            image_stems.add(path.stem)
    audit.images = len(image_stems)

    foreign_shas = foreign_shas or set()
    if foreign_shas:
        for path in images_dir.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                try:
                    digest = sha256_file(path)
                except OSError:
                    continue
                if digest in foreign_shas:
                    audit.foreign_masks[path.name] = (
                        "byte-identical to an image from a different dataset; "
                        "this is not a genuine IDRiD photograph"
                    )
        if audit.foreign_masks:
            audit.notes.append(
                f"{len(audit.foreign_masks)} IDRiD image(s) are byte-identical to another "
                "dataset's images"
            )

    mask_stems_by_channel: Dict[str, Set[str]] = {code: set() for code in IDRID_CHANNEL_ORDER}
    for code, directory in (layout["lesion_dirs"] or {}).items():
        if not directory:
            audit.missing_masks[code] = len(image_stems)
            audit.channels[code] = 0
            continue
        folder = Path(directory)
        count = 0
        for path in folder.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in MASK_SUFFIXES:
                continue
            count += 1
            reasons: List[str] = []

            rel_parts = path.relative_to(root).parts
            if len(rel_parts) < 2 or "Segmentation Groundtruths" not in rel_parts[0]:
                reasons.append("mask is not inside '2. All Segmentation Groundtruths'")

            match = _RE_IDRID_MASK.match(path.name)
            if match is None:
                reasons.append("filename is not an official IDRiD annotation name")
            else:
                if match.group(2).upper() != code:
                    reasons.append(
                        f"channel {match.group(2).upper()} does not match folder channel {code}"
                    )
                mask_stems_by_channel[code].add(f"IDRiD_{match.group(1)}")
                if f"IDRiD_{match.group(1)}" not in image_stems:
                    reasons.append(
                        f"no official IDRiD image with stem IDRiD_{match.group(1)}"
                    )

            if _RE_SUSPICIOUS.search(path.name) or _RE_SUSPICIOUS.search(str(path.parent)):
                reasons.append("filename or directory carries a generated/synthetic marker")

            sidecar = path.with_suffix(path.suffix + ".provenance.json")
            if sidecar.is_file():
                try:
                    text = sidecar.read_text(encoding="utf-8", errors="replace").lower()
                except OSError:
                    text = ""
                if any(word in text for word in ("generated", "threshold", "synthetic", "fabricat")):
                    reasons.append("provenance sidecar states the mask was generated")

            try:
                digest = sha256_file(path)
            except OSError:
                digest = ""
            if digest and digest in foreign_shas:
                audit.foreign_masks[path.name] = (
                    "mask file is byte-identical to another dataset's image"
                )

            if reasons:
                audit.fabricated[path.relative_to(root).as_posix()] = reasons

        audit.channels[code] = count
        missing = sorted(image_stems - mask_stems_by_channel[code])
        orphans = sorted(mask_stems_by_channel[code] - image_stems)
        if missing:
            audit.missing_masks[code] = len(missing)
        if orphans:
            audit.orphan_masks[code] = len(orphans)

    audit.pairing_ok = (
        audit.images > 0
        and not audit.missing_masks
        and not audit.orphan_masks
        and all(audit.channels.get(code, 0) >= audit.images for code in IDRID_CHANNEL_ORDER)
    )
    audit.status = VERIFIED if audit.usable else BLOCKED
    if audit.fabricated:
        audit.notes.append(f"{len(audit.fabricated)} fabricated mask(s) rejected")
    if audit.orphan_masks:
        audit.notes.append(f"orphan masks: {audit.orphan_masks}")
    if audit.missing_masks:
        audit.notes.append(f"missing masks: {audit.missing_masks}")
    return audit


# ---------------------------------------------------------------------------
# Duplicates and split groups
# ---------------------------------------------------------------------------

def exact_duplicate_groups(entries: Sequence[ImageEntry]) -> Dict[str, List[int]]:
    by_sha: Dict[str, List[int]] = defaultdict(list)
    for index, entry in enumerate(entries):
        if entry.sha256 and entry.decode_ok:
            by_sha[entry.sha256].append(index)
    return {sha: idx for sha, idx in by_sha.items() if len(idx) > 1}


def near_duplicate_pairs(
    entries: Sequence[ImageEntry],
    *,
    dhash_max: int = DHASH_MAX,
    ahash_max: int = AHASH_MAX,
) -> List[Tuple[int, int, int, int]]:
    """Return ``(i, j, dhash_distance, ahash_distance)`` perceptual pairs."""
    import numpy as np

    usable = [i for i, e in enumerate(entries) if e.decode_ok and e.dhash and e.ahash]
    if len(usable) < 2:
        return []
    d = np.array([int(entries[i].dhash, 16) for i in usable], dtype=np.uint64)
    a = np.array([int(entries[i].ahash, 16) for i in usable], dtype=np.uint64)

    pairs: List[Tuple[int, int, int, int]] = []
    n = len(usable)
    for i in range(n):
        for j in range(i + 1, n):
            di = int(d[i] ^ d[j])
            if di > dhash_max:
                continue
            aj = int(a[i] ^ a[j])
            if aj > ahash_max:
                continue
            pairs.append((usable[i], usable[j], di, aj))
    return pairs


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

    def union(self, first: str, second: str) -> None:
        ra, rb = self.find(first), self.find(second)
        if ra != rb:
            if ra > rb:
                ra, rb = rb, ra
            self.parent[rb] = ra


@dataclass
class Grouping:
    group_id_by_index: Dict[int, str] = field(default_factory=dict)
    components: Dict[str, List[int]] = field(default_factory=dict)
    conflicts: Dict[str, List[int]] = field(default_factory=dict)
    reason_by_root: Dict[str, str] = field(default_factory=dict)

    def group_of(self, index: int) -> str:
        return self.group_id_by_index.get(index, "")


def build_grouping(entries: Sequence[ImageEntry], pairs: Sequence[Tuple[int, int, int, int]]) -> Grouping:
    """Union patient keys, exact duplicates and perceptual duplicates into groups."""
    uf = UnionFind()
    for index, entry in enumerate(entries):
        uf.find(f"i{index}")
        if entry.group_key:
            uf.union(f"i{index}", f"k:{entry.group_key}")
    # Exact duplicates are unioned here rather than only by the caller, so an
    # identical image can never be split just because its caller forgot to pass
    # the SHA-256 group through.
    by_sha: Dict[str, List[int]] = defaultdict(list)
    for index, entry in enumerate(entries):
        if entry.sha256:
            by_sha[entry.sha256].append(index)
    for members in by_sha.values():
        for other in members[1:]:
            uf.union(f"i{members[0]}", f"i{other}")
    for first, second, _d, _a in pairs:
        uf.union(f"i{first}", f"i{second}")

    components: Dict[str, List[int]] = defaultdict(list)
    for index in range(len(entries)):
        components[uf.find(f"i{index}")].append(index)

    grouping = Grouping()
    ordered = sorted(
        components.items(),
        key=lambda item: min(entries[i].rel_path for i in item[1]),
    )
    for number, (root, members) in enumerate(ordered, start=1):
        group_id = f"SG-{number:05d}"
        labels = {entries[i].mapped_label for i in members}
        if len(labels) > 1:
            grouping.conflicts[root] = members
            grouping.reason_by_root[root] = (
                "members of one duplicate/patient group carry different five-class labels "
                f"({sorted(str(label) for label in labels)}); the group cannot be graded or split"
            )
        for index in members:
            grouping.group_id_by_index[index] = group_id
        grouping.components[group_id] = members
    return grouping


def allocate_splits(
    entries: Sequence[ImageEntry],
    grouping: Grouping,
    *,
    included: Sequence[int],
    seed: int,
    ratios: Dict[str, float],
) -> Dict[int, str]:
    """Group-aware stratified split; every group lands in exactly one partition.

    Groups are visited largest-first and assigned to the partition with the
    largest remaining deficit for that class, which keeps the class ratios close
    without ever splitting a group.
    """
    import random

    partitions = ("train", "validation", "test")
    included_set = set(included)
    by_class: Dict[int, List[List[int]]] = defaultdict(list)
    for group_id, members in grouping.components.items():
        usable = [i for i in members if i in included_set]
        if not usable:
            continue
        labels = {entries[i].mapped_label for i in members}
        if len(labels) > 1:
            continue
        label = entries[usable[0]].mapped_label
        if label is None:
            continue
        by_class[int(label)].append(usable)

    rng = random.Random(seed)
    assignment: Dict[int, str] = {}
    for label in sorted(by_class):
        groups = by_class[label]
        groups.sort(key=lambda members: (-len(members), min(entries[i].rel_path for i in members)))
        rng.shuffle(groups)
        groups.sort(key=lambda members: -len(members))
        totals = {partition: 0 for partition in partitions}
        target = {
            partition: float(ratios.get(partition, 0.0)) * sum(len(g) for g in groups)
            for partition in partitions
        }
        for members in groups:
            best = partitions[0]
            best_deficit = None
            for partition in partitions:
                deficit = target[partition] - totals[partition]
                if best_deficit is None or deficit > best_deficit:
                    best, best_deficit = partition, deficit
            for index in members:
                assignment[index] = best
            totals[best] += len(members)

    # Official partitions win: MMRDR's `ts` images are an external test set and
    # must never appear in train or validation.
    for index, entry in enumerate(entries):
        if index in assignment and entry.official_partition == "test":
            group = grouping.group_id_by_index.get(index)
            members = grouping.components.get(group, [index])
            for member in members:
                if member in assignment:
                    assignment[member] = "external_test"
    return assignment


def assert_split_isolation(
    entries: Sequence[ImageEntry],
    grouping: Grouping,
    assignment: Dict[int, str],
) -> List[str]:
    """Return every way in which a group, hash or patient key straddles partitions."""
    problems: List[str] = []
    partitions_by_group: Dict[str, Set[str]] = defaultdict(set)
    partitions_by_hash: Dict[str, Set[str]] = defaultdict(set)
    partitions_by_patient: Dict[str, Set[str]] = defaultdict(set)
    for index, split in assignment.items():
        entry = entries[index]
        partitions_by_group[grouping.group_id_by_index.get(index, entry.key)].add(split)
        if entry.sha256:
            partitions_by_hash[entry.sha256].add(split)
        if entry.group_key:
            partitions_by_patient[entry.group_key].add(split)
    for name, partitions in (
        ("split group", partitions_by_group),
        ("identical image (sha256)", partitions_by_hash),
        ("patient/eye group", partitions_by_patient),
    ):
        for key, values in partitions.items():
            if len(values) > 1:
                problems.append(f"{name} {key!r} spans partitions {sorted(values)}")
    return problems


def cross_source_duplicates(entries: Sequence[ImageEntry]) -> List[Dict[str, Any]]:
    """Images whose bytes appear under more than one registered dataset.

    This is the detector for foreign images: an MMRDR photograph copied into the
    IDRiD directory, or into a Mendeley class folder, shares a SHA-256 with the
    image in its real source.
    """
    by_sha: Dict[str, List[int]] = defaultdict(list)
    for index, entry in enumerate(entries):
        if entry.sha256:
            by_sha[entry.sha256].append(index)
    findings: List[Dict[str, Any]] = []
    for digest, indices in by_sha.items():
        sources = {entries[i].source for i in indices}
        if len(sources) < 2:
            continue
        findings.append(
            {
                "sha256": digest,
                "sources": sorted(sources),
                "members": [
                    {
                        "source": entries[i].source,
                        "rel_path": entries[i].rel_path,
                        "original_label": entries[i].original_label,
                    }
                    for i in indices
                ],
            }
        )
    findings.sort(key=lambda f: (f["sources"], f["members"][0]["rel_path"]))
    return findings


# ---------------------------------------------------------------------------
# Manifest stamping and stale-artifact refusal
# ---------------------------------------------------------------------------

def manifest_digest(manifest_dir: Path, filenames: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(filenames):
        path = Path(manifest_dir) / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def write_manifest_stamp(
    manifest_dir: Path,
    *,
    files: Sequence[str],
    counts: Dict[str, Any],
    source_statuses: Dict[str, str],
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Stamp a manifest set with its content digest and the policy that produced it."""
    manifest_dir = Path(manifest_dir)
    payload = {
        "policy_version": POLICY_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "digest": manifest_digest(manifest_dir, files),
        "files": list(files),
        "counts": counts,
        "source_statuses": source_statuses,
        "contaminated": False,
        "clean": True,
    }
    if extra:
        payload.update(extra)
    path = manifest_dir / MANIFEST_STAMP
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def require_clean_manifests(manifest_dir: Path) -> Dict[str, Any]:
    """Refuse an unstamped, stale or edited manifest set."""
    manifest_dir = Path(manifest_dir)
    stamp_path = manifest_dir / MANIFEST_STAMP
    if not stamp_path.is_file():
        raise IntegrityError(
            f"{manifest_dir} has no {MANIFEST_STAMP}. The manifests were not produced or "
            f"checked by the clean policy ({POLICY_VERSION}); rebuild them with "
            "`python scripts/audit_datasets.py` and `RUN.bat --prepare-only`."
        )
    try:
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise IntegrityError(f"manifest stamp is unreadable: {stamp_path} ({exc})") from exc
    if stamp.get("policy_version") != POLICY_VERSION:
        raise IntegrityError(
            f"manifest stamp policy {stamp.get('policy_version')!r} is not the current "
            f"policy {POLICY_VERSION!r}; rebuild the manifests."
        )
    if stamp.get("contaminated") or not stamp.get("clean", False):
        raise IntegrityError(
            "the manifest stamp marks this dataset as contaminated or unchecked; "
            "it must not be used for training or evaluation."
        )
    files = list(stamp.get("files") or [])
    missing = [name for name in files if not (manifest_dir / name).is_file()]
    if missing:
        raise IntegrityError(f"stamped manifest files are missing: {missing}")
    current = manifest_digest(manifest_dir, files)
    if current != stamp.get("digest"):
        raise IntegrityError(
            "manifests changed after they were stamped (digest mismatch). They may have been "
            "edited or replaced by a contaminated run; rebuild and re-audit before training."
        )
    return stamp


def collect_stale_artifacts(root: Path) -> List[Dict[str, str]]:
    """List artifacts that predate the clean policy and must not be reused."""
    root = Path(root)
    artifacts: List[Dict[str, str]] = []

    def add(path: Path, category: str, reason: str) -> None:
        if path.exists():
            artifacts.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "category": category,
                    "status": "invalid_for_final_evaluation",
                    "reason": reason,
                }
            )

    checkpoint_reason = (
        "trained before the clean-manifest policy; no clean manifest digest is recorded, "
        "so its training data cannot be shown to be free of fabricated labels/masks"
    )
    for pattern in ("*.pt", "*.pth", "*.onnx"):
        for path in sorted(root.rglob(pattern)):
            if ".git" in path.parts or ".venv" in path.parts:
                continue
            add(path, "checkpoint", checkpoint_reason)

    metric_reason = "metrics computed from a manifest that predates the clean policy"
    for relative in (
        "outputs/evaluation/metrics_summary.csv",
        "outputs/segmentation/segmentation_metrics.json",
        "outputs/segmentation/segmentation_summary.json",
        "outputs/reports/final_report.html",
        "demo/results/metrics.json",
        "demo/results/DEMO_RESULTS.md",
    ):
        add(root / relative, "metric_or_report", metric_reason)

    manifest_reason = "manifest built before the clean policy; provenance and duplicate audit incomplete"
    for relative in (
        "provisional_primary_split/provisional_primary_split.csv",
        "audit_output/pretraining_audit/PROVISIONAL_ELIGIBILITY_PRIMARY.csv",
        "audit_output/pretraining_audit/PROVISIONAL_ELIGIBILITY_SECONDARY.csv",
        "audit_output/pretraining_audit/PROVENANCE_MANIFEST.csv",
        "demo/manifests/selected_train.csv",
        "demo/manifests/selected_test.csv",
        "demo/manifests/dataset_inventory.csv",
    ):
        add(root / relative, "manifest", manifest_reason)

    return artifacts


def write_artifact_registry(root: Path, out_dir: Path) -> Path:
    """Write (or refresh) the registry of invalidated artifacts."""
    root = Path(root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = collect_stale_artifacts(root)
    payload = {
        "policy_version": POLICY_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": (
            "Artifacts listed here were produced before the clean data policy. They are kept "
            "for history only and must not be cited as final results or loaded for evaluation."
        ),
        "artifacts": artifacts,
    }
    path = out_dir / REGISTRY_FILENAME
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def assert_artifact_usable(path: Path, registry_path: Path) -> None:
    """Raise when a checkpoint/report is registered as invalid for final evaluation."""
    registry_path = Path(registry_path)
    if not registry_path.is_file():
        raise IntegrityError(
            f"no artifact registry at {registry_path}; run `python scripts/audit_datasets.py` "
            "before reusing a checkpoint or reporting a metric."
        )
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise IntegrityError(f"artifact registry is unreadable: {registry_path} ({exc})") from exc
    if payload.get("policy_version") != POLICY_VERSION:
        raise IntegrityError(
            "artifact registry was written by a different policy version; re-run the audit."
        )
    target = Path(path).resolve()
    for entry in payload.get("artifacts") or []:
        candidate = (registry_path.parent.parent / entry["path"]).resolve()
        if candidate == target:
            raise IntegrityError(
                f"{entry['path']} is registered as {entry['status']}: {entry['reason']}. "
                "Retrain under the clean manifests instead of reusing it."
            )


# ---------------------------------------------------------------------------
# Cohort eligibility
# ---------------------------------------------------------------------------

#: Images whose downsampled luminance standard deviation is below this carry no
#: retinal structure and are excluded (measured, not guessed).
NEAR_UNIFORM_STD = 8.0


def ineligibility_reason(entry: ImageEntry) -> str:
    """Return why an image may not enter the clean five-class cohort, or ``""``.

    This is the single eligibility rule used by the audit and by the tests; a
    missing, unverified or conflicting entry can never slip through because two
    code paths disagree.
    """
    if entry.exclusion_reason:
        return entry.exclusion_reason
    if not entry.decode_ok:
        return f"unreadable_image:{entry.error}"
    if entry.mapped_label is None:
        return "unmapped_label:no_declared_five_class_rule"
    if entry.provenance_status == UNKNOWN:
        return "unverified_provenance:filename_family_unrecognised"
    if entry.lum_std is not None and entry.lum_std < NEAR_UNIFORM_STD:
        entry.quality_flags = "near_uniform_low_std"
        return "near_uniform_image:no_retinal_structure"
    return ""


# ---------------------------------------------------------------------------
# Small helpers shared by the audit writer
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})
    return target


def per_class_counts(entries: Iterable[ImageEntry], class_names: Sequence[str]) -> Dict[str, int]:
    counts = {name: 0 for name in class_names}
    for entry in entries:
        if entry.mapped_label is not None and 0 <= entry.mapped_label < len(class_names):
            counts[class_names[entry.mapped_label]] += 1
    return counts
