"""Prepare the 50-train / 10-test demonstration dataset from the V03 fundus archive.

What this script does, in order
------------------------------
1. Locate and open the ZIP archive (without modifying it) and list its internal
   structure before extracting anything.
2. Extract it safely into ``datasets/demo_fundus_v03`` (inside ``.gitignore``),
   rejecting absolute paths and ``..`` traversal.
3. Verify every image decodes, hash every file, and flag duplicate files.
4. Map the archive's folder names onto the repository's five-class taxonomy,
   applying the repository's existing provisional merges (Very Severe NPDR ->
   Severe NPDR, Advanced PDR -> PDR) - never inventing a new mapping.
5. Draw a class-aware, seeded split: 50 training images (41 fit + 9 validation)
   and 10 test images, with no overlap.
6. Write CSV manifests with relative paths only (no private local paths are ever
   written to a committed file).

The archive is read-only from this script's point of view: it is never deleted,
moved or rewritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
CONFIG_PATH = DEMO_DIR / "config.json"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MANIFEST_FIELDS = [
    "filename",
    "rel_path",
    "source_class",
    "mapped_label",
    "class_name",
    "used_for",
    "split",
    "sha256",
    "width",
    "height",
    "mapping_provenance",
]


# ---------------------------------------------------------------------------
# Configuration and small utilities
# ---------------------------------------------------------------------------

def load_demo_config(path: Path | str = CONFIG_PATH) -> Dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Demo configuration not found: {source}")
    with source.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _expand(candidate: str) -> Path:
    text = os.path.expandvars(str(candidate))
    if text.startswith("~"):
        text = os.path.expanduser(text)
    return Path(text)


def resolve_archive(config: Dict[str, Any], override: Optional[str] = None) -> Path:
    """Find the dataset ZIP: ``--zip`` wins, then ``DEMO_DATASET_ZIP``, then candidates."""
    tried: List[str] = []
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        tried.append(str(candidate))

    env = os.environ.get("DEMO_DATASET_ZIP", "").strip()
    if env:
        candidate = Path(env).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        tried.append(str(candidate))

    for entry in config.get("dataset", {}).get("archive_candidates", []):
        candidate = _expand(entry)
        if candidate.is_file():
            return candidate.resolve()
        tried.append(str(candidate))
        # A version-agnostic fallback inside the same folder (V02, V04, ...).
        pattern = re.compile(r"Dataset from fundus images.*\.zip$", re.IGNORECASE)
        if candidate.parent.is_dir():
            matches = sorted(
                p for p in candidate.parent.iterdir()
                if p.is_file() and pattern.search(p.name)
            )
            if matches:
                print(
                    "NOTE: the configured archive was not found; using the closest "
                    f"matching archive instead: {matches[-1].name}"
                )
                return matches[-1].resolve()

    raise FileNotFoundError(
        "Could not find the fundus dataset ZIP archive.\n"
        "Looked for:\n  " + "\n  ".join(tried) + "\n"
        "Fix: pass the path explicitly, e.g.\n"
        '  python demo/prepare_data.py --zip "C:/path/to/Dataset from fundus images ...V03.zip"\n'
        "or set the DEMO_DATASET_ZIP environment variable."
    )


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_member(name: str) -> str:
    """Return the top-level folder of a ZIP member ('' when at the archive root)."""
    parts = [p for p in name.replace("\\", "/").split("/") if p]
    return parts[0] if len(parts) > 1 else ""


def is_safe_member(name: str) -> bool:
    normalised = name.replace("\\", "/")
    if normalised.startswith("/") or re.match(r"^[A-Za-z]:", normalised):
        return False
    return ".." not in normalised.split("/")


# ---------------------------------------------------------------------------
# Step 1-2: inspect, then extract
# ---------------------------------------------------------------------------

def inspect_archive(zf: zipfile.ZipFile, config: Dict[str, Any]) -> Dict[str, Any]:
    """Describe the archive's internal structure before any extraction."""
    suffixes = {s.lower() for s in config["dataset"]["image_suffixes"]}
    per_folder: Dict[str, Counter] = defaultdict(Counter)
    images = 0
    others: List[str] = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        suffix = Path(info.filename).suffix.lower()
        folder = classify_member(info.filename)
        per_folder[folder][suffix] += 1
        if suffix in suffixes:
            images += 1
        else:
            others.append(info.filename)

    print("Archive structure (before extraction)")
    print(f"  archive      : {zf.filename}")
    for folder in sorted(per_folder, key=lambda x: (x == "", x)):
        counts = per_folder[folder]
        label = folder if folder else "(archive root)"
        detail = ", ".join(f"{n} x {suffix or '(no suffix)'}" for suffix, n in sorted(counts.items()))
        print(f"  {label:<32} {detail}")
    print(f"  image files  : {images}")
    if others:
        print(f"  non-image    : {others}")

    expected = set(config["label_mapping"])
    present = {folder for folder, counts in per_folder.items() if counts.get(".jpg") or folder}
    missing = sorted(expected - present)
    unexpected = sorted(f for f in present if f and f not in expected)
    if missing:
        raise ValueError(
            "The archive does not contain every expected DR folder.\n"
            f"Missing: {missing}\n"
            "Refusing to guess folders or remap labels."
        )
    if unexpected:
        print(f"  NOTE: extra top-level folders/files ignored: {unexpected}")

    return {
        "per_folder": {k: dict(v) for k, v in per_folder.items()},
        "image_files": images,
        "non_image_entries": others,
    }


def safe_extract(zf: zipfile.ZipFile, destination: Path, *, force: bool = False) -> int:
    """Extract members that sit inside the destination, resuming when possible."""
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    extracted = 0
    skipped = 0
    for info in zf.infolist():
        if info.is_dir():
            continue
        if not is_safe_member(info.filename):
            raise ValueError(f"Refusing to extract unsafe archive member: {info.filename!r}")
        target = (destination / info.filename.replace("\\", "/")).resolve()
        if destination not in target.parents and target != destination:
            raise ValueError(f"Refusing to extract outside the dataset folder: {info.filename!r}")
        if target.is_file() and not force and target.stat().st_size == info.file_size:
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as source, target.open("wb") as sink:
            sink.write(source.read())
        extracted += 1
    print(f"Extracted {extracted} file(s), reused {skipped} already-extracted file(s) -> {destination}")
    return extracted


# ---------------------------------------------------------------------------
# Step 3-4: verify, hash, and map labels
# ---------------------------------------------------------------------------

def verify_and_inventory(
    extract_root: Path, config: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Decode-check and hash every image; never trust a file just because it exists."""
    from PIL import Image

    suffixes = {s.lower() for s in config["dataset"]["image_suffixes"]}
    mapping = config["label_mapping"]
    rows: List[Dict[str, Any]] = []
    corrupt: List[Dict[str, str]] = []
    unmapped: List[str] = []

    for folder in sorted(mapping):
        folder_dir = extract_root / folder
        if not folder_dir.is_dir():
            raise FileNotFoundError(f"Expected class folder is missing after extraction: {folder_dir}")
        files = sorted(p for p in folder_dir.iterdir() if p.is_file() and p.suffix.lower() in suffixes)
        for path in files:
            rel = path.relative_to(extract_root).as_posix()
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    width, height = image.size
                    image.convert("RGB").load()
            except Exception as exc:  # unreadable / truncated file
                corrupt.append({"rel_path": rel, "source_class": folder, "error": f"{type(exc).__name__}: {exc}"})
                continue
            spec = mapping[folder]
            rows.append(
                {
                    "filename": path.name,
                    "rel_path": rel,
                    "source_class": folder,
                    "mapped_label": int(spec["mapped_label"]),
                    "class_name": spec["class_name"],
                    "used_for": "",
                    "split": "",
                    "sha256": sha256_of(path),
                    "width": int(width),
                    "height": int(height),
                    "mapping_provenance": spec["provenance"],
                }
            )

    # Any image that sits in a folder we do not recognise is reported, not guessed.
    known = set(mapping)
    for path in sorted(extract_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes:
            top = path.relative_to(extract_root).parts[0]
            if top not in known:
                unmapped.append(path.relative_to(extract_root).as_posix())

    summary = {
        "usable_images": len(rows),
        "corrupt_images": corrupt,
        "images_in_unknown_folders": unmapped,
        "per_source_class": dict(Counter(row["source_class"] for row in rows)),
        "per_mapped_class": dict(Counter(row["class_name"] for row in rows)),
    }
    return rows, summary


def find_duplicates(rows: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["sha256"]].append(row)
    return {digest: members for digest, members in groups.items() if len(members) > 1}


# ---------------------------------------------------------------------------
# Step 5-6: class-aware sampling and manifests
# ---------------------------------------------------------------------------

def _pick(pool: List[Dict[str, Any]], count: int, *, seed: int, tag: str, label: int) -> List[Dict[str, Any]]:
    if count == 0:
        return []
    if len(pool) < count:
        raise ValueError(
            f"Not enough usable images for class {label} to draw {count} image(s) for {tag}: "
            f"only {len(pool)} available. Lower the quota in demo/config.json instead of "
            "duplicating images."
        )
    rng = random.Random(f"{seed}|{tag}|{label}")
    return rng.sample(pool, count)


def select_splits(
    rows: List[Dict[str, Any]], duplicates: Dict[str, List[Dict[str, Any]]], config: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Seeded, class-aware, disjoint selection of 50 training and 10 test images."""
    seed = int(config["experiment"]["seed"])
    sampling = config["sampling"]

    duplicate_of = {}
    for digest, members in duplicates.items():
        ordered = sorted(members, key=lambda r: r["rel_path"])
        for member in ordered[1:]:
            duplicate_of[member["rel_path"]] = ordered[0]["rel_path"]

    by_class: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["rel_path"] in duplicate_of:
            continue  # identical bytes already represented by a kept file
        by_class[int(row["mapped_label"])].append(row)
    for label in by_class:
        by_class[label].sort(key=lambda r: r["filename"])

    selected: List[Dict[str, Any]] = []
    chosen: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}
    for label in sorted(by_class):
        pool = list(by_class[label])
        n_test = int(sampling["test_per_class"].get(str(label), 0))
        n_val = int(sampling["validation_per_class"].get(str(label), 0))
        n_fit = int(sampling["fit_per_class"].get(str(label), 0))

        test_rows = _pick(pool, n_test, seed=seed, tag="test", label=label)
        for row in test_rows:
            row["used_for"], row["split"] = "test", "test"
        pool = [r for r in pool if r not in test_rows]

        val_rows = _pick(pool, n_val, seed=seed, tag="validation", label=label)
        for row in val_rows:
            row["used_for"], row["split"] = "validation", "train"
        pool = [r for r in pool if r not in val_rows]

        fit_rows = _pick(pool, n_fit, seed=seed, tag="fit", label=label)
        for row in fit_rows:
            row["used_for"], row["split"] = "fit", "train"

        selected.extend(fit_rows + val_rows + test_rows)
        chosen[label] = {"fit": fit_rows, "validation": val_rows, "test": test_rows}

    expected_train = int(config["experiment"]["expected_train_total"])
    expected_test = int(config["experiment"]["expected_test_total"])
    n_train = sum(1 for r in selected if r["split"] == "train")
    n_test = sum(1 for r in selected if r["split"] == "test")
    if n_train != expected_train or n_test != expected_test:
        raise ValueError(
            f"Sampling produced {n_train} training and {n_test} test images; "
            f"the experiment requires exactly {expected_train} and {expected_test}. "
            "Check demo/config.json sampling quotas."
        )

    names_train = {r["rel_path"] for r in selected if r["split"] == "train"}
    names_test = {r["rel_path"] for r in selected if r["split"] == "test"}
    hashes_train = {r["sha256"] for r in selected if r["split"] == "train"}
    hashes_test = {r["sha256"] for r in selected if r["split"] == "test"}
    if names_train & names_test or hashes_train & hashes_test:
        raise AssertionError("Train and test partitions overlap; this must never happen.")

    detail = {
        str(label): {
            part: {
                "count": len(part_rows),
                "class_name": part_rows[0]["class_name"] if part_rows else config["taxonomy"]["classes"][label],
                "files": [r["filename"] for r in part_rows],
            }
            for part, part_rows in chosen[label].items()
        }
        for label in sorted(chosen)
    }
    return selected, detail


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_manifests(
    config: Dict[str, Any],
    selected: List[Dict[str, Any]],
    rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
    duplicates: Dict[str, List[Dict[str, Any]]],
    archive_info: Dict[str, Any],
    archive_path: Path,
) -> Dict[str, str]:
    manifest_dir = REPO_ROOT / config["dataset"]["manifest_dir"]
    manifest_dir.mkdir(parents=True, exist_ok=True)

    train_rows = sorted(
        (r for r in selected if r["split"] == "train"), key=lambda r: (int(r["mapped_label"]), r["filename"])
    )
    test_rows = sorted(
        (r for r in selected if r["split"] == "test"), key=lambda r: (int(r["mapped_label"]), r["filename"])
    )

    written = {}
    written["train"] = str(write_csv(manifest_dir / "selected_train.csv", train_rows, MANIFEST_FIELDS))
    written["test"] = str(write_csv(manifest_dir / "selected_test.csv", test_rows, MANIFEST_FIELDS))
    written["val"] = str(
        write_csv(
            manifest_dir / "selected_train_validation.csv",
            [r for r in train_rows if r["used_for"] == "validation"],
            MANIFEST_FIELDS,
        )
    )

    inventory_fields = ["filename", "rel_path", "source_class", "mapped_label", "class_name", "sha256", "width", "height"]
    written["inventory"] = str(write_csv(manifest_dir / "dataset_inventory.csv", rows, inventory_fields))

    duplicate_rows = [
        {
            "sha256": digest,
            "kept_file": sorted(members, key=lambda r: r["rel_path"])[0]["rel_path"],
            "duplicate_file": member["rel_path"],
        }
        for digest, members in sorted(duplicates.items())
        for member in sorted(members, key=lambda r: r["rel_path"])[1:]
    ]
    written["duplicates"] = str(
        write_csv(manifest_dir / "duplicates.csv", duplicate_rows, ["sha256", "kept_file", "duplicate_file"])
    )

    payload = {
        "archive_name": archive_path.name,
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_untouched": True,
        "archive_structure": archive_info,
        "seed": int(config["experiment"]["seed"]),
        "usable_images_in_archive": summary["usable_images"],
        "images_per_source_class": summary["per_source_class"],
        "images_per_mapped_class": summary["per_mapped_class"],
        "corrupt_images": summary["corrupt_images"],
        "duplicate_file_count": len(duplicate_rows),
        "images_in_unknown_folders": summary["images_in_unknown_folders"],
        "train_total": len(train_rows),
        "test_total": len(test_rows),
        "train_per_class": dict(Counter(int(r["mapped_label"]) for r in train_rows)),
        "test_per_class": dict(Counter(int(r["mapped_label"]) for r in test_rows)),
        "overlap_check": "disjoint by filename and sha256",
    }
    summary_path = manifest_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    written["summary"] = str(summary_path)
    return written


# ---------------------------------------------------------------------------
# Shared helpers used by train.py / evaluate.py
# ---------------------------------------------------------------------------

def manifest_path(config: Dict[str, Any], name: str) -> Path:
    return REPO_ROOT / config["dataset"]["manifest_dir"] / name


def extract_root(config: Dict[str, Any]) -> Path:
    return (REPO_ROOT / config["dataset"]["extract_dir"]).resolve()


def load_manifest(name: str, config: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """Load one selected manifest and make row paths absolute for runtime use.

    The committed CSV keeps only relative paths; the absolute path is attached in
    memory here and is never written back to disk.
    """
    from src.data import load_manifest_rows

    config = config or load_demo_config()
    path = manifest_path(config, name)
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {path}\nRun `python demo/prepare_data.py` first."
        )
    rows = load_manifest_rows(path)
    root = extract_root(config)
    for row in rows:
        row["image_path"] = str((root / row["rel_path"]).resolve())
        row["source_dataset"] = "demo_fundus_v03"
        row["original_label"] = row.get("source_class", "")
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", help="Path to the fundus dataset ZIP (default: auto-discovery).")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to demo/config.json.")
    parser.add_argument("--force-extract", action="store_true", help="Re-extract files that already exist.")
    parser.add_argument("--skip-crc", action="store_true", help="Skip the full-archive CRC test (faster).")
    args = parser.parse_args(argv)

    config = load_demo_config(args.config)
    archive = resolve_archive(config, args.zip)
    if not archive.is_file():
        raise FileNotFoundError(f"Dataset archive is not a readable file: {archive}")

    print(f"Dataset archive : {archive}")
    print(f"Archive size    : {archive.stat().st_size / 1024**3:.2f} GiB")

    try:
        zf = zipfile.ZipFile(archive)
    except zipfile.BadZipFile as exc:
        raise SystemExit(f"The dataset ZIP is not readable as a ZIP archive: {exc}") from exc

    with zf:
        info = inspect_archive(zf, config)
        if not args.skip_crc:
            print("Checking archive integrity (CRC over every member)...")
            bad = zf.testzip()
            if bad is not None:
                raise SystemExit(f"Archive is corrupt: CRC check failed on {bad!r}")
            print("Archive integrity: OK")

        destination = extract_root(config)
        safe_extract(zf, destination, force=args.force_extract)

    rows, summary = verify_and_inventory(destination, config)
    duplicates = find_duplicates(rows)

    print()
    print("Usable images per class (after decode check and duplicate removal from the pools)")
    duplicate_hashes = set(duplicates)
    for label, name in enumerate(config["taxonomy"]["classes"]):
        usable = sum(1 for r in rows if int(r["mapped_label"]) == label)
        unique = sum(1 for r in rows if int(r["mapped_label"]) == label and r["sha256"] not in duplicate_hashes)
        sources = sorted({r["source_class"] for r in rows if int(r["mapped_label"]) == label})
        print(f"  [{label}] {name:<16} usable={usable:<4} unique={unique:<4} from {sources}")

    if summary["corrupt_images"]:
        print(f"  corrupt images excluded: {len(summary['corrupt_images'])}")
        for item in summary["corrupt_images"]:
            print(f"    - {item['rel_path']}: {item['error']}")
    else:
        print("  corrupt images excluded: 0")
    print(f"  duplicate files found  : {sum(len(m) - 1 for m in duplicates.values())}")

    selected, detail = select_splits(rows, duplicates, config)
    written = write_manifests(config, selected, rows, summary, duplicates, info, archive)

    n_fit = sum(1 for r in selected if r["used_for"] == "fit")
    n_val = sum(1 for r in selected if r["used_for"] == "validation")
    n_test = sum(1 for r in selected if r["used_for"] == "test")
    print()
    print(f"Selected: {n_fit} fit + {n_val} validation = 50 training images, {n_test} test images")
    for label in sorted(detail, key=int):
        entry = detail[label]
        print(
            f"  [{label}] {config['taxonomy']['classes'][int(label)]:<16} "
            f"fit={entry['fit']['count']:<3} val={entry['validation']['count']:<3} test={entry['test']['count']:<3}"
        )
    print()
    print("Manifests written:")
    for key, path in written.items():
        print(f"  {key:<10} {Path(path).relative_to(REPO_ROOT).as_posix()}")
    print()
    print("Next step: python demo/train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
