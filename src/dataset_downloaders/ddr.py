"""DDR (OIA-DDR) acquisition.

Access model, verified against the published repository on 2026-11-05
---------------------------------------------------------------------
The label repository lives at https://github.com/nkicsl/DDR-dataset, but the
images themselves are **not** served by that repository: they are distributed as
a zip split into ten chunks hosted on Google Drive and Baidu Drive. Neither
endpoint exposes an anonymous, rclone-free HTTP download, and the dataset licence
is CC BY-NC-SA 4.0 (non-commercial).

This module therefore does what the project's provenance rules allow:

1. validates an already-supplied local DDR tree (``DR_DDR_ROOT`` or
   ``paths.ddr_dir``) against its real layout;
2. reports exactly what is missing when it is absent, with the manual steps;
3. never fabricates a download, never re-labels another dataset's images as DDR,
   and never claims success because a directory merely exists.

Expected layout (as shipped inside ``DDR-dataset.zip``)::

    <root>/
    └── DDR-dataset/
        └── DR_grading/
            ├── train.csv      id_code, diagnosis
            ├── valid.csv      id_code, diagnosis
            ├── test.csv       id_code, diagnosis
            └── <image files>  (one folder per official partition, or a flat set)

``diagnosis`` is the official DDR grade: 0-4 map verbatim onto the five-class
taxonomy, and 5 means *ungradable* and is excluded (never mapped to No DR).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import Config
from .base import (
    DownloadError,
    DownloadManifest,
    DownloadResult,
    count_files,
    find_dirs_named,
    join_split_zip,
    require_free_space,
    safe_extract_zip,
    verify_zip,
)
from .gdrive import DriveSession, download_file_from_drive, folder_id_of

SOURCE = "ddr"

#: Minimum free space worth checking before telling the user where to extract.
EXPECTED_MIN_GB = 8.0

#: Official public Drive folder that holds DDR-dataset.zip.001 ... .010.
OFFICIAL_DRIVE_FOLDER = "https://drive.google.com/drive/folders/1z6tSFmxW_aNayUqVxx6h6bY4kwGzUTEC"

#: The official archive is published as ten byte-split chunks.
EXPECTED_CHUNKS = 10

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def read_label_csv(path: Path) -> Dict[str, str]:
    """Return ``{id_code: diagnosis}`` for one official DDR label file.

    Column names are matched case-insensitively; a file without the documented
    columns is rejected rather than guessed at.
    """
    rows: Dict[str, str] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row")
        lowered = {str(name).strip().lower(): name for name in reader.fieldnames}
        id_column = lowered.get("id_code") or lowered.get("image") or lowered.get("filename")
        label_column = lowered.get("diagnosis") or lowered.get("grade") or lowered.get("label")
        if id_column is None or label_column is None:
            raise ValueError(
                f"{path} must contain an id column and a diagnosis column; found "
                f"{reader.fieldnames}"
            )
        for row in reader:
            identifier = str(row.get(id_column, "") or "").strip()
            if identifier:
                rows[identifier] = str(row.get(label_column, "") or "").strip()
    return rows


def locate_grading_dir(search_root: Path) -> Optional[Path]:
    """Find the ``DR_grading`` directory anywhere under ``search_root``."""
    root = Path(search_root)
    direct = root / "DR_grading"
    if direct.is_dir():
        return direct
    candidates = [
        path for path in find_dirs_named(root, "DR_grading") if path.is_dir()
    ]
    return candidates[0] if candidates else None


def validate_ddr(root: Path) -> Dict[str, Any]:
    """Validate a supplied DDR tree and report its official partitions.

    ``present`` is only true when the grading directory, all three label files and
    at least one labelled image are found.
    """
    info: Dict[str, Any] = {
        "root": str(root),
        "root_exists": Path(root).is_dir(),
        "present": False,
        "grading_dir": None,
        "label_files": {},
        "images": 0,
        "labelled_images": 0,
        "partitions": {},
        "ungradable_rows": 0,
    }
    if not info["root_exists"]:
        info["reason"] = f"DDR root does not exist: {root}"
        return info

    grading = locate_grading_dir(Path(root))
    if grading is None:
        info["reason"] = f"no DR_grading directory found under {root}"
        return info
    info["grading_dir"] = str(grading)

    label_tables: Dict[str, Dict[str, str]] = {}
    for partition in ("train", "valid", "test"):
        candidate = grading / f"{partition}.csv"
        if not candidate.is_file():
            info["reason"] = f"missing official DDR label file: {candidate}"
            return info
        try:
            table = read_label_csv(candidate)
        except Exception as exc:
            info["reason"] = f"unreadable DDR label file {candidate}: {exc}"
            return info
        label_tables[partition] = table
        info["label_files"][partition] = str(candidate)

    image_count = count_files(Path(root), _IMAGE_SUFFIXES)
    info["images"] = image_count
    if image_count == 0:
        info["reason"] = f"no fundus images found under {root}"
        return info

    # Count how many labelled rows actually have a matching image on disk.
    stems: set[str] = set()
    for path in Path(root).rglob("*"):
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            stems.add(path.stem.lower())

    labelled = 0
    partitions: Dict[str, int] = {}
    ungradable = 0
    for partition, table in label_tables.items():
        matched = 0
        for identifier, grade in table.items():
            if Path(identifier).stem.lower() in stems:
                matched += 1
                if grade.strip() == "5":
                    ungradable += 1
        partitions[partition] = matched
        labelled += matched
    info["partitions"] = partitions
    info["labelled_images"] = labelled
    info["ungradable_rows"] = ungradable

    if labelled == 0:
        info["reason"] = (
            "the DDR label files do not match any image on disk; the images are probably "
            "not extracted next to their CSV, or they are a different dataset"
        )
        return info

    info["present"] = True
    return info


def _instructions(root: Path, reason: str) -> List[str]:
    return [
        "DDR images are distributed as 10 byte-split zip chunks on Google Drive / Baidu "
        "Drive and are not anonymously downloadable over HTTP.",
        "  1. Open https://github.com/nkicsl/DDR-dataset and follow the drive links in its "
        "README.",
        "  2. Download DDR-dataset.zip.001 ... DDR-dataset.zip.010.",
        "  3. Recombine them:  cat DDR-dataset.zip.0* > DDR-dataset.zip   (Unix)   or "
        "copy /b DDR-dataset.zip.001+DDR-dataset.zip.002+... DDR-dataset.zip  (Windows).",
        "  4. Extract the archive so the DDR-dataset/DR_grading folder with train.csv, "
        f"valid.csv, test.csv and the images sits under: {root}",
        f"  5. Or point the pipeline at an existing copy: set DR_DDR_ROOT, or run "
        f'RUN.bat --ddr-root "PATH".',
        "  6. Re-run RUN.bat --resume; the loader validates the layout and label files "
        "before using anything.",
        "DDR is licensed CC BY-NC-SA 4.0: non-commercial research use only.",
        f"Current blocker: {reason}",
    ]


def _chunk_plan(files) -> List[Any]:
    """Return the official chunks in numeric order, rejecting anything else."""
    chunks = [item for item in files if item.part_index is not None]
    chunks.sort(key=lambda item: item.part_index)
    if not chunks:
        raise DownloadError(
            "the Drive folder does not contain DDR-dataset.zip.NNN chunks"
        )
    indexes = [item.part_index for item in chunks]
    if indexes != list(range(1, len(chunks) + 1)):
        raise DownloadError(f"DDR chunk numbering is not contiguous: {indexes}")
    return chunks


def _automatic_download(
    cfg: Config,
    logger,
    manifest: DownloadManifest,
    target: Path,
) -> DownloadResult:
    """Retrieve the ten official chunks from Drive, then join, verify and extract."""
    folder = str(cfg.get("datasets.ddr.drive_folder_url") or OFFICIAL_DRIVE_FOLDER)
    cache_dir = cfg.path("paths.cache_dir", create=True) / SOURCE
    session = DriveSession()

    try:
        files = session.list_folder(folder_id_of(folder))
        chunks = _chunk_plan(files)
    except DownloadError as exc:
        manifest.add("drive_list_failed", source=SOURCE, error=str(exc))
        return DownloadResult(
            source=SOURCE,
            status=DownloadResult.BLOCKED,
            detail=f"could not resolve the official DDR Drive folder: {exc}",
            root=target,
            requirements=_instructions(target, str(exc)) + [f"Official folder: {folder}"],
            metadata={"drive_folder": folder, "auto_download_attempted": True},
        )

    expected_chunks = int(cfg.get("datasets.ddr.expected_chunks", EXPECTED_CHUNKS))
    if len(chunks) != expected_chunks:
        logger.warning(
            "DDR Drive folder lists %d chunk(s); the official release has %d",
            len(chunks), expected_chunks,
        )

    # Probe the sizes once, so the free-space decision is measured, not guessed.
    sizes: Dict[str, Optional[int]] = {}
    for chunk in chunks:
        sizes[chunk.name] = session.probe_size(chunk.id)
    known_total = sum(size or 0 for size in sizes.values())
    if known_total:
        try:
            require_free_space(target, int(known_total * 2.2), logger=logger)
        except DownloadError as exc:
            manifest.add("space_check_failed", source=SOURCE, error=str(exc))
            return DownloadResult(
                source=SOURCE,
                status=DownloadResult.BLOCKED,
                detail=str(exc),
                root=target,
                requirements=[f"Free roughly {known_total * 2.2 / 1024**3:.0f} GB and retry"],
            )

    downloaded: List[Path] = []
    for chunk in chunks:
        destination = cache_dir / chunk.name
        try:
            path = download_file_from_drive(
                chunk.id,
                destination,
                session=session,
                expected_size=sizes.get(chunk.name),
                chunk_size=int(cfg.get("download.chunk_size", 1048576)),
                timeout=float(cfg.get("download.timeout_seconds", 120)),
                resume=bool(cfg.get("download.resume", True)),
                max_retries=int(cfg.get("download.max_retries", 5)),
                backoff=float(cfg.get("download.retry_backoff_seconds", 3)),
                logger=logger,
            )
            downloaded.append(path)
            manifest.add("download", source=SOURCE, file=str(path), bytes=path.stat().st_size, ok=True)
        except Exception as exc:
            manifest.add("download", source=SOURCE, file=chunk.name, ok=False, error=str(exc))
            logger.error("DDR chunk %s failed: %s", chunk.name, exc)
            return DownloadResult(
                source=SOURCE,
                status=DownloadResult.BLOCKED,
                detail=(
                    f"downloaded {len(downloaded)} of {len(chunks)} DDR chunks; the transfer "
                    f"stopped with: {exc}. Re-run to resume the remaining chunks."
                ),
                root=target,
                files=[str(p) for p in downloaded],
                requirements=_instructions(target, str(exc)) + [f"Official folder: {folder}"],
                metadata={"chunks_downloaded": len(downloaded), "chunks_expected": len(chunks)},
            )

    try:
        merged = join_split_zip(downloaded, cache_dir / "DDR-dataset.zip", logger=logger)
        verify_zip(merged)
        manifest.add("verify", source=SOURCE, archive=str(merged), ok=True)
        safe_extract_zip(merged, target, logger=logger)
        manifest.add("extract", source=SOURCE, archive=str(merged),
                     destination=str(target), ok=True)
    except Exception as exc:
        manifest.add("extract", source=SOURCE, ok=False, error=str(exc))
        return DownloadResult(
            source=SOURCE,
            status=DownloadResult.FAILED,
            detail=f"DDR archive reconstruction/extraction failed: {exc}",
            root=target,
            files=[str(p) for p in downloaded],
        )

    layout = validate_ddr(target)
    if not layout.get("present"):
        return DownloadResult(
            source=SOURCE,
            status=DownloadResult.FAILED,
            detail=(
                "the DDR archive was downloaded and extracted but its label files do not "
                f"match the images: {layout.get('reason')}"
            ),
            root=target,
            metadata={"layout": layout},
        )
    return DownloadResult(
        source=SOURCE,
        status=DownloadResult.OK,
        detail=(
            f"downloaded {len(downloaded)} official DDR chunk(s) "
            f"({known_total / 1024**3:.1f} GB), joined, verified and extracted"
        ),
        root=target,
        files=[str(p) for p in downloaded],
        metadata={
            "layout": layout,
            "provenance": "downloaded from the official Google Drive folder",
            "chunks": len(downloaded),
            "bytes": known_total,
        },
    )


def run(
    cfg: Config,
    logger,
    manifest: DownloadManifest,
    *,
    root: Optional[Path] = None,
    skip_download: bool = False,
) -> DownloadResult:
    """Acquire DDR automatically, or validate a supplied copy, or report the blocker."""
    target = Path(root) if root else cfg.ddr_root()
    layout = validate_ddr(target)

    manifest.add(
        "inspect",
        source=SOURCE,
        root=str(target),
        present=layout.get("present"),
        images=layout.get("images"),
        labelled_images=layout.get("labelled_images"),
        partitions=layout.get("partitions"),
        auto_download=not skip_download,
        drive_folder=str(cfg.get("datasets.ddr.drive_folder_url") or OFFICIAL_DRIVE_FOLDER),
    )

    if layout.get("present"):
        logger.info(
            "DDR already present: %d labelled images (%s) under %s",
            layout.get("labelled_images", 0),
            layout.get("partitions"),
            target,
        )
        return DownloadResult(
            source=SOURCE,
            status=DownloadResult.OK_ALREADY_PRESENT,
            detail=(
                f"DDR validated: {layout.get('labelled_images', 0)} labelled images "
                f"across official partitions {layout.get('partitions')}"
            ),
            root=target,
            metadata={
                "layout": layout,
                "provenance": "manually supplied (Google Drive / Baidu Drive chunks)",
            },
        )

    if skip_download:
        logger.info("DDR download skipped by request (--skip-download)")
        return DownloadResult(
            source=SOURCE,
            status=DownloadResult.SKIPPED,
            detail="DDR download skipped by request (--skip-download)",
            root=target,
            metadata={"layout": layout},
        )

    if bool(cfg.get("datasets.ddr.auto_download", True)):
        logger.info("DDR not present locally; attempting the official Google Drive folder")
        return _automatic_download(cfg, logger, manifest, target)

    reason = str(layout.get("reason") or "DDR data not found")
    logger.warning("DDR unavailable: %s", reason)
    return DownloadResult(
        source=SOURCE,
        status=DownloadResult.BLOCKED,
        detail=reason,
        root=target,
        requirements=_instructions(target, reason),
        metadata={"layout": layout, "requires_manual_download": True},
    )
