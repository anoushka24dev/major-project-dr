"""IDRiD acquisition (Indian Diabetic Retinopathy Image Dataset).

Access model, verified on 2026-11-03
------------------------------------
Both official hosts gate the download behind registration and terms acceptance:

* IEEE DataPort  https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
* Grand Challenge https://idrid.grand-challenge.org/

No anonymous download endpoint exists. Access restrictions are **not** bypassed
here. This module therefore:

1. detects whether the user has already supplied IDRiD;
2. validates the exact folder layout the segmentation stage needs, including
   1:1 image/mask pairing;
3. prints precise, actionable instructions when something is missing;
4. lets the pipeline resume automatically once the files appear.

IDRiD layout expected (as published):

    idrid/
      1. Original Images/
        a. Training Set/IDRiD_01.jpg ...
        b. Testing Set/IDRiD_81.jpg ...          (optional here)
      2. All Segmentation Groundtruths/
        a. Training Set/
          1. Microaneurysms/IDRiD_01_MA.tif ...
          2. Haemorrhages/IDRiD_01_HE.tif ...
          3. Hard Exudates/IDRiD_01_EX.tif ...
          4. Soft Exudates/IDRiD_01_SE.tif ...
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import Config
from .base import (
    DownloadManifest,
    DownloadResult,
    count_files,
    download_file,
    extract_archive,
    find_dirs_named,
)

SOURCE = "idrid"

#: Lesion sub-folder name -> channel code, in the fixed channel order.
LESION_SUBDIRS = {
    "1. Microaneurysms": "MA",
    "2. Haemorrhages": "HE",
    "3. Hard Exudates": "EX",
    "4. Soft Exudates": "SE",
}

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def _resolve(root: Path, relative: str) -> Optional[Path]:
    """Locate a required relative path, tolerating a single wrapper directory."""
    direct = root / relative
    if direct.is_dir():
        return direct
    if not root.is_dir():
        return None
    # Tolerate one extra wrapper folder, e.g. idrid/IDRiD/1. Original Images/...
    for child in sorted(root.iterdir()):
        if child.is_dir():
            nested = child / relative
            if nested.is_dir():
                return nested
    return None


def _find_any(root: Path, name: str) -> Optional[Path]:
    direct = root / name
    if direct.is_dir():
        return direct
    found = find_dirs_named(root, name, max_depth=5)
    return found[0] if found else None


def discover_layout(root: Path) -> Dict[str, Any]:
    """Inspect the IDRiD tree and report what is present or missing."""
    root = Path(root)
    report: Dict[str, Any] = {
        "root": str(root),
        "root_exists": root.is_dir(),
        "images_dir": None,
        "lesion_dirs": {},
        "missing": [],
        "image_count": 0,
        "lesion_counts": {},
    }
    if not report["root_exists"]:
        report["missing"].append(f"IDRiD root does not exist: {root}")
        for folder, code in LESION_SUBDIRS.items():
            report["lesion_dirs"][code] = None
        return report

    images_dir = _find_any(root, "a. Training Set")
    if images_dir is not None and not any(
        p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES for p in images_dir.iterdir()
    ):
        # "a. Training Set" also exists under the ground-truths tree; prefer the
        # one that actually contains images.
        images_dir = None
    if images_dir is None:
        for candidate in sorted(root.rglob("a. Training Set")):
            if candidate.is_dir() and any(
                p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES for p in candidate.iterdir()
            ):
                images_dir = candidate
                break
    report["images_dir"] = str(images_dir) if images_dir else None
    if images_dir is None:
        report["missing"].append(
            "could not find the training images folder ('1. Original Images/a. Training Set')"
        )
    else:
        report["image_count"] = count_files(images_dir, IMAGE_SUFFIXES)

    groundtruth_root = _find_any(root, "a. Training Set")
    for folder, code in LESION_SUBDIRS.items():
        found = None
        for candidate in root.rglob(folder):
            if candidate.is_dir() and any(
                p.suffix.lower() in IMAGE_SUFFIXES for p in candidate.iterdir()
            ):
                found = candidate
                break
        report["lesion_dirs"][code] = str(found) if found else None
        if found is None:
            report["missing"].append(
                f"missing lesion annotation folder {folder!r} (channel {code})"
            )
        else:
            report["lesion_counts"][code] = count_files(found, IMAGE_SUFFIXES)

    return report


def _stems(directory: Optional[Path], pattern_suffix: str = "") -> List[str]:
    if directory is None:
        return []
    out: List[str] = []
    for path in sorted(Path(directory).iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        stem = path.stem
        if pattern_suffix and stem.upper().endswith(pattern_suffix.upper()):
            stem = stem[: -len(pattern_suffix)]
        out.append(stem)
    return out


def pairing_report(layout: Dict[str, Any]) -> Dict[str, Any]:
    """Verify that every image has one mask per lesion channel and vice versa."""
    images_dir = Path(layout["images_dir"]) if layout.get("images_dir") else None
    image_stems = set(_stems(images_dir)) if images_dir else set()
    result: Dict[str, Any] = {
        "image_count": len(image_stems),
        "per_channel": {},
        "missing_masks": {},
        "orphan_masks": {},
        "unpaired": bool(image_stems) and False,
    }
    if not image_stems:
        result["unpaired"] = True
        result["reason"] = "no images found"
        return result

    ok = True
    for code, directory in (layout.get("lesion_dirs") or {}).items():
        if not directory:
            result["per_channel"][code] = 0
            result["missing_masks"][code] = len(image_stems)
            ok = False
            continue
        mask_stems = set(_stems(Path(directory), pattern_suffix=f"_{code}"))
        missing = sorted(image_stems - mask_stems)
        orphans = sorted(mask_stems - image_stems)
        result["per_channel"][code] = len(mask_stems)
        if missing:
            result["missing_masks"][code] = len(missing)
            ok = False
        if orphans:
            result["orphan_masks"][code] = len(orphans)
    result["unpaired"] = not ok
    return result


def manual_instructions(cfg: Config, target: Path) -> List[str]:
    """Exact, actionable steps for a human who must supply IDRiD."""
    return [
        "IDRiD is not downloadable without an account: both official hosts require "
        "registration and acceptance of their terms.",
        f"  1. Register at {cfg.get('datasets.idrid.page_url')}",
        f"     (mirror: {cfg.get('datasets.idrid.mirror_url')})",
        "  2. Accept the dataset terms and download the archive(s) for:",
        "       - '1. Original Images'      (training set images)",
        "       - '2. All Segmentation Groundtruths'  (MA, HE, EX, SE annotations)",
        f"  3. Extract so the two numbered folders sit directly inside: {target}",
        "     Expected layout:",
        "       <IDRiD root>/1. Original Images/a. Training Set/IDRiD_01.jpg",
        "       <IDRiD root>/2. All Segmentation Groundtruths/a. Training Set/"
        "1. Microaneurysms/IDRiD_01_MA.tif",
        "       <IDRiD root>/2. All Segmentation Groundtruths/a. Training Set/"
        "2. Haemorrhages/IDRiD_01_HE.tif",
        "       <IDRiD root>/2. All Segmentation Groundtruths/a. Training Set/"
        "3. Hard Exudates/IDRiD_01_EX.tif",
        "       <IDRiD root>/2. All Segmentation Groundtruths/a. Training Set/"
        "4. Soft Exudates/IDRiD_01_SE.tif",
        "  4. Re-run RUN.bat (or `RUN.bat --resume`); the pipeline validates the pairing "
        "and continues automatically.",
        f"  Alternatively set DR_IDRID_ROOT to an existing extracted copy.",
        "  Or, after signing in, start the download in the browser and copy the archive "
        "URL(s) and the session Cookie header into the environment, then re-run:",
        "      set DR_IDRID_URLS=<space-separated archive urls>",
        "      set DR_IDRID_COOKIE=<cookie header value>",
        "  Both values stay in your environment: they are never written to config, "
        "logs or the repository.",
    ]


def authenticated_sources() -> Dict[str, Any]:
    """Read the optional authenticated-download configuration from the environment.

    IEEE DataPort serves the archives only to a signed-in session. Rather than
    storing or automating a password, the pipeline accepts the *already resolved*
    archive URLs and the session cookie the user obtained after signing in:

        DR_IDRID_URLS    space-separated direct archive URLs
        DR_IDRID_COOKIE  the Cookie header value for that session

    Both are read from the environment only, never written to the repository or a
    log line, and the cookie is redacted from every message.
    """
    raw_urls = str(os.environ.get("DR_IDRID_URLS") or "").strip()
    urls = [part for part in raw_urls.split() if part]
    cookie = str(os.environ.get("DR_IDRID_COOKIE") or "").strip()
    return {"urls": urls, "cookie": cookie, "configured": bool(urls)}


def _download_authenticated(
    cfg: Config,
    logger,
    manifest: DownloadManifest,
    target: Path,
    sources: Dict[str, Any],
) -> List[Path]:
    """Download the configured IDRiD archives with the user's session cookie."""
    cache_dir = cfg.path("paths.cache_dir", create=True) / SOURCE
    headers = {"Cookie": sources["cookie"]} if sources.get("cookie") else {}
    downloaded: List[Path] = []
    for index, url in enumerate(sources["urls"]):
        name = Path(url.split("?")[0]).name or f"idrid_archive_{index + 1}.zip"
        try:
            path = download_file(
                url,
                cache_dir / name,
                chunk_size=int(cfg.get("download.chunk_size", 1048576)),
                timeout=float(cfg.get("download.timeout_seconds", 120)),
                resume=bool(cfg.get("download.resume", True)),
                max_retries=int(cfg.get("download.max_retries", 5)),
                backoff=float(cfg.get("download.retry_backoff_seconds", 3)),
                logger=logger,
                headers=headers,
            )
        except Exception as exc:
            manifest.add("download", source=SOURCE, url=url.split("?")[0], ok=False, error=str(exc))
            logger.error("IDRiD archive failed: %s", exc)
            continue
        manifest.add("download", source=SOURCE, file=str(path), bytes=path.stat().st_size, ok=True)
        downloaded.append(path)
        if path.suffix.lower() in {".zip", ".tar", ".gz", ".tgz"}:
            try:
                extract_archive(path, target, logger=logger)
                manifest.add("extract", source=SOURCE, archive=str(path), ok=True)
            except Exception as exc:
                manifest.add("extract", source=SOURCE, archive=str(path), ok=False, error=str(exc))
                logger.error("IDRiD extraction failed: %s", exc)
    return downloaded


def run(
    cfg: Config,
    logger,
    manifest: DownloadManifest,
    *,
    root: Optional[Path] = None,
    skip_download: bool = False,
) -> DownloadResult:
    """Download authorized IDRiD archives, validate a supplied copy, or report the blocker."""
    target = Path(root) if root else cfg.idrid_root()
    layout = discover_layout(target)
    source_config = authenticated_sources()
    manifest.add(
        "inspect",
        source=SOURCE,
        root=str(target),
        image_count=layout.get("image_count"),
        missing=layout.get("missing"),
        auto_download=bool(source_config["configured"]) and not skip_download,
        authenticated_urls=len(source_config["urls"]),
        cookie_present=bool(source_config["cookie"]),
        access_requirement="registration + terms acceptance",
    )

    if not layout["root_exists"] and source_config["configured"] and not skip_download:
        logger.info("IDRiD: %d authenticated archive URL(s) configured; downloading",
                    len(source_config["urls"]))
        downloaded = _download_authenticated(cfg, logger, manifest, target, source_config)
        layout = discover_layout(target)
        if not downloaded:
            logger.error("every configured IDRiD archive failed to download")
        elif not layout.get("image_count"):
            logger.error("IDRiD archives downloaded but no images were found after extraction")

    if not layout["root_exists"]:
        logger.warning("IDRiD root does not exist yet: %s", target)
        return DownloadResult(
            source=SOURCE,
            status=DownloadResult.BLOCKED,
            detail=(
                "IDRiD requires an authenticated download; no files are present at the "
                "expected location and no DR_IDRID_URLS were configured"
            ),
            root=target,
            requirements=manual_instructions(cfg, target),
            metadata={"layout": layout, "requires_registration": True},
        )

    if layout["missing"] or not layout.get("image_count"):
        detail = "IDRiD tree is present but incomplete: " + "; ".join(layout["missing"][:6])
        logger.warning(detail)
        return DownloadResult(
            source=SOURCE,
            status=DownloadResult.BLOCKED,
            detail=detail,
            root=target,
            requirements=manual_instructions(cfg, target),
            metadata={"layout": layout, "requires_registration": True},
        )

    pairing = pairing_report(layout)
    if pairing.get("unpaired"):
        detail = (
            "IDRiD images and lesion annotations are not paired 1:1 "
            f"(missing masks: {pairing.get('missing_masks')}, "
            f"orphan masks: {pairing.get('orphan_masks')})"
        )
        logger.error(detail)
        return DownloadResult(
            source=SOURCE,
            status=DownloadResult.FAILED,
            detail=detail,
            root=target,
            requirements=[
                "Every image needs one annotation per lesion channel (MA, HE, EX, SE). "
                "Check that all four ground-truth folders were downloaded completely.",
            ],
            metadata={"layout": layout, "pairing": pairing},
        )

    logger.info(
        "IDRiD validated: %d images with paired MA/HE/EX/SE annotations",
        pairing.get("image_count", 0),
    )
    return DownloadResult(
        source=SOURCE,
        status=DownloadResult.OK_ALREADY_PRESENT,
        detail=(
            f"IDRiD validated: {pairing.get('image_count', 0)} images with 1:1 lesion masks "
            "for MA, HE, EX, SE"
        ),
        root=target,
        metadata={
            "layout": layout,
            "pairing": pairing,
            "provenance": "manually supplied (registration required)",
            "requires_registration": True,
        },
    )
