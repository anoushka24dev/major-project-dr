"""Mendeley Data acquisition (dataset ``6ry9n79gb4``, DOI 10.17632/6ry9n79gb4.1).

Access model, verified against the live API on 2026-11-03
---------------------------------------------------------
The anonymous public API is used, exactly as the Mendeley Data website does it:

    GET {api_base}/{dataset_id}?version={n}
    GET {api_base}/{dataset_id}/files?folder_id=root&version={n}

For this record the second call currently returns `[]` and the metadata reports
``"size": 0``: the record exposes **no deposited files**, and the dataset page
renders an empty file list with "Download All" disabled. This is a provider-side
state, not a bug in the request: the same endpoint returns populated listings for
other public datasets.

Consequently this module never claims a download that did not happen. It:

1. always records the live API response in the download manifest;
2. uses an already-present local copy of the corpus when one is configured, and
   labels its provenance honestly as a local copy rather than a download;
3. otherwise returns ``blocked`` with the exact destination folder and the exact
   manual step required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import Config
from .base import (
    DownloadError,
    DownloadManifest,
    DownloadResult,
    count_files,
    download_file,
    extract_archive,
    http_json,
    require_free_space,
)

SOURCE = "mendeley"


def _metadata(cfg: Config, logger) -> Dict[str, Any]:
    base = str(cfg.require("datasets.mendeley.api_base")).rstrip("/")
    dataset_id = str(cfg.require("datasets.mendeley.dataset_id"))
    version = int(cfg.require("datasets.mendeley.version"))
    meta = http_json(f"{base}/{dataset_id}?version={version}", logger=logger)
    return meta if isinstance(meta, dict) else {}


def _file_listing(cfg: Config, logger) -> List[Dict[str, Any]]:
    base = str(cfg.require("datasets.mendeley.api_base")).rstrip("/")
    dataset_id = str(cfg.require("datasets.mendeley.dataset_id"))
    version = int(cfg.require("datasets.mendeley.version"))
    listing = http_json(
        f"{base}/{dataset_id}/files?folder_id=root&version={version}", logger=logger
    )
    return listing if isinstance(listing, list) else []


def _class_folders_present(root: Path, expected: List[str]) -> List[str]:
    return [name for name in expected if (root / name).is_dir()]


def _looks_like_corpus(root: Path, expected: List[str]) -> bool:
    """True when at least half the expected class folders and some images exist."""
    if not root or not root.is_dir():
        return False
    found = _class_folders_present(root, expected)
    if len(found) < max(1, len(expected) // 2):
        return False
    return count_files(root) > 0


def run(
    cfg: Config,
    logger,
    manifest: DownloadManifest,
    *,
    fundus_root: Optional[Path] = None,
    force: bool = False,
) -> DownloadResult:
    """Acquire the Mendeley corpus, or report precisely why it is unavailable."""
    expected = [str(x) for x in (cfg.get("datasets.mendeley.expected_class_folders") or [])]
    target_dir = cfg.path("paths.mendeley_dir", create=True)
    cache_dir = cfg.path("paths.cache_dir", create=True) / SOURCE

    metadata: Dict[str, Any] = {}
    listing: List[Dict[str, Any]] = []
    provider_error: Optional[str] = None
    try:
        metadata = _metadata(cfg, logger)
        listing = _file_listing(cfg, logger)
        manifest.add(
            "metadata",
            source=SOURCE,
            url=cfg.require("datasets.mendeley.api_base"),
            dataset_id=cfg.require("datasets.mendeley.dataset_id"),
            version=cfg.require("datasets.mendeley.version"),
            api_size=metadata.get("size"),
            api_file_count=len(listing),
        )
    except Exception as exc:
        provider_error = f"{type(exc).__name__}: {exc}"
        logger.warning("Mendeley API unavailable: %s", provider_error)
        manifest.add("metadata_failed", source=SOURCE, error=provider_error)

    meta_summary = {
        "dataset_id": cfg.get("datasets.mendeley.dataset_id"),
        "version": cfg.get("datasets.mendeley.version"),
        "doi": cfg.get("datasets.mendeley.doi"),
        "licence": cfg.get("datasets.mendeley.licence"),
        "api_name": metadata.get("name"),
        "api_size_bytes": metadata.get("size"),
        "api_file_count": len(listing),
        "provider_error": provider_error,
    }

    # 1. Files published by the provider -> download them properly.
    if listing:
        logger.info("Mendeley API lists %d file(s); downloading", len(listing))
        downloaded: List[str] = []
        total_bytes = 0
        for entry in listing:
            details = entry.get("content_details") or {}
            url = details.get("download_url")
            name = entry.get("filename") or Path(str(url or "")).name
            size = details.get("size") or entry.get("size")
            if not url:
                logger.warning("skipping %s: no download_url in API response", name)
                continue
            total_bytes += int(size or 0)
            try:
                require_free_space(target_dir, int(size or 0))
                dest = download_file(
                    url,
                    cache_dir / str(name),
                    expected_sha256=details.get("sha256_hash"),
                    expected_size=int(size) if size else None,
                    chunk_size=int(cfg.get("download.chunk_size", 1048576)),
                    timeout=float(cfg.get("download.timeout_seconds", 120)),
                    resume=bool(cfg.get("download.resume", True)),
                    max_retries=int(cfg.get("download.max_retries", 5)),
                    backoff=float(cfg.get("download.retry_backoff_seconds", 3)),
                    logger=logger,
                )
                manifest.add(
                    "download", source=SOURCE, file=str(dest), bytes=dest.stat().st_size, ok=True
                )
                downloaded.append(str(dest))
            except Exception as exc:
                manifest.add("download", source=SOURCE, file=str(name), ok=False, error=str(exc))
                logger.error("failed to download %s: %s", name, exc)

        for path_str in downloaded:
            path = Path(path_str)
            if path.suffix.lower() in {".zip", ".tar", ".gz", ".tgz"}:
                try:
                    extract_archive(path, target_dir, logger=logger)
                    manifest.add("extract", source=SOURCE, archive=str(path), destination=str(target_dir), ok=True)
                except Exception as exc:
                    manifest.add("extract", source=SOURCE, archive=str(path), ok=False, error=str(exc))
                    logger.error("failed to extract %s: %s", path, exc)

        if _looks_like_corpus(target_dir, expected):
            missing = set(expected) - set(_class_folders_present(target_dir, expected))
            return DownloadResult(
                source=SOURCE,
                status=DownloadResult.OK,
                detail=f"downloaded {len(downloaded)} file(s) from the Mendeley public API",
                root=target_dir,
                files=downloaded,
                metadata={**meta_summary, "missing_class_folders": sorted(missing)},
            )
        return DownloadResult(
            source=SOURCE,
            status=DownloadResult.FAILED,
            detail=(
                "files were downloaded but the expected class folders were not found "
                f"under {target_dir}"
            ),
            root=target_dir,
            files=downloaded,
            metadata=meta_summary,
        )

    # 2. No files published. Prefer an already-present local copy, honestly labelled.
    candidate_roots: List[Path] = []
    if fundus_root is not None:
        candidate_roots.append(Path(fundus_root))
    candidate_roots.append(target_dir)
    for candidate in candidate_roots:
        if _looks_like_corpus(candidate, expected):
            missing = sorted(set(expected) - set(_class_folders_present(candidate, expected)))
            logger.warning(
                "using a LOCAL COPY of the Mendeley corpus at %s (the Mendeley API currently "
                "publishes no files for this record)", candidate,
            )
            manifest.add(
                "local_copy_used",
                source=SOURCE,
                root=str(candidate),
                image_count=count_files(candidate),
                api_file_count=0,
            )
            note = (
                "LOCAL PRE-EXISTING COPY, not a download. The Mendeley public API reports "
                f"size={metadata.get('size')} and lists {len(listing)} file(s) for record "
                f"{cfg.get('datasets.mendeley.dataset_id')} v{cfg.get('datasets.mendeley.version')}."
            )
            return DownloadResult(
                source=SOURCE,
                status=DownloadResult.OK_LOCAL,
                detail=note,
                root=candidate,
                metadata={
                    **meta_summary,
                    "provenance": "local_pre_existing_copy",
                    "image_count": count_files(candidate),
                    "missing_class_folders": missing,
                },
            )

    # 3. Nothing usable anywhere: report the exact limitation and manual step.
    instructions = [
        f"The Mendeley record {cfg.get('datasets.mendeley.dataset_id')} "
        f"(v{cfg.get('datasets.mendeley.version')}, DOI {cfg.get('datasets.mendeley.doi')}) "
        "currently exposes no downloadable files through its public API.",
        f"Provider page : {cfg.get('datasets.mendeley.page_url')}",
        f"API response  : size={metadata.get('size')!r}, files=[] (HTTP 200)",
        f"Manual step   : open the provider page in a browser and use 'Download All' if it is "
        f"enabled there, then extract so that the 13 class folders sit directly inside:",
        f"                {target_dir}",
        "Expected folders: " + ", ".join(expected),
        "Alternative   : if you already have this corpus elsewhere, point the pipeline at it with",
        "                set DR_FUNDUS_ROOT=<path to the folder containing the 13 class folders>",
        "The pipeline will validate whatever is placed there before using it.",
    ]
    return DownloadResult(
        source=SOURCE,
        status=DownloadResult.BLOCKED,
        detail=(
            "Mendeley exposes no downloadable files for this record (API size=0, empty listing); "
            "no local copy was found either"
        ),
        root=target_dir,
        requirements=instructions,
        metadata={**meta_summary, "provenance": "none"},
    )
