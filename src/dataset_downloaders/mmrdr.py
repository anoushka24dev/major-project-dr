"""MMRDR acquisition (Figshare article 29423747).

Access model, verified against the live API on 2026-11-03
---------------------------------------------------------
Figshare serves this article anonymously. The API reports 10 files:

    MMRDR.zip.001 ... MMRDR.zip.008   2,097,152,000 bytes each
    MMRDR.zip.009                     1,833,101,731 bytes
    README.md                         2,459 bytes

i.e. a ~18.7 GB byte-split zip. The README (fetched and read) documents the
layout and is the authority for the label mapping used by the rest of the
pipeline:

    MMRDR/MMRDR-CFP/img/{tr,ts}*.jpg    + MMRDR/MMRDR-CFP/FP.csv
    MMRDR/MMRDR-OCT/...                 (OCT, DME grades 0-2  -> excluded)
    MMRDR/MMRDR-UWF/...                 (ultra-widefield       -> excluded)

``FP.csv`` carries ``grade`` in 0..4 meaning
No DR / Mild NPDR / Moderate NPDR / Severe NPDR / PDR — a verbatim match for the
project's five-class taxonomy — plus ``lr`` for laterality.

Only the CFP subset is used for the fundus classifier. The official ``tr``/``ts``
filename prefixes are preserved: ``ts`` images become an ``external_test``
partition and are never trained on.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import Config
from .base import (
    DownloadManifest,
    DownloadResult,
    count_files,
    download_file,
    find_dirs_named,
    http_json,
    join_split_zip,
    require_free_space,
    safe_extract_zip,
)

SOURCE = "mmrdr"

_PART_RE = re.compile(r"^(?P<stem>.+)\.zip\.(?P<index>\d{3})$", re.IGNORECASE)


def fetch_article(cfg: Config, logger) -> Dict[str, Any]:
    """Return the Figshare article metadata."""
    base = str(cfg.require("datasets.mmrdr.api_base")).rstrip("/")
    article_id = int(cfg.require("datasets.mmrdr.article_id"))
    meta = http_json(f"{base}/{article_id}", logger=logger)
    return meta if isinstance(meta, dict) else {}


def classify_files(article: Dict[str, Any]) -> Dict[str, Any]:
    """Split the article file list into archive parts and other files."""
    files = article.get("files") or []
    parts: List[Dict[str, Any]] = []
    others: List[Dict[str, Any]] = []
    for entry in files:
        name = str(entry.get("name") or "")
        match = _PART_RE.match(name)
        if match:
            parts.append(
                {
                    "name": name,
                    "index": int(match.group("index")),
                    "size": int(entry.get("size") or 0),
                    "download_url": entry.get("download_url"),
                    "id": entry.get("id"),
                }
            )
        else:
            others.append(
                {
                    "name": name,
                    "size": int(entry.get("size") or 0),
                    "download_url": entry.get("download_url"),
                }
            )
    parts.sort(key=lambda p: p["index"])
    return {
        "parts": parts,
        "others": others,
        "total_bytes": sum(p["size"] for p in parts),
        "part_count": len(parts),
    }


def locate_cfp_root(search_root: Path) -> Optional[Path]:
    """Find the ``MMRDR-CFP`` directory anywhere under ``search_root``."""
    direct = Path(search_root) / "MMRDR-CFP"
    if direct.is_dir():
        return direct
    found = find_dirs_named(search_root, "MMRDR-CFP")
    return found[0] if found else None


def validate_cfp(cfp_root: Optional[Path]) -> Dict[str, Any]:
    """Check that the CFP subset has images and its annotation CSV."""
    if cfp_root is None or not cfp_root.is_dir():
        return {"present": False, "reason": "MMRDR-CFP directory not found"}
    img_dir = cfp_root / "img"
    csv_candidates = [p for p in cfp_root.glob("*.csv") if p.name.upper().startswith("FP")]
    images = count_files(img_dir) if img_dir.is_dir() else 0
    if images == 0:
        return {"present": False, "reason": f"no CFP images under {img_dir}"}
    if not csv_candidates:
        return {"present": False, "reason": f"no FP.csv under {cfp_root}"}
    return {
        "present": True,
        "img_dir": str(img_dir),
        "csv": str(sorted(csv_candidates)[0]),
        "image_count": images,
    }


def run(
    cfg: Config,
    logger,
    manifest: DownloadManifest,
    *,
    skip_download: bool = False,
) -> DownloadResult:
    """Acquire MMRDR and validate the CFP subset."""
    mmrdr_dir = cfg.path("paths.mmrdr_dir", create=True)
    cache_dir = cfg.path("paths.cache_dir", create=True) / SOURCE

    existing_cfp = locate_cfp_root(mmrdr_dir)
    existing_check = validate_cfp(existing_cfp)
    if existing_check.get("present") and existing_check.get("image_count", 0) > 0:
        logger.info("MMRDR CFP subset already present at %s", existing_cfp)
        manifest.add("already_present", source=SOURCE, root=str(existing_cfp),
                     image_count=existing_check.get("image_count"))
        return DownloadResult(
            source=SOURCE,
            status=DownloadResult.OK_ALREADY_PRESENT,
            detail="MMRDR-CFP subset already extracted and validated",
            root=Path(existing_cfp) if existing_cfp else mmrdr_dir,
            metadata={"cfp": existing_check},
        )

    try:
        article = fetch_article(cfg, logger)
    except Exception as exc:
        logger.error("MMRDR metadata request failed: %s", exc)
        manifest.add("metadata_failed", source=SOURCE, error=str(exc))
        return DownloadResult(
            source=SOURCE,
            status=DownloadResult.BLOCKED,
            detail=f"could not reach the Figshare API ({type(exc).__name__}: {exc})",
            root=mmrdr_dir,
            requirements=[
                f"Figshare article: {cfg.get('datasets.mmrdr.page_url')}",
                f"Manual step: download MMRDR.zip.001..009 and README.md, concatenate the parts "
                f"in order into one zip, extract it, and place the resulting MMRDR folder under {mmrdr_dir}",
            ],
        )

    plan = classify_files(article)
    total_gb = plan["total_bytes"] / 1024**3
    logger.info(
        "MMRDR: %d archive part(s), %.2f GB total, title=%r",
        plan["part_count"], total_gb, article.get("title"),
    )
    manifest.add(
        "metadata",
        source=SOURCE,
        article_id=cfg.require("datasets.mmrdr.article_id"),
        title=article.get("title"),
        part_count=plan["part_count"],
        total_bytes=plan["total_bytes"],
    )

    if skip_download:
        return DownloadResult(
            source=SOURCE,
            status=DownloadResult.SKIPPED,
            detail="MMRDR download skipped by request (--skip-download)",
            root=mmrdr_dir,
            metadata={"part_count": plan["part_count"], "total_bytes": plan["total_bytes"]},
        )

    expected_parts = int(cfg.get("datasets.mmrdr.expected_parts", 9))
    if plan["part_count"] and plan["part_count"] != expected_parts:
        logger.warning(
            "Figshare now publishes %d parts (config expects %d); continuing with what exists",
            plan["part_count"], expected_parts,
        )

    if not plan["parts"]:
        return DownloadResult(
            source=SOURCE,
            status=DownloadResult.BLOCKED,
            detail="the Figshare article lists no MMRDR.zip.* archive parts",
            root=mmrdr_dir,
            requirements=[f"Check {cfg.get('datasets.mmrdr.page_url')} for the current file list."],
        )

    # A split zip must be joined AND extracted: budget for both.
    try:
        require_free_space(mmrdr_dir, plan["total_bytes"] * 2, logger=logger)
    except Exception as exc:
        manifest.add("space_check_failed", source=SOURCE, error=str(exc))
        return DownloadResult(
            source=SOURCE,
            status=DownloadResult.BLOCKED,
            detail=str(exc),
            root=mmrdr_dir,
            requirements=[
                f"MMRDR needs roughly {total_gb * 2:.0f} GB free (download + reconstructed zip).",
            ],
        )

    downloaded: List[Path] = []
    for part in plan["parts"]:
        url = part.get("download_url")
        if not url:
            logger.warning("part %s has no download_url; skipping", part["name"])
            continue
        try:
            path = download_file(
                url,
                cache_dir / part["name"],
                expected_size=part["size"] or None,
                chunk_size=int(cfg.get("download.chunk_size", 1048576)),
                timeout=float(cfg.get("download.timeout_seconds", 120)),
                resume=bool(cfg.get("download.resume", True)),
                max_retries=int(cfg.get("download.max_retries", 5)),
                backoff=float(cfg.get("download.retry_backoff_seconds", 3)),
                logger=logger,
            )
            downloaded.append(path)
            manifest.add("download", source=SOURCE, file=str(path),
                         bytes=path.stat().st_size, ok=True)
        except Exception as exc:
            manifest.add("download", source=SOURCE, file=part["name"], ok=False, error=str(exc))
            logger.error("failed to download %s: %s", part["name"], exc)

    if len(downloaded) != plan["part_count"]:
        return DownloadResult(
            source=SOURCE,
            status=DownloadResult.FAILED,
            detail=(
                f"downloaded {len(downloaded)} of {plan['part_count']} archive parts; "
                "re-run to resume the remaining parts"
            ),
            root=mmrdr_dir,
            files=[str(p) for p in downloaded],
            metadata={"part_count": plan["part_count"], "total_bytes": plan["total_bytes"]},
        )

    try:
        merged = join_split_zip(downloaded, cache_dir / "MMRDR.zip", logger=logger)
        manifest.add("join", source=SOURCE, archive=str(merged), ok=True)
        safe_extract_zip(merged, mmrdr_dir, logger=logger)
        manifest.add("extract", source=SOURCE, archive=str(merged),
                     destination=str(mmrdr_dir), ok=True)
    except Exception as exc:
        manifest.add("extract", source=SOURCE, ok=False, error=str(exc))
        return DownloadResult(
            source=SOURCE,
            status=DownloadResult.FAILED,
            detail=f"archive reconstruction/extraction failed: {exc}",
            root=mmrdr_dir,
            files=[str(p) for p in downloaded],
        )

    cfp_root = locate_cfp_root(mmrdr_dir)
    check = validate_cfp(cfp_root)
    if not check.get("present"):
        return DownloadResult(
            source=SOURCE,
            status=DownloadResult.FAILED,
            detail=f"extraction finished but the CFP subset is unusable: {check.get('reason')}",
            root=mmrdr_dir,
            metadata={"cfp": check},
        )

    return DownloadResult(
        source=SOURCE,
        status=DownloadResult.OK,
        detail=f"downloaded {len(downloaded)} part(s) ({total_gb:.1f} GB), reconstructed and extracted",
        root=Path(cfp_root) if cfp_root else mmrdr_dir,
        files=[str(p) for p in downloaded],
        metadata={"cfp": check, "part_count": plan["part_count"], "total_bytes": plan["total_bytes"]},
    )
