"""Dataset acquisition orchestrator used by the download stage.

Responsibilities (Gate 1 of the workflow)
-----------------------------------------
1. check connectivity before spending time on transfers;
2. check free disk space before multi-gigabyte archives;
3. inspect what is already present, so a second run tranfers nothing again;
4. download whatever is missing, resuming partial transfers;
5. verify sizes/checksums, extract safely, validate the expected structure and
   labels per source (each adapter does its own verification);
6. write a machine-readable and human-readable availability report that names
   every blocked source with the exact reason and the exact next step.

The orchestrator never upgrades a status: a source is ``ok`` only if its adapter
validated real files, and one failing source never prevents the others from being
reported.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..config import Config
from .base import DownloadManifest, DownloadResult, count_files, free_space_bytes

#: Acquisition order. Cheap/authoritative sources first; the segmentation-only
#: source last, because classification can proceed without it.
SOURCE_ORDER: Sequence[str] = ("mendeley", "mmrdr", "aptos", "ddr", "idrid")

CLASSIFICATION_SOURCES = {"mendeley", "mmrdr", "aptos", "ddr"}

#: Connectivity probe: small, stable, and not a dataset host.
CONNECTIVITY_PROBES: Sequence[str] = (
    "https://api.figshare.com/v2/articles/29423747",
    "https://data.mendeley.com/public-api/datasets/6ry9n79gb4?version=1",
)


def check_connectivity(timeout: float = 8.0) -> Dict[str, Any]:
    """Return whether at least one provider host answers, and which ones failed."""
    reachable: List[str] = []
    failures: Dict[str, str] = {}
    for url in CONNECTIVITY_PROBES:
        try:
            request = urllib.request.Request(url, method="GET", headers={"User-Agent": "dr-pipeline/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response.read(1)
                reachable.append(url)
        except Exception as exc:  # network errors are expected here
            failures[url] = f"{type(exc).__name__}: {exc}"
    return {
        "online": bool(reachable),
        "reachable": reachable,
        "failed": failures,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def disk_report(cfg: Config) -> Dict[str, Any]:
    """Free space on the drive that will hold the datasets and the outputs."""
    datasets_dir = cfg.path("paths.datasets_dir", create=True)
    free = free_space_bytes(datasets_dir)
    return {
        "datasets_dir": str(datasets_dir),
        "free_bytes": free,
        "free_gb": round(free / 1024**3, 2) if free is not None else None,
        "required_gb_estimate": {
            # Documented, measured from each provider's published file list.
            "mmrdr": 37.0,          # 18.7 GB parts + reconstructed zip + extracted CFP
            "ddr": 16.0,            # 10 chunks + reconstructed zip
            "aptos": 5.0,
            "mendeley": 2.0,
            "idrid": 1.0,
        },
    }


def _call(source: str, cfg: Config, logger, manifest: DownloadManifest, *, skip_download: bool) -> DownloadResult:
    """Call one adapter with the arguments it supports."""
    if source == "mendeley":
        from . import mendeley as module

        return module.run(cfg, logger, manifest, fundus_root=cfg.fundus_root())
    if source == "mmrdr":
        from . import mmrdr as module

        return module.run(
            cfg, logger, manifest,
            root=cfg.mmrdr_root(),
            skip_download=skip_download,
        )
    if source == "aptos":
        from . import aptos as module

        return module.run(cfg, logger, manifest, skip_download=skip_download)
    if source == "ddr":
        from . import ddr as module

        return module.run(cfg, logger, manifest, skip_download=skip_download)
    if source == "idrid":
        from . import idrid as module

        return module.run(cfg, logger, manifest, skip_download=skip_download)
    raise ValueError(f"unknown dataset source: {source}")


def _source_root(cfg: Config, source: str) -> Optional[Path]:
    try:
        if source == "mendeley":
            return cfg.fundus_root() or cfg.path("paths.mendeley_dir")
        if source == "mmrdr":
            return cfg.mmrdr_root()
        if source == "aptos":
            return cfg.aptos_root()
        if source == "ddr":
            return cfg.ddr_root()
        if source == "idrid":
            return cfg.idrid_root()
    except Exception:
        return None
    return None


def dataset_inventory(cfg: Config) -> Dict[str, Any]:
    """Measured on-disk inventory for every source, independent of its status."""
    inventory: Dict[str, Any] = {}
    for source in SOURCE_ORDER:
        root = _source_root(cfg, source)
        entry: Dict[str, Any] = {"root": str(root) if root else None, "images": 0}
        if root is not None and Path(root).is_dir():
            entry["images"] = count_files(Path(root))
        inventory[source] = entry
    return inventory


def write_report(cfg: Config, payload: Dict[str, Any]) -> Dict[str, str]:
    """Write the availability report as JSON and Markdown."""
    manifests = cfg.outputs_dir("manifests")
    json_path = manifests / "download_report.json"
    json_path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")

    lines = [
        "# Dataset acquisition report",
        "",
        f"- Generated: {payload.get('generated')}",
        f"- Internet: {'reachable' if payload.get('connectivity', {}).get('online') else 'NOT reachable'}",
        f"- Free space: {payload.get('disk', {}).get('free_gb')} GB at {payload.get('disk', {}).get('datasets_dir')}",
        "",
        "| Source | Status | Images on disk | Detail |",
        "|---|---|---|---|",
    ]
    for source, result in (payload.get("sources") or {}).items():
        detail = str(result.get("detail") or "").replace("|", "/").replace("\n", " ")
        images = (payload.get("inventory", {}).get(source) or {}).get("images", 0)
        lines.append(f"| {source} | {result.get('status')} | {images} | {detail} |")

    blocked = [
        (name, result) for name, result in (payload.get("sources") or {}).items()
        if result.get("status") in {"blocked", "failed"}
    ]
    lines.extend(["", "## Required actions", ""])
    if not blocked:
        lines.append("None: every source is present and validated.")
    for name, result in blocked:
        lines.append(f"### {name} — {result.get('status')}")
        lines.append("")
        lines.append(str(result.get("detail") or ""))
        for requirement in result.get("requirements") or []:
            lines.append(f"- {requirement}")
        lines.append("")

    lines.extend([
        "## Gate 1",
        "",
        payload.get("gate1_note", ""),
        "",
    ])
    md_path = manifests / "dataset_availability.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def run_download_stage(
    cfg: Config,
    logger,
    events=None,
    *,
    skip_download: bool = False,
    only: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Acquire and validate every dataset, then report what is usable."""
    manifest = DownloadManifest(cfg.outputs_dir("manifests") / "download_manifest.json")
    connectivity = check_connectivity()
    disk = disk_report(cfg)

    logger.info(
        "connectivity: %s | free space at %s: %s GB",
        "online" if connectivity["online"] else "OFFLINE",
        disk["datasets_dir"],
        disk["free_gb"],
    )
    if events is not None:
        events.write("download_preflight", online=connectivity["online"],
                     free_gb=disk["free_gb"])

    selected = [source for source in SOURCE_ORDER if not only or source in set(only)]
    results: Dict[str, Dict[str, Any]] = {}
    for source in selected:
        root = _source_root(cfg, source)
        existing = count_files(Path(root)) if root and Path(root).is_dir() else 0
        logger.info("")
        logger.info("--- %s (%d file(s) already on disk) ---", source, existing)
        try:
            result = _call(source, cfg, logger, manifest, skip_download=skip_download)
        except Exception as exc:  # an adapter bug must not hide the other sources
            logger.error("%s adapter raised: %s: %s", source, type(exc).__name__, exc)
            result = DownloadResult(
                source=source,
                status=DownloadResult.FAILED,
                detail=f"{type(exc).__name__}: {exc}",
                root=root,
            )
        results[source] = result.as_dict()
        logger.info("%s -> %s", source, result.status)

    inventory = dataset_inventory(cfg)
    usable = [name for name, payload in results.items() if payload.get("usable")]
    usable_classification = [name for name in usable if name in CLASSIFICATION_SOURCES]
    blocked = {
        name: payload["status"]
        for name, payload in results.items()
        if payload["status"] in {"blocked", "failed"}
    }
    offline = not connectivity["online"]
    gate1_ready = bool(usable_classification)

    if offline:
        gate_note = (
            "No provider host was reachable, so only locally present datasets could be "
            "validated. Re-run with an internet connection to acquire the missing sources."
        )
    elif gate1_ready:
        gate_note = (
            "At least one classification source is validated, so preparation may proceed. "
            "Blocked sources stay blocked; nothing is substituted for them."
        )
    else:
        gate_note = (
            "No classification source could be validated. Preparation and training must "
            "not start until at least one genuine source is present."
        )

    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "connectivity": connectivity,
        "disk": disk,
        "sources": results,
        "inventory": inventory,
        "usable": usable,
        "usable_classification": usable_classification,
        "blocked": blocked,
        "gate1_ready": gate1_ready,
        "gate1_note": gate_note,
        "skip_download": skip_download,
        "manifest_entries": len(manifest.entries),
    }
    paths = write_report(cfg, payload)
    payload["report"] = paths

    if events is not None:
        events.write(
            "download_done",
            usable=usable,
            blocked=blocked,
            gate1_ready=gate1_ready,
        )

    if not usable:
        return {
            "status": "blocked",
            "sources": results,
            "inventory": inventory,
            "blocked": blocked,
            "gate1_ready": False,
            "report": paths,
            "detail": f"no dataset is usable: { {name: r['status'] for name, r in results.items()} }",
        }
    if not usable_classification:
        return {
            "status": "blocked",
            "sources": results,
            "inventory": inventory,
            "blocked": blocked,
            "gate1_ready": False,
            "report": paths,
            "detail": (
                "no classification source is available "
                f"(Mendeley, MMRDR, APTOS, DDR all unavailable): {blocked}"
            ),
        }
    status = "completed" if not blocked else "partial"
    return {
        "status": status,
        "sources": results,
        "inventory": inventory,
        "usable": usable,
        "usable_classification": usable_classification,
        "blocked": blocked,
        "gate1_ready": gate1_ready,
        "report": paths,
        "detail": None if not blocked else f"blocked sources: {blocked}",
    }
