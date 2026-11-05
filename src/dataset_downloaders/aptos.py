"""APTOS 2019 Blindness Detection adapter.

Access model, verified against the published Kaggle dataset page on 2026-11-03
----------------------------------------------------------------------------
APTOS is distributed through Kaggle. The only supported programmatic path is the
Kaggle API (``kaggle datasets download ...``), which requires:

* the ``kaggle`` CLI installed and on ``PATH``
* a ``~/.kaggle/kaggle.json`` token with download scope (created by the user from
  their Kaggle account, never generated or stored by this pipeline)
* acceptance of the competition/dataset terms on the Kaggle website

This module:

1. detects whether the Kaggle CLI is available and whether credentials exist;
2. checks local disk space against the expected archive size;
3. reports the exact status for APTOS (downloaded / present / blocked / failed);
4. when anything is missing, prints precise, actionable instructions and stops
   gracefully, allowing the user (or another mode) to continue with the other sources.

It never:

* creates, writes or commits a ``kaggle.json`` token;
* hardcodes usernames, keys, passwords or API secrets;
* claims a download that did not complete or that did not validate.
"""

from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import Config
from .base import (
    DownloadError,
    DownloadManifest,
    DownloadResult,
    count_files,
    require_free_space,
)

SOURCE = "aptos"

#: The official source is a Kaggle competition; a few mirrors exist as datasets.
KAGGLE_COMPETITION = "aptos2019-blindness-detection"
KAGGLE_DATASET_HANDLE = "aptos2019-blindness-detection"

#: Approximate downloaded+extracted footprint used for the pre-download space guard.
#: The exact size is checked after download; this is only a pre-flight guard.
EXPECTED_MIN_GB = 0.5


def _home_kaggle_dir() -> Path:
    return Path.home() / ".kaggle"


def _kaggle_token_path() -> Path:
    return _home_kaggle_dir() / "kaggle.json"


def _kaggle_cli_available() -> bool:
    try:
        subprocess.run(
            ["kaggle", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return True
    except Exception:
        return False


def _kaggle_credentials_present() -> bool:
    token = _kaggle_token_path()
    if not token.is_file():
        return False
    try:
        import json as _json
        payload = _json.loads(token.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(payload.get("username")) and bool(payload.get("key"))


def _aptos_folder_present(root: Path) -> Dict[str, Any]:
    """Report what APTOS data is already present under ``root``."""
    root = Path(root)
    info: Dict[str, Any] = {
        "root": str(root),
        "root_exists": root.is_dir(),
        "csv_present": False,
        "image_count": 0,
        "images_dir": None,
    }
    if not info["root_exists"]:
        return info
    # APTOS ships a train CSV alongside a train image folder. We accept either of a
    # few common layouts without assuming a particular one.
    csv_candidate = None
    for name in ("train.csv", "train_labels.csv", "trainLabels.csv"):
        candidate = root / name
        if candidate.is_file():
            csv_candidate = candidate
            break
    info["csv_present"] = bool(csv_candidate)
    info["csv_path"] = str(csv_candidate) if csv_candidate else None
    # Locate an image folder by scanning a small, bounded set of likely names.
    for name in ("train_images", "trainImages", "train", "input", "train_images_heritage"):
        candidate = root / name
        if candidate.is_dir():
            n = count_files(candidate, (".png", ".jpg", ".jpeg"))
            if n > 0:
                info["images_dir"] = str(candidate)
                info["image_count"] = n
                break
    return info


def verify_aptos_labels(root: Path) -> Dict[str, Any]:
    """Verify every train.csv id has an image, and report the label distribution."""
    root = Path(root)
    layout = _aptos_folder_present(root)
    info: Dict[str, Any] = {"ok": False, "rows": 0, "images": 0, "missing_images": 0,
                           "label_counts": {}, "reason": None}
    csv_path = layout.get("csv_path")
    images_dir = layout.get("images_dir")
    if not csv_path or not images_dir:
        info["reason"] = "train.csv or the image directory is missing"
        return info
    with Path(str(csv_path)).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = [str(name).strip().lower() for name in (reader.fieldnames or [])]
    id_field = None
    label_field = None
    for candidate in ("id_code", "image", "filename", "id"):
        if candidate in fields:
            id_field = fields.index(candidate)
            break
    for candidate in ("diagnosis", "level", "label", "grade"):
        if candidate in fields:
            label_field = fields.index(candidate)
            break
    if id_field is None or label_field is None:
        info["reason"] = f"train.csv must contain an id and a diagnosis column; found {fields}"
        return info

    stems = {p.stem.lower() for p in Path(images_dir).iterdir() if p.is_file()}
    missing = 0
    labels: Dict[str, int] = {}
    keyed = list(rows[0].keys()) if rows else []
    id_key = keyed[id_field] if keyed else ""
    label_key = keyed[label_field] if keyed else ""
    for row in rows:
        identifier = str(row.get(id_key, "")).strip().lower()
        if identifier and identifier not in stems:
            missing += 1
        grade = str(row.get(label_key, "")).strip()
        labels[grade] = labels.get(grade, 0) + 1
    info.update({
        "rows": len(rows),
        "images": len(stems),
        "missing_images": missing,
        "label_counts": dict(sorted(labels.items())),
        "ok": bool(rows) and missing == 0,
    })
    if not info["ok"] and missing:
        info["reason"] = f"{missing} train.csv row(s) have no matching image"
    return info


def _instructions(cfg: Config, root: Path, reason: str) -> List[str]:
    """Precise, actionable steps for a human who must supply APTOS."""
    steps = [
        "APTOS 2019 is distributed through Kaggle and requires the Kaggle API plus a "
        "valid Kaggle account/token; it is not anonymously downloadable.",
        f"  1. Create a Kaggle account at https://www.kaggle.com and accept the "
        f"competition rules at https://www.kaggle.com/c/{KAGGLE_DATASET_HANDLE}/rules.",
        "  2. Generate a Kaggle API token from your account settings (Account -> "
        "Create New API Token). This downloads ``kaggle.json``.",
        f"  3. Place ``kaggle.json`` in {str(_kaggle_token_path())} with mode 600 "
        f"(read/write for the user only).",
        "  4. Ensure the ``kaggle`` CLI is installed and on PATH.",
        f"  5. Extract the downloaded archive so the APTOS tree lives under: {root}",
        "  6. Re-run RUN.bat (or `RUN.bat --resume`); the pipeline validates the layout "
        "and continues automatically.",
        f"  Alternatively set DR_APTOS_ROOT to an existing extracted copy.",
        "  Note: this pipeline never asks for your token on the command line, never "
        "writes it anywhere and never commits it. The token stays in ~/.kaggle/kaggle.json.",
        f"Current blocker: {reason}",
    ]
    return steps


def _kaggle_download(dataset_handle: str, destination: Path, cfg: Config, logger) -> Path:
    """Download a Kaggle dataset via the CLI, with retries and a clean failure message.

    The CLI writes into ``destination``. We do not attempt to parse its output beyond
    detecting success/failure.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    cfg_path = _kaggle_token_path()
    if cfg_path.is_file():
        env["KAGGLE_CONFIG_DIR"] = str(cfg_path.parent)
    commands = [
        ["kaggle", "competitions", "download", "-c", dataset_handle,
         "-p", str(destination), "--force", "--unzip"],
        ["kaggle", "datasets", "download", "-d", dataset_handle,
         "-p", str(destination), "--force", "--unzip"],
    ]
    last_error = ""
    for cmd in commands:
        logger.info("Kaggle download command: %s", " ".join(cmd))
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=1800,
                env=env,
                text=True,
            )
        except FileNotFoundError:
            raise DownloadError(
                "The ``kaggle`` CLI was not found on PATH. Install it and retry."
            )
        except subprocess.TimeoutExpired:
            raise DownloadError(
                "The Kaggle download timed out after 30 minutes. Check the network and retry."
            )
        if proc.returncode == 0:
            return destination
        output = (proc.stdout or "")[:3000]
        last_error = f"Kaggle CLI exited with code {proc.returncode}.\n{output}"
        lowered = output.lower()
        if "403" in lowered or "accept" in lowered or "rules" in lowered:
            raise DownloadError(
                "Kaggle refused the download: the competition rules for "
                f"{dataset_handle} have not been accepted by this account. Open "
                f"https://www.kaggle.com/c/{dataset_handle}/rules, accept them, then re-run.\n"
                f"{output}"
            )
        if "401" in lowered or "unauthorized" in lowered:
            raise DownloadError(
                "Kaggle rejected the API token (401). Create a fresh token at "
                "https://www.kaggle.com/settings and replace ~/.kaggle/kaggle.json.\n"
                f"{output}"
            )
    raise DownloadError(last_error or "Kaggle download failed")


def run(
    cfg: Config,
    logger,
    manifest: DownloadManifest,
    *,
    root: Optional[Path] = None,
    force: bool = False,
    skip_download: bool = False,
) -> DownloadResult:
    """Acquire APTOS, or report precisely why it is unavailable."""
    target = Path(root) if root else cfg.aptos_root()
    layout = _aptos_folder_present(target)

    manifest.add(
        "inspect",
        source=SOURCE,
        root=str(target),
        csv_present=layout.get("csv_present"),
        image_count=layout.get("image_count"),
        images_dir=layout.get("images_dir"),
        kaggle_cli=_kaggle_cli_available(),
        kaggle_credentials=_kaggle_credentials_present(),
        auto_download=True,
    )

    if layout.get("image_count", 0) > 0 and layout.get("csv_present"):
        records = verify_aptos_labels(target)
        manifest.add("verify_labels", source=SOURCE, ok=records.get("ok"), records=records)
        logger.info(
            "APTOS already present: %d images + train CSV under %s",
            layout.get("image_count", 0),
            target,
        )
        return DownloadResult(
            source=SOURCE,
            status=(
                DownloadResult.OK_ALREADY_PRESENT if records.get("ok") else DownloadResult.FAILED
            ),
            detail=(
                f"APTOS validated: {layout.get('image_count', 0)} images and a train CSV "
                f"under {target}"
                if records.get("ok")
                else f"APTOS labels do not match the images: {records.get('reason')}"
            ),
            root=target,
            metadata={
                "layout": layout,
                "records": records,
                "provenance": "pre-existing copy",
            },
        )

    # Not present yet: report blockers honestly.
    missing: List[str] = []
    if not _kaggle_cli_available():
        missing.append("Kaggle CLI not found on PATH")
    if not _kaggle_credentials_present():
        missing.append("no valid Kaggle API token at " + str(_kaggle_token_path()))
    if missing:
        detail = "APTOS requires the Kaggle API; " + "; ".join(missing) + "."
        logger.warning(detail)
        return DownloadResult(
            source=SOURCE,
            status=DownloadResult.BLOCKED,
            detail=detail,
            root=target,
            requirements=_instructions(cfg, target, " + ".join(missing)),
            metadata={"layout": layout, "requires_kaggle_api": True},
        )

    if skip_download:
        logger.info("APTOS download skipped by request (--skip-download)")
        return DownloadResult(
            source=SOURCE,
            status=DownloadResult.SKIPPED,
            detail="APTOS download skipped by request (--skip-download)",
            root=target,
            metadata={"layout": layout, "requires_kaggle_api": True},
        )

    # Credentials + CLI present, but data absent. Attempt automatic download.
    free = require_free_space(
        target,
        int(EXPECTED_MIN_GB * 1024**3),
        logger=logger,
        multiplier=1.5,
    )
    try:
        _kaggle_download(KAGGLE_DATASET_HANDLE, target, cfg, logger)
    except Exception as exc:
        logger.error("APTOS Kaggle download failed: %s", exc)
        manifest.add(
            "download_failed",
            source=SOURCE,
            root=str(target),
            error=str(exc),
        )
        return DownloadResult(
            source=SOURCE,
            status=DownloadResult.FAILED,
            detail=f"APTOS Kaggle download failed: {exc}",
            root=target,
            requirements=_instructions(cfg, target, f"Kaggle download failed: {exc}"),
            metadata={"layout": layout},
        )

    # Validate what was downloaded.
    after = _aptos_folder_present(target)
    records = verify_aptos_labels(target) if after.get("csv_present") else {"ok": False}
    if not after.get("image_count", 0) or not after.get("csv_present") or not records.get("ok"):
        detail = (
            "APTOS download completed but the expected train CSV/images were not found "
            f"under {target}."
            if not records.get("ok")
            else f"APTOS labels do not match the images: {records.get('reason')}"
        )
        logger.error(detail)
        manifest.add(
            "download_incomplete",
            source=SOURCE,
            root=str(target),
            layout=after,
        )
        return DownloadResult(
            source=SOURCE,
            status=DownloadResult.FAILED,
            detail=detail,
            root=target,
            requirements=_instructions(
                cfg, target, "downloaded archive did not contain the expected APTOS data"
            ),
            metadata={"layout": after},
        )

    logger.info(
        "APTOS downloaded and validated: %d images + train CSV under %s",
        after.get("image_count", 0),
        target,
    )
    manifest.add("verify_labels", source=SOURCE, ok=True, records=records)
    return DownloadResult(
        source=SOURCE,
        status=DownloadResult.OK,
        detail=(
            f"APTOS downloaded and validated: {after.get('image_count', 0)} images and a "
            f"train CSV under {target}"
        ),
        root=target,
        metadata={
            "layout": after,
            "records": records,
            "provenance": "downloaded via the official Kaggle competition API",
        },
    )
