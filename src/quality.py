"""Image-quality assessment: unreadable, blurred, under/over-exposed images.

The assessment is **advisory**. Every decision is written to
``outputs/manifests/image_quality.csv`` and nothing is deleted. When
``quality.exclude_flagged`` is true, the flags are applied on the *next*
``prepare`` run (which reads this CSV), so exclusion is always a recorded,
re-runnable decision rather than a silent filter.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from .config import Config

QUALITY_FILENAME = "image_quality.csv"
QUALITY_COLUMNS = (
    "image_path",
    "source_dataset",
    "split",
    "readable",
    "width",
    "height",
    "mode",
    "mean_intensity",
    "laplacian_variance",
    "dark_fraction",
    "blurred",
    "underexposed",
    "overexposed",
    "quality_flags",
    "gradable",
    "action",
)


class QualityAssessmentError(RuntimeError):
    """Raised when quality assessment cannot be performed."""


def _laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian, inside the fundus region."""
    if gray.size == 0:
        return 0.0
    padded = np.pad(gray, 1, mode="edge")
    laplace = (
        padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
        - 4.0 * padded[1:-1, 1:-1]
    )
    return float(laplace.var())


def assess_image(path: Path, *, blur_threshold: float, dark_threshold: float,
                 bright_threshold: float, dark_fraction_threshold: float) -> Dict[str, Any]:
    """Assess one image and return its quality row fields."""
    result: Dict[str, Any] = {
        "readable": False,
        "width": 0, "height": 0, "mode": "",
        "mean_intensity": "",
        "laplacian_variance": "",
        "dark_fraction": "",
        "blurred": "",
        "underexposed": "",
        "overexposed": "",
        "quality_flags": "",
        "gradable": "unknown",
    }
    flags: List[str] = []
    try:
        with Image.open(path) as image:
            image.load()
            result["readable"] = True
            result["width"] = int(image.width)
            result["height"] = int(image.height)
            result["mode"] = str(image.mode)
            # Downscale for speed; blur is judged on a consistently sized image.
            working = image.convert("L")
            if max(working.size) > 512:
                scale = 512 / max(working.size)
                working = working.resize(
                    (max(1, int(working.width * scale)), max(1, int(working.height * scale))),
                    Image.BILINEAR,
                )
            gray = np.asarray(working, dtype=np.float64)
    except Exception as exc:
        result["quality_flags"] = f"unreadable:{type(exc).__name__}"
        result["gradable"] = "no"
        return result

    # Restrict statistics to the retinal area so padding does not dominate.
    tissue = gray > 12.0
    if tissue.mean() < 0.2:
        tissue = np.ones_like(gray, dtype=bool)

    mean_intensity = float(gray[tissue].mean()) if tissue.any() else float(gray.mean())
    variance = _laplacian_variance(gray)
    dark_fraction = float((gray < 10.0).mean())

    blurred = variance < float(blur_threshold)
    underexposed = mean_intensity < float(dark_threshold)
    overexposed = mean_intensity > float(bright_threshold)
    if dark_fraction > float(dark_fraction_threshold):
        flags.append("mostly_black")

    result.update(
        {
            "mean_intensity": round(mean_intensity, 3),
            "laplacian_variance": round(variance, 3),
            "dark_fraction": round(dark_fraction, 4),
            "blurred": "yes" if blurred else "no",
            "underexposed": "yes" if underexposed else "no",
            "overexposed": "yes" if overexposed else "no",
        }
    )
    if blurred:
        flags.append("blurred")
    if underexposed:
        flags.append("underexposed")
    if overexposed:
        flags.append("overexposed")

    result["quality_flags"] = ";".join(flags)
    result["gradable"] = "review" if flags else "yes"
    return result


def load_quality_rows(path: Path) -> Dict[str, Dict[str, str]]:
    """Load a previously written quality CSV keyed by image path."""
    target = Path(path)
    if not target.is_file():
        return {}
    with target.open(newline="", encoding="utf-8-sig") as handle:
        return {row["image_path"]: row for row in csv.DictReader(handle)}


def run_quality_assessment(cfg: Config, logger, events=None, *, limit: Optional[int] = None) -> Dict[str, Any]:
    """Assess every included image in the unified manifest."""
    if not bool(cfg.get("quality.enabled", True)):
        logger.info("Quality assessment disabled in configuration")
        return {"status": "skipped", "reason": "quality.enabled is false"}

    manifests = cfg.outputs_dir("manifests")
    all_images = manifests / "all_images.csv"
    if not all_images.is_file():
        raise QualityAssessmentError(
            f"No unified manifest at {all_images}; run the 'prepare' stage first."
        )

    with all_images.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("included") == "yes"]
    if limit:
        rows = rows[: int(limit)]
    if not rows:
        raise QualityAssessmentError("The unified manifest contains no included images.")

    blur_threshold = float(cfg.get("quality.blur_threshold", 12.0))
    dark_threshold = float(cfg.get("quality.dark_threshold", 25.0))
    bright_threshold = float(cfg.get("quality.bright_threshold", 230.0))
    dark_fraction_threshold = float(cfg.get("quality.dark_fraction_threshold", 0.75))
    exclude_flagged = bool(cfg.get("quality.exclude_flagged", False))

    out_rows: List[Dict[str, Any]] = []
    unreadable = 0
    flagged = 0
    for index, row in enumerate(rows, start=1):
        path = Path(row.get("image_path", ""))
        assessment = assess_image(
            path,
            blur_threshold=blur_threshold,
            dark_threshold=dark_threshold,
            bright_threshold=bright_threshold,
            dark_fraction_threshold=dark_fraction_threshold,
        )
        if not assessment["readable"]:
            unreadable += 1
        elif assessment["quality_flags"]:
            flagged += 1
        out_rows.append(
            {
                "image_path": row.get("image_path", ""),
                "source_dataset": row.get("source_dataset", ""),
                "split": row.get("split", ""),
                **assessment,
                "action": (
                    "flagged_for_review"
                    if assessment["quality_flags"]
                    else "kept"
                ),
            }
        )
        if index % 1000 == 0:
            logger.info("quality: assessed %d/%d images", index, len(rows))

    target = manifests / QUALITY_FILENAME
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(QUALITY_COLUMNS))
        writer.writeheader()
        for row in out_rows:
            writer.writerow({key: row.get(key, "") for key in QUALITY_COLUMNS})

    flag_counts: Dict[str, int] = {}
    for row in out_rows:
        for flag in str(row["quality_flags"]).split(";"):
            if flag:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1

    summary = {
        "status": "completed",
        "images_assessed": len(out_rows),
        "unreadable": unreadable,
        "flagged_for_review": flagged,
        "flag_counts": flag_counts,
        "thresholds": {
            "blur_laplacian_variance": blur_threshold,
            "dark_mean_intensity": dark_threshold,
            "bright_mean_intensity": bright_threshold,
            "dark_fraction": dark_fraction_threshold,
        },
        "exclusion_policy": {
            "exclude_flagged_configured": exclude_flagged,
            "applied": False,
            "note": (
                "Flagged images were recorded, not removed. "
                + (
                    "quality.exclude_flagged is true, so they will be excluded by the NEXT "
                    "prepare run, keeping the decision recorded and reversible."
                    if exclude_flagged
                    else "Set quality.exclude_flagged=true and re-run prepare to exclude them."
                )
            ),
        },
        "csv": str(target),
        "advisory_note": (
            "These heuristics are advisory screening aids, not a validated gradability "
            "score. No image was deleted."
        ),
    }
    report_path = cfg.outputs_dir("evaluation") / "quality_summary.json"
    report_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info(
        "quality: %d assessed, %d unreadable, %d flagged for review -> %s",
        len(out_rows), unreadable, flagged, target,
    )
    if events is not None:
        events.write(
            "quality_done",
            assessed=len(out_rows),
            unreadable=unreadable,
            flagged=flagged,
        )
    return summary
