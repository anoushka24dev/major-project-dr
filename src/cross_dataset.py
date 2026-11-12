"""Cross-dataset evaluation: does the combined model generalise across sources?

Two things are reported separately, because they answer different questions:

1. **Source-wise performance on the internal held-out test partition.** Because the
   unified test split contains images from both Mendeley and MMRDR, splitting the
   metrics by source shows whether one source is systematically harder.

2. **Performance on MMRDR's own official test partition** (the ``ts`` images, kept
   as ``external_test`` and never trained on). This is the honest
   cross-dataset number, since those images were held out by the dataset authors
   rather than by this pipeline.

If only one source contributed to the test partition, no cross-dataset claim is
made; the file states that instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import ClassifierDataset, load_manifest_rows
from .evaluate_classification import evaluate_rows_with_checkpoints
from .hardware import detect_hardware, resolve_device
from .metrics import classification_metrics, predictive_entropy


class CrossDatasetError(RuntimeError):
    """Raised when cross-dataset evaluation cannot be produced honestly."""


def _metrics_for_subset(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    class_names: List[str],
    label: str,
) -> Dict[str, Any]:
    if len(y_true) == 0:
        return {"available": False, "reason": f"no samples for {label}"}
    y_pred = probabilities.argmax(axis=1)
    metrics = classification_metrics(
        y_true, y_pred, probabilities=probabilities, class_names=class_names, compute_auc=True
    )
    metrics["subset"] = label
    metrics["available"] = True
    return metrics


def _evaluate_manifest(
    cfg: Config,
    manifest_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    amp_enabled: bool,
    class_names: List[str],
    batch_size: int,
    num_workers: int,
    logger=None,
) -> Dict[str, Any]:
    rows = load_manifest_rows(manifest_path)
    stacked, truth, _meta = evaluate_rows_with_checkpoints(
        cfg,
        rows,
        checkpoint_path,
        device=device,
        amp_enabled=amp_enabled,
        batch_size=batch_size,
        num_workers=num_workers,
        logger=logger,
    )
    return {
        "samples": int(len(truth)),
        "metrics": _metrics_for_subset(stacked, truth, class_names, manifest_path.stem),
        "probabilities": stacked,
        "y_true": truth,
    }


def run_cross_dataset(cfg: Config, logger, events=None) -> Dict[str, Any]:
    out_dir = cfg.outputs_dir("evaluation")
    checkpoint_path = cfg.outputs_dir("classification") / "best_resnet50.pt"
    if not checkpoint_path.is_file() and not cfg.get("evaluation.ensemble_checkpoints"):
        raise CrossDatasetError(f"No classifier checkpoint at {checkpoint_path}.")

    class_names = list(cfg.classes)
    device = resolve_device("auto")
    hardware = detect_hardware(
        mixed_precision_preference=cfg.mixed_precision_setting("classification"),
        disk_probe=cfg.root,
    )
    amp_enabled = bool(hardware.mixed_precision)
    batch_size = cfg.effective_batch_size(
        "classification", fallback=hardware.recommended_classification_batch_size
    )
    num_workers = cfg.effective_num_workers("classification") or hardware.recommended_num_workers

    result: Dict[str, Any] = {
        "available": True,
        "method": (
            "The same frozen classifier is applied to each dataset partition separately. "
            "No images from these partitions were used for training or checkpoint selection."
        ),
        "within_internal_test": {},
        "external_partition": {},
        "domain_shift": {},
    }

    probability_path = cfg.outputs_dir("classification") / "test_probabilities.npz"
    if probability_path.is_file():
        data = np.load(probability_path, allow_pickle=True)
        probabilities = data["probabilities"]
        y_true = data["y_true"]
        sources = np.asarray([str(s) for s in data["sources"]])
        for source in sorted(set(sources.tolist())):
            mask = sources == source
            result["within_internal_test"][source] = _metrics_for_subset(
                probabilities[mask], y_true[mask], class_names, f"internal_test:{source}"
            )
        result["within_internal_test"]["combined"] = _metrics_for_subset(
            probabilities, y_true, class_names, "internal_test:combined"
        )
    else:
        result["within_internal_test"] = {
            "available": False,
            "reason": "test probabilities not found; run classification evaluation first",
        }

    external_manifest = cfg.outputs_dir("manifests") / "external_test_manifest.csv"
    external_saved: Optional[np.ndarray] = None
    if external_manifest.is_file():
        logger.info("Evaluating the MMRDR official test partition: %s", external_manifest)
        evaluated = _evaluate_manifest(
            cfg, external_manifest, checkpoint_path, device, amp_enabled, class_names, batch_size, num_workers, logger
        )
        external_saved = evaluated["probabilities"]
        metrics = evaluated["metrics"]
        entropy = predictive_entropy(external_saved)
        metrics["mean_confidence"] = float(external_saved.max(axis=1).mean())
        metrics["mean_entropy_nats"] = float(entropy.mean())
        result["external_partition"] = {
            "name": "mmrdr_official_test",
            "manifest": str(external_manifest),
            "num_samples": evaluated["samples"],
            "metrics": metrics,
        }
        np.savez_compressed(
            out_dir / "external_test_probabilities.npz",
            probabilities=external_saved,
            y_true=evaluated["y_true"],
        )
    else:
        result["external_partition"] = {
            "available": False,
            "reason": (
                "No MMRDR official test partition (external_test_manifest.csv). It is created "
                "by the prepare stage when MMRDR-CFP `ts` images are present."
            ),
        }

    # ---- interpretable comparison -------------------------------------
    internal = result["within_internal_test"]
    internal_sources = [key for key in internal if key not in {"combined", "available", "reason"}]
    gaps: Dict[str, Any] = {}
    if len(internal_sources) >= 2:
        for key in internal_sources:
            entry = internal[key]
            if not entry.get("available"):
                continue
            gaps[key] = {
                "accuracy_delta_vs_combined": entry["accuracy"] - internal["combined"]["accuracy"],
                "macro_f1_delta_vs_combined": entry["macro_f1"] - internal["combined"]["macro_f1"],
            }
        result["domain_shift"] = {
            "comparable": True,
            "basis": "source-wise metrics on the internal held-out test partition",
            "deltas": gaps,
            "interpretation": (
                "Negative deltas indicate that a source is harder for the combined model. This "
                "is a descriptive observation about domain shift, not proof of a cause."
            ),
        }
    else:
        result["domain_shift"] = {
            "comparable": False,
            "reason": (
                "Only one dataset source is present in the test partition ("
                + ", ".join(internal_sources) + "). A cross-dataset comparison between "
                "Mendeley and MMRDR requires both to be available, so none is reported."
            ),
        }

    external_metrics = (result["external_partition"] or {}).get("metrics") or {}
    if external_metrics.get("available") and internal.get("combined", {}).get("available"):
        result["domain_shift"]["external_comparison"] = {
            "internal_accuracy": internal["combined"]["accuracy"],
            "external_accuracy": external_metrics["accuracy"],
            "accuracy_delta": external_metrics["accuracy"] - internal["combined"]["accuracy"],
            "internal_macro_f1": internal["combined"]["macro_f1"],
            "external_macro_f1": external_metrics["macro_f1"],
            "macro_f1_delta": external_metrics["macro_f1"] - internal["combined"]["macro_f1"],
            "interpretation": (
                "MMRDR's own held-out test images versus the internal test mix. A drop here is "
                "the expected signature of domain shift between acquisition protocols."
            ),
        }

    result["caveats"] = [
        "Cross-dataset comparison is only meaningful where label definitions and imaging "
        "modalities match; only the MMRDR CFP subset is used, and its grades map 1:1.",
        "MMRDR-CFP publishes no patient identifier, so patient-independent generalisation "
        "is not claimed for that source.",
        "The five-class mapping is provisional for the Mendeley merges (Very Severe NPDR -> "
        "Severe NPDR, Advanced PDR -> PDR).",
    ]

    path = out_dir / "cross_dataset_results.json"
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    logger.info("cross-dataset results written to %s", path)
    if events is not None:
        events.write(
            "cross_dataset_done",
            comparable=result["domain_shift"].get("comparable", False),
            external_available=bool(external_metrics.get("available")),
        )
    return {
        "status": "completed",
        "results_json": str(path),
        "comparable": result["domain_shift"].get("comparable", False),
        "external_available": bool(external_metrics.get("available")),
    }
