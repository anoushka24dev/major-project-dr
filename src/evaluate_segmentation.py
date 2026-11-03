"""Held-out lesion segmentation evaluation and prediction overlays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .config import Config
from .data import LESION_ORDER, SegmentationDataset, build_idrid_pairs, split_idrid_pairs
from .hardware import detect_hardware, resolve_device
from .metrics import iou_per_channel, segmentation_metrics
from .models import build_unet4
from .train_segmentation import predict_dataset


class SegmentationEvaluationError(RuntimeError):
    """Raised when segmentation cannot be evaluated honestly."""


def _overlay(image: torch.Tensor, mask: np.ndarray, colour) -> Image.Image:
    """Blend one binary lesion mask over the model input image."""
    array = (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8).copy()
    tint = np.zeros_like(array)
    tint[mask] = colour
    blended = (array * 0.6 + tint * 0.4).astype(np.uint8)
    return Image.fromarray(blended, mode="RGB")


def run_evaluation(cfg: Config, logger, events=None) -> Dict[str, Any]:
    out_dir = cfg.outputs_dir("segmentation")
    checkpoint_path = out_dir / "best_lesion_unet.pt"
    if not checkpoint_path.is_file():
        summary_path = out_dir / "segmentation_summary.json"
        reason = "no trained U-Net checkpoint"
        if summary_path.is_file():
            try:
                reason = json.loads(summary_path.read_text(encoding="utf-8")).get("detail", reason)
            except Exception:
                pass
        logger.warning("Segmentation evaluation skipped: %s", reason)
        payload = {"available": False, "reason": reason, "lesion_order": list(LESION_ORDER)}
        (out_dir / "segmentation_metrics.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        if events is not None:
            events.write("segmentation_eval_skipped", reason=reason)
        return {"status": "blocked", "reason": reason}

    pairing = build_idrid_pairs(cfg, logger)
    if not pairing.get("available"):
        raise SegmentationEvaluationError(
            f"IDRiD pairs are unavailable: {pairing.get('reason')}"
        )

    ratios = cfg.section("split")["ratios"]
    split = split_idrid_pairs(pairing["pairs"], ratios, cfg.seed)
    test_pairs = split["test"]
    if not test_pairs:
        raise SegmentationEvaluationError("No held-out IDRiD pairs to evaluate.")

    image_size = int(cfg.require("preprocessing.segmentation.image_size"))
    summary_path = out_dir / "segmentation_summary.json"
    thresholds: List[float] = [0.5] * len(LESION_ORDER)
    if summary_path.is_file():
        try:
            stored = json.loads(summary_path.read_text(encoding="utf-8"))
            stored_thresholds = stored.get("thresholds") or {}
            thresholds = [float(stored_thresholds.get(channel, 0.5)) for channel in LESION_ORDER]
        except Exception:
            pass

    device = resolve_device("auto")
    hardware = detect_hardware(
        mixed_precision_preference=cfg.mixed_precision_setting("segmentation"),
        disk_probe=cfg.root,
    )
    amp_enabled = bool(hardware.mixed_precision)
    batch_size = cfg.effective_batch_size(
        "segmentation", fallback=hardware.recommended_segmentation_batch_size
    )
    num_workers = cfg.effective_num_workers("segmentation") or hardware.recommended_num_workers

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_unet4(
        in_channels=int(cfg.require("segmentation.in_channels")),
        out_channels=int(cfg.require("segmentation.out_channels")),
        base_filters=int(cfg.require("segmentation.base_filters")),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    dataset = SegmentationDataset(test_pairs, cfg, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    ids, probabilities, targets = predict_dataset(
        model, loader, device, amp_enabled=amp_enabled
    )

    metrics = segmentation_metrics(probabilities, targets, LESION_ORDER, thresholds=thresholds)
    metrics.update(
        {
            "partition": "test",
            "num_images": len(ids),
            "checkpoint": str(checkpoint_path),
            "thresholds_source": "validation (tuned in train_segmentation)",
            "note": (
                "Lesion segmentation Dice/IoU. Distinct from the Grad-CAM "
                "explanation-overlap Dice in explanation_validation.json."
            ),
        }
    )

    prediction_dir = out_dir / "lesion_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prediction_dir / "test_probabilities.npz",
        ids=np.array(ids),
        probabilities=probabilities,
        targets=targets,
        thresholds=np.array(thresholds),
    )

    colours = {
        "MA": (255, 96, 96),
        "HE": (96, 255, 96),
        "EX": (96, 96, 255),
        "SE": (255, 255, 96),
    }
    max_overlays = int(cfg.get("evaluation.segmentation_overlays", 8) or 8)
    written = 0
    for index in range(min(max_overlays, len(ids))):
        image, _mask, meta = dataset[index]
        for channel_index, channel in enumerate(LESION_ORDER):
            predicted = probabilities[index, channel_index] >= thresholds[channel_index]
            overlay = _overlay(image, predicted, colours[channel])
            overlay.save(prediction_dir / f"{meta['image_id']}_{channel}.png")
            written += 1

    metrics["overlays_written"] = written
    (out_dir / "segmentation_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )
    logger.info(
        "segmentation test: macro Dice %.4f | macro IoU %.4f (%d images)",
        metrics["macro_dice"], metrics["macro_iou"], len(ids),
    )
    if events is not None:
        events.write(
            "segmentation_eval_done",
            macro_dice=metrics["macro_dice"],
            macro_iou=metrics["macro_iou"],
            num_images=len(ids),
        )
    return {
        "status": "completed",
        "macro_dice": metrics["macro_dice"],
        "macro_iou": metrics["macro_iou"],
        "num_images": len(ids),
        "metrics_json": str(out_dir / "segmentation_metrics.json"),
        "overlays_written": written,
    }
