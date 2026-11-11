"""Grad-CAM visualisation and lesion-based explanation validation.

Grad-CAM is computed with the repository's existing, tested engine
(``colab/gradcam.py``) through :mod:`src.legacy_bridge`: the same ``GradCAM``
class, the same ``layer4[-1]`` target selection, the same overlay renderer and the
same attention statistics. No second Grad-CAM implementation exists.

Interpretation limits stated up front
-------------------------------------
* Grad-CAM shows which *spatial regions contributed to the logit*. It is not
  evidence that the model used clinically valid features, and a hot spot is not a
  diagnosed lesion.
* By construction the map is produced by global-average-pooling gradients, so it
  is coarse (7x7 for a 224 px input). Individual pixels must not be read as
  lesions.
* Two Dice numbers exist in this project and they are **not** comparable:
  lesion-segmentation Dice (U-Net vs IDRiD annotations) and Grad-CAM
  explanation-overlap Dice (binarised CAM vs annotations). Both are reported, in
  separate files, with separate keys.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from .config import Config
from .data import (
    LESION_ORDER,
    ClassifierDataset,
    build_idrid_pairs,
    load_manifest_rows,
    load_rgb,
    normalize,
    resize_and_pad_mask,
    split_idrid_pairs,
    to_tensor,
)
from .hardware import detect_hardware, resolve_device
from .legacy_bridge import load_gradcam
from .metrics import binarise_cam, mask_overlap_scores, predictive_entropy
from .models import build_resnet50


class ExplainabilityError(RuntimeError):
    """Raised when explanation output cannot be produced honestly."""


def _slug(text: str) -> str:
    out = []
    for character in str(text):
        if character.isalnum():
            out.append(character.lower())
        elif out and out[-1] != "_":
            out.append("_")
    return "".join(out).strip("_") or "item"


def _heatmap_image(cam: np.ndarray, grad_mod) -> Image.Image:
    """Render a CAM as a standalone heatmap PNG."""
    lookup = getattr(grad_mod, "_JET_LUT", None)
    indices = np.clip((np.clip(cam, 0.0, 1.0) * 255.0).astype(np.int32), 0, 255)
    if lookup is not None:
        return Image.fromarray(lookup[indices].astype(np.uint8), mode="RGB")
    # Fallback: greyscale ramp, still a faithful visualisation of the map.
    return Image.fromarray((indices).astype(np.uint8), mode="L").convert("RGB")


def _select_examples(rows: Sequence[Dict[str, str]], per_class: int, seed: int) -> List[Dict[str, str]]:
    """Deterministically pick up to ``per_class`` rows for each class."""
    import random

    by_class: Dict[int, List[Dict[str, str]]] = {}
    for row in rows:
        by_class.setdefault(int(row["mapped_label_int"]), []).append(row)
    selected: List[Dict[str, str]] = []
    rng = random.Random(seed)
    for label in sorted(by_class):
        candidates = sorted(by_class[label], key=lambda r: r.get("image_path", ""))
        rng.shuffle(candidates)
        selected.extend(candidates[:per_class])
    return selected


def _predict_and_cam(
    model,
    cam_engine,
    dataset: ClassifierDataset,
    index: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return ``(probabilities, cam, base_rgb, meta)`` for one dataset item."""
    tensor, _label, meta = dataset[index]
    inputs = tensor.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=1).float().cpu().numpy()[0]
    predicted = int(probabilities.argmax())
    cam = cam_engine(inputs, [predicted])[0]

    # Rebuild the processed base image for the overlay (same pipeline as training).
    from .data import _resize_and_pad

    base_image = _resize_and_pad(load_rgb(Path(meta["image_path"])), dataset.image_size, dataset.root)
    base_rgb = np.asarray(base_image.convert("RGB"), dtype=np.uint8)
    return probabilities, cam, base_rgb, meta


def run_explainability(cfg: Config, logger, events=None) -> Dict[str, Any]:
    """Generate Grad-CAM visualisations and, when possible, validate them."""
    out_root = cfg.outputs_dir("explainability")
    cam_dir = out_root / "gradcam"
    cam_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = cfg.outputs_dir("classification") / "best_resnet50.pt"
    if not checkpoint_path.is_file():
        raise ExplainabilityError(
            f"No classifier checkpoint at {checkpoint_path}; run training first."
        )

    class_names = list(cfg.classes)
    image_size = int(cfg.require("preprocessing.classification.image_size"))
    per_class = int(cfg.get("evaluation.gradcam.num_examples", 3))
    percentile = float(cfg.get("evaluation.gradcam.binarise_percentile", 90))

    device = resolve_device("auto")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_resnet50(num_classes=len(class_names), pretrained=False, allow_weight_download=False).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    grad_mod = load_gradcam(cfg.root)
    layer_name, layer = grad_mod.resolve_cam_layer(model)
    facts = grad_mod.layer_facts(model, layer_name, layer, image_size)
    engine = grad_mod.GradCAM(model, layer, device=device)
    logger.info("Grad-CAM target layer: %s (%s)", facts["module_name"], facts["module_type"])

    test_path = cfg.outputs_dir("manifests") / "test_manifest.csv"
    if not test_path.is_file():
        raise ExplainabilityError(f"No test manifest at {test_path}.")

    rows = load_manifest_rows(test_path)
    selected = _select_examples(rows, per_class, cfg.seed)
    dataset = ClassifierDataset(selected, cfg, train=False, image_size=image_size)

    records: List[Dict[str, Any]] = []
    for index, row in enumerate(selected):
        probabilities, cam, base_rgb, meta = _predict_and_cam(model, engine, dataset, index, device)
        predicted = int(probabilities.argmax())
        truth = int(row["mapped_label_int"])
        stem = _slug(Path(row["image_path"]).stem)[:40]
        class_dir = cam_dir / _slug(class_names[predicted])
        class_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{index:03d}_{stem}"

        Image.fromarray(base_rgb).save(class_dir / f"{prefix}_original.png")
        _heatmap_image(cam, grad_mod).save(class_dir / f"{prefix}_gradcam.png")
        grad_mod.render_overlay(base_rgb, cam).save(class_dir / f"{prefix}_overlay.png")

        stats = grad_mod.attention_stats(cam, grad_mod.fundus_mask(base_rgb))
        records.append(
            {
                "image_id": row.get("image_path", ""),
                "true_index": truth,
                "true_class": class_names[truth],
                "predicted_index": predicted,
                "predicted_class": class_names[predicted],
                "confidence": float(probabilities[predicted]),
                "entropy_nats": float(predictive_entropy(probabilities[None, :])[0]),
                "correct": int(predicted == truth),
                "source_dataset": row.get("source_dataset", ""),
                "fundus_mass": round(float(stats["fundus_mass"]), 6),
                "top20_mass": round(float(stats["top20_mass"]), 6),
                "centroid_radius": round(float(stats["centroid_radius"]), 6),
                "original_png": str(class_dir / f"{prefix}_original.png"),
                "heatmap_png": str(class_dir / f"{prefix}_gradcam.png"),
                "overlay_png": str(class_dir / f"{prefix}_overlay.png"),
            }
        )

    predictions_csv = cam_dir / "gradcam_predictions.csv"
    with predictions_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    logger.info("Grad-CAM written for %d images -> %s", len(records), cam_dir)

    # ---- lesion-based explanation validation --------------------------
    validation = _validate_against_lesions(
        cfg, logger, model, engine, grad_mod, class_names, image_size, device, percentile
    )
    validation["gradcam_visualisation"] = {
        "images": len(records),
        "per_class_requested": per_class,
        "layer": facts,
        "predictions_csv": str(predictions_csv),
        "output_dir": str(cam_dir),
        "correct": sum(r["correct"] for r in records),
        "incorrect": sum(1 - r["correct"] for r in records),
    }
    validation["interpretation_note"] = (
        "Grad-CAM shows which regions contributed to the predicted logit. It is not "
        "proof that the model used clinically valid evidence, and the coarse 7x7 map "
        "must not be read as a lesion localisation."
    )
    path = out_root / "explanation_validation.json"
    path.write_text(json.dumps(validation, indent=2, default=str), encoding="utf-8")

    if events is not None:
        events.write(
            "explainability_done",
            images=len(records),
            lesion_validation=validation.get("lesion_overlap", {}).get("available"),
        )
    return {
        "status": "completed",
        "images": len(records),
        "output_dir": str(cam_dir),
        "validation_json": str(path),
        "lesion_validation_available": validation.get("lesion_overlap", {}).get("available", False),
    }


def _validate_against_lesions(
    cfg: Config,
    logger,
    model,
    engine,
    grad_mod,
    class_names: List[str],
    image_size: int,
    device: torch.device,
    percentile: float,
) -> Dict[str, Any]:
    """Compare binarised Grad-CAM maps with IDRiD lesion annotations."""
    pairing = build_idrid_pairs(cfg, logger)
    if not pairing.get("available"):
        return {
            "lesion_overlap": {
                "available": False,
                "reason": pairing.get("reason", "IDRiD annotations unavailable"),
                "note": (
                    "Explanation validation against lesion annotations requires IDRiD. "
                    "Nothing was computed; no placeholder metric is reported."
                ),
            }
        }

    pairs = pairing["pairs"]
    ratios = cfg.section("split")["ratios"]
    split = split_idrid_pairs(pairs, ratios, cfg.seed)
    candidates = split["test"] or split["validation"]
    limit = int(cfg.get("evaluation.gradcam.lesion_validation_images", 12) or 12)
    chosen = candidates[:limit]

    per_image: List[Dict[str, Any]] = []
    for pair in chosen:
        image = load_rgb(Path(pair["image_path"]))
        from .data import _resize_and_pad

        base_image = _resize_and_pad(image, image_size, cfg.root)
        tensor = normalize(to_tensor(base_image)).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1).float().cpu().numpy()[0]
        predicted = int(probabilities.argmax())
        cam = engine(tensor, [predicted])[0]
        binary_cam = binarise_cam(cam, percentile)

        masks: Dict[str, np.ndarray] = {}
        union = np.zeros((image_size, image_size), dtype=bool)
        for channel in LESION_ORDER:
            with Image.open(Path(pair[f"mask_{channel}"])) as mask_image:
                array = resize_and_pad_mask(mask_image, image_size)
            masks[channel] = array
            union |= array

        overall = mask_overlap_scores(binary_cam, union)
        per_channel = {
            channel: mask_overlap_scores(binary_cam, masks[channel]) for channel in LESION_ORDER
        }
        per_image.append(
            {
                "image_id": pair["image_id"],
                "predicted_class": class_names[predicted],
                "confidence": float(probabilities[predicted]),
                "lesion_pixel_fraction": float(union.mean()),
                "cam_binarise_percentile": percentile,
                "dice_vs_lesion_union": overall["dice"],
                "iou_vs_lesion_union": overall["iou"],
                "per_lesion": per_channel,
            }
        )

    def _mean(values: Sequence[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    aggregate = {
        "mean_dice_vs_lesion_union": _mean([r["dice_vs_lesion_union"] for r in per_image]),
        "mean_iou_vs_lesion_union": _mean([r["iou_vs_lesion_union"] for r in per_image]),
        "per_lesion_mean_dice": {
            channel: _mean([r["per_lesion"][channel]["dice"] for r in per_image])
            for channel in LESION_ORDER
        },
        "per_lesion_mean_iou": {
            channel: _mean([r["per_lesion"][channel]["iou"] for r in per_image])
            for channel in LESION_ORDER
        },
        "num_images": len(per_image),
        "images": per_image,
    }

    return {
        "lesion_overlap": {
            "available": True,
            "method": (
                "The Grad-CAM map for the predicted class is min-max normalised, binarised "
                f"at its {percentile:g}th percentile, and compared with the union of the four "
                "IDRiD lesion masks (and with each mask separately). Both masks are "
                "transformed with the same scale-and-pad geometry as the model input."
            ),
            "thresholding": {
                "binarise_percentile": percentile,
                "chosen_on": "configured value; not tuned on the test partition",
            },
            "evaluation_partition": "IDRiD held-out pairs (never used for classifier training)",
            "external_data_note": (
                "IDRiD is not part of the classifier's training data, so these images are "
                "external to it. This measures explanation/annotation agreement on external "
                "fundus images, not agreement on the classifier's own test set."
            ),
            "aggregate": aggregate,
        },
        "metric_separation": {
            "segmentation_dice": {
                "source": "outputs/segmentation/segmentation_metrics.json",
                "meaning": "U-Net predicted lesion masks vs IDRiD annotations",
            },
            "explanation_overlap_dice": {
                "source": "this file, lesion_overlap.aggregate.mean_dice_vs_lesion_union",
                "meaning": "binarised Grad-CAM regions vs IDRiD annotations",
            },
            "note": (
                "These two Dice values measure different quantities and must never be "
                "compared or averaged together."
            ),
        },
    }
