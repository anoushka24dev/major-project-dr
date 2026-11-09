"""Held-out test evaluation for the five-class ResNet-50.

The test partition is opened **only** here, after training has finished and its
checkpoint has been frozen. Nothing in this module selects a threshold, a
checkpoint or a hyper-parameter: it reports what the model does on data it has
never seen.

Outputs (``outputs/classification``)
------------------------------------
``classification_metrics.json``  full metric set including per-class metrics,
                               confusion matrix, Cohen's kappa and one-vs-rest ROC-AUC
``classification_report.csv``    one row per class plus macro/weighted rows
``confusion_matrix.png``         count-annotated confusion matrix
``roc_curves.png``               one-vs-rest ROC curves with per-class AUC
``test_predictions.csv``         per-image truth, prediction, confidence, entropy
``test_probabilities.npz``       raw probabilities, reused by calibration and cross-dataset
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import ClassifierDataset, load_manifest_rows
from .hardware import detect_hardware, resolve_device
from .metrics import (
    classification_metrics,
    predictive_entropy,
)
from .models import build_resnet50
from .tta import (
    TTAError,
    build_views as build_tta_views,
    predict as tta_predict,
    selected_view_names,
    selection_path,
)


class EvaluationError(RuntimeError):
    """Raised when evaluation cannot produce an honest result."""


def _write_report_csv(path: Path, metrics: Dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for name in metrics["class_order"]:
        stats = metrics["per_class"][name]
        rows.append(
            {
                "class": name,
                "precision": round(stats["precision"], 6),
                "recall_sensitivity": round(stats["recall"], 6),
                "specificity": round(stats["specificity"], 6),
                "f1": round(stats["f1"], 6),
                "support": stats["support"],
                "predicted": stats["predicted"],
            }
        )
    rows.append(
        {
            "class": "macro_avg",
            "precision": round(metrics["macro_precision"], 6),
            "recall_sensitivity": round(metrics["macro_recall"], 6),
            "specificity": round(metrics["macro_specificity"], 6),
            "f1": round(metrics["macro_f1"], 6),
            "support": metrics["num_samples"],
            "predicted": metrics["num_samples"],
        }
    )
    rows.append(
        {
            "class": "weighted_f1",
            "precision": "",
            "recall_sensitivity": "",
            "specificity": "",
            "f1": round(metrics["weighted_f1"], 6),
            "support": metrics["num_samples"],
            "predicted": metrics["num_samples"],
        }
    )
    rows.append(
        {
            "class": "overall",
            "precision": "",
            "recall_sensitivity": "",
            "specificity": "",
            "f1": "",
            "support": metrics["num_samples"],
            "predicted": f"accuracy={metrics['accuracy']:.6f}; kappa={metrics['cohen_kappa']:.6f}",
        }
    )
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return target


def _plot_confusion(matrix: np.ndarray, class_names: List[str], path: Path, title: str) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    threshold = matrix.max() / 2.0 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, int(matrix[i, j]), ha="center", va="center", fontsize=8,
                color="white" if matrix[i, j] > threshold else "black",
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _posthoc_for_published_checkpoint(cfg: Config, out_dir: Path) -> Optional[Dict[str, Any]]:
    """Return the post-hoc parameters attached to the published checkpoint, if any.

    Only the experiment whose checkpoint was actually published can contribute
    thresholds or a temperature, so an experiment's parameters can never be applied
    to a different model.
    """
    provenance = out_dir / "checkpoint_provenance.json"
    if not provenance.is_file():
        return None
    try:
        published = json.loads(provenance.read_text(encoding="utf-8"))
    except Exception:
        return None
    experiment = published.get("source_experiment")
    if not experiment:
        return None
    parameter_path = cfg.outputs_dir("experiments", str(experiment)) / "posthoc.json"
    if not parameter_path.is_file():
        return None
    try:
        payload = json.loads(parameter_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    payload["source_experiment"] = experiment
    payload["parameter_file"] = str(parameter_path)
    return payload


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _supplementary_metrics(
    cfg: Config,
    posthoc: Optional[Dict[str, Any]],
    *,
    probabilities: np.ndarray,
    logits: np.ndarray,
    y_true: np.ndarray,
    class_names: List[str],
    out_dir: Path,
    all_meta: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Turn validation-fitted parameters into clearly labelled supplementary results.

    The headline metrics are always the plain argmax of the model's own probabilities.
    Thresholds and temperature are reported here so a threshold shift can never be
    presented as a model improvement.
    """
    supplementary: Dict[str, Any] = {}
    if not posthoc:
        return supplementary

    offsets = posthoc.get("class_offsets") or {}
    if offsets:
        vector = np.asarray([float(offsets.get(name, 0.0)) for name in class_names])
        thresholded_predictions = (probabilities + vector).argmax(axis=1)
        thresholded = classification_metrics(
            y_true, thresholded_predictions, probabilities=probabilities,
            class_names=class_names, compute_auc=False,
        )
        supplementary["thresholded"] = {
            "class_offsets": dict(offsets),
            "accuracy": thresholded["accuracy"],
            "macro_f1": thresholded["macro_f1"],
            "macro_precision": thresholded["macro_precision"],
            "macro_recall": thresholded["macro_recall"],
            "predicted_class_counts": {
                name: stats["predicted"] for name, stats in thresholded["per_class"].items()
            },
            "selected_on": "validation fold of experiment " + str(posthoc.get("source_experiment")),
            "note": (
                "Offsets are added to each class's softmax score before argmax. They can "
                "shift the precision/recall balance; they do not change the model."
            ),
        }
        with (Path(out_dir) / "test_predictions_thresholded.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "image_path", "source_dataset", "true_index", "predicted_index",
                "predicted_class", "confidence", "correct",
            ])
            for index, meta in enumerate(all_meta):
                truth = int(y_true[index])
                predicted = int(thresholded_predictions[index])
                writer.writerow([
                    meta.get("image_path", ""),
                    meta.get("source_dataset", ""),
                    truth,
                    predicted,
                    class_names[predicted],
                    f"{float(probabilities[index].max()):.6f}",
                    int(truth == predicted),
                ])

    temperature = posthoc.get("temperature")
    if temperature:
        scaled = _softmax_rows(logits / float(temperature))
        scaled_metrics = classification_metrics(
            y_true, scaled.argmax(axis=1), probabilities=scaled,
            class_names=class_names, compute_auc=False,
        )
        supplementary["temperature_scaled"] = {
            "temperature": float(temperature),
            "accuracy": scaled_metrics["accuracy"],
            "macro_f1": scaled_metrics["macro_f1"],
            "mean_confidence": float(scaled.max(axis=1).mean()),
            "selected_on": "validation fold of experiment " + str(posthoc.get("source_experiment")),
            "note": (
                "One-parameter temperature scaling cannot change predictions; accuracy "
                "is reported only to show that it did not."
            ),
        }

    if supplementary:
        supplementary["posthoc_parameter_file"] = posthoc.get("parameter_file")
    return supplementary


def _plot_roc(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    class_names: List[str],
    auc_info: Dict[str, Any],
    path: Path,
) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
    except Exception:
        return None

    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    plotted = 0
    for index, name in enumerate(class_names):
        binary = (np.asarray(y_true) == index).astype(int)
        if binary.sum() == 0 or binary.sum() == len(binary):
            continue
        fpr, tpr, _ = roc_curve(binary, probabilities[:, index])
        value = auc_info.get("per_class", {}).get(str(index))
        label = f"{name} (AUC={value:.3f})" if isinstance(value, float) else name
        ax.plot(fpr, tpr, linewidth=1.4, label=label)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return None
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-rest ROC curves (held-out test set)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_evaluation(cfg: Config, logger, events=None) -> Dict[str, Any]:
    """Evaluate the trained classifier on the untouched test partition."""
    out_dir = cfg.outputs_dir("classification")
    manifests = cfg.outputs_dir("manifests")
    test_path = manifests / "test_manifest.csv"
    checkpoint_path = out_dir / "best_resnet50.pt"

    if not checkpoint_path.is_file():
        raise EvaluationError(
            f"No trained classifier checkpoint at {checkpoint_path}. "
            "Run the classification training stage first."
        )
    if not test_path.is_file():
        raise EvaluationError(
            f"No test manifest at {test_path}. Run the 'prepare' stage first."
        )

    class_names = list(cfg.classes)
    image_size = int(cfg.require("preprocessing.classification.image_size"))
    rows = load_manifest_rows(test_path)
    logger.info("Evaluating on %d held-out test images", len(rows))

    device = resolve_device("auto")
    hardware = detect_hardware(
        mixed_precision_preference=cfg.mixed_precision_setting("classification"),
        disk_probe=cfg.root,
    )
    batch_size = cfg.effective_batch_size(
        "classification", fallback=hardware.recommended_classification_batch_size
    )
    num_workers = cfg.effective_num_workers("classification") or hardware.recommended_num_workers

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    fingerprint = payload.get("fingerprint") or {}
    if list(fingerprint.get("classes") or []) and list(fingerprint["classes"]) != class_names:
        raise EvaluationError(
            "The checkpoint's class order does not match the configured taxonomy: "
            f"checkpoint={fingerprint.get('classes')}, config={class_names}"
        )
    if int(fingerprint.get("image_size", image_size)) != image_size:
        raise EvaluationError(
            f"Checkpoint was trained at image_size={fingerprint.get('image_size')} but the "
            f"config now says {image_size}."
        )

    model = build_resnet50(
        num_classes=len(class_names), pretrained=False, allow_weight_download=False
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    dataset = ClassifierDataset(rows, cfg, train=False, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # TTA is applied only when the validation comparison already selected it. With no
    # saved selection this is exactly one forward pass per batch, i.e. ordinary
    # inference. The choice was made before this partition was opened.
    selection_file = selection_path(cfg)
    try:
        selected_views = selected_view_names(cfg)
    except TTAError as exc:
        logger.warning("TTA configuration is invalid (%s); using ordinary inference", exc)
        selected_views = None
    view_functions = build_tta_views(selected_views) if selected_views else None

    inference = tta_predict(
        model,
        loader,
        device,
        views=view_functions,
        num_classes=len(class_names),
    )
    probabilities = inference["probabilities"]
    logits_array = inference["logits"]
    y_true = np.asarray(inference["labels"])
    all_meta = inference["meta"]
    y_pred = probabilities.argmax(axis=1)

    metrics = classification_metrics(
        y_true, y_pred, probabilities=probabilities, class_names=class_names,
        compute_auc=bool(cfg.get("evaluation.roc_auc.enabled", True)),
    )

    # E5 / E6: report the effect of validation-selected thresholds and temperature
    # scaling as clearly separated supplementary results, never as the headline metric.
    posthoc = _posthoc_for_published_checkpoint(cfg, out_dir)
    supplementary = _supplementary_metrics(
        cfg,
        posthoc,
        probabilities=probabilities,
        logits=logits_array,
        y_true=y_true,
        class_names=class_names,
        out_dir=out_dir,
        all_meta=all_meta,
    )
    if supplementary:
        metrics["supplementary"] = supplementary
    metrics["partition"] = "test"
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["checkpoint_fingerprint"] = fingerprint
    metrics["provisional_label_policy"] = bool(cfg.get("project.provisional_label_policy", True))
    metrics["integrity_note"] = (
        "Reported on the held-out test partition only. No threshold, checkpoint or "
        "hyper-parameter was chosen using this partition."
    )
    metrics["inference"] = {
        "mode": inference["mode"],
        "views": list(inference["views"]),
        "reduction": (
            "mean of per-view class probabilities"
            if inference["mode"] == "tta"
            else "single forward pass"
        ),
        "selection_file": str(selection_file) if selection_file.is_file() else None,
        "selected_on": (
            "validation partition" if inference["mode"] == "tta" else "not applied"
        ),
        "note": (
            "TTA, when used, was selected on the validation partition before this "
            "partition was opened. The test metrics below are reported, not tuned."
        ),
    }

    metrics_path = out_dir / "classification_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    _write_report_csv(out_dir / "classification_report.csv", metrics)

    matrix = np.asarray(metrics["confusion_matrix"])
    confusion_png = _plot_confusion(
        matrix, class_names, out_dir / "confusion_matrix.png",
        "ResNet-50 five-class confusion matrix (held-out test set)",
    )
    auc_info = metrics.get("roc_auc", {})
    roc_png = _plot_roc(
        probabilities, y_true, class_names, auc_info, out_dir / "roc_curves.png"
    )

    entropy = predictive_entropy(probabilities)
    predictions_csv = out_dir / "test_predictions.csv"
    with predictions_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["image_path", "source_dataset", "original_label", "true_index", "true_class",
             "predicted_index", "predicted_class", "confidence", "entropy_nats", "correct"]
        )
        for index, meta in enumerate(all_meta):
            truth = int(y_true[index])
            predicted = int(y_pred[index])
            writer.writerow(
                [
                    meta.get("image_path", ""),
                    meta.get("source_dataset", ""),
                    meta.get("original_label", ""),
                    truth, class_names[truth],
                    predicted, class_names[predicted],
                    f"{float(probabilities[index].max()):.6f}",
                    f"{float(entropy[index]):.6f}",
                    int(truth == predicted),
                ]
            )

    np.savez_compressed(
        out_dir / "test_probabilities.npz",
        probabilities=probabilities,
        logits=logits_array,
        y_true=y_true,
        y_pred=y_pred,
        sources=np.array([m.get("source_dataset", "") for m in all_meta]),
        paths=np.array([m.get("image_path", "") for m in all_meta]),
    )

    summary = {
        "status": "completed",
        "partition": "test",
        "num_samples": int(len(y_true)),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "cohen_kappa": metrics["cohen_kappa"],
        "roc_auc_macro": (auc_info or {}).get("macro"),
        "metrics_json": str(metrics_path),
        "report_csv": str(out_dir / "classification_report.csv"),
        "confusion_matrix_png": str(confusion_png) if confusion_png else None,
        "roc_curves_png": str(roc_png) if roc_png else None,
        "predictions_csv": str(predictions_csv),
        "device": str(device),
        "posthoc_applied": sorted(supplementary) if supplementary else [],
        "inference_mode": inference["mode"],
        "inference_views": list(inference["views"]),
    }
    logger.info(
        "test: accuracy %.4f | macro F1 %.4f | kappa %.4f | AUC %s",
        metrics["accuracy"], metrics["macro_f1"], metrics["cohen_kappa"],
        summary["roc_auc_macro"],
    )
    if events is not None:
        events.write("classification_eval_done", **summary)
    return summary
