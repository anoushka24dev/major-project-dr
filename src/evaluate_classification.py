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
from .models import build_classifier, predict_probs_with_tta


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


def _resolve_checkpoint_specs(cfg: Config, default_checkpoint: Path) -> List[Tuple[Path, float]]:
    raw = cfg.get("evaluation.ensemble_checkpoints")
    if not raw:
        return [(default_checkpoint, 1.0)]
    specs: List[Tuple[Path, float]] = []
    for entry in raw:
        if isinstance(entry, dict):
            path_str = entry.get("path")
            weight = float(entry.get("weight", 1.0))
        else:
            path_str = str(entry)
            weight = 1.0
        if not path_str:
            continue
        p = Path(path_str)
        if not p.is_absolute():
            p = cfg.root / p
        specs.append((p, weight))
    return specs or [(default_checkpoint, 1.0)]


def evaluate_rows_with_checkpoints(
    cfg: Config,
    rows: Sequence[Dict[str, str]],
    default_checkpoint: Path,
    *,
    device: torch.device,
    amp_enabled: bool,
    batch_size: int,
    num_workers: int,
    logger=None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Run single-checkpoint or weighted multi-checkpoint evaluation with optional TTA."""
    from .data import precache_classification_images

    class_names = list(cfg.classes)
    default_image_size = int(cfg.require("preprocessing.classification.image_size"))
    default_arch = str(cfg.get("classification.architecture", "resnet50"))
    tta_mode = cfg.get("evaluation.tta", False)
    specs = _resolve_checkpoint_specs(cfg, default_checkpoint)

    total_weight = 0.0
    blended_probs: Optional[np.ndarray] = None
    y_true_out: Optional[np.ndarray] = None
    meta_out: List[Dict[str, Any]] = []

    for ckpt_path, weight in specs:
        if not ckpt_path.is_file():
            raise EvaluationError(f"Missing classifier checkpoint at {ckpt_path}")
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        fp = payload.get("fingerprint") or {}
        if list(fp.get("classes") or []) and list(fp["classes"]) != class_names:
            raise EvaluationError(
                "The checkpoint's class order does not match the configured taxonomy: "
                f"checkpoint={fp.get('classes')}, config={class_names}"
            )
        if not cfg.get("evaluation.ensemble_checkpoints") and int(fp.get("image_size", default_image_size)) != default_image_size:
            raise EvaluationError(
                f"Checkpoint was trained at image_size={fp.get('image_size')} but the "
                f"config now says {default_image_size}."
            )
        arch = str(fp.get("architecture") or default_arch)
        img_size = int(fp.get("image_size") or default_image_size)
        pool_mode = str(fp.get("pooling") or "avg")
        contrast_mode = str(fp.get("enhance_contrast") or "none")
        precache_classification_images(
            rows, cfg, logger=logger, image_size=img_size, enhance_contrast=contrast_mode
        )

        model = build_classifier(
            architecture=arch,
            num_classes=len(class_names),
            pretrained=False,
            allow_weight_download=False,
            pooling=pool_mode,
        ).to(device)
        try:
            model.load_state_dict(payload["model_state"])
        except RuntimeError:
            dropout_guess = float(cfg.get("classification.dropout", 0.2) or 0.2)
            model = build_classifier(
                architecture=arch,
                num_classes=len(class_names),
                pretrained=False,
                allow_weight_download=False,
                dropout=dropout_guess,
                pooling=pool_mode,
            ).to(device)
            model.load_state_dict(payload["model_state"])
        model.eval()

        dataset = ClassifierDataset(
            rows, cfg, train=False, image_size=img_size, enhance_contrast=contrast_mode
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        probs_chunks: List[np.ndarray] = []
        y_list: List[int] = []
        collected_meta: List[Dict[str, Any]] = []
        with torch.no_grad():
            for inputs, labels, meta in loader:
                probs = predict_probs_with_tta(
                    model, inputs, tta=tta_mode, device=device, amp_enabled=amp_enabled
                )
                probs_chunks.append(probs.cpu().numpy())
                y_list.extend(int(v) for v in labels)
                if not meta_out:
                    count = inputs.size(0)
                    collected_meta.extend(
                        {
                            key: (meta[key][i] if isinstance(meta[key], (list, tuple)) else meta[key])
                            for key in meta
                        }
                        for i in range(count)
                    )

        member_probs = np.concatenate(probs_chunks, axis=0)
        if blended_probs is None:
            blended_probs = member_probs * weight
            y_true_out = np.asarray(y_list)
            meta_out = collected_meta
        else:
            blended_probs = blended_probs + (member_probs * weight)
        total_weight += weight
        if logger:
            logger.info(
                "Evaluated checkpoint %s (arch=%s, size=%d, weight=%.2f, tta=%s)",
                ckpt_path.name, arch, img_size, weight, tta_mode,
            )

    assert blended_probs is not None and y_true_out is not None
    return blended_probs / max(total_weight, 1e-8), y_true_out, meta_out


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

    probabilities, y_true, all_meta = evaluate_rows_with_checkpoints(
        cfg,
        rows,
        checkpoint_path,
        device=device,
        amp_enabled=bool(hardware.mixed_precision),
        batch_size=batch_size,
        num_workers=num_workers,
        logger=logger,
    )
    y_pred = probabilities.argmax(axis=1)
    ensemble_specs = cfg.get("evaluation.ensemble_checkpoints") or []
    if ensemble_specs:
        fingerprint = {
            "architecture": "ensemble",
            "members": ensemble_specs,
            "tta": cfg.get("evaluation.tta", False),
            "classes": class_names,
        }
        arch_label = f"Ensemble ({len(ensemble_specs)} models)"
    else:
        payload_meta = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        fingerprint = payload_meta.get("fingerprint") or {}
        arch_label = str(fingerprint.get("architecture") or cfg.get("classification.architecture", "Classifier"))

    metrics = classification_metrics(
        y_true, y_pred, probabilities=probabilities, class_names=class_names,
        compute_auc=bool(cfg.get("evaluation.roc_auc.enabled", True)),
    )
    metrics["partition"] = "test"
    metrics["checkpoint"] = str(checkpoint_path) if not ensemble_specs else "ensemble"
    metrics["checkpoint_fingerprint"] = fingerprint
    metrics["provisional_label_policy"] = bool(cfg.get("project.provisional_label_policy", True))
    metrics["integrity_note"] = (
        "Reported on the held-out test partition only. No threshold, checkpoint or "
        "hyper-parameter was chosen using this partition."
    )

    metrics_path = out_dir / "classification_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    _write_report_csv(out_dir / "classification_report.csv", metrics)

    matrix = np.asarray(metrics["confusion_matrix"])
    confusion_png = _plot_confusion(
        matrix, class_names, out_dir / "confusion_matrix.png",
        f"{arch_label} five-class confusion matrix (held-out test set)",
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
    }
    logger.info(
        "test: accuracy %.4f | macro F1 %.4f | kappa %.4f | AUC %s",
        metrics["accuracy"], metrics["macro_f1"], metrics["cohen_kappa"],
        summary["roc_auc_macro"],
    )
    if events is not None:
        events.write("classification_eval_done", **summary)
    return summary
