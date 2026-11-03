"""Classification, segmentation, calibration and uncertainty metrics.

All functions are pure NumPy and take plain arrays, so they are testable without
a dataset or a GPU. Nothing here fabricates values: metric families that are not
mathematically defined for the given inputs return ``None`` with a reason, rather
than a plausible-looking number.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-12


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Integer confusion matrix with rows = truth, columns = prediction."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, prediction in zip(np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()):
        if 0 <= int(truth) < num_classes and 0 <= int(prediction) < num_classes:
            matrix[int(truth), int(prediction)] += 1
    return matrix


def cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Cohen's kappa for multi-class agreement."""
    matrix = confusion_matrix(y_true, y_pred, num_classes).astype(np.float64)
    total = matrix.sum()
    if total <= 0:
        return 0.0
    observed = np.trace(matrix) / total
    row_totals = matrix.sum(axis=1) / total
    col_totals = matrix.sum(axis=0) / total
    expected = float((row_totals * col_totals).sum())
    if abs(1.0 - expected) < EPS:
        return 0.0
    return float((observed - expected) / (1.0 - expected))


def one_vs_rest_auc(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    num_classes: int,
) -> Dict[str, Any]:
    """One-vs-rest ROC-AUC per class plus the macro average.

    Returns ``{"per_class": {...}, "macro": float|None, "reason": str}``. A class
    with only one outcome present makes AUC undefined for that class; it is
    reported as ``None`` with a reason instead of being invented.
    """
    y_true = np.asarray(y_true).ravel()
    probabilities = np.asarray(probabilities, dtype=np.float64)
    per_class: Dict[str, Any] = {}
    macro_values: List[float] = []
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return {
            "per_class": {},
            "macro": None,
            "reason": "scikit-learn is not installed, so ROC-AUC was not computed",
        }

    for index in range(num_classes):
        binary = (y_true == index).astype(int)
        positives = int(binary.sum())
        if positives == 0 or positives == len(binary):
            per_class[str(index)] = None
            continue
        try:
            score = float(roc_auc_score(binary, probabilities[:, index]))
            per_class[str(index)] = score
            macro_values.append(score)
        except Exception as exc:  # pragma: no cover - defensive
            per_class[str(index)] = None
            per_class.setdefault("errors", {})[str(index)] = str(exc)

    macro = float(np.mean(macro_values)) if macro_values else None
    reason = "" if macro is not None else "not enough class diversity in the evaluation set"
    return {"per_class": per_class, "macro": macro, "reason": reason}


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    probabilities: Optional[np.ndarray] = None,
    class_names: Sequence[str],
    compute_auc: bool = True,
) -> Dict[str, Any]:
    """Full classification metric set for the five-class task."""
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)
    num_classes = len(class_names)
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    total = int(matrix.sum())

    per_class: Dict[str, Dict[str, Any]] = {}
    for index, name in enumerate(class_names):
        tp = int(matrix[index, index])
        fn = int(matrix[index, :].sum() - tp)
        fp = int(matrix[:, index].sum() - tp)
        tn = int(total - tp - fn - fp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "sensitivity": float(recall),
            "specificity": float(specificity),
            "f1": float(f1),
            "support": int(matrix[index, :].sum()),
            "predicted": int(matrix[:, index].sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    macro_precision = float(np.mean([per_class[n]["precision"] for n in class_names]))
    macro_recall = float(np.mean([per_class[n]["recall"] for n in class_names]))
    macro_f1 = float(np.mean([per_class[n]["f1"] for n in class_names]))
    macro_specificity = float(np.mean([per_class[n]["specificity"] for n in class_names]))
    accuracy = float(np.trace(matrix) / total) if total else 0.0

    result: Dict[str, Any] = {
        "num_samples": total,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "macro_sensitivity": macro_recall,
        "macro_specificity": macro_specificity,
        "weighted_f1": float(
            sum(per_class[n]["f1"] * per_class[n]["support"] for n in class_names) / total
        ) if total else 0.0,
        "cohen_kappa": cohen_kappa(y_true, y_pred, num_classes),
        "per_class": per_class,
        "class_order": list(class_names),
        "confusion_matrix": matrix.tolist(),
    }

    if compute_auc:
        if probabilities is None:
            result["roc_auc"] = {
                "per_class": {},
                "macro": None,
                "reason": "no probabilities supplied",
            }
        else:
            result["roc_auc"] = one_vs_rest_auc(probabilities, y_true, num_classes)
    return result


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def dice_per_channel(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    thresholds: Sequence[float] | float = 0.5,
    smooth: float = 1e-6,
) -> np.ndarray:
    """Per-channel Dice coefficient for ``(N, C, H, W)`` probability tensors."""
    probs = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    channels = probs.shape[1]
    if np.isscalar(thresholds):
        thresholds = [float(thresholds)] * channels
    scores = np.zeros(channels, dtype=np.float64)
    for channel in range(channels):
        predicted = (probs[:, channel] >= float(thresholds[channel])).astype(np.float64)
        truth = targets[:, channel]
        intersection = float((predicted * truth).sum())
        denominator = float(predicted.sum() + truth.sum())
        scores[channel] = (2.0 * intersection + smooth) / (denominator + smooth)
    return scores


def iou_per_channel(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    thresholds: Sequence[float] | float = 0.5,
    smooth: float = 1e-6,
) -> np.ndarray:
    """Per-channel IoU (Jaccard) for ``(N, C, H, W)`` probability tensors."""
    probs = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    channels = probs.shape[1]
    if np.isscalar(thresholds):
        thresholds = [float(thresholds)] * channels
    scores = np.zeros(channels, dtype=np.float64)
    for channel in range(channels):
        predicted = (probs[:, channel] >= float(thresholds[channel])).astype(np.float64)
        truth = targets[:, channel]
        intersection = float((predicted * truth).sum())
        union = float(predicted.sum() + truth.sum() - intersection)
        scores[channel] = (intersection + smooth) / (union + smooth)
    return scores


def segmentation_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    channel_names: Sequence[str],
    *,
    thresholds: Sequence[float] | float = 0.5,
) -> Dict[str, Any]:
    """Per-lesion Dice/IoU plus macro and micro aggregates."""
    dice = dice_per_channel(probabilities, targets, thresholds=thresholds)
    iou = iou_per_channel(probabilities, targets, thresholds=thresholds)
    per_channel = {
        name: {"dice": float(dice[i]), "iou": float(iou[i])}
        for i, name in enumerate(channel_names)
    }
    return {
        "per_channel": per_channel,
        "macro_dice": float(np.mean(dice)),
        "macro_iou": float(np.mean(iou)),
        "thresholds": (
            [float(t) for t in thresholds] if not np.isscalar(thresholds) else float(thresholds)
        ),
        "channels": list(channel_names),
    }


def binarise_cam(cam: np.ndarray, percentile: float) -> np.ndarray:
    """Threshold a CAM at a percentile of its own activation distribution."""
    cam = np.asarray(cam, dtype=np.float64)
    if cam.size == 0:
        return np.zeros_like(cam, dtype=bool)
    threshold = float(np.percentile(cam, float(percentile)))
    out = cam >= threshold
    if not out.any():  # degenerate CAM: fall back to the single hottest pixel
        out = cam >= cam.max()
    return out


def mask_overlap_scores(
    binary_prediction: np.ndarray,
    binary_truth: np.ndarray,
    *,
    smooth: float = 1e-6,
) -> Dict[str, float]:
    """Dice and IoU between two boolean masks."""
    predicted = np.asarray(binary_prediction, dtype=bool)
    truth = np.asarray(binary_truth, dtype=bool)
    intersection = float(np.logical_and(predicted, truth).sum())
    predicted_sum = float(predicted.sum())
    truth_sum = float(truth.sum())
    dice = (2.0 * intersection + smooth) / (predicted_sum + truth_sum + smooth)
    union = predicted_sum + truth_sum - intersection
    iou = (intersection + smooth) / (union + smooth)
    return {"dice": float(dice), "iou": float(iou)}


# ---------------------------------------------------------------------------
# Calibration and uncertainty
# ---------------------------------------------------------------------------

def predictive_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Shannon entropy (nats) of each row of class probabilities."""
    probs = np.asarray(probabilities, dtype=np.float64)
    probs = np.clip(probs, EPS, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)


def calibration_bins(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    *,
    bins: int = 15,
    strategy: str = "equal_width",
) -> Dict[str, Any]:
    """Bin-level calibration statistics for the reliability diagram."""
    probs = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(y_true).ravel().astype(int)
    if strategy != "equal_width":
        raise ValueError(
            f"Unsupported calibration binning strategy {strategy!r}. "
            "Only 'equal_width' is implemented; the pipeline records this explicitly."
        )
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y_true).astype(np.float64)

    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    rows: List[Dict[str, Any]] = []
    ece = 0.0
    total = len(confidences)
    for index in range(int(bins)):
        low, high = float(edges[index]), float(edges[index + 1])
        if index == int(bins) - 1:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {"bin": index, "lower": low, "upper": high, "count": 0,
                 "mean_confidence": None, "accuracy": None, "gap": None}
            )
            continue
        mean_confidence = float(confidences[mask].mean())
        accuracy = float(correct[mask].mean())
        gap = abs(accuracy - mean_confidence)
        ece += (count / total) * gap if total else 0.0
        rows.append(
            {
                "bin": index,
                "lower": low,
                "upper": high,
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "gap": gap,
            }
        )
    return {"bins": rows, "ece": float(ece), "binning_strategy": "equal_width", "num_bins": int(bins)}


def expected_calibration_error(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    *,
    bins: int = 15,
    strategy: str = "equal_width",
) -> float:
    """Expected Calibration Error, equal-width binning over max-probability."""
    return calibration_bins(probabilities, y_true, bins=bins, strategy=strategy)["ece"]


def uncertainty_report(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    *,
    entropy_threshold: float,
    confidence_threshold: float,
    max_entropy: Optional[float] = None,
) -> Dict[str, Any]:
    """Summarise uncertainty and flag predictions that need human review."""
    probs = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(y_true).ravel().astype(int)
    confidences = probs.max(axis=1)
    entropy = predictive_entropy(probs)
    predictions = probs.argmax(axis=1)

    low_confidence = confidences < float(confidence_threshold)
    high_entropy = entropy > float(entropy_threshold)
    flagged = low_confidence | high_entropy

    correct = predictions == y_true
    return {
        "num_samples": int(len(probs)),
        "confidence": {
            "mean": float(confidences.mean()),
            "median": float(np.median(confidences)),
            "min": float(confidences.min()),
            "max": float(confidences.max()),
        },
        "entropy": {
            "mean": float(entropy.mean()),
            "median": float(np.median(entropy)),
            "max": float(entropy.max()),
            "max_possible_nats": float(np.log(probs.shape[1])),
        },
        "thresholds": {
            "confidence_threshold": float(confidence_threshold),
            "entropy_threshold": float(entropy_threshold),
        },
        "review_flags": {
            "low_confidence": int(low_confidence.sum()),
            "high_entropy": int(high_entropy.sum()),
            "either": int(flagged.sum()),
            "fraction_flagged": float(flagged.mean()),
        },
        "accuracy_split_by_flag": {
            "flagged": float(correct[flagged].mean()) if flagged.any() else None,
            "not_flagged": float(correct[~flagged].mean()) if (~flagged).any() else None,
        },
        "policy_note": (
            "A flagged prediction is routed to human review. It is NOT a confirmed "
            "diagnosis and the flag is not a clinical decision rule."
        ),
    }
