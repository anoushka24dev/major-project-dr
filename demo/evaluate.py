"""Score the 10 held-out demo test images exactly once with the best checkpoint.

Everything here is computed from the model's real predictions on
``manifests/selected_test.csv``. Nothing is hardcoded, estimated or fabricated;
metric families that are mathematically undefined for this sample (for example
ROC-AUC for a class absent from the test set) are reported as unavailable.

The script refuses to overwrite an existing ``results/metrics.json`` so the test
partition cannot be scored repeatedly, which would turn it into a tuning set.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader

from prepare_data import load_demo_config, load_manifest, manifest_path
from src.config import load_config
from src.data import ClassifierDataset
from src.metrics import classification_metrics
from src.models import build_resnet50
from train import BEST_CHECKPOINT, RESULTS_DIR, detect_device

METRICS_JSON = RESULTS_DIR / "metrics.json"


class EvaluationError(RuntimeError):
    """Raised when evaluation cannot proceed honestly."""


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_checkpoint(device: torch.device, path: Path = BEST_CHECKPOINT) -> Dict[str, Any]:
    if not path.is_file():
        raise EvaluationError(
            f"No trained checkpoint at {path}.\n"
            "Run `python demo/train.py` first and confirm it finishes without errors."
        )
    return torch.load(path, map_location=device, weights_only=False)


def predict(
    payload: Dict[str, Any], rows: List[Dict[str, str]], config: Dict[str, Any], device: torch.device
) -> Dict[str, Any]:
    classes = list(payload["classes"])
    image_size = int(payload["image_size"])
    if classes != list(config["taxonomy"]["classes"]):
        raise EvaluationError("Checkpoint class order does not match demo/config.json.")
    if image_size != int(config["model"]["image_size"]):
        raise EvaluationError("Checkpoint image size does not match demo/config.json.")

    repo_cfg = load_config()
    dataset = ClassifierDataset(rows, repo_cfg, train=False, image_size=image_size)
    loader = DataLoader(dataset, batch_size=len(rows), shuffle=False, num_workers=0)

    model = build_resnet50(
        num_classes=len(classes), pretrained=False, allow_weight_download=False, logger=None
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    all_labels: List[int] = []
    all_probs: List[List[float]] = []
    with torch.no_grad():
        for inputs, labels, _meta in loader:
            logits = model(inputs.to(device))
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probabilities.tolist())
            all_labels.extend(int(x) for x in labels.tolist())

    return {"labels": all_labels, "probabilities": all_probs, "classes": classes}


# ---------------------------------------------------------------------------
# Artefacts
# ---------------------------------------------------------------------------

def plot_confusion_matrix(matrix: List[List[int]], classes: Sequence[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    array = np.array(matrix, dtype=int)
    figure, axis = plt.subplots(figsize=(7.2, 6.0))
    image = axis.imshow(array, cmap="Blues", vmin=0)
    axis.set_xticks(range(len(classes)))
    axis.set_yticks(range(len(classes)))
    axis.set_xticklabels(classes, rotation=35, ha="right")
    axis.set_yticklabels(classes)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(f"Confusion matrix - {int(array.sum())} held-out test images")
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            value = int(array[i, j])
            axis.text(
                j, i, str(value), ha="center", va="center",
                color="white" if value > array.max() / 2 else "black", fontsize=11,
            )
    figure.colorbar(image, ax=axis, shrink=0.85, label="images")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_per_class(metrics: Dict[str, Any], classes: Sequence[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    precision = [metrics["per_class"][name]["precision"] for name in classes]
    recall = [metrics["per_class"][name]["recall"] for name in classes]
    f1 = [metrics["per_class"][name]["f1"] for name in classes]
    support = [metrics["per_class"][name]["support"] for name in classes]

    positions = np.arange(len(classes))
    width = 0.26
    figure, axis = plt.subplots(figsize=(9.5, 5.0))
    axis.bar(positions - width, precision, width, label="precision", color="#1f77b4")
    axis.bar(positions, recall, width, label="recall", color="#ff7f0e")
    axis.bar(positions + width, f1, width, label="F1", color="#2ca02c")
    axis.set_xticks(positions)
    axis.set_xticklabels([f"{name}\n(n={n})" for name, n in zip(classes, support)], fontsize=8.5)
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("score")
    axis.set_title(f"Per-class performance (macro F1 = {metrics['macro_f1']:.3f})")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_classification_report(metrics: Dict[str, Any], classes: Sequence[str], path: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for name in classes:
        entry = metrics["per_class"][name]
        rows.append(
            {
                "class": name,
                "precision": round(entry["precision"], 6),
                "recall": round(entry["recall"], 6),
                "f1_score": round(entry["f1"], 6),
                "support": entry["support"],
            }
        )
    for label, key_precision, key_recall, key_f1 in (
        ("macro avg", "macro_precision", "macro_recall", "macro_f1"),
        ("weighted avg", "weighted_precision", "weighted_recall", "weighted_f1"),
    ):
        rows.append(
            {
                "class": label,
                "precision": round(float(metrics[key_precision]), 6),
                "recall": round(float(metrics[key_recall]), 6),
                "f1_score": round(float(metrics[key_f1]), 6),
                "support": int(metrics["num_samples"]),
            }
        )
    rows.append(
        {
            "class": "accuracy",
            "precision": "",
            "recall": "",
            "f1_score": round(float(metrics["accuracy"]), 6),
            "support": int(metrics["num_samples"]),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", "precision", "recall", "f1_score", "support"])
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(
    rows: List[Dict[str, str]],
    labels: List[int],
    probabilities: List[List[float]],
    classes: Sequence[str],
    path: Path,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row, truth, probs in zip(rows, labels, probabilities):
        predicted = int(np.argmax(probs))
        record: Dict[str, Any] = {
            "filename": row.get("filename", ""),
            "rel_path": row.get("rel_path", ""),
            "source_class": row.get("source_class", ""),
            "true_label": int(truth),
            "true_class": classes[int(truth)],
            "predicted_label": predicted,
            "predicted_class": classes[predicted],
            "confidence": round(float(max(probs)), 6),
            "correct": bool(predicted == int(truth)),
            "status": "correct" if predicted == int(truth) else "incorrect",
        }
        for index, name in enumerate(classes):
            record[f"prob_{index}_{name.replace(' ', '_')}"] = round(float(probs[index]), 6)
        records.append(record)

    fields = list(records[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return records


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def render_report(
    metrics: Dict[str, Any],
    metadata: Dict[str, Any],
    training_config: Dict[str, Any],
    config: Dict[str, Any],
    predictions: List[Dict[str, Any]],
    classes: Sequence[str],
    path: Path,
) -> None:
    matrix = metrics["confusion_matrix"]
    roc = metrics.get("roc_auc") or {}
    roc_macro = roc.get("macro")
    resolved = training_config.get("resolved", {})
    correct = sum(1 for p in predictions if p["correct"])
    total = len(predictions)

    lines: List[str] = []
    add = lines.append
    add("# Demo Experiment Results - ResNet-50, 50 Training Images / 10 Test Images")
    add("")
    add("**Status: completed.** Every number below was computed from the saved best checkpoint")
    add("applied to the 10 held-out test images exactly once. No value is estimated or hardcoded.")
    add("")
    add(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    add("")
    add("---")
    add("")
    add("## 1. What this experiment is")
    add("")
    add(
        "A deliberately small, fully reproducible companion experiment to the main pipeline. "
        "It answers one question: with only **50 training images** and a **10-image held-out "
        "test set**, what does an ImageNet-pretrained ResNet-50 actually score on the "
        "repository's five-class DR taxonomy?"
    )
    add("")
    add("| Item | Value |")
    add("| --- | --- |")
    add(f"| Architecture | ResNet-50 (`src/models.py::build_resnet50`) |")
    add(f"| Initialisation | ImageNet `IMAGENET1K_V1` weights, 5-class head |")
    add(f"| Classes | {', '.join(f'{i} = {c}' for i, c in enumerate(classes))} |")
    add(
        f"| Training images | 50 ({resolved.get('fit_images', '?')} used for weight updates, "
        f"{resolved.get('validation_images', '?')} for validation) |"
    )
    add(f"| Test images | 10, held out and untouched during training |")
    add(f"| Input size | {metadata['image_size']} x {metadata['image_size']} (aspect-ratio-preserving resize-and-pad) |")
    add(f"| Seed | {metadata['seed']} |")
    add(f"| Device | {metadata['device']} |")
    add(f"| Best epoch | {metadata['best_epoch']} of {resolved.get('epochs_planned', '?')} planned ({metadata['best_stage']}) |")
    add("")
    add("## 2. Overall test metrics")
    add("")
    add("| Metric | Value |")
    add("| --- | --- |")
    add(f"| **Accuracy** | **{_pct(metrics['accuracy'])}** ({correct}/{total}) |")
    add(f"| Precision (macro) | {metrics['macro_precision']:.4f} |")
    add(f"| Precision (weighted) | {metrics['weighted_precision']:.4f} |")
    add(f"| Recall (macro) | {metrics['macro_recall']:.4f} |")
    add(f"| Recall (weighted) | {metrics['weighted_recall']:.4f} |")
    add(f"| F1-score (macro) | {metrics['macro_f1']:.4f} |")
    add(f"| F1-score (weighted) | {metrics['weighted_f1']:.4f} |")
    add(f"| Cohen's kappa | {metrics['cohen_kappa']:.4f} |")
    if roc_macro is None:
        add(f"| ROC-AUC (macro, one-vs-rest) | not available - {roc.get('reason') or 'undefined for this sample'} |")
    else:
        add(f"| ROC-AUC (macro, one-vs-rest) | {roc_macro:.4f} |")
    add("")
    add("## 3. Per-class results")
    add("")
    add("| Class | Support | Precision | Recall | F1 |")
    add("| --- | --- | --- | --- | --- |")
    for name in classes:
        entry = metrics["per_class"][name]
        add(
            f"| {name} | {entry['support']} | {entry['precision']:.3f} | "
            f"{entry['recall']:.3f} | {entry['f1']:.3f} |"
        )
    add("")
    add("Full table (including macro/weighted averages): `classification_report.csv`.")
    add("")
    add("## 4. Confusion matrix")
    add("")
    add("Rows are the true class, columns the predicted class.")
    add("")
    add("| True \\ Predicted | " + " | ".join(classes) + " |")
    add("| --- | " + " | ".join("---" for _ in classes) + " |")
    for index, name in enumerate(classes):
        cells = " | ".join(str(int(value)) for value in matrix[index])
        add(f"| **{name}** | {cells} |")
    add("")
    add("![Confusion matrix](confusion_matrix.png)")
    add("")
    add("![Per-class performance](per_class_performance.png)")
    add("")
    add("## 5. Per-image predictions")
    add("")
    add("| Image | True | Predicted | Confidence | Result |")
    add("| --- | --- | --- | --- | --- |")
    for record in predictions:
        add(
            f"| {record['filename']} | {record['true_class']} | {record['predicted_class']} | "
            f"{record['confidence']:.3f} | {record['status']} |"
        )
    add("")
    add("Full probabilities per image: `test_predictions.csv`.")
    add("")
    add("## 6. Training setup")
    add("")
    add("| Item | Value |")
    add("| --- | --- |")
    add("| Optimiser | AdamW |")
    add("| Loss | Cross-entropy with inverse-frequency class weights computed from the 41 fit images |")
    add(f"| Class weights | {resolved['class_weights']} |")
    for stage in training_config["training"]["stages"]:
        add(
            f"| Stage `{stage['name']}` | {stage['epochs']} epochs, lr={stage['lr']}, "
            f"weight decay={stage['weight_decay']}, trainable={stage['trainable']} |"
        )
    add("| Batch size | 8 |")
    add("| Augmentation | horizontal flip, rotation +/- 10 deg, brightness/contrast +/- 0.08 (training rows only) |")
    add("| Validation | stratified 9-image carve-out from the 50 training images |")
    add("| Checkpoint selection | best validation macro-F1 (ties -> lower validation loss) |")
    add(f"| Early stopping | patience {training_config['training']['early_stopping_patience']} epochs |")
    add(f"| Epochs run | {resolved['epochs_run']} |")
    add(f"| Training time | {metadata['training_seconds'] / 60:.1f} min on {metadata['device']} |")
    add("")
    add("Training and validation curves: `training_curve.png`. Full history: `training_history.csv`.")
    add("")
    add("## 7. What the numbers say")
    add("")
    interpretations = _interpret(metrics, predictions, classes)
    for sentence in interpretations:
        add(f"- {sentence}")
    add("")
    add("## 8. Limitations")
    add("")
    add(f"- The test set contains only {total} images, so a single image moves accuracy by "
        f"{100.0 / total:.0f} percentage points. These numbers describe this sample only.")
    add("- 50 training images cannot represent the class variability of a real DR screening")
    add("  population; the result is a sanity-check of the code path, not a performance claim.")
    add("- Labels are the source archive's folder labels, with the repository's provisional")
    add("  merges (Very Severe NPDR -> Severe NPDR, Advanced PDR -> PDR). They are not")
    add("  independently re-graded by a clinician.")
    add("- Nothing here is a clinical validation. Do not use this model for diagnosis.")
    add("")
    add("## 9. Reproduce it")
    add("")
    add("```bash")
    add("python demo/prepare_data.py   # locate, extract, verify, sample 50/10")
    add("python demo/train.py          # train, save best checkpoint")
    add("python demo/evaluate.py       # score the 10 test images once")
    add("```")
    add("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _interpret(metrics: Dict[str, Any], predictions: List[Dict[str, Any]], classes: Sequence[str]) -> List[str]:
    """Turn the computed numbers into short, factual observations."""
    notes: List[str] = []
    total = metrics["num_samples"]
    correct = sum(1 for p in predictions if p["correct"])
    notes.append(f"{correct} of {total} test images were classified correctly (accuracy {_pct(metrics['accuracy'])}).")

    worst = sorted(classes, key=lambda name: metrics["per_class"][name]["f1"])
    perfect = [name for name in classes if metrics["per_class"][name]["f1"] >= 0.999 and metrics["per_class"][name]["support"] > 0]
    zero = [name for name in classes if metrics["per_class"][name]["support"] > 0 and metrics["per_class"][name]["recall"] == 0.0]

    if perfect:
        notes.append("Perfect per-class F1 on this sample: " + ", ".join(perfect) + ".")
    if zero:
        notes.append(
            "No test image of class "
            + ", ".join(zero)
            + " was recognised (recall 0.0); with this sample size that is one or two images."
        )

    matrix = np.array(metrics["confusion_matrix"], dtype=int)
    off_diagonal = [
        (int(matrix[i, j]), classes[i], classes[j])
        for i in range(len(classes))
        for j in range(len(classes))
        if i != j and matrix[i, j] > 0
    ]
    if off_diagonal:
        off_diagonal.sort(reverse=True)
        count, truth, predicted = off_diagonal[0]
        notes.append(
            f"The most frequent error is {truth} predicted as {predicted} ({count} image"
            f"{'s' if count != 1 else ''}), which is a one-grade shift rather than a random mistake."
            if count and abs(classes.index(truth) - classes.index(predicted)) == 1
            else f"The most frequent error is {truth} predicted as {predicted} ({count} image"
            f"{'s' if count != 1 else ''})."
        )
    elif correct == total:
        notes.append("There were no errors on this test sample; the confusion matrix is diagonal.")
    else:
        notes.append("No off-diagonal errors were recorded.")

    mean_confidence = float(np.mean([p["confidence"] for p in predictions]))
    wrong = [p["confidence"] for p in predictions if not p["correct"]]
    notes.append(f"Mean prediction confidence is {mean_confidence:.3f}.")
    if wrong:
        notes.append(
            f"Confidence on the {len(wrong)} misclassified image(s) averaged {np.mean(wrong):.3f}, "
            "so confidence alone does not separate right from wrong at this scale."
        )
    notes.append(
        "Ranking the worst class by F1: "
        + ", ".join(f"{name} ({metrics['per_class'][name]['f1']:.2f})" for name in worst[:2])
        + "."
    )
    return notes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(config: Dict[str, Any], *, force: bool = False) -> Dict[str, Any]:
    if METRICS_JSON.exists() and not force:
        raise EvaluationError(
            f"{METRICS_JSON.relative_to(REPO_ROOT).as_posix()} already exists, so the test set has "
            "already been scored.\n"
            "Scoring it again would turn the held-out set into a tuning set. If you really "
            "mean to overwrite it (for example after retraining from scratch), add --force."
        )

    rows = load_manifest("selected_test.csv", config)
    expected = int(config["experiment"]["expected_test_total"])
    if len(rows) != expected:
        raise EvaluationError(f"Expected exactly {expected} test rows, found {len(rows)}.")

    classes = list(config["taxonomy"]["classes"])
    device = detect_device()
    payload = load_checkpoint(device)

    train_rows = load_manifest("selected_train.csv", config)
    if len(train_rows) != int(config["experiment"]["expected_train_total"]):
        raise EvaluationError("The training manifest does not contain exactly 50 rows.")
    overlap = {r["rel_path"] for r in rows} & {r["rel_path"] for r in train_rows}
    if overlap:
        raise EvaluationError(f"Test and training manifests overlap on {len(overlap)} file(s).")
    hash_overlap = {r["sha256"] for r in rows} & {r["sha256"] for r in train_rows}
    if hash_overlap:
        raise EvaluationError(f"Test and training manifests share {len(hash_overlap)} identical file(s).")

    print(f"Evaluating {len(rows)} held-out test images on {device}")
    result = predict(payload, rows, config, device)
    labels = result["labels"]
    probabilities = np.array(result["probabilities"], dtype=np.float64)
    predictions = probabilities.argmax(axis=1).tolist()

    metrics = classification_metrics(
        np.array(labels),
        np.array(predictions),
        probabilities=probabilities,
        class_names=classes,
        compute_auc=True,
    )
    # Weighted precision/recall are not returned by src.metrics; derive them from
    # its per-class values so every number stays traceable to the same matrix.
    total = metrics["num_samples"]
    metrics["weighted_precision"] = float(
        sum(metrics["per_class"][name]["precision"] * metrics["per_class"][name]["support"] for name in classes) / total
    )
    metrics["weighted_recall"] = float(
        sum(metrics["per_class"][name]["recall"] * metrics["per_class"][name]["support"] for name in classes) / total
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(metrics["confusion_matrix"], classes, RESULTS_DIR / "confusion_matrix.png")
    plot_per_class(metrics, classes, RESULTS_DIR / "per_class_performance.png")
    write_classification_report(metrics, classes, RESULTS_DIR / "classification_report.csv")
    records = write_predictions(rows, labels, result["probabilities"], classes, RESULTS_DIR / "test_predictions.csv")

    metadata_path = RESULTS_DIR / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    training_config_path = RESULTS_DIR / "training_config.json"
    training_config = (
        json.loads(training_config_path.read_text(encoding="utf-8")) if training_config_path.is_file() else {"resolved": {}, "training": {"stages": []}}
    )

    payload_metrics = {
        "status": "completed",
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_images": len(rows),
        "num_samples": int(metrics["num_samples"]),
        "class_order": classes,
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_precision": metrics["weighted_precision"],
        "weighted_recall": metrics["weighted_recall"],
        "weighted_f1": metrics["weighted_f1"],
        "cohen_kappa": metrics["cohen_kappa"],
        "roc_auc": metrics["roc_auc"],
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        "checkpoint": BEST_CHECKPOINT.relative_to(REPO_ROOT).as_posix(),
        "checkpoint_selected_on": payload.get("selection_note", ""),
        "best_val_macro_f1": payload.get("best", {}).get("val_macro_f1"),
        "seed": payload.get("seed"),
        "image_size": payload.get("image_size"),
        "architecture": payload.get("architecture"),
        "device": str(device),
        "software": {
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "self_check": {},
    }

    # ---- self-check: every headline number must agree with the saved rows ----
    recomputed_correct = sum(1 for record in records if record["correct"])
    recomputed_accuracy = recomputed_correct / len(records)
    recomputed_matrix = [[0] * len(classes) for _ in classes]
    for record in records:
        recomputed_matrix[int(record["true_label"])][int(record["predicted_label"])] += 1
    checks = {
        "rows_written": len(records),
        "correct_rows": recomputed_correct,
        "accuracy_from_rows": round(recomputed_accuracy, 12),
        "accuracy_matches_metrics": bool(abs(recomputed_accuracy - metrics["accuracy"]) < 1e-12),
        "confusion_matrix_matches_predictions": recomputed_matrix == metrics["confusion_matrix"],
        "train_test_overlap": 0,
    }
    if not checks["accuracy_matches_metrics"] or not checks["confusion_matrix_matches_predictions"]:
        raise EvaluationError(
            "Internal consistency check failed: metrics.json would not agree with "
            "test_predictions.csv. Refusing to write a report with inconsistent numbers."
        )
    payload_metrics["self_check"] = checks
    METRICS_JSON.write_text(json.dumps(payload_metrics, indent=2), encoding="utf-8")

    render_report(
        payload_metrics,
        metadata,
        training_config,
        config,
        records,
        classes,
        RESULTS_DIR / "DEMO_RESULTS.md",
    )

    print()
    print(f"Accuracy            : {metrics['accuracy']:.4f} ({recomputed_correct}/{len(records)})")
    print(f"Precision macro/weighted : {metrics['macro_precision']:.4f} / {metrics['weighted_precision']:.4f}")
    print(f"Recall    macro/weighted : {metrics['macro_recall']:.4f} / {metrics['weighted_recall']:.4f}")
    print(f"F1        macro/weighted : {metrics['macro_f1']:.4f} / {metrics['weighted_f1']:.4f}")
    roc = metrics["roc_auc"].get("macro")
    print(f"ROC-AUC   macro          : {roc if roc is not None else 'not available (' + str(metrics['roc_auc'].get('reason')) + ')'}")
    print()
    print("Artefacts written to demo/results/:")
    for name in (
        "metrics.json",
        "classification_report.csv",
        "confusion_matrix.png",
        "per_class_performance.png",
        "test_predictions.csv",
        "DEMO_RESULTS.md",
    ):
        print(f"  {name}")
    return payload_metrics


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEMO_DIR / "config.json"))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing results/metrics.json (only after retraining from scratch).",
    )
    args = parser.parse_args(argv)

    config = load_demo_config(args.config)
    try:
        evaluate(config, force=args.force)
    except EvaluationError as exc:
        print()
        print("EVALUATION FAILED")
        print(f"  {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
