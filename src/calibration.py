"""Calibration and uncertainty analysis of the held-out test predictions.

ECE binning strategy
--------------------
Equal-width binning of the **max softmax probability** (confidence) into
``evaluation.calibration.bins`` bins on [0, 1], with the top bin closed on the
right. ECE is the sample-weighted mean gap between bin accuracy and bin mean
confidence. This is stated in the output JSON as
``calibration.binning_strategy = "equal_width"`` so the number is reproducible and
comparable. No other binning scheme is silently substituted.

Uncertainty
-----------
Two complementary signals are reported: softmax confidence and predictive
entropy in nats (maximum ``ln 5 = 1.609`` for five classes). Predictions that
cross the configured thresholds are flagged for human review. A flagged
prediction is *not* a diagnosis and the flag is not a clinical decision rule.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .config import Config
from .metrics import calibration_bins, uncertainty_report


class CalibrationError(RuntimeError):
    """Raised when calibration cannot be computed from real predictions."""


def _reliability_diagram(bins: Dict[str, Any], ece: float, path: Path) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    rows = bins["bins"]
    centres, accuracies, confidences, counts = [], [], [], []
    for row in rows:
        if row["count"] == 0:
            continue
        centres.append((row["lower"] + row["upper"]) / 2.0)
        accuracies.append(row["accuracy"])
        confidences.append(row["mean_confidence"])
        counts.append(row["count"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.4))
    ax1.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="perfect calibration")
    if centres:
        ax1.plot(confidences, accuracies, marker="o", linewidth=1.5, label="model")
        ax1.bar(confidences, accuracies, width=0.05, alpha=0.25)
    ax1.set_xlabel("Mean predicted confidence")
    ax1.set_ylabel("Empirical accuracy")
    ax1.set_title(f"Reliability diagram (ECE = {ece:.4f})")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=8)

    if counts:
        ax2.bar(centres, counts, width=0.9 / max(1, len(centres)))
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Number of predictions")
    ax2.set_title("Confidence histogram")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _posthoc_temperature(cfg: Config, classification_dir: Path) -> Optional[float]:
    """Return the validation-fitted temperature for the published checkpoint, if any."""
    provenance = Path(classification_dir) / "checkpoint_provenance.json"
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
    temperature = payload.get("temperature")
    try:
        return float(temperature) if temperature else None
    except (TypeError, ValueError):
        return None


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def run_calibration(cfg: Config, logger, events=None) -> Dict[str, Any]:
    out_dir = cfg.outputs_dir("evaluation")
    probability_path = cfg.outputs_dir("classification") / "test_probabilities.npz"
    if not probability_path.is_file():
        raise CalibrationError(
            f"Test probabilities not found at {probability_path}. "
            "Run the classification evaluation stage first."
        )

    data = np.load(probability_path, allow_pickle=True)
    probabilities = data["probabilities"]
    y_true = data["y_true"]
    y_pred = data["y_pred"]
    if probabilities.size == 0:
        raise CalibrationError("Test probabilities file is empty.")

    bins_count = int(cfg.get("evaluation.calibration.bins", 15))
    strategy = str(cfg.get("evaluation.calibration.strategy", "equal_width"))
    binning = calibration_bins(probabilities, y_true, bins=bins_count, strategy=strategy)

    entropy_threshold = float(cfg.get("evaluation.uncertainty.entropy_threshold", 1.10))
    confidence_threshold = float(cfg.get("evaluation.uncertainty.confidence_threshold", 0.60))
    uncertainty = uncertainty_report(
        probabilities, y_true,
        entropy_threshold=entropy_threshold,
        confidence_threshold=confidence_threshold,
    )

    overall_accuracy = float((np.asarray(y_pred) == np.asarray(y_true)).mean())

    # E6: if the published checkpoint has a validation-fitted temperature, report the
    # calibrated ECE next to the unscaled one. The headline ECE stays unscaled so the
    # two are never confused.
    temperature_scaled: Optional[Dict[str, Any]] = None
    temperature = _posthoc_temperature(cfg, probability_path.parent)
    if temperature and "logits" in data:
        scaled_probabilities = _softmax_rows(np.asarray(data["logits"]) / temperature)
        scaled_binning = calibration_bins(
            scaled_probabilities, y_true, bins=bins_count, strategy=strategy
        )
        temperature_scaled = {
            "temperature": temperature,
            "ece": scaled_binning["ece"],
            "accuracy": float(
                (scaled_probabilities.argmax(axis=1) == np.asarray(y_true)).mean()
            ),
            "note": (
                "Temperature fitted on the validation fold. Temperature scaling has one "
                "parameter and cannot change the predicted class, so accuracy should be "
                "unchanged; only confidence and ECE move."
            ),
        }

    diagram_path = _reliability_diagram(binning, binning["ece"], out_dir / "reliability_diagram.png")

    result = {
        "partition": "test",
        "num_samples": int(len(y_true)),
        "accuracy": overall_accuracy,
        "calibration": {
            "ece": binning["ece"],
            "num_bins": binning["num_bins"],
            "binning_strategy": binning["binning_strategy"],
            "binning_description": (
                "Equal-width bins over the max softmax probability in [0, 1]; the final bin "
                "includes 1.0. ECE is the count-weighted |accuracy - mean confidence|."
            ),
            "bins": binning["bins"],
            **(
                {"temperature_scaled": temperature_scaled}
                if temperature_scaled
                else {}
            ),
        },
        "uncertainty": uncertainty,
        "review_policy": {
            "confidence_threshold": confidence_threshold,
            "entropy_threshold_nats": entropy_threshold,
            "note": (
                "Predictions below the confidence threshold or above the entropy threshold "
                "are routed to human review. They are reported as uncertain, not as "
                "confirmed diagnoses, and no clinical decision rule is implied."
            ),
        },
        "reliability_diagram": str(diagram_path) if diagram_path else None,
    }

    path = out_dir / "calibration_results.json"
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    logger.info(
        "calibration: ECE %.4f over %d bins | %.1f%% of test predictions flagged for review",
        binning["ece"], bins_count, 100.0 * uncertainty["review_flags"]["fraction_flagged"],
    )
    if events is not None:
        events.write(
            "calibration_done",
            ece=binning["ece"],
            flagged_fraction=uncertainty["review_flags"]["fraction_flagged"],
        )
    return {
        "status": "completed",
        "ece": binning["ece"],
        "results_json": str(path),
        "reliability_diagram": str(diagram_path) if diagram_path else None,
        "fraction_flagged": uncertainty["review_flags"]["fraction_flagged"],
    }
