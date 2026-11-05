"""Stratified group K-fold cross-validation for the five-class classifier.

The K-fold protocol here is a *training* protocol, not a YAML flag:

* folds are assigned over the **development pool only** (the ``train`` and
  ``validation`` manifests produced by the prepare stage); the held-out ``test``
  and ``external_test`` partitions can never enter a fold, and a run refuses to
  start if a manifest row carries one of those split names;
* assignment is grouped by ``group_id`` (patient / duplicate group) and
  stratified by the five-class label, so a group is never split between a fold's
  training and validation halves;
* every fold trains a fresh model through :func:`src.experiment_engine.run_experiment`
  (same builder, same augmentation, same loss variants, same checkpoint format),
  validates on its own fold, and writes its checkpoint, history, metrics and
  predictions;
* the folds are aggregated two ways: fold-level mean ± standard deviation, and
  pooled out-of-fold predictions, which are the more reliable overall estimate
  and are saved as one file;
* the final model is retrained on the complete development pool with a fixed
  epoch budget taken from the folds (median best epoch), so no fold validation
  data is reused for early stopping during the final fit.

If the data cannot support the requested number of folds, the runner reports
``blocked`` with the measured reason instead of silently reducing the fold count.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .config import Config
from .data import ClassifierDataset, load_manifest_rows

CV_SUMMARY = "cv_summary.json"
FOLD_METRICS_CSV = "fold_metrics.csv"
OOF_PREDICTIONS = "oof_predictions.csv"

#: Metric keys aggregated across folds (all are higher-is-better except loss).
AGGREGATE_KEYS = (
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "cohen_kappa",
)

#: Splits that may appear in a fold. Anything else is rejected.
DEVELOPMENT_SPLITS = {"train", "validation", "dev", ""}


class FoldError(RuntimeError):
    """Raised when a leakage-safe K-fold assignment cannot be produced."""


@dataclass
class FoldPlan:
    """Result of assigning every development row to exactly one fold."""

    assignments: List[int]
    n_splits: int
    seed: int
    grouped: bool
    summary: Dict[str, Any]


def _label_of(row: Dict[str, Any]) -> int:
    value = row.get("mapped_label_int")
    if value is None or value == "":
        value = row.get("mapped_label")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise FoldError(
            f"row has no integer mapped label: {row.get('image_path')!r} "
            f"(mapped_label={row.get('mapped_label')!r})"
        ) from exc


def _group_of(row: Dict[str, Any]) -> Optional[str]:
    for key in ("group_id", "patient_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return None


def load_development_rows(cfg: Config) -> List[Dict[str, Any]]:
    """Load the development pool (train + validation manifests), de-duplicated."""
    manifests = cfg.outputs_dir("manifests")
    train_path = manifests / "train_manifest.csv"
    val_path = manifests / "validation_manifest.csv"
    if not train_path.is_file():
        raise FoldError(
            f"the training manifest is missing ({train_path}); run the prepare stage first"
        )
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for path in (train_path, val_path):
        if not path.is_file():
            continue
        try:
            rows_in_file = load_manifest_rows(path)
        except ValueError:
            # A header-only manifest is a legitimate state (no validation rows yet).
            continue
        for row in rows_in_file:
            key = str(row.get("image_path") or "")
            if key and key in seen:
                continue
            seen.add(key)
            rows.append(row)
    if not rows:
        raise FoldError("the development manifests contain no rows")

    forbidden = [
        row for row in rows
        if str(row.get("split") or "").strip().lower() not in DEVELOPMENT_SPLITS
    ]
    if forbidden:
        raise FoldError(
            "refusing to cross-validate on non-development rows: "
            + ", ".join(sorted({str(r.get("split")) for r in forbidden})[:5])
        )
    return rows


def assign_folds(
    rows: Sequence[Dict[str, Any]],
    *,
    n_splits: int,
    seed: int,
    allow_ungrouped: bool = False,
) -> FoldPlan:
    """Assign each row to one of ``n_splits`` grouped, stratified folds.

    ``allow_ungrouped`` defaults to False: a partial or missing group identifier
    raises instead of silently falling back to ordinary stratification, because
    that fallback can leak the same patient into training and validation.
    """
    if n_splits < 2:
        raise FoldError(f"n_splits must be at least 2, got {n_splits}")
    if len(rows) < n_splits:
        raise FoldError(
            f"only {len(rows)} development rows for {n_splits} folds; "
            "at least one row per fold is required"
        )

    labels = np.array([_label_of(row) for row in rows], dtype=int)
    groups = [_group_of(row) for row in rows]
    ungrouped = sum(1 for group in groups if group is None)
    if ungrouped and not allow_ungrouped:
        raise FoldError(
            f"{ungrouped} of {len(rows)} development rows have no group id, so a grouped "
            "K-fold split cannot be guaranteed leakage-free. Fix grouping in the prepare "
            "stage, or set run.cv.allow_ungrouped_fallback: true to accept ordinary "
            "stratified folds and record the limitation."
        )

    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    unique_groups = {group for group in groups if group is not None}
    per_class_groups: Dict[int, set] = {}
    for label, group in zip(labels, groups):
        per_class_groups.setdefault(int(label), set()).add(group or f"__ungrouped_{id(label)}")
    minimum_groups = min(len(value) for value in per_class_groups.values())

    assignments = np.full(len(rows), -1, dtype=int)
    if ungrouped:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        iterator = splitter.split(np.zeros(len(rows)), labels)
        grouped = False
    else:
        if minimum_groups < n_splits:
            raise FoldError(
                f"the rarest class has only {minimum_groups} distinct group(s), fewer than "
                f"the {n_splits} requested folds; reduce run.cv.n_splits or merge classes"
            )
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        iterator = splitter.split(np.zeros(len(rows)), labels, np.array(groups))
        grouped = True

    for fold, (_train_index, val_index) in enumerate(iterator):
        assignments[val_index] = fold
    if (assignments < 0).any():
        raise FoldError("the splitter did not assign every development row to a fold")

    # Verify the invariants the protocol promises, on the actual assignment.
    group_folds: Dict[str, set] = {}
    for group, fold in zip(groups, assignments):
        if group is not None:
            group_folds.setdefault(group, set()).add(int(fold))
    spanning = {group: sorted(folds) for group, folds in group_folds.items() if len(folds) > 1}
    if spanning:
        raise FoldError(
            f"{len(spanning)} group(s) span more than one fold, e.g. "
            f"{list(spanning.items())[:3]}"
        )

    per_fold_counts: Dict[int, int] = {}
    per_fold_classes: Dict[int, set] = {}
    for label, fold in zip(labels, assignments):
        per_fold_counts[int(fold)] = per_fold_counts.get(int(fold), 0) + 1
        per_fold_classes.setdefault(int(fold), set()).add(int(label))
    empty = [fold for fold in range(n_splits) if per_fold_counts.get(fold, 0) == 0]
    if empty:
        raise FoldError(f"fold(s) {empty} would have no validation rows")
    missing_training = {
        fold: sorted(set(int(x) for x in labels) - per_fold_classes[fold])
        for fold in range(n_splits)
    }

    summary = {
        "n_splits": n_splits,
        "seed": seed,
        "grouped": grouped,
        "rows": len(rows),
        "unique_groups": len(unique_groups),
        "ungrouped_rows": ungrouped,
        "per_fold_rows": {str(k): v for k, v in sorted(per_fold_counts.items())},
        "per_fold_class_counts": {
            str(fold): {
                str(label): int(sum(1 for l, f in zip(labels, assignments) if int(f) == fold and int(l) == label))
                for label in sorted(set(int(x) for x in labels))
            }
            for fold in range(n_splits)
        },
        "classes_absent_from_some_folds": {
            str(fold): values for fold, values in missing_training.items() if values
        },
        "assignment": "StratifiedGroupKFold" if grouped else "StratifiedKFold (ungrouped fallback)",
    }
    return FoldPlan(
        assignments=[int(x) for x in assignments],
        n_splits=n_splits,
        seed=seed,
        grouped=grouped,
        summary=summary,
    )


def write_fold_manifests(
    cfg: Config,
    rows: Sequence[Dict[str, Any]],
    plan: FoldPlan,
    *,
    experiment_id: str,
) -> List[Dict[str, Any]]:
    """Write one train/validation manifest pair per fold, plus their audit."""
    from .prepare import MANIFEST_COLUMNS

    base = cfg.outputs_dir("manifests", "folds", experiment_id)
    folds: List[Dict[str, Any]] = []
    for fold in range(plan.n_splits):
        fold_dir = base / f"fold_{fold + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_rows = [row for row, a in zip(rows, plan.assignments) if a != fold]
        val_rows = [row for row, a in zip(rows, plan.assignments) if a == fold]
        for name, subset in (("train_manifest.csv", train_rows), ("validation_manifest.csv", val_rows)):
            with (fold_dir / name).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS), extrasaction="ignore")
                writer.writeheader()
                for row in subset:
                    writer.writerow({key: row.get(key, "") for key in MANIFEST_COLUMNS})
        payload = {
            "fold": fold + 1,
            "experiment": experiment_id,
            "n_splits": plan.n_splits,
            "seed": plan.seed,
            "grouped": plan.grouped,
            "train_rows": len(train_rows),
            "validation_rows": len(val_rows),
            "validation_groups": sorted(
                {str(r.get("group_id") or r.get("patient_id") or "") for r in val_rows}
            ),
            "train_class_counts": _class_counts(train_rows),
            "validation_class_counts": _class_counts(val_rows),
            "validation_sources": _source_counts(val_rows),
        }
        (fold_dir / "fold_summary.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
        folds.append({"fold": fold + 1, "dir": str(fold_dir), "train_rows": train_rows,
                      "validation_rows": val_rows, "summary": payload})
    (base / "fold_plan.json").write_text(
        json.dumps({"experiment": experiment_id, **plan.summary}, indent=1), encoding="utf-8"
    )
    return folds


def _class_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        key = str(row.get("class_name") or _label_of(row))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _source_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        key = str(row.get("source_dataset") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def summarise_folds(fold_metrics: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean and sample standard deviation (ddof=1) of the fold-level metrics.

    Fold metrics are *not* independent observations of the same quantity, so this
    is reported as descriptive spread, alongside the pooled out-of-fold estimate.
    """
    if not fold_metrics:
        return {"folds": 0, "metrics": {}}
    metrics: Dict[str, Dict[str, Any]] = {}
    for key in AGGREGATE_KEYS:
        values = [float(m[key]) for m in fold_metrics if m.get(key) is not None]
        if not values:
            continue
        metrics[key] = {
            "mean": round(statistics.fmean(values), 6),
            "std": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
            "min": round(min(values), 6),
            "max": round(max(values), 6),
            "n": len(values),
        }
    return {
        "folds": len(fold_metrics),
        "aggregation": "fold-level mean and sample standard deviation (ddof=1)",
        "metrics": metrics,
    }


def _predictions_frame(
    rows: Sequence[Dict[str, Any]],
    labels: np.ndarray,
    logits: np.ndarray,
    *,
    fold: int,
    class_names: Sequence[str],
) -> List[Dict[str, Any]]:
    probabilities = _softmax(logits)
    frame: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        record = {
            "image_path": row.get("image_path", ""),
            "source_dataset": row.get("source_dataset", ""),
            "group_id": row.get("group_id", ""),
            "true_label": int(labels[index]),
            "predicted_label": int(probabilities[index].argmax()),
            "fold": fold,
        }
        for class_index, name in enumerate(class_names):
            record[f"p_{class_index}_{name}"] = round(float(probabilities[index][class_index]), 6)
        frame.append(record)
    return frame


def _softmax(logits: np.ndarray) -> np.ndarray:
    if logits.size == 0:
        return logits
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _write_frame(path: Path, frame: Sequence[Dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for record in frame:
        for key in record:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(frame)
    return path


def run_cross_validation(
    cfg: Config,
    logger,
    events=None,
    *,
    experiment_id: str,
    n_splits: Optional[int] = None,
    seed: Optional[int] = None,
    resume: bool = True,
    force_retrain: bool = False,
    rows: Optional[Sequence[Dict[str, Any]]] = None,
    trainer: Optional[Callable[..., Dict[str, Any]]] = None,
    predictor: Optional[Callable[..., tuple]] = None,
) -> Dict[str, Any]:
    """Run grouped stratified K-fold and return fold, aggregate and OOF results.

    ``trainer`` exists so the protocol can be tested without fitting a network; it
    must accept ``(cfg, logger, experiment_id, train_rows, validation_rows,
    fold_dir, resume, force_retrain)`` and return a summary dict with at least
    ``checkpoint`` and ``best_epoch``.
    """
    n_splits = int(n_splits or cfg.get("run.cv.n_splits", 5) or 5)
    seed = int(seed if seed is not None else cfg.get("run.cv.seed", cfg.seed) or cfg.seed)
    allow_ungrouped = bool(cfg.get("run.cv.allow_ungrouped_fallback", False))

    try:
        development_rows = list(rows) if rows is not None else load_development_rows(cfg)
        plan = assign_folds(
            development_rows, n_splits=n_splits, seed=seed, allow_ungrouped=allow_ungrouped
        )
    except FoldError as exc:
        logger.error("K-fold is blocked: %s", exc)
        if events is not None:
            events.write("cv_blocked", experiment=experiment_id, reason=str(exc))
        return {
            "status": "blocked",
            "experiment": experiment_id,
            "reason": str(exc),
            "n_splits": n_splits,
            "folds": [],
        }

    folds = write_fold_manifests(cfg, development_rows, plan, experiment_id=experiment_id)
    logger.info(
        "K-fold plan: %d folds, %d rows, %d groups (%s), seed %d",
        plan.n_splits, plan.summary["rows"], plan.summary["unique_groups"],
        plan.summary["assignment"], plan.seed,
    )

    class_names = list(cfg.classes)
    output_root = cfg.outputs_dir("experiments", experiment_id)
    fold_results: List[Dict[str, Any]] = []
    oof_frame: List[Dict[str, Any]] = []
    oof_labels: List[int] = []
    oof_predictions: List[int] = []
    oof_probabilities: List[List[float]] = []
    started = time.time()

    for entry in folds:
        fold = entry["fold"]
        fold_dir = Path(entry["dir"]).parent.parent / f"{experiment_id}_fold_{fold}"
        fold_output = output_root / f"fold_{fold}"
        fold_output.mkdir(parents=True, exist_ok=True)
        fold_started = time.time()
        fold_experiment = f"{experiment_id}_fold_{fold}"
        logger.info("--- fold %d/%d (%d train / %d validation) ---", fold, plan.n_splits,
                    len(entry["train_rows"]), len(entry["validation_rows"]))

        try:
            if trainer is not None:
                result = trainer(
                    cfg,
                    logger,
                    experiment_id=fold_experiment,
                    train_rows=entry["train_rows"],
                    validation_rows=entry["validation_rows"],
                    fold_dir=fold_dir,
                    resume=resume,
                    force_retrain=force_retrain,
                )
            else:
                result = _train_fold(
                    cfg,
                    logger,
                    experiment_id=fold_experiment,
                    train_rows=entry["train_rows"],
                    validation_rows=entry["validation_rows"],
                    resume=resume,
                    force_retrain=force_retrain,
                )
        except Exception as exc:
            logger.error("fold %d failed: %s: %s", fold, type(exc).__name__, exc)
            if events is not None:
                events.write("cv_fold_failed", experiment=experiment_id, fold=fold, error=str(exc))
            return {
                "status": "failed",
                "experiment": experiment_id,
                "reason": f"fold {fold} failed: {type(exc).__name__}: {exc}",
                "folds": fold_results,
                "n_splits": plan.n_splits,
            }

        checkpoint = result.get("checkpoint")
        if not checkpoint or not Path(str(checkpoint)).is_file():
            return {
                "status": "failed",
                "experiment": experiment_id,
                "reason": f"fold {fold} produced no checkpoint",
                "folds": fold_results,
                "n_splits": plan.n_splits,
            }

        try:
            if predictor is not None:
                logits, labels, metrics = predictor(
                    cfg, Path(str(checkpoint)), entry["validation_rows"]
                )
            else:
                logits, labels, metrics = _fold_predictions(
                    cfg, Path(str(checkpoint)), entry["validation_rows"]
                )
        except Exception as exc:
            logger.error("fold %d could not be evaluated: %s", fold, exc)
            return {
                "status": "failed",
                "experiment": experiment_id,
                "reason": f"fold {fold} evaluation failed: {type(exc).__name__}: {exc}",
                "folds": fold_results,
                "n_splits": plan.n_splits,
            }

        frame = _predictions_frame(
            entry["validation_rows"], labels, logits, fold=fold, class_names=class_names
        )
        _write_frame(fold_output / "predictions.csv", frame)
        (fold_output / "metrics.json").write_text(
            json.dumps(
                {
                    "fold": fold,
                    "experiment": fold_experiment,
                    "checkpoint": str(checkpoint),
                    "best_epoch": result.get("best_epoch"),
                    "epochs_completed": result.get("epochs_completed"),
                    "train_rows": len(entry["train_rows"]),
                    "validation_rows": len(entry["validation_rows"]),
                    "metrics": metrics,
                    "class_distribution": entry["summary"],
                    "seconds": round(time.time() - fold_started, 2),
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        (fold_output / "config.yaml").write_text(
            json.dumps(result.get("config") or {}, indent=1), encoding="utf-8"
        )
        history = result.get("history_csv")
        if history and Path(str(history)).is_file():
            import shutil

            shutil.copyfile(history, fold_output / "history.csv")
        if Path(str(checkpoint)).is_file():
            import shutil

            shutil.copyfile(checkpoint, fold_output / "best_model.pt")

        oof_frame.extend(frame)
        oof_labels.extend(labels.tolist())
        oof_predictions.extend([int(r["predicted_label"]) for r in frame])
        oof_probabilities.extend(logits.tolist())
        fold_results.append(
            {
                "fold": fold,
                "experiment": fold_experiment,
                "checkpoint": str(checkpoint),
                "best_epoch": result.get("best_epoch"),
                "metrics": metrics,
                "output_dir": str(fold_output),
                "seconds": round(time.time() - fold_started, 2),
            }
        )
        if events is not None:
            events.write("cv_fold_done", experiment=experiment_id, fold=fold,
                         macro_f1=metrics.get("macro_f1"))

    from .metrics import classification_metrics

    pooled = classification_metrics(
        np.array(oof_labels, dtype=int),
        np.array(oof_predictions, dtype=int),
        probabilities=_softmax(np.array(oof_probabilities, dtype=float)),
        class_names=class_names,
    )
    fold_metrics = [result["metrics"] for result in fold_results]
    summary = {
        "status": "completed",
        "experiment": experiment_id,
        "n_splits": plan.n_splits,
        "seed": plan.seed,
        "grouped": plan.grouped,
        "plan": plan.summary,
        "aggregation_note": (
            "Fold metrics are summarised as mean ± sample standard deviation across folds. "
            "The pooled out-of-fold metrics are computed from each development image's "
            "prediction by the one fold that held it out, so every image contributes exactly "
            "once and the estimate is not an average of averages."
        ),
        "folds": fold_results,
        "fold_summary": summarise_folds(fold_metrics),
        "pooled_out_of_fold": pooled,
        "oof_rows": len(oof_frame),
        "seconds": round(time.time() - started, 2),
        "fold_metrics_csv": str(output_root / FOLD_METRICS_CSV),
        "oof_predictions_csv": str(output_root / OOF_PREDICTIONS),
    }
    _write_frame(output_root / OOF_PREDICTIONS, oof_frame)
    with (output_root / FOLD_METRICS_CSV).open("w", newline="", encoding="utf-8") as handle:
        fields = ["fold", "experiment", "best_epoch"] + list(AGGREGATE_KEYS)
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for result in fold_results:
            writer.writerow(
                {
                    "fold": result["fold"],
                    "experiment": result["experiment"],
                    "best_epoch": result.get("best_epoch"),
                    **{key: result["metrics"].get(key) for key in AGGREGATE_KEYS},
                }
            )
    (output_root / CV_SUMMARY).write_text(json.dumps(summary, indent=1), encoding="utf-8")

    covered = {row["image_path"] for row in oof_frame}
    expected = {str(row.get("image_path")) for row in development_rows}
    summary["oof_coverage"] = {
        "development_rows": len(expected),
        "predicted_rows": len(covered),
        "complete": covered == expected,
    }
    if events is not None:
        events.write(
            "cv_done", experiment=experiment_id, folds=plan.n_splits,
            pooled_macro_f1=pooled.get("macro_f1"),
            complete=summary["oof_coverage"]["complete"],
        )
    logger.info(
        "K-fold complete: pooled macro-F1 %.4f, fold mean %.4f ± %.4f",
        float(pooled.get("macro_f1") or 0.0),
        summary["fold_summary"]["metrics"]["macro_f1"]["mean"],
        summary["fold_summary"]["metrics"]["macro_f1"]["std"],
    )
    return summary


def _train_fold(
    cfg: Config,
    logger,
    *,
    experiment_id: str,
    train_rows: Sequence[Dict[str, Any]],
    validation_rows: Sequence[Dict[str, Any]],
    resume: bool,
    force_retrain: bool,
    fixed_epochs: Optional[int] = None,
) -> Dict[str, Any]:
    """Train one fold through the shared experiment engine."""
    from .experiment_engine import run_experiment

    return run_experiment(
        cfg,
        logger,
        None,
        experiment_id=experiment_id,
        resume=resume,
        force_retrain=force_retrain,
        train_rows=train_rows,
        validation_rows=validation_rows,
        fixed_epochs=fixed_epochs,
    )


def _fold_predictions(
    cfg: Config,
    checkpoint: Path,
    validation_rows: Sequence[Dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load the fold checkpoint and predict its held-out rows, in row order."""
    from .experiment_engine import _dataloader, _validation_predictions, _softmax
    from .hardware import resolve_device
    from .metrics import classification_metrics

    image_size = int(cfg.require("preprocessing.classification.image_size"))
    raw_batch = cfg.get("classification.batch_size", 16)
    try:
        batch_size = int(raw_batch)
    except (TypeError, ValueError):
        batch_size = 16
    dataset = ClassifierDataset(list(validation_rows), cfg, train=False, image_size=image_size)
    loader = _dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, seed=cfg.seed)
    device = resolve_device("auto")
    logits, labels = _validation_predictions(
        cfg, checkpoint, loader, device, bool(cfg.get("classification.mixed_precision", False))
    )
    metrics = classification_metrics(
        labels,
        logits.argmax(axis=1),
        probabilities=_softmax(logits),
        class_names=list(cfg.classes),
    )
    return logits, labels, metrics


def retrain_full_development(
    cfg: Config,
    logger,
    events=None,
    *,
    experiment_id: str,
    epochs: int,
    resume: bool = True,
    force_retrain: bool = False,
    trainer: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Retrain the selected configuration on the whole development pool.

    The epoch budget comes from the folds, so no validation-based early stopping
    reuses fold data. The model of the final epoch is kept.
    """
    rows = load_development_rows(cfg)
    if trainer is not None:
        result = trainer(
            cfg, logger, experiment_id=experiment_id,
            train_rows=rows, validation_rows=rows, fold_dir=None,
            resume=resume, force_retrain=force_retrain, fixed_epochs=epochs,
        )
    else:
        result = _train_fold(
            cfg, logger, experiment_id=experiment_id,
            train_rows=rows, validation_rows=rows,
            resume=resume, force_retrain=force_retrain, fixed_epochs=epochs,
        )
    payload = {
        "status": result.get("status"),
        "experiment": experiment_id,
        "checkpoint": result.get("checkpoint"),
        "epochs": epochs,
        "development_rows": len(rows),
        "selection": "fixed_epoch_budget_from_cv",
        "note": (
            "Retrained on the complete development pool with no early stopping; the final "
            "epoch's weights are kept, so no fold validation data selected this checkpoint."
        ),
    }
    cfg.outputs_dir("experiments", experiment_id).joinpath("final_retrain.json").write_text(
        json.dumps(payload, indent=1), encoding="utf-8"
    )
    if events is not None:
        events.write("cv_final_retrain", experiment=experiment_id, epochs=epochs,
                     checkpoint=result.get("checkpoint"))
    return payload


def publish_cv_checkpoint(
    cfg: Config,
    checkpoint: Path,
    *,
    cv_summary: Dict[str, Any],
    retrain: Dict[str, Any],
    logger=None,
) -> Path:
    """Publish the retrained CV model for the evaluation stage, with provenance."""
    import shutil

    out_dir = cfg.outputs_dir("classification")
    published = out_dir / "best_resnet50.pt"
    shutil.copyfile(checkpoint, published)
    (out_dir / "checkpoint_provenance.json").write_text(
        json.dumps(
            {
                "published_checkpoint": str(published),
                "source_experiment": retrain.get("experiment"),
                "source_checkpoint": str(checkpoint),
                "selection_basis": (
                    "configuration selected from grouped stratified K-fold out-of-fold "
                    "results; the final model was retrained on the complete development pool "
                    "and no test or external-test sample influenced training or selection"
                ),
                "cv": {
                    "n_splits": cv_summary.get("n_splits"),
                    "seed": cv_summary.get("seed"),
                    "grouped": cv_summary.get("grouped"),
                    "fold_macro_f1_mean": (
                        cv_summary.get("fold_summary", {}).get("metrics", {})
                        .get("macro_f1", {}).get("mean")
                    ),
                    "fold_macro_f1_std": (
                        cv_summary.get("fold_summary", {}).get("metrics", {})
                        .get("macro_f1", {}).get("std")
                    ),
                    "pooled_oof_macro_f1": (
                        cv_summary.get("pooled_out_of_fold", {}).get("macro_f1")
                    ),
                },
                "retrain": retrain,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    if logger:
        logger.info("published the cross-validated model to %s", published)
    return published


def run_cv_protocol(
    cfg: Config,
    logger,
    events=None,
    *,
    n_splits: Optional[int] = None,
    resume: bool = True,
    force_retrain: bool = False,
    trainer: Optional[Callable[..., Dict[str, Any]]] = None,
    predictor: Optional[Callable[..., tuple]] = None,
) -> Dict[str, Any]:
    """Run K-fold, retrain on the full development pool, and publish the model."""
    summary = run_cross_validation(
        cfg, logger, events,
        experiment_id="cv_baseline",
        n_splits=n_splits,
        resume=resume,
        force_retrain=force_retrain,
        trainer=trainer,
        predictor=predictor,
    )
    if summary.get("status") != "completed":
        return summary

    try:
        epochs = median_best_epoch(summary["folds"])
        retrain = retrain_full_development(
            cfg, logger, events,
            experiment_id="cv_final",
            epochs=epochs,
            resume=resume,
            force_retrain=force_retrain,
            trainer=trainer,
        )
    except Exception as exc:
        logger.error("final retrain failed: %s: %s", type(exc).__name__, exc)
        return {**summary, "status": "failed", "reason": f"final retrain failed: {exc}"}

    checkpoint = retrain.get("checkpoint")
    if checkpoint and Path(str(checkpoint)).is_file():
        published = publish_cv_checkpoint(
            cfg, Path(str(checkpoint)), cv_summary=summary, retrain=retrain, logger=logger
        )
        summary["published_checkpoint"] = str(published)
    else:
        summary["published_checkpoint"] = None
        summary["status"] = "failed"
        summary["reason"] = "the final retrain produced no checkpoint"
    summary["retrain"] = retrain
    cfg.outputs_dir("experiments", "cv_baseline").joinpath(CV_SUMMARY).write_text(
        json.dumps(summary, indent=1), encoding="utf-8"
    )
    return summary


def median_best_epoch(folds: Sequence[Dict[str, Any]]) -> int:
    """Median (best_epoch + 1) across folds, rounded up, minimum 1."""
    values = [
        int(fold["best_epoch"]) + 1
        for fold in folds
        if fold.get("best_epoch") is not None and int(fold["best_epoch"]) >= 0
    ]
    if not values:
        raise FoldError("no fold reported a best epoch; cannot derive a retraining budget")
    return max(1, int(math.ceil(statistics.median(values))))
