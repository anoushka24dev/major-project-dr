"""K-fold protocol invariants: grouping, leakage, reproducibility and aggregation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src import kfold


def make_rows(classes: int = 5, groups_per_class: int = 12, images_per_group: int = 2):
    rows = []
    for label in range(classes):
        for group in range(groups_per_class):
            for image in range(images_per_group):
                rows.append(
                    {
                        "image_path": f"/data/c{label}/g{group}_{image}.png",
                        "rel_path": f"c{label}/g{group}_{image}.png",
                        "source_dataset": "mendeley" if label % 2 else "mmrdr",
                        "original_label": f"class-{label}",
                        "mapped_label": str(label),
                        "mapped_label_int": label,
                        "class_name": f"class-{label}",
                        "patient_id": f"{label}-{group}",
                        "group_id": f"src:{label}-{group}",
                        "split": "train",
                        "included": "True",
                        "manifest_version": "test",
                        "group_key": f"g{label}-{group}",
                    }
                )
    return rows


def test_every_row_is_validated_exactly_once():
    rows = make_rows()
    plan = kfold.assign_folds(rows, n_splits=5, seed=12345)
    assert len(plan.assignments) == len(rows)
    assert set(plan.assignments) == set(range(5))
    counts = {fold: plan.assignments.count(fold) for fold in range(5)}
    assert sum(counts.values()) == len(rows)
    assert max(counts.values()) - min(counts.values()) <= 4


def test_groups_never_span_folds():
    rows = make_rows()
    plan = kfold.assign_folds(rows, n_splits=5, seed=12345)
    seen = {}
    for row, fold in zip(rows, plan.assignments):
        seen.setdefault(row["group_id"], set()).add(fold)
    assert all(len(folds) == 1 for folds in seen.values())


def test_folds_are_reproducible_for_the_same_seed():
    rows = make_rows()
    a = kfold.assign_folds(rows, n_splits=5, seed=12345)
    b = kfold.assign_folds(rows, n_splits=5, seed=12345)
    assert a.assignments == b.assignments


def test_fold_assignment_is_affected_by_the_seed():
    rows = make_rows()
    a = kfold.assign_folds(rows, n_splits=5, seed=12345)
    b = kfold.assign_folds(rows, n_splits=5, seed=999)
    assert a.assignments != b.assignments


def test_ungrouped_rows_are_rejected_without_an_explicit_opt_in():
    rows = make_rows()
    for row in rows:
        row["group_id"] = ""
        row["patient_id"] = ""
    with pytest.raises(kfold.FoldError, match="group"):
        kfold.assign_folds(rows, n_splits=5, seed=12345)


def test_too_few_groups_per_class_is_reported_not_silently_reduced():
    rows = make_rows(groups_per_class=3)
    with pytest.raises(kfold.FoldError, match="distinct group"):
        kfold.assign_folds(rows, n_splits=5, seed=12345)


def test_fold_assignment_records_class_distributions():
    rows = make_rows()
    plan = kfold.assign_folds(rows, n_splits=5, seed=12345)
    for fold in range(5):
        counts = plan.summary["per_fold_class_counts"][str(fold)]
        assert sum(counts.values()) == plan.summary["per_fold_rows"][str(fold)]
        assert len(counts) == 5


def test_external_rows_are_refused(tmp_path, monkeypatch):
    from src.config import load_config

    cfg = load_config(None, root=Path.cwd())
    manifests = cfg.outputs_dir("manifests")
    created = []
    try:
        train = manifests / "train_manifest.csv"
        val = manifests / "validation_manifest.csv"
        for path in (train, val):
            created.append(path.exists())
        external = dict(make_rows(classes=1, groups_per_class=1)[0], split="external_test")
        fields = sorted({key for row in [external] for key in row})
        train.write_text("", encoding="utf-8")
        with train.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(external)
        if val.exists():
            val.unlink()
        with pytest.raises(kfold.FoldError, match="non-development"):
            kfold.load_development_rows(cfg)
    finally:
        # Restore: remove the manifest this test wrote so later tests are unaffected.
        (manifests / "train_manifest.csv").unlink(missing_ok=True)


def test_write_fold_manifests_partitions_the_development_pool(tmp_path):
    from src.config import load_config

    cfg = load_config(None, root=Path.cwd())
    cfg._data.setdefault("paths", {})["outputs_dir"] = str(tmp_path / "outputs")
    rows = make_rows(groups_per_class=6, images_per_group=1)
    plan = kfold.assign_folds(rows, n_splits=3, seed=7)
    folds = kfold.write_fold_manifests(cfg, rows, plan, experiment_id="E1")
    assert len(folds) == 3
    seen_train, seen_val = set(), set()
    for entry in folds:
        train_paths = {row["image_path"] for row in entry["train_rows"]}
        val_paths = {row["image_path"] for row in entry["validation_rows"]}
        assert not (train_paths & val_paths)
        assert val_paths
        seen_val |= val_paths
        seen_train |= train_paths
        stored = Path(entry["dir"]) / "fold_summary.json"
        payload = json.loads(stored.read_text(encoding="utf-8"))
        assert payload["train_rows"] == len(train_paths)
        assert payload["validation_rows"] == len(val_paths)
    assert seen_val == {row["image_path"] for row in rows}
    assert len(seen_train) == len(rows)


def test_summarise_folds_uses_mean_and_sample_std():
    metrics = [
        {"accuracy": 0.5, "balanced_accuracy": 0.4, "macro_precision": 0.3,
         "macro_recall": 0.2, "macro_f1": 0.25, "cohen_kappa": 0.1},
        {"accuracy": 0.7, "balanced_accuracy": 0.6, "macro_precision": 0.5,
         "macro_recall": 0.4, "macro_f1": 0.45, "cohen_kappa": 0.3},
    ]
    summary = kfold.summarise_folds(metrics)
    assert summary["metrics"]["macro_f1"]["mean"] == pytest.approx(0.35)
    assert summary["metrics"]["macro_f1"]["std"] == pytest.approx(np.std([0.25, 0.45], ddof=1), abs=1e-6)
    assert summary["metrics"]["accuracy"]["min"] == 0.5
    assert summary["metrics"]["accuracy"]["max"] == 0.7


def fake_trainer(cfg, logger, *, experiment_id, train_rows, validation_rows, fold_dir,
                 resume, force_retrain, fixed_epochs=None):
    """Stand-in for the GPU trainer: writes a checkpoint file and a summary."""
    root = cfg.path("paths.checkpoints_dir", create=True) / experiment_id / "best"
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "best_model.pt"
    checkpoint.write_bytes(b"fake-weights")
    published = root.parent / "best_resnet50.pt"
    published.write_bytes(b"fake-weights")
    return {
        "status": "completed",
        "checkpoint": str(published),
        "best_epoch": 1 if fixed_epochs is None else fixed_epochs - 1,
        "epochs_completed": 2 if fixed_epochs is None else fixed_epochs,
        "config": {"experiment": experiment_id, "train_rows": len(train_rows)},
        "history_csv": None,
    }


def fake_predictor(cfg, checkpoint, validation_rows):
    """Perfect predictions derived from the row labels (no images needed)."""
    from src.metrics import classification_metrics

    labels = np.array([int(row["mapped_label_int"]) for row in validation_rows])
    logits = np.zeros((len(labels), 5))
    logits[np.arange(len(labels)), labels] = 10.0
    metrics = classification_metrics(labels, labels, class_names=list(cfg.classes))
    return logits, labels, metrics


def test_run_cross_validation_produces_complete_oof_predictions(tmp_path):
    from src.config import load_config

    cfg = load_config(None, root=Path.cwd())
    cfg._data.setdefault("paths", {})["outputs_dir"] = str(tmp_path / "outputs")
    cfg._data.setdefault("paths", {})["checkpoints_dir"] = str(tmp_path / "checkpoints")
    rows = make_rows(groups_per_class=8, images_per_group=1)

    summary = kfold.run_cross_validation(
        cfg, _NullLogger(), None,
        experiment_id="cv_test", n_splits=4, seed=12345,
        rows=rows, trainer=fake_trainer, predictor=fake_predictor,
    )
    assert summary["status"] == "completed"
    assert summary["n_splits"] == 4
    assert len(summary["folds"]) == 4
    assert summary["oof_coverage"]["complete"] is True
    assert summary["oof_rows"] == len(rows)
    assert summary["pooled_out_of_fold"]["macro_f1"] == pytest.approx(1.0)

    oof = Path(summary["oof_predictions_csv"])
    assert oof.is_file()
    with oof.open(encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    assert len(records) == len(rows)
    assert {int(r["true_label"]) for r in records} == set(range(5))
    for fold in range(1, 5):
        fold_dir = Path(summary["folds"][fold - 1]["output_dir"])
        assert (fold_dir / "metrics.json").is_file()
        assert (fold_dir / "predictions.csv").is_file()
        assert (fold_dir / "best_model.pt").is_file()
        payload = json.loads((fold_dir / "metrics.json").read_text(encoding="utf-8"))
        assert payload["fold"] == fold
        assert payload["validation_rows"] > 0

    # Fold metrics CSV has one row per fold plus a summary of mean ± std.
    with (Path(summary["fold_metrics_csv"])).open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 4
    saved = json.loads((Path(summary["fold_metrics_csv"]).parent / "cv_summary.json").read_text(encoding="utf-8"))
    assert saved["fold_summary"]["folds"] == 4


def test_run_cross_validation_reports_blocked_when_data_cannot_support_folds(tmp_path):
    from src.config import load_config

    cfg = load_config(None, root=Path.cwd())
    cfg._data.setdefault("paths", {})["outputs_dir"] = str(tmp_path / "outputs")
    rows = make_rows(groups_per_class=1, images_per_group=1)  # 5 rows, one group each
    summary = kfold.run_cross_validation(
        cfg, _NullLogger(), None, experiment_id="cv_blocked", n_splits=5, rows=rows,
        trainer=fake_trainer, predictor=fake_predictor,
    )
    assert summary["status"] == "blocked"
    assert "group" in summary["reason"]
    assert summary["folds"] == []


def test_median_best_epoch_is_robust_to_one_outlier():
    folds = [{"best_epoch": 2}, {"best_epoch": 3}, {"best_epoch": 40}]
    assert kfold.median_best_epoch(folds) == 4  # median of (best_epoch + 1)
    with pytest.raises(kfold.FoldError):
        kfold.median_best_epoch([{"best_epoch": None}])


def test_run_cv_protocol_retrains_and_publishes(tmp_path):
    from src.config import load_config

    cfg = load_config(None, root=Path.cwd())
    cfg._data.setdefault("paths", {})["outputs_dir"] = str(tmp_path / "outputs")
    cfg._data.setdefault("paths", {})["checkpoints_dir"] = str(tmp_path / "checkpoints")
    rows = make_rows(groups_per_class=6, images_per_group=1)
    # The retrain path loads the development manifests, so point them at this pool.
    manifests = cfg.outputs_dir("manifests")
    fields = list(rows[0].keys())
    for name, subset in (("train_manifest.csv", rows), ("validation_manifest.csv", [])):
        with (manifests / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(subset)

    summary = kfold.run_cv_protocol(
        cfg, _NullLogger(), None, n_splits=3,
        trainer=fake_trainer, predictor=fake_predictor, force_retrain=True,
    )
    assert summary["status"] == "completed"
    published = Path(summary["published_checkpoint"])
    assert published.is_file()
    provenance = json.loads(
        (published.parent / "checkpoint_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["source_experiment"] == "cv_final"
    assert provenance["cv"]["n_splits"] == 3
    assert summary["retrain"]["selection"] == "fixed_epoch_budget_from_cv"


class _NullLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass
