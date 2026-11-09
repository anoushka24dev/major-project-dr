"""Integration checks for the E0..E6 experiment series and final_candidate.

No training runs here. These tests assert what the suite is supposed to guarantee:
every experiment resolves to a complete, distinct recipe through the same engine,
and the comparison artefact records each experiment's loss, learning rates,
augmentation, scheduler, fine-tuning strategy, checkpoint and selection metric.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config  # noqa: E402
from src.experiment_engine import FocalLoss, build_loss_fn  # noqa: E402
from src.experiments import (  # noqa: E402
    AUGMENTATION_VARIANTS,
    EXPERIMENT_CONFIG_OVERRIDES,
    EXPERIMENT_IDS,
    _build_comparison,
    experiment_ids,
    final_candidate_config,
    select_best_by_validation,
)

KNOWN_LOSSES = {"cross_entropy", "weighted_cross_entropy", "focal"}


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_the_suite_covers_e0_to_e6_and_the_final_candidate():
    assert EXPERIMENT_IDS == ("E0", "E1", "E2", "E3", "E4", "E5", "E6", "final_candidate")
    assert set(EXPERIMENT_CONFIG_OVERRIDES) == set(EXPERIMENT_IDS)


def test_every_experiment_declares_a_complete_recipe(cfg):
    """Each experiment resolves through the config to all seven required facts."""
    scheduler = cfg.get("classification.scheduler")
    base_lr = float(cfg.get("classification.learning_rate"))
    head_lr = float(cfg.get("experiments.head_learning_rate"))
    backbone_multiplier = float(cfg.get("experiments.backbone_lr_multiplier"))
    monitor = cfg.get("classification.monitor")
    runs_root = cfg.path("paths.runs_dir", create=False) / "experiments"

    records = {}
    for eid in EXPERIMENT_IDS:
        overrides = EXPERIMENT_CONFIG_OVERRIDES[eid]
        mode = overrides["experiments.augmentation_mode"]
        freeze_epochs = int(overrides["experiments.two_stage_freeze_epochs"])

        record = {
            "loss": overrides["experiments.loss"],
            "learning_rates": (
                {"head": head_lr, "backbone": head_lr * backbone_multiplier}
                if freeze_epochs > 0
                else {"all_parameters": base_lr}
            ),
            "augmentation": AUGMENTATION_VARIANTS[mode],
            "scheduler": scheduler,
            "fine_tuning": "two-stage (head warm-up, then unfreeze)" if freeze_epochs else "single-stage",
            "checkpoint": str(runs_root / eid),
            "selection_metric": monitor,
        }
        records[eid] = record

        assert record["loss"] in KNOWN_LOSSES
        assert mode in AUGMENTATION_VARIANTS
        assert freeze_epochs in {0, 1, 2, 3, 4, 5}
        for field in ("loss", "learning_rates", "augmentation", "scheduler",
                      "fine_tuning", "checkpoint", "selection_metric"):
            assert record[field] not in (None, "", {}), f"{eid}.{field} is unset"
        assert record["scheduler"] == "cosine"
        assert record["selection_metric"] == "val_macro_f1"

    # Every experiment owns a distinct checkpoint directory.
    checkpoints = [record["checkpoint"] for record in records.values()]
    assert len(set(checkpoints)) == len(checkpoints)


def test_loss_variants_build_the_expected_loss_objects(cfg):
    plain, plain_meta = build_loss_fn(cfg, loss_variant="cross_entropy")
    assert type(plain).__name__ == "CrossEntropyLoss"
    assert plain_meta["variant"] == "cross_entropy"

    weighted, weighted_meta = build_loss_fn(cfg, loss_variant="weighted_cross_entropy")
    assert type(weighted).__name__ == "CrossEntropyLoss"
    assert weighted_meta["variant"] == "weighted_cross_entropy"

    focal, focal_meta = build_loss_fn(cfg, loss_variant="focal")
    assert isinstance(focal, FocalLoss)
    assert focal_meta["variant"] == "focal"
    assert focal.gamma == pytest.approx(float(cfg.get("experiments.focal_gamma")))
    alpha = float(cfg.get("experiments.focal_alpha"))
    assert focal.alpha is not None and float(focal.alpha) == pytest.approx(alpha)


def test_final_candidate_inherits_the_best_validation_experiment(cfg):
    assert final_candidate_config(cfg, "E3") == EXPERIMENT_CONFIG_OVERRIDES["E3"]
    # With no completed training experiment it falls back to the E0 recipe.
    assert final_candidate_config(cfg, None) == EXPERIMENT_CONFIG_OVERRIDES["E0"]


def test_selection_uses_validation_macro_f1_only():
    individual = {
        "E0": {"status": "completed", "best_metric_value": 0.61},
        "E1": {"status": "failed", "best_metric_value": 0.99},
        "E2": {"status": "completed", "best_metric_value": 0.64},
        "E3": {"status": "completed"},
        "final_candidate": {"status": "completed", "best_metric_value": 0.50},
    }
    assert select_best_by_validation(individual) == "E2"
    assert select_best_by_validation({}) is None
    assert select_best_by_validation({"E0": {"status": "reused", "best_metric_value": 0.3}}) == "E0"


def test_comparison_records_every_field_for_every_experiment(cfg):
    individual = {
        eid: {
            "status": "reused",
            "best_metric_value": 0.5,
            "best_epoch": 2,
            "epochs_completed": 3,
            "checkpoint": f"checkpoints/{eid}/best/model.pt",
            "config": {
                "loss_variant": EXPERIMENT_CONFIG_OVERRIDES[eid]["experiments.loss"],
                "augmentation_mode": EXPERIMENT_CONFIG_OVERRIDES[eid]["experiments.augmentation_mode"],
                "two_stage_freeze_epochs": EXPERIMENT_CONFIG_OVERRIDES[eid][
                    "experiments.two_stage_freeze_epochs"
                ],
            },
        }
        for eid in EXPERIMENT_IDS
    }

    comparison = _build_comparison(cfg, individual, EXPERIMENT_IDS)

    rows = {row["experiment"]: row for row in comparison["experiments"]}
    assert set(rows) == set(EXPERIMENT_IDS)
    for eid, row in rows.items():
        assert row["loss_variant"] == EXPERIMENT_CONFIG_OVERRIDES[eid]["experiments.loss"]
        assert row["augmentation_mode"] == EXPERIMENT_CONFIG_OVERRIDES[eid]["experiments.augmentation_mode"]
        assert row["checkpoint"]
        assert row["best_val_macro_f1"] == pytest.approx(0.5)
    assert "validation" in comparison["note"].lower()
    assert "test" in comparison["note"].lower()


def test_experiment_selection_contract():
    assert experiment_ids(["all"]) == list(EXPERIMENT_IDS)
    assert experiment_ids(None) == list(EXPERIMENT_IDS)
    assert experiment_ids(["E0", "E4"]) == ["E0", "E4"]
    with pytest.raises(ValueError):
        experiment_ids(["E9"])
    with pytest.raises(ValueError):
        experiment_ids(["final_candidate"])


def test_augmentation_variants_are_retinal_specific():
    for mode in ("default", "retinal"):
        variant = AUGMENTATION_VARIANTS[mode]
        assert variant["horizontal_flip"] is True
        assert variant["vertical_flip"] is False
        assert 0 < variant["rotation_degrees"] <= 15
    assert AUGMENTATION_VARIANTS["retinal"]["rotation_degrees"] >= (
        AUGMENTATION_VARIANTS["default"]["rotation_degrees"]
    )


def test_training_augmentation_is_never_applied_to_evaluation_data(cfg):
    from src.data import ClassifierDataset

    rows = [
        {"image_path": f"image_{index}.png", "mapped_label_int": str(index % 5), "source_dataset": "mendeley"}
        for index in range(5)
    ]
    training = ClassifierDataset(rows, cfg, train=True)
    evaluation = ClassifierDataset(rows, cfg, train=False)

    assert training.hflip is True
    assert training.rotation > 0
    assert evaluation.hflip is False
    assert evaluation.rotation == 0.0
    assert evaluation.brightness == 0.0
    assert evaluation.contrast == 0.0
