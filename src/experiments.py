"""Experiment suite: definitions, configuration and orchestration.

One ResNet-50, one data pipeline, one evaluation path, many experiments. Each
experiment is a named configuration that selects which loss, training strategy,
augmentation, threshold and calibration variant is used; the actual training is
delegated to :func:`src.experiment_engine.run_experiment`, which reuses the same
builder, dataset class, augmentation policy, checkpoint machinery and metrics.

The comparison tree described in the task is implemented here as a set of named
experiments plus a single "final_candidate" that is selected from the best of
E0..E6 by validation macro-F1. The final candidate is evaluated once, on the
untouched test set, with its own checkpoint that is never reused by the other
experiments.

What is NOT done here
---------------------
* Threshold tuning is validation-only and is described/selected here but delegated
  to the evaluation stage when it depends on the held-out predictions.
* Temperature scaling is a calibration-stage concern; the experiment only records
  the chosen variant and the calibration stage applies it.
* No experiment is claimed better than another unless the comparison report shows
  it; the README and the final report state the comparison explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .config import Config
from .logutil import get_logger

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

#: Public experiment identifiers. Order is the natural reading order of the
#: comparison (E0 baseline, then the incremental variants, then the final candidate).
EXPERIMENT_IDS = (
    "E0",
    "E1",
    "E2",
    "E3",
    "E4",
    "E5",
    "E6",
    "final_candidate",
)

EXPERIMENT_NAMES: Dict[str, str] = {
    "E0": "Existing baseline (ResNet-50, class-weighted cross-entropy, mild augmentation)",
    "E1": "Class-weighted cross-entropy (weights computed from the training fold only)",
    "E2": "E1 + two-stage ResNet-50 fine-tuning (frozen backbone head warm-up, then unfreeze)",
    "E3": "E2 + retinal-specific training augmentation (small rotation, hflip, mild brightness/contrast)",
    "E4": "E2 + multiclass focal loss (vs the best cross-entropy configuration)",
    "E5": "Best training configuration + validation-selected class-specific threshold optimisation",
    "E6": "Best configuration + ECE / reliability diagram / predictive entropy / temperature scaling",
    "final_candidate": "Selected best of E0..E6 by validation macro-F1, evaluated once on the held-out test set",
}

#: Which config keys change between experiments. Used by the orchestrator and by
#: tests to assert that the suite is driven by configuration, not by copy-pasted code.
EXPERIMENT_CONFIG_OVERRIDES: Dict[str, Dict[str, Any]] = {
    # E0 — baseline. Uses the configured classification.loss-style behaviour but we
    # express it as an explicit experiment for consistency. The existing pipeline's
    # single run already behaves like E0; the experiment engine makes the comparison
    # explicit and reproducible.
    "E0": {
        "experiments.loss": "cross_entropy",
        "experiments.two_stage_freeze_epochs": 0,
        "experiments.augmentation_mode": "default",  # the default mild policy
    },
    # E1 — explicit class-weighted cross-entropy (same weights source as E0, explicit).
    "E1": {
        "experiments.loss": "weighted_cross_entropy",
        "experiments.two_stage_freeze_epochs": 0,
        "experiments.augmentation_mode": "default",
    },
    # E2 — two-stage fine-tuning on top of E1.
    "E2": {
        "experiments.loss": "weighted_cross_entropy",
        "experiments.two_stage_freeze_epochs": 3,
        "experiments.augmentation_mode": "default",
    },
    # E3 — E2 + stronger (but still justified) retinal augmentation.
    "E3": {
        "experiments.loss": "weighted_cross_entropy",
        "experiments.two_stage_freeze_epochs": 3,
        "experiments.augmentation_mode": "retinal",
    },
    # E4 — focal loss as a controlled alternative to the best cross-entropy config (E2/E3).
    "E4": {
        "experiments.loss": "focal",
        "experiments.two_stage_freeze_epochs": 3,
        "experiments.augmentation_mode": "retinal",
    },
    # E5 / E6 select from E2/E3 best; the exact selection rule is described in the
    # orchestrator below. Their training config mirrors the selected best training config.
    "E5": {
        "experiments.loss": "weighted_cross_entropy",
        "experiments.two_stage_freeze_epochs": 3,
        "experiments.augmentation_mode": "retinal",
        "experiments.threshold_optimisation": True,
    },
    "E6": {
        "experiments.loss": "weighted_cross_entropy",
        "experiments.two_stage_freeze_epochs": 3,
        "experiments.augmentation_mode": "retinal",
        "experiments.calibration_evaluation": True,
    },
    # final_candidate reuses the best of E0..E6 by validation macro-F1.
    "final_candidate": {
        "experiments.loss": "weighted_cross_entropy",
        "experiments.two_stage_freeze_epochs": 3,
        "experiments.augmentation_mode": "retinal",
    },
}

#: Augmentation policy variants. "default" is the mild policy already in config.yaml.
#: "retinal" adds small rotation + mild colour variation but remains training-only and
#: never distorts lesions. Both are configured under preprocessing.classification.augmentation.
AUGMENTATION_VARIANTS: Dict[str, Dict[str, Any]] = {
    "default": {
        "horizontal_flip": True,
        "rotation_degrees": 10,
        "brightness": 0.08,
        "contrast": 0.08,
        "random_crop": False,
        "vertical_flip": False,
    },
    "retinal": {
        "horizontal_flip": True,
        "rotation_degrees": 15,
        "brightness": 0.12,
        "contrast": 0.12,
        "colour_jitter": 0.06,
        "random_crop": False,
        "vertical_flip": False,
    },
}


def describe_experiment(experiment_id: str) -> Dict[str, Any]:
    """Return a human-readable description + the config overrides for one experiment."""
    return {
        "id": experiment_id,
        "name": EXPERIMENT_NAMES.get(experiment_id, experiment_id),
        "overrides": dict(EXPERIMENT_CONFIG_OVERRIDES.get(experiment_id, {})),
        "is_final_candidate": experiment_id == "final_candidate",
    }


def experiment_ids(selected: Optional[Sequence[str]] = None) -> List[str]:
    """Return the experiment ids to run: explicit list, "all", or a sensible default."""
    if selected is None:
        return list(EXPERIMENT_IDS)
    if len(selected) == 1 and str(selected[0]).strip().lower() == "all":
        return list(EXPERIMENT_IDS)
    out: List[str] = []
    for sid in selected:
        if str(sid) not in EXPERIMENT_IDS:
            raise ValueError(
                f"Unknown experiment {sid!r}. Known: {sorted(EXPERIMENT_IDS)}"
            )
        if str(sid) not in out:
            out.append(str(sid))
    # final_candidate is meaningless without its dependencies; require at least one
    # of E0..E6 if the final candidate is requested.
    if "final_candidate" in out and not any(e in out for e in EXPERIMENT_IDS[:-1]):
        raise ValueError(
            "final_candidate requires at least one of E0..E6 to be selected as well."
        )
    return out


# ---------------------------------------------------------------------------
# Best-experiment selection (validation only, never test)
# ---------------------------------------------------------------------------

def select_best_by_validation(results: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """Return the experiment id with the best validation macro-F1.

    Only experiments that completed and reported a validation macro-F1 are eligible.
    This is a *descriptive* selection for the final candidate; it is recorded in the
    output so the choice can be reproduced and critiqued.
    """
    best: Optional[str] = None
    best_value: float = -1.0
    for eid, summary in results.items():
        if summary.get("status") not in {"completed", "reused"}:
            continue
        val = summary.get("best_metric_value")
        if val is None:
            continue
        if val > best_value:
            best_value = float(val)
            best = eid
    return best


def final_candidate_config(
    cfg: Config,
    best_training_experiment: Optional[str],
) -> Dict[str, Any]:
    """Configuration for the final candidate, based on the best training experiment.

    The final candidate reuses the best training configuration (loss, two-stage
    schedule, augmentation) and is evaluated once on the held-out test set. If no
    training experiment completed, it falls back to E0.
    """
    base = dict(EXPERIMENT_CONFIG_OVERRIDES.get("E0", {}))
    if best_training_experiment and best_training_experiment in EXPERIMENT_CONFIG_OVERRIDES:
        base.update(EXPERIMENT_CONFIG_OVERRIDES[best_training_experiment])
    return base


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_selected_experiments(
    cfg: Config,
    logger,
    events=None,
    *,
    experiments: Sequence[str],
    resume: bool = True,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """Run the requested experiments and return a comparison summary.

    Each experiment is run by the same engine, so fold assignments, augmentation
    policy, checkpoint/fingerprint machinery and metrics are identical across
    experiments. The only differences are those declared in
    :data:`EXPERIMENT_CONFIG_OVERRIDES`.

    ``final_candidate`` is run **last** and inherits the training configuration of
    the best training experiment by validation macro-F1. That selection uses the
    validation fold only; the test partition is never consulted here.
    """
    logger = logger or get_logger()
    selected = experiment_ids(experiments)
    individual: Dict[str, Dict[str, Any]] = {}

    for eid in [name for name in selected if name != "final_candidate"]:
        logger.info("=" * 70)
        logger.info("EXPERIMENT %s — %s", eid, EXPERIMENT_NAMES.get(eid, eid))
        logger.info("=" * 70)
        try:
            individual[eid] = _run_one(cfg, logger, events, eid, resume, force_retrain)
        except Exception as exc:  # noqa: BLE001 - one failure must not stop the rest
            logger.error("Experiment %s failed: %s: %s", eid, type(exc).__name__, exc)
            individual[eid] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "experiment": eid,
            }
            if events is not None:
                events.write("experiment_failed", experiment=eid, error=str(exc))

    if "final_candidate" in selected:
        best = select_best_by_validation(individual)
        logger.info("=" * 70)
        logger.info(
            "EXPERIMENT final_candidate — derived from %s (validation macro-F1)",
            best or "E0 (no training experiment completed)",
        )
        logger.info("=" * 70)
        try:
            summary = _run_one(
                cfg, logger, events, "final_candidate", resume, force_retrain,
                overrides=final_candidate_config(cfg, best),
            )
            summary["derived_from_experiment"] = best or "E0"
            individual["final_candidate"] = summary
        except Exception as exc:  # noqa: BLE001
            logger.error("final_candidate failed: %s: %s", type(exc).__name__, exc)
            individual["final_candidate"] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "experiment": "final_candidate",
            }
            if events is not None:
                events.write("experiment_failed", experiment="final_candidate", error=str(exc))

    comparison = _build_comparison(cfg, individual, selected)
    return {
        "selected": selected,
        "individual": individual,
        "comparison": comparison,
        "best_by_validation_macro_f1": select_best_by_validation(
            {k: v for k, v in individual.items() if k != "final_candidate"}
        ),
    }


def _run_one(
    cfg: Config,
    logger,
    events,
    eid: str,
    resume: bool,
    force_retrain: bool,
    *,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from .experiment_engine import run_experiment

    # Apply the experiment's config overrides on top of the base config. The overrides
    # only touch experiments.* keys, so they cannot change dataset, taxonomy, split or
    # preprocessing geometry in ways that would break reproducibility.
    applied = dict(overrides if overrides is not None else EXPERIMENT_CONFIG_OVERRIDES.get(eid, {}))
    for key, value in applied.items():
        if key.startswith("experiments."):
            cfg._data.setdefault("experiments", {})[key.split(".", 1)[1]] = value

    # The augmentation variant is materialised by the engine into the training-only
    # augmentation policy, so E3 really trains differently from E2.
    cfg._data.setdefault("experiments", {})["augmentation_mode"] = applied.get(
        "experiments.augmentation_mode", "default"
    )

    return run_experiment(
        cfg,
        logger,
        events,
        experiment_id=eid,
        resume=resume,
        force_retrain=force_retrain,
    )


def promote_final_candidate(
    cfg: Config,
    individual: Dict[str, Dict[str, Any]],
    selected: Sequence[str],
    *,
    logger=None,
) -> Optional[Path]:
    """Publish one experiment checkpoint as the classifier the evaluation stage loads.

    Preference order: the explicit ``final_candidate`` checkpoint, then the best
    training experiment by validation macro-F1. The published file is a byte copy;
    the source experiment directory is left untouched, so every experiment keeps its
    own checkpoint.
    """
    import shutil

    logger = logger or get_logger()
    candidates: List[str] = []
    if "final_candidate" in individual:
        candidates.append("final_candidate")
    best = select_best_by_validation(
        {k: v for k, v in individual.items() if k != "final_candidate"}
    )
    if best:
        candidates.append(best)
    candidates.extend(eid for eid in selected if eid not in candidates)

    out_dir = cfg.outputs_dir("classification")
    published = out_dir / "best_resnet50.pt"
    for eid in candidates:
        summary = individual.get(eid) or {}
        if summary.get("status") not in {"completed", "reused"}:
            continue
        source = summary.get("checkpoint")
        if not source or not Path(source).is_file():
            continue
        shutil.copyfile(source, published)
        note = out_dir / "checkpoint_provenance.json"
        note.write_text(
            json.dumps(
                {
                    "published_checkpoint": str(published),
                    "source_experiment": eid,
                    "source_checkpoint": str(source),
                    "selection_basis": (
                        "final_candidate inherits the best validation macro-F1 training "
                        "configuration; the test partition was never used for this choice"
                        if eid == "final_candidate"
                        else "highest validation macro-F1 among the training experiments"
                    ),
                    "all_experiments": {
                        name: {
                            "status": summary.get("status"),
                            "checkpoint": summary.get("checkpoint"),
                            "best_val_macro_f1": summary.get("best_metric_value"),
                        }
                        for name, summary in individual.items()
                    },
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        logger.info("Published experiment %s checkpoint as %s", eid, published)
        return published

    logger.warning(
        "No experiment produced a usable checkpoint; the evaluation stage will not run "
        "against an unvalidated model."
    )
    return None


def _build_comparison(
    cfg: Config,
    individual: Dict[str, Dict[str, Any]],
    selected: Sequence[str],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for eid in selected:
        summary = individual.get(eid, {})
        status = summary.get("status", "not_run")
        rows.append(
            {
                "experiment": eid,
                "name": EXPERIMENT_NAMES.get(eid, eid),
                "status": status,
                "best_epoch": summary.get("best_epoch"),
                "best_val_macro_f1": summary.get("best_metric_value"),
                "epochs_completed": summary.get("epochs_completed"),
                "checkpoint": summary.get("checkpoint"),
                "loss_variant": (summary.get("config") or {}).get("loss_variant"),
                "augmentation_mode": (
                    (summary.get("config") or {}).get("augmentation_mode")
                    or (summary.get("config") or {}).get("augmentation")
                ),
                "two_stage_freeze_epochs": (
                    (summary.get("config") or {}).get("two_stage_freeze_epochs")
                ),
                "error": summary.get("error"),
            }
        )
    comparison = {
        "experiments": rows,
        "best_by_validation_macro_f1": select_best_by_validation(individual),
        "note": (
            "Comparison is based on validation macro-F1 from each experiment's own "
            "validation fold. The held-out test set is used only for the final "
            "candidate (and, where run, by the evaluation stage), never for selecting "
            "between experiments. No experiment is reported as better than another "
            "unless this table shows it."
        ),
    }
    return comparison


def write_experiment_comparison(
    cfg: Config,
    comparison: Dict[str, Any],
    individual: Dict[str, Dict[str, Any]],
    selected: Sequence[str],
) -> Path:
    """Write the experiment comparison artefact under outputs/experiments/."""
    out_dir = cfg.outputs_dir("experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "manifest_version": cfg.manifest_version,
        "selected_experiments": list(selected),
        "comparison": comparison,
        "individual_summaries": {k: v for k, v in individual.items()},
    }
    path = out_dir / "experiment_comparison.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def write_experiment_csvs(cfg: Config, individual: Dict[str, Dict[str, Any]]) -> Dict[str, Path]:
    """Write flat CSVs that summarise each experiment's result for the report."""
    out_dir = cfg.outputs_dir("experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    rows: List[Dict[str, Any]] = []
    for eid, summary in individual.items():
        config = summary.get("config") or {}
        rows.append(
            {
                "experiment": eid,
                "status": summary.get("status"),
                "best_val_macro_f1": summary.get("best_metric_value"),
                "best_epoch": summary.get("best_epoch"),
                "epochs_completed": summary.get("epochs_completed"),
                "loss_variant": config.get("loss_variant"),
                "two_stage_freeze_epochs": config.get("two_stage_freeze_epochs"),
                "checkpoint": summary.get("checkpoint"),
            }
        )
    csv_path = out_dir / "experiment_comparison.csv"
    import csv as _csv

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        if rows:
            writer = _csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    paths["experiment_comparison.csv"] = csv_path
    return paths
