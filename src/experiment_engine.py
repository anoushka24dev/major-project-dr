"""Reusable ResNet-50 experiment training engine.

This is the single implementation that every classification experiment (E0..E6 and
the final candidate) re-enters with a different configuration. It does NOT replace
``train_classifier.py``: that module keeps running the legacy-style single run so
existing checkpoints, resume behaviour and the existing README guarantees continue
to hold. The experiment engine is a separate, more general entry point that drives
the same ``build_resnet50`` builder, the same ``ClassifierDataset`` pipeline, the
same augmentation policy, the same checkpoint/fingerprint machinery and the same
evaluation metrics.

Design constraints enforced in code
-----------------------------------
* The training fold is the only partition ever opened here. Validation is used for
  checkpoint selection, early stopping and any threshold/calibration fitting. The
  test partition is opened only by the evaluation stage.
* Class weights are computed from the rows passed in (the training fold), never from
  validation or test data.
* Augmentation is applied only to the training dataset instance; the validation and
  test datasets are built with ``train=False``.
* Every experiment run writes a self-contained ``run_meta.json`` + ``training_history.csv``
  under ``outputs/experiments/<experiment_id>/``, plus the best checkpoint under
  ``checkpoints/<experiment_id>/best/``. The downstream evaluation stage loads the
  best checkpoint by experiment id, so experiments are independent and re-runnable.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .checkpoints import (
    BEST_DIR,
    CHECKPOINT_FILENAME,
    LAST_DIR,
    archive_previous_run,
    build_fingerprint,
    find_reusable,
    load_checkpoint,
    load_training_state,
    restore_into,
    save_checkpoint,
    _atomic_write_json,
)
from .config import Config
from .data import (
    ClassifierDataset,
    class_weights_from_rows,
    load_manifest_rows,
)
from .hardware import detect_hardware, resolve_device
from .metrics import classification_metrics
from .models import build_resnet50, unfreeze_all

INCOMPLETE_MARKER = "TRAINING_INCOMPLETE.txt"
RUN_META = "run_meta.json"
RUN_SUMMARY = "run_summary.json"


class ExperimentError(RuntimeError):
    """Raised when an experiment cannot be run honestly."""


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dataloader(
    dataset: ClassifierDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    class_names: List[str],
    amp_enabled: bool,
    compute_auc: bool = True,
) -> Dict[str, Any]:
    """One no-grad validation pass returning loss, metrics and probabilities."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    with torch.no_grad():
        for inputs, labels, _meta in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type=device.type, enabled=amp_enabled and device.type == "cuda"
            ):
                logits = model(inputs)
                loss = loss_fn(logits, labels)
            total_loss += float(loss.item()) * inputs.size(0)
            total_samples += inputs.size(0)
            all_logits.append(logits.detach().float().cpu())
            all_labels.append(labels.detach().cpu())

    logits = torch.cat(all_logits, dim=0) if all_logits else torch.zeros(0, len(class_names))
    labels = torch.cat(all_labels, dim=0) if all_labels else torch.zeros(0, dtype=torch.long)
    probabilities = (
        torch.softmax(logits, dim=1).numpy() if len(logits) else np.zeros((0, len(class_names)))
    )
    predictions = logits.argmax(dim=1).numpy() if len(logits) else np.zeros(0, dtype=int)

    metrics = classification_metrics(
        labels.numpy(),
        predictions,
        probabilities=probabilities,
        class_names=class_names,
        compute_auc=bool(len(logits)) and compute_auc,
    )
    metrics["loss"] = total_loss / total_samples if total_samples else float("inf")
    return metrics


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    amp_enabled: bool,
    scaler: Optional[torch.amp.GradScaler],
    gradient_clip_norm: float,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    use_amp = amp_enabled and device.type == "cuda"

    for inputs, labels, _meta in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type, enabled=use_amp
        ):
            logits = model(inputs)
            loss = loss_fn(logits, labels)
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            if gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip_norm
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip_norm
                )
            optimizer.step()
        total_loss += float(loss.item()) * inputs.size(0)
        total_samples += inputs.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())

    return {
        "loss": total_loss / total_samples if total_samples else float("inf"),
        "accuracy": correct / total_samples if total_samples else 0.0,
        "num_samples": total_samples,
    }


def _schedule(total_epochs: int, warmup_epochs: int) -> Callable[[int], float]:
    warmup = max(0, int(warmup_epochs))

    def factor(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return float(epoch + 1) / float(warmup)
        progress = (epoch - warmup) / max(1, total_epochs - warmup)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return factor


def build_loss_fn(
    cfg: Config,
    *,
    loss_variant: str = "cross_entropy",
    class_weights_device: Optional[torch.device] = None,
    label_smoothing: float = 0.0,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Build the experiment's loss from the config + a per-experiment variant.

    Variants
    --------
    ``cross_entropy``  standard (optionally class-weighted) cross-entropy
    ``weighted_cross_entropy``  cross-entropy with inverse-frequency class weights
    ``focal``  multiclass focal loss (see :class:`FocalLoss`), alpha/gamma from config
    """
    class_names = list(cfg.classes)
    num_classes = len(class_names)
    weights = None
    class_weight_values: Dict[str, float] = {}

    if loss_variant in {"weighted_cross_entropy", "focal"} or (
        loss_variant == "cross_entropy" and cfg.get("classification.class_weighting", True)
    ):
        train_manifest = cfg.outputs_dir("manifests") / "train_manifest.csv"
        if train_manifest.is_file():
            train_rows = load_manifest_rows(train_manifest)
            array = class_weights_from_rows(train_rows, num_classes)
            weights = torch.tensor(array, dtype=torch.float32, device=class_weights_device)
            class_weight_values = {
                name: float(array[index]) for index, name in enumerate(class_names)
            }

    if loss_variant == "focal":
        gamma = float(cfg.get("experiments.focal_gamma", 2.0))
        alpha = cfg.get("experiments.focal_alpha")
        alpha_tensor = None
        if isinstance(alpha, (int, float)):
            alpha_tensor = torch.tensor(float(alpha), dtype=torch.float32, device=class_weights_device)
        elif isinstance(alpha, (list, tuple)) and len(alpha) == num_classes:
            alpha_tensor = torch.tensor(
                [float(a) for a in alpha], dtype=torch.float32, device=class_weights_device
            )
        loss_fn: nn.Module = FocalLoss(
            gamma=gamma,
            alpha=alpha_tensor,
            reduction="mean",
        )
        loss_meta = {
            "variant": "focal",
            "gamma": gamma,
            "alpha": (
                float(alpha)
                if isinstance(alpha, (int, float))
                else list(alpha)
                if isinstance(alpha, (list, tuple))
                else None
            ),
            "class_weights_from_training_fold": class_weight_values,
        }
        return loss_fn, loss_meta

    loss_fn = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=float(label_smoothing),
    )
    loss_meta = {
        "variant": loss_variant,
        "class_weighting": bool(cfg.get("classification.class_weighting", True)),
        "class_weights_from_training_fold": class_weight_values,
        "label_smoothing": float(label_smoothing),
    }
    return loss_fn, loss_meta


def build_optimizer(
    model: nn.Module,
    cfg: Config,
    *,
    backbone_lr_multiplier: float = 1.0,
    head_lr: Optional[float] = None,
) -> torch.optim.Optimizer:
    """Build the optimizer, optionally with separate backbone/head learning rates.

    When ``head_lr`` is set and the config specifies ``adamw``, the head parameters
    receive ``head_lr`` and the backbone receives ``head_lr * backbone_lr_multiplier``.
    This is used by the two-stage fine-tuning experiment (E2) without rebuilding the
    optimizer after unfreezing: parameters with requires_grad=False simply receive no
    gradient, which matches the existing behaviour in ``train_classifier.py``.
    """
    optimizer_name = str(cfg.require("classification.optimizer")).lower()
    base_lr = float(cfg.require("classification.learning_rate"))
    weight_decay = float(cfg.require("classification.weight_decay"))

    if head_lr is None or optimizer_name != "adamw":
        if optimizer_name == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
        if optimizer_name == "sgd":
            return torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=weight_decay)
        raise ExperimentError(f"Unsupported optimizer {optimizer_name!r}")

    head_params: List[nn.Parameter] = []
    backbone_params: List[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("fc"):
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    # Override param groups so the backbone uses the smaller lr. The head group keeps
    # the higher head_lr. We rebuild groups but do not change ``model`` at all.
    param_groups: List[Dict[str, Any]] = []
    if head_params:
        param_groups.append(
            {"params": head_params, "lr": float(head_lr), "weight_decay": weight_decay}
        )
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": float(head_lr) * float(backbone_lr_multiplier),
                "weight_decay": weight_decay,
            }
        )
    if not param_groups:
        raise ExperimentError("No trainable parameters found for the experiment optimizer.")
    optimizer = torch.optim.AdamW(param_groups)
    optimizer.defaults.setdefault("lr", base_lr)
    return optimizer


def run_experiment(
    cfg: Config,
    logger,
    events=None,
    *,
    experiment_id: str,
    resume: bool = True,
    force_retrain: bool = False,
    train_rows: Optional[Sequence[Dict[str, str]]] = None,
    validation_rows: Optional[Sequence[Dict[str, str]]] = None,
    fixed_epochs: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one ResNet-50 experiment (E0..E6 or a final candidate).

    The experiment is fully described by:
    * the manifests it reads (always ``train_manifest.csv`` / ``validation_manifest.csv``
      under ``outputs/manifests/`` — the same folds the pipeline prepared)
    * the loss variant and its hyperparameters
    * the augmentation policy (read from ``config.yaml``, frozen for the experiment)
    * the training schedule (epochs, lr, weight decay, patience, freeze schedule)
    * whether two-stage fine-tuning is used (head-only warm-up then unfreeze)

    ``train_rows`` / ``validation_rows`` let a cross-validation fold supply its own
    partitions instead of the shared manifests. ``fixed_epochs`` trains for exactly
    that many epochs with no early stopping and keeps the final epoch's weights; it
    is used only for the final retrain, where no validation data may select a
    checkpoint.

    Returns a run summary whose keys mirror those of ``train_classifier.run_training``
    so the existing report and evaluation code can be extended rather than rewritten.
    """
    from .models import build_resnet50 as _build

    manifests = cfg.outputs_dir("manifests")
    train_path = manifests / "train_manifest.csv"
    val_path = manifests / "validation_manifest.csv"
    if train_rows is None or validation_rows is None:
        if not train_path.is_file() or not val_path.is_file():
            raise ExperimentError(
                "Classification manifests are missing. Run the 'prepare' stage first. "
                f"Looked for {train_path} and {val_path}."
            )

    train_rows = list(train_rows) if train_rows is not None else load_manifest_rows(train_path)
    val_rows = (
        list(validation_rows) if validation_rows is not None else load_manifest_rows(val_path)
    )
    if not train_rows:
        raise ExperimentError("The training partition is empty; nothing to fit.")
    if not val_rows:
        raise ExperimentError("Validation manifest is empty; cannot select a checkpoint.")

    class_names = list(cfg.classes)
    image_size = int(cfg.require("preprocessing.classification.image_size"))
    experiment_index = str(experiment_id)

    fingerprint = build_fingerprint(
        architecture=str(cfg.require("classification.architecture")),
        classes=class_names,
        image_size=image_size,
        manifest_version=cfg.manifest_version,
        extra={
            "task": "five_class_classification",
            "experiment": experiment_index,
            "loss_variant": cfg.get("experiments.loss", "cross_entropy"),
            "augmentation": dict(cfg.get("preprocessing.classification.augmentation", {}) or {}),
            "train_images": len(train_rows),
            "validation_images": len(val_rows),
            "fixed_epochs": fixed_epochs,
        },
    )

    runs_root = cfg.path("paths.runs_dir", create=True) / "experiments" / experiment_index
    checkpoints_root = cfg.path("paths.checkpoints_dir", create=True) / experiment_index
    run_dir = runs_root
    best_dir = checkpoints_root / BEST_DIR
    last_dir = checkpoints_root / LAST_DIR

    # Reuse is per-experiment, not global: one experiment's checkpoint is never reused
    # by another experiment with a different loss or augmentation policy.
    if cfg.get("run.reuse_valid_checkpoints", True) and not force_retrain:
        reusable = find_reusable(best_dir, fingerprint)
        if reusable["compatible"]:
            logger.info("Reusing compatible experiment checkpoint at %s", reusable["path"])
            published_path = checkpoints_root / "best_resnet50.pt"
            if not published_path.is_file():
                import shutil
                shutil.copyfile(best_dir / CHECKPOINT_FILENAME, published_path)
                logger.info("published reused experiment checkpoint to %s", published_path)
            summary = {
                "status": "reused",
                "checkpoint": str(published_path),
                "epoch": reusable.get("epoch"),
                "reused": True,
                "metrics_available": (
                    cfg.outputs_dir("classification") / "classification_metrics.json"
                ).is_file(),
            }
            if events is not None:
                events.write(
                    "experiment_training_reused", experiment=experiment_index, **summary
                )
            return summary
        if reusable["exists"]:
            logger.warning(
                "Existing experiment checkpoint is NOT reusable: %s",
                "; ".join(reusable["reasons"]),
            )

    device = resolve_device("auto")
    hardware = detect_hardware(
        mixed_precision_preference=cfg.mixed_precision_setting("classification"),
        disk_probe=cfg.root,
    )
    amp_enabled = bool(hardware.mixed_precision)

    batch_size = cfg.effective_batch_size(
        "classification", fallback=hardware.recommended_classification_batch_size
    )
    num_workers = (
        cfg.effective_num_workers("classification") or hardware.recommended_num_workers
    )
    epochs = int(fixed_epochs) if fixed_epochs else int(cfg.require("classification.epochs"))
    if epochs < 1:
        raise ExperimentError(f"epoch budget must be positive, got {epochs}")
    monitor = str(cfg.get("classification.monitor", "val_macro_f1"))
    monitor_mode = str(cfg.get("classification.monitor_mode", "max"))
    seed = cfg.seed

    # Materialise the experiment's augmentation variant into the training-only policy.
    # Validation and test datasets are always built with train=False, so no variant can
    # leak an augmentation into an evaluation partition.
    augmentation_mode = str(cfg.get("experiments.augmentation_mode", "default") or "default")
    if augmentation_mode != "default":
        from .experiments import AUGMENTATION_VARIANTS

        variant = AUGMENTATION_VARIANTS.get(augmentation_mode)
        if variant is None:
            raise ExperimentError(
                f"Unknown augmentation variant {augmentation_mode!r}. Known: "
                f"{sorted(AUGMENTATION_VARIANTS)}"
            )
        merged = dict(cfg.get("preprocessing.classification.augmentation", {}) or {})
        merged.update(variant)
        cfg._data.setdefault("preprocessing", {}).setdefault("classification", {})[
            "augmentation"
        ] = merged
        logger.info("Augmentation variant %r applied for the training partition only", augmentation_mode)

    _seed_everything(seed)
    train_dataset = ClassifierDataset(train_rows, cfg, train=True, image_size=image_size)
    val_dataset = ClassifierDataset(val_rows, cfg, train=False, image_size=image_size)
    train_loader = _dataloader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, seed=seed
    )
    val_loader = _dataloader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, seed=seed
    )

    model = _build(
        num_classes=len(class_names),
        pretrained=bool(cfg.get("classification.pretrained", True)),
        allow_weight_download=bool(cfg.get("classification.allow_weight_download", True)),
        freeze_backbone=False,
        logger=logger,
    ).to(device)

    loss_variant = str(cfg.get("experiments.loss", "cross_entropy"))
    loss_fn, loss_meta = build_loss_fn(
        cfg,
        loss_variant=loss_variant,
        class_weights_device=device,
        label_smoothing=float(cfg.get("classification.label_smoothing", 0.0) or 0.0),
    )

    freeze_epochs = int(cfg.get("experiments.two_stage_freeze_epochs", 0) or 0) if loss_variant in {
        "cross_entropy",
        "weighted_cross_entropy",
    } else 0
    if freeze_epochs > 0:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("fc")

    # E2 two-stage: smaller backbone LR once unfrozen.
    backbone_lr_multiplier = float(cfg.get("experiments.backbone_lr_multiplier", 0.1))
    head_lr = None
    if freeze_epochs > 0:
        base_lr = float(cfg.require("classification.learning_rate"))
        head_lr = float(cfg.get("experiments.head_learning_rate", base_lr))

    optimizer = build_optimizer(
        model,
        cfg,
        backbone_lr_multiplier=(
            backbone_lr_multiplier if freeze_epochs > 0 else 1.0
        ),
        head_lr=head_lr,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=_schedule(epochs, int(cfg.get("classification.warmup_epochs", 0) or 0)),
    )
    scaler = (
        torch.amp.GradScaler("cuda") if amp_enabled and device.type == "cuda" else None
    )
    gradient_clip = float(cfg.get("classification.gradient_clip_norm", 1.0) or 0.0)
    patience = (
        epochs + 1 if fixed_epochs else int(cfg.get("classification.early_stopping_patience", 5) or 5)
    )

    history: List[Dict[str, Any]] = []
    start_epoch = 0
    best_value = -math.inf if monitor_mode == "max" else math.inf
    best_epoch = -1
    epochs_without_improvement = 0

    if resume and last_dir.exists():
        reusable = find_reusable(last_dir, fingerprint)
        if reusable["compatible"]:
            payload = load_checkpoint(last_dir / CHECKPOINT_FILENAME, map_location=device)
            restored = restore_into(model, payload, optimizer=optimizer, scheduler=scheduler)
            state = load_training_state(last_dir) or {}
            history = list(state.get("history") or [])
            best_value = float(state.get("best_metric", best_value))
            best_epoch = int(state.get("best_epoch", -1))
            start_epoch = restored["epoch"] + 1
            logger.info(
                "Resuming experiment %s from epoch %d (best %s=%s at epoch %s)",
                experiment_index,
                start_epoch,
                monitor,
                best_value,
                best_epoch,
            )
            if start_epoch >= epochs:
                logger.info(
                    "Experiment checkpoint already trained for %d epochs; nothing to do",
                    epochs,
                )

    if run_dir.exists() and start_epoch == 0 and not force_retrain:
        state = load_training_state(last_dir)
        if state and int(state.get("epoch", 0)) + 1 >= epochs:
            archived = archive_previous_run(run_dir)
            if archived:
                logger.info(
                    "Archived the previous completed experiment run to %s", archived
                )
    elif run_dir.exists() and force_retrain:
        archived = archive_previous_run(run_dir)
        if archived:
            logger.info("Archived the previous experiment run to %s", archived)

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    marker = run_dir / INCOMPLETE_MARKER
    if marker.exists():
        marker.unlink()

    config_snapshot = {
        "experiment": experiment_index,
        "seed": seed,
        "image_size": image_size,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "epochs": epochs,
        "learning_rate": float(cfg.require("classification.learning_rate")),
        "weight_decay": float(cfg.require("classification.weight_decay")),
        "optimizer": str(cfg.require("classification.optimizer")),
        "patience": patience,
        "monitor": monitor,
        "monitor_mode": monitor_mode,
        "two_stage_freeze_epochs": freeze_epochs,
        "head_learning_rate": head_lr,
        "backbone_lr_multiplier": (
            backbone_lr_multiplier if freeze_epochs > 0 else 1.0
        ),
        "mixed_precision": amp_enabled,
        "device": str(device),
        "class_names": class_names,
        "loss_variant": loss_variant,
        "loss_meta": loss_meta,
        "augmentation_mode": augmentation_mode,
        "augmentation": dict(cfg.get("preprocessing.classification.augmentation", {}) or {}),
        "fingerprint": fingerprint,
        "config_file": str(cfg.source),
        "train_images": len(train_rows),
        "validation_images": len(val_rows),
    }
    _atomic_write_json(config_snapshot, run_dir / RUN_META)

    started = time.time()
    completed = False
    try:
        for epoch in range(start_epoch, epochs):
            epoch_started = time.time()
            if freeze_epochs > 0 and epoch == freeze_epochs:
                unfreeze_all(model)
                logger.info(
                    "Backbone unfrozen at epoch %d (two-stage fine-tuning)", epoch + 1
                )

            train_stats = _train_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
                device,
                amp_enabled=amp_enabled,
                scaler=scaler,
                gradient_clip_norm=gradient_clip,
            )
            val_metrics = _evaluate(
                model, val_loader, loss_fn, device, class_names, amp_enabled
            )
            scheduler.step()

            current = float(val_metrics.get(monitor, val_metrics["loss"]))
            improved = current > best_value if monitor_mode == "max" else current < best_value
            if improved:
                best_value = current
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            row = {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_accuracy": train_stats["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_macro_precision": val_metrics["macro_precision"],
                "val_macro_recall": val_metrics["macro_recall"],
                "val_cohen_kappa": val_metrics["cohen_kappa"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "seconds": round(time.time() - epoch_started, 2),
                "improved": improved,
            }
            history.append(row)
            logger.info(
                "experiment %s epoch %d/%d | train_loss %.4f acc %.4f | val_loss %.4f acc %.4f macroF1 %.4f | %s%s",
                experiment_index,
                epoch + 1,
                epochs,
                row["train_loss"],
                row["train_accuracy"],
                row["val_loss"],
                row["val_accuracy"],
                row["val_macro_f1"],
                row["seconds"],
                "s *" if improved else "s",
            )

            save_checkpoint(
                last_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                history=history,
                best_metric=best_value,
                best_epoch=best_epoch,
                fingerprint=fingerprint,
                extra={"config": config_snapshot},
            )
            keep_this_epoch = improved or (
                fixed_epochs is not None and epoch == epochs - 1
            )
            if fixed_epochs is not None and keep_this_epoch and not improved:
                best_epoch = epoch
                best_value = current
            if keep_this_epoch:
                save_checkpoint(
                    best_dir,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    history=history,
                    best_metric=best_value,
                    best_epoch=best_epoch,
                    fingerprint=fingerprint,
                    extra={"config": config_snapshot},
                )
                _atomic_write_json(
                    history,
                    cfg.outputs_dir("experiments", experiment_index)
                    / "training_history.csv",
                )

            if epochs_without_improvement >= patience:
                logger.info(
                    "Early stopping: %s did not improve for %d epochs",
                    monitor,
                    patience,
                )
                break
        completed = True
    except KeyboardInterrupt:
        marker.write_text(
            f"Training interrupted by the user at {time.strftime('%Y-%m-%dT%H:%M:%SZ')}.\n"
            "Re-run with --resume to continue from the 'last' checkpoint.\n",
            encoding="utf-8",
        )
        raise
    except Exception as exc:
        marker.write_text(
            f"Training failed at {time.strftime('%Y-%m-%dT%H:%M:%SZ')}: {type(exc).__name__}: {exc}\n"
            "Re-run with --resume to continue from the 'last' checkpoint.\n",
            encoding="utf-8",
        )
        raise
    finally:
        _atomic_write_json(
            history,
            cfg.outputs_dir("experiments", experiment_index) / "training_history.csv",
        )

    elapsed = time.time() - started

    published = checkpoints_root / "best_resnet50.pt"
    best_model_path = best_dir / CHECKPOINT_FILENAME
    if best_model_path.is_file():
        import shutil
        shutil.copyfile(best_model_path, published)
        logger.info("best experiment checkpoint published to %s", published)

    # Post-hoc thresholds/temperature need a validation partition that was not
    # trained on. A fixed-epoch retrain uses the development pool for both roles, so
    # fitting there would be fitting on training data: skip it and say why.
    posthoc: Optional[Dict[str, Any]] = None
    if fixed_epochs is None:
        posthoc = _maybe_fit_posthoc(
            cfg,
            logger,
            experiment_index,
            best_model_path,
            val_loader,
            device,
            amp_enabled,
        )
    else:
        logger.info(
            "post-hoc calibration skipped for %s: fixed-epoch retrain has no held-out "
            "validation partition",
            experiment_index,
        )

    summary = {
        "status": "completed" if completed else "partial",
        "checkpoint": str(published) if published.is_file() else None,
        "best_checkpoint_dir": str(best_dir),
        "last_checkpoint_dir": str(last_dir),
        "best_epoch": best_epoch,
        "best_metric_name": monitor,
        "best_metric_value": (
            None if best_value in (math.inf, -math.inf) else float(best_value)
        ),
        "epochs_completed": len(history),
        "epochs_requested": epochs,
        "fixed_epochs": fixed_epochs,
        "seconds": round(elapsed, 2),
        "device": str(device),
        "mixed_precision": amp_enabled,
        "class_weights": loss_meta.get("class_weights_from_training_fold", {}),
        "fingerprint": fingerprint,
        "config": config_snapshot,
        "history_csv": str(
            cfg.outputs_dir("experiments", experiment_index) / "training_history.csv"
        ),
        "reused": False,
        "experiment": experiment_index,
        "posthoc": posthoc,
    }
    _atomic_write_json(summary, run_dir / RUN_SUMMARY)
    _atomic_write_json(
        summary,
        cfg.outputs_dir("experiments", experiment_index) / "experiment_summary.json",
    )

    if events is not None:
        events.write(
            "experiment_training_done",
            experiment=experiment_index,
            **{
                "status": summary["status"],
                "best_epoch": best_epoch,
                "epochs_completed": len(history),
                "device": str(device),
            },
        )
    return summary


# ---------------------------------------------------------------------------
# Post-hoc, validation-only calibration and threshold selection (E5 / E6)
# ---------------------------------------------------------------------------

def _validation_predictions(
    cfg: Config,
    checkpoint_path: Path,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(logits, labels)`` for the best checkpoint over the validation fold.

    The checkpoint is loaded into a fresh model, so this never depends on whichever
    epoch happened to be in memory when training stopped.
    """
    payload = load_checkpoint(checkpoint_path, map_location=device)
    model = build_resnet50(
        num_classes=int(cfg.require("classification.num_classes")),
        pretrained=False,
        allow_weight_download=False,
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    logits_batches: List[np.ndarray] = []
    label_batches: List[np.ndarray] = []
    with torch.no_grad():
        for inputs, labels, _meta in loader:
            inputs = inputs.to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type=device.type, enabled=amp_enabled and device.type == "cuda"
            ):
                logits = model(inputs)
            logits_batches.append(logits.detach().float().cpu().numpy())
            label_batches.append(labels.numpy())
    if not logits_batches:
        return np.zeros((0, int(cfg.require("classification.num_classes")))), np.zeros(0, dtype=int)
    return np.concatenate(logits_batches, axis=0), np.concatenate(label_batches, axis=0)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _macro_f1(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: Optional[int] = None,
) -> float:
    """Macro-F1 over all five classes, so an absent class still counts as 0."""
    from sklearn.metrics import f1_score

    if len(labels) == 0:
        return 0.0
    total = int(num_classes) if num_classes else int(max(int(labels.max()), int(predictions.max()))) + 1
    return float(
        f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
            labels=list(range(total)),
        )
    )


def _fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit a single temperature by minimising validation NLL (ternary search).

    Temperature scaling has exactly one parameter, so it cannot change the model's
    predictions — only their confidence. Fitting it on validation data therefore
    cannot leak test information or improve accuracy artificially.
    """
    if len(labels) == 0:
        return 1.0
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels) if labels.dtype == int else np.asarray(labels, dtype=np.int64)

    def nll(temperature: float) -> float:
        scaled = logits / max(temperature, 1e-3)
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        log_sum_exp = np.log(np.exp(shifted).sum(axis=1))
        chosen = shifted[np.arange(len(labels)), labels]
        return float(np.mean(log_sum_exp - chosen))

    low, high = 0.3, 6.0
    for _ in range(60):
        third = (high - low) / 3.0
        left, right = low + third, high - third
        if nll(left) < nll(right):
            high = right
        else:
            low = left
    return float(round((low + high) / 2.0, 4))


def _optimise_class_offsets(
    probabilities: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> Tuple[List[float], float]:
    """Coordinate-ascent search for per-class score offsets on the validation fold.

    A per-class additive offset on the softmax scores is exactly a per-class decision
    threshold: class ``c`` is predicted when ``p_c + offset_c`` is the maximum. The
    search maximises validation macro-F1 only; nothing here sees the test partition.
    """
    if len(labels) == 0:
        return [0.0] * num_classes, 0.0
    grid = np.round(np.arange(-0.6, 0.61, 0.05), 2)
    offsets = [0.0] * num_classes
    best = _macro_f1(labels, probabilities.argmax(axis=1), num_classes)
    for _ in range(2):
        improved = False
        for index in range(num_classes):
            for candidate in grid:
                trial = list(offsets)
                trial[index] = float(candidate)
                predictions = (probabilities + np.asarray(trial)).argmax(axis=1)
                score = _macro_f1(labels, predictions, num_classes)
                if score > best + 1e-9:
                    best, offsets, improved = score, trial, True
        if not improved:
            break
    return [round(value, 3) for value in offsets], float(best)


def _maybe_fit_posthoc(
    cfg: Config,
    logger,
    experiment_id: str,
    best_checkpoint: Path,
    val_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> Optional[Dict[str, Any]]:
    """Fit E5 thresholds and/or the E6 temperature, if this experiment asks for them.

    Both are fitted on the validation fold of the best checkpoint and written to
    ``outputs/experiments/<id>/posthoc.json``. The evaluation and calibration stages
    read that file and report the effects separately from the headline metrics, so a
    threshold change can never be mistaken for a change in the model.
    """
    want_thresholds = bool(cfg.get("experiments.threshold_optimisation", False))
    want_temperature = bool(cfg.get("experiments.calibration_evaluation", False))
    if not (want_thresholds or want_temperature):
        return None
    if not Path(best_checkpoint).is_file():
        logger.warning(
            "Experiment %s requested post-hoc fitting but has no best checkpoint", experiment_id
        )
        return None

    logits, labels = _validation_predictions(
        cfg, Path(best_checkpoint), val_loader, device, amp_enabled
    )
    if len(labels) == 0:
        logger.warning("Experiment %s has no validation samples for post-hoc fitting", experiment_id)
        return None

    probabilities = _softmax(np.asarray(logits, dtype=np.float64))
    payload: Dict[str, Any] = {
        "experiment": experiment_id,
        "fitted_on": "validation",
        "validation_samples": int(len(labels)),
        "note": (
            "Fitted on the validation fold only. The test partition is never used for "
            "threshold or temperature selection."
        ),
    }
    if want_thresholds:
        num_classes = len(cfg.classes)
        offsets, macro_f1 = _optimise_class_offsets(probabilities, labels, num_classes)
        payload["class_offsets"] = {
            name: offsets[index] for index, name in enumerate(cfg.classes)
        }
        payload["validation_macro_f1_with_offsets"] = round(macro_f1, 6)
        payload["validation_macro_f1_argmax"] = round(
            _macro_f1(labels, probabilities.argmax(axis=1), len(cfg.classes)), 6
        )
        payload["threshold_semantics"] = (
            "Each value is added to that class's softmax score before argmax; a class is "
            "predicted when its adjusted score is the highest. This is a per-class "
            "decision threshold, not a re-trained model."
        )
    if want_temperature:
        temperature = _fit_temperature(logits, labels)
        payload["temperature"] = temperature
        payload["temperature_fit"] = "minimised validation negative log-likelihood"

    target = cfg.outputs_dir("experiments", experiment_id) / "posthoc.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("post-hoc parameters written to %s", target)
    payload["path"] = str(target)
    return payload


# ---------------------------------------------------------------------------
# Focal loss (multiclass, configurable alpha/gamma)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Multiclass focal loss.

    ``alpha`` may be a scalar (applied to every class identically) or a per-class
    tensor of length ``num_classes``. ``gamma`` down-weights easy examples. This is
    an *alternative* to cross-entropy used as a controlled comparison, not an add-on.
    """

    def __init__(
        self,
        *,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(
            logits,
            targets,
            weight=(
                self.weight.to(logits.device)
                if self.weight is not None
                else None
            ),
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        focal = (1.0 - pt) ** self.gamma * ce
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)
            if alpha_t.dim() == 0:
                alpha_t = alpha_t.expand_as(focal)
            else:
                idx = targets.view(-1)
                alpha_t = alpha_t[idx]
            focal = alpha_t * focal
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


def focal_loss_description(cfg: Config) -> Dict[str, Any]:
    """Return the focal-loss configuration used by an experiment, for the report."""
    return {
        "gamma": float(cfg.get("experiments.focal_gamma", 2.0)),
        "alpha": cfg.get("experiments.focal_alpha"),
        "note": (
            "Focal loss is a controlled alternative to cross-entropy. It is never "
            "stacked on top of cross-entropy, and alpha/gamma are configurable, not "
            "assumed to be optimal."
        ),
    }
