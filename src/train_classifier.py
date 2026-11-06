"""ResNet-50 training on the combined five-class manifest.

Guarantees
----------
* Only ``train_manifest.csv`` and ``validation_manifest.csv`` are ever loaded here.
  The test partition is not opened, scored or inspected by this module.
* Class weights and the best checkpoint come from the training and validation
  partitions only.
* Checkpoints record a fingerprint; a checkpoint whose architecture, class order,
  image size or manifest version differs is never reused or resumed.
* An interrupted run leaves a ``TRAINING_INCOMPLETE.txt`` marker and its ``last``
  checkpoint is resumable.
* A completed run is not silently overwritten: the previous directory is archived.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    write_history_csv,
)
from .hardware import detect_hardware, resolve_device
from .metrics import classification_metrics

INCOMPLETE_MARKER = "TRAINING_INCOMPLETE.txt"
RUN_META = "run_meta.json"
RUN_SUMMARY = "run_summary.json"


class TrainingError(RuntimeError):
    """Raised when training cannot proceed honestly."""


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
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled and device.type == "cuda"):
                logits = model(inputs)
                loss = loss_fn(logits, labels)
            total_loss += float(loss.item()) * inputs.size(0)
            total_samples += inputs.size(0)
            all_logits.append(logits.detach().float().cpu())
            all_labels.append(labels.detach().cpu())

    logits = torch.cat(all_logits, dim=0) if all_logits else torch.zeros(0, len(class_names))
    labels = torch.cat(all_labels, dim=0) if all_labels else torch.zeros(0, dtype=torch.long)
    probabilities = torch.softmax(logits, dim=1).numpy() if len(logits) else np.zeros((0, len(class_names)))
    predictions = logits.argmax(dim=1).numpy() if len(logits) else np.zeros(0, dtype=int)

    metrics = classification_metrics(
        labels.numpy(), predictions,
        probabilities=probabilities,
        class_names=class_names,
        compute_auc=bool(len(logits)),
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
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(inputs)
            loss = loss_fn(logits, labels)
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            if gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
        total_loss += float(loss.item()) * inputs.size(0)
        total_samples += inputs.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())

    return {
        "loss": total_loss / total_samples if total_samples else float("inf"),
        "accuracy": correct / total_samples if total_samples else 0.0,
        "num_samples": total_samples,
    }


def _schedule(total_epochs: int, warmup_epochs: int):
    """Linear warm-up followed by cosine decay, as a per-epoch multiplier."""
    warmup = max(0, int(warmup_epochs))

    def factor(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return float(epoch + 1) / float(warmup)
        progress = (epoch - warmup) / max(1, total_epochs - warmup)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return factor


def run_training(
    cfg: Config,
    logger,
    events=None,
    *,
    force_retrain: bool = False,
    resume: bool = True,
) -> Dict[str, Any]:
    """Train (or reuse) the five-class ResNet-50."""
    from .models import build_resnet50, unfreeze_all

    manifests = cfg.outputs_dir("manifests")
    train_path = manifests / "train_manifest.csv"
    val_path = manifests / "validation_manifest.csv"
    if not train_path.is_file() or not val_path.is_file():
        raise TrainingError(
            "Classification manifests are missing. Run the 'prepare' stage first. "
            f"Looked for {train_path} and {val_path}."
        )

    train_rows = load_manifest_rows(train_path)
    val_rows = load_manifest_rows(val_path)
    if not val_rows:
        raise TrainingError("Validation manifest is empty; cannot select a checkpoint.")

    class_names = list(cfg.classes)
    image_size = int(cfg.require("preprocessing.classification.image_size"))
    fingerprint = build_fingerprint(
        architecture=str(cfg.require("classification.architecture")),
        classes=class_names,
        image_size=image_size,
        manifest_version=cfg.manifest_version,
        extra={"task": "five_class_classification"},
    )

    runs_root = cfg.path("paths.runs_dir", create=True) / "classification"
    run_dir = runs_root
    best_dir = run_dir / BEST_DIR
    last_dir = run_dir / LAST_DIR

    if cfg.get("run.reuse_valid_checkpoints", True) and not force_retrain:
        reusable = find_reusable(best_dir, fingerprint)
        if reusable["compatible"]:
            logger.info("Reusing compatible checkpoint at %s", reusable["path"])
            metrics_path = cfg.outputs_dir("classification") / "classification_metrics.json"
            # Reuse must still publish the artefact downstream stages expect.
            published_path = cfg.outputs_dir("classification") / "best_resnet50.pt"
            if not published_path.is_file():
                import shutil

                shutil.copyfile(best_dir / CHECKPOINT_FILENAME, published_path)
                logger.info("published reused checkpoint to %s", published_path)
            summary = {
                "status": "reused",
                "checkpoint": str(published_path),
                "epoch": reusable.get("epoch"),
                "reused": True,
                "metrics_available": metrics_path.is_file(),
            }
            if events is not None:
                events.write("classification_training_reused", **summary)
            return summary
        if reusable["exists"]:
            logger.warning(
                "Existing checkpoint is NOT reusable: %s", "; ".join(reusable["reasons"])
            )

    device = resolve_device("auto")
    hardware = detect_hardware(
        mixed_precision_preference=cfg.mixed_precision_setting("classification"),
        disk_probe=cfg.root,
    )
    amp_enabled = bool(hardware.mixed_precision)

    batch_size = cfg.effective_batch_size("classification", fallback=hardware.recommended_classification_batch_size)
    num_workers = cfg.effective_num_workers("classification") or hardware.recommended_num_workers
    epochs = int(cfg.require("classification.epochs"))
    lr = float(cfg.require("classification.learning_rate"))
    weight_decay = float(cfg.require("classification.weight_decay"))
    patience = int(cfg.require("classification.early_stopping_patience"))
    freeze_epochs = int(cfg.get("classification.freeze_backbone_epochs", 0) or 0)
    gradient_clip = float(cfg.get("classification.gradient_clip_norm", 1.0) or 0.0)
    label_smoothing = float(cfg.get("classification.label_smoothing", 0.0) or 0.0)
    monitor = str(cfg.get("classification.monitor", "val_macro_f1"))
    monitor_mode = str(cfg.get("classification.monitor_mode", "max"))
    seed = cfg.seed

    logger.info(
        "Classification training: device=%s, amp=%s, batch=%d, workers=%d, epochs=%d, "
        "lr=%g, wd=%g, freeze_backbone_epochs=%d",
        device, amp_enabled, batch_size, num_workers, epochs, lr, weight_decay, freeze_epochs,
    )

    _seed_everything(seed)
    train_dataset = ClassifierDataset(train_rows, cfg, train=True, image_size=image_size)
    val_dataset = ClassifierDataset(val_rows, cfg, train=False, image_size=image_size)
    train_loader = _dataloader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, seed=seed)
    val_loader = _dataloader(val_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, seed=seed)

    model = build_resnet50(
        num_classes=len(class_names),
        pretrained=bool(cfg.get("classification.pretrained", True)),
        allow_weight_download=bool(cfg.get("classification.allow_weight_download", True)),
        freeze_backbone=freeze_epochs > 0,
        logger=logger,
    ).to(device)

    weights = None
    class_weight_values: Dict[str, float] = {}
    if bool(cfg.get("classification.class_weighting", True)):
        array = class_weights_from_rows(train_rows, len(class_names))
        weights = torch.tensor(array, dtype=torch.float32, device=device)
        class_weight_values = {
            name: float(array[index]) for index, name in enumerate(class_names)
        }
        logger.info("class weights (from training partition): %s", class_weight_values)

    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
    optimizer_name = str(cfg.require("classification.optimizer")).lower()
    if optimizer_name == "adamw":
        # All parameters are registered with the optimizer even during the frozen
        # warm-up: parameters with requires_grad=False simply receive no gradient,
        # and unfreezing later works without rebuilding (and losing) optimizer state.
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    else:
        raise TrainingError(
            f"Unsupported optimizer {optimizer_name!r}; configure 'adamw' or 'sgd'."
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_schedule(epochs, int(cfg.get("classification.warmup_epochs", 0) or 0))
    )
    scaler = torch.amp.GradScaler("cuda") if amp_enabled and device.type == "cuda" else None

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
                "Resuming from epoch %d (best %s=%s at epoch %s)",
                start_epoch, monitor, best_value, best_epoch,
            )
            if start_epoch >= epochs:
                logger.info("Checkpoint already trained for %d epochs; nothing to do", epochs)
        else:
            logger.info("No resumable 'last' checkpoint (%s); starting fresh",
                        "; ".join(reusable["reasons"]))

    if run_dir.exists() and start_epoch == 0 and not force_retrain:
        state = load_training_state(last_dir)
        if state and int(state.get("epoch", 0)) + 1 >= epochs:
            archived = archive_previous_run(run_dir)
            if archived:
                logger.info("Archived the previous completed classification run to %s", archived)
    elif run_dir.exists() and force_retrain:
        archived = archive_previous_run(run_dir)
        if archived:
            logger.info("Archived the previous classification run to %s", archived)

    run_dir.mkdir(parents=True, exist_ok=True)
    marker = run_dir / INCOMPLETE_MARKER
    if marker.exists():
        marker.unlink()

    config_snapshot = {
        "seed": seed,
        "image_size": image_size,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "epochs": epochs,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "optimizer": optimizer_name,
        "patience": patience,
        "monitor": monitor,
        "monitor_mode": monitor_mode,
        "freeze_backbone_epochs": freeze_epochs,
        "mixed_precision": amp_enabled,
        "device": str(device),
        "class_names": class_names,
        "class_weights": class_weight_values,
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
                logger.info("Backbone unfrozen at epoch %d", epoch + 1)

            train_stats = _train_epoch(
                model, train_loader, loss_fn, optimizer, device,
                amp_enabled=amp_enabled, scaler=scaler, gradient_clip_norm=gradient_clip,
            )
            val_metrics = _evaluate(model, val_loader, loss_fn, device, class_names, amp_enabled)
            scheduler.step()

            monitor_key = monitor[4:] if monitor.startswith("val_") else monitor
            current = float(val_metrics.get(monitor_key, val_metrics.get(monitor, val_metrics["loss"])))
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
                "epoch %d/%d | train_loss %.4f acc %.4f | val_loss %.4f acc %.4f macroF1 %.4f | %s%s",
                epoch + 1, epochs, row["train_loss"], row["train_accuracy"],
                row["val_loss"], row["val_accuracy"], row["val_macro_f1"],
                row["seconds"], "s *" if improved else "s",
            )

            save_checkpoint(
                last_dir, model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, history=history, best_metric=best_value, best_epoch=best_epoch,
                fingerprint=fingerprint, extra={"config": config_snapshot},
            )
            if improved:
                save_checkpoint(
                    best_dir, model=model, optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch, history=history, best_metric=best_value, best_epoch=best_epoch,
                    fingerprint=fingerprint, extra={"config": config_snapshot},
                )
                write_history_csv(cfg.outputs_dir("classification") / "training_history.csv", history)

            if epochs_without_improvement >= patience:
                logger.info(
                    "Early stopping: %s did not improve for %d epochs", monitor, patience
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
        write_history_csv(cfg.outputs_dir("classification") / "training_history.csv", history)

    elapsed = time.time() - started

    # Publish the best checkpoint where the request expects it.
    published = cfg.outputs_dir("classification") / "best_resnet50.pt"
    best_model_path = best_dir / CHECKPOINT_FILENAME
    if best_model_path.is_file():
        import shutil

        shutil.copyfile(best_model_path, published)
        logger.info("best checkpoint published to %s", published)

    summary = {
        "status": "completed" if completed else "partial",
        "checkpoint": str(published),
        "best_checkpoint_dir": str(best_dir),
        "last_checkpoint_dir": str(last_dir),
        "best_epoch": best_epoch,
        "best_metric_name": monitor,
        "best_metric_value": None if best_value in (math.inf, -math.inf) else float(best_value),
        "epochs_completed": len(history),
        "epochs_requested": epochs,
        "seconds": round(elapsed, 2),
        "device": str(device),
        "mixed_precision": amp_enabled,
        "class_weights": class_weight_values,
        "fingerprint": fingerprint,
        "config": config_snapshot,
        "history_csv": str(cfg.outputs_dir("classification") / "training_history.csv"),
        "reused": False,
    }
    _atomic_write_json(summary, run_dir / RUN_SUMMARY)
    _atomic_write_json(summary, cfg.outputs_dir("classification") / "training_summary.json")

    if events is not None:
        events.write("classification_training_done", **{
            "status": summary["status"],
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "device": str(device),
        })
    return summary
