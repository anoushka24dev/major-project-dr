"""Train a five-class ResNet-50 on the 50-image demonstration training subset.

Guarantees
----------
* Only ``manifests/selected_train.csv`` (50 rows: 41 fit + 9 validation) is ever
  loaded here. ``selected_test.csv`` is not opened, so the 10 test images cannot
  influence training, augmentation, class weights, early stopping or checkpoint
  selection.
* Validation is carved out of the 50 training images; it is never drawn from the
  test partition.
* Preprocessing and augmentation are the repository's own implementations
  (``src.data.ClassifierDataset`` driven by ``config/config.yaml``), and the model
  comes from ``src.models.build_resnet50``.
* An interrupted run leaves ``model/TRAINING_INCOMPLETE.txt`` and a resumable
  ``last_model.pt``; re-run with ``--resume`` to continue.
* A completed run is never silently overwritten: the previous ``results/``
  training artefacts are moved to ``results/previous_run_<timestamp>/``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from prepare_data import (
    extract_root,
    load_demo_config,
    load_manifest,
    manifest_path,
)
from src.config import load_config
from src.data import ClassifierDataset, class_weights_from_rows, write_history_csv
from src.metrics import classification_metrics
from src.models import build_resnet50

RESULTS_DIR = DEMO_DIR / "results"
MODEL_DIR = DEMO_DIR / "model"
BEST_CHECKPOINT = MODEL_DIR / "best_model.pt"
LAST_CHECKPOINT = MODEL_DIR / "last_model.pt"
INCOMPLETE_MARKER = MODEL_DIR / "TRAINING_INCOMPLETE.txt"


class TrainingError(RuntimeError):
    """Raised when training cannot proceed honestly."""


# ---------------------------------------------------------------------------
# Reproducibility and small helpers
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device() -> torch.device:
    from src.hardware import resolve_device

    return resolve_device("auto")


def set_trainable(model: nn.Module, prefixes: List[str]) -> List[str]:
    """Freeze everything except parameters whose name starts with a prefix."""
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(prefix) for prefix in prefixes)
    return [name for name, p in model.named_parameters() if p.requires_grad]


def freeze_batchnorm(model: nn.Module) -> None:
    """Keep BatchNorm in eval mode: 41 images/epoch cannot estimate its statistics."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def trainable_summary(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}


def rows_digest(rows: List[Dict[str, str]]) -> str:
    import hashlib

    joined = "\n".join(sorted(f"{r['rel_path']}|{r['mapped_label']}" for r in rows))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def build_schedule(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    schedule: List[Dict[str, Any]] = []
    for stage_index, stage in enumerate(config["training"]["stages"]):
        for epoch_in_stage in range(int(stage["epochs"])):
            schedule.append(
                {
                    "stage_index": stage_index,
                    "stage_name": stage["name"],
                    "epoch_in_stage": epoch_in_stage,
                    "stage_epochs": int(stage["epochs"]),
                    "trainable": list(stage["trainable"]),            "lr": float(stage["lr"]),
                            "lr_head": stage.get("lr_head"),
                            "weight_decay": float(stage["weight_decay"]),
                        }
                    )
    if not schedule:
        raise TrainingError("demo/config.json defines no training epochs.")
    return schedule


# ---------------------------------------------------------------------------
# Model / loader construction
# ---------------------------------------------------------------------------

def make_model(config: Dict[str, Any], *, pretrained: bool, logger=print) -> nn.Module:
    num_classes = int(config["model"]["num_classes"])
    if num_classes != len(config["taxonomy"]["classes"]):
        raise TrainingError("model.num_classes does not match the configured taxonomy.")
    try:
        return build_resnet50(
            num_classes=num_classes,
            pretrained=pretrained,
            allow_weight_download=True,
            logger=None,
        )
    except Exception as exc:
        raise TrainingError(
            "Could not build ResNet-50 with ImageNet weights.\n"
            f"Underlying error: {type(exc).__name__}: {exc}\n"
            "Fix: connect to the internet once so torchvision can download "
            "resnet50 IMAGENET1K_V1 weights (~98 MB), then re-run."
        ) from exc


def make_loader(
    rows: List[Dict[str, str]],
    repo_cfg,
    *,
    train: bool,
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    sampler: Optional[torch.utils.data.Sampler] = None,
) -> DataLoader:
    dataset = ClassifierDataset(rows, repo_cfg, train=train, image_size=image_size)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(train and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=None if sampler is not None else generator,
    )


def class_balanced_sampler(rows: List[Dict[str, str]], seed: int) -> WeightedRandomSampler:
    """Draw class-balanced minibatches from the fit rows only.

    With 41 images the plain shuffle order can leave rare grades very rare inside a
    batch, which pushes the model towards the majority class. The sampler corrects
    that without touching the loss or the validation/test data.
    """
    counts = Counter(int(row["mapped_label_int"]) for row in rows)
    weights = [1.0 / counts[int(row["mapped_label_int"])] for row in rows]
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(rows), replacement=True, generator=generator)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[float, float, List[int], List[int], List[List[float]]]:
    """One pass; trains when an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    if training:
        freeze_batchnorm(model)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    labels_out: List[int] = []
    preds_out: List[int] = []
    probs_out: List[List[float]] = []

    with torch.set_grad_enabled(training):
        for inputs, labels, _meta in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            batch = int(labels.numel())
            total_loss += float(loss.item()) * batch
            probabilities = torch.softmax(logits.detach(), dim=1).cpu().numpy()
            predictions = probabilities.argmax(axis=1)
            total_correct += int((predictions == labels.cpu().numpy()).sum())
            total_samples += batch
            labels_out.extend(int(x) for x in labels.cpu().tolist())
            preds_out.extend(int(x) for x in predictions.tolist())
            probs_out.extend(probabilities.tolist())

    return (
        total_loss / max(1, total_samples),
        total_correct / max(1, total_samples),
        labels_out,
        preds_out,
        probs_out,
    )


# ---------------------------------------------------------------------------
# Artefacts
# ---------------------------------------------------------------------------

def plot_training_curve(history: List[Dict[str, Any]], path: Path, best_epoch: Optional[int]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(row["global_epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    val_loss = [float(row["val_loss"]) for row in history]
    train_acc = [float(row["train_accuracy"]) for row in history]
    val_acc = [float(row["val_accuracy"]) for row in history]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].plot(epochs, train_loss, label="train loss", color="#1f77b4")
    axes[0].plot(epochs, val_loss, label="validation loss", color="#d62728")
    axes[0].set_title("Loss per epoch")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train accuracy", color="#1f77b4")
    axes[1].plot(epochs, val_acc, label="validation accuracy", color="#d62728")
    axes[1].set_title("Accuracy per epoch")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    stages = [row["stage"] for row in history]
    for index in range(1, len(stages)):
        if stages[index] != stages[index - 1]:
            for axis in axes:
                axis.axvline(epochs[index] - 0.5, color="#888888", linestyle="--", linewidth=1)
    if best_epoch is not None:
        for axis in axes:
            axis.scatter([best_epoch], [axis.get_ylim()[0]], marker="^", color="#2ca02c", zorder=5, s=40)

    figure.suptitle("ResNet-50 demo training (50 training images, validation carved from them)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def archive_previous_results() -> Optional[Path]:
    """Move previous *training* artefacts aside instead of overwriting them."""
    targets = [
        RESULTS_DIR / "training_history.csv",
        RESULTS_DIR / "training_curve.png",
        RESULTS_DIR / "training_config.json",
        RESULTS_DIR / "model_metadata.json",
    ]
    existing = [p for p in targets if p.exists()]
    if not existing:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = RESULTS_DIR / f"previous_run_{stamp}"
    target.mkdir(parents=True, exist_ok=True)
    for path in existing:
        shutil.move(str(path), str(target / path.name))
    return target


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(
    config: Dict[str, Any],
    *,
    resume: bool = False,
    overwrite: bool = False,
    epochs_override: Optional[int] = None,
) -> Dict[str, Any]:
    seed = int(config["experiment"]["seed"])
    image_size = int(config["model"]["image_size"])
    classes = list(config["taxonomy"]["classes"])
    num_classes = len(classes)
    training_cfg = config["training"]
    batch_size = int(training_cfg["batch_size"])
    num_workers = int(training_cfg["num_workers"])
    patience = int(training_cfg["early_stopping_patience"])

    # ---- data (training manifest only) -----------------------------------
    train_rows = load_manifest("selected_train.csv", config)
    if len(train_rows) != int(config["experiment"]["expected_train_total"]):
        raise TrainingError(
            f"Expected exactly {config['experiment']['expected_train_total']} training rows, "
            f"found {len(train_rows)}."
        )
    if manifest_path(config, "selected_test.csv").is_file():
        import csv

        with manifest_path(config, "selected_test.csv").open(newline="", encoding="utf-8") as handle:
            test_rows = list(csv.DictReader(handle))
        overlap = {r["rel_path"] for r in train_rows} & {r["rel_path"] for r in test_rows}
        if overlap:
            raise TrainingError(f"Training and test manifests overlap on {len(overlap)} file(s); refusing to train.")

    fit_rows = [r for r in train_rows if r.get("used_for") == "fit"]
    val_rows = [r for r in train_rows if r.get("used_for") == "validation"]
    if not fit_rows or not val_rows:
        raise TrainingError(
            "The training manifest must contain both 'fit' and 'validation' rows "
            f"(found {len(fit_rows)} fit, {len(val_rows)} validation)."
        )
    labels_seen = sorted({int(r["mapped_label_int"]) for r in train_rows})
    if labels_seen != list(range(num_classes)):
        raise TrainingError(f"Training manifest covers classes {labels_seen}; expected 0..{num_classes - 1}.")

    # ---- repository config (preprocessing + augmentation) ----------------
    repo_cfg = load_config()
    if int(repo_cfg.seed) != seed:
        print(
            f"WARNING: repository seed ({repo_cfg.seed}) differs from the demo seed ({seed}); "
            "augmentation randomness follows the repository seed."
        )
    if int(repo_cfg.require("preprocessing.classification.image_size")) != image_size:
        print(
            "WARNING: repository image size differs from demo/config.json; "
            f"using {image_size} for training and evaluation."
        )

    device = detect_device()
    seed_everything(seed)

    sampler = None
    if bool(training_cfg.get("class_balanced_sampling", False)):
        sampler = class_balanced_sampler(fit_rows, seed)
        print("Class-balanced sampling: ON (fit rows only)")
    fit_loader = make_loader(
        fit_rows, repo_cfg, train=True, image_size=image_size,
        batch_size=batch_size, num_workers=num_workers, seed=seed, sampler=sampler,
    )
    val_loader = make_loader(
        val_rows, repo_cfg, train=False, image_size=image_size,
        batch_size=batch_size, num_workers=0, seed=seed,
    )

    class_weights = class_weights_from_rows(fit_rows, num_classes)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # ---- model ------------------------------------------------------------
    model = make_model(config, pretrained=bool(config["model"]["pretrained"]))
    model.to(device)

    schedule = build_schedule(config)
    if epochs_override is not None:
        schedule = schedule[: int(epochs_override)]
        if not schedule:
            raise TrainingError("--epochs must be at least 1.")

    config_snapshot = {
        "experiment": config["experiment"]["name"],
        "seed": seed,
        "image_size": image_size,
        "num_classes": num_classes,
        "classes": classes,
        "label_mapping": config["label_mapping"],
        "sampling": config["sampling"],
        "training": training_cfg,
        "augmentation": config["augmentation"],
        "model": config["model"],
    }
    fingerprint = rows_digest(train_rows)

    history: List[Dict[str, Any]] = []
    best = {
        "val_macro_f1": -1.0,
        "val_loss": float("inf"),
        "global_epoch": -1,
        "stage": "",
        "val_accuracy": 0.0,
    }
    start_epoch = 0
    epochs_without_improvement = 0
    optimizer: Optional[torch.optim.Optimizer] = None
    scheduler = None
    current_stage_index = -1

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if resume:
        if not LAST_CHECKPOINT.is_file():
            raise TrainingError(
                f"--resume was requested but no checkpoint exists at {LAST_CHECKPOINT}.\n"
                "Run without --resume to start a fresh run."
            )
        payload = torch.load(LAST_CHECKPOINT, map_location=device, weights_only=False)
        if payload.get("rows_digest") != fingerprint:
            raise TrainingError(
                "The existing checkpoint was trained on a different set of training images. "
                "Refusing to resume. Delete demo/model/ and re-run without --resume."
            )
        model.load_state_dict(payload["model_state"])
        history = list(payload.get("history", []))
        best = dict(payload.get("best", best))
        start_epoch = int(payload.get("next_epoch", 0)) + 1
        epochs_without_improvement = int(payload.get("epochs_without_improvement", 0))
        print(f"Resuming from epoch {start_epoch + 1} of {len(schedule)}.")
    elif not overwrite:
        archived = archive_previous_results()
        if archived is not None:
            print(f"Previous training artefacts archived to {archived.relative_to(REPO_ROOT).as_posix()}")
    if INCOMPLETE_MARKER.exists():
        INCOMPLETE_MARKER.unlink()

    training_started = time.time()
    print()
    print("Training configuration")
    print(f"  device                 : {device}")
    print(f"  seed                   : {seed}")
    print(f"  fit / validation images: {len(fit_rows)} / {len(val_rows)} (test images: {int(config['experiment']['expected_test_total'])}, unused here)")
    print(f"  image size             : {image_size}")
    print(f"  batch size             : {batch_size}")
    print(f"  class weights          : {[round(float(w), 3) for w in class_weights]}")
    print(f"  class weight counts    : {dict(Counter(int(r['mapped_label_int']) for r in fit_rows))}")
    print(f"  total epochs           : {len(schedule)}")

    try:
        for epoch_index, plan in enumerate(schedule):
            if epoch_index < start_epoch:
                continue

            if plan["stage_index"] != current_stage_index:
                current_stage_index = plan["stage_index"]
                trainable_names = set_trainable(model, plan["trainable"])
                head_params = [
                    p for n, p in model.named_parameters() if p.requires_grad and n.startswith("fc.")
                ]
                body_params = [
                    p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("fc.")
                ]
                if plan.get("lr_head") and head_params and body_params:
                    parameter_groups = [
                        {"params": head_params, "lr": float(plan["lr_head"])},
                        {"params": body_params, "lr": plan["lr"]},
                    ]
                else:
                    parameter_groups = [{"params": head_params + body_params, "lr": plan["lr"]}]
                optimizer = torch.optim.AdamW(
                    parameter_groups, lr=plan["lr"], weight_decay=plan["weight_decay"]
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, plan["stage_epochs"]), eta_min=plan["lr"] * 0.05
                )
                if resume and epoch_index == start_epoch and plan["epoch_in_stage"] > 0:
                    optimizer.load_state_dict(payload["optimizer_state"])
                    for _ in range(plan["epoch_in_stage"]):
                        scheduler.step()
                print()
                print(
                    f"Stage {plan['stage_index'] + 1}/{len(training_cfg['stages'])}: {plan['stage_name']} "
                    f"({len(trainable_names)} tensors, lr={plan['lr']}"
                    + (f", head lr={plan['lr_head']}" if plan.get("lr_head") else "")
                    + f", wd={plan['weight_decay']})"
                )
                if resume and epoch_index == start_epoch:
                    pass  # verbose resume already reported above
                else:
                    print(f"  fine-tuning: {sorted({n.split('.')[0] for n in trainable_names})}")

            epoch_started = time.time()
            current_lr = float(optimizer.param_groups[0]["lr"])
            train_loss, train_acc, _, _, _ = run_epoch(model, fit_loader, criterion, device, optimizer)
            val_loss, val_acc, val_true, val_pred, val_probs = run_epoch(model, val_loader, criterion, device)
            if scheduler is not None:
                scheduler.step()

            val_metrics = classification_metrics(
                np.array(val_true),
                np.array(val_pred),
                probabilities=np.array(val_probs),
                class_names=classes,
                compute_auc=False,
            )
            val_macro_f1 = float(val_metrics["macro_f1"])

            improved = (val_macro_f1 > best["val_macro_f1"] + 1e-9) or (
                abs(val_macro_f1 - best["val_macro_f1"]) <= 1e-9 and val_loss < best["val_loss"] - 1e-9
            )
            if improved:
                best = {
                    "val_macro_f1": val_macro_f1,
                    "val_loss": float(val_loss),
                    "global_epoch": epoch_index,
                    "stage": plan["stage_name"],
                    "val_accuracy": float(val_acc),
                }
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "classes": classes,
                        "num_classes": num_classes,
                        "seed": seed,
                        "image_size": image_size,
                        "architecture": "resnet50",
                        "pretrained": bool(config["model"]["pretrained"]),
                        "rows_digest": fingerprint,
                        "best": best,
                        "config_snapshot": config_snapshot,
                        "selection_note": "selected on validation images carved out of the 50 training images; test images were never loaded",
                        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    },
                    BEST_CHECKPOINT,
                )
            else:
                epochs_without_improvement += 1

            record = {
                "stage": plan["stage_name"],
                "global_epoch": epoch_index,
                "stage_epoch": plan["epoch_in_stage"],
                "lr": current_lr,
                "train_loss": round(train_loss, 6),
                "train_accuracy": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_accuracy": round(val_acc, 6),
                "val_macro_f1": round(val_macro_f1, 6),
                "epoch_seconds": round(time.time() - epoch_started, 2),
            }
            history.append(record)
            write_history_csv(RESULTS_DIR / "training_history.csv", history)

            marker = " *" if improved else ""
            print(
                f"  epoch {epoch_index + 1:02d}/{len(schedule)} [{plan['stage_name']}] "
                f"lr={current_lr:.2e} train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_macro_f1={val_macro_f1:.3f}{marker}"
            )

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
                    "next_epoch": epoch_index,
                    "history": history,
                    "best": best,
                    "epochs_without_improvement": epochs_without_improvement,
                    "rows_digest": fingerprint,
                    "classes": classes,
                    "seed": seed,
                    "image_size": image_size,
                },
                LAST_CHECKPOINT,
            )

            if epochs_without_improvement >= patience:
                print(
                    f"  early stopping: no validation improvement in {patience} epoch(s)."
                )
                break

        if not BEST_CHECKPOINT.is_file():
            raise TrainingError(
                "Training finished without producing a best checkpoint. This usually means "
                "every epoch failed to evaluate; check the log above."
            )

        elapsed = time.time() - training_started
        plot_training_curve(history, RESULTS_DIR / "training_curve.png", best["global_epoch"])

        meta = {
            "experiment": config["experiment"]["name"],
            "status": "completed",
            "architecture": "resnet50",
            "model_builder": "src.models.build_resnet50",
            "pretrained_weights": config["model"]["pretrained"],
            "num_classes": num_classes,
            "class_index_to_name": {str(i): name for i, name in enumerate(classes)},
            "label_mapping": config["label_mapping"],
            "seed": seed,
            "image_size": image_size,
            "preprocessing": config["model"]["preprocessing"],
            "augmentation": config["augmentation"],
            "device": str(device),
            "torch_version": torch.__version__,
            "torchvision_version": _torchvision_version(),
            "python_version": platform.python_version(),
            "fit_images": [r["rel_path"] for r in fit_rows],
            "validation_images": [r["rel_path"] for r in val_rows],
            "fit_class_counts": {str(k): v for k, v in sorted(Counter(int(r["mapped_label_int"]) for r in fit_rows).items())},
            "validation_class_counts": {str(k): v for k, v in sorted(Counter(int(r["mapped_label_int"]) for r in val_rows).items())},
            "training_manifest": manifest_path(config, "selected_train.csv").relative_to(REPO_ROOT).as_posix(),
            "test_manifest": manifest_path(config, "selected_test.csv").relative_to(REPO_ROOT).as_posix(),
            "test_images_used_in_training": 0,
            "selected_files_digest": fingerprint,
            "best_checkpoint": BEST_CHECKPOINT.relative_to(REPO_ROOT).as_posix(),
            "best_epoch": int(best["global_epoch"]) + 1,
            "best_stage": best["stage"],
            "best_val_macro_f1": round(float(best["val_macro_f1"]), 6),
            "best_val_loss": round(float(best["val_loss"]), 6),
            "best_val_accuracy": round(float(best["val_accuracy"]), 6),
            "epochs_completed": len(history),
            "training_seconds": round(elapsed, 1),
            "early_stopped": len(history) < len(schedule),
            "class_weights": {str(i): round(float(w), 6) for i, w in enumerate(class_weights)},
            "class_balanced_sampling": bool(training_cfg.get("class_balanced_sampling", False)),
            "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (RESULTS_DIR / "model_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        training_config = {
            "experiment": config["experiment"],
            "model": config["model"],
            "training": config["training"],
            "sampling": config["sampling"],
            "taxonomy": config["taxonomy"],
            "label_mapping": config["label_mapping"],
            "augmentation": config["augmentation"],
            "resolved": {
                "device": str(device),
                "torch_version": torch.__version__,
                "torchvision_version": _torchvision_version(),
                "python_version": platform.python_version(),
                "batch_size": batch_size,
                "num_workers": num_workers,
                "fit_images": len(fit_rows),
                "validation_images": len(val_rows),
                "epochs_planned": len(build_schedule(config)),
                "epochs_run": len(history),
                "class_weights": [round(float(w), 6) for w in class_weights],
                "class_balanced_sampling": bool(training_cfg.get("class_balanced_sampling", False)),
                "optimizer": "AdamW",
                "loss": "CrossEntropyLoss(weight=class_weights)",
                "scheduler": "CosineAnnealingLR per stage",
                "batchnorm": training_cfg["batchnorm"],
            },
        }
        (RESULTS_DIR / "training_config.json").write_text(json.dumps(training_config, indent=2), encoding="utf-8")

        print()
        print(f"Best epoch {meta['best_epoch']} ({meta['best_stage']}): "
              f"val_macro_f1={meta['best_val_macro_f1']:.3f} val_acc={meta['best_val_accuracy']:.3f}")
        print(f"Training took {elapsed / 60:.1f} min on {device}")
        print(f"Checkpoint: {BEST_CHECKPOINT.relative_to(REPO_ROOT).as_posix()}")
        print()
        print("Next step: python demo/evaluate.py")
        return meta

    except BaseException as exc:  # KeyboardInterrupt included: never fail silently
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        marker = (
            "TRAINING DID NOT COMPLETE\n"
            "========================\n"
            f"When       : {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
            f"Error type : {type(exc).__name__}\n"
            f"Message    : {exc}\n\n"
            "What is still valid:\n"
            f"  - demo/results/training_history.csv ({len(history)} epoch(s) recorded)\n"
            f"  - demo/model/last_model.pt (resumable)\n\n"
            "How to continue:\n"
            "  python demo/train.py --resume\n"
            "If the error is about a missing checkpoint or mismatched images, delete the\n"
            "demo/model folder and run `python demo/train.py` from scratch.\n"
        )
        INCOMPLETE_MARKER.write_text(marker, encoding="utf-8")
        print()
        print(f"Training stopped: {type(exc).__name__}: {exc}")
        print(f"A marker was written to {INCOMPLETE_MARKER.relative_to(REPO_ROOT).as_posix()}")
        raise


def _torchvision_version() -> str:
    try:
        import torchvision

        return str(torchvision.__version__)
    except Exception:  # pragma: no cover
        return "unavailable"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEMO_DIR / "config.json"))
    parser.add_argument("--resume", action="store_true", help="Continue from demo/model/last_model.pt.")
    parser.add_argument("--overwrite", action="store_true", help="Keep previous artefacts in place.")
    parser.add_argument("--epochs", type=int, default=None, help="Optional cap on total epochs (debugging).")
    args = parser.parse_args(argv)

    config = load_demo_config(args.config)
    if not (extract_root(config)).is_dir():
        raise SystemExit(
            "The demo dataset has not been prepared yet.\n"
            "Run this first:\n  python demo/prepare_data.py"
        )
    try:
        train(config, resume=args.resume, overwrite=args.overwrite, epochs_override=args.epochs)
    except TrainingError as exc:
        print()
        print("TRAINING FAILED")
        print(f"  {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
