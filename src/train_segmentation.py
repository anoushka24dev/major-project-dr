"""Four-channel lesion U-Net training and prediction on IDRiD.

The model produces one independent probability map per lesion type, in the fixed
channel order MA, HE, EX, SE. Each channel is a separate binary problem because
lesions co-occur.

What this module will and will not do
-------------------------------------
* It refuses to run when IDRiD images and masks are not present and correctly
  paired — it never trains on mismatched annotations, and never reports a
  segmentation result that did not happen.
* Per-lesion decision thresholds are tuned on the **validation** partition only
  and then frozen for the test partition.
* Nothing here is a clinical measurement of lesion burden.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
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
from .data import LESION_ORDER, SegmentationDataset, build_idrid_pairs, split_idrid_pairs, write_history_csv
from .hardware import detect_hardware, resolve_device
from .metrics import dice_per_channel, iou_per_channel, segmentation_metrics


class SegmentationError(RuntimeError):
    """Raised when segmentation cannot proceed honestly."""


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(dataset, *, batch_size: int, shuffle: bool, num_workers: int, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def predict_dataset(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp_enabled: bool = False,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Collect probabilities and targets for a whole dataset."""
    model.eval()
    ids: List[str] = []
    probabilities: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    for images, masks, meta in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled and device.type == "cuda"):
            logits = model(images)
        probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
        targets.append(masks.numpy())
        batch_ids = meta.get("image_id")
        if batch_ids is None:
            batch_ids = [f"sample_{len(ids) + offset}" for offset in range(images.size(0))]
        elif isinstance(batch_ids, str):
            batch_ids = [batch_ids]
        ids.extend(list(batch_ids))
    if not probabilities:
        return ids, np.zeros((0, 4, 1, 1)), np.zeros((0, 4, 1, 1))
    return ids, np.concatenate(probabilities, axis=0), np.concatenate(targets, axis=0)


def tune_thresholds(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    start: float,
    stop: float,
    step: float,
) -> Dict[str, Any]:
    """Choose a per-lesion threshold that maximises validation Dice.

    Only validation data may be passed to this function.
    """
    grid = np.arange(float(start), float(stop) + 1e-9, float(step))
    chosen: List[float] = []
    detail: Dict[str, Any] = {}
    for channel_index, channel in enumerate(LESION_ORDER):
        best_threshold = 0.5
        best_dice = -1.0
        scores: List[Dict[str, float]] = []
        for threshold in grid:
            dice = float(
                dice_per_channel(
                    probabilities, targets, thresholds=[threshold] * 4
                )[channel_index]
            )
            scores.append({"threshold": float(threshold), "dice": dice})
            if dice > best_dice:
                best_dice = dice
                best_threshold = float(threshold)
        chosen.append(best_threshold)
        detail[channel] = {
            "threshold": best_threshold,
            "val_dice": best_dice,
            "curve": scores,
        }
    return {"thresholds": chosen, "per_channel": detail}


def run_training(
    cfg: Config,
    logger,
    events=None,
    *,
    force_retrain: bool = False,
    resume: bool = True,
) -> Dict[str, Any]:
    """Train (or reuse) the four-channel lesion U-Net."""
    from .models import DiceBCELoss, build_unet4, unfreeze_all  # noqa: F401

    pairing = build_idrid_pairs(cfg, logger)
    if not pairing.get("available"):
        message = (
            "IDRiD segmentation data is unavailable, so no U-Net was trained: "
            f"{pairing.get('reason', 'unknown reason')}. "
            "This is recorded as 'blocked', not as a result."
        )
        logger.warning(message)
        summary = {
            "status": "blocked",
            "reason": pairing.get("reason"),
            "detail": message,
            "idrid": {k: v for k, v in pairing.items() if k != "pairs"},
            "checkpoint": None,
        }
        _atomic_write_json(summary, cfg.outputs_dir("segmentation") / "segmentation_summary.json")
        if events is not None:
            events.write("segmentation_blocked", reason=pairing.get("reason"))
        return summary

    pairs = pairing["pairs"]
    channel_names = list(cfg.lesion_channels)
    image_size = int(cfg.require("preprocessing.segmentation.image_size"))
    fingerprint = build_fingerprint(
        architecture=str(cfg.require("segmentation.architecture")),
        classes=channel_names,
        image_size=image_size,
        manifest_version=cfg.manifest_version,
        extra={
            "task": "four_channel_lesion_segmentation",
            "lesion_order": list(LESION_ORDER),
            "idrid_paired_images": len(pairs),
        },
    )

    run_dir = cfg.path("paths.runs_dir", create=True) / "segmentation"
    best_dir = run_dir / BEST_DIR
    last_dir = run_dir / LAST_DIR

    if cfg.get("run.reuse_valid_checkpoints", True) and not force_retrain:
        reusable = find_reusable(best_dir, fingerprint)
        if reusable["compatible"]:
            logger.info("Reusing compatible segmentation checkpoint at %s", reusable["path"])
            # Reuse must still publish the artefact downstream stages expect, and keep
            # the thresholds the checkpoint was tuned with.
            published_path = cfg.outputs_dir("segmentation") / "best_lesion_unet.pt"
            if not published_path.is_file():
                import shutil

                shutil.copyfile(best_dir / CHECKPOINT_FILENAME, published_path)
                logger.info("published reused segmentation checkpoint to %s", published_path)
            existing_summary = cfg.outputs_dir("segmentation") / "segmentation_summary.json"
            thresholds: Dict[str, Any] = {}
            if existing_summary.is_file():
                try:
                    thresholds = json.loads(
                        existing_summary.read_text(encoding="utf-8")
                    ).get("thresholds", {})
                except Exception:
                    thresholds = {}
            summary = {
                "status": "reused",
                "checkpoint": str(published_path),
                "epoch": reusable.get("epoch"),
                "reused": True,
                "thresholds": thresholds,
            }
            if events is not None:
                events.write("segmentation_training_reused", **summary)
            return summary

    split_ratios = cfg.section("split")["ratios"]
    split = split_idrid_pairs(pairs, split_ratios, cfg.seed)
    logger.info(
        "IDRiD split: train=%d validation=%d test=%d",
        len(split["train"]), len(split["validation"]), len(split["test"]),
    )

    device = resolve_device("auto")
    hardware = detect_hardware(
        mixed_precision_preference=cfg.mixed_precision_setting("segmentation"),
        disk_probe=cfg.root,
    )
    amp_enabled = bool(hardware.mixed_precision)
    batch_size = cfg.effective_batch_size(
        "segmentation", fallback=hardware.recommended_segmentation_batch_size
    )
    num_workers = cfg.effective_num_workers("segmentation") or hardware.recommended_num_workers
    epochs = int(cfg.require("segmentation.epochs"))
    lr = float(cfg.require("segmentation.learning_rate"))
    weight_decay = float(cfg.require("segmentation.weight_decay"))
    patience = int(cfg.require("segmentation.early_stopping_patience"))
    seed = cfg.seed

    _seed_everything(seed)
    train_ds = SegmentationDataset(split["train"], cfg, image_size=image_size)
    val_ds = SegmentationDataset(split["validation"], cfg, image_size=image_size)
    test_ds = SegmentationDataset(split["test"], cfg, image_size=image_size) if split["test"] else None
    train_loader = _loader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, seed=seed)
    val_loader = _loader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, seed=seed)
    test_loader = (
        _loader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, seed=seed)
        if test_ds is not None else None
    )

    model = build_unet4(
        in_channels=int(cfg.require("segmentation.in_channels")),
        out_channels=int(cfg.require("segmentation.out_channels")),
        base_filters=int(cfg.require("segmentation.base_filters")),
        logger=logger,
    ).to(device)

    loss_cfg = cfg.get("segmentation.loss", {}) or {}
    loss_fn = DiceBCELoss(
        bce_weight=float(loss_cfg.get("bce_weight", 0.5)),
        dice_weight=float(loss_cfg.get("dice_weight", 0.5)),
    )
    optimizer_name = str(cfg.require("segmentation.optimizer")).lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise SegmentationError(f"Unsupported optimizer {optimizer_name!r}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    scaler = torch.amp.GradScaler("cuda") if amp_enabled and device.type == "cuda" else None

    history: List[Dict[str, Any]] = []
    best_dice = -1.0
    best_loss = float("inf")
    best_epoch = -1
    start_epoch = 0

    if resume and last_dir.exists():
        reusable = find_reusable(last_dir, fingerprint)
        if reusable["compatible"]:
            payload = load_checkpoint(last_dir / CHECKPOINT_FILENAME, map_location=device)
            restored = restore_into(model, payload, optimizer=optimizer, scheduler=scheduler)
            state = load_training_state(last_dir) or {}
            history = list(state.get("history") or [])
            best_dice = float(state.get("best_metric", -1.0))
            best_epoch = int(state.get("best_epoch", -1))
            start_epoch = restored["epoch"] + 1
            logger.info("Resuming segmentation from epoch %d", start_epoch)

    if run_dir.exists() and (start_epoch == 0 or force_retrain):
        archived = archive_previous_run(run_dir)
        if archived:
            logger.info("Archived the previous segmentation run to %s", archived)

    run_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = {
        "seed": seed,
        "image_size": image_size,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "epochs": epochs,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "optimizer": optimizer_name,
        "loss": loss_cfg,
        "device": str(device),
        "mixed_precision": amp_enabled,
        "lesion_order": list(LESION_ORDER),
        "fingerprint": fingerprint,
        "train_pairs": len(split["train"]),
        "validation_pairs": len(split["validation"]),
        "test_pairs": len(split["test"]),
        "idrid_root": pairing.get("root"),
    }
    _atomic_write_json(config_snapshot, run_dir / "run_meta.json")

    started = time.time()
    gradient_clip = 1.0
    for epoch in range(start_epoch, epochs):
        epoch_started = time.time()
        model.train()
        total_loss = 0.0
        total_samples = 0
        for images, masks, _meta in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled and device.type == "cuda"):
                logits = model(images)
                loss = loss_fn(logits, masks)
            if scaler is not None and amp_enabled and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
            total_loss += float(loss.item()) * images.size(0)
            total_samples += images.size(0)
        scheduler.step()

        val_ids, val_probs, val_targets = predict_dataset(model, val_loader, device, amp_enabled=amp_enabled)
        dice_scores = dice_per_channel(val_probs, val_targets, thresholds=0.5)
        iou_scores = iou_per_channel(val_probs, val_targets, thresholds=0.5)
        current_loss = total_loss / total_samples if total_samples else float("inf")
        improved = (mean_dice > best_dice) if (mean_dice > 0.0 or best_dice > 0.0) else (current_loss < best_loss)
        if improved:
            best_dice = max(best_dice, mean_dice)
            best_loss = min(best_loss, current_loss)
            best_epoch = epoch

        row = {
            "epoch": epoch,
            "train_loss": current_loss,
            "val_macro_dice@0.5": mean_dice,
            "val_macro_iou@0.5": float(np.mean(iou_scores)),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - epoch_started, 2),
            "improved": improved,
        }
        for index, channel in enumerate(LESION_ORDER):
            row[f"val_dice_{channel}@0.5"] = float(dice_scores[index])
            row[f"val_iou_{channel}@0.5"] = float(iou_scores[index])
        history.append(row)
        logger.info(
            "epoch %d/%d | loss %.4f | val macro Dice %.4f | %.1fs%s",
            epoch + 1, epochs, row["train_loss"], mean_dice, row["seconds"],
            " *" if improved else "",
        )

        save_checkpoint(
            last_dir, model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch,
            history=history, best_metric=best_dice, best_epoch=best_epoch,
            fingerprint=fingerprint, extra={"config": config_snapshot},
        )
        if improved:
            save_checkpoint(
                best_dir, model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch,
                history=history, best_metric=best_dice, best_epoch=best_epoch,
                fingerprint=fingerprint, extra={"config": config_snapshot},
            )
        if epoch - best_epoch >= patience:
            logger.info("Early stopping: validation Dice did not improve for %d epochs", patience)
            break

    write_history_csv(cfg.outputs_dir("segmentation") / "training_history.csv", history)

    # ---- threshold tuning on validation only --------------------------
    thresholds = [0.5] * len(LESION_ORDER)
    threshold_report: Dict[str, Any] = {"method": "fixed 0.5 (tuning disabled)"}
    search_cfg = cfg.get("segmentation.threshold_search", {}) or {}
    best_path = best_dir / CHECKPOINT_FILENAME
    if search_cfg.get("enabled", True) and best_path.is_file():
        payload = load_checkpoint(best_path, map_location=device)
        model.load_state_dict(payload["model_state"])
        _val_ids, val_probs, val_targets = predict_dataset(
            model, val_loader, device, amp_enabled=amp_enabled
        )
        tuned = tune_thresholds(
            val_probs, val_targets,
            start=float(search_cfg.get("start", 0.1)),
            stop=float(search_cfg.get("stop", 0.9)),
            step=float(search_cfg.get("step", 0.05)),
        )
        thresholds = tuned["thresholds"]
        threshold_report = {
            "method": "per-lesion threshold maximising validation Dice",
            "selected_on": "validation",
            "thresholds": dict(zip(LESION_ORDER, thresholds)),
            "per_channel": tuned["per_channel"],
        }

    # ---- test evaluation ---------------------------------------------
    test_metrics: Dict[str, Any] = {"available": False}
    if test_loader is not None and best_path.is_file():
        payload = load_checkpoint(best_path, map_location=device)
        model.load_state_dict(payload["model_state"])
        ids, probs, targets = predict_dataset(model, test_loader, device, amp_enabled=amp_enabled)
        test_metrics = segmentation_metrics(probs, targets, LESION_ORDER, thresholds=thresholds)
        test_metrics.update(
            {"available": True, "num_images": int(len(ids)), "selected_by": "validation Dice"}
        )
        # Prediction tensors and overlays are written by the dedicated
        # evaluate_segmentation stage so there is exactly one producer.
    else:
        test_metrics["reason"] = "no held-out IDRiD pairs or no checkpoint"

    published = cfg.outputs_dir("segmentation") / "best_lesion_unet.pt"
    if best_path.is_file():
        import shutil

        shutil.copyfile(best_path, published)

    summary = {
        "status": "completed",
        "checkpoint": str(published) if best_path.is_file() else None,
        "best_epoch": best_epoch,
        "best_val_macro_dice": None if best_dice < 0 else float(best_dice),
        "epochs_completed": len(history),
        "seconds": round(time.time() - started, 2),
        "device": str(device),
        "mixed_precision": amp_enabled,
        "thresholds": dict(zip(LESION_ORDER, thresholds)),
        "threshold_selection": threshold_report,
        "test_metrics": test_metrics,
        "fingerprint": fingerprint,
        "idrid": {k: v for k, v in pairing.items() if k != "pairs"},
    }
    # `segmentation_metrics.json` is owned by the evaluate_segmentation stage, which
    # recomputes the same test metrics and additionally writes prediction overlays.
    _atomic_write_json(summary, cfg.outputs_dir("segmentation") / "segmentation_summary.json")
    if events is not None:
        events.write("segmentation_training_done", **{
            "best_epoch": best_epoch,
            "best_val_macro_dice": summary["best_val_macro_dice"],
            "test_macro_dice": (test_metrics or {}).get("macro_dice"),
        })
    return summary
