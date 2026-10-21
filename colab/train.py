"""
Minimal, guarded frozen ResNet-50 training pipeline for the provisional primary
five-band diabetic-retinopathy task.

This module is deliberately small and defensive. It does NOT:
- modify the existing dataset, split CSV, split report, or audit artifacts;
- regenerate, reshuffle, or alter the existing train/validation/test assignment;
- use center crops or random crops at any stage;
- train on, tune on, early-stop on, or select checkpoints using the test partition;
- download pretrained weights or install packages without an explicit opt-in;
- silently overwrite a previous run, or present an incomplete run as completed;
- claim clinical validity, GPU execution, or Colab compatibility.

It DOES:
- load the existing split CSV unchanged;
- use deterministic, aspect-ratio-preserving resize-and-pad preprocessing;
- apply ImageNet normalization;
- compute class weights from training labels only;
- train only on the training partition and evaluate on validation each epoch;
- select the best checkpoint by validation loss and support early stopping;
- keep the test partition for a separate, explicitly invoked final evaluation;
- write configuration, per-epoch history, metrics, and checkpoints to a
  user-configurable persistent output directory.

Provisional status
------------------
The five-band primary label mapping is provisional (see
``audit_output/pretraining_audit/LABEL_POLICY_DRAFT.md``). Its two merges
(Very Severe NPDR -> Severe NPDR, Advanced PDR -> PDR) are not specialist-validated.
Nothing produced by this module is a clinical result.

Interruption behaviour
----------------------
Resume is NOT implemented. If a run is interrupted, the output directory is
marked with ``TRAINING_INCOMPLETE.txt`` and its manifest is set to
``"incomplete"``; that run must be restarted from scratch. Per-epoch checkpoints
and optimizer state are still written, so a future resume implementation has the
state it needs, but this version does not claim to resume.

Overwrite behaviour
-------------------
A run refuses to start when the output directory already holds a run. With an
explicit ``overwrite=True`` the previous run is discarded only *after* the cheap
validation steps (dataset construction, class weights, model construction) have
succeeded, so a bad dataset or an unavailable torchvision can no longer destroy a
previous run.

Device handling
---------------
The weighted loss keeps its class weights in a registered buffer. It is moved to
the training device explicitly, because a CPU weight against CUDA logits raises a
device-mismatch RuntimeError on the very first training step.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from colab_helpers import (
    BAND_ORDER,
    assert_head_is_trainable,
    build_frozen_resnet50,
    default_normalize,
    filter_by_partition,
    load_rgb,
    load_split_csv,
    preprocessing_inference,
    pretrained_weights_available,
    weights_cache_path,
)

IMAGENET_MEAN, IMAGENET_STD = default_normalize()

RUN_META_NAME = "run_meta.json"
RUN_MANIFEST_NAME = "run_manifest.json"
RUN_SUMMARY_NAME = "run_summary.json"
INCOMPLETE_MARKER_NAME = "TRAINING_INCOMPLETE.txt"
BEST_CHECKPOINT_DIR_NAME = "best_checkpoint"
TEST_EVAL_DIR_NAME = "final_test_evaluation"

PROVISIONAL_NOTE = (
    "Provisional five-band mapping. Not a clinical result. "
    "Severe NPDR includes Very Severe NPDR; PDR includes Advanced PDR."
)


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def _to_tensor(pil_image) -> torch.Tensor:
    """Convert an RGB PIL image to a float32 CHW tensor in [0, 1]."""
    arr = np.asarray(pil_image, dtype=np.float32) / 255.0
    if arr.ndim == 2:  # defensive: single-channel image
        arr = np.stack([arr] * 3, axis=-1)
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


def _normalize(
    tensor: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    """Apply per-channel normalization (out-of-place)."""
    m = torch.as_tensor(list(mean), dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    s = torch.as_tensor(list(std), dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return (tensor - m) / s


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PrimarySplitDataset(Dataset):
    """PyTorch dataset over one partition of the existing split CSV.

    Image preprocessing is deterministic, aspect-ratio-preserving resize-and-pad
    followed by ImageNet normalization. No random crops and no center crops are
    used. Decoding delegates to the existing ``dataset.decoding.load_rgb``.

    This dataset does not know or care which partition it was handed. Partition
    selection is the caller's responsibility; ``run_training`` only ever
    constructs it from train and validation rows.
    """

    def __init__(
        self,
        split_rows: Sequence[Dict[str, str]],
        dataset_root: Path,
        *,
        image_size: int = 224,
    ) -> None:
        if not split_rows:
            raise ValueError("split_rows must be non-empty")
        self._rows = list(split_rows)
        self._dataset_root = Path(dataset_root)
        self._image_size = image_size
        self._preprocess = preprocessing_inference(image_size)
        self._validate()

    def _validate(self) -> None:
        """Fail fast when rows are unusable, rather than mid-epoch."""
        for row in self._rows:
            path = self._dataset_root / row["image_path"]
            if not path.exists():
                raise FileNotFoundError(f"Image does not exist: {path}")
            if row.get("primary_class") not in BAND_ORDER:
                raise ValueError(
                    f"Unexpected primary_class {row.get('primary_class')!r} on row {row['image_path']!r}"
                )

    @property
    def image_size(self) -> int:
        return self._image_size

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, Dict[str, str]]:
        row = self._rows[index]
        path = self._dataset_root / row["image_path"]
        img = load_rgb(path)
        tensor = _to_tensor(self._preprocess(img))
        tensor = _normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        label = BAND_ORDER.index(row["primary_class"])
        return tensor, label, row


# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------

TrainingState = Dict[str, Any]


def initial_training_state(
    *,
    epoch: int = 0,
    best_val_loss: float = float("inf"),
    best_epoch: int = 0,
    best_state: Optional[dict] = None,
    patience_counter: int = 0,
    train_history: Optional[List[Dict[str, Any]]] = None,
    val_history: Optional[List[Dict[str, Any]]] = None,
) -> TrainingState:
    """Build a fresh training-state dict."""
    return {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_state": best_state,
        "patience_counter": patience_counter,
        "train_history": list(train_history or []),
        "val_history": list(val_history or []),
    }


def training_state_checksum(state: TrainingState) -> Dict[str, Any]:
    """Return a small integrity snapshot of a training state."""
    return {
        "epoch": state["epoch"],
        "best_val_loss": state["best_val_loss"],
        "best_epoch": state["best_epoch"],
        "patience_counter": state["patience_counter"],
        "train_history_len": len(state["train_history"]),
        "val_history_len": len(state["val_history"]),
        "has_best_state": state["best_state"] is not None,
    }


def load_training_state(path: Path) -> Dict[str, Any]:
    """Load a ``{"state": ..., "config": ...}`` metadata file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the metadata is missing the required ``state``/``config`` keys.
    """
    if not path.exists():
        raise FileNotFoundError(f"Training state file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "state" not in data or "config" not in data:
        raise ValueError("Checkpoint metadata missing required keys: state, config")
    return data


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
    """Compute accuracy, per-class metrics, macro aggregates, and confusion matrix.

    ``confusion_matrix`` is returned as a list of lists of Python ``int`` values so
    it serializes cleanly to JSON. ``per_class`` is keyed by the provisional band
    names in ``BAND_ORDER``.
    """
    preds = logits.argmax(dim=1)
    n = int(labels.size(0))
    correct = int((preds == labels).sum().item())
    accuracy = correct / n if n else 0.0

    k = len(BAND_ORDER)
    confusion = np.zeros((k, k), dtype=np.int64)
    labels_np = labels.detach().cpu().numpy()
    preds_np = preds.detach().cpu().numpy()
    for y_true, y_pred in zip(labels_np.ravel(), preds_np.ravel()):
        confusion[int(y_true), int(y_pred)] += 1

    per_class: Dict[str, Dict[str, Any]] = {}
    for i, name in enumerate(BAND_ORDER):
        tp = int(confusion[i, i])
        support = int(confusion[i].sum())
        pred_pos = int(confusion[:, i].sum())
        precision = tp / pred_pos if pred_pos else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted": pred_pos,
            "tp": tp,
        }

    macro_precision = float(np.mean([per_class[name]["precision"] for name in BAND_ORDER]))
    macro_recall = float(np.mean([per_class[name]["recall"] for name in BAND_ORDER]))
    macro_f1 = float(np.mean([per_class[name]["f1"] for name in BAND_ORDER]))

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "band_order": list(BAND_ORDER),
        "per_class": per_class,
        "confusion_matrix": [[int(v) for v in row] for row in confusion.tolist()],
        "num_samples": n,
        "provisional_note": PROVISIONAL_NOTE,
    }


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    """Run one no-grad pass and return loss plus full metrics."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_logits: Optional[torch.Tensor] = None
    all_labels: Optional[torch.Tensor] = None

    with torch.no_grad():
        for inputs, labels, _ in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += float(loss.item()) * inputs.size(0)
            total_samples += inputs.size(0)

            logits = outputs.detach().cpu()
            batch_labels = labels.detach().cpu()
            if all_logits is None:
                all_logits, all_labels = logits, batch_labels
            else:
                all_logits = torch.cat([all_logits, logits], dim=0)
                all_labels = torch.cat([all_labels, batch_labels], dim=0)

    metrics = compute_metrics(all_logits, all_labels) if all_logits is not None else {
        "accuracy": 0.0,
        "macro_precision": 0.0,
        "macro_recall": 0.0,
        "macro_f1": 0.0,
        "band_order": list(BAND_ORDER),
        "per_class": {},
        "confusion_matrix": [[0] * len(BAND_ORDER) for _ in BAND_ORDER],
        "num_samples": 0,
        "provisional_note": PROVISIONAL_NOTE,
    }
    metrics["loss"] = total_loss / total_samples if total_samples else float("inf")
    metrics["num_samples"] = total_samples
    return metrics


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, Any]:
    """Run one training pass over the supplied loader."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for inputs, labels, _ in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * inputs.size(0)
        total_samples += inputs.size(0)

    return {
        "loss": total_loss / total_samples if total_samples else float("inf"),
        "num_samples": total_samples,
    }


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def compute_class_weights_from_rows(
    split_rows: Sequence[Dict[str, str]],
    *,
    class_order: Sequence[str] = BAND_ORDER,
) -> Dict[str, float]:
    """Compute inverse-frequency class weights from the supplied rows only.

    ``weight[class] = N / (K * n_class)``

    This function is partition-agnostic by design; the guarantee that only
    training rows are passed in is enforced by :func:`run_training`, and the
    tests pin that behaviour at the call site.

    Raises
    ------
    ValueError
        If there are no rows, or if any class in ``class_order`` has no examples.
    """
    from collections import Counter

    counts = Counter(
        row["primary_class"] for row in split_rows if row.get("primary_class") in class_order
    )
    n = sum(counts.values())
    k = len(class_order)
    if n == 0:
        raise ValueError("No labels available to compute class weights")
    weights: Dict[str, float] = {}
    for name in class_order:
        c = counts.get(name, 0)
        if c == 0:
            raise ValueError(
                f"Training partition has no examples for class {name!r}; "
                "cannot compute class weights"
            )
        weights[name] = n / (k * c)
    return weights


def make_weighted_loss_fn(
    class_weights: Mapping[str, float],
    *,
    device: Optional[torch.device] = None,
) -> nn.CrossEntropyLoss:
    """Build a weighted CrossEntropyLoss ordered by the provisional band order.

    ``nn.CrossEntropyLoss`` keeps ``weight`` in a registered buffer, so it can be
    moved with ``nn.Module.to``. When ``device`` is given the weight is placed on
    that device immediately.

    This matters: a CPU weight applied to CUDA logits raises
    ``RuntimeError: Expected all tensors to be on the same device`` on the first
    training step, and moving only the model does not fix it.
    """
    if not class_weights:
        loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    else:
        missing = [name for name in BAND_ORDER if name not in class_weights]
        if missing:
            raise ValueError(f"class_weights is missing entries for: {missing}")
        weights = torch.tensor(
            [float(class_weights[name]) for name in BAND_ORDER], dtype=torch.float32
        )
        loss = nn.CrossEntropyLoss(weight=weights)

    if device is not None:
        loss = loss.to(device)
    return loss


# ---------------------------------------------------------------------------
# Device / dependency guards
# ---------------------------------------------------------------------------

def device_report() -> Dict[str, Any]:
    """Describe the *actual* runtime device state. Reports facts only."""
    report: Dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_device_name": None,
        "cuda_device_count": 0,
    }
    if report["cuda_available"]:
        report["cuda_device_count"] = int(torch.cuda.device_count())
        try:
            report["cuda_device_name"] = torch.cuda.get_device_name(0)
        except Exception:  # pragma: no cover - driver-level failures
            report["cuda_device_name"] = "unavailable"
    return report


def resolve_device() -> torch.device:
    """Return the actual runtime device (CUDA only if it really is available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def update_best_tracking(
    *,
    best_val_loss: float,
    best_epoch: int,
    patience_counter: int,
    epoch: int,
    val_loss: float,
) -> Tuple[bool, float, int, int]:
    """Apply the validation-based checkpoint-selection and early-stopping rule.

    Returns ``(is_best, best_val_loss, best_epoch, patience_counter)``.

    This is the single place the selection rule lives, and it consults the
    validation loss only. It takes no test-partition input by construction, which
    is what keeps model selection independent of the held-out test set.
    """
    if val_loss < best_val_loss:
        return True, float(val_loss), int(epoch), 0
    return False, best_val_loss, best_epoch, patience_counter + 1


def confirm_guard_contract() -> Dict[str, bool]:
    """Report the opt-in flags this pipeline requires before it will do work.

    Kept as a function so the notebook and the tests can assert they agree on the
    guard contract instead of duplicating the flag names in prose.
    """
    return {
        "training_requires_confirm": True,
        "training_requires_smoke_test": True,
        "final_evaluation_requires_confirm": True,
    }


def require_torchvision() -> None:
    """Fail clearly (and in one place) when torchvision is unavailable."""
    try:
        import torchvision  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "torchvision is required for the frozen ResNet-50 baseline but is not "
            "importable in this runtime. Install the pinned requirements from "
            "colab/requirements-colab.txt before training."
        ) from exc


def ensure_training_preconditions(*, allow_weight_download: bool) -> Dict[str, Any]:
    """Fail clearly before any training work starts.

    Checks torch/torchvision importability, the pretrained-weight cache, and the
    real device state.
    """
    require_torchvision()

    if not allow_weight_download and not pretrained_weights_available():
        try:
            cache_hint = str(weights_cache_path())
        except Exception:  # pragma: no cover - torch itself unavailable
            cache_hint = "the torch hub checkpoint directory"
        raise FileNotFoundError(
            f"ImageNet ResNet-50 weights are not cached locally at {cache_hint}. Re-run with "
            "allow_weight_download=True to permit the one-time download, or pre-seed the "
            "torch cache."
        )

    return device_report()


# ---------------------------------------------------------------------------
# Output directory / overwrite protection
# ---------------------------------------------------------------------------

def _existing_run_summary(output_dir: Path) -> Optional[Dict[str, Any]]:
    """Return the existing run manifest if the directory already holds a run."""
    manifest_path = output_dir / RUN_MANIFEST_NAME
    if manifest_path.is_file():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"status": "unreadable"}
    if (output_dir / RUN_META_NAME).is_file():
        return {"status": "unknown"}
    return None


def resolve_output_dir(output_dir: Path, *, overwrite: bool = False) -> Path:
    """Validate the output directory and return it. **Never deletes anything.**

    Raises
    ------
    FileExistsError
        If the path exists but is not a directory, or if it already contains a
        previous run and ``overwrite`` is not explicitly True.
    """
    out = Path(output_dir).expanduser()
    if out.exists():
        if not out.is_dir():
            raise FileExistsError(f"Output path exists and is not a directory: {out}")
        existing = _existing_run_summary(out)
        if existing is not None and not overwrite:
            raise FileExistsError(
                f"Output directory already contains a run (status={existing.get('status')!r}): {out}\n"
                "Refusing to overwrite. Choose a new output directory, or pass "
                "overwrite=True to explicitly replace the previous run."
            )
    out.mkdir(parents=True, exist_ok=True)
    return out


def clear_output_dir(output_dir: Path) -> Path:
    """Delete an existing run directory and recreate it empty.

    This is destructive and is deliberately a separate call so the pipeline can
    defer it until cheap validation has already succeeded.
    """
    out = Path(output_dir).expanduser()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    return out


def prepare_output_dir(output_dir: Path, *, overwrite: bool = False) -> Path:
    """Validate the output directory, replacing a previous run only if asked.

    Convenience wrapper: :func:`resolve_output_dir` plus, when ``overwrite=True``
    and a previous run is present, :func:`clear_output_dir`.
    """
    out = resolve_output_dir(output_dir, overwrite=overwrite)
    if overwrite and _existing_run_summary(out) is not None:
        clear_output_dir(out)
    return out


def write_run_meta(output_dir: Path, config: Dict[str, Any]) -> Path:
    """Write the run configuration to ``run_meta.json``."""
    meta_path = output_dir / RUN_META_NAME
    meta_path.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    return meta_path


def write_run_manifest(
    output_dir: Path,
    config: Dict[str, Any],
    state: TrainingState,
    *,
    status: str,
    note: Optional[str] = None,
) -> Path:
    """Write/refresh ``run_manifest.json`` with an explicit run status."""
    manifest_path = output_dir / RUN_MANIFEST_NAME
    payload: Dict[str, Any] = {
        "status": status,
        "config": config,
        "best_epoch": state["best_epoch"],
        "best_val_loss": state["best_val_loss"],
        "train_epochs": len(state["train_history"]),
        "val_epochs": len(state["val_history"]),
        "resume_supported": bool(config.get("resume_supported", False)),
        "provisional_note": PROVISIONAL_NOTE,
    }
    if config.get("resumed_from"):
        payload["resumed_from"] = config["resumed_from"]
    if note:
        payload["note"] = note
    manifest_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return manifest_path


def mark_incomplete(output_dir: Path, reason: str) -> Path:
    """Mark an interrupted/failed run so it is never mistaken for a completed one."""
    marker_path = output_dir / INCOMPLETE_MARKER_NAME
    marker_path.write_text(
        "This output directory contains an INCOMPLETE training run.\n"
        f"Reason: {reason}\n"
        "The completed epochs are checkpointed. Continue from the last finished epoch with\n"
        "    python -m colab.pipeline train --confirm --resume\n"
        "(or --overwrite to discard this run and start fresh). Until the run completes,\n"
        "do not treat any metric, checkpoint, or summary here as a final result.\n",
        encoding="utf-8",
    )
    return marker_path


# ---------------------------------------------------------------------------
# Checkpoint IO
# ---------------------------------------------------------------------------

def _atomic_save(payload: Dict[str, Any], path: Path) -> Path:
    """``torch.save`` to a temp file, then rename.

    A crash (or a Colab disconnect) mid-write must not leave a truncated
    ``model.pt`` that :func:`latest_epoch_checkpoint` would count as a usable resume
    point. The rename is atomic on the same filesystem, and the temp file is removed
    on failure so nothing partial survives.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def _atomic_copy(source: Path, target: Path) -> Path:
    """Copy a file via a temp name + rename, for the best-checkpoint mirror."""
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    try:
        shutil.copyfile(source, tmp)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()
    return target


def save_checkpoint(
    output_dir: Path,
    state: TrainingState,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    epoch: int,
    *,
    is_best: bool = False,
) -> Path:
    """Save an epoch checkpoint, and refresh the best checkpoint when ``is_best``."""
    epoch_dir = output_dir / "checkpoints" / f"epoch_{epoch:04d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    collab_compat = {
        "torchvision_version_marker": "resnet50_imagenet1k_v1",
        "head_class_count": int(config.get("num_classes") or len(BAND_ORDER)),
        "band_order": list(BAND_ORDER),
        "preprocessing": {
            "kind": "resize_and_pad",
            "image_size": config.get("image_size", 224),
            "normalize": {
                "mean": list(IMAGENET_MEAN),
                "std": list(IMAGENET_STD),
            },
        },
    }

    model_path = epoch_dir / "model.pt"
    _atomic_save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "collab_compat": collab_compat,
        },
        model_path,
    )

    meta_path = epoch_dir / "training_state.json"
    _atomic_write_json(
        meta_path,
        {
            "state": state,
            "config": config,
            "collab_compat": collab_compat,
            "saved_at": time.time(),
            "is_best": is_best,
        },
    )

    if is_best:
        best_dir = output_dir / BEST_CHECKPOINT_DIR_NAME
        best_dir.mkdir(parents=True, exist_ok=True)
        _atomic_copy(model_path, best_dir / "model.pt")
        _atomic_copy(meta_path, best_dir / "training_state.json")

    return epoch_dir


def inspect_checkpoint_metadata(path: Path) -> Dict[str, Any]:
    """Return the metadata blob from a checkpoint directory, without loading weights."""
    return load_training_state(Path(path) / "training_state.json")


def checkpoint_compat_report(path: Path) -> Dict[str, Any]:
    """Summarize a checkpoint's metadata and verify it matches this pipeline.

    Raises
    ------
    ValueError
        If the checkpoint predates the current metadata contract, targets a
        different model, or has a head size that does not match ``BAND_ORDER``.
    """
    path = Path(path)
    meta = inspect_checkpoint_metadata(path)
    config = meta.get("config", {}) or {}
    # collab_compat is written both at the metadata top level and inside the
    # weights payload; accept either so older/partial checkpoints are diagnosable.
    collab = meta.get("collab_compat") or config.get("collab_compat")
    if collab is None:
        raise ValueError(
            "Checkpoint is missing 'collab_compat' metadata; it may predate this pipeline version."
        )
    head_count = collab.get("head_class_count")
    if head_count != len(BAND_ORDER):
        raise ValueError(
            f"Checkpoint head class count {head_count} does not match the current "
            f"BAND_ORDER length {len(BAND_ORDER)}."
        )
    if config.get("model") != "resnet50_frozen":
        raise ValueError(
            f"Checkpoint model type {config.get('model')!r} is not 'resnet50_frozen'."
        )
    if not (path / "model.pt").is_file():
        raise FileNotFoundError(f"Checkpoint weights missing: {path / 'model.pt'}")

    state = meta.get("state", {}) or {}
    return {
        "checkpoint_dir": str(path),
        "checkpoint_epoch": state.get("epoch"),
        "best_epoch": state.get("best_epoch"),
        "best_val_loss": state.get("best_val_loss"),
        "collab_compat": collab,
        "config_summary": {
            "task": config.get("task"),
            "image_size": config.get("image_size"),
            "seed": config.get("seed"),
            "device": config.get("device"),
        },
        "compatible": True,
    }


def save_run_summary(output_dir: Path, config: Dict[str, Any], state: TrainingState) -> Path:
    """Write per-epoch history plus final validation metrics for a completed run."""
    summary_path = output_dir / RUN_SUMMARY_NAME
    # Atomic: a disconnect while writing the summary must not leave truncated JSON
    # that the artifact verification would report as a corrupt completed run.
    summary_path = _atomic_write_json(
        summary_path,
        {
            "status": "completed",
            "config": config,
            "final_state": state,
            "train_history": state["train_history"],
            "val_history": state["val_history"],
            "best_epoch": state["best_epoch"],
            "best_val_loss": state["best_val_loss"],
            "provisional_note": PROVISIONAL_NOTE,
        },
    )
    # Remove any stale incomplete marker: this run really did finish.
    marker = output_dir / INCOMPLETE_MARKER_NAME
    if marker.exists():
        marker.unlink()
    return summary_path


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

RESUME_STATE_NAME = "resume_state.json"


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Return parsed JSON, or None when the file is absent or unreadable."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> Path:
    """Write JSON via a temp file + replace, so an interrupted write cannot corrupt it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return path


def latest_epoch_checkpoint(output_dir: Path) -> Tuple[Optional[Path], Optional[int]]:
    """Return ``(epoch_dir, epoch_index)`` for the newest complete epoch checkpoint.

    Only folders that actually contain a ``model.pt`` count, so a crash while
    creating ``checkpoints/epoch_XXXX`` cannot be mistaken for a usable resume
    point.
    """
    root = Path(output_dir) / "checkpoints"
    if not root.is_dir():
        return None, None
    found: List[Tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("epoch_"):
            continue
        try:
            index = int(child.name.split("epoch_", 1)[1])
        except (IndexError, ValueError):
            continue
        if (child / "model.pt").is_file():
            found.append((index, child))
    if not found:
        return None, None
    index, path = max(found, key=lambda item: item[0])
    return path, index


def write_resume_state(
    output_dir: Path,
    config: Dict[str, Any],
    state: TrainingState,
    *,
    epochs_completed: int,
) -> Path:
    """Persist everything needed to continue this run after a disconnect.

    ``state['best_state']`` holds live tensors and is deliberately excluded: the
    best weights already live in ``best_checkpoint/model.pt``, so the resume
    pointer only carries history, best-loss bookkeeping, and the early-stopping
    counter.
    """
    payload = {
        "epochs_completed": int(epochs_completed),
        "state": {k: v for k, v in state.items() if k != "best_state"},
        "config": config,
        "resume_supported": True,
        "saved_at": time.time(),
    }
    return _atomic_write_json(Path(output_dir) / RESUME_STATE_NAME, payload)


def _state_from_json_state(raw: Dict[str, Any]) -> TrainingState:
    """Rebuild a training state from a saved JSON blob (best_state dropped)."""
    return {
        "epoch": int(raw.get("epoch") or 0),
        "best_val_loss": float(raw.get("best_val_loss", float("inf"))),
        "best_epoch": int(raw.get("best_epoch") or 0),
        "best_state": None,
        "patience_counter": int(raw.get("patience_counter") or 0),
        "train_history": list(raw.get("train_history") or []),
        "val_history": list(raw.get("val_history") or []),
    }


def plan_resume(output_dir: Path, *, requested_epochs: int) -> Dict[str, Any]:
    """Work out whether ``output_dir`` can be resumed, and from where.

    Read-only: it never writes, deletes, or loads model weights. Raises a plain
    error with a next step rather than a traceback, because every failure here is
    user-actionable (wrong directory, finished run, missing checkpoints).
    """
    out = Path(output_dir)
    if not out.is_dir():
        raise FileNotFoundError(
            f"Nothing to resume: the output directory does not exist yet: {out}\n"
            "Start a new run without --resume first."
        )

    manifest = _read_json(out / RUN_MANIFEST_NAME)
    meta = _read_json(out / RUN_META_NAME)
    if manifest is None and meta is None:
        raise FileNotFoundError(
            f"Nothing to resume at {out}: it holds no {RUN_MANIFEST_NAME} and no {RUN_META_NAME}.\n"
            "Point --output-dir at an existing run, or start a new one without --resume."
        )

    status = str((manifest or {}).get("status") or "unknown")
    epoch_dir, last_epoch = latest_epoch_checkpoint(out)
    if last_epoch is None or epoch_dir is None:
        raise FileNotFoundError(
            f"Cannot resume {out}: there is no checkpoints/epoch_XXXX/model.pt to continue from.\n"
            "A run that never finished its first epoch cannot be resumed; start it again with "
            "--overwrite."
        )

    previous_epochs = int(last_epoch) + 1
    pointer = _read_json(out / RESUME_STATE_NAME)
    if pointer is not None:
        completed = int(pointer.get("epochs_completed") or previous_epochs)
        if completed > previous_epochs:
            completed = previous_epochs
        previous_epochs = completed

    if status == "completed":
        trained = int((manifest or {}).get("train_epochs") or previous_epochs)
        if int(requested_epochs) <= trained:
            raise ValueError(
                f"The run at {out} already finished {trained} epoch(s) and its manifest says "
                f"'completed'. Requesting {int(requested_epochs)} epoch(s) would train nothing.\n"
                f"Pass --epochs greater than {trained} to extend the run, or --overwrite to start "
                "a fresh run in this directory."
            )

    # Prefer the explicit resume pointer; fall back to the epoch checkpoint metadata.
    raw_state: Optional[Dict[str, Any]] = None
    if pointer is not None and int(pointer.get("epochs_completed") or 0) == previous_epochs:
        raw_state = pointer.get("state")
    if not raw_state:
        epoch_meta = _read_json(epoch_dir / "training_state.json") or {}
        raw_state = epoch_meta.get("state")
    if not raw_state:
        raise FileNotFoundError(
            f"Cannot resume {out}: {epoch_dir / 'training_state.json'} has no 'state' block, so "
            "the best-loss and early-stopping bookkeeping is unrecoverable."
        )

    state = _state_from_json_state(raw_state)

    best_dir = out / BEST_CHECKPOINT_DIR_NAME
    if state["best_epoch"] >= 0 and not (best_dir / "model.pt").is_file() and state["best_val_loss"] != float("inf"):
        raise FileNotFoundError(
            f"Cannot resume {out}: the run selected a best epoch ({state['best_epoch']}) but "
            f"{best_dir / 'model.pt'} is missing, so the validation-selected weights are gone.\n"
            "Restore that folder from a backup, or start a fresh run with --overwrite."
        )

    return {
        "epoch_dir": epoch_dir,
        "last_epoch": int(last_epoch),
        "previous_epochs": previous_epochs,
        "status": status,
        "state": state,
    }


def _torch_load(path: Path, device: torch.device) -> Tuple[Dict[str, Any], bool]:
    """Load a checkpoint, preferring the safe ``weights_only=True`` path.

    Falls back only for this pipeline's own checkpoints (which embed an optimizer
    state dict and plain metadata), and says so in the returned flag.
    """
    try:
        return torch.load(path, map_location=device, weights_only=True), True
    except Exception:
        return torch.load(path, map_location=device, weights_only=False), False


def load_epoch_checkpoint(
    epoch_dir: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> Dict[str, Any]:
    """Restore model (and optimizer) weights from one epoch checkpoint.

    Returns ``{'epoch', 'weights_only'}``. Raises with the exact path and reason
    when the checkpoint is unusable, instead of letting ``load_state_dict`` fail
    with an opaque size-mismatch traceback.
    """
    epoch_dir = Path(epoch_dir)
    model_path = epoch_dir / "model.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint weights missing: {model_path}")

    payload, weights_only = _torch_load(model_path, device)
    if "model_state_dict" not in payload:
        raise ValueError(f"Checkpoint {model_path} has no 'model_state_dict' entry.")
    try:
        model.load_state_dict(payload["model_state_dict"])
    except RuntimeError as exc:
        raise ValueError(
            f"Checkpoint {model_path} does not match this model architecture: {exc}"
        ) from exc

    if optimizer is not None and payload.get("optimizer_state_dict"):
        try:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        except (ValueError, KeyError) as exc:
            raise ValueError(
                f"Checkpoint {model_path} has an optimizer state that this optimizer cannot "
                f"restore: {exc}"
            ) from exc

    return {"epoch": int(payload.get("epoch", -1)), "weights_only": weights_only}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training(
    *,
    split_dir: Path,
    dataset_root: Path,
    output_dir: Path,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 3,
    image_size: int = 224,
    seed: int = 12345,
    device: Optional[torch.device] = None,
    confirm_training: bool = False,
    allow_weight_download: bool = False,
    overwrite: bool = False,
    smoke_test_passed: bool = False,
    resume: bool = False,
) -> Dict[str, Any]:
    """Run the guarded frozen-ResNet-50 training loop.

    Guards (all must pass before any training work happens)
    -----------------------------------------------------
    - ``confirm_training`` must be True: the explicit opt-in.
    - ``smoke_test_passed`` must be True: the notebook sets this only after the
      smoke test exits 0.
    - torchvision must be importable.
    - ImageNet weights must be cached, unless ``allow_weight_download=True``.
    - The output directory must be new, unless ``overwrite=True`` or ``resume=True``.

    The test partition is never loaded, scored, or inspected here. Final test
    evaluation is a separate, explicitly invoked step
    (:func:`run_final_evaluation`).

    Resume
    ------
    ``resume=True`` continues the run already in ``output_dir``: it restores the
    newest complete epoch checkpoint (model **and** optimizer state), the epoch
    histories, the best-validation bookkeeping, and the early-stopping counter,
    then trains from the next epoch up to ``epochs``. It never deletes the
    existing run, and it refuses (with the reason) when there is nothing usable
    to resume. ``resume=True`` together with ``overwrite=True`` is rejected as
    contradictory.

    A ``completed`` run can only be resumed to *extend* it (``epochs`` greater
    than the epochs already trained); asking for the same or fewer epochs is a
    no-op and is rejected instead of silently rewriting a finished result.

    Without ``resume=True`` an interrupted run is still marked incomplete and
    must be restarted from scratch.
    """
    if not confirm_training:
        raise PermissionError(
            "Training is disabled by default. Pass confirm_training=True (the notebook "
            "does this only after you set RUN_TRAINING = True) to start a run."
        )
    if not smoke_test_passed:
        raise PermissionError(
            "Refusing to train before the smoke test has passed. Run "
            "`python -m pytest colab/smoke_test.py` (or the notebook smoke cell) first "
            "and pass smoke_test_passed=True only when it exits 0."
        )
    if int(epochs) < 1:
        raise ValueError(
            f"epochs must be at least 1 (got {epochs!r}); refusing to write a 'completed' "
            "run that trained no epochs."
        )

    runtime = ensure_training_preconditions(allow_weight_download=allow_weight_download)

    split_dir = Path(split_dir)
    dataset_root = Path(dataset_root)
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    rows = load_split_csv(split_dir)
    train_rows = filter_by_partition(rows, "train")
    val_rows = filter_by_partition(rows, "validation")
    if not train_rows:
        raise ValueError("Split CSV contains no training rows.")
    if not val_rows:
        raise ValueError("Split CSV contains no validation rows.")

    # Invariant: the training path must never see test rows.
    train_paths = {r["image_path"] for r in train_rows}
    val_paths = {r["image_path"] for r in val_rows}
    test_paths = {r["image_path"] for r in filter_by_partition(rows, "test")}
    if train_paths & test_paths or val_paths & test_paths:
        raise ValueError("Split invariant violated: test rows overlap train/validation rows.")

    if resume and overwrite:
        raise ValueError(
            "resume=True and overwrite=True are contradictory: resuming continues the existing "
            "run, while overwriting discards it. Choose one."
        )

    # Validation only: nothing is deleted here. The destructive clear happens
    # further down, once cheap validation has succeeded. ``resume`` must be allowed
    # to target an existing run, and it never deletes anything: ``replace_existing``
    # below still depends on the explicit ``overwrite`` alone.
    out = resolve_output_dir(output_dir, overwrite=bool(overwrite or resume))
    replace_existing = bool(overwrite and _existing_run_summary(out) is not None)

    # Read-only planning: raises a plain, actionable error when this directory
    # cannot be resumed, before any dataset, model, or file work happens.
    resume_plan = plan_resume(out, requested_epochs=int(epochs)) if resume else None

    if device is None:
        device = resolve_device()

    config: Dict[str, Any] = {
        "task": "primary_five_band_provisional",
        "model": "resnet50_frozen",
        "split_dir": str(split_dir),
        "dataset_root": str(dataset_root),
        "output_dir": str(out),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "patience": patience,
        "image_size": image_size,
        "seed": seed,
        "device": str(device),
        "num_classes": len(BAND_ORDER),
        "band_order": list(BAND_ORDER),
        "preprocessing": "resize_and_pad+imagenet_normalize",
        "crops": "none",
        "class_weights_source": "train_partition_only",
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "runtime": runtime,
        "provisional_note": PROVISIONAL_NOTE,
        "resume_supported": True,
        "resume": bool(resume),
    }

    if resume_plan is not None:
        state = resume_plan["state"]
        start_epoch = resume_plan["last_epoch"] + 1
        config["resumed_from"] = {
            "checkpoint_dir": str(resume_plan["epoch_dir"]),
            "last_completed_epoch": resume_plan["last_epoch"],
            "previous_epochs": resume_plan["previous_epochs"],
            "previous_status": resume_plan["status"],
        }
        config["epochs"] = int(epochs)
    else:
        state = initial_training_state()
        start_epoch = 0

    # Tracks whether this run has taken ownership of the output directory. Until it
    # has, the except-handler must not stamp a status onto whatever is already there.
    owns_output_dir = False

    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        train_ds = PrimarySplitDataset(train_rows, dataset_root, image_size=image_size)
        val_ds = PrimarySplitDataset(val_rows, dataset_root, image_size=image_size)
        if set(r["image_path"] for r in train_ds._rows) & test_paths:  # pragma: no cover
            raise ValueError("Split invariant violated: test rows reached the training dataset.")

        class_weights = compute_class_weights_from_rows(train_rows)
        config["class_weights"] = class_weights

        # Build the model BEFORE touching the output directory, so that a failure
        # here (missing images, unavailable weights) cannot destroy a previous run.
        model = build_frozen_resnet50(
            num_classes=len(BAND_ORDER), allow_download=allow_weight_download
        )
        assert_head_is_trainable(model)
        # The loss holds its class weights in a buffer and must live on the same
        # device as the logits; a CPU weight against CUDA logits fails immediately.
        loss_fn = make_weighted_loss_fn(class_weights, device=device)

        # Cheap validation has now succeeded, so it is finally safe to discard a
        # previous run at this path -- and only with an explicit overwrite=True.
        if replace_existing:
            clear_output_dir(out)
        out.mkdir(parents=True, exist_ok=True)
        write_run_meta(out, config)
        write_run_manifest(out, config, state, status="in_progress")
        owns_output_dir = True

        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        if resume_plan is not None:
            restored = load_epoch_checkpoint(resume_plan["epoch_dir"], model, optimizer, device)
            # A resumed run owns the directory again, so a stale incomplete marker
            # from the interrupted attempt must not outlive this takeover.
            marker = out / INCOMPLETE_MARKER_NAME
            if marker.exists():
                marker.unlink()
            print(json.dumps({
                "event": "resume_start",
                "checkpoint_dir": str(resume_plan["epoch_dir"]),
                "restored_epoch": restored["epoch"],
                "weights_only": restored["weights_only"],
                "start_epoch": start_epoch,
                "target_epochs": int(epochs),
                "best_epoch": state["best_epoch"],
                "best_val_loss": state["best_val_loss"],
                "patience_counter": state["patience_counter"],
                "history_epochs": len(state["train_history"]),
            }))

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False
        )

        print(json.dumps({
            "event": "training_start",
            "device": str(device),
            "cuda_available": runtime["cuda_available"],
            "cuda_device_name": runtime["cuda_device_name"],
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "epochs": epochs,
            "class_weights": class_weights,
        }))

        # On resume these MUST come from the restored state, not from their initial
        # values: starting again at inf/0 would make the first resumed epoch look
        # like a new best (overwriting the real best checkpoint with worse weights)
        # and would silently reset the early-stopping counter.
        best_state: Optional[dict] = state["best_state"]
        best_val_loss = float(state["best_val_loss"])
        best_epoch = int(state["best_epoch"])
        patience_counter = int(state["patience_counter"])
        epoch = -1

        for epoch in range(start_epoch, int(epochs)):
            epoch_start = time.time()
            train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, device)
            val_metrics = evaluate_epoch(model, val_loader, loss_fn, device)
            epoch_seconds = time.time() - epoch_start

            train_metrics["epoch"] = epoch
            val_metrics["epoch"] = epoch
            train_metrics["epoch_time_seconds"] = epoch_seconds
            state["train_history"].append(train_metrics)
            state["val_history"].append(val_metrics)

            is_best, best_val_loss, best_epoch, patience_counter = update_best_tracking(
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                patience_counter=patience_counter,
                epoch=epoch,
                val_loss=float(val_metrics["loss"]),
            )
            if is_best:
                best_state = {
                    "model_state_dict": {k: v.clone() for k, v in model.state_dict().items()},
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                }

            state["epoch"] = epoch
            state["best_val_loss"] = best_val_loss
            state["best_epoch"] = best_epoch
            state["best_state"] = best_state
            state["patience_counter"] = patience_counter

            save_checkpoint(out, state, model, optimizer, config, epoch, is_best=is_best)
            write_resume_state(out, config, state, epochs_completed=epoch + 1)
            write_run_manifest(out, config, state, status="in_progress")

            print(json.dumps({
                "event": "epoch_end",
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
                "epoch_seconds": round(epoch_seconds, 3),
                "device": str(device),
            }))

            if patience_counter >= patience:
                print(json.dumps({
                    "event": "early_stop",
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                }))
                break

        save_run_summary(out, config, state)
        write_resume_state(out, config, state, epochs_completed=len(state["train_history"]))
        note = "Validation-selected run. Test partition not evaluated here."
        if start_epoch >= int(epochs):
            note = (
                "Resumed run had already reached the requested epoch count; no additional "
                "training was performed. " + note
            )
        manifest = write_run_manifest(out, config, state, status="completed", note=note)
        print(json.dumps({
            "event": "training_complete",
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "epochs_run": len(state["train_history"]),
            "epochs_requested": int(epochs),
            "started_at_epoch": start_epoch,
            "resumed": resume_plan is not None,
            "manifest": str(manifest),
        }))

        return {
            "config": config,
            "final_state": state,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_checkpoint_dir": out / BEST_CHECKPOINT_DIR_NAME,
            "output_dir": out,
            "run_summary_path": out / RUN_SUMMARY_NAME,
            "device": str(device),
            "runtime": runtime,
            "resumed_from": config.get("resumed_from"),
        }
    except BaseException as exc:
        # Only stamp a status once this run actually owns the directory, so that a
        # failure during validation cannot relabel a previous run as incomplete.
        if owns_output_dir:
            try:
                write_run_manifest(
                    out, config, state, status="incomplete", note=f"{type(exc).__name__}: {exc}"
                )
                mark_incomplete(out, f"{type(exc).__name__}: {exc}")
            except Exception:  # pragma: no cover - best-effort bookkeeping
                pass
        raise


# ---------------------------------------------------------------------------
# Final test evaluation (separate, explicitly invoked step)
# ---------------------------------------------------------------------------

def run_final_evaluation(
    *,
    split_dir: Path,
    dataset_root: Path,
    output_dir: Path,
    checkpoint_dir: Optional[Path] = None,
    batch_size: int = 32,
    image_size: int = 224,
    device: Optional[torch.device] = None,
    confirm_final_evaluation: bool = False,
) -> Dict[str, Any]:
    """Evaluate the selected checkpoint on the held-out test partition.

    This is intentionally a separate entry point from :func:`run_training` so the
    test partition is never touched during training, hyperparameter selection,
    early stopping, or checkpoint selection. It must be explicitly requested.
    """
    if not confirm_final_evaluation:
        raise PermissionError(
            "Final test evaluation is disabled by default. Pass "
            "confirm_final_evaluation=True to score the held-out test partition."
        )

    output_dir = Path(output_dir)
    best_dir = Path(checkpoint_dir) if checkpoint_dir is not None else (output_dir / BEST_CHECKPOINT_DIR_NAME)
    if not best_dir.exists():
        raise FileNotFoundError(
            f"Best checkpoint directory not found at {best_dir}. "
            "Run training first, or pass checkpoint_dir explicitly."
        )

    compat = checkpoint_compat_report(best_dir)
    meta = inspect_checkpoint_metadata(best_dir)
    config = meta.get("config", {}) or {}

    split_dir = Path(split_dir)
    dataset_root = Path(dataset_root)
    rows = load_split_csv(split_dir)
    test_rows = filter_by_partition(rows, "test")
    if not test_rows:
        raise ValueError("Split CSV contains no test rows for final evaluation.")

    train_paths = {r["image_path"] for r in filter_by_partition(rows, "train")}
    val_paths = {r["image_path"] for r in filter_by_partition(rows, "validation")}
    test_paths = {r["image_path"] for r in test_rows}
    if test_paths & (train_paths | val_paths):
        raise ValueError("Split invariant violated: test rows overlap train/validation rows.")

    if device is None:
        device = resolve_device()

    # Evaluation restores the trained weights, so the ImageNet weights are neither
    # needed nor downloaded here: the architecture is built with weights=None. The
    # load below uses strict key matching, so this still validates the architecture.
    # The dependency check comes after the path/checkpoint checks so that the cheap,
    # actionable errors (no checkpoint yet, wrong split) are reported first.
    require_torchvision()
    model = build_frozen_resnet50(num_classes=len(BAND_ORDER), pretrained=False)
    payload = torch.load(best_dir / "model.pt", map_location=device, weights_only=True)
    if "model_state_dict" not in payload:
        raise ValueError(
            f"Checkpoint at {best_dir / 'model.pt'} has no 'model_state_dict' entry."
        )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()

    class_weights = config.get("class_weights")
    if not class_weights:
        raise ValueError(
            "Checkpoint metadata is missing 'class_weights'; cannot reproduce the "
            "training-time weighted loss for the reported test loss."
        )
    loss_fn = make_weighted_loss_fn(class_weights, device=device)

    test_ds = PrimarySplitDataset(test_rows, dataset_root, image_size=image_size)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False
    )

    metrics = evaluate_epoch(model, test_loader, loss_fn, device)

    output_eval_dir = output_dir / TEST_EVAL_DIR_NAME
    output_eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = output_eval_dir / "final_test_metrics.json"
    eval_path.write_text(
        json.dumps(
            {
                "partition": "test",
                "split_source": str(split_dir),
                "checkpoint_dir": str(best_dir),
                "checkpoint_epoch": compat["checkpoint_epoch"],
                "best_epoch": compat["best_epoch"],
                "best_val_loss": compat["best_val_loss"],
                "image_size": image_size,
                "device": str(device),
                "band_order": list(BAND_ORDER),
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "per_class": metrics["per_class"],
                "confusion_matrix": metrics["confusion_matrix"],
                "num_samples": metrics["num_samples"],
                "provisional_note": PROVISIONAL_NOTE,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(json.dumps({
        "event": "final_evaluation_done",
        "partition": "test",
        "num_samples": metrics["num_samples"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "output_path": str(eval_path),
        "device": str(device),
    }))

    return {"metrics": metrics, "output_path": str(eval_path), "compatibility": compat}
