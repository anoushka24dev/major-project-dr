"""Checkpoint saving, fingerprinting and safe reuse.

A checkpoint is only reusable when the things that would change its meaning still
match: the architecture, the class count and order, the image size, the manifest
version and the label mapping. Those facts are combined into a fingerprint that is
stored inside every checkpoint and re-checked before any resume.

`torch.save` writes to a temporary file and is then moved into place, so an
interrupted save can never leave a half-written checkpoint that looks valid.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

CHECKPOINT_FILENAME = "model.pt"
STATE_FILENAME = "training_state.json"
BEST_DIR = "best_checkpoint"
LAST_DIR = "last_checkpoint"
CHECKPOINT_FORMAT_VERSION = 2


def _atomic_save(payload: Dict[str, Any], path: Path) -> Path:
    """Pickle to ``path`` via a temporary file plus atomic replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)
    return target


def _atomic_write_json(payload: Dict[str, Any], path: Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    os.replace(temporary, target)
    return target


def build_fingerprint(
    *,
    architecture: str,
    classes: Sequence[str],
    image_size: int,
    manifest_version: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a checkpoint fingerprint (plus a short digest of it)."""
    payload: Dict[str, Any] = {
        "architecture": str(architecture),
        "classes": list(classes),
        "num_classes": len(classes),
        "image_size": int(image_size),
        "manifest_version": str(manifest_version),
    }
    if extra:
        payload.update(extra)
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    payload["digest"] = hashlib.sha256(encoded).hexdigest()[:16]
    return payload


def compare_fingerprints(
    stored: Optional[Dict[str, Any]],
    current: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """Compare a stored fingerprint with the current one.

    Returns ``(compatible, reasons)``. A missing stored fingerprint is treated as
    incompatible: an unverifiable checkpoint must not be silently trusted.
    """
    if not stored:
        return False, ["checkpoint has no fingerprint"]
    reasons: List[str] = []
    for key in ("architecture", "num_classes", "image_size", "manifest_version"):
        if stored.get(key) != current.get(key):
            reasons.append(
                f"{key} differs: checkpoint={stored.get(key)!r}, current={current.get(key)!r}"
            )
    if list(stored.get("classes") or []) != list(current.get("classes") or []):
        reasons.append(
            f"class order differs: checkpoint={stored.get('classes')}, current={current.get('classes')}"
        )
    return (not reasons), reasons


def save_checkpoint(
    directory: Path,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    epoch: int,
    history: Sequence[Dict[str, Any]],
    best_metric: float,
    best_epoch: int,
    fingerprint: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Write ``model.pt`` + ``training_state.json`` into ``directory``."""
    target = Path(directory)
    payload: Dict[str, Any] = {
        "checkpoint_format": CHECKPOINT_FORMAT_VERSION,
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "fingerprint": fingerprint,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        payload.update(extra)
    model_path = _atomic_save(payload, target / CHECKPOINT_FILENAME)

    state = {
        "format": CHECKPOINT_FORMAT_VERSION,
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "best_epoch": int(best_epoch),
        "history": list(history),
        "fingerprint": fingerprint,
        "saved_at": payload["saved_at"],
    }
    state_path = _atomic_write_json(state, target / STATE_FILENAME)
    return {"model": str(model_path), "state": str(state_path)}


def load_checkpoint(path: Path, *, map_location: Any = "cpu") -> Dict[str, Any]:
    """Load a checkpoint payload, raising a clear error when unusable."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {target}")
    try:
        payload = torch.load(target, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Could not load checkpoint {target}: {exc}") from exc
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise RuntimeError(f"{target} is not a pipeline checkpoint (missing 'model_state')")
    return payload


def load_training_state(directory: Path) -> Optional[Dict[str, Any]]:
    """Read ``training_state.json`` if it exists."""
    path = Path(directory) / STATE_FILENAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_reusable(
    directory: Path,
    current_fingerprint: Dict[str, Any],
) -> Dict[str, Any]:
    """Report whether ``directory`` holds a checkpoint worth reusing."""
    target = Path(directory)
    model_path = target / CHECKPOINT_FILENAME
    result: Dict[str, Any] = {
        "path": str(model_path),
        "exists": model_path.is_file(),
        "compatible": False,
        "reasons": [],
        "state": load_training_state(target),
    }
    if not result["exists"]:
        result["reasons"].append("no checkpoint file")
        return result
    try:
        payload = load_checkpoint(model_path)
    except Exception as exc:
        result["reasons"].append(str(exc))
        return result
    compatible, reasons = compare_fingerprints(payload.get("fingerprint"), current_fingerprint)
    result["compatible"] = compatible
    result["reasons"] = reasons
    result["epoch"] = payload.get("epoch")
    result["saved_at"] = payload.get("saved_at")
    return result


def restore_into(
    model: torch.nn.Module,
    payload: Dict[str, Any],
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> Dict[str, Any]:
    """Load model (and optionally optimizer/scheduler) state from a payload."""
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state"):
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state"):
        scheduler.load_state_dict(payload["scheduler_state"])
    return {
        "epoch": int(payload.get("epoch", 0)),
        "has_optimizer_state": bool(payload.get("optimizer_state")),
        "has_scheduler_state": bool(payload.get("scheduler_state")),
    }


def archive_previous_run(run_dir: Path) -> Optional[Path]:
    """Move an existing run directory aside instead of overwriting it."""
    source = Path(run_dir)
    if not source.exists():
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    destination = source.with_name(f"{source.name}_superseded_{stamp}")
    counter = 1
    while destination.exists():
        counter += 1
        destination = source.with_name(f"{source.name}_superseded_{stamp}_{counter}")
    os.replace(source, destination)
    return destination
