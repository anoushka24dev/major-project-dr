"""Optional test-time augmentation (TTA) for the five-class classifier.

What TTA is here
----------------
A TTA view is a *medically defensible* perturbation of the already preprocessed
image tensor (resized, padded, ImageNet-normalised):

``identity``        the unmodified image (ordinary inference);
``hflip``           horizontal mirror. The retinal mosaic is near-symmetric for the
                    purpose of "is DR present, and how severe", so a mirrored view
                    is still a valid photograph of the same eye;
``rotate_p10``      +10 degrees rotation. This is inside the 10-15 degree range the
``rotate_m10``      training augmentation already uses, and does not move the optic
                    disc/macula relationship materially. The wedges exposed by the
                    rotation are filled with the normalised value of a black pixel,
                    exactly like the border the capture itself produces;
``brightness_p10``  +10% global illumination gain (camera/exposure change);
``brightness_m10``  -10% global illumination gain.

Deliberately **not** used: vertical flip, random cropping, large rotations and
channel permutation. Those are not anatomically neutral for fundus images: they
would make the averaged answer respond to a question the model was never asked.

How the views are combined
--------------------------
Class **probabilities** are averaged, never labels. Averaging one-hot decisions is a
vote that throws away confidence and can turn a 0.51/0.49 split into a hard tie; the
arithmetic mean of the per-view probability vectors keeps the five-class output
order and is renormalised into a proper probability vector.

Where the choice is made
------------------------
Ordinary inference (one forward pass) is always available and is the default. TTA is
only applied when a comparison **on the validation partition** selects it; the result
is written to ``outputs/tta/tta_selection.json`` and the evaluation stage reads that
file. The test partition is opened after the choice is already on disk, so TTA can
never be tuned on test data. No accuracy improvement is claimed anywhere: the
selection file records the measured validation metrics for both modes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .data import IMAGENET_MEAN, IMAGENET_STD


class TTAError(RuntimeError):
    """Raised when TTA cannot be applied honestly."""


#: Every view TTA may contain, in a fixed order.
VIEW_NAMES: Tuple[str, ...] = (
    "identity",
    "hflip",
    "rotate_p10",
    "rotate_m10",
    "brightness_p10",
    "brightness_m10",
)

#: Default averaged views: the four perturbations plus the unmodified image.
DEFAULT_VIEWS: Tuple[str, ...] = (
    "hflip",
    "rotate_p10",
    "rotate_m10",
    "brightness_p10",
    "brightness_m10",
)

_SELECTION_DIR = "tta"
_SELECTION_FILENAME = "tta_selection.json"
_ROTATION_DEGREES = 10.0
_GAIN = 0.10

#: The normalised value of a black (zero-intensity) pixel, per channel. Used as the
#: rotation fill so a rotated view has the same black border a real capture has
#: instead of an artificial grey wedge.
_BLACK_PIXEL = tuple(
    (0.0 - mean) / std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD)
)


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def _gain(tensor: torch.Tensor, factor: float) -> torch.Tensor:
    """Multiply pixel intensity by ``factor`` exactly, in normalised space.

    ``x_norm = (x - mean) / std`` so ``(g*x - mean) / std = g*x_norm + mean*(g-1)/std``.
    Applying that identity keeps the view a genuine exposure change rather than an
    arbitrary offset in the standardised tensor.
    """
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(1, -1, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(1, -1, 1, 1)
    return tensor * factor + mean * (factor - 1.0) / std


def _rotate(tensor: torch.Tensor, degrees: float) -> torch.Tensor:
    import torchvision.transforms.functional as TF

    return TF.rotate(
        tensor,
        degrees,
        interpolation=TF.InterpolationMode.BILINEAR,
        fill=list(_BLACK_PIXEL),
    )


def _view_function(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if name == "identity":
        return lambda tensor: tensor
    if name == "hflip":
        return lambda tensor: torch.flip(tensor, dims=[-1])
    if name == "rotate_p10":
        return lambda tensor: _rotate(tensor, _ROTATION_DEGREES)
    if name == "rotate_m10":
        return lambda tensor: _rotate(tensor, -_ROTATION_DEGREES)
    if name == "brightness_p10":
        return lambda tensor: _gain(tensor, 1.0 + _GAIN)
    if name == "brightness_m10":
        return lambda tensor: _gain(tensor, 1.0 - _GAIN)
    raise TTAError(f"Unknown TTA view {name!r}. Known views: {list(VIEW_NAMES)}")


def build_views(
    names: Optional[Sequence[str]] = None,
    *,
    include_original: bool = True,
) -> List[Tuple[str, Callable[[torch.Tensor], torch.Tensor]]]:
    """Return ``[(name, function), ...]`` for the requested views.

    ``include_original`` prepends the unmodified image, so the averaged prediction
    always contains the model's ordinary answer as one of its terms.
    """
    if names is None:
        requested = list(DEFAULT_VIEWS)
    elif isinstance(names, str):
        # A single name from configuration must not be iterated character by character.
        requested = [names]
    else:
        requested = [str(name) for name in names]
    ordered: List[str] = ["identity"] if include_original else []
    for name in requested:
        text = str(name)
        if text not in VIEW_NAMES:
            raise TTAError(
                f"Unknown TTA view {text!r} in inference.tta.transforms. "
                f"Known views: {list(VIEW_NAMES)}"
            )
        if text not in ordered:
            ordered.append(text)
    if not ordered:
        raise TTAError("TTA was requested but no views were configured.")
    return [(name, _view_function(name)) for name in ordered]


# ---------------------------------------------------------------------------
# Probability averaging
# ---------------------------------------------------------------------------

def mean_probabilities(
    per_view: Sequence[np.ndarray],
    *,
    num_classes: Optional[int] = None,
    renormalise: bool = True,
) -> np.ndarray:
    """Average class-probability matrices from several views.

    Each element must be a ``(samples, classes)`` matrix of probabilities in the same
    class order. The result is their arithmetic mean, renormalised per row, so it is
    itself a probability vector over the same five classes.
    """
    if not per_view:
        raise TTAError("mean_probabilities needs at least one view.")

    arrays = [np.asarray(view, dtype=np.float64) for view in per_view]
    shape = arrays[0].shape
    if len(shape) != 2:
        raise TTAError(
            f"Every view must be a 2-D (samples, classes) matrix; got shape {shape}."
        )
    if num_classes is not None and shape[1] != num_classes:
        raise TTAError(
            f"A view has {shape[1]} classes but the taxonomy has {num_classes}."
        )
    for index, array in enumerate(arrays):
        if array.shape != shape:
            raise TTAError(
                f"View {index} has shape {array.shape}; every view must have {shape}."
            )
        if not np.isfinite(array).all():
            raise TTAError(f"View {index} contains non-finite values.")
        if float(array.min()) < -1e-6:
            raise TTAError(f"View {index} contains negative 'probabilities'.")

    averaged = np.stack(arrays, axis=0).mean(axis=0)
    if renormalise:
        totals = averaged.sum(axis=1, keepdims=True)
        if (totals <= 0).any():
            raise TTAError("A row of the averaged probabilities sums to zero.")
        averaged = averaged / totals
    return averaged


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _unpack_batch(batch: Any) -> Tuple[torch.Tensor, Any, Any]:
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            return batch[0], batch[1], batch[2]
        if len(batch) == 2:
            return batch[0], batch[1], None
    raise TTAError(
        "Unsupported batch layout for TTA inference: expected (inputs, labels[, meta])."
    )


def _meta_records(meta: Any, count: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if isinstance(meta, dict):
        for index in range(count):
            record: Dict[str, Any] = {}
            for key, value in meta.items():
                if isinstance(value, (list, tuple)):
                    record[key] = value[index]
                elif torch.is_tensor(value) and value.dim() >= 1 and value.size(0) == count:
                    record[key] = value[index].item()
                else:
                    record[key] = value
            records.append(record)
        return records
    if isinstance(meta, (list, tuple)):
        for entry in meta:
            records.append(dict(entry) if isinstance(entry, dict) else {})
        return records
    return [{} for _ in range(count)]


def _forward(
    model: nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
    amp_enabled: bool,
) -> torch.Tensor:
    with torch.amp.autocast(
        device_type=device.type, enabled=amp_enabled and device.type == "cuda"
    ):
        return model(inputs)


def predict(
    model: nn.Module,
    loader: Any,
    device: torch.device | str,
    *,
    views: Optional[Sequence[Tuple[str, Callable[[torch.Tensor], torch.Tensor]]]] = None,
    num_classes: Optional[int] = None,
    amp_enabled: bool = False,
) -> Dict[str, Any]:
    """Run one partition through the model, with or without TTA.

    ``views=None`` (or an empty list) is ordinary inference: exactly one forward pass
    per batch. Otherwise every view is passed through the model and the class
    probabilities are averaged per sample, in the model's class order.

    Returned ``logits`` always come from the unmodified image, so temperature scaling
    stays a property of the model rather than of the averaged views.
    """
    device = device if isinstance(device, torch.device) else torch.device(str(device))
    view_list = list(views or [])
    used_views = [name for name, _function in view_list] or ["identity"]

    per_view: Dict[str, List[np.ndarray]] = {name: [] for name in used_views}
    ordinary_logits: List[np.ndarray] = []
    labels_out: List[int] = []
    meta_out: List[Dict[str, Any]] = []
    forward_passes = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs, labels, meta = _unpack_batch(batch)
            inputs = inputs.to(device, non_blocking=True)

            logits = _forward(model, inputs, device, amp_enabled)
            forward_passes += 1
            ordinary_logits.append(logits.detach().float().cpu().numpy())
            identity_probabilities = (
                torch.softmax(logits, dim=1).float().cpu().numpy()
            )
            if "identity" in per_view:
                per_view["identity"].append(identity_probabilities)

            for name, function in view_list:
                if name == "identity":
                    continue
                view_logits = _forward(model, function(inputs), device, amp_enabled)
                forward_passes += 1
                per_view[name].append(
                    torch.softmax(view_logits, dim=1).float().cpu().numpy()
                )

            if labels is not None:
                labels_out.extend(int(value) for value in labels)
            meta_out.extend(_meta_records(meta, int(inputs.size(0))))

    def _stack(chunks: List[np.ndarray], width: int) -> np.ndarray:
        if not chunks:
            return np.zeros((0, width), dtype=np.float64)
        return np.concatenate(chunks, axis=0)

    width = num_classes or (len(ordinary_logits[0][0]) if ordinary_logits else 0)
    stacked = {name: _stack(chunks, width) for name, chunks in per_view.items()}

    if len(used_views) == 1 and used_views[0] == "identity":
        probabilities = stacked["identity"]
    else:
        probabilities = mean_probabilities(
            [stacked[name] for name in used_views], num_classes=num_classes
        )

    return {
        "probabilities": probabilities,
        "logits": _stack(ordinary_logits, width),
        "labels": np.asarray(labels_out, dtype=int),
        "meta": meta_out,
        "views": used_views,
        "forward_passes": forward_passes,
        "mode": "ordinary" if len(used_views) == 1 else "tta",
    }


# ---------------------------------------------------------------------------
# Configuration and the saved selection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TTAOptions:
    """Resolved ``inference.tta`` configuration."""

    enabled: bool = False
    transforms: Tuple[str, ...] = DEFAULT_VIEWS
    include_original: bool = True
    reduction: str = "mean"
    selection_metric: str = "macro_f1"
    compare_on_validation: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "transforms": list(self.transforms),
            "include_original": self.include_original,
            "reduction": self.reduction,
            "selection_metric": self.selection_metric,
            "compare_on_validation": self.compare_on_validation,
        }


def options_from_config(cfg: Config) -> TTAOptions:
    """Read ``inference.tta`` from the configuration, rejecting unsupported values."""
    section = cfg.get("inference.tta", {}) or {}
    if not isinstance(section, dict):
        raise TTAError("inference.tta must be a mapping.")

    transforms = section.get("transforms")
    if transforms is None:
        transform_tuple: Tuple[str, ...] = DEFAULT_VIEWS
    elif isinstance(transforms, str):
        transform_tuple = (transforms,)
    else:
        transform_tuple = tuple(str(name) for name in transforms)
    for name in transform_tuple:
        if name not in VIEW_NAMES:
            raise TTAError(
                f"inference.tta.transforms contains unknown view {name!r}. "
                f"Known views: {list(VIEW_NAMES)}"
            )

    reduction = str(section.get("reduction", "mean")).strip().lower()
    if reduction != "mean":
        raise TTAError(
            f"inference.tta.reduction={reduction!r} is not supported. Only 'mean' "
            "(the arithmetic mean of class probabilities) is implemented."
        )

    metric = str(section.get("selection_metric", "macro_f1")).strip().lower()
    if metric not in {"macro_f1", "accuracy"}:
        raise TTAError(
            f"inference.tta.selection_metric={metric!r} is not supported; use "
            "'macro_f1' or 'accuracy'."
        )

    return TTAOptions(
        enabled=bool(section.get("enabled", False)),
        transforms=transform_tuple,
        include_original=bool(section.get("include_original", True)),
        reduction=reduction,
        selection_metric=metric,
        compare_on_validation=bool(section.get("compare_on_validation", True)),
    )


def selection_path(cfg: Config) -> Path:
    return cfg.outputs_dir(_SELECTION_DIR) / _SELECTION_FILENAME


def save_selection(cfg: Config, payload: Dict[str, Any]) -> Path:
    path = selection_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def load_selection(cfg: Config) -> Optional[Dict[str, Any]]:
    path = selection_path(cfg)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def selected_view_names(cfg: Config) -> Optional[List[str]]:
    """Views the evaluation stage must use, or ``None`` for ordinary inference.

    Ordinary inference is the default in every situation except one: TTA is enabled
    *and* the saved validation comparison selected it.
    """
    options = options_from_config(cfg)
    if not options.enabled:
        return None
    payload = load_selection(cfg)
    if not payload:
        return None
    if str(payload.get("selected", "ordinary")) != "tta":
        return None
    views = payload.get("views") or list(payload.get("options", {}).get("transforms") or options.transforms)
    names = [str(name) for name in views]
    if not options.include_original:
        return names
    return ["identity"] + [name for name in names if name != "identity"]


# ---------------------------------------------------------------------------
# Validation comparison
# ---------------------------------------------------------------------------

def _metric_block(
    probabilities: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    *,
    loss: Optional[float] = None,
) -> Dict[str, Any]:
    from .metrics import classification_metrics

    if len(labels) == 0:
        return {"num_samples": 0}
    metrics = classification_metrics(
        labels,
        probabilities.argmax(axis=1),
        probabilities=probabilities,
        class_names=list(class_names),
        compute_auc=False,
    )
    block = {
        "num_samples": int(len(labels)),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "cohen_kappa": float(metrics["cohen_kappa"]),
        "roc_auc": "not computed for this comparison",
    }
    if loss is not None:
        block["loss"] = float(loss)
    return block


def compare_on_validation(
    cfg: Config,
    logger,
    events=None,
    *,
    checkpoint: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Compare ordinary inference with TTA on the validation partition.

    Returns ``None`` when TTA is disabled. Otherwise both modes are measured on the
    same validation rows and the selected one is written to
    ``outputs/tta/tta_selection.json``. A tie keeps ordinary inference: added
    complexity has to earn its place with a strictly better validation score.
    """
    from torch.utils.data import DataLoader

    from .data import ClassifierDataset, load_manifest_rows
    from .hardware import detect_hardware, resolve_device
    from .models import build_resnet50

    options = options_from_config(cfg)
    if not options.enabled:
        logger.info("TTA disabled (inference.tta.enabled=false); ordinary inference only")
        return None
    if not options.compare_on_validation:
        logger.info("TTA enabled without a validation comparison (compare_on_validation=false)")
        return None

    class_names = list(cfg.classes)
    manifest = cfg.outputs_dir("manifests") / "validation_manifest.csv"
    if not manifest.is_file():
        raise TTAError(
            f"No validation manifest at {manifest}. Run the 'prepare' stage before "
            "selecting a TTA configuration."
        )
    rows = load_manifest_rows(manifest)
    if not rows:
        raise TTAError(f"The validation manifest at {manifest} contains no rows.")

    target = Path(checkpoint) if checkpoint else (
        cfg.outputs_dir("classification") / "best_resnet50.pt"
    )
    if not target.is_file():
        raise TTAError(
            f"No trained checkpoint at {target}. TTA can only be selected for a model "
            "that has already been trained."
        )

    image_size = int(cfg.require("preprocessing.classification.image_size"))
    device = resolve_device("auto")
    hardware = detect_hardware(
        mixed_precision_preference=cfg.mixed_precision_setting("classification"),
        disk_probe=cfg.root,
    )
    batch_size = cfg.effective_batch_size(
        "classification", fallback=hardware.recommended_classification_batch_size
    )
    num_workers = cfg.effective_num_workers("classification") or hardware.recommended_num_workers

    payload = torch.load(target, map_location=device, weights_only=False)
    fingerprint = payload.get("fingerprint") or {}
    if list(fingerprint.get("classes") or []) and list(fingerprint["classes"]) != class_names:
        raise TTAError(
            "The checkpoint's class order does not match the configured taxonomy: "
            f"checkpoint={fingerprint.get('classes')}, config={class_names}"
        )

    model = build_resnet50(
        num_classes=len(class_names), pretrained=False, allow_weight_download=False
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    dataset = ClassifierDataset(rows, cfg, train=False, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    views = build_views(options.transforms, include_original=options.include_original)
    logger.info(
        "Comparing ordinary inference with TTA (%d views) on %d validation images",
        len(views), len(rows),
    )
    ordinary = predict(model, loader, device, views=None, num_classes=len(class_names))
    augmented = predict(model, loader, device, views=views, num_classes=len(class_names))

    ordinary_block = _metric_block(ordinary["probabilities"], ordinary["labels"], class_names)
    tta_block = _metric_block(augmented["probabilities"], augmented["labels"], class_names)

    metric = options.selection_metric
    ordinary_score = float(ordinary_block.get(metric, 0.0))
    tta_score = float(tta_block.get(metric, 0.0))
    selected = "tta" if tta_score > ordinary_score + 1e-9 else "ordinary"

    selection: Dict[str, Any] = {
        "status": "completed",
        "selected": selected,
        "selection_metric": metric,
        "selection_partition": "validation",
        "validation_manifest": str(manifest),
        "validation_samples": int(len(rows)),
        "checkpoint": str(target),
        "options": options.as_dict(),
        "views": [name for name, _function in views],
        "reduction": "mean of class probabilities",
        "ordinary": ordinary_block,
        "tta": tta_block,
        "delta": {
            "macro_f1": round(tta_block.get("macro_f1", 0.0) - ordinary_block.get("macro_f1", 0.0), 6),
            "accuracy": round(tta_block.get("accuracy", 0.0) - ordinary_block.get("accuracy", 0.0), 6),
        },
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": (
            "Ordinary inference and TTA were compared on the validation partition only. "
            "The test partition was not consulted, so this choice is not tuned on test "
            "data. A tie keeps ordinary inference; no improvement is claimed beyond the "
            "numbers recorded here."
        ),
    }
    path = save_selection(cfg, selection)
    selection["selection_file"] = str(path)

    logger.info(
        "TTA comparison: ordinary macro-F1 %.4f | TTA macro-F1 %.4f -> selected %s",
        ordinary_score, tta_score, selected,
    )
    if events is not None:
        events.write(
            "tta_selection",
            selected=selected,
            metric=metric,
            ordinary=ordinary_score,
            tta=tta_score,
            views=selection["views"],
            selection_file=str(path),
        )
    return selection
