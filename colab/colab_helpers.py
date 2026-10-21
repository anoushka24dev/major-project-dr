"""
Colab helper utilities for the diabetic retinopathy training baseline.

This module is meant to be uploaded with the rest of the project into a Colab
runtime and imported from a notebook or script. It does not install packages,
train a model, download pretrained weights, or modify any dataset or split file.

The five-band primary label mapping is provisional. See
``audit_output/pretraining_audit/LABEL_POLICY_DRAFT.md`` for its documented
limitations. Nothing here implies specialist validation.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Constants that mirror the existing split artifact schema
# ---------------------------------------------------------------------------

SPLIT_CSV_NAME = "provisional_primary_split.csv"

# Provisional primary-task band order. This must stay aligned with
# dataset.labels.PRIMARY_MAPPING.
BAND_ORDER = [
    "No apparent DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "PDR",
]

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

DATASET_ENV = "DR_FUNDUS_ROOT"
SPLIT_DIR_ENV = "DR_SPLIT_DIR"

DATASET_DIR_NAME = "Diabetic Retinopathy and Retinal Diseases Fundus I"


def resolve_dataset_root() -> Path:
    """Return the dataset root directory for the uploaded project.

    Resolution order:
    1. ``DR_FUNDUS_ROOT`` environment variable, if set and present.
    2. A Google Drive mount path if Drive is mounted.
    3. A local Colab runtime folder, if the project was uploaded there.
    4. Raise a clear error if none of the above works.
    """
    env = os.environ.get(DATASET_ENV, "").strip()
    if env:
        candidate = Path(env)
        if candidate.exists():
            return candidate

    mounted = _try_drive_root()
    if mounted:
        candidate = mounted / DATASET_DIR_NAME
        if candidate.exists():
            return candidate

    local = Path("/content/major_project")
    if local.exists():
        candidate = local.parent / DATASET_DIR_NAME
        if candidate.exists():
            return candidate

    raise RuntimeError(
        "Dataset root not found. Set DR_FUNDUS_ROOT, mount Google Drive and place the "
        "dataset under it, or upload the project to /content/major_project."
    )


def resolve_split_dir() -> Path:
    """Return the directory that holds the existing split artifact folder.

    The split artifact lives in ``provisional_primary_split/``, which is expected to
    sit alongside the dataset or alongside the uploaded project.
    """
    env = os.environ.get(SPLIT_DIR_ENV, "").strip()
    if env:
        candidate = Path(env)
        if candidate.is_dir():
            return candidate

    mounted = _try_drive_root()
    if mounted:
        candidate = mounted / "major_project" / "provisional_primary_split"
        if candidate.is_dir():
            return candidate

    local_project = Path("/content/major_project")
    if local_project.is_dir():
        candidate = local_project / "provisional_primary_split"
        if candidate.is_dir():
            return candidate

    raise RuntimeError(
        "Split directory not found. Set DR_SPLIT_DIR, mount Google Drive with the "
        "project under it, or upload the project to /content/major_project."
    )


def _try_drive_root() -> Path | None:
    """Return a Drive root path if Drive appears to be mounted, otherwise None."""
    for candidate in (
        Path("/content/drive/MyDrive"),
        Path("/drive/MyDrive"),
    ):
        if candidate.exists():
            return candidate
    return None


def split_csv_path(split_dir: Path) -> Path:
    """Return the existing split CSV path from a split directory."""
    csv_path = split_dir / SPLIT_CSV_NAME
    if not csv_path.is_file():
        raise FileNotFoundError(f"Expected split CSV at {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Split CSV loading
# ---------------------------------------------------------------------------

def load_split_csv(split_dir: Path) -> List[Dict[str, str]]:
    """Load the existing split assignment without modifying it.

    Returns each row as a dict with the existing CSV columns.
    """
    csv_path = split_csv_path(split_dir)
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Split CSV has no data rows: {csv_path}")
    return rows


def filter_by_partition(rows: Sequence[Dict[str, str]], partition: str) -> List[Dict[str, str]]:
    """Return the rows for one partition, preserving order."""
    return [r for r in rows if r.get("partition") == partition]


def partitions_present(rows: Sequence[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    """Return a mapping from partition name to its rows."""
    out: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        out.setdefault(r.get("partition", ""), []).append(r)
    return out


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _resize_and_pad(image, target_width: int, target_height: int):  # -> Image.Image
    """Resize with aspect-ratio preservation and pad to exactly target size.

    No center crop and no random crop is used.
    """
    from PIL import Image

    src_w, src_h = image.size
    scale = min(target_width / src_w, target_height / src_h)
    new_w = max(1, round(src_w * scale))
    new_h = max(1, round(src_h * scale))

    resized = image.resize((new_w, new_h), Image.LANCZOS)
    padded = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_w) // 2
    paste_y = (target_height - new_h) // 2
    padded.paste(resized, (paste_x, paste_y))

    # Record the content box so callers can reason about padding if needed.
    padded.info["content_box"] = (paste_x, paste_y, paste_x + new_w, paste_y + new_h)
    return padded


def preprocessing_train(source_size: int = 224):
    """Deterministic training preprocessing factory.

    The baseline uses deterministic, aspect-ratio-preserving resize-and-pad
    preprocessing. No random crop and no center crop is applied. A later
    version may add controlled augmentation; this stays minimal for now.
    """
    from PIL import Image  # noqa: F401  (documents the expected input type)

    def _apply(image):
        return _resize_and_pad(image, source_size, source_size)

    return _apply


def preprocessing_inference(source_size: int = 224):
    """Deterministic preprocessing for validation and test partitions."""
    from PIL import Image  # noqa: F401

    def _apply(image):
        return _resize_and_pad(image, source_size, source_size)

    return _apply


def preprocessing_inference_factory(source_size: int = 224):
    """Alias of :func:`preprocessing_inference` for notebook-facing calls."""
    return preprocessing_inference(source_size)


def default_normalize() -> Tuple[List[float], List[float]]:
    """ImageNet statistics used for ResNet-50 preprocessing."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return mean, std


# ---------------------------------------------------------------------------
# Image decoding (delegates to the existing dataset package)
# ---------------------------------------------------------------------------

def load_rgb(path: Path):
    """Load an image and return an RGB PIL image.

    Thin wrapper around ``dataset.decoding.load_rgb`` so Colab-facing code uses
    the existing, already-tested decoder rather than a parallel implementation.
    """
    from dataset.decoding import load_rgb as _load_rgb

    return _load_rgb(path)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

#: torchvision's file name for ResNet50_Weights.IMAGENET1K_V1. Used only as a
#: fallback when torchvision is not importable; the live URL is preferred otherwise.
RESNET50_IMAGENET1K_V1_FILENAME = "resnet50-0676ba61.pth"


def weights_cache_path() -> Path:
    """Return the expected local cache path for the ImageNet ResNet-50 weights.

    Lookup only; nothing is downloaded here. Works without torchvision installed by
    falling back to the pinned weight file name.
    """
    import torch

    filename = RESNET50_IMAGENET1K_V1_FILENAME
    try:
        import torchvision.models as models

        filename = models.ResNet50_Weights.IMAGENET1K_V1.url.rsplit("/", 1)[-1]
    except Exception:
        pass  # torchvision absent or changed shape: use the pinned fallback name
    return Path(torch.hub.get_dir()) / "checkpoints" / filename


def pretrained_weights_available() -> bool:
    """Return True when the ImageNet ResNet-50 weights are already cached locally.

    Never raises: this is a guard helper, and a missing or unbrowsable cache simply
    means "not available yet".
    """
    try:
        exact = weights_cache_path()
        if exact.is_file():
            return True
        # Tolerate a differently named cached copy from another torchvision version.
        return any(exact.parent.glob("resnet50-*.pth"))
    except Exception:
        return False


def build_frozen_resnet50(
    num_classes: int = 5,
    *,
    allow_download: bool = False,
    pretrained: bool = True,
):
    """Build a ResNet-50 with frozen backbone and a new ``num_classes`` head.

    Requires ``torch`` and ``torchvision`` to be importable.

    Parameters
    ----------
    num_classes:
        Number of output logits for the new classification head.
    allow_download:
        Only meaningful when ``pretrained=True``. When False (the default), this
        raises a clear error if the ImageNet weights are not already present in the
        local torch cache, so the pipeline never silently starts a network download.
        Set True to permit torchvision to download the weights.
    pretrained:
        When False, build the identical architecture with ``weights=None``: no
        ImageNet weights are required and nothing is downloaded. Use this to
        validate the architecture/head contract, or to rebuild the net before
        restoring a trained checkpoint.
    """
    import torch
    import torchvision.models as models

    if pretrained:
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        if not allow_download and not pretrained_weights_available():
            raise FileNotFoundError(
                "ImageNet ResNet-50 weights are not cached locally at "
                f"{weights_cache_path()}. Pass allow_download=True to permit the "
                "download, or download them once and re-run."
            )
    else:
        weights = None

    model = models.resnet50(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)

    return model


def assert_head_is_trainable(model, head_name: str = "fc") -> bool:
    """Assert the classifier head is trainable and the backbone is frozen.

    Raises AssertionError with a clear message when the model does not match the
    expected frozen-backbone-plus-trainable-head contract.
    """
    import torch.nn as nn

    head = getattr(model, head_name, None)
    assert head is not None, f"Model does not have a {head_name!r} attribute."
    assert isinstance(head, nn.Linear), f"Model.{head_name!r} is not a Linear layer."

    backbone_trainable = [
        name
        for name, p in model.named_parameters()
        if not name.startswith(head_name) and p.requires_grad
    ]
    assert not backbone_trainable, (
        f"Expected backbone parameters to be frozen, but these are trainable: "
        f"{backbone_trainable[:5]}"
    )
    assert all(p.requires_grad for p in head.parameters()), (
        "Expected head parameters to be trainable."
    )
    return True


def inspect_model_structure(model) -> Dict[str, object]:
    """Return a small structural summary without running a forward pass."""
    head_name = "fc"
    trainable = []
    frozen = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable.append((name, tuple(param.shape)))
        else:
            frozen += 1
    return {
        "type": type(model).__name__,
        "head_name": head_name,
        "head_out_features": getattr(getattr(model, head_name, None), "out_features", None),
        "trainable_parameters": trainable,
        "frozen_parameter_count": frozen,
    }
