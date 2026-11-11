"""Datasets and preprocessing for classification and lesion segmentation.

Preprocessing contract
----------------------
Classification uses the repository's existing deterministic, aspect-ratio
preserving resize-and-pad followed by ImageNet normalisation, imported from
``colab/colab_helpers.py`` through :mod:`src.legacy_bridge`. Using the *same*
function as the legacy baseline guarantees that the two paths preprocess
identically, and that no centre/random crop can ever remove a border lesion.

Augmentation is applied **only** to the training partition and is deliberately
mild: horizontal flip (anatomically admissible for fundus laterality), a small
rotation, and small brightness/contrast jitter. Vertical flips and random crops
are disabled in configuration because they can change clinically meaningful
lesion appearance or remove lesions entirely.

Segmentation resizes images with bilinear interpolation and masks with nearest
neighbour, so mask values stay binary and stay aligned with the image.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import Config
from .legacy_bridge import load_colab_helpers

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# ---------------------------------------------------------------------------
# Preprocessing primitives
# ---------------------------------------------------------------------------

def _resize_and_pad(image: Image.Image, size: int, root: Path) -> Image.Image:
    """Deterministic resize-and-pad, delegated to the existing implementation."""
    helpers = load_colab_helpers(root)
    return helpers._resize_and_pad(image, size, size)


def to_tensor(image: Image.Image) -> torch.Tensor:
    """RGB PIL image -> float32 CHW tensor in [0, 1]."""
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


def normalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(-1, 1, 1)
    return (tensor - mean) / std


def load_rgb(path: Path) -> Image.Image:
    """Decode an image to RGB, using the repository's tested decoder."""
    from dataset.decoding import load_rgb as _load_rgb

    return _load_rgb(Path(path))


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------

def load_manifest_rows(path: Path) -> List[Dict[str, str]]:
    """Load a manifest CSV into typed-ish dicts."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    for row in rows:
        row["mapped_label_int"] = int(row["mapped_label"]) if row.get("mapped_label") else -1
    return rows


def resolve_row_path(row: Dict[str, str], roots: Dict[str, Path]) -> Optional[Path]:
    """Resolve a manifest row to an existing file."""
    stored = row.get("image_path") or ""
    candidate = Path(stored)
    if candidate.is_file():
        return candidate
    rel = row.get("rel_path") or ""
    root = roots.get(str(row.get("source_dataset", "")))
    if root is not None and rel:
        alternative = Path(root) / rel
        if alternative.is_file():
            return alternative
    return None


def source_roots(cfg: Config) -> Dict[str, Path]:
    """Map source dataset name -> root used to resolve relative paths."""
    roots: Dict[str, Path] = {}
    fundus = cfg.fundus_root()
    mendeley = fundus if (fundus and fundus.is_dir()) else cfg.path("paths.mendeley_dir")
    roots["mendeley"] = mendeley
    roots["mmrdr"] = cfg.path("paths.mmrdr_dir")
    roots["aptos2019"] = cfg.root / "datasets" / "aptos2019"
    return roots


# ---------------------------------------------------------------------------
# Classification dataset
# ---------------------------------------------------------------------------

class ClassifierDataset(Dataset):
    """One partition of a unified manifest."""

    def __init__(
        self,
        rows: Sequence[Dict[str, str]],
        cfg: Config,
        *,
        train: bool = False,
        image_size: Optional[int] = None,
    ) -> None:
        if not rows:
            raise ValueError("ClassifierDataset requires at least one row")
        self.rows = list(rows)
        self.cfg = cfg
        self.root = cfg.root
        self.image_size = int(image_size or cfg.require("preprocessing.classification.image_size"))
        self.train = bool(train)
        self.roots = source_roots(cfg)

        aug = cfg.get("preprocessing.classification.augmentation", {}) or {}
        self.hflip = bool(aug.get("horizontal_flip", True)) and self.train
        self.rotation = float(aug.get("rotation_degrees", 0) or 0) if self.train else 0.0
        self.brightness = float(aug.get("brightness", 0) or 0) if self.train else 0.0
        self.contrast = float(aug.get("contrast", 0) or 0) if self.train else 0.0
        if bool(aug.get("random_crop", False)):
            raise ValueError(
                "preprocessing.classification.augmentation.random_crop is enabled but random "
                "cropping is not implemented because it can remove retinal lesions. "
                "Set it to false."
            )
        if bool(aug.get("vertical_flip", False)):
            raise ValueError(
                "preprocessing.classification.augmentation.vertical_flip is enabled but the "
                "pipeline does not implement it: it is not an anatomically neutral transform "
                "for fundus images. Set it to false."
            )
        self._rng = np.random.default_rng(cfg.seed)

    def __len__(self) -> int:
        return len(self.rows)

    def _augment(self, image: Image.Image) -> Image.Image:
        import torchvision.transforms.functional as TF

        if self.hflip and self._rng.random() < 0.5:
            image = TF.hflip(image)
        if self.rotation > 0:
            angle = float(self._rng.uniform(-self.rotation, self.rotation))
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        if self.brightness > 0:
            factor = float(1.0 + self._rng.uniform(-self.brightness, self.brightness))
            image = TF.adjust_brightness(image, factor)
        if self.contrast > 0:
            factor = float(1.0 + self._rng.uniform(-self.contrast, self.contrast))
            image = TF.adjust_contrast(image, factor)
        return image

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        row = self.rows[index]
        path = resolve_row_path(row, self.roots)
        if path is None:
            raise FileNotFoundError(
                f"Manifest row points to a missing image: {row.get('image_path')!r} "
                f"(source {row.get('source_dataset')!r})"
            )
        image = load_rgb(path)
        if self.train:
            image = self._augment(image)
        image = _resize_and_pad(image, self.image_size, self.root)
        tensor = normalize(to_tensor(image))

        meta = {
            "image_path": str(path),
            "source_dataset": row.get("source_dataset", ""),
            "original_label": row.get("original_label", ""),
            "class_name": row.get("class_name", ""),
            "split": row.get("split", ""),
            "image_hash": row.get("image_hash", ""),
        }
        return tensor, int(row["mapped_label_int"]), meta


def class_weights_from_rows(rows: Sequence[Dict[str, str]], num_classes: int) -> np.ndarray:
    """Inverse-frequency class weights computed from the given rows only.

    Computed from the training partition, never from validation or test data.
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    for row in rows:
        label = int(row.get("mapped_label_int", -1))
        if 0 <= label < num_classes:
            counts[label] += 1
    present = counts > 0
    weights = np.zeros(num_classes, dtype=np.float64)
    if present.sum() == 0:
        return np.ones(num_classes, dtype=np.float64)
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    weights[~present] = 1.0
    return weights


# ---------------------------------------------------------------------------
# IDRiD segmentation pairs
# ---------------------------------------------------------------------------

LESION_ORDER = ("MA", "HE", "EX", "SE")
_LESION_FOLDER_HINTS = {
    "MA": ("1. Microaneurysms", "microaneurysm", "ma"),
    "HE": ("2. Haemorrhages", "hemorrhage", "haemorrhage", "he"),
    "EX": ("3. Hard Exudates", "hard exudate", "ex"),
    "SE": ("4. Soft Exudates", "soft exudate", "se"),
}


def _files_with_suffix(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def _find_lesion_dir(root: Path, channel: str) -> Optional[Path]:
    """Locate one lesion annotation directory by its published name."""
    hints = _LESION_FOLDER_HINTS[channel]
    for hint in hints:
        exact = [p for p in root.rglob(hints[0]) if p.is_dir()]
        if exact:
            return sorted(exact)[0]
    for hint in hints:
        for path in sorted(root.rglob("*")):
            if path.is_dir() and hint.lower() in path.name.lower():
                if _files_with_suffix(path):
                    return path
    return None


def _mask_stem(path: Path, channel: str) -> str:
    stem = path.stem
    suffix = f"_{channel}"
    if stem.upper().endswith(suffix.upper()):
        return stem[: -len(suffix)]
    return stem


def build_idrid_pairs(cfg: Config, logger) -> Dict[str, Any]:
    """Pair IDRiD images with their four lesion masks and validate the pairing.

    Returns ``{"available": bool, "pairs": [...], "reason": str, ...}``. When any
    image lacks a mask (or vice versa) the pairing is rejected: the pipeline will
    not train a segmented model on silently mismatched annotations.
    """
    root = cfg.idrid_root()
    info: Dict[str, Any] = {"available": False, "root": str(root), "pairs": []}
    if not root.is_dir():
        info["reason"] = f"IDRiD root not found: {root}"
        return info

    images_dir: Optional[Path] = None
    for candidate in sorted(root.rglob("a. Training Set")):
        if candidate.is_dir() and _files_with_suffix(candidate):
            images_dir = candidate
            break
    if images_dir is None:
        info["reason"] = (
            "could not locate the IDRiD training images folder "
            "('1. Original Images/a. Training Set')"
        )
        return info

    lesion_dirs: Dict[str, Optional[Path]] = {}
    for channel in LESION_ORDER:
        lesion_dirs[channel] = _find_lesion_dir(root, channel)
        if lesion_dirs[channel] is None:
            info["reason"] = f"missing lesion annotation folder for channel {channel}"
            info["lesion_dirs"] = {k: (str(v) if v else None) for k, v in lesion_dirs.items()}
            return info

    images = {p.stem: p for p in _files_with_suffix(images_dir)}
    masks: Dict[str, Dict[str, Path]] = {channel: {} for channel in LESION_ORDER}
    for channel in LESION_ORDER:
        for path in _files_with_suffix(lesion_dirs[channel]):
            masks[channel][_mask_stem(path, channel)] = path

    common = set(images)
    for channel in LESION_ORDER:
        common &= set(masks[channel])
    missing_masks = sorted(set(images) - common)
    orphan_masks = {
        channel: sorted(set(masks[channel]) - set(images)) for channel in LESION_ORDER
    }

    pairs = [
        {
            "image_id": stem,
            "image_path": str(images[stem]),
            **{f"mask_{channel}": str(masks[channel][stem]) for channel in LESION_ORDER},
        }
        for stem in sorted(common)
    ]

    info.update(
        {
            "available": bool(pairs),
            "pairs": pairs,
            "images_dir": str(images_dir),
            "lesion_dirs": {k: str(v) for k, v in lesion_dirs.items()},
            "image_count": len(images),
            "paired_count": len(pairs),
            "images_without_all_masks": len(missing_masks),
            "images_without_all_masks_examples": missing_masks[:10],
            "orphan_masks": {k: len(v) for k, v in orphan_masks.items()},
        }
    )
    if not pairs:
        info["reason"] = "no image had a complete set of four lesion masks"
        return info

    manifest_dir = cfg.outputs_dir("manifests")
    pairs_csv = manifest_dir / "idrid_pairs.csv"
    with pairs_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["image_id", "image_path", "mask_MA", "mask_HE", "mask_EX", "mask_SE"]
        )
        writer.writeheader()
        writer.writerows(pairs)
    info["pairs_csv"] = str(pairs_csv)

    if missing_masks:
        logger.warning(
            "%d IDRiD image(s) lack at least one lesion mask and were dropped from segmentation",
            len(missing_masks),
        )
    logger.info("IDRiD: %d fully paired image/mask sets", len(pairs))
    return info


def split_idrid_pairs(
    pairs: Sequence[Dict[str, str]],
    ratios: Dict[str, float],
    seed: int,
) -> Dict[str, List[Dict[str, str]]]:
    """Deterministic disjoint split of the IDRiD pairs (no patient IDs published)."""
    import random

    ordered = list(pairs)
    random.Random(seed).shuffle(ordered)
    total = len(ordered)
    n_train = max(1, int(round(total * float(ratios["train"]))))
    n_val = max(1, int(round(total * float(ratios["validation"]))))
    if n_train + n_val >= total:
        n_val = max(1, total - n_train - 1)
    n_train = max(1, min(n_train, total - 2))
    n_val = max(1, min(n_val, total - n_train - 1))
    return {
        "train": ordered[:n_train],
        "validation": ordered[n_train:n_train + n_val],
        "test": ordered[n_train + n_val:],
    }


def resize_and_pad_mask(mask_image: Image.Image, target_size: int) -> np.ndarray:
    """Decode a binary lesion mask and align it with ``_resize_and_pad`` geometry.

    Converts palette/colour masks (including IDRiD's ``P``-mode red annotations
    whose luminance drops below 128 under ``convert("L")``) via max-channel
    intensity before nearest-neighbour scaling and centered padding.
    """
    if mask_image.mode in ("P", "PA", "RGB", "RGBA", "CMYK", "YCbCr"):
        array = np.asarray(mask_image.convert("RGB"))[..., :3].max(axis=-1)
    else:
        array = np.asarray(mask_image)
    threshold = int(array.max()) // 2
    canonical = Image.fromarray(((array > threshold).astype(np.uint8) * 255), mode="L")
    width, height = canonical.size
    scale = min(target_size / width, target_size / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    resized = canonical.resize((new_width, new_height), Image.NEAREST)
    canvas = Image.new("L", (target_size, target_size), 0)
    canvas.paste(resized, ((target_size - new_width) // 2, (target_size - new_height) // 2))
    return np.asarray(canvas) > 127


class SegmentationDataset(Dataset):
    """IDRiD image + four binary lesion masks."""

    def __init__(
        self,
        pairs: Sequence[Dict[str, str]],
        cfg: Config,
        *,
        image_size: Optional[int] = None,
    ) -> None:
        if not pairs:
            raise ValueError("SegmentationDataset requires at least one image/mask pair")
        self.pairs = list(pairs)
        self.cfg = cfg
        self.root = cfg.root
        self.image_size = int(image_size or cfg.require("preprocessing.segmentation.image_size"))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, str]]:
        pair = self.pairs[index]
        image = load_rgb(Path(pair["image_path"]))
        base = _resize_and_pad(image, self.image_size, self.root)
        tensor = normalize(to_tensor(base))

        masks = np.zeros((len(LESION_ORDER), self.image_size, self.image_size), dtype=np.float32)
        for channel_index, channel in enumerate(LESION_ORDER):
            mask_path = Path(pair[f"mask_{channel}"])
            with Image.open(mask_path) as mask_image:
                masks[channel_index] = resize_and_pad_mask(mask_image, self.image_size).astype(np.float32)

        meta = {"image_id": pair["image_id"], "image_path": pair["image_path"]}
        return tensor, torch.from_numpy(masks), meta


def write_history_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> Path:
    """Write a per-epoch training history CSV."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return target
    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return target
