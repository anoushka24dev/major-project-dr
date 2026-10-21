"""Image decoding and validation for fundus images.

This module uses Pillow only. It does not normalize, resize, or otherwise
prepare images for a model. It is intended for deterministic loading and
validity checks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

from PIL import Image


class DecodeError(ValueError):
    """Raised when an image cannot be opened, verified, or converted to RGB."""


def load_rgb(path: Path) -> Image.Image:
    """Open an image, verify it, convert to RGB, and return the PIL image.

    Raises
    ------
    DecodeError
        If the file cannot be opened or decoded, or cannot be converted to RGB.
    """
    if not path.exists():
        raise DecodeError(f"Image does not exist: {path}")
    if not path.is_file():
        raise DecodeError(f"Path is not a file: {path}")

    try:
        with Image.open(path) as im:
            im.verify()
    except Exception as e:
        raise DecodeError(f"Image verification failed for {path}: {e}") from e

    try:
        with Image.open(path) as im:
            im.load()
            rgb = im.convert("RGB")
    except Exception as e:
        raise DecodeError(f"Image decode/convert failed for {path}: {e}") from e

    return rgb


def image_dimensions(im: Image.Image) -> Tuple[int, int]:
    """Return (width, height) for a PIL image."""
    return im.size  # (width, height)


def to_numpy_rgb(im: Image.Image):
    """Convert an RGB PIL image to a NumPy array with shape (H, W, 3) and dtype uint8."""
    import numpy as np

    arr = np.asarray(im)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise DecodeError(f"Expected RGB array with shape (H, W, 3), got {arr.shape}")
    if arr.dtype != np.uint8:
        raise DecodeError(f"Expected uint8 array, got dtype {arr.dtype}")
    return arr


def load_report(path: Path) -> dict:
    """Decode an image and return a small dict of metadata for inspection/tests."""
    im = load_rgb(path)
    width, height = image_dimensions(im)
    return {
        "path": str(path),
        "width": width,
        "height": height,
        "mode": im.mode,
        "channels": 3,
        "has_alpha": False,
    }
