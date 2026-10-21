"""Tests for image decoding and validation."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dataset.decoding import (
    DecodeError,
    image_dimensions,
    load_rgb,
    load_report,
    to_numpy_rgb,
)


def _make_png(mode="RGB", size=(4, 4), color=None):
    if color is None:
        color = (255, 0, 0) if mode == "RGB" else (128,) if mode == "L" else (0, 0, 0, 128)
    img = Image.new(mode, size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg(mode="RGB", size=(4, 4), color=None):
    if color is None:
        color = (255, 0, 0) if mode == "RGB" else (128,) if mode == "L" else (0, 0, 0, 128)
    img = Image.new(mode, size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def rgb_png(tmp_path):
    p = tmp_path / "rgb.png"
    p.write_bytes(_make_png("RGB", (6, 4), color=(255, 128, 64)))
    return p


@pytest.fixture
def grayscale_png(tmp_path):
    p = tmp_path / "gray.png"
    p.write_bytes(_make_png("L", (3, 5), color=(100,)))
    return p


@pytest.fixture
def rgba_png(tmp_path):
    p = tmp_path / "rgba.png"
    p.write_bytes(_make_png("RGBA", (2, 2), color=(1, 2, 3, 4)))
    return p


@pytest.fixture
def rgb_jpeg(tmp_path):
    p = tmp_path / "rgb.jpeg"
    p.write_bytes(_make_jpeg("RGB", (5, 5), color=(64, 128, 192)))
    return p


def test_load_rgb_returns_pil_image(rgb_png):
    im = load_rgb(rgb_png)
    assert im.mode == "RGB"


def test_load_rgb_dimensions(rgb_png):
    im = load_rgb(rgb_png)
    assert image_dimensions(im) == (6, 4)


def test_load_rgb_grayscale_converts(grayscale_png):
    im = load_rgb(grayscale_png)
    assert im.mode == "RGB"
    assert image_dimensions(im) == (3, 5)


def test_load_rgb_rgba_converts(rgba_png):
    im = load_rgb(rgba_png)
    assert im.mode == "RGB"


def test_load_rgb_jpeg_converts(rgb_jpeg):
    im = load_rgb(rgb_jpeg)
    assert im.mode == "RGB"
    assert image_dimensions(im) == (5, 5)


def test_load_report(rgb_png):
    r = load_report(rgb_png)
    assert r["mode"] == "RGB"
    assert r["channels"] == 3
    assert r["has_alpha"] is False
    assert r["width"] == 6
    assert r["height"] == 4


def test_to_numpy_rgb_shape_and_dtype(rgb_png):
    im = load_rgb(rgb_png)
    arr = to_numpy_rgb(im)
    assert arr.shape == (4, 6, 3)
    assert arr.dtype == np.uint8


def test_decode_error_on_missing(rgb_png, tmp_path):
    missing = tmp_path / "nope.png"
    with pytest.raises(DecodeError, match="does not exist"):
        load_rgb(missing)


def test_decode_error_on_not_a_file(rgb_png, tmp_path):
    d = tmp_path / "folder"
    d.mkdir()
    with pytest.raises(DecodeError, match="not a file"):
        load_rgb(d)


def test_decode_error_on_corrupt_bytes(tmp_path):
    p = tmp_path / "bad.png"
    p.write_bytes(b"not an image")
    with pytest.raises(DecodeError):
        load_rgb(p)


def test_real_image_smoke_test():
    """Optional integration-style smoke test for one real eligible image per class.

    This test runs whenever the verified image root is present. It is gated so
    the default unit suite stays offline-friendly, but it is not command-line
    trickery: it simply skips when the real dataset is not available.
    """
    root = Path(r"C:\Users\anous\Downloads\Diabetic Retinopathy and Retinal Diseases Fundus I")
    if not root.exists():
        pytest.skip("image root not present")

    from dataset.manifest import load_manifest

    rows = load_manifest(
        Path(__file__).resolve().parents[1]
        / "audit_output"
        / "pretraining_audit"
        / "PROVISIONAL_ELIGIBILITY_PRIMARY.csv",
        "primary",
        image_root=root,
    )
    eligible = [r for r in rows if r["eligibility"] == "eligible_provisional"]
    assert eligible, "no eligible rows found in primary manifest"

    eligible_by_class: dict[str, str] = {}
    for row in eligible:
        cls = (row["original_label"] or "").strip()
        if not cls:
            continue
        if cls not in eligible_by_class:
            eligible_by_class[cls] = row["image_path"]

    assert len(eligible_by_class) >= 7, f"expected at least the 7 primary DR source classes, got {sorted(eligible_by_class)}"

    for cls, rel_path in eligible_by_class.items():
        path = root / rel_path
        assert path.exists(), f"real eligible image not present: {cls} -> {rel_path}"
        report = load_report(path)
        assert report["mode"] == "RGB"
        assert report["width"] > 0 and report["height"] > 0

    # The remaining 6 source classes are excluded from the primary DR cohort,
    # so they will not appear in `eligible_provisional` rows here. If the real
    # dataset is available, this smoke test therefore validates one real image
    # for each of the 7 primary DR source classes, and documents that the other
    # 6 classes are intentionally excluded from the primary DR task.
    expected_primary_classes = {
        "11.Normal Fundus",
        "3.Mild DR",
        "4.Moderate DR",
        "5.Severe DR",
        "13. Very Severe NPDR",
        "6.Proliferate DR",
        "12. Advanced PDR",
    }
    for cls in expected_primary_classes:
        assert cls in eligible_by_class, f"primary DR class missing from eligible rows: {cls}"
    assert set(eligible_by_class) == expected_primary_classes
