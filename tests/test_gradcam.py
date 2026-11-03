"""Tests for the Grad-CAM explainability stage: ``colab.gradcam``.

Contract pinned here:

* the CAM target is ResNet-50's ``layer4[-1]`` (the last spatially-resolved stage),
  not a surface Conv2d and not anything downstream of ``avgpool``;
* Grad-CAM never trains: the module contains no optimizer, no ``.step()`` call, and
  never writes a checkpoint;
* a CAM is a finite, non-negative map in [0, 1] with the same spatial size as the
  input, and it peaks where the explained logit actually depends on the input;
* example selection prefers confident matches and confident mistakes per band;
* the report carries the layer justification, the limitation list, and the run
  metadata, because those are what make the images defensible.

Nothing here downloads ImageNet weights: the architecture is always built with
``pretrained=False`` and a toy model is used for the gradient semantics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "colab"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

torch = pytest.importorskip("torch")

from colab import gradcam  # noqa: E402
from colab_helpers import BAND_ORDER  # noqa: E402

GRADCAM_PY = REPO_ROOT / "colab" / "gradcam.py"


# ---------------------------------------------------------------------------
# Toy model: logit for class 1 is the spatial mean of a positive 1x1 conv, so the
# CAM must peak wherever the input is bright.
# ---------------------------------------------------------------------------

class TinyNet(torch.nn.Module):
    def __init__(self, in_size: int = 16) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)
        self.fc = torch.nn.Linear(4, 2)
        with torch.no_grad():
            self.conv.weight.fill_(1.0)
            self.conv.bias.zero_()
            self.fc.weight[0].fill_(-1.0)
            self.fc.weight[1].fill_(1.0)
            self.fc.bias.zero_()

    def forward(self, x):
        act = torch.relu(self.conv(x))
        pooled = act.mean(dim=(2, 3))
        return self.fc(pooled)


def _blob_input(size: int = 16, top: int = 4, left: int = 5) -> torch.Tensor:
    x = torch.zeros(1, 3, size, size)
    x[:, :, top:top + 3, left:left + 3] = 1.0
    return x


# ---------------------------------------------------------------------------
# Layer selection
# ---------------------------------------------------------------------------

def test_resolve_cam_layer_picks_the_last_stage_of_resnet50():
    pytest.importorskip("torchvision")
    from colab_helpers import build_frozen_resnet50

    model = build_frozen_resnet50(num_classes=5, pretrained=False)
    name, layer = gradcam.resolve_cam_layer(model)

    assert name == "layer4.2"
    assert layer is model.layer4[-1]
    # The target must be upstream of avgpool, i.e. it still has a spatial axis.
    assert isinstance(layer, torch.nn.Module)
    assert layer is not model.avgpool
    assert layer is not model.fc


def test_resolve_cam_layer_accepts_an_explicit_layer():
    model = TinyNet()
    name, layer = gradcam.resolve_cam_layer(model, model.conv)
    assert name == "conv"
    assert layer is model.conv


def test_layer_facts_records_the_spatial_map():
    model = TinyNet()
    name, layer = gradcam.resolve_cam_layer(model, model.conv)
    facts = gradcam.layer_facts(model, name, layer, 224)
    assert facts["module_name"] == "conv"
    assert facts["out_channels"] == 4
    assert "7x7" in facts["spatial_map"]
    assert "avgpool" in facts["justification"]


# ---------------------------------------------------------------------------
# CAM semantics
# ---------------------------------------------------------------------------

def test_cam_is_bounded_and_sized_like_the_input():
    model = TinyNet()
    engine = gradcam.GradCAM(model, model.conv)
    cam = engine(_blob_input(16), [1])

    assert cam.shape == (1, 16, 16)
    assert np.isfinite(cam).all()
    assert cam.min() >= 0.0
    assert cam.max() == pytest.approx(1.0)


def test_cam_peaks_where_the_explained_logit_depends_on_the_input():
    model = TinyNet()
    engine = gradcam.GradCAM(model, model.conv)
    cam = engine(_blob_input(16, top=4, left=5), [1])[0]

    peak_row, peak_col = np.unravel_index(int(np.argmax(cam)), cam.shape)
    assert abs(int(peak_row) - 5) <= 1
    assert abs(int(peak_col) - 6) <= 1


def test_cam_rejects_a_batch_index_mismatch():
    model = TinyNet()
    engine = gradcam.GradCAM(model, model.conv)
    with pytest.raises(ValueError):
        engine(_blob_input(16), [1, 2])


def test_cam_requires_a_gradient_path():
    model = TinyNet()
    engine = gradcam.GradCAM(model, model.conv)
    # A no-grad forward inside the engine's hook cannot happen (it enables grad), but
    # a target layer that is not on the forward path must fail loudly instead of
    # returning a silent zero map.
    dead = torch.nn.Conv2d(3, 4, kernel_size=1)
    engine.model = model
    engine.target_layer = dead
    with pytest.raises(RuntimeError):
        engine(_blob_input(16), [1])


# ---------------------------------------------------------------------------
# Overlays and statistics
# ---------------------------------------------------------------------------

def test_overlay_preserves_size_and_writes_a_png(tmp_path):
    base = (np.random.RandomState(0).rand(16, 16, 3) * 255).astype(np.uint8)
    cam = np.zeros((16, 16), dtype=np.float32)
    cam[4, 4] = 1.0
    overlay = gradcam.render_overlay(base, cam, alpha=0.5)

    assert overlay.size == (16, 16)
    assert overlay.mode == "RGB"
    path = tmp_path / "overlay.png"
    overlay.save(path)
    assert path.is_file() and path.stat().st_size > 0


def test_overlay_rejects_a_shape_mismatch():
    base = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        gradcam.render_overlay(base, np.zeros((8, 8), dtype=np.float32))


def test_attention_stats_are_normalized_and_concentrated():
    cam = np.zeros((9, 9), dtype=np.float32)
    cam[4, 4] = 1.0  # exact centre of an odd-sized map
    mask = np.ones((9, 9), dtype=bool)
    stats = gradcam.attention_stats(cam, mask)

    assert stats["fundus_mass"] == pytest.approx(1.0)
    assert stats["top20_mass"] == pytest.approx(1.0)
    assert stats["centroid_radius"] == pytest.approx(0.0, abs=1e-6)
    assert stats["peak_x"] == pytest.approx(0.0, abs=1e-6)
    assert stats["peak_y"] == pytest.approx(0.0, abs=1e-6)


def test_attention_stats_flag_activation_on_padding():
    cam = np.zeros((10, 10), dtype=np.float32)
    cam[0, 0] = 1.0  # corner: the pad, not the fundus
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    stats = gradcam.attention_stats(cam, mask)

    assert stats["fundus_mass"] == pytest.approx(0.0)
    assert stats["centroid_radius"] == pytest.approx(1.0)


def test_fundus_mask_ignores_black_padding():
    base = np.zeros((10, 10, 3), dtype=np.uint8)
    base[2:8, 2:8] = 200
    mask = gradcam.fundus_mask(base)
    assert not mask[0, 0]
    assert mask[5, 5]


# ---------------------------------------------------------------------------
# Example selection
# ---------------------------------------------------------------------------

def _record(band: str, correct: bool, confidence: float, name: str) -> dict:
    pred = band if correct else BAND_ORDER[(BAND_ORDER.index(band) + 1) % len(BAND_ORDER)]
    return {
        "image_path": f"{name}.png",
        "true_index": BAND_ORDER.index(band),
        "true_class": band,
        "pred_index": BAND_ORDER.index(pred),
        "pred_class": pred,
        "prob_pred": confidence,
        "prob_true": confidence if correct else 1.0 - confidence,
        "correct": correct,
    }


def test_selection_prefers_confident_matches_and_confident_mistakes():
    records = [
        _record("Mild NPDR", True, 0.55, "mild_a"),
        _record("Mild NPDR", True, 0.91, "mild_b"),
        _record("Mild NPDR", False, 0.60, "mild_c"),
        _record("Mild NPDR", False, 0.88, "mild_d"),
        _record("PDR", True, 0.99, "pdr_a"),
    ]
    picks = gradcam.select_examples(records, per_class=1)

    assert [r["image_path"] for r in picks["correct"]["Mild NPDR"]] == ["mild_b.png"]
    assert [r["image_path"] for r in picks["incorrect"]["Mild NPDR"]] == ["mild_d.png"]
    assert [r["image_path"] for r in picks["correct"]["PDR"]] == ["pdr_a.png"]
    assert picks["incorrect"]["PDR"] == []
    # Bands with no scanned rows are present but empty, never missing keys.
    assert set(picks["correct"]) == set(BAND_ORDER)


def test_selection_respects_per_class_zero():
    records = [_record("PDR", True, 0.9, "pdr_a")]
    picks = gradcam.select_examples(records, per_class=0)
    assert picks["correct"]["PDR"] == []


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _entries():
    return [
        {
            "image_path": "a.png",
            "true_class": "Mild NPDR",
            "pred_class": "Moderate NPDR",
            "correct": False,
            "prob_true": 0.2,
            "prob_pred": 0.7,
            "cammed_class": "Moderate NPDR",
            "overlay_rel": "explainability/gradcam/validation/incorrect/mild_npdr/a.png",
            "stats": {
                "fundus_mass": 0.9,
                "top20_mass": 0.7,
                "centroid_x": 0.1,
                "centroid_y": 0.0,
                "centroid_radius": 0.1,
                "peak_x": 0.2,
                "peak_y": 0.0,
            },
        }
    ]


def test_report_contains_metadata_layer_and_caveats():
    entries = _entries()
    per_band = {
        band: {
            "correct": gradcam.aggregate_attention([e for e in entries if e["correct"] and e["true_class"] == band]),
            "incorrect": gradcam.aggregate_attention([e for e in entries if not e["correct"] and e["true_class"] == band]),
        }
        for band in BAND_ORDER
    }
    text = gradcam.build_report(
        run_dir=Path("cpu_runs/demo"),
        checkpoint_dir=Path("cpu_runs/demo/best_checkpoint"),
        compat={"checkpoint_epoch": 6, "best_epoch": 6, "best_val_loss": 0.573},
        meta={"config": {"image_size": 224}},
        layer_info={"module_name": "layer4.2", "module_type": "Bottleneck", "out_channels": 2048,
                    "spatial_map": "7x7 for a 224x224 input", "downsample_factor": 32,
                    "justification": "avgpool collapses space, so layer4[-1] is the last option."},
        partition="validation",
        rows_scanned=10,
        rows_total=100,
        scan_accuracy={"count": 10, "correct": 7, "accuracy": 0.7,
                       "per_band": {b: {"count": 2, "correct": 1, "accuracy": 0.5} for b in BAND_ORDER}},
        entries=entries,
        per_band=per_band,
        overlay_root=Path("explainability/gradcam/validation"),
        montage_paths={"Mild NPDR": "explainability/gradcam/validation/montage_mild_npdr.png"},
        image_size=224,
        weights_only=True,
        elapsed=1.5,
        per_class=2,
    )

    assert "Grad-CAM Attention Report" in text
    assert "layer4.2" in text
    assert "best validation loss" in text.lower()
    for band in BAND_ORDER:
        assert band in text
    assert "not lesion segmentation" in text
    assert "Limitations" in text
    assert "Mild NPDR -> Moderate NPDR" in text or "Moderate NPDR" in text


def test_aggregate_attention_handles_no_entries():
    assert gradcam.aggregate_attention([]) == {"count": 0}


# ---------------------------------------------------------------------------
# Preprocessing parity with the evaluated dataset
# ---------------------------------------------------------------------------

def test_model_input_matches_the_training_dataset_tensor(tmp_path):
    """The CAM input must be bit-identical to what evaluation fed the network."""
    pytest.importorskip("torchvision")
    from PIL import Image

    import colab.train as train_mod

    # A deliberately non-square image so resize-and-pad differs from a plain resize.
    src = tmp_path / "img.png"
    Image.fromarray(
        (np.random.RandomState(3).rand(90, 160, 3) * 255).astype(np.uint8), mode="RGB"
    ).save(src)

    row = {"image_path": "img.png", "primary_class": BAND_ORDER[2], "partition": "validation"}
    dataset = train_mod.PrimarySplitDataset([row], tmp_path, image_size=224)
    expected, label, _row = dataset[0]
    actual, base = gradcam.load_model_input(src, 224)

    assert label == 2
    assert base.size == (224, 224)
    assert actual.shape == expected.shape == (3, 224, 224)
    assert torch.allclose(actual, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Read-only contract
# ---------------------------------------------------------------------------

def test_module_never_trains_and_never_writes_checkpoints():
    source = GRADCAM_PY.read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert "torch.save" not in source
    assert "run_training" not in source
    assert "model.train()" not in source

    # Code-level check (not prose): no optimizer step and no parameter backward pass.
    # Gradients come from ``torch.autograd.grad`` against the activation tensor only.
    import ast

    called = {
        node.func.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "step" not in called
    assert "backward" not in called


def test_cli_refuses_the_test_partition_without_confirm(tmp_path):
    with pytest.raises(PermissionError):
        gradcam.run_gradcam(
            project_root=REPO_ROOT,
            output_dir=tmp_path,
            partition="test",
            confirm=False,
        )
