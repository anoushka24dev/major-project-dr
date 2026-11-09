"""Tests for optional test-time augmentation (TTA).

Only synthetic tensors and a tiny linear model are used: no dataset, no checkpoint
and no training. The tests pin down the two properties the feature must have --
class probabilities are averaged (never labels) and the five-class output shape and
order are preserved -- plus the rule that ordinary inference stays available and is
the default until a validation comparison selects TTA.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import Config  # noqa: E402
from src.tta import (  # noqa: E402
    DEFAULT_VIEWS,
    VIEW_NAMES,
    TTAError,
    build_views,
    mean_probabilities,
    options_from_config,
    predict,
    save_selection,
    selected_view_names,
    selection_path,
)

NUM_CLASSES = 5


class CountingModel(nn.Module):
    """Tiny stand-in for the classifier that counts forward passes."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 8 * 8, num_classes)
        self.calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.fc(inputs.reshape(inputs.shape[0], -1))


class SyntheticDataset(Dataset):
    def __init__(self, count: int = 6, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.rand(count, 3, 8, 8, generator=generator)
        self.labels = torch.arange(count) % NUM_CLASSES

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            self.images[index],
            int(self.labels[index]),
            {"image_path": f"synthetic_{index}.png", "source_dataset": "synthetic"},
        )


def _loader(count: int = 6, batch_size: int = 2) -> DataLoader:
    return DataLoader(SyntheticDataset(count), batch_size=batch_size, shuffle=False)


def _cfg(tmp_path: Path, **tta_overrides) -> Config:
    section = {
        "enabled": True,
        "compare_on_validation": True,
        "reduction": "mean",
        "selection_metric": "macro_f1",
        "include_original": True,
        "transforms": list(DEFAULT_VIEWS),
    }
    section.update(tta_overrides)
    return Config(
        {
            "paths": {"outputs_dir": str(tmp_path / "outputs")},
            "inference": {"tta": section},
        },
        root=tmp_path,
        source=tmp_path / "config.yaml",
    )


# ---------------------------------------------------------------------------
# Probability averaging
# ---------------------------------------------------------------------------

def test_mean_probabilities_is_the_arithmetic_mean():
    first = np.array([[0.9, 0.1, 0.0, 0.0, 0.0]])
    second = np.array([[0.2, 0.8, 0.0, 0.0, 0.0]])

    averaged = mean_probabilities([first, second])

    assert averaged.shape == (1, NUM_CLASSES)
    assert averaged[0, 0] == pytest.approx(0.55)
    assert averaged[0, 1] == pytest.approx(0.45)
    assert averaged.sum(axis=1) == pytest.approx([1.0])


def test_probabilities_are_averaged_not_voted_as_labels():
    """A label vote would produce one-hot entries; a probability mean cannot."""
    views = [
        np.array([[0.7, 0.1, 0.1, 0.05, 0.05]]),
        np.array([[0.1, 0.7, 0.1, 0.05, 0.05]]),
        np.array([[0.1, 0.1, 0.7, 0.05, 0.05]]),
    ]

    averaged = mean_probabilities(views)

    assert averaged[0, :3] == pytest.approx([0.3, 0.3, 0.3])
    # No entry is 0 or 1, so this is not an averaged set of one-hot decisions.
    assert np.all(averaged > 0.0)
    assert np.all(averaged < 1.0)
    # The average of the three one-hot votes would have been [1/3, 1/3, 1/3, 0, 0].
    assert averaged[0, 3] > 0.0 and averaged[0, 4] > 0.0


def test_mean_probabilities_preserves_class_order():
    per_class = np.eye(NUM_CLASSES)
    averaged = mean_probabilities([per_class, per_class])
    np.testing.assert_allclose(averaged, per_class, atol=1e-12)
    # Class index 3 stays class index 3: the order is never permuted.
    assert int(np.argmax(averaged[3])) == 3


@pytest.mark.parametrize(
    "views",
    [
        [],
        [np.zeros((2, 5))],
        [np.zeros((2, 5)), np.zeros((3, 5))],
        [np.zeros((2, NUM_CLASSES)), np.full((2, NUM_CLASSES), np.nan)],
        [np.full((2, NUM_CLASSES), -0.5)],
    ],
)
def test_mean_probabilities_rejects_unusable_input(views):
    with pytest.raises(TTAError):
        mean_probabilities(views)


def test_mean_probabilities_rejects_the_wrong_number_of_classes():
    with pytest.raises(TTAError):
        mean_probabilities([np.full((1, 4), 0.25)], num_classes=NUM_CLASSES)


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def test_default_views_are_medically_defensible():
    assert "vflip" not in VIEW_NAMES
    assert "vertical_flip" not in VIEW_NAMES
    assert "random_crop" not in VIEW_NAMES
    assert set(DEFAULT_VIEWS) <= set(VIEW_NAMES)
    # The mirror, the two small rotations and the two exposure changes.
    assert "hflip" in DEFAULT_VIEWS
    assert {"rotate_p10", "rotate_m10"} <= set(DEFAULT_VIEWS)
    assert {"brightness_p10", "brightness_m10"} <= set(DEFAULT_VIEWS)


def test_each_view_keeps_the_image_shape():
    batch = torch.rand(2, 3, 32, 32)
    for name, function in build_views(VIEW_NAMES, include_original=False):
        transformed = function(batch.clone())
        assert transformed.shape == batch.shape, name
        assert torch.isfinite(transformed).all(), name


def test_hflip_mirrors_and_brightness_changes_exposure():
    batch = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4).repeat(1, 3, 1, 1)
    views = dict(build_views("hflip", include_original=False))

    flipped = views["hflip"](batch)
    assert torch.equal(flipped, torch.flip(batch, dims=[-1]))

    brighter = dict(build_views("brightness_p10", include_original=False))["brightness_p10"](batch)
    darker = dict(build_views("brightness_m10", include_original=False))["brightness_m10"](batch)
    assert brighter.mean() > batch.mean() > darker.mean()


def test_unknown_view_is_rejected():
    with pytest.raises(TTAError):
        build_views(["not_a_view"], include_original=True)


# ---------------------------------------------------------------------------
# Inference: shape, ordinary default, forward-pass accounting
# ---------------------------------------------------------------------------

def test_tta_output_keeps_the_five_class_shape_and_probability_semantics():
    model = CountingModel()
    loader = _loader(count=6, batch_size=2)
    result = predict(
        model,
        loader,
        "cpu",
        views=build_views(DEFAULT_VIEWS, include_original=True),
        num_classes=NUM_CLASSES,
    )

    probabilities = result["probabilities"]
    assert probabilities.shape == (6, NUM_CLASSES)
    assert result["logits"].shape == (6, NUM_CLASSES)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(6), atol=1e-6)
    assert (probabilities >= 0.0).all()
    assert result["mode"] == "tta"
    assert result["views"][0] == "identity"
    assert set(result["views"]) == {"identity", *DEFAULT_VIEWS}
    # Labels and metadata are untouched by augmentation.
    assert list(result["labels"]) == list(SyntheticDataset(6).labels.numpy())
    assert result["meta"][0]["image_path"] == "synthetic_0.png"


def test_ordinary_inference_is_a_single_forward_pass_per_batch():
    model = CountingModel()
    result = predict(model, _loader(count=6, batch_size=2), "cpu", num_classes=NUM_CLASSES)

    assert model.calls == 3  # six samples, batch size two
    assert result["mode"] == "ordinary"
    assert result["views"] == ["identity"]
    assert result["forward_passes"] == 3
    assert result["probabilities"].shape == (6, NUM_CLASSES)


def test_tta_runs_one_pass_per_view_per_batch():
    model = CountingModel()
    views = build_views(DEFAULT_VIEWS, include_original=True)
    result = predict(model, _loader(count=6, batch_size=2), "cpu", views=views, num_classes=NUM_CLASSES)

    assert model.calls == 3 * len(views)
    assert result["forward_passes"] == 3 * len(views)


def test_tta_logits_come_from_the_unmodified_image():
    """Temperature scaling must stay a property of the model, not of the views."""
    model = CountingModel()
    ordinary = predict(model, _loader(count=4, batch_size=4), "cpu", num_classes=NUM_CLASSES)
    augmented = predict(
        model,
        _loader(count=4, batch_size=4),
        "cpu",
        views=build_views(DEFAULT_VIEWS, include_original=True),
        num_classes=NUM_CLASSES,
    )
    np.testing.assert_allclose(augmented["logits"], ordinary["logits"], atol=1e-6)


# ---------------------------------------------------------------------------
# Configuration and the saved selection
# ---------------------------------------------------------------------------

def test_options_default_to_disabled(tmp_path):
    """A missing inference.tta section must never switch TTA on."""
    cfg = Config({"paths": {"outputs_dir": str(tmp_path / "out")}}, root=tmp_path, source=tmp_path / "c.yaml")
    assert options_from_config(cfg).enabled is False
    assert selected_view_names(cfg) is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"reduction": "vote"},
        {"selection_metric": "auc"},
        {"transforms": ["random_crop"]},
    ],
)
def test_unsupported_tta_configuration_is_rejected(tmp_path, overrides):
    with pytest.raises(TTAError):
        options_from_config(_cfg(tmp_path, **overrides))


def test_saved_selection_controls_the_evaluation_views(tmp_path):
    cfg = _cfg(tmp_path)
    assert selected_view_names(cfg) is None  # no comparison has run yet

    path = save_selection(
        cfg,
        {"selected": "tta", "views": list(DEFAULT_VIEWS), "selection_partition": "validation"},
    )
    assert path == selection_path(cfg)
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["selection_partition"] == "validation"

    views = selected_view_names(cfg)
    assert views[0] == "identity"
    assert set(views) == {"identity", *DEFAULT_VIEWS}

    # Selecting ordinary inference switches TTA off again.
    save_selection(cfg, {"selected": "ordinary", "views": list(DEFAULT_VIEWS)})
    assert selected_view_names(cfg) is None


def test_disabled_tta_ignores_a_saved_selection(tmp_path):
    cfg = _cfg(tmp_path, enabled=False)
    save_selection(cfg, {"selected": "tta", "views": list(DEFAULT_VIEWS)})
    assert options_from_config(cfg).enabled is False
    assert selected_view_names(cfg) is None
