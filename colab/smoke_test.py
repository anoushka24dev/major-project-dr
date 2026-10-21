"""
Fast, offline smoke test for the Colab training pipeline.

Run it with pytest (the notebook does exactly this):

    python -m pytest colab/smoke_test.py -q

It must exit 0 before any training cell is allowed to run. It does NOT install
packages, download pretrained weights, start training, or modify any dataset,
split, or audit artifact.

Path resolution
---------------
- Split CSV: ``DR_SPLIT_DIR`` -> ``<repo>/provisional_primary_split`` ->
  ``/content/major_project/provisional_primary_split``.
- Image root: ``DR_SMOKE_IMAGE_ROOT`` -> ``DR_FUNDUS_ROOT`` -> Drive ->
  ``/content/major_project`` sibling folder.

Tests that need the image corpus are skipped (not silently passed) with a clear
reason when no corpus is reachable, so this file stays runnable on a bare
checkout without pretending to have verified anything.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
COLAB_DIR = REPO_ROOT / "colab"
if str(COLAB_DIR) not in sys.path:
    sys.path.insert(0, str(COLAB_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab_helpers import (  # noqa: E402
    BAND_ORDER,
    filter_by_partition,
    load_rgb,
    load_split_csv,
    preprocessing_inference,
)
from colab.train import (  # noqa: E402
    PrimarySplitDataset,
    compute_class_weights_from_rows,
    compute_metrics,
    confirm_guard_contract,
    initial_training_state,
    make_weighted_loss_fn,
    save_checkpoint,
    training_state_checksum,
)

DATASET_DIR_NAME = "Diabetic Retinopathy and Retinal Diseases Fundus I"
#: The documented entry point. ``train_on_colab.ipynb`` is the older, superseded
#: notebook; it is no longer what the project tells anyone to run.
NOTEBOOK_PATH = COLAB_DIR / "train_on_colab_fixed.ipynb"


# ---------------------------------------------------------------------------
# Environment discovery
# ---------------------------------------------------------------------------

def _candidate_split_dirs():
    env = os.environ.get("DR_SPLIT_DIR", "").strip()
    if env:
        yield Path(env)
    yield REPO_ROOT / "provisional_primary_split"
    yield Path("/content/major_project/provisional_primary_split")


def _candidate_image_roots():
    env = os.environ.get("DR_SMOKE_IMAGE_ROOT", "").strip()
    if env:
        yield Path(env)
    env = os.environ.get("DR_FUNDUS_ROOT", "").strip()
    if env:
        yield Path(env)
    mount = Path("/content/drive/MyDrive")
    if mount.exists():
        yield mount / DATASET_DIR_NAME
    yield REPO_ROOT.parent / DATASET_DIR_NAME
    yield Path("/content/major_project").parent / DATASET_DIR_NAME


def find_split_dir():
    for candidate in _candidate_split_dirs():
        if (candidate / "provisional_primary_split.csv").is_file():
            return candidate
    return None


def find_image_root():
    for candidate in _candidate_image_roots():
        if candidate.is_dir():
            return candidate
    return None


SPLIT_DIR = find_split_dir()
IMAGE_ROOT = find_image_root()

needs_split = pytest.mark.skipif(
    SPLIT_DIR is None,
    reason="No split directory reachable; set DR_SPLIT_DIR to run the split smoke checks.",
)
needs_images = pytest.mark.skipif(
    IMAGE_ROOT is None or SPLIT_DIR is None,
    reason="No image corpus reachable; set DR_SMOKE_IMAGE_ROOT (or DR_FUNDUS_ROOT) to run image checks.",
)


def _split_rows():
    assert SPLIT_DIR is not None
    return load_split_csv(SPLIT_DIR)


# ---------------------------------------------------------------------------
# Constants and split loading
# ---------------------------------------------------------------------------

def test_band_order_is_the_provisional_five_band_order():
    assert BAND_ORDER == [
        "No apparent DR",
        "Mild NPDR",
        "Moderate NPDR",
        "Severe NPDR",
        "PDR",
    ]


@needs_split
def test_split_csv_loads_and_partitions_are_exactly_the_expected_three():
    rows = _split_rows()
    assert len(rows) == 14201
    assert {r["partition"] for r in rows} == {"train", "validation", "test"}


@needs_split
def test_partitions_are_disjoint_and_cover_the_split():
    rows = _split_rows()
    train = {r["image_path"] for r in filter_by_partition(rows, "train")}
    val = {r["image_path"] for r in filter_by_partition(rows, "validation")}
    test = {r["image_path"] for r in filter_by_partition(rows, "test")}
    assert len(train) + len(val) + len(test) == len(rows)
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)


# ---------------------------------------------------------------------------
# Dataset package integration
# ---------------------------------------------------------------------------

def test_dataset_label_mapping_imports_and_matches_band_order():
    """The Colab band order must come from the existing label mapping module."""
    from dataset.labels import PRIMARY_MAPPING, primary_task_label

    assert set(PRIMARY_MAPPING.values()) == set(BAND_ORDER)
    assert primary_task_label("11.Normal Fundus") == "No apparent DR"
    assert primary_task_label("13. Very Severe NPDR") == "Severe NPDR"  # provisional merge
    assert primary_task_label("12. Advanced PDR") == "PDR"  # provisional merge


def test_dataset_image_decoding_is_the_one_the_pipeline_uses():
    """colab_helpers.load_rgb must delegate to dataset.decoding.load_rgb."""
    import dataset.decoding as decoding

    assert load_rgb.__module__ == "colab_helpers"
    assert hasattr(decoding, "load_rgb")


@needs_images
def test_real_image_decodes_through_the_dataset_package():
    rows = _split_rows()
    row = filter_by_partition(rows, "train")[0]
    image = load_rgb(IMAGE_ROOT / row["image_path"])
    assert image.mode == "RGB"
    assert image.size[0] > 0 and image.size[1] > 0


@needs_images
def test_dataset_item_is_normalized_tensor_with_correct_label():
    rows = _split_rows()
    train_rows = filter_by_partition(rows, "train")
    ds = PrimarySplitDataset(train_rows[:4], IMAGE_ROOT, image_size=64)
    tensor, label, row = ds[0]
    assert tuple(tensor.shape) == (3, 64, 64)
    assert tensor.dtype.is_floating_point
    assert -5.0 < float(tensor.min()) and float(tensor.max()) < 5.0  # normalized range
    assert BAND_ORDER[label] == row["primary_class"]


def test_preprocessing_is_resize_and_pad_without_crops():
    from PIL import Image

    preprocess = preprocessing_inference(224)
    out = preprocess(Image.new("RGB", (400, 300), color=(10, 20, 30)))
    assert out.size == (224, 224)
    # aspect ratio preserved: 400x300 content becomes 224x168, centred with padding
    content_box = out.info.get("content_box")
    assert content_box is not None
    width = content_box[2] - content_box[0]
    height = content_box[3] - content_box[1]
    assert (width, height) == (224, 168)
    assert abs((width / height) - (400 / 300)) < 0.01


# ---------------------------------------------------------------------------
# Class weights (training labels only)
# ---------------------------------------------------------------------------

@needs_split
def test_class_weights_are_derived_from_training_rows_only():
    from collections import Counter

    rows = _split_rows()
    train_rows = filter_by_partition(rows, "train")
    weights = compute_class_weights_from_rows(train_rows)

    assert set(weights) == set(BAND_ORDER)
    counts = Counter(r["primary_class"] for r in train_rows)
    n = sum(counts.values())
    for name in BAND_ORDER:
        assert weights[name] == pytest.approx(n / (len(BAND_ORDER) * counts[name]))
    # The rarest training band must carry the largest weight.
    assert weights["PDR"] > weights["Moderate NPDR"]


def test_class_weights_refuse_a_partition_missing_a_band():
    with pytest.raises(ValueError, match="no examples for class"):
        compute_class_weights_from_rows(
            [{"primary_class": "No apparent DR"}, {"primary_class": "No apparent DR"}]
        )


def test_weighted_loss_follows_the_provisional_band_order():
    import torch

    weights = {name: float(i + 1) for i, name in enumerate(BAND_ORDER)}
    loss_fn = make_weighted_loss_fn(weights)
    assert torch.equal(loss_fn.weight, torch.tensor([float(i + 1) for i in range(len(BAND_ORDER))]))


# ---------------------------------------------------------------------------
# Metrics contract
# ---------------------------------------------------------------------------

def test_metrics_report_accuracy_macros_per_class_and_confusion_matrix():
    import torch

    logits = torch.tensor([
        [2.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 4.0],
    ])
    labels = torch.tensor([0, 2, 4])
    m = compute_metrics(logits, labels)

    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "per_class",
                "confusion_matrix", "num_samples", "band_order"):
        assert key in m, key
    assert m["accuracy"] == pytest.approx(1.0)
    assert set(m["per_class"]) == set(BAND_ORDER)
    assert m["num_samples"] == 3
    assert len(m["confusion_matrix"]) == len(BAND_ORDER)
    assert all(len(row) == len(BAND_ORDER) for row in m["confusion_matrix"])
    assert all(isinstance(v, int) for row in m["confusion_matrix"] for v in row)
    assert m["per_class"]["Moderate NPDR"]["support"] == 1


# ---------------------------------------------------------------------------
# Checkpointing / output guards
# ---------------------------------------------------------------------------

def test_training_state_checksum_reports_best_state_presence():
    state = initial_training_state(
        epoch=2, best_val_loss=0.75, best_epoch=1, patience_counter=0, best_state={"dummy": True}
    )
    snap = training_state_checksum(state)
    assert snap["epoch"] == 2
    assert snap["best_epoch"] == 1
    assert snap["has_best_state"] is True
    assert training_state_checksum(initial_training_state())["has_best_state"] is False


def test_checkpoint_round_trip_writes_best_copy_and_compat_metadata(tmp_path):
    import torch

    from colab.train import checkpoint_compat_report

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    state = initial_training_state(epoch=0, best_val_loss=1.0)
    config = {
        "task": "primary_five_band_provisional",
        "model": "resnet50_frozen",
        "image_size": 224,
        "seed": 12345,
        "num_classes": len(BAND_ORDER),
        "class_weights": {name: 1.0 for name in BAND_ORDER},
    }

    saved = save_checkpoint(tmp_path, state, model, optimizer, config, epoch=0, is_best=True)
    assert (saved / "model.pt").is_file()
    assert (saved / "training_state.json").is_file()
    assert (tmp_path / "best_checkpoint" / "model.pt").is_file()

    report = checkpoint_compat_report(tmp_path / "best_checkpoint")
    assert report["compatible"] is True
    assert report["collab_compat"]["head_class_count"] == len(BAND_ORDER)
    assert report["collab_compat"]["preprocessing"]["kind"] == "resize_and_pad"


def test_overwrite_protection_refuses_an_existing_run(tmp_path):
    from colab.train import prepare_output_dir, write_run_meta

    out = prepare_output_dir(tmp_path / "run_a")
    write_run_meta(out, {"task": "primary_five_band_provisional"})

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_output_dir(tmp_path / "run_a")

    # Explicit overwrite is allowed and clears the previous run.
    out2 = prepare_output_dir(tmp_path / "run_a", overwrite=True)
    assert not (out2 / "run_meta.json").exists()


# ---------------------------------------------------------------------------
# Training guards
# ---------------------------------------------------------------------------

def test_training_is_disabled_without_an_explicit_opt_in(tmp_path):
    from colab.train import run_training

    with pytest.raises(PermissionError, match="disabled by default"):
        run_training(
            split_dir=REPO_ROOT / "provisional_primary_split",
            dataset_root=tmp_path,
            output_dir=tmp_path / "out",
        )


def test_training_refuses_before_the_smoke_test_passes(tmp_path):
    from colab.train import run_training

    with pytest.raises(PermissionError, match="smoke test has passed"):
        run_training(
            split_dir=REPO_ROOT / "provisional_primary_split",
            dataset_root=tmp_path,
            output_dir=tmp_path / "out",
            confirm_training=True,
            smoke_test_passed=False,
        )


def test_final_evaluation_is_disabled_until_explicitly_requested(tmp_path):
    from colab.train import run_final_evaluation

    with pytest.raises(PermissionError, match="disabled by default"):
        run_final_evaluation(
            split_dir=REPO_ROOT / "provisional_primary_split",
            dataset_root=tmp_path,
            output_dir=tmp_path / "out",
        )


def test_final_evaluation_requires_a_best_checkpoint(tmp_path):
    from colab.train import run_final_evaluation

    with pytest.raises(FileNotFoundError, match="Best checkpoint directory not found"):
        run_final_evaluation(
            split_dir=REPO_ROOT / "provisional_primary_split",
            dataset_root=tmp_path,
            output_dir=tmp_path / "out",
            confirm_final_evaluation=True,
        )


# ---------------------------------------------------------------------------
# Notebook structure
# ---------------------------------------------------------------------------

def test_notebook_exists_and_is_valid_json():
    import json

    assert NOTEBOOK_PATH.is_file()
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert nb["nbformat"] >= 4
    assert len(nb["cells"]) > 0


def test_notebook_keeps_the_smoke_and_training_guards():
    """The notebook still gates training, and now does it through preflight.

    The notebook is thin, so the smoke gate is no longer hand-written cell code: it
    lives in ``colab.pipeline.preflight``, which runs this very suite and fails on a
    non-zero exit. Assert both halves of that contract.
    """
    import inspect

    import colab.pipeline as pipeline

    text = NOTEBOOK_PATH.read_text(encoding="utf-8")
    assert "RUN_TRAINING = False" in text
    assert "RUN_FINAL_EVALUATION = False" in text
    assert "if not RUN_TRAINING:" in text
    assert "raise SystemExit" in text  # a failed preflight stops the cell
    assert "pipeline.preflight(" in text
    assert "run_smoke=True" in text

    # The gate itself: preflight runs the smoke suite and treats a failure as fatal.
    assert inspect.signature(pipeline.preflight).parameters["run_smoke"].default is True
    src = inspect.getsource(pipeline.preflight)
    assert "run_smoke_test(" in src
    assert 'smoke.get("status") == "failed"' in src
    assert '"smoke_test", "fail"' in src


def test_notebook_integrates_the_dataset_package_visibly():
    """The notebook's entry point reaches the dataset package, visibly.

    The chain is notebook -> ``colab.pipeline`` -> ``colab.train`` ->
    ``colab_helpers`` -> ``dataset.decoding``. Each link is checked, so moving the
    decoding path cannot silently orphan the dataset package again.
    """
    text = NOTEBOOK_PATH.read_text(encoding="utf-8")
    assert "colab.pipeline" in text

    pipeline_src = (COLAB_DIR / "pipeline.py").read_text(encoding="utf-8")
    train_src = (COLAB_DIR / "train.py").read_text(encoding="utf-8")
    helpers_src = (COLAB_DIR / "colab_helpers.py").read_text(encoding="utf-8")

    assert "colab.train" in pipeline_src
    assert "colab_helpers" in train_src
    assert "dataset.decoding" in helpers_src
    # And the decode really does come from the dataset package at runtime.
    from dataset import decoding

    assert callable(decoding.load_rgb)


def test_notebook_references_only_files_that_exist():
    import json

    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    text = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    referenced = {
        "colab.pipeline": COLAB_DIR / "pipeline.py",
        "colab.setup_helpers": COLAB_DIR / "setup_helpers.py",
    }
    for name, path in referenced.items():
        assert name in text, f"notebook should reference {name}"
        assert path.is_file(), f"{name} is referenced by the notebook but does not exist"

    # Everything the entry point needs at runtime must also be present.
    for path in (
        COLAB_DIR / "requirements-colab.txt",
        COLAB_DIR / "smoke_test.py",
        COLAB_DIR / "train.py",
        COLAB_DIR / "colab_helpers.py",
    ):
        assert path.is_file(), f"the pipeline needs {path.name} but it does not exist"


def test_confirm_guard_contract_matches_the_notebook_flags():
    """The library's guard contract and the notebook opt-in flags must agree."""
    contract = confirm_guard_contract()
    assert contract == {
        "training_requires_confirm": True,
        "training_requires_smoke_test": True,
        "final_evaluation_requires_confirm": True,
    }
