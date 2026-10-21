"""
Tests for the guarded Colab training pipeline and its Colab-facing helpers.

These tests validate the training contract without modifying the existing dataset
or split artifacts, without installing packages, without downloading pretrained
weights, and without running a real training run.

Image-dependent checks sample a handful of images rather than decoding a whole
partition, so the suite stays fast enough to run alongside the rest of the repo
tests. If no image corpus is reachable, those specific checks are skipped with an
explicit reason instead of being silently weakened.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "colab"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from colab_helpers import (  # noqa: E402
    BAND_ORDER,
    filter_by_partition,
    load_rgb,
    load_split_csv,
    preprocessing_inference,
)
from colab.train import (  # noqa: E402
    PrimarySplitDataset,
    checkpoint_compat_report,
    compute_class_weights_from_rows,
    compute_metrics,
    confirm_guard_contract,
    device_report,
    initial_training_state,
    inspect_checkpoint_metadata,
    latest_epoch_checkpoint,
    load_training_state,
    make_weighted_loss_fn,
    mark_incomplete,
    prepare_output_dir,
    run_final_evaluation,
    run_training,
    save_checkpoint,
    save_run_summary,
    train_epoch,
    evaluate_epoch,
    update_best_tracking,
    write_run_manifest,
    write_run_meta,
)
from dataset.labels import PRIMARY_MAPPING, is_non_dr_source, primary_task_label  # noqa: E402

AUDIT_DIR = REPO_ROOT / "audit_output" / "pretraining_audit"
PRIMARY_CSV = AUDIT_DIR / "PROVISIONAL_ELIGIBILITY_PRIMARY.csv"
SPLIT_DIR = REPO_ROOT / "provisional_primary_split"
SPLIT_CSV = SPLIT_DIR / "provisional_primary_split.csv"
#: The documented entry point (``train_on_colab.ipynb`` is the superseded notebook).
NOTEBOOK_PATH = REPO_ROOT / "colab" / "train_on_colab_fixed.ipynb"

DATASET_DIR_NAME = "Diabetic Retinopathy and Retinal Diseases Fundus I"


def _find_image_root():
    env = os.environ.get("DR_SMOKE_IMAGE_ROOT", "").strip()
    if env and Path(env).is_dir():
        return Path(env)
    env = os.environ.get("DR_FUNDUS_ROOT", "").strip()
    if env and Path(env).is_dir():
        return Path(env)
    for candidate in (
        Path(r"C:\Users\anous\Downloads") / DATASET_DIR_NAME,
        REPO_ROOT.parent / DATASET_DIR_NAME,
        Path("/content/major_project").parent / DATASET_DIR_NAME,
        Path("/content/drive/MyDrive") / DATASET_DIR_NAME,
    ):
        if candidate.is_dir():
            return candidate
    return None


IMAGE_ROOT = _find_image_root()

needs_images = pytest.mark.skipif(
    IMAGE_ROOT is None,
    reason="No image corpus reachable; set DR_SMOKE_IMAGE_ROOT to run image-dependent checks.",
)

#: Small, deterministic sample of rows used by image-dependent checks.
SAMPLE_ROWS_PER_PARTITION = 3


def _rows():
    return load_split_csv(SPLIT_DIR)


# ---------------------------------------------------------------------------
# Split loading and partition filtering
# ---------------------------------------------------------------------------

def test_split_csv_exists_and_loads():
    assert SPLIT_CSV.is_file()
    rows = _rows()
    assert len(rows) == 14201


def test_split_has_exactly_the_expected_partitions():
    rows = _rows()
    assert {r["partition"] for r in rows} == {"train", "validation", "test"}


def test_partition_filtering_is_correct_and_complete():
    rows = _rows()
    train = filter_by_partition(rows, "train")
    val = filter_by_partition(rows, "validation")
    test = filter_by_partition(rows, "test")

    assert all(r["partition"] == "train" for r in train)
    assert all(r["partition"] == "validation" for r in val)
    assert all(r["partition"] == "test" for r in test)
    assert (len(train), len(val), len(test)) == (9924, 2133, 2144)

    combined = Counter(r["image_path"] for r in train + val + test)
    assert combined == Counter(r["image_path"] for r in rows)
    assert max(combined.values()) == 1


def test_partitions_are_pairwise_disjoint():
    rows = _rows()
    train = {r["image_path"] for r in filter_by_partition(rows, "train")}
    val = {r["image_path"] for r in filter_by_partition(rows, "validation")}
    test = {r["image_path"] for r in filter_by_partition(rows, "test")}

    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


def test_filter_by_partition_returns_empty_for_unknown_partition():
    rows = _rows()
    assert filter_by_partition(rows, "holdout") == []


# ---------------------------------------------------------------------------
# Class weights come from training labels only
# ---------------------------------------------------------------------------

def test_class_weights_use_training_rows_only():
    rows = _rows()
    train_rows = filter_by_partition(rows, "train")
    train_counts = Counter(r["primary_class"] for r in train_rows)
    weights = compute_class_weights_from_rows(train_rows)

    assert set(weights) == set(BAND_ORDER)
    n = sum(train_counts.values())
    for name in BAND_ORDER:
        assert weights[name] == pytest.approx(n / (len(BAND_ORDER) * train_counts[name]), abs=1e-9)


def test_class_weights_are_unaffected_by_validation_and_test_counts():
    """Rebalancing the val/test partitions must not move the weights."""
    rows = _rows()
    train_rows = filter_by_partition(rows, "train")
    baseline = compute_class_weights_from_rows(train_rows)

    shuffled_others = [
        {**r, "primary_class": "PDR"} for r in rows if r["partition"] != "train"
    ]
    assert shuffled_others  # sanity: there really are non-train rows
    after = compute_class_weights_from_rows(train_rows)
    assert after == baseline


def test_rarest_training_band_receives_the_largest_weight():
    rows = _rows()
    train_rows = filter_by_partition(rows, "train")
    counts = Counter(r["primary_class"] for r in train_rows)
    weights = compute_class_weights_from_rows(train_rows)
    rarest = min(BAND_ORDER, key=lambda name: counts[name])
    assert weights[rarest] == max(weights.values())


def test_class_weights_reject_a_partition_missing_a_band():
    with pytest.raises(ValueError, match="no examples for class"):
        compute_class_weights_from_rows([{"primary_class": "No apparent DR"}])


def test_class_weights_reject_an_empty_partition():
    with pytest.raises(ValueError, match="No labels available"):
        compute_class_weights_from_rows([])


def test_weighted_loss_matches_the_provisional_band_order():
    weights = {name: float(i + 1) for i, name in enumerate(BAND_ORDER)}
    loss_fn = make_weighted_loss_fn(weights)
    assert loss_fn.weight is not None
    assert torch.equal(
        loss_fn.weight, torch.tensor([float(i + 1) for i in range(len(BAND_ORDER))])
    )


def test_weighted_loss_rejects_incomplete_weights():
    with pytest.raises(ValueError, match="missing entries"):
        make_weighted_loss_fn({"No apparent DR": 1.0})


# ---------------------------------------------------------------------------
# The training path never uses test rows
# ---------------------------------------------------------------------------

@needs_images
def test_train_dataset_is_built_from_training_rows_only():
    rows = _rows()
    train_rows = filter_by_partition(rows, "train")
    test_paths = {r["image_path"] for r in filter_by_partition(rows, "test")}

    ds = PrimarySplitDataset(train_rows, IMAGE_ROOT, image_size=64)
    assert len(ds) == len(train_rows)

    ds_paths = {ds[i][2]["image_path"] for i in range(SAMPLE_ROWS_PER_PARTITION)}
    assert ds_paths.issubset({r["image_path"] for r in train_rows})
    assert ds_paths.isdisjoint(test_paths)


@needs_images
def test_train_and_validation_datasets_are_disjoint():
    rows = _rows()
    train_ds = PrimarySplitDataset(filter_by_partition(rows, "train"), IMAGE_ROOT, image_size=64)
    val_ds = PrimarySplitDataset(filter_by_partition(rows, "validation"), IMAGE_ROOT, image_size=64)

    train_paths = {train_ds[i][2]["image_path"] for i in range(SAMPLE_ROWS_PER_PARTITION)}
    val_paths = {val_ds[i][2]["image_path"] for i in range(SAMPLE_ROWS_PER_PARTITION)}
    assert train_paths.isdisjoint(val_paths)


@needs_images
def test_dataset_items_are_imagenet_normalized_with_the_right_label():
    rows = _rows()
    train_rows = filter_by_partition(rows, "train")
    ds = PrimarySplitDataset(train_rows, IMAGE_ROOT, image_size=64)

    for index in range(SAMPLE_ROWS_PER_PARTITION):
        tensor, label, row = ds[index]
        assert tuple(tensor.shape) == (3, 64, 64)
        assert tensor.dtype == torch.float32
        assert 0 <= label < len(BAND_ORDER)
        assert BAND_ORDER[label] == row["primary_class"]
        # ImageNet normalization keeps values in a modest range; raw [0,1] pixels
        # could not exceed ~1, so a max above 2 proves normalization was applied.
        assert float(tensor.min()) < 0.0
        assert float(tensor.max()) <= 3.0


@needs_images
def test_dataset_rejects_a_row_with_an_unknown_class():
    rows = _rows()
    bad = [dict(filter_by_partition(rows, "train")[0], primary_class="Not A Band")]
    with pytest.raises(ValueError, match="Unexpected primary_class"):
        PrimarySplitDataset(bad, IMAGE_ROOT, image_size=64)


def test_dataset_rejects_missing_images():
    with pytest.raises(FileNotFoundError, match="Image does not exist"):
        PrimarySplitDataset(
            [{"image_path": "definitely/not/here.png", "primary_class": BAND_ORDER[0]}],
            Path(tempfile.gettempdir()),
            image_size=32,
        )


def test_run_training_declares_no_test_partition_use():
    """The only partition loaders in run_training are train and validation."""
    import inspect

    source = inspect.getsource(run_training)
    assert 'filter_by_partition(rows, "train")' in source
    assert 'filter_by_partition(rows, "validation")' in source
    assert 'filter_by_partition(rows, "test")' in source  # used only for the overlap invariant
    # No test-partition dataset or loader may be constructed inside training.
    assert "PrimarySplitDataset(test_rows" not in source
    assert "test_loader" not in source


# ---------------------------------------------------------------------------
# Validation-based checkpoint selection and early stopping
# ---------------------------------------------------------------------------

def test_selection_rule_improves_on_lower_validation_loss():
    is_best, best_val_loss, best_epoch, patience = update_best_tracking(
        best_val_loss=1.0, best_epoch=0, patience_counter=2, epoch=3, val_loss=0.8
    )
    assert is_best is True
    assert best_val_loss == pytest.approx(0.8)
    assert best_epoch == 3
    assert patience == 0  # improvement resets patience


def test_selection_rule_keeps_previous_best_and_counts_patience():
    is_best, best_val_loss, best_epoch, patience = update_best_tracking(
        best_val_loss=0.8, best_epoch=3, patience_counter=1, epoch=4, val_loss=0.9
    )
    assert is_best is False
    assert best_val_loss == pytest.approx(0.8)
    assert best_epoch == 3
    assert patience == 2


def test_equal_validation_loss_is_not_an_improvement():
    is_best, _, best_epoch, patience = update_best_tracking(
        best_val_loss=0.8, best_epoch=3, patience_counter=0, epoch=4, val_loss=0.8
    )
    assert is_best is False
    assert best_epoch == 3
    assert patience == 1


def test_selection_rule_supports_the_early_stopping_contract():
    best_val_loss, best_epoch, patience, patience_limit = float("inf"), 0, 0, 2
    history = [1.0, 0.7, 0.9, 0.95]  # best at epoch 1, then two non-improvements
    stopped_at = None
    for epoch, val_loss in enumerate(history):
        enabled, best_val_loss, best_epoch, patience = update_best_tracking(
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            patience_counter=patience,
            epoch=epoch,
            val_loss=val_loss,
        )
        if patience >= patience_limit:
            stopped_at = epoch
            break

    assert best_epoch == 1
    assert best_val_loss == pytest.approx(0.7)
    assert stopped_at == 3  # two epochs after the best, as `patience=2` requires
    assert not enabled  # the stopping epoch was not itself an improvement


def test_best_checkpoint_written_only_when_selection_says_best(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    config = {"task": "primary_five_band_provisional"}

    # A non-best epoch must not create the best checkpoint at all.
    state = initial_training_state(epoch=0, best_val_loss=0.5)
    save_checkpoint(tmp_path, state, model, optimizer, config, epoch=0, is_best=False)
    best_dir = tmp_path / "best_checkpoint"
    assert not best_dir.exists()

    # A best epoch creates it, and a later non-best epoch must not overwrite it.
    state = initial_training_state(epoch=1, best_val_loss=0.4, best_epoch=1)
    save_checkpoint(tmp_path, state, model, optimizer, config, epoch=1, is_best=True)
    best_meta = json.loads((best_dir / "training_state.json").read_text(encoding="utf-8"))
    assert best_meta["state"]["best_epoch"] == 1
    assert best_meta["is_best"] is True

    marker = (best_dir / "model.pt").read_bytes()
    state = initial_training_state(epoch=2, best_val_loss=0.4, best_epoch=1)
    save_checkpoint(tmp_path, state, model, optimizer, config, epoch=2, is_best=False)
    assert (best_dir / "model.pt").read_bytes() == marker
    assert json.loads((best_dir / "training_state.json").read_text(encoding="utf-8")) == best_meta


def test_every_epoch_checkpoint_is_preserved(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    config = {"task": "primary_five_band_provisional"}
    for epoch in range(3):
        state = initial_training_state(epoch=epoch, best_val_loss=0.5, best_epoch=0)
        save_checkpoint(tmp_path, state, model, optimizer, config, epoch, is_best=epoch == 0)

    for epoch in range(3):
        epoch_dir = tmp_path / "checkpoints" / f"epoch_{epoch:04d}"
        assert (epoch_dir / "model.pt").is_file()
        assert (epoch_dir / "training_state.json").is_file()


def test_history_records_every_epoch():
    state = initial_training_state()
    state["train_history"].append({"loss": 1.0})
    state["val_history"].append({"loss": 1.2})
    assert len(state["train_history"]) == 1
    assert len(state["val_history"]) == 1


def test_train_and_evaluate_epoch_return_loss_and_sample_counts():
    """A tiny real forward/backward pass proves the epoch loops are executable."""
    model = torch.nn.Linear(3 * 4 * 4, len(BAND_ORDER))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = make_weighted_loss_fn({name: 1.0 for name in BAND_ORDER})

    batches = [
        (torch.randn(2, 3 * 4 * 4), torch.tensor([0, 4]), {"image_path": "x"}),
        (torch.randn(2, 3 * 4 * 4), torch.tensor([1, 2]), {"image_path": "y"}),
    ]
    device = torch.device("cpu")

    train_metrics = train_epoch(model, batches, loss_fn, optimizer, device)
    assert train_metrics["num_samples"] == 4
    assert train_metrics["loss"] > 0

    val_metrics = evaluate_epoch(model, batches, loss_fn, device)
    assert val_metrics["num_samples"] == 4
    assert 0.0 <= val_metrics["accuracy"] <= 1.0
    assert set(val_metrics["per_class"]) == set(BAND_ORDER)


# ---------------------------------------------------------------------------
# Metrics contract
# ---------------------------------------------------------------------------

def test_metrics_report_every_required_field():
    logits = torch.tensor([
        [2.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 4.0],
    ])
    labels = torch.tensor([0, 2, 4])
    m = compute_metrics(logits, labels)

    for key in (
        "accuracy", "macro_precision", "macro_recall", "macro_f1",
        "per_class", "confusion_matrix", "num_samples", "band_order",
    ):
        assert key in m, key

    assert m["accuracy"] == pytest.approx(1.0)
    assert set(m["per_class"]) == set(BAND_ORDER)
    assert m["num_samples"] == 3
    assert m["band_order"] == BAND_ORDER
    for name in BAND_ORDER:
        assert set(m["per_class"][name]) >= {"precision", "recall", "f1", "support"}


def test_confusion_matrix_is_square_integers_partitioning_all_samples():
    logits = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]])
    labels = torch.tensor([0, 1])
    conf = compute_metrics(logits, labels)["confusion_matrix"]

    assert len(conf) == len(BAND_ORDER)
    assert all(len(row) == len(BAND_ORDER) for row in conf)
    assert all(isinstance(v, int) for row in conf for v in row)
    assert sum(sum(row) for row in conf) == 2
    assert conf[0][0] == 1 and conf[1][1] == 1


def test_perfect_predictions_give_unit_macros():
    logits = torch.eye(len(BAND_ORDER)) * 10.0
    labels = torch.arange(len(BAND_ORDER))
    m = compute_metrics(logits, labels)
    assert m["macro_precision"] == pytest.approx(1.0)
    assert m["macro_recall"] == pytest.approx(1.0)
    assert m["macro_f1"] == pytest.approx(1.0)


def test_metrics_are_json_serializable():
    logits = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
    labels = torch.tensor([0])
    payload = json.dumps(compute_metrics(logits, labels))
    assert json.loads(payload)["num_samples"] == 1


# ---------------------------------------------------------------------------
# Training guards prevent accidental execution
# ---------------------------------------------------------------------------

def test_training_is_disabled_without_an_explicit_opt_in(tmp_path):
    with pytest.raises(PermissionError, match="disabled by default"):
        run_training(
            split_dir=SPLIT_DIR,
            dataset_root=tmp_path,
            output_dir=tmp_path / "out",
        )


def test_training_refuses_before_the_smoke_test_has_passed(tmp_path):
    with pytest.raises(PermissionError, match="smoke test has passed"):
        run_training(
            split_dir=SPLIT_DIR,
            dataset_root=tmp_path,
            output_dir=tmp_path / "out",
            confirm_training=True,
            smoke_test_passed=False,
        )


def test_final_evaluation_is_disabled_without_an_explicit_request(tmp_path):
    with pytest.raises(PermissionError, match="disabled by default"):
        run_final_evaluation(
            split_dir=SPLIT_DIR,
            dataset_root=tmp_path,
            output_dir=tmp_path / "out",
        )


def test_final_evaluation_requires_a_best_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="Best checkpoint directory not found"):
        run_final_evaluation(
            split_dir=SPLIT_DIR,
            dataset_root=tmp_path,
            output_dir=tmp_path / "out",
            confirm_final_evaluation=True,
        )


def test_guard_contract_is_explicit_and_matches_the_notebook():
    """The library contract, the notebook opt-ins, and the pipeline forwarding agree."""
    import inspect

    import colab.pipeline as pipeline

    assert confirm_guard_contract() == {
        "training_requires_confirm": True,
        "training_requires_smoke_test": True,
        "final_evaluation_requires_confirm": True,
    }

    text = NOTEBOOK_PATH.read_text(encoding="utf-8")
    assert "RUN_TRAINING = False" in text
    assert "RUN_FINAL_EVALUATION = False" in text
    assert "raise SystemExit" in text

    # The notebook opts in; the pipeline is what actually sets the library guards.
    train_src = inspect.getsource(pipeline.train_run)
    assert "confirm_training=True" in train_src
    assert "smoke_test_passed=True" in train_src
    eval_src = inspect.getsource(pipeline.evaluate_run)
    assert "confirm_final_evaluation=True" in eval_src


def test_notebook_runs_the_smoke_test_under_pytest():
    """The smoke gate stays a real pytest run, now owned by ``preflight``."""
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "pipeline.preflight(" in code
    assert "run_smoke=True" in code

    helpers_src = (REPO_ROOT / "colab" / "setup_helpers.py").read_text(encoding="utf-8")
    # Running the module directly would define tests and exit 0 without executing
    # any of them, so the suite has to be invoked through pytest.
    assert '"-m", "pytest"' in helpers_src
    assert "returncode" in helpers_src


def test_weight_guards_never_raise_even_without_torchvision():
    """Both weight helpers must be safe to call when torchvision is missing."""
    from colab_helpers import pretrained_weights_available, weights_cache_path

    assert isinstance(pretrained_weights_available(), bool)
    assert isinstance(weights_cache_path(), Path)


def test_training_fails_clearly_when_the_stack_is_missing(tmp_path):
    """The precondition check must name the problem instead of failing obscurely.

    Whatever is genuinely absent in this runtime (torchvision, or the cached
    ImageNet weights behind it), the refusal must be the named, actionable error
    - not an escaped traceback and not a silent pass.
    """
    from colab_helpers import pretrained_weights_available
    from colab.train import ensure_training_preconditions

    try:
        import torchvision  # noqa: F401

        has_torchvision = True
    except ImportError:
        has_torchvision = False

    if not has_torchvision:
        with pytest.raises(RuntimeError, match="torchvision is required"):
            ensure_training_preconditions(allow_weight_download=False)
    elif not pretrained_weights_available():
        with pytest.raises(FileNotFoundError, match="not cached locally"):
            ensure_training_preconditions(allow_weight_download=False)
    else:
        report = ensure_training_preconditions(allow_weight_download=False)
        assert report["device"] in {"cpu", "cuda"}


def test_missing_weights_are_reported_not_downloaded(tmp_path):
    """With no cached weights, training must refuse instead of fetching them."""
    from colab_helpers import pretrained_weights_available
    from colab.train import ensure_training_preconditions

    if pretrained_weights_available():
        pytest.skip("ImageNet weights are already cached in this environment")

    try:
        import torchvision  # noqa: F401
    except ImportError:
        pytest.skip("torchvision is not installed, so the torchvision precondition fires first")

    with pytest.raises(FileNotFoundError, match="not cached locally"):
        ensure_training_preconditions(allow_weight_download=False)


def test_device_report_describes_the_actual_runtime():
    report = device_report()
    assert report["device"] == ("cuda" if torch.cuda.is_available() else "cpu")
    assert report["cuda_available"] == torch.cuda.is_available()
    if not torch.cuda.is_available():
        assert report["cuda_device_name"] is None


# ---------------------------------------------------------------------------
# Output collision / overwrite protection
# ---------------------------------------------------------------------------

def test_new_output_directory_is_created(tmp_path):
    out = prepare_output_dir(tmp_path / "run_a")
    assert out.is_dir()


def test_existing_run_directory_is_not_overwritten(tmp_path):
    out = prepare_output_dir(tmp_path / "run_a")
    write_run_meta(out, {"task": "primary_five_band_provisional", "seed": 12345})

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_output_dir(tmp_path / "run_a")

    # The previous run's metadata must survive the refusal untouched.
    assert json.loads((out / "run_meta.json").read_text(encoding="utf-8"))["seed"] == 12345


def test_explicit_overwrite_replaces_the_previous_run(tmp_path):
    out = prepare_output_dir(tmp_path / "run_a")
    write_run_meta(out, {"task": "previous"})

    out2 = prepare_output_dir(tmp_path / "run_a", overwrite=True)
    assert out2 == out
    assert not (out2 / "run_meta.json").exists()


def test_output_path_that_is_a_file_is_rejected(tmp_path):
    target = tmp_path / "not_a_dir"
    target.write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not a directory"):
        prepare_output_dir(target)


def test_run_meta_and_manifest_round_trip(tmp_path):
    out = prepare_output_dir(tmp_path / "run_a")
    config = {"task": "primary_five_band_provisional", "epochs": 10, "seed": 12345}
    write_run_meta(out, config)

    state = initial_training_state(epoch=2, best_val_loss=0.5, best_epoch=1)
    manifest_path = write_run_manifest(out, config, state, status="in_progress")

    assert json.loads((out / "run_meta.json").read_text(encoding="utf-8")) == config
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "in_progress"
    assert manifest["best_epoch"] == 1
    assert manifest["resume_supported"] is False


def test_interrupted_run_is_marked_incomplete_and_never_completed(tmp_path):
    out = prepare_output_dir(tmp_path / "run_a")
    config = {"task": "primary_five_band_provisional"}
    state = initial_training_state(epoch=1, best_val_loss=0.5)

    write_run_manifest(out, config, state, status="incomplete", note="KeyboardInterrupt")
    marker = mark_incomplete(out, "KeyboardInterrupt")

    assert marker.is_file()
    assert "INCOMPLETE" in marker.read_text(encoding="utf-8")
    assert json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))["status"] == "incomplete"
    assert not (out / "run_summary.json").exists()


def test_completed_run_summary_clears_the_incomplete_marker(tmp_path):
    out = prepare_output_dir(tmp_path / "run_a")
    config = {"task": "primary_five_band_provisional"}
    state = initial_training_state(epoch=1, best_val_loss=0.5, best_epoch=1)
    state["train_history"].append({"loss": 1.0, "epoch": 0})
    state["val_history"].append({"loss": 0.5, "epoch": 0})

    mark_incomplete(out, "simulated interruption")
    summary_path = save_run_summary(out, config, state)

    assert summary_path.is_file()
    assert not (out / "TRAINING_INCOMPLETE.txt").exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert summary["best_epoch"] == 1
    assert len(summary["train_history"]) == 1
    assert len(summary["val_history"]) == 1


# ---------------------------------------------------------------------------
# Checkpoint metadata and compatibility
# ---------------------------------------------------------------------------

def _write_checkpoint(tmp_path, *, model_name="resnet50_frozen", head_count=len(BAND_ORDER)):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    state = initial_training_state(epoch=0, best_val_loss=1.0)
    config = {
        "task": "primary_five_band_provisional",
        "model": model_name,
        "image_size": 224,
        "seed": 12345,
        "num_classes": head_count,
        "class_weights": {name: 1.0 for name in BAND_ORDER},
    }
    save_checkpoint(tmp_path, state, model, optimizer, config, epoch=0, is_best=True)
    return tmp_path / "best_checkpoint"


def test_checkpoint_metadata_has_the_required_shape(tmp_path):
    best_dir = _write_checkpoint(tmp_path)
    meta = inspect_checkpoint_metadata(best_dir)
    assert set(meta) >= {"state", "config", "collab_compat"}
    assert meta["state"]["best_val_loss"] == pytest.approx(1.0)


def test_checkpoint_compatibility_report_accepts_a_matching_checkpoint(tmp_path):
    best_dir = _write_checkpoint(tmp_path)
    report = checkpoint_compat_report(best_dir)
    assert report["compatible"] is True
    assert report["collab_compat"]["head_class_count"] == len(BAND_ORDER)


def test_checkpoint_compatibility_rejects_a_mismatched_head(tmp_path):
    best_dir = _write_checkpoint(tmp_path, head_count=13)
    with pytest.raises(ValueError, match="does not match"):
        checkpoint_compat_report(best_dir)


def test_checkpoint_compatibility_rejects_a_different_model(tmp_path):
    best_dir = _write_checkpoint(tmp_path, model_name="efficientnet_b0")
    with pytest.raises(ValueError, match="resnet50_frozen"):
        checkpoint_compat_report(best_dir)


def test_checkpoint_compatibility_flags_missing_weights(tmp_path):
    best_dir = _write_checkpoint(tmp_path)
    (best_dir / "model.pt").unlink()
    with pytest.raises(FileNotFoundError, match="weights missing"):
        checkpoint_compat_report(best_dir)


def test_checkpoint_compatibility_flags_legacy_metadata(tmp_path):
    best_dir = _write_checkpoint(tmp_path)
    meta_path = best_dir / "training_state.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.pop("collab_compat")
    meta["config"].pop("collab_compat", None)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ValueError, match="collab_compat"):
        checkpoint_compat_report(best_dir)


def test_load_training_state_rejects_a_wrong_shape(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"state": {"epoch": 1}}), encoding="utf-8")
    with pytest.raises(ValueError, match="required keys"):
        load_training_state(path)


def test_load_training_state_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_training_state(tmp_path / "nope.json")


# ---------------------------------------------------------------------------
# Resize-and-pad preprocessing
# ---------------------------------------------------------------------------

def test_preprocessing_resizes_and_pads_without_cropping():
    from PIL import Image

    preprocess = preprocessing_inference(224)
    img = Image.new("RGB", (400, 300), color=(10, 20, 30))
    out = preprocess(img)

    assert out.size == (224, 224)
    assert out.mode == "RGB"

    left, top, right, bottom = out.info["content_box"]
    width, height = right - left, bottom - top
    assert (width, height) == (224, 168)  # 400x300 scaled to fit, not cropped
    assert abs((width / height) - (400 / 300)) < 0.01
    assert (left, top) == (0, 28)  # centred vertically


def test_preprocessing_is_deterministic():
    from PIL import Image

    preprocess = preprocessing_inference(224)
    img = Image.new("RGB", (400, 300), color=(10, 20, 30))
    assert preprocess(img).tobytes() == preprocess(img).tobytes()


def test_preprocessing_handles_tall_and_square_inputs():
    """Every input shape follows the documented resize-and-pad rule exactly."""
    from PIL import Image

    preprocess = preprocessing_inference(224)
    for size in ((200, 400), (200, 200), (500, 100)):
        width, height = size
        scale = min(224 / width, 224 / height)
        expected_w = max(1, round(width * scale))
        expected_h = max(1, round(height * scale))

        out = preprocess(Image.new("RGB", size, color=(1, 2, 3)))
        assert out.size == (224, 224)
        left, top, right, bottom = out.info["content_box"]
        assert (right - left, bottom - top) == (expected_w, expected_h)
        assert (left, top) == ((224 - expected_w) // 2, (224 - expected_h) // 2)


def test_preprocessing_pads_with_black_not_a_crop():
    from PIL import Image

    # A 2x1 source becomes 224x112 content plus padding, so both colours exist.
    out = preprocessing_inference(224)(Image.new("RGB", (2, 1), color=(255, 0, 0)))
    left, top, right, bottom = out.info["content_box"]
    assert (top, bottom) == (56, 168)  # centred 112-pixel-tall content

    assert out.getpixel((112, 112)) == (255, 0, 0)  # inside the content box
    assert out.getpixel((112, 0)) == (0, 0, 0)  # padding above it
    assert out.getpixel((112, 223)) == (0, 0, 0)  # padding below it
    assert (left, right) == (0, 224)


def test_preprocessing_train_matches_inference():
    from PIL import Image

    from colab_helpers import preprocessing_train

    img = Image.new("RGB", (321, 210), color=(7, 8, 9))
    assert preprocessing_train(224)(img).tobytes() == preprocessing_inference(224)(img).tobytes()


# ---------------------------------------------------------------------------
# Label mapping integration
# ---------------------------------------------------------------------------

def test_band_order_matches_the_provisional_label_mapping():
    assert set(PRIMARY_MAPPING.values()) == set(BAND_ORDER)


def test_provisional_merges_are_preserved_and_documented():
    assert primary_task_label("13. Very Severe NPDR") == "Severe NPDR"
    assert primary_task_label("12. Advanced PDR") == "PDR"
    assert primary_task_label("11.Normal Fundus") == "No apparent DR"
    assert primary_task_label("3.Mild DR") == "Mild NPDR"
    assert primary_task_label("4.Moderate DR") == "Moderate NPDR"
    assert primary_task_label("5.Severe DR") == "Severe NPDR"


def test_non_primary_labels_are_rejected_or_flagged():
    with pytest.raises(ValueError, match="Unknown or non-primary"):
        primary_task_label("7.Cataract")
    assert is_non_dr_source("7.Cataract") is True
    assert is_non_dr_source("4.Moderate DR") is False


def test_split_csv_primary_class_matches_the_manifest_mapping():
    """Every split row's band must equal dataset.labels applied to its source label."""
    from dataset.eligibility import filter_primary_eligible
    from dataset.manifest import load_manifest

    rows = _rows()
    manifest_rows = list(
        filter_primary_eligible(
            load_manifest(PRIMARY_CSV, "primary", image_root=IMAGE_ROOT, require_files_exist=False)
        )
    )
    expected_by_path = {
        r["image_path"]: primary_task_label(r["original_label"]) for r in manifest_rows
    }

    mismatches = [
        (r["image_path"], r["primary_class"], expected_by_path[r["image_path"]])
        for r in rows
        if r["image_path"] in expected_by_path
        and r["primary_class"] != expected_by_path[r["image_path"]]
    ]
    assert mismatches == []
    # The split must be fully covered by the manifest, not silently short.
    assert sum(1 for r in rows if r["image_path"] in expected_by_path) == len(rows)


def test_smoke_test_module_is_discoverable_and_is_a_pytest_suite():
    smoke = REPO_ROOT / "colab" / "smoke_test.py"
    source = smoke.read_text(encoding="utf-8")
    assert "def test_" in source
    assert "pytest" in source


# ---------------------------------------------------------------------------
# Notebook code cells must be valid Python
# ---------------------------------------------------------------------------

def test_notebook_code_cells_compile():
    """Every code cell must at least parse.

    A notebook is JSON, so a broken string literal introduced while editing it stays
    invisible until Colab raises ``SyntaxError`` mid-session. This compiles each cell
    so the failure surfaces here instead.
    """
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells, "notebook has no code cells"

    failures = []
    for index, cell in enumerate(code_cells):
        source = "".join(cell["source"])
        try:
            compile(source, f"<notebook code cell {index}>", "exec")
        except SyntaxError as exc:
            failures.append(f"code cell {index}: {exc}")
    assert failures == [], "notebook code cells do not compile:\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# The weighted loss must live on the training device
# ---------------------------------------------------------------------------

def test_cpu_class_weights_fail_against_logits_on_another_device():
    """Regression: the weighted loss must be placed on the training device.

    On a CUDA runtime a CPU class-weight tensor against CUDA logits raises
    ``RuntimeError: Expected all tensors to be on the same device`` on the very
    first training step, and moving only the model does not fix it. The ``meta``
    device reproduces the same failure class locally, so this stays executable on a
    CPU-only machine, and it shows the fix removing the failure.
    """
    weights = {name: 1.0 for name in BAND_ORDER}
    logits = torch.zeros(2, len(BAND_ORDER), device="meta")
    labels = torch.zeros(2, dtype=torch.long, device="meta")

    unmoved = make_weighted_loss_fn(weights)
    assert unmoved.weight.device.type == "cpu"
    with pytest.raises(RuntimeError, match="not on the expected device"):
        unmoved(logits, labels)

    moved = make_weighted_loss_fn(weights, device=logits.device)
    assert moved.weight.device.type == "meta"
    moved(logits, labels)  # no device error once the weight follows the logits


def test_weighted_loss_defaults_to_cpu_and_can_be_moved_by_to():
    weights = {name: 1.0 for name in BAND_ORDER}
    assert make_weighted_loss_fn(weights).weight.device.type == "cpu"
    assert make_weighted_loss_fn(weights, device=torch.device("cpu")).weight.device.type == "cpu"
    assert make_weighted_loss_fn(weights, device=torch.device("meta")).weight.device.type == "meta"


# ---------------------------------------------------------------------------
# Download-free ResNet-50 construction (torchvision faked, helper code real)
# ---------------------------------------------------------------------------

class _FakeResNet(torch.nn.Module):
    """Minimal resnet50 stand-in: a frozen 'conv1' plus a replaceable 'fc' head."""

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=3)
        self.fc = torch.nn.Linear(8, 1000)


class _FakeResNet50Weights:
    class IMAGENET1K_V1:
        url = "https://download.pytorch.org/models/resnet50-0676ba61.pth"


class _RecordingResNet50Builder:
    """Records every ``resnet50(...)`` call so downloads are observable."""

    def __init__(self):
        self.calls = []

    def __call__(self, weights=None, **kwargs):
        self.calls.append({"weights": weights, **kwargs})
        return _FakeResNet()


def _install_fake_torchvision(monkeypatch, builder):
    """Install a stand-in torchvision so the real helper code can be exercised.

    torchvision cannot be installed in this environment, and a test must never
    download weights, so the two attributes the helper touches are faked here.
    """
    import types

    torchvision = types.ModuleType("torchvision")
    models = types.ModuleType("torchvision.models")
    models.resnet50 = builder
    models.ResNet50_Weights = _FakeResNet50Weights
    torchvision.models = models
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.models", models)


def test_build_frozen_resnet50_without_weights_never_enters_the_pretrained_path(monkeypatch):
    """``pretrained=False`` must build the architecture and nothing else."""
    from colab_helpers import build_frozen_resnet50

    builder = _RecordingResNet50Builder()
    _install_fake_torchvision(monkeypatch, builder)

    model = build_frozen_resnet50(num_classes=len(BAND_ORDER), pretrained=False)

    assert [call["weights"] for call in builder.calls] == [None]
    assert model.fc.out_features == len(BAND_ORDER)
    assert all(p.requires_grad for p in model.fc.parameters())
    assert all(
        not p.requires_grad
        for name, p in model.named_parameters()
        if not name.startswith("fc")
    )


def test_build_frozen_resnet50_refuses_to_download_absent_weights(monkeypatch):
    """With no cached weights and no opt-in, the constructor is never reached."""
    from colab_helpers import build_frozen_resnet50, pretrained_weights_available

    if pretrained_weights_available():
        pytest.skip("ImageNet weights are already cached, so the refusal path is unreachable")

    builder = _RecordingResNet50Builder()
    _install_fake_torchvision(monkeypatch, builder)

    with pytest.raises(FileNotFoundError, match="not cached locally"):
        build_frozen_resnet50(num_classes=len(BAND_ORDER), allow_download=False)
    # Raising before the constructor is what guarantees no silent download.
    assert builder.calls == []


def test_build_frozen_resnet50_requests_the_imagenet_enum_only_when_opted_in(monkeypatch):
    """The ImageNet enum is used only after an explicit, cached-weights decision."""
    import colab_helpers as H

    builder = _RecordingResNet50Builder()
    _install_fake_torchvision(monkeypatch, builder)
    monkeypatch.setattr(H, "pretrained_weights_available", lambda: True)

    H.build_frozen_resnet50(num_classes=len(BAND_ORDER), allow_download=False)

    assert builder.calls[0]["weights"] is _FakeResNet50Weights.IMAGENET1K_V1


# ---------------------------------------------------------------------------
# Overwrite safety is deferred until cheap validation succeeds
# ---------------------------------------------------------------------------

_FAKE_RUNTIME = {
    "torch_version": torch.__version__,
    "cuda_available": False,
    "device": "cpu",
    "cuda_device_name": None,
    "cuda_device_count": 0,
}


def _previous_completed_run(output_dir):
    """Write the minimum artifacts that mark a directory as an existing run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_manifest.json").write_text(
        json.dumps({"status": "completed", "note": "previous run"}), encoding="utf-8"
    )
    (output_dir / "run_summary.json").write_text(
        json.dumps({"status": "completed", "marker": "keep me"}), encoding="utf-8"
    )
    return {p.name: p.read_bytes() for p in sorted(output_dir.iterdir())}


def test_overwrite_does_not_destroy_a_previous_run_when_validation_fails(tmp_path, monkeypatch):
    """A failed start must not delete or relabel the run it was asked to replace."""
    import colab.train as train_mod

    out = tmp_path / "existing_run"
    before = _previous_completed_run(out)
    monkeypatch.setattr(
        train_mod, "ensure_training_preconditions", lambda *, allow_weight_download: _FAKE_RUNTIME
    )

    # tmp_path holds no images, so dataset construction fails after the overwrite
    # decision has already been made but before anything destructive happened.
    with pytest.raises((FileNotFoundError, RuntimeError)):
        train_mod.run_training(
            split_dir=SPLIT_DIR,
            dataset_root=tmp_path,
            output_dir=out,
            epochs=1,
            image_size=32,
            device=torch.device("cpu"),
            confirm_training=True,
            smoke_test_passed=True,
            overwrite=True,
        )

    after = {p.name: p.read_bytes() for p in sorted(out.iterdir())}
    assert after == before
    assert not (out / "TRAINING_INCOMPLETE.txt").exists()
    assert json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))["status"] == "completed"


def test_failed_start_does_not_relabel_a_previous_run(tmp_path, monkeypatch):
    """Even without overwrite, a failure must leave the previous run's status alone."""
    import colab.train as train_mod

    out = tmp_path / "existing_run"
    before = _previous_completed_run(out)
    monkeypatch.setattr(
        train_mod, "ensure_training_preconditions", lambda *, allow_weight_download: _FAKE_RUNTIME
    )

    with pytest.raises(FileNotFoundError):
        train_mod.run_training(
            split_dir=SPLIT_DIR,
            dataset_root=tmp_path / "missing_dataset_root",
            output_dir=out,
            epochs=1,
            device=torch.device("cpu"),
            confirm_training=True,
            smoke_test_passed=True,
        )

    after = {p.name: p.read_bytes() for p in sorted(out.iterdir())}
    assert after == before
    assert not (out / "TRAINING_INCOMPLETE.txt").exists()


# ---------------------------------------------------------------------------
# End-to-end mini runs of the real pipeline (weights/model/dataset size stubbed)
# ---------------------------------------------------------------------------

MODEL_IMAGE_SIZE = 32
_TINY_MODEL_INPUT = 3 * MODEL_IMAGE_SIZE * MODEL_IMAGE_SIZE


class _TinyResNetLike(torch.nn.Module):
    """Stands in for frozen ResNet-50: frozen backbone plus a trainable 'fc' head."""

    def __init__(self, num_classes: int = len(BAND_ORDER)) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(_TINY_MODEL_INPUT, 16)
        self.fc = torch.nn.Linear(16, num_classes)
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.fc(torch.tanh(self.backbone(x.flatten(1))))


def _mini_subsets():
    """Pick a few real split rows: one per band for train, a handful for the rest."""
    rows = _rows()
    train_rows, seen = [], set()
    for row in filter_by_partition(rows, "train"):
        if row["primary_class"] not in seen:
            seen.add(row["primary_class"])
            train_rows.append(row)
    val_rows = filter_by_partition(rows, "validation")[:3]
    test_rows = filter_by_partition(rows, "test")[:3]
    return train_rows, val_rows, test_rows


def _install_mini_run_stubs(monkeypatch, train_rows, val_rows, test_rows, decoded, built, loss_devices):
    """Point the real pipeline at a tiny model and a tiny dataset slice."""
    import colab.train as train_mod

    real_load_rgb = train_mod.load_rgb
    real_make_loss = train_mod.make_weighted_loss_fn
    real_filter = train_mod.filter_by_partition

    def _recording_load_rgb(path, *args, **kwargs):
        decoded.append(str(path))
        return real_load_rgb(path, *args, **kwargs)

    def _recording_make_loss(class_weights, *, device=None):
        loss_devices.append(device)
        return real_make_loss(class_weights, device=device)

    def _stub_builder(num_classes=len(BAND_ORDER), *, allow_download=False, pretrained=True, **kwargs):
        built.append(
            {"num_classes": num_classes, "allow_download": allow_download, "pretrained": pretrained}
        )
        return _TinyResNetLike(num_classes)

    def _stub_filter(rows, partition):
        if partition == "train":
            return list(train_rows)
        if partition == "validation":
            return list(val_rows)
        if partition == "test":
            return list(test_rows)
        return real_filter(rows, partition)

    monkeypatch.setattr(train_mod, "load_rgb", _recording_load_rgb)
    monkeypatch.setattr(train_mod, "make_weighted_loss_fn", _recording_make_loss)
    monkeypatch.setattr(train_mod, "build_frozen_resnet50", _stub_builder)
    monkeypatch.setattr(train_mod, "filter_by_partition", _stub_filter)
    monkeypatch.setattr(
        train_mod, "ensure_training_preconditions", lambda *, allow_weight_download: _FAKE_RUNTIME
    )
    return train_mod


@needs_images
def test_mini_training_run_is_validation_selected_and_never_reads_test_images(tmp_path, monkeypatch):
    """Execute the real training loop end to end on a handful of real images.

    Only the ImageNet weights/model and the split size are stubbed; the dataset,
    preprocessing, weighting, loop, checkpointing, and manifest bookkeeping are the
    real code paths.
    """
    train_rows, val_rows, test_rows = _mini_subsets()
    if len({row["primary_class"] for row in train_rows}) != len(BAND_ORDER):
        pytest.skip("the training partition does not expose every band")

    decoded, built, loss_devices = [], [], []
    train_mod = _install_mini_run_stubs(
        monkeypatch, train_rows, val_rows, test_rows, decoded, built, loss_devices
    )

    out = tmp_path / "mini_run"
    result = train_mod.run_training(
        split_dir=SPLIT_DIR,
        dataset_root=IMAGE_ROOT,
        output_dir=out,
        epochs=2,
        batch_size=2,
        patience=1,
        image_size=MODEL_IMAGE_SIZE,
        device=torch.device("cpu"),
        confirm_training=True,
        smoke_test_passed=True,
    )

    summary = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))

    # Selection followed the validation loss history and never left initial values.
    val_losses = [entry["loss"] for entry in summary["val_history"]]
    assert val_losses and result["best_val_loss"] < float("inf")
    assert result["best_val_loss"] == pytest.approx(min(val_losses))
    assert result["best_epoch"] == val_losses.index(min(val_losses))

    # Metrics describe the tiny validation subset, not the test partition.
    assert summary["status"] == "completed"
    assert len(summary["train_history"]) == len(summary["val_history"]) >= 1
    assert summary["val_history"][0]["num_samples"] == len(val_rows)

    # The saved best checkpoint is the one selection picked.
    best_state = json.loads(
        (out / "best_checkpoint" / "training_state.json").read_text(encoding="utf-8")
    )
    assert best_state["state"]["best_epoch"] == result["best_epoch"]
    assert best_state["is_best"] is True

    # Class weights came from the training rows only, and match them exactly.
    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["class_weights_source"] == "train_partition_only"
    assert set(meta["class_weights"]) == set(BAND_ORDER)
    assert meta["class_weights"] == pytest.approx(compute_class_weights_from_rows(train_rows))

    # No test image was ever decoded during training.
    test_paths = {str(IMAGE_ROOT / row["image_path"]) for row in test_rows}
    train_val_paths = {
        str(IMAGE_ROOT / row["image_path"]) for row in list(train_rows) + list(val_rows)
    }
    assert decoded, "the mini run decoded no images, so the isolation check proves nothing"
    assert test_paths.isdisjoint(decoded)
    assert set(decoded) <= train_val_paths

    # The weighted loss was placed on the device the model trained on.
    assert loss_devices and set(loss_devices) == {torch.device("cpu")}

    # The frozen five-band model was built through the shared builder contract.
    assert built[0]["num_classes"] == len(BAND_ORDER)

    # Artifacts and run status.
    assert (out / "checkpoints" / "epoch_0000" / "model.pt").is_file()
    assert (out / "best_checkpoint" / "model.pt").is_file()
    assert not (out / "TRAINING_INCOMPLETE.txt").exists()
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    # This pipeline version checkpoints enough to resume an interrupted run.
    assert manifest["resume_supported"] is True
    assert result["resumed_from"] is None


@needs_images
def test_mini_run_stops_early_after_patience_is_exhausted(tmp_path, monkeypatch):
    """Early stopping halts the loop and still records a completed run."""
    train_rows, val_rows, test_rows = _mini_subsets()
    if len({row["primary_class"] for row in train_rows}) != len(BAND_ORDER):
        pytest.skip("the training partition does not expose every band")

    decoded, built, loss_devices = [], [], []
    train_mod = _install_mini_run_stubs(
        monkeypatch, train_rows, val_rows, test_rows, decoded, built, loss_devices
    )

    # Freeze the validation loss at a single value so nothing ever improves.
    monkeypatch.setattr(
        train_mod,
        "evaluate_epoch",
        lambda model, loader, loss_fn, device: {"loss": 1.0, "accuracy": 0.0, "macro_f1": 0.0},
    )

    out = tmp_path / "early_stop_run"
    train_mod.run_training(
        split_dir=SPLIT_DIR,
        dataset_root=IMAGE_ROOT,
        output_dir=out,
        epochs=10,
        batch_size=2,
        patience=2,
        image_size=MODEL_IMAGE_SIZE,
        device=torch.device("cpu"),
        confirm_training=True,
        smoke_test_passed=True,
    )

    summary = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
    assert len(summary["train_history"]) == 3  # first epoch sets the best, then two flat epochs
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"


@needs_images
def test_mini_final_evaluation_scores_only_the_test_partition(tmp_path, monkeypatch):
    """The real final-evaluation path runs against the test partition alone."""
    train_rows, val_rows, test_rows = _mini_subsets()

    out = tmp_path / "mini_run"
    out.mkdir(parents=True)
    model = _TinyResNetLike(len(BAND_ORDER))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    config = {
        "task": "primary_five_band_provisional",
        "model": "resnet50_frozen",
        "num_classes": len(BAND_ORDER),
        "image_size": MODEL_IMAGE_SIZE,
        "seed": 12345,
        "class_weights": compute_class_weights_from_rows(train_rows),
    }
    save_checkpoint(
        out, initial_training_state(epoch=0, best_val_loss=0.25), model, optimizer, config, epoch=0, is_best=True
    )

    decoded, built, loss_devices = [], [], []
    train_mod = _install_mini_run_stubs(
        monkeypatch, train_rows, val_rows, test_rows, decoded, built, loss_devices
    )
    monkeypatch.setattr(train_mod, "require_torchvision", lambda: None)

    result = train_mod.run_final_evaluation(
        split_dir=SPLIT_DIR,
        dataset_root=IMAGE_ROOT,
        output_dir=out,
        batch_size=2,
        image_size=MODEL_IMAGE_SIZE,
        device=torch.device("cpu"),
        confirm_final_evaluation=True,
    )

    # The architecture is rebuilt without ImageNet weights: nothing to download.
    assert built[0]["pretrained"] is False

    assert result["metrics"]["num_samples"] == len(test_rows)
    test_paths = {str(IMAGE_ROOT / row["image_path"]) for row in test_rows}
    assert decoded and set(decoded) == test_paths

    eval_path = out / "final_test_evaluation" / "final_test_metrics.json"
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    assert payload["partition"] == "test"
    assert payload["num_samples"] == len(test_rows)
    assert payload["band_order"] == list(BAND_ORDER)
    assert len(payload["confusion_matrix"]) == len(BAND_ORDER)


# ---------------------------------------------------------------------------
# Notebook <-> module consistency (imports, call signatures, cell order)
# ---------------------------------------------------------------------------

def _notebook_code_cells():
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_imports_of_project_modules_resolve():
    """Every project import in every code cell must name something that exists."""
    import ast
    import importlib

    problems = []
    for index, source in enumerate(_notebook_code_cells()):
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in {"colab", "colab_helpers", "dataset"}:
                try:
                    module = importlib.import_module(node.module)
                except ImportError as exc:
                    problems.append(f"cell {index}: cannot import {node.module}: {exc}")
                    continue
                for alias in node.names:
                    if not hasattr(module, alias.name):
                        problems.append(f"cell {index}: {node.module}.{alias.name} does not exist")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in {"colab", "colab_helpers", "dataset"}:
                        try:
                            importlib.import_module(alias.name)
                        except ImportError as exc:
                            problems.append(f"cell {index}: cannot import {alias.name}: {exc}")
    assert problems == [], "notebook project imports do not resolve:\n" + "\n".join(problems)


@pytest.mark.parametrize(
    "caller_name,func_name",
    [("train_run", "run_training"), ("evaluate_run", "run_final_evaluation")],
)
def test_pipeline_calls_match_the_train_function_signatures(caller_name, func_name):
    """The entry points the notebook calls must match the real train signatures.

    The notebook only ever calls ``colab.pipeline``; the pipeline is what has to pass
    the right keyword arguments through, so this inspects the pipeline's own calls.
    """
    import ast
    import inspect
    import textwrap

    import colab.pipeline as pipeline

    func = {"run_training": run_training, "run_final_evaluation": run_final_evaluation}[func_name]
    signature = inspect.signature(func)
    required = {
        name
        for name, param in signature.parameters.items()
        if param.default is inspect.Parameter.empty
    }
    assert not any(
        param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for param in signature.parameters.values()
    ), f"{func_name} should stay keyword-only"

    source = textwrap.dedent(inspect.getsource(getattr(pipeline, caller_name)))
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", getattr(node.func, "id", None)) == func_name
    ]

    assert calls, f"{caller_name} never calls {func_name}"
    for node in calls:
        kwargs = {keyword.arg for keyword in node.keywords}
        assert not node.args, f"{caller_name} passes positional args to {func_name}"
        assert kwargs <= set(signature.parameters), (
            f"unknown {func_name} arguments from {caller_name}: "
            f"{sorted(kwargs - set(signature.parameters))}"
        )
        assert required <= kwargs, (
            f"{caller_name} omits required {func_name} arguments: {sorted(required - kwargs)}"
        )
        # The guards are always forwarded as True; the caller owns the opt-in itself.
        guard = {
            "run_training": {"confirm_training", "smoke_test_passed"},
            "run_final_evaluation": {"confirm_final_evaluation"},
        }[func_name]
        assert kwargs >= guard, kwargs


def test_notebook_never_calls_the_train_functions_directly():
    """The notebook is thin: it must go through ``colab.pipeline``, not around it."""
    blob = "\n".join(_notebook_code_cells())
    assert "colab.pipeline" in blob
    for name in ("run_training(", "run_final_evaluation("):
        assert name not in blob


# ---------------------------------------------------------------------------
# Genuine resume: the real loop, a real corpus, no GPU and no weights
# ---------------------------------------------------------------------------

def _synthetic_canary(tmp_path):
    """Build the 75-row derived split and decodable images at the canonical paths.

    No real corpus is needed, so these resume checks run everywhere (including on a
    machine with no fundus images), and no pretrained weights are downloaded.
    """
    import csv as _csv

    import colab.setup_helpers as sh
    from PIL import Image

    split_dir = tmp_path / "canary_split"
    split_dir.mkdir(parents=True)
    csv_path = split_dir / "provisional_primary_split.csv"
    sh.build_canary_split(
        SPLIT_CSV, csv_path, train_per_class=10, val_per_class=5, test_per_class=0
    )

    corpus = tmp_path / "corpus"
    rows = list(_csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    for row in rows:
        path = corpus / row["image_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), (10, 20, 30)).save(path)
    return split_dir, corpus, rows


def _stub_model_and_runtime(monkeypatch):
    import colab.train as train_mod

    def _builder(num_classes=len(BAND_ORDER), *, allow_download=False, pretrained=True, **kwargs):
        return _TinyResNetLike(num_classes)

    monkeypatch.setattr(train_mod, "build_frozen_resnet50", _builder)
    monkeypatch.setattr(
        train_mod, "ensure_training_preconditions", lambda *, allow_weight_download: _FAKE_RUNTIME
    )
    return train_mod


def _run(train_mod, split_dir, corpus, out, *, epochs, resume=False, patience=5):
    return train_mod.run_training(
        split_dir=split_dir,
        dataset_root=corpus,
        output_dir=out,
        epochs=epochs,
        batch_size=8,
        patience=patience,
        image_size=MODEL_IMAGE_SIZE,
        device=torch.device("cpu"),
        confirm_training=True,
        smoke_test_passed=True,
        resume=resume,
    )


def _events(captured: str):
    return [json.loads(line) for line in captured.splitlines() if line.strip().startswith("{")]


def test_resume_continues_from_the_last_complete_epoch_and_keeps_best_tracking(
    tmp_path, monkeypatch, capsys
):
    split_dir, corpus, rows = _synthetic_canary(tmp_path)
    train_mod = _stub_model_and_runtime(monkeypatch)
    out = tmp_path / "run"

    # Spy on the optimizer the loop is handed, to prove its state is restored.
    optimizer_state_sizes = []
    real_train_epoch = train_mod.train_epoch

    def _spy_train_epoch(model, loader, loss_fn, optimizer, device):
        optimizer_state_sizes.append(len(optimizer.state))
        return real_train_epoch(model, loader, loss_fn, optimizer, device)

    monkeypatch.setattr(train_mod, "train_epoch", _spy_train_epoch)

    first = _run(train_mod, split_dir, corpus, out, epochs=1)
    assert len(first["final_state"]["train_history"]) == 1
    best_after_first = first["best_val_loss"]
    capsys.readouterr()  # discard the first run's log lines

    # Simulate a Colab disconnect: the run is left mid-flight and marked incomplete.
    (out / "TRAINING_INCOMPLETE.txt").write_text("interrupted\n", encoding="utf-8")

    resumed = _run(train_mod, split_dir, corpus, out, epochs=3, resume=True)

    history = resumed["final_state"]["train_history"]
    assert [entry["epoch"] for entry in history] == [0, 1, 2]
    assert [entry["epoch"] for entry in resumed["final_state"]["val_history"]] == [0, 1, 2]

    # The best-loss / early-stopping trackers were restored, not reset: the reported
    # best is the minimum over all three epochs, not just the resumed ones.
    val_losses = [entry["loss"] for entry in resumed["final_state"]["val_history"]]
    assert resumed["best_val_loss"] == pytest.approx(min(val_losses))
    assert resumed["best_epoch"] == val_losses.index(min(val_losses))
    assert resumed["resumed_from"]["last_completed_epoch"] == 0

    # Nothing was deleted or retrained: epoch 0 kept its checkpoint, 1 and 2 are new.
    for epoch in (0, 1, 2):
        assert (out / "checkpoints" / f"epoch_{epoch:04d}" / "model.pt").is_file()
    assert (out / "best_checkpoint" / "model.pt").is_file()
    assert not (out / "TRAINING_INCOMPLETE.txt").exists()

    # The loop resumed at epoch 1 and the optimizer came back with its state.
    assert optimizer_state_sizes[0] == 0, "the first epoch starts with an empty optimizer"
    assert optimizer_state_sizes[1] > 0, "resume must restore the optimizer state"
    assert all(size > 0 for size in optimizer_state_sizes[1:])

    events = _events(capsys.readouterr().out)
    start = next(event for event in events if event.get("event") == "resume_start")
    assert start["start_epoch"] == 1
    assert start["history_epochs"] == 1
    assert start["best_val_loss"] == pytest.approx(best_after_first)

    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["resume_supported"] is True
    assert manifest["resumed_from"]["last_completed_epoch"] == 0

    # The resume pointer is refreshed, so a second disconnect resumes from epoch 2.
    pointer = json.loads((out / "resume_state.json").read_text(encoding="utf-8"))
    assert pointer["epochs_completed"] == 3
    assert pointer["resume_supported"] is True


def test_resume_without_a_checkpoint_explains_the_next_step(tmp_path, monkeypatch):
    split_dir, corpus, _ = _synthetic_canary(tmp_path)
    train_mod = _stub_model_and_runtime(monkeypatch)
    out = tmp_path / "never_started"
    out.mkdir()

    with pytest.raises(FileNotFoundError) as excinfo:
        _run(train_mod, split_dir, corpus, out, epochs=2, resume=True)
    message = str(excinfo.value)
    assert "Nothing to resume" in message or "Cannot resume" in message
    assert "overwrite" in message or "without --resume" in message


def test_resume_and_overwrite_together_are_rejected(tmp_path, monkeypatch):
    split_dir, corpus, _ = _synthetic_canary(tmp_path)
    train_mod = _stub_model_and_runtime(monkeypatch)

    with pytest.raises(ValueError, match="contradictory"):
        train_mod.run_training(
            split_dir=split_dir,
            dataset_root=corpus,
            output_dir=tmp_path / "run",
            epochs=2,
            image_size=MODEL_IMAGE_SIZE,
            device=torch.device("cpu"),
            confirm_training=True,
            smoke_test_passed=True,
            resume=True,
            overwrite=True,
        )


def test_a_completed_run_refuses_to_be_resumed_for_the_same_epoch_count(tmp_path, monkeypatch):
    split_dir, corpus, _ = _synthetic_canary(tmp_path)
    train_mod = _stub_model_and_runtime(monkeypatch)
    out = tmp_path / "run"

    _run(train_mod, split_dir, corpus, out, epochs=1)

    with pytest.raises(ValueError, match="already finished"):
        _run(train_mod, split_dir, corpus, out, epochs=1, resume=True)


def test_a_completed_run_can_be_extended_by_resuming_with_more_epochs(tmp_path, monkeypatch):
    split_dir, corpus, _ = _synthetic_canary(tmp_path)
    train_mod = _stub_model_and_runtime(monkeypatch)
    out = tmp_path / "run"

    _run(train_mod, split_dir, corpus, out, epochs=1)
    extended = _run(train_mod, split_dir, corpus, out, epochs=2, resume=True)

    assert [entry["epoch"] for entry in extended["final_state"]["train_history"]] == [0, 1]
    val_losses = [entry["loss"] for entry in extended["final_state"]["val_history"]]
    assert extended["best_val_loss"] == pytest.approx(min(val_losses))
    assert (out / "checkpoints" / "epoch_0001" / "model.pt").is_file()


def test_resume_never_reads_the_test_partition(tmp_path, monkeypatch):
    """The canary has no test rows, so nothing to leak; assert that explicitly."""
    split_dir, corpus, rows = _synthetic_canary(tmp_path)
    assert not [row for row in rows if row["partition"] == "test"]
    train_mod = _stub_model_and_runtime(monkeypatch)
    out = tmp_path / "run"

    _run(train_mod, split_dir, corpus, out, epochs=1)
    _run(train_mod, split_dir, corpus, out, epochs=2, resume=True)

    assert not (out / "final_test_evaluation").exists()


# ---------------------------------------------------------------------------
# Checkpoints are written atomically (a disconnect must not leave a torn model.pt)
# ---------------------------------------------------------------------------

def _checkpoint_pieces(tmp_path):
    out = tmp_path / "run"
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    config = {"num_classes": len(BAND_ORDER), "image_size": MODEL_IMAGE_SIZE}
    return out, model, optimizer, config, initial_training_state()


def test_a_checkpoint_is_written_atomically_and_loads(tmp_path):
    import colab.train as train_mod

    out, model, optimizer, config, state = _checkpoint_pieces(tmp_path)
    train_mod.save_checkpoint(out, state, model, optimizer, config, 0, is_best=True)

    epoch_dir = out / "checkpoints" / "epoch_0000"
    payload = torch.load(epoch_dir / "model.pt", map_location="cpu", weights_only=True)
    assert "model_state_dict" in payload
    assert "optimizer_state_dict" in payload

    # No temp files survive a successful write, anywhere in the run.
    assert not list(out.rglob("*.tmp"))

    # The best copy is byte-identical to the epoch checkpoint it mirrors.
    assert (out / "best_checkpoint" / "model.pt").read_bytes() == (epoch_dir / "model.pt").read_bytes()
    assert json.loads((epoch_dir / "training_state.json").read_text(encoding="utf-8"))["is_best"] is True


def test_a_failed_checkpoint_write_keeps_the_previous_checkpoint_and_no_temp_file(
    tmp_path, monkeypatch
):
    import colab.train as train_mod

    out, model, optimizer, config, state = _checkpoint_pieces(tmp_path)
    train_mod.save_checkpoint(out, state, model, optimizer, config, 0, is_best=True)
    good = (out / "checkpoints" / "epoch_0000" / "model.pt").read_bytes()

    def _explode(payload, path, *args, **kwargs):
        # Simulate a crash part-way through the write.
        Path(path).write_bytes(b"partial garbage")
        raise OSError("disk went away")

    monkeypatch.setattr(train_mod.torch, "save", _explode)
    with pytest.raises(OSError, match="disk went away"):
        train_mod.save_checkpoint(out, state, model, optimizer, config, 1, is_best=True)
    monkeypatch.undo()

    # The good checkpoint is untouched, no half-written file remains, and the failed
    # epoch is not advertised as a resume point.
    assert (out / "checkpoints" / "epoch_0000" / "model.pt").read_bytes() == good
    assert (out / "best_checkpoint" / "model.pt").read_bytes() == good
    assert not list(out.rglob("*.tmp"))
    assert not (out / "checkpoints" / "epoch_0001" / "model.pt").exists()
    epoch_dir, index = train_mod.latest_epoch_checkpoint(out)
    assert index == 0 and epoch_dir == out / "checkpoints" / "epoch_0000"


def test_latest_epoch_checkpoint_ignores_an_epoch_without_weights(tmp_path):
    import colab.train as train_mod

    out, model, optimizer, config, state = _checkpoint_pieces(tmp_path)
    train_mod.save_checkpoint(out, state, model, optimizer, config, 1, is_best=True)

    # A crash while saving epoch 2 leaves the metadata but no usable weights.
    broken = out / "checkpoints" / "epoch_0002"
    broken.mkdir(parents=True)
    (broken / "training_state.json").write_text(json.dumps({"state": state}), encoding="utf-8")

    epoch_dir, index = train_mod.latest_epoch_checkpoint(out)
    assert index == 1
    assert epoch_dir == out / "checkpoints" / "epoch_0001"
