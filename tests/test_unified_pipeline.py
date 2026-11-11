"""Tests for the unified one-click pipeline (``src/``).

These tests are self-contained: they create small synthetic fixtures in temporary
directories and never touch the real dataset, the audit artefacts, or the network.

Synthetic fixtures exist **only** to exercise software behaviour. No metric
produced by these tests is a research result.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import metrics as M  # noqa: E402
from src.checkpoints import (  # noqa: E402
    build_fingerprint,
    compare_fingerprints,
    find_reusable,
    load_checkpoint,
    save_checkpoint,
)
from src.config import ConfigError, load_config, validate_config  # noqa: E402
from src.data import (  # noqa: E402
    LESION_ORDER,
    build_idrid_pairs,
    class_weights_from_rows,
    _resize_and_pad,
)
from src.dataset_downloaders.base import (  # noqa: E402
    DownloadError,
    UnsafeArchiveError,
    download_file,
    safe_extract_zip,
)
from src.hardware import detect_hardware, resolve_device  # noqa: E402
from src.models import build_resnet50, build_unet4  # noqa: E402
from src.taxonomy import (  # noqa: E402
    LabelMapper,
    UnknownLabelError,
    cross_check_mendeley_policy,
)

MENDELEY_FOLDERS = {
    "11.Normal Fundus": "Normalfundus",
    "3.Mild DR": "MildDR",
    "4.Moderate DR": "ModerateDR",
    "5.Severe DR": "SevereDR",
    "6.Proliferate DR": "ProliferateDR",
    "13. Very Severe NPDR": "VerySevereNPDR",
    "12. Advanced PDR": "AdvancedPDR",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_config(tmp_path: Path, **overrides) -> Path:
    """Copy the repository config into tmp_path with path overrides applied."""
    base = yaml.safe_load((PROJECT_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    base["paths"].update(
        {
            "outputs_dir": "outputs",
            "cache_dir": "cache",
            "runs_dir": "runs",
            "logs_dir": "logs",
            "datasets_dir": "datasets",
            "mendeley_dir": "datasets/mendeley",
            "mmrdr_dir": "datasets/mmrdr",
            "idrid_dir": "datasets/idrid",
            "legacy_fundus_root": "",
            "idrid_manual_root": "",
        }
    )
    for dotted, value in overrides.items():
        node = base
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    target = Path(tmp_path) / "config.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    return target


def _make_fundus_corpus(root: Path, per_class: int = 4, seed: int = 1) -> Path:
    """Create a tiny Mendeley-shaped corpus: one folder per DR grade."""
    rng = np.random.default_rng(seed)
    grades = {0: 40, 1: 60, 2: 80, 3: 110, 4: 150}
    for folder, stem in MENDELEY_FOLDERS.items():
        directory = root / folder
        directory.mkdir(parents=True, exist_ok=True)
        grade = 0 if folder == "11.Normal Fundus" else (
            1 if "Mild" in folder else 2 if "Moderate" in folder else
            3 if "Severe" in folder else 4
        )
        for index in range(per_class):
            width, height = 48, 64
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            canvas[..., 0] = grades[grade]
            image = Image.fromarray(canvas)
            draw = ImageDraw.Draw(image)
            draw.ellipse([4, 3, width - 4, height - 3], outline=(200, 140, 80), width=2)
            for _ in range((grade + 1) * 3):
                x = int(rng.integers(8, width - 8))
                y = int(rng.integers(6, height - 8))
                radius = int(rng.integers(2, 4))
                draw.ellipse([x, y, x + radius, y + radius], fill=(240, 220, 90))
            image.save(directory / f"{stem}{500 + index}_left_{index}_1.jpeg")
    # A cataract folder must be excluded, never mapped to a DR grade.
    cataract = root / "7.Cataract"
    cataract.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((48, 64, 3), 90, dtype=np.uint8)).save(
        cataract / "Cataract900_left_0_0.jpeg"
    )
    return root


def _make_idrid(root: Path, count: int = 6, seed: int = 2) -> Path:
    images = root / "1. Original Images" / "a. Training Set"
    groundtruth = root / "2. All Segmentation Groundtruths" / "a. Training Set"
    subdirs = {
        "MA": "1. Microaneurysms",
        "HE": "2. Haemorrhages",
        "EX": "3. Hard Exudates",
        "SE": "4. Soft Exudates",
    }
    images.mkdir(parents=True, exist_ok=True)
    for name in subdirs.values():
        (groundtruth / name).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for index in range(count):
        stem = f"IDRiD_{index + 1:02d}"
        height, width = 56, 72
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[..., 1] = 40
        image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image)
        draw.ellipse([6, 4, width - 6, height - 4], outline=(190, 140, 80), width=2)
        for channel, subdir in subdirs.items():
            mask = Image.new("L", (width, height), 0)
            mask_draw = ImageDraw.Draw(mask)
            cx = int(rng.integers(14, width - 14))
            cy = int(rng.integers(12, height - 12))
            radius = int(rng.integers(2, 5))
            mask_draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=255)
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(240, 230, 120))
            mask.save(groundtruth / subdir / f"{stem}_{channel}.tif")
        image.save(images / f"{stem}.jpg")
    return root


@pytest.fixture()
def pipeline_env(tmp_path):
    """A temporary pipeline with a tiny corpus and its own config."""
    corpus = _make_fundus_corpus(tmp_path / "datasets" / "mendeley")
    config_path = _write_config(tmp_path)
    config = load_config(config_path, root=tmp_path)
    return {"root": tmp_path, "config_path": config_path, "config": config, "corpus": corpus}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_config_loads_and_validates(pipeline_env):
    cfg = pipeline_env["config"]
    assert cfg.classes == ["No apparent DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "PDR"]
    assert cfg.lesion_channels == ["MA", "HE", "EX", "SE"]
    assert cfg.seed == 12345
    assert len(cfg.display_names) == 5


def test_config_rejects_wrong_class_count(tmp_path):
    config_path = _write_config(tmp_path, **{"taxonomy.classes": ["A", "B", "C"]})
    with pytest.raises(ConfigError):
        load_config(config_path, root=tmp_path)


def test_config_rejects_bad_lesion_order(tmp_path):
    config_path = _write_config(tmp_path, **{"taxonomy.lesion_channels": ["MA", "EX", "HE", "SE"]})
    with pytest.raises(ConfigError):
        load_config(config_path, root=tmp_path)


def test_config_rejects_bad_split_ratios(tmp_path):
    config_path = _write_config(tmp_path, **{"split.ratios": {"train": 0.5, "validation": 0.2, "test": 0.2}})
    with pytest.raises(ConfigError):
        load_config(config_path, root=tmp_path)


def test_config_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml", root=tmp_path)


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

def test_mendeley_mapping_matches_repository_policy(pipeline_env):
    result = cross_check_mendeley_policy(pipeline_env["config"])
    assert result["consistent"] is True, result


def test_mendeley_merges_are_flagged(pipeline_env):
    mapper = LabelMapper(pipeline_env["config"])
    very_severe = mapper.resolve("mendeley", "13. Very Severe NPDR")
    advanced = mapper.resolve("mendeley", "12. Advanced PDR")
    assert very_severe.mapped_label == 3 and very_severe.decision == "mapped_merged"
    assert advanced.mapped_label == 4 and advanced.decision == "mapped_merged"


def test_non_dr_folder_is_excluded_with_a_reason(pipeline_env):
    mapper = LabelMapper(pipeline_env["config"])
    decision = mapper.resolve("mendeley", "7.Cataract")
    assert decision.mapped_label is None
    assert decision.decision == "excluded"
    assert "not a DR severity grade" in decision.exclusion_reason


def test_unknown_label_raises_instead_of_guessing(pipeline_env):
    mapper = LabelMapper(pipeline_env["config"])
    with pytest.raises(UnknownLabelError):
        mapper.resolve("mendeley", "99. Invented DR Grade")


def test_mmrdr_grades_map_one_to_one(pipeline_env):
    mapper = LabelMapper(pipeline_env["config"])
    for grade, expected in [("0", 0), ("1", 1), ("2", 2), ("3", 3), ("4", 4)]:
        assert mapper.resolve("mmrdr", grade).mapped_label == expected
    with pytest.raises(UnknownLabelError):
        mapper.resolve("mmrdr", "5")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def test_resnet50_has_five_output_logits():
    model = build_resnet50(num_classes=5, pretrained=False)
    model.eval()
    with torch.no_grad():
        logits = model(torch.zeros(2, 3, 224, 224))
    assert tuple(logits.shape) == (2, 5)


def test_resnet50_head_shape_matches_legacy_builder():
    """The unified and legacy builders must agree on the classifier head."""
    from src.legacy_bridge import load_colab_helpers

    helpers = load_colab_helpers(PROJECT_ROOT)
    legacy = helpers.build_frozen_resnet50(num_classes=5, pretrained=False)
    unified = build_resnet50(num_classes=5, pretrained=False)
    assert legacy.fc.in_features == unified.fc.in_features
    assert legacy.fc.out_features == unified.fc.out_features == 5
    assert set(legacy.state_dict().keys()) == set(unified.state_dict().keys())


def test_unet_outputs_four_lesion_channels():
    model = build_unet4(base_filters=8)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 128, 128))
    assert tuple(output.shape) == (1, 4, 128, 128)


def test_unet_probabilities_are_bounded():
    model = build_unet4(base_filters=8)
    probabilities = model.predict_proba(torch.zeros(1, 3, 64, 64))
    assert float(probabilities.min()) >= 0.0 and float(probabilities.max()) <= 1.0


# ---------------------------------------------------------------------------
# Preprocessing and data
# ---------------------------------------------------------------------------

def test_resize_and_pad_preserves_aspect_ratio_without_cropping():
    image = Image.fromarray((np.random.rand(40, 80, 3) * 255).astype(np.uint8))
    result = _resize_and_pad(image, 224, PROJECT_ROOT)
    assert result.size == (224, 224)
    box = result.info["content_box"]
    content_width = box[2] - box[0]
    content_height = box[3] - box[1]
    # 2:1 source inside a square canvas: full width used, letterboxed vertically.
    assert content_width == 224
    assert content_height == 112
    assert box[1] == (224 - 112) // 2


def test_load_rgb_converts_to_three_channels(tmp_path):
    from src.data import load_rgb

    gray = tmp_path / "gray.png"
    Image.fromarray(np.full((10, 12), 128, dtype=np.uint8)).save(gray)
    loaded = load_rgb(gray)
    assert loaded.mode == "RGB"
    assert loaded.size == (12, 10)


def test_idrid_pairing_builds_complete_sets(tmp_path):
    _make_idrid(tmp_path / "idrid", count=4)
    config_path = _write_config(
        tmp_path, **{"paths.idrid_dir": str(tmp_path / "idrid")}
    )
    cfg = load_config(config_path, root=tmp_path)
    pairing = build_idrid_pairs(cfg, _null_logger())
    assert pairing["available"] is True
    assert pairing["paired_count"] == 4
    for pair in pairing["pairs"]:
        for channel in LESION_ORDER:
            assert Path(pair[f"mask_{channel}"]).is_file()


def test_idrid_pairing_detects_a_missing_mask(tmp_path):
    root = _make_idrid(tmp_path / "idrid", count=3)
    victim = next((root / "2. All Segmentation Groundtruths" / "a. Training Set"
                   / "3. Hard Exudates").glob("*.tif"))
    victim.unlink()
    config_path = _write_config(tmp_path, **{"paths.idrid_dir": str(root)})
    cfg = load_config(config_path, root=tmp_path)
    pairing = build_idrid_pairs(cfg, _null_logger())
    assert pairing["paired_count"] == 2
    assert pairing["images_without_all_masks"] == 1


def test_segmentation_dataset_mask_channel_order(tmp_path):
    from src.data import SegmentationDataset

    root = _make_idrid(tmp_path / "idrid", count=2)
    config_path = _write_config(tmp_path, **{"paths.idrid_dir": str(root)})
    cfg = load_config(config_path, root=tmp_path)
    pairing = build_idrid_pairs(cfg, _null_logger())
    dataset = SegmentationDataset(pairing["pairs"], cfg, image_size=32)
    image, masks, meta = dataset[0]
    assert tuple(image.shape) == (3, 32, 32)
    assert tuple(masks.shape) == (4, 32, 32)
    assert masks.dtype == torch.float32
    assert set(np.unique(masks.numpy()).tolist()) <= {0.0, 1.0}
    assert meta["image_id"].startswith("IDRiD_")


# ---------------------------------------------------------------------------
# Manifest preparation, duplicates and leakage
# ---------------------------------------------------------------------------

def test_prepare_creates_manifests_and_audits(pipeline_env, monkeypatch):
    from src.prepare import run_prepare

    monkeypatch.setenv("DR_FUNDUS_ROOT", str(pipeline_env["corpus"]))
    summary = run_prepare(pipeline_env["config"], _null_logger())

    outputs = pipeline_env["root"] / "outputs" / "manifests"
    for name in ("train_manifest.csv", "validation_manifest.csv", "test_manifest.csv",
                 "dataset_summary.json", "label_mapping_audit.csv", "duplicate_audit.csv"):
        assert (outputs / name).is_file(), name

    assert summary["counts"]["images_included"] > 0
    # The cataract folder must never enter the DR cohort.
    assert summary["counts"]["excluded_by_reason"]
    assert any("not a DR severity grade" in reason for reason in summary["counts"]["excluded_by_reason"])
    # Only Mendeley is present here, so the run must call itself partial.
    assert summary["mode"] == "partial"
    assert "NOT a combined" in summary["combined_note"]


def test_label_mapping_audit_records_merges_and_exclusions(pipeline_env, monkeypatch):
    from src.prepare import run_prepare

    monkeypatch.setenv("DR_FUNDUS_ROOT", str(pipeline_env["corpus"]))
    run_prepare(pipeline_env["config"], _null_logger())
    audit = pipeline_env["root"] / "outputs" / "manifests" / "label_mapping_audit.csv"
    with audit.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    decisions = {row["original_label"]: row for row in rows}
    assert decisions["13. Very Severe NPDR"]["mapping_decision"] == "mapped_merged"
    assert decisions["7.Cataract"]["mapping_decision"] == "excluded"
    assert decisions["7.Cataract"]["exclusion_reason"]
    assert all(row["image_count"] for row in rows)


def test_split_partitions_are_disjoint_and_cover_everything(pipeline_env, monkeypatch):
    from src.prepare import run_prepare

    monkeypatch.setenv("DR_FUNDUS_ROOT", str(pipeline_env["corpus"]))
    summary = run_prepare(pipeline_env["config"], _null_logger())

    manifests = pipeline_env["root"] / "outputs" / "manifests"
    paths = []
    for name in ("train_manifest.csv", "validation_manifest.csv", "test_manifest.csv"):
        with (manifests / name).open(newline="", encoding="utf-8") as handle:
            paths.append({row["image_path"] for row in csv.DictReader(handle)})
    assert not (paths[0] & paths[1])
    assert not (paths[0] & paths[2])
    assert not (paths[1] & paths[2])
    total = sum(len(part) for part in paths)
    assert total == summary["counts"]["images_included"]


def test_duplicate_detection_finds_exact_and_near_copies(pipeline_env, monkeypatch):
    from src.prepare import run_prepare

    corpus = pipeline_env["corpus"]
    source = sorted((corpus / "3.Mild DR").glob("*.jpeg"))[0]
    shutil.copy(source, corpus / "3.Mild DR" / "dupe500_left_0_0.jpeg")
    Image.open(source).save(corpus / "3.Mild DR" / "near500_left_0_0.jpeg", quality=88)

    monkeypatch.setenv("DR_FUNDUS_ROOT", str(corpus))
    summary = run_prepare(pipeline_env["config"], _null_logger())
    duplicates = summary["duplicates"]
    assert duplicates["exact_duplicate_groups"] >= 1
    assert duplicates["near_duplicate_pairs_same_label"] >= 1

    audit = pipeline_env["root"] / "outputs" / "manifests" / "duplicate_audit.csv"
    with audit.open(newline="", encoding="utf-8") as handle:
        kinds = {row["kind"] for row in csv.DictReader(handle)}
    assert "exact" in kinds


def test_no_duplicate_reaches_two_partitions(pipeline_env, monkeypatch):
    """The core leakage guarantee: identical bytes cannot straddle partitions."""
    from src.prepare import run_prepare

    corpus = pipeline_env["corpus"]
    source = sorted((corpus / "3.Mild DR").glob("*.jpeg"))[0]
    shutil.copy(source, corpus / "3.Mild DR" / "copy700_left_0_0.jpeg")

    monkeypatch.setenv("DR_FUNDUS_ROOT", str(corpus))
    run_prepare(pipeline_env["config"], _null_logger())

    manifests = pipeline_env["root"] / "outputs" / "manifests"
    hash_to_partition = {}
    for name, partition in (("train_manifest.csv", "train"),
                            ("validation_manifest.csv", "validation"),
                            ("test_manifest.csv", "test")):
        with (manifests / name).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                digest = row["image_hash"]
                assert digest not in hash_to_partition or hash_to_partition[digest] == partition
                hash_to_partition[digest] = partition


def test_patient_groups_never_span_partitions(pipeline_env, monkeypatch):
    from src.prepare import run_prepare

    monkeypatch.setenv("DR_FUNDUS_ROOT", str(pipeline_env["corpus"]))
    run_prepare(pipeline_env["config"], _null_logger())

    manifests = pipeline_env["root"] / "outputs" / "manifests"
    group_to_partition = {}
    for name, partition in (("train_manifest.csv", "train"),
                            ("validation_manifest.csv", "validation"),
                            ("test_manifest.csv", "test")):
        with (manifests / name).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                group = row["group_id"]
                if not group:
                    continue
                assert group_to_partition.get(group, partition) == partition
                group_to_partition[group] = partition


def test_prepare_refuses_when_no_data_exists(tmp_path):
    from src.prepare import PrepareError, run_prepare

    config_path = _write_config(tmp_path)
    cfg = load_config(config_path, root=tmp_path)
    with pytest.raises(PrepareError):
        run_prepare(cfg, _null_logger())


def test_class_weights_are_inverse_frequency():
    rows = (
        [{"mapped_label_int": 0}] * 8
        + [{"mapped_label_int": 1}] * 2
    )
    weights = class_weights_from_rows(rows, 5)
    assert weights[0] < weights[1]
    assert len(weights) == 5


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_confusion_and_kappa():
    y_true = np.array([0, 1, 2, 3, 4, 1])
    y_pred = np.array([0, 1, 2, 3, 3, 1])
    matrix = M.confusion_matrix(y_true, y_pred, 5)
    assert int(matrix.sum()) == 6
    assert 0.0 <= M.cohen_kappa(y_true, y_pred, 5) <= 1.0


def test_per_class_specificity_and_sensitivity():
    names = ["No apparent DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "PDR"]
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    y_pred = np.array([0, 0, 1, 2, 2, 2, 3, 4, 4, 4])
    result = M.classification_metrics(y_true, y_pred, class_names=names, compute_auc=False)
    stats = result["per_class"]["Mild NPDR"]
    assert stats["precision"] == 1.0
    assert stats["sensitivity"] == stats["recall"]
    assert 0.0 <= stats["specificity"] <= 1.0
    assert result["num_samples"] == 10


def test_auc_is_reported_as_unavailable_when_undefined():
    names = ["No apparent DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "PDR"]
    y_true = np.zeros(5, dtype=int)  # only one class present
    probabilities = np.tile(np.array([0.6, 0.1, 0.1, 0.1, 0.1]), (5, 1))
    info = M.one_vs_rest_auc(probabilities, y_true, 5)
    assert info["macro"] is None
    assert info["reason"]


def test_dice_and_iou_are_one_for_perfect_overlap():
    probabilities = np.zeros((1, 4, 8, 8))
    targets = np.zeros((1, 4, 8, 8))
    probabilities[0, :, 2:5, 2:5] = 1.0
    targets[0, :, 2:5, 2:5] = 1.0
    dice = M.dice_per_channel(probabilities, targets)
    iou = M.iou_per_channel(probabilities, targets)
    assert np.allclose(dice, 1.0)
    assert np.allclose(iou, 1.0)


def test_dice_is_zero_for_disjoint_masks():
    probabilities = np.zeros((1, 4, 8, 8))
    targets = np.zeros((1, 4, 8, 8))
    probabilities[0, :, 0:2, 0:2] = 1.0
    targets[0, :, 4:6, 4:6] = 1.0
    assert np.all(M.dice_per_channel(probabilities, targets) < 1e-5)


def test_ece_is_zero_for_perfectly_calibrated_predictions():
    probabilities = np.tile(np.array([1.0, 0.0, 0.0, 0.0, 0.0]), (10, 1))
    y_true = np.zeros(10, dtype=int)
    assert M.expected_calibration_error(probabilities, y_true, bins=10) == pytest.approx(0.0, abs=1e-9)


def test_ece_is_large_for_confidently_wrong_predictions():
    probabilities = np.tile(np.array([0.95, 0.0125, 0.0125, 0.0125, 0.0125]), (10, 1))
    y_true = np.ones(10, dtype=int)  # every prediction is wrong
    assert M.expected_calibration_error(probabilities, y_true, bins=10) > 0.9


def test_calibration_rejects_unknown_binning_strategy():
    probabilities = np.tile(np.array([0.6, 0.1, 0.1, 0.1, 0.1]), (4, 1))
    y_true = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError):
        M.calibration_bins(probabilities, y_true, bins=5, strategy="equal_mass")


def test_entropy_is_maximal_for_a_uniform_distribution():
    uniform = np.full((1, 5), 0.2)
    concentrated = np.array([[0.97, 0.01, 0.01, 0.005, 0.005]])
    assert M.predictive_entropy(uniform)[0] == pytest.approx(np.log(5), abs=1e-6)
    assert M.predictive_entropy(concentrated)[0] < M.predictive_entropy(uniform)[0]


def test_uncertainty_flags_low_confidence_predictions():
    confident = np.array([[0.9, 0.025, 0.025, 0.025, 0.025]])
    unsure = np.array([[0.3, 0.25, 0.2, 0.15, 0.1]])
    report = M.uncertainty_report(
        np.vstack([confident, unsure]), np.array([0, 1]),
        entropy_threshold=1.10, confidence_threshold=0.60,
    )
    assert report["review_flags"]["either"] == 1
    assert "NOT a confirmed diagnosis" in report["policy_note"]


def test_binarise_cam_returns_a_boolean_mask():
    cam = np.linspace(0.0, 1.0, 100).reshape(10, 10)
    mask = M.binarise_cam(cam, 90)
    assert mask.dtype == bool
    assert mask.sum() >= 1


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def test_checkpoint_round_trip(tmp_path):
    model = build_resnet50(num_classes=5, pretrained=False)
    fingerprint = build_fingerprint(
        architecture="resnet50", classes=["a", "b", "c", "d", "e"],
        image_size=224, manifest_version="test-v1",
    )
    paths = save_checkpoint(
        tmp_path / "ckpt", model=model, optimizer=None, scheduler=None, epoch=3,
        history=[{"epoch": 0}], best_metric=0.5, best_epoch=0, fingerprint=fingerprint,
    )
    payload = load_checkpoint(Path(paths["model"]))
    assert payload["epoch"] == 3
    assert payload["fingerprint"]["digest"] == fingerprint["digest"]

    restored = build_resnet50(num_classes=5, pretrained=False)
    restored.load_state_dict(payload["model_state"])
    for left, right in zip(model.parameters(), restored.parameters()):
        assert torch.allclose(left, right)


def test_fingerprint_mismatch_is_detected(tmp_path):
    model = build_resnet50(num_classes=5, pretrained=False)
    stored = build_fingerprint(
        architecture="resnet50", classes=["a", "b", "c", "d", "e"],
        image_size=224, manifest_version="v1",
    )
    save_checkpoint(
        tmp_path / "ckpt", model=model, optimizer=None, scheduler=None, epoch=1,
        history=[], best_metric=1.0, best_epoch=0, fingerprint=stored,
    )
    changed = build_fingerprint(
        architecture="resnet50", classes=["a", "b", "c", "d", "e"],
        image_size=128, manifest_version="v2",
    )
    reusable = find_reusable(tmp_path / "ckpt", changed)
    assert reusable["compatible"] is False
    assert any("image_size" in reason for reason in reusable["reasons"])
    assert any("manifest_version" in reason for reason in reusable["reasons"])


def test_class_order_change_invalidates_a_checkpoint():
    stored = build_fingerprint(
        architecture="resnet50", classes=["a", "b", "c", "d", "e"],
        image_size=224, manifest_version="v1",
    )
    reordered = build_fingerprint(
        architecture="resnet50", classes=["b", "a", "c", "d", "e"],
        image_size=224, manifest_version="v1",
    )
    compatible, reasons = compare_fingerprints(stored, reordered)
    assert compatible is False
    assert any("class order" in reason for reason in reasons)


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def test_hardware_report_is_self_consistent():
    report = detect_hardware(disk_probe=PROJECT_ROOT)
    assert report.os_name
    assert report.python_version.startswith("3.")
    assert report.cpu_count >= 1
    assert report.recommended_classification_batch_size >= 1
    assert report.recommended_segmentation_batch_size >= 1
    # A present GPU must not be reported as usable without torch agreeing.
    if report.gpu_present_but_unusable and not report.cuda_available:
        assert any("nvidia-smi" in note for note in report.notes)


def test_mixed_precision_disabled_without_cuda():
    report = detect_hardware(mixed_precision_preference="auto", disk_probe=PROJECT_ROOT)
    if not report.cuda_available:
        assert report.mixed_precision is False
        assert report.selected_device == "cpu"


def test_resolve_device_falls_back_to_cpu():
    device = resolve_device("cuda:0")
    assert device.type in {"cpu", "cuda"}


# ---------------------------------------------------------------------------
# Download machinery
# ---------------------------------------------------------------------------

def test_download_file_copies_and_verifies(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"x" * 4096)
    destination = tmp_path / "nested" / "target.bin"
    result = download_file(source.as_uri(), destination, expected_size=4096, logger=_null_logger())
    assert result.read_bytes() == b"x" * 4096


def test_download_file_rejects_a_size_mismatch(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"y" * 100)
    with pytest.raises(DownloadError):
        download_file(source.as_uri(), tmp_path / "out.bin", expected_size=999, max_retries=1)


def test_download_file_rejects_a_checksum_mismatch(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"z" * 64)
    with pytest.raises(DownloadError):
        download_file(
            source.as_uri(), tmp_path / "out.bin",
            expected_sha256="0" * 64, max_retries=1,
        )


def test_safe_extract_rejects_directory_traversal(tmp_path):
    archive_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../../escaped.txt", "nope")
    with pytest.raises(UnsafeArchiveError):
        safe_extract_zip(archive_path, tmp_path / "extract", logger=_null_logger())
    assert not (tmp_path.parent / "escaped.txt").exists()


def test_safe_extract_accepts_a_normal_archive(tmp_path):
    archive_path = tmp_path / "good.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("folder/file.txt", "ok")
    destination = safe_extract_zip(archive_path, tmp_path / "extract", logger=_null_logger())
    assert (destination / "folder" / "file.txt").read_text() == "ok"


# ---------------------------------------------------------------------------
# Pipeline error handling
# ---------------------------------------------------------------------------

def test_cli_validate_config_succeeds():
    completed = subprocess.run(
        [sys.executable, "-m", "src.pipeline", "--validate-config"],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Configuration OK" in completed.stdout


def test_evaluation_without_a_checkpoint_fails_clearly(tmp_path):
    from src.evaluate_classification import EvaluationError, run_evaluation

    config_path = _write_config(tmp_path)
    cfg = load_config(config_path, root=tmp_path)
    with pytest.raises(EvaluationError):
        run_evaluation(cfg, _null_logger())


def test_clinical_fusion_refuses_to_fabricate():
    from src import clinical_status

    assert clinical_status.is_implemented() is False
    assert clinical_status.STATUS_RECORD["status"] == "proposed"
    with pytest.raises(NotImplementedError):
        clinical_status.refuse_synthetic_clinical_data()


def test_training_refuses_without_manifests(tmp_path):
    from src.train_classifier import TrainingError, run_training

    config_path = _write_config(tmp_path)
    cfg = load_config(config_path, root=tmp_path)
    with pytest.raises(TrainingError):
        run_training(cfg, _null_logger(), resume=False)


# ---------------------------------------------------------------------------
# End-to-end smoke test
# ---------------------------------------------------------------------------

def test_smoke_full_chain_on_synthetic_fixture(tmp_path, monkeypatch):
    """Load an image, run both models, score Dice/IoU, save and reload a checkpoint.

    This validates plumbing only. The values are meaningless as research results.
    """
    from src.data import ClassifierDataset, SegmentationDataset, load_manifest_rows
    from src.prepare import run_prepare

    corpus = _make_fundus_corpus(tmp_path / "datasets" / "mendeley", per_class=4)
    idrid_root = _make_idrid(tmp_path / "datasets" / "idrid", count=6)
    config_path = _write_config(
        tmp_path,
        **{
            "paths.idrid_dir": str(idrid_root),
            "preprocessing.classification.image_size": 64,
            "preprocessing.segmentation.image_size": 32,
            "classification.batch_size": 2,
            "classification.num_workers": 0,
            "segmentation.base_filters": 8,
        },
    )
    cfg = load_config(config_path, root=tmp_path)
    monkeypatch.setenv("DR_FUNDUS_ROOT", str(corpus))

    summary = run_prepare(cfg, _null_logger())
    assert summary["counts"]["images_included"] > 0

    # 1-3: classification image -> ResNet-50 -> five logits
    rows = load_manifest_rows(tmp_path / "outputs" / "manifests" / "train_manifest.csv")
    dataset = ClassifierDataset(rows[:4], cfg, train=False)
    tensor, label, meta = dataset[0]
    assert 0 <= label < 5
    assert meta["image_path"].endswith((".jpeg", ".jpg", ".png"))
    model = build_resnet50(num_classes=5, pretrained=False)
    model.eval()
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0))
    assert tuple(logits.shape) == (1, 5)

    # 4-6: IDRiD pair -> U-Net -> four channels
    pairing = build_idrid_pairs(cfg, _null_logger())
    assert pairing["available"] is True
    seg_dataset = SegmentationDataset(pairing["pairs"][:2], cfg, image_size=32)
    image, mask, _meta = seg_dataset[0]
    unet = build_unet4(base_filters=8)
    unet.eval()
    with torch.no_grad():
        logits = unet(image.unsqueeze(0))
    assert tuple(logits.shape) == (1, 4, 32, 32)

    # 7: sample Dice and IoU
    probabilities = torch.sigmoid(logits).numpy()
    targets = mask.numpy()[None, ...]
    dice = M.dice_per_channel(probabilities, targets)
    iou = M.iou_per_channel(probabilities, targets)
    assert dice.shape == (4,) and iou.shape == (4,)
    assert all(0.0 <= float(value) <= 1.0 for value in dice)

    # 8: checkpoint save and reload
    fingerprint = build_fingerprint(
        architecture="unet4", classes=list(cfg.lesion_channels),
        image_size=32, manifest_version=cfg.manifest_version,
    )
    paths = save_checkpoint(
        tmp_path / "ckpt", model=unet, optimizer=None, scheduler=None, epoch=0,
        history=[], best_metric=float(dice.mean()), best_epoch=0, fingerprint=fingerprint,
    )
    reloaded = build_unet4(base_filters=8)
    payload = load_checkpoint(Path(paths["model"]))
    reloaded.load_state_dict(payload["model_state"])
    reloaded.eval()
    with torch.no_grad():
        again = reloaded(image.unsqueeze(0))
    assert torch.allclose(logits, again, atol=1e-6)

    # 9: missing real datasets are reported, not fabricated
    from src.train_segmentation import run_training as run_seg_training

    empty_root = tmp_path / "empty_idrid"
    empty_root.mkdir()
    config_path_2 = _write_config(tmp_path / "second", **{"paths.idrid_dir": str(empty_root)})
    cfg2 = load_config(config_path_2, root=tmp_path / "second")
    result = run_seg_training(cfg2, _null_logger(), resume=False)
    assert result["status"] == "blocked"
    assert "IDRiD" in (result["detail"] or "")


def test_resize_and_pad_mask_preserves_palette_red_and_geometry(tmp_path):
    from src.data import SegmentationDataset, resize_and_pad_mask
    from src.legacy_bridge import load_colab_helpers

    load_colab_helpers(PROJECT_ROOT)

    # Simulate IDRiD palette TIFF: index 0 = black (0, 0, 0), index 1 = red (255, 0, 0).
    palette = [0, 0, 0, 255, 0, 0] + [0] * (254 * 3)
    palette_img = Image.fromarray(np.ones((32, 64), dtype=np.uint8), mode="P")
    palette_img.putpalette(palette)
    mask = resize_and_pad_mask(palette_img, 32)
    assert mask.shape == (32, 32)
    assert mask.dtype == bool
    assert not mask[:8].any() and not mask[24:].any()
    assert mask[8:24].all()

    empty = Image.fromarray(np.zeros((32, 64), dtype=np.uint8), mode="L")
    assert not resize_and_pad_mask(empty, 32).any()

    # Overwrite MA channel of a synthetic IDRiD fixture with a palette red TIFF.
    root = _make_idrid(tmp_path / "idrid_palette", count=1)
    ma_path = root / "2. All Segmentation Groundtruths" / "a. Training Set" / "1. Microaneurysms" / "IDRiD_01_MA.tif"
    ma_raw = np.zeros((56, 72), dtype=np.uint8)
    ma_raw[20:36, 28:44] = 1
    ma_tif = Image.fromarray(ma_raw, mode="P")
    ma_tif.putpalette(palette)
    ma_tif.save(ma_path)

    cfg = load_config(_write_config(tmp_path, **{"paths.idrid_dir": str(root)}), root=tmp_path)
    pairing = build_idrid_pairs(cfg, _null_logger())
    _image, masks, _meta = SegmentationDataset(pairing["pairs"], cfg, image_size=32)[0]
    assert float(masks[0].sum()) > 0.0


def test_classifier_resume_continues_incomplete_run_and_unfreezes(tmp_path, monkeypatch):
    from src import models as models_mod
    from src import train_classifier
    from src.legacy_bridge import load_colab_helpers
    from src.prepare import run_prepare

    load_colab_helpers(PROJECT_ROOT)
    real_dataloader = train_classifier._dataloader
    monkeypatch.setattr(
        train_classifier,
        "_dataloader",
        lambda ds, *, batch_size, shuffle, num_workers, seed: real_dataloader(
            ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, seed=seed
        ),
    )

    corpus = _make_fundus_corpus(tmp_path / "datasets" / "mendeley", per_class=3)
    cfg_path = _write_config(
        tmp_path,
        **{
            "classification.pretrained": False,
            "classification.epochs": 3,
            "classification.freeze_backbone_epochs": 1,
            "classification.early_stopping_patience": 3,
            "classification.batch_size": 4,
            "classification.num_workers": 0,
            "preprocessing.classification.image_size": 32,
        },
    )
    cfg = load_config(cfg_path, root=tmp_path)
    monkeypatch.setenv("DR_FUNDUS_ROOT", str(corpus))
    run_prepare(cfg, _null_logger())

    # Simulate an interrupted run at epoch 2/3 (0-indexed epoch=1) without run_summary.json.
    run_dir = cfg.path("paths.runs_dir", create=True) / "classification"
    fingerprint = build_fingerprint(
        architecture="resnet50",
        classes=list(cfg.classes),
        image_size=32,
        manifest_version=cfg.manifest_version,
        extra={"task": "five_class_classification"},
    )
    seed_model = models_mod.build_resnet50(
        num_classes=5, pretrained=False, allow_weight_download=False, freeze_backbone=True
    )
    history = [
        {"epoch": 1, "train_loss": 1.6, "train_accuracy": 0.2, "val_loss": 1.6, "val_accuracy": 0.2, "val_macro_f1": 0.1, "lr": 1e-4, "epoch_seconds": 0.1},
        {"epoch": 2, "train_loss": 1.5, "train_accuracy": 0.25, "val_loss": 1.5, "val_accuracy": 0.25, "val_macro_f1": 0.15, "lr": 1e-4, "epoch_seconds": 0.1},
    ]
    for target in (run_dir / "best_checkpoint", run_dir / "last_checkpoint"):
        save_checkpoint(
            target, model=seed_model, optimizer=None, scheduler=None,
            epoch=1, history=history, best_metric=0.15, best_epoch=1, fingerprint=fingerprint,
        )

    unfrozen_calls = []
    real_unfreeze = models_mod.unfreeze_all

    def spy_unfreeze(model):
        unfrozen_calls.append(True)
        return real_unfreeze(model)

    monkeypatch.setattr(models_mod, "unfreeze_all", spy_unfreeze)
    result = train_classifier.run_training(cfg, _null_logger(), force_retrain=False, resume=True)
    assert result["status"] == "completed"
    assert result["reused"] is False
    assert len(unfrozen_calls) == 1
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["epochs_completed"] == 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _null_logger():
    import logging

    logger = logging.getLogger("test-unified-pipeline")
    logger.handlers = [logging.NullHandler()]
    logger.propagate = False
    logger.setLevel(logging.CRITICAL)
    return logger
