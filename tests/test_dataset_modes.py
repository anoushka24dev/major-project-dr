"""Regression tests for the APTOS/DDR adapters, dataset modes and experiments.

Everything here is synthetic: small generated images and CSVs in ``tmp_path``. No
test touches the real datasets, the network, or a GPU, and no metric produced here
is a research result.

What is pinned
--------------
* APTOS and DDR images keep their own provenance and original labels.
* DDR grade 5 (ungradable) is excluded, never mapped to No DR.
* DDR's official partitions are preserved and its official test images never train.
* ``combined`` trains on every validated source; ``external`` holds the non-development
  sources out entirely, and the partition policy is enforced in code, not by convention.
* A missing source produces a blocked/skipped status with instructions — never fake data.
* The experiment series is driven by configuration, the final candidate inherits the
  best training configuration, and only validation data is used for post-hoc fitting.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class _NullLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


def _null_logger() -> _NullLogger:
    return _NullLogger()


def _write_config(tmp_path: Path, **overrides) -> Path:
    base = yaml.safe_load((PROJECT_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    base["paths"].update(
        {
            "outputs_dir": "outputs",
            "cache_dir": "cache",
            "runs_dir": "runs",
            "logs_dir": "logs",
            "checkpoints_dir": "checkpoints",
            "datasets_dir": "datasets",
            "mendeley_dir": "datasets/mendeley",
            "mmrdr_dir": "datasets/mmrdr",
            "aptos_dir": "datasets/aptos",
            "ddr_dir": "datasets/ddr",
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


def _image(path: Path, seed: int, size: int = 32) -> Path:
    rng = np.random.default_rng(seed)
    canvas = rng.integers(20, 220, size=(size, size, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)
    return path


def _make_mendeley(root: Path) -> Path:
    """Three DR grades, enough images for a three-way split."""
    for folder, seed_base in (
        ("11.Normal Fundus", 100),
        ("3.Mild DR", 200),
        ("6.Proliferate DR", 300),
    ):
        # Six per class: the group-balanced allocator needs more than four to fill
        # train, validation and test for every class.
        for index in range(6):
            _image(root / folder / f"image{seed_base + index}.jpeg", seed_base + index)
    return root


def _make_aptos(root: Path) -> Path:
    images = root / "train_images"
    rows = [
        ("aptos0001", 0),
        ("aptos0002", 1),
        ("aptos0003", 2),
        ("aptos0004", 3),
        ("aptos0005", 4),
        ("aptos0006", 4),
    ]
    for offset, (identifier, _grade) in enumerate(rows):
        _image(images / f"{identifier}.png", 400 + offset)
    with (root / "train.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id_code", "diagnosis"])
        writer.writerows(rows)
    return root


def _make_ddr(root: Path) -> Path:
    grading = root / "DDR-dataset" / "DR_grading"
    plan = {
        "train": [("ddr_tr_1", 0), ("ddr_tr_2", 2), ("ddr_tr_3", 3)],
        "valid": [("ddr_va_1", 1), ("ddr_va_2", 4)],
        "test": [("ddr_te_1", 0), ("ddr_te_2", 4)],
    }
    ungradable = [("ddr_tr_bad", 5)]
    offset = 0
    for partition, entries in plan.items():
        for identifier, grade in entries:
            _image(grading / partition / f"{identifier}.jpg", 500 + offset)
            offset += 1
        with (grading / f"{partition}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id_code", "diagnosis"])
            writer.writerows(entries)
    for identifier, grade in ungradable:
        _image(grading / "train" / f"{identifier}.jpg", 600)
    with (grading / "train.csv").open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(ungradable)
    return root


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

def test_aptos_layout_is_validated_and_labelled(tmp_path):
    from src.dataset_downloaders import aptos as aptos_mod
    from src.dataset_downloaders.base import DownloadManifest

    root = _make_aptos(tmp_path / "aptos")
    cfg = __import__("src.config", fromlist=["load_config"]).load_config(
        _write_config(tmp_path, **{"paths.aptos_dir": str(root)}), root=tmp_path
    )
    manifest = DownloadManifest(tmp_path / "download_manifest.json")
    result = aptos_mod.run(cfg, _null_logger(), manifest)

    assert result.status == result.OK_ALREADY_PRESENT
    assert result.usable
    assert result.metadata["layout"]["image_count"] == 6


def test_aptos_reports_blocked_without_kaggle_credentials(tmp_path, monkeypatch):
    from src.dataset_downloaders import aptos as aptos_mod
    from src.dataset_downloaders.base import DownloadManifest

    cfg = __import__("src.config", fromlist=["load_config"]).load_config(
        _write_config(tmp_path, **{"paths.aptos_dir": str(tmp_path / "missing")}),
        root=tmp_path,
    )
    monkeypatch.setattr(aptos_mod, "_kaggle_cli_available", lambda: False)
    monkeypatch.setattr(aptos_mod, "_kaggle_credentials_present", lambda: False)
    result = aptos_mod.run(cfg, _null_logger(), DownloadManifest(tmp_path / "m.json"))

    assert result.status == result.BLOCKED
    assert result.requirements
    assert any("Kaggle" in step for step in result.requirements)


def test_aptos_skip_download_never_transfers(tmp_path, monkeypatch):
    from src.dataset_downloaders import aptos as aptos_mod
    from src.dataset_downloaders.base import DownloadManifest

    cfg = __import__("src.config", fromlist=["load_config"]).load_config(
        _write_config(tmp_path, **{"paths.aptos_dir": str(tmp_path / "missing")}),
        root=tmp_path,
    )
    monkeypatch.setattr(aptos_mod, "_kaggle_cli_available", lambda: True)
    monkeypatch.setattr(aptos_mod, "_kaggle_credentials_present", lambda: True)

    def _explode(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("a skipped download must not start a transfer")

    monkeypatch.setattr(aptos_mod, "_kaggle_download", _explode)
    result = aptos_mod.run(
        cfg, _null_logger(), DownloadManifest(tmp_path / "m.json"), skip_download=True
    )
    assert result.status == result.SKIPPED


def test_ddr_layout_and_official_partitions_are_validated(tmp_path):
    from src.dataset_downloaders import ddr as ddr_mod

    root = _make_ddr(tmp_path / "ddr")
    layout = ddr_mod.validate_ddr(root)
    assert layout["present"] is True
    assert layout["partitions"] == {"train": 4, "valid": 2, "test": 2}
    assert layout["ungradable_rows"] == 1


def test_ddr_missing_copy_is_blocked_not_faked(tmp_path):
    from src.dataset_downloaders import ddr as ddr_mod
    from src.dataset_downloaders.base import DownloadManifest

    cfg = __import__("src.config", fromlist=["load_config"]).load_config(
        _write_config(tmp_path, **{"paths.ddr_dir": str(tmp_path / "absent")}), root=tmp_path
    )
    result = ddr_mod.run(cfg, _null_logger(), DownloadManifest(tmp_path / "m.json"))
    assert result.status == result.BLOCKED
    assert any("Google Drive" in step for step in result.requirements)


def test_ddr_label_file_without_images_is_rejected(tmp_path):
    from src.dataset_downloaders import ddr as ddr_mod

    grading = tmp_path / "ddr" / "DR_grading"
    grading.mkdir(parents=True)
    for partition in ("train", "valid", "test"):
        (grading / f"{partition}.csv").write_text(
            "id_code,diagnosis\nabc,0\n", encoding="utf-8"
        )
    layout = ddr_mod.validate_ddr(tmp_path / "ddr")
    assert layout["present"] is False
    assert "no fundus images" in layout["reason"]


# ---------------------------------------------------------------------------
# Provenance and label correctness through the manifest stage
# ---------------------------------------------------------------------------

def test_aptos_and_ddr_keep_provenance_and_labels(tmp_path, monkeypatch):
    from src.prepare import run_prepare

    mendeley = _make_mendeley(tmp_path / "mendeley")
    aptos = _make_aptos(tmp_path / "aptos")
    ddr = _make_ddr(tmp_path / "ddr")
    config_path = _write_config(tmp_path)
    cfg = __import__("src.config", fromlist=["load_config"]).load_config(config_path, root=tmp_path)
    monkeypatch.setenv("DR_FUNDUS_ROOT", str(mendeley))
    monkeypatch.setenv("DR_APTOS_ROOT", str(aptos))
    monkeypatch.setenv("DR_DDR_ROOT", str(ddr))

    summary = run_prepare(cfg, _null_logger())

    for name in ("mendeley", "aptos", "ddr"):
        assert summary["sources"][name]["available"] is True

    rows = _read_manifest(tmp_path / "outputs" / "manifests" / "all_images.csv")
    by_source = {}
    for row in rows:
        by_source.setdefault(row["source_dataset"], []).append(row)

    # Provenance is never rewritten: each source carries its own dataset identity.
    assert set(by_source) == {"mendeley", "aptos", "ddr"}
    assert all(row["original_label"].startswith("diagnosis=") for row in by_source["aptos"])
    # APTOS diagnosis 4 -> PDR, verbatim.
    assert {row["class_name"] for row in by_source["aptos"] if "diagnosis=4" in row["original_label"]} == {"PDR"}
    # MMRDR images can never be recorded as Mendeley: there is no mmrdr source here,
    # and every mendeley row points inside the Mendeley corpus.
    assert all(str(mendeley) in row["image_path"] for row in by_source["mendeley"])

    # DDR grade 5 is excluded with an explicit reason, not mapped to No DR.
    ungradable = [row for row in by_source["ddr"] if row["original_label"] == "diagnosis=5"]
    assert ungradable
    assert all(row["included"] == "no" for row in ungradable)
    assert all("ungradable" in row["exclusion_reason"] for row in ungradable)
    assert not any(
        row["class_name"] == "No apparent DR" and row["original_label"] == "diagnosis=5"
        for row in by_source["ddr"]
    )
    # DDR official partition membership survives into the manifest.
    partitions = {row["rel_path"].split("/")[2] for row in by_source["ddr"] if row["rel_path"].startswith("DDR-dataset/DR_grading/")}
    assert {"train", "valid", "test"} <= partitions


def test_combined_mode_trains_on_every_source_but_not_official_tests(tmp_path, monkeypatch):
    from src.prepare import run_prepare

    mendeley = _make_mendeley(tmp_path / "mendeley")
    aptos = _make_aptos(tmp_path / "aptos")
    ddr = _make_ddr(tmp_path / "ddr")
    cfg = __import__("src.config", fromlist=["load_config"]).load_config(
        _write_config(tmp_path), root=tmp_path
    )
    monkeypatch.setenv("DR_FUNDUS_ROOT", str(mendeley))
    monkeypatch.setenv("DR_APTOS_ROOT", str(aptos))
    monkeypatch.setenv("DR_DDR_ROOT", str(ddr))

    summary = run_prepare(cfg, _null_logger(), dataset_mode="combined")
    assert summary["mode"] == "combined"
    manifests = tmp_path / "outputs" / "manifests"

    internal = []
    for name in ("train_manifest.csv", "validation_manifest.csv", "test_manifest.csv"):
        path = manifests / name
        if path.is_file():
            internal.extend(_read_manifest(path))
    internal_sources = {row["source_dataset"] for row in internal}
    assert {"mendeley", "aptos", "ddr"} <= internal_sources

    external = _read_manifest(manifests / "external_test_manifest.csv")
    assert external
    # Only DDR's official test partition is held out in combined mode.
    assert {row["source_dataset"] for row in external} == {"ddr"}
    assert all("ddr_te_" in row["image_path"] for row in external)
    assert not ({row["image_path"] for row in external} & {row["image_path"] for row in internal})

    # The label audit names every source independently.
    audit_sources = {row["source_dataset"] for row in _read_manifest(manifests / "label_mapping_audit.csv")}
    assert {"mendeley", "aptos", "ddr"} <= audit_sources


def test_external_mode_holds_non_development_sources_out_entirely(tmp_path, monkeypatch):
    from src.prepare import run_prepare

    mendeley = _make_mendeley(tmp_path / "mendeley")
    aptos = _make_aptos(tmp_path / "aptos")
    ddr = _make_ddr(tmp_path / "ddr")
    cfg = __import__("src.config", fromlist=["load_config"]).load_config(
        _write_config(tmp_path), root=tmp_path
    )
    monkeypatch.setenv("DR_FUNDUS_ROOT", str(mendeley))
    monkeypatch.setenv("DR_APTOS_ROOT", str(aptos))
    monkeypatch.setenv("DR_DDR_ROOT", str(ddr))

    summary = run_prepare(cfg, _null_logger(), dataset_mode="external")
    assert summary["mode"] == "external"
    assert set(cfg.development_sources) == {"mendeley", "mmrdr"}

    manifests = tmp_path / "outputs" / "manifests"
    internal = []
    for name in ("train_manifest.csv", "validation_manifest.csv", "test_manifest.csv"):
        path = manifests / name
        assert path.is_file(), name
        internal.extend(_read_manifest(path))
    assert {row["source_dataset"] for row in internal} == {"mendeley"}

    external = _read_manifest(manifests / "external_test_manifest.csv")
    assert {row["source_dataset"] for row in external} == {"aptos", "ddr"}
    assert not ({row["image_path"] for row in external} & {row["image_path"] for row in internal})
    assert summary["external_partition_composition"] == {"aptos": 6, "ddr": 7}


def test_external_mode_without_a_development_source_fails_loudly(tmp_path, monkeypatch):
    from src.prepare import PrepareError, run_prepare

    aptos = _make_aptos(tmp_path / "aptos")
    cfg = __import__("src.config", fromlist=["load_config"]).load_config(
        _write_config(tmp_path), root=tmp_path
    )
    monkeypatch.setenv("DR_APTOS_ROOT", str(aptos))
    monkeypatch.delenv("DR_FUNDUS_ROOT", raising=False)

    with pytest.raises(PrepareError) as excinfo:
        run_prepare(cfg, _null_logger(), dataset_mode="external")
    message = str(excinfo.value)
    assert "External-validation mode" in message
    assert "mendeley" in message


def test_unknown_source_label_stops_the_pipeline(tmp_path):
    from src.taxonomy import LabelMapper, UnknownLabelError

    cfg = __import__("src.config", fromlist=["load_config"]).load_config(
        _write_config(tmp_path), root=tmp_path
    )
    mapper = LabelMapper(cfg)
    with pytest.raises(UnknownLabelError):
        mapper.resolve("ddr", "7")
    # Grade 5 is declared as an exclusion, not as a grade.
    decision = mapper.resolve("ddr", "5")
    assert decision.mapped_label is None
    assert decision.exclusion_reason


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def test_parser_accepts_dataset_modes_and_experiments():
    from src.pipeline import MODES, build_parser

    assert {"combined", "external"} <= set(MODES)
    args = build_parser().parse_args(["--mode", "combined", "--experiments", "all"])
    assert args.mode == "combined"
    assert args.experiments == ["all"]
    assert args.dataset_mode is None

    args = build_parser().parse_args(["--prepare-only", "--dataset-mode", "external"])
    assert args.mode_alias == "prepare"
    assert args.dataset_mode == "external"


# ---------------------------------------------------------------------------
# Experiment series
# ---------------------------------------------------------------------------

def test_experiment_ids_require_a_training_experiment_for_the_final_candidate():
    from src.experiments import experiment_ids

    assert experiment_ids(["all"]) == list(experiment_ids())[: len(experiment_ids())]
    assert experiment_ids(["E0", "E0", "E4"]) == ["E0", "E4"]
    with pytest.raises(ValueError):
        experiment_ids(["final_candidate"])
    with pytest.raises(ValueError):
        experiment_ids(["E9"])


def test_augmentation_variants_are_distinct_and_training_only():
    from src.experiments import AUGMENTATION_VARIANTS

    assert AUGMENTATION_VARIANTS["default"] != AUGMENTATION_VARIANTS["retinal"]
    # No variant may enable a transform the pipeline refuses for fundus images.
    for variant in AUGMENTATION_VARIANTS.values():
        assert variant["random_crop"] is False
        assert variant["vertical_flip"] is False


def test_final_candidate_inherits_the_best_training_configuration(tmp_path):
    from src.experiments import EXPERIMENT_CONFIG_OVERRIDES, final_candidate_config, select_best_by_validation

    cfg = __import__("src.config", fromlist=["load_config"]).load_config(
        _write_config(tmp_path), root=tmp_path
    )
    results = {
        "E0": {"status": "completed", "best_metric_value": 0.31},
        "E3": {"status": "completed", "best_metric_value": 0.42},
        "E4": {"status": "failed", "best_metric_value": 0.99},
        "E5": {"status": "completed", "best_metric_value": None},
    }
    best = select_best_by_validation(results)
    assert best == "E3"
    assert final_candidate_config(cfg, best) == EXPERIMENT_CONFIG_OVERRIDES["E3"]


def test_promote_final_candidate_publishes_one_checkpoint_with_provenance(tmp_path):
    from src.experiments import promote_final_candidate

    cfg = __import__("src.config", fromlist=["load_config"]).load_config(
        _write_config(tmp_path), root=tmp_path
    )
    checkpoint_dir = tmp_path / "checkpoints" / "E2"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "best_resnet50.pt"
    checkpoint.write_bytes(b"synthetic-checkpoint")

    individual = {
        "E0": {"status": "completed", "checkpoint": str(tmp_path / "nope.pt"), "best_metric_value": 0.2},
        "E2": {"status": "completed", "checkpoint": str(checkpoint), "best_metric_value": 0.5},
    }
    published = promote_final_candidate(cfg, individual, ["E0", "E2"], logger=_null_logger())
    assert published is not None
    assert published.is_file()
    assert published.read_bytes() == b"synthetic-checkpoint"

    provenance = json.loads(
        (published.parent / "checkpoint_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["source_experiment"] == "E2"
    assert "validation" in provenance["selection_basis"]


def test_promotion_refuses_when_no_experiment_completed(tmp_path):
    from src.experiments import promote_final_candidate

    cfg = __import__("src.config", fromlist=["load_config"]).load_config(
        _write_config(tmp_path), root=tmp_path
    )
    published = promote_final_candidate(
        cfg, {"E0": {"status": "failed", "error": "boom"}}, ["E0"], logger=_null_logger()
    )
    assert published is None
    assert not (tmp_path / "outputs" / "classification" / "best_resnet50.pt").exists()


# ---------------------------------------------------------------------------
# Post-hoc fitting is validation-only and never changes the model
# ---------------------------------------------------------------------------

def test_temperature_fit_stays_in_bounds_and_cannot_change_predictions():
    from src.experiment_engine import _fit_temperature, _softmax

    rng = np.random.default_rng(7)
    logits = rng.normal(size=(64, 5))
    labels = rng.integers(0, 5, size=64)
    temperature = _fit_temperature(logits, labels)
    assert 0.3 <= temperature <= 6.0
    assert (_softmax(logits).argmax(axis=1) == _softmax(logits / temperature).argmax(axis=1)).all()


def test_posthoc_parameters_are_attached_only_to_the_published_experiment(tmp_path):
    from src.evaluate_classification import _posthoc_for_published_checkpoint

    cfg = __import__("src.config", fromlist=["load_config"]).load_config(
        _write_config(tmp_path), root=tmp_path
    )
    out = tmp_path / "outputs" / "classification"
    out.mkdir(parents=True)
    assert _posthoc_for_published_checkpoint(cfg, out) is None

    (out / "checkpoint_provenance.json").write_text(
        json.dumps({"source_experiment": "E5"}), encoding="utf-8"
    )
    # The provenance names E5, but E5 has no post-hoc file yet.
    assert _posthoc_for_published_checkpoint(cfg, out) is None

    experiment_dir = tmp_path / "outputs" / "experiments" / "E5"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "posthoc.json").write_text(
        json.dumps({"class_offsets": {"PDR": 0.3}, "temperature": 1.5}), encoding="utf-8"
    )
    posthoc = _posthoc_for_published_checkpoint(cfg, out)
    assert posthoc is not None
    assert posthoc["source_experiment"] == "E5"


def test_supplementary_metrics_are_separate_from_the_headline_metrics(tmp_path):
    from src.evaluate_classification import _supplementary_metrics

    cfg = __import__("src.config", fromlist=["load_config"]).load_config(
        _write_config(tmp_path), root=tmp_path
    )
    out = tmp_path / "outputs" / "classification"
    out.mkdir(parents=True)
    class_names = cfg.classes
    y_true = np.asarray([0, 1, 4, 4])
    probabilities = np.asarray(
        [
            [0.80, 0.10, 0.05, 0.03, 0.02],
            [0.30, 0.40, 0.10, 0.10, 0.10],
            [0.40, 0.05, 0.05, 0.05, 0.45],
            [0.40, 0.05, 0.05, 0.05, 0.45],
        ]
    )
    logits = np.log(probabilities)
    posthoc = {
        "class_offsets": {name: 0.0 for name in class_names}
        | {"PDR": 0.25},
        "temperature": 1.4,
        "source_experiment": "E6",
        "parameter_file": "posthoc.json",
    }
    supplementary = _supplementary_metrics(
        cfg,
        posthoc,
        probabilities=probabilities,
        logits=logits,
        y_true=y_true,
        class_names=class_names,
        out_dir=out,
        all_meta=[{"image_path": f"image{index}"} for index in range(len(y_true))],
    )
    assert set(supplementary) >= {"thresholded", "temperature_scaled"}
    assert set(supplementary["thresholded"]["predicted_class_counts"]) == set(class_names)
    assert supplementary["temperature_scaled"]["temperature"] == pytest.approx(1.4)
    # Temperature scaling cannot change the predicted class.
    assert supplementary["temperature_scaled"]["accuracy"] == pytest.approx(
        float((probabilities.argmax(axis=1) == y_true).mean())
    )
    csv_path = out / "test_predictions_thresholded.csv"
    assert csv_path.is_file()
    assert len(csv_path.read_text(encoding="utf-8").strip().splitlines()) == len(y_true) + 1


def test_class_offsets_can_recover_an_under_predicted_class():
    from src.experiment_engine import _macro_f1, _optimise_class_offsets

    # Class 1 is systematically just below class 0, so argmax never predicts it.
    probabilities = np.asarray(
        [
            [0.70, 0.30, 0.00, 0.00, 0.00],
            [0.70, 0.30, 0.00, 0.00, 0.00],
            [0.70, 0.30, 0.00, 0.00, 0.00],
            [0.52, 0.48, 0.00, 0.00, 0.00],
            [0.52, 0.48, 0.00, 0.00, 0.00],
            [0.52, 0.48, 0.00, 0.00, 0.00],
        ]
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    baseline = _macro_f1(labels, probabilities.argmax(axis=1))
    offsets, fitted = _optimise_class_offsets(probabilities, labels, 5)
    assert fitted >= baseline
    assert fitted > baseline
    # Either raising class 1 or lowering class 0 solves it; both are valid per-class
    # decision thresholds, so only "some offset was moved" is asserted.
    assert any(abs(value) > 0 for value in offsets)
