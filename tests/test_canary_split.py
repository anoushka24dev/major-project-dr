"""Focused tests for the derived canary split contract.

These cover two things:

1. ``colab.setup_helpers.validate_canary_split`` - a validator that is *separate*
   from ``validate_split_artifact`` because a canary deliberately has no test
   partition. The full validator is untouched and still rejects a canary.
2. ``colab.pipeline.canary_run`` - the canary entry point the notebook calls: the
   CSV must be written to the exact path the validator and ``run_training`` read,
   the canary must be validated (with image paths + decoding) before training,
   and the weight-download consent must default to off.

Nothing here trains a model, downloads weights, or touches the pinned split or
any audit artifact. Synthetic images are written under ``tmp_path`` so the
image-path and decoding checks run without the real corpus.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "colab"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import colab.setup_helpers as sh  # noqa: E402
import colab_helpers as ch  # noqa: E402

SPLIT_CSV = REPO_ROOT / "provisional_primary_split" / "provisional_primary_split.csv"
NOTEBOOK = REPO_ROOT / "colab" / "train_on_colab_fixed.ipynb"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _notebook_cell(label: str) -> str:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        first = "".join(cell["source"]).splitlines()[0].lstrip()
        m = re.match(r"# (\S+)\.\s", first)
        if m and m.group(1) == label:
            return "".join(cell["source"])
    raise AssertionError(f"code cell #{label} not found in {NOTEBOOK}")


def _build_canary(tmp_path: Path, *, train: int = 10, val: int = 5):
    """Build the 50/25 canary at the canonical split-dir layout."""
    split_dir = tmp_path / "canary_split"
    split_dir.mkdir(parents=True, exist_ok=True)
    csv_path = split_dir / "provisional_primary_split.csv"
    meta = sh.build_canary_split(
        SPLIT_CSV,
        csv_path,
        train_per_class=train,
        val_per_class=val,
        test_per_class=0,
    )
    return split_dir, csv_path, meta


def _rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def _write_rows(csv_path: Path, rows: List[Dict[str, str]]) -> None:
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _materialize_images(rows, root: Path, *, size=(8, 6)) -> None:
    from PIL import Image

    for row in rows:
        path = root / row["image_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, (12, 34, 56)).save(path)


def _check(validation, name: str) -> Dict[str, object]:
    for item in validation["checks"]:
        if item["name"] == name:
            return item
    raise AssertionError(f"check {name!r} not reported; got {[c['name'] for c in validation['checks']]}")


def _failing_names(validation) -> set:
    return {c["name"] for c in validation["checks"] if not c["ok"]}


# ---------------------------------------------------------------------------
# path contract: the CSV has to live where the readers look
# ---------------------------------------------------------------------------

def test_canary_csv_is_where_the_trainer_loads_the_split(tmp_path):
    """``run_training`` loads ``split_dir / provisional_primary_split.csv``."""
    split_dir, csv_path, meta = _build_canary(tmp_path)

    assert csv_path == split_dir / ch.SPLIT_CSV_NAME
    assert csv_path.is_file()
    assert meta["out_path"] == str(csv_path)

    # The loader run_training actually uses must see the canary rows.
    loaded = ch.load_split_csv(split_dir)
    assert len(loaded) == 75
    assert len(loaded) == meta["rows"]


def test_a_canary_written_outside_the_split_dir_fails_loudly(tmp_path):
    """Writing the CSV to the output-dir root must not validate as a split dir."""
    out_root = tmp_path / "canary"
    empty_split_dir = out_root / "canary_split"
    empty_split_dir.mkdir(parents=True, exist_ok=True)
    misplaced_csv = out_root / "canary_provisional_split.csv"

    sh.build_canary_split(SPLIT_CSV, misplaced_csv, train_per_class=10, val_per_class=5, test_per_class=0)
    assert misplaced_csv.is_file()

    validation = sh.validate_canary_split(empty_split_dir, dataset_root=tmp_path)

    assert validation["ok"] is False
    assert "canary_csv_at_expected_path" in _failing_names(validation)
    detail = _check(validation, "canary_csv_at_expected_path")["detail"]
    assert "provisional_primary_split.csv" in detail


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------

def test_validate_canary_split_accepts_the_50_25_0_canary(tmp_path):
    split_dir, csv_path, meta = _build_canary(tmp_path)
    rows = _rows(csv_path)

    dataset_root = tmp_path / "corpus"
    _materialize_images(rows, dataset_root)

    validation = sh.validate_canary_split(
        split_dir,
        dataset_root=dataset_root,
        expected_train=50,
        expected_validation=25,
        expected_test=0,
        train_per_class=10,
        val_per_class=5,
        expected_classes=sh.EXPECTED_CLASSES,
        decode_per_class=1,
    )

    assert validation["ok"] is True, [c for c in validation["checks"] if not c["ok"]]
    assert validation["rows"] == 75
    assert validation["partition_counts"] == {"train": 50, "validation": 25}

    names = {c["name"] for c in validation["checks"]}
    for required in (
        "canary_csv_at_expected_path",
        "canary_total_rows",
        "canary_train_rows",
        "canary_validation_rows",
        "canary_classes_are_the_five_bands",
        "canary_all_bands_in_train",
        "canary_all_bands_in_validation",
        "canary_train_per_class",
        "canary_validation_per_class",
        "canary_image_paths_unique",
        "canary_path_prefix_matches_label",
        "canary_no_group_leakage",
        "canary_image_paths_exist",
        "canary_images_decode",
    ):
        assert required in names, required
        assert _check(validation, required)["ok"] is True, _check(validation, required)


def test_validate_canary_split_decodes_one_image_per_band(tmp_path):
    split_dir, csv_path, _ = _build_canary(tmp_path)
    rows = _rows(csv_path)
    dataset_root = tmp_path / "corpus"
    _materialize_images(rows, dataset_root, size=(11, 7))

    validation = sh.validate_canary_split(split_dir, dataset_root=dataset_root, decode_per_class=2)

    assert _check(validation, "canary_images_decode")["ok"] is True
    # 5 bands x 2 samples
    assert "10 canary image" in _check(validation, "canary_images_decode")["detail"]


# ---------------------------------------------------------------------------
# shape / integrity rejections
# ---------------------------------------------------------------------------

def test_validate_canary_split_rejects_a_canary_of_the_wrong_size(tmp_path):
    split_dir, csv_path, _ = _build_canary(tmp_path, train=4, val=2)
    rows = _rows(csv_path)
    dataset_root = tmp_path / "corpus"
    _materialize_images(rows, dataset_root)

    # Same expected shape the notebook passes: 10 train / 5 validation per band.
    validation = sh.validate_canary_split(
        split_dir,
        dataset_root=dataset_root,
        train_per_class=10,
        val_per_class=5,
    )

    assert validation["ok"] is False
    failing = _failing_names(validation)
    assert {
        "canary_total_rows",
        "canary_train_rows",
        "canary_validation_rows",
        "canary_train_per_class",
        "canary_validation_per_class",
    } <= failing


def test_validate_canary_split_requires_every_band_in_validation(tmp_path):
    split_dir, csv_path, _ = _build_canary(tmp_path)
    rows = _rows(csv_path)
    kept = [r for r in rows if not (r["partition"] == "validation" and r["primary_class"] == "PDR")]
    _write_rows(csv_path, kept)

    dataset_root = tmp_path / "corpus"
    _materialize_images(kept, dataset_root)

    validation = sh.validate_canary_split(
        split_dir,
        dataset_root=dataset_root,
        train_per_class=10,
        val_per_class=5,
    )

    assert validation["ok"] is False
    failing = _failing_names(validation)
    assert "canary_all_bands_in_validation" in failing
    assert "canary_validation_per_class" in failing


def test_validate_canary_split_detects_group_leakage(tmp_path):
    split_dir, csv_path, _ = _build_canary(tmp_path)
    rows = _rows(csv_path)

    train_group = next(r["group_id"] for r in rows if r["partition"] == "train" and r["group_id"])
    victim = next(r for r in rows if r["partition"] == "validation")
    victim["group_id"] = train_group  # same group now spans train + validation
    _write_rows(csv_path, rows)

    dataset_root = tmp_path / "corpus"
    _materialize_images(rows, dataset_root)

    validation = sh.validate_canary_split(split_dir, dataset_root=dataset_root)

    assert validation["ok"] is False
    assert "canary_no_group_leakage" in _failing_names(validation)


def test_validate_canary_split_detects_a_relabelled_row(tmp_path):
    split_dir, csv_path, _ = _build_canary(tmp_path)
    rows = _rows(csv_path)
    rows[0]["primary_class"] = "PDR" if rows[0]["primary_class"] != "PDR" else "Mild NPDR"
    _write_rows(csv_path, rows)

    dataset_root = tmp_path / "corpus"
    _materialize_images(rows, dataset_root)

    validation = sh.validate_canary_split(split_dir, dataset_root=dataset_root)

    assert validation["ok"] is False
    assert "canary_path_prefix_matches_label" in _failing_names(validation)


# ---------------------------------------------------------------------------
# image checks fail closed
# ---------------------------------------------------------------------------

def test_validate_canary_split_fails_closed_without_a_dataset_root(tmp_path):
    split_dir, _, _ = _build_canary(tmp_path)

    validation = sh.validate_canary_split(split_dir, dataset_root=None)

    assert validation["ok"] is False
    assert "canary_image_paths_exist" in _failing_names(validation)
    assert "canary_images_decode" in _failing_names(validation)
    assert "failing closed" in _check(validation, "canary_images_decode")["detail"]


def test_validate_canary_split_reports_missing_images(tmp_path):
    split_dir, csv_path, _ = _build_canary(tmp_path)
    rows = _rows(csv_path)
    dataset_root = tmp_path / "corpus"
    _materialize_images(rows[:-1], dataset_root)  # leave one image out

    validation = sh.validate_canary_split(split_dir, dataset_root=dataset_root)

    assert validation["ok"] is False
    assert "canary_image_paths_exist" in _failing_names(validation)
    assert "1 missing" in _check(validation, "canary_image_paths_exist")["detail"]


def test_validate_canary_split_reports_an_undecodable_image(tmp_path):
    split_dir, csv_path, _ = _build_canary(tmp_path)
    rows = _rows(csv_path)
    dataset_root = tmp_path / "corpus"
    _materialize_images(rows, dataset_root)

    # Corrupt the first validation image of one band: it exists but cannot decode.
    target = next(r for r in rows if r["partition"] == "validation" and r["primary_class"] == "PDR")
    (dataset_root / target["image_path"]).write_bytes(b"not an image")

    validation = sh.validate_canary_split(split_dir, dataset_root=dataset_root)

    assert validation["ok"] is False
    assert _check(validation, "canary_image_paths_exist")["ok"] is True
    assert "canary_images_decode" in _failing_names(validation)


# ---------------------------------------------------------------------------
# the full-split validator is NOT weakened
# ---------------------------------------------------------------------------

def test_full_split_validator_still_rejects_a_canary(tmp_path):
    split_dir, _, _ = _build_canary(tmp_path)

    validation = sh.validate_split_artifact(
        split_dir,
        project_root=REPO_ROOT,
        expected_rows=sh.EXPECTED_ROWS,
        expected_partition_counts=sh.EXPECTED_PARTITION_COUNTS,
        expected_classes=sh.EXPECTED_CLASSES,
        image_root=None,
        path_check="sample",
        decode_sample_per_stratum=0,
        run_protocol_validation=False,
    )

    assert validation["ok"] is False
    failing = {c["name"] for c in validation["checks"] if not c["ok"]}
    assert "row_count" in failing
    assert "partition_counts_match_audited_split" in failing
    assert "every_band_in_every_partition" in failing


# ---------------------------------------------------------------------------
# The canary entry point the notebook calls: colab.pipeline.canary_run
# ---------------------------------------------------------------------------

import inspect  # noqa: E402

import colab.pipeline as pipeline  # noqa: E402


def _canary_env(tmp_path: Path, *, materialize="all"):
    """Build a canary + a synthetic corpus under ``tmp_path``.

    ``materialize`` is ``"all"``, ``"missing_one"``, or ``None`` (no corpus at all).
    """
    out = tmp_path / "outputs"
    corpus = tmp_path / "corpus"

    probe_dir = tmp_path / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    sh.build_canary_split(
        SPLIT_CSV,
        probe_dir / ch.SPLIT_CSV_NAME,
        train_per_class=10,
        val_per_class=5,
        test_per_class=0,
    )
    rows = _rows(probe_dir / ch.SPLIT_CSV_NAME)
    if materialize == "all":
        _materialize_images(rows, corpus)
    elif materialize == "missing_one":
        _materialize_images(rows[:-1], corpus)
    return out, corpus, rows


def _record_training(monkeypatch, calls, *, device="cpu"):
    """Replace the real training loop with a recorder, and the eval with one too."""
    import colab.train as train_mod

    def _stub(**kwargs):
        calls.append(kwargs)
        return {
            "best_epoch": 0,
            "best_val_loss": 0.5,
            "output_dir": Path(kwargs["output_dir"]),
            "final_state": {"train_history": [{}]},
            "device": device,
            "best_checkpoint_dir": Path(kwargs["output_dir"]) / "best_checkpoint",
        }

    monkeypatch.setattr(train_mod, "run_training", _stub)


def test_canary_csv_is_written_where_the_trainer_loads_the_split(tmp_path, monkeypatch):
    out, corpus, _ = _canary_env(tmp_path)
    _record_training(monkeypatch, [])

    result = pipeline.canary_run(
        project_root=REPO_ROOT, dataset_root=corpus, output_dir=out
    )

    split_dir = out / "canary_split"
    csv_path = split_dir / ch.SPLIT_CSV_NAME
    assert result["canary_split_dir"] == str(split_dir)
    assert csv_path.is_file()
    assert result["rows"] == 75
    assert result["partition_counts"] == {"train": 50, "validation": 25}
    assert result["validated"] is True
    assert result["trained"] is False

    # The loader run_training actually uses must see the canary rows.
    assert len(ch.load_split_csv(split_dir)) == 75

    # The old, broken layouts are gone for good.
    assert not (split_dir / "canary_provisional_split.csv").exists()
    assert not (out / "canary_provisional_split.csv").exists()


def test_canary_is_validated_before_training_is_reached(tmp_path, monkeypatch):
    out, corpus, _ = _canary_env(tmp_path, materialize="missing_one")
    calls = []
    _record_training(monkeypatch, calls)

    with pytest.raises(RuntimeError, match="Canary split failed validation"):
        pipeline.canary_run(
            project_root=REPO_ROOT, dataset_root=corpus, output_dir=out, train=True, confirm=True
        )

    assert calls == [], "run_training must not be reached when the canary is invalid"
    assert not (out / "checkpoints").exists()
    assert not (out / "run_manifest.json").exists()


def test_canary_training_requires_explicit_weight_consent(tmp_path, monkeypatch):
    out, corpus, _ = _canary_env(tmp_path)
    calls = []
    _record_training(monkeypatch, calls)
    monkeypatch.setattr(ch, "pretrained_weights_available", lambda: False)

    with pytest.raises(FileNotFoundError, match="weights are not cached"):
        pipeline.canary_run(
            project_root=REPO_ROOT, dataset_root=corpus, output_dir=out, train=True, confirm=True
        )

    assert calls == []
    # The canary itself was still built and validated at the canonical path.
    assert (out / "canary_split" / ch.SPLIT_CSV_NAME).is_file()


def test_canary_training_requires_confirm(tmp_path, monkeypatch):
    out, corpus, _ = _canary_env(tmp_path)
    calls = []
    _record_training(monkeypatch, calls)
    monkeypatch.setattr(ch, "pretrained_weights_available", lambda: True)

    with pytest.raises(PermissionError, match="--confirm"):
        pipeline.canary_run(
            project_root=REPO_ROOT, dataset_root=corpus, output_dir=out, train=True
        )

    assert calls == []


def test_canary_trains_on_the_canary_split_with_the_real_guards(tmp_path, monkeypatch):
    out, corpus, _ = _canary_env(tmp_path)
    calls = []
    _record_training(monkeypatch, calls)
    monkeypatch.setattr(ch, "pretrained_weights_available", lambda: True)

    result = pipeline.canary_run(
        project_root=REPO_ROOT,
        dataset_root=corpus,
        output_dir=out,
        train=True,
        confirm=True,
        epochs=2,
        batch_size=8,
    )

    assert result["trained"] is True
    assert len(calls) == 1
    kwargs = calls[0]
    assert Path(kwargs["split_dir"]) == out / "canary_split"
    assert Path(kwargs["output_dir"]) == out
    assert kwargs["confirm_training"] is True
    assert kwargs["smoke_test_passed"] is True
    assert kwargs["allow_weight_download"] is False
    assert kwargs["overwrite"] is True
    assert kwargs["seed"] == 12345
    assert Path(kwargs["dataset_root"]) == corpus
    assert len(ch.load_split_csv(kwargs["split_dir"])) == 75

    # A canary has no test partition, so nothing may score one.
    rows = ch.load_split_csv(out / "canary_split")
    assert not [r for r in rows if r["partition"] == "test"]
    assert "no test partition" in result["note"].lower()
    assert not (out / "final_test_evaluation").exists()


def test_canary_fails_closed_when_the_corpus_is_unresolved(tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    calls = []
    _record_training(monkeypatch, calls)
    monkeypatch.setattr(
        sh, "resolve_data_root", lambda *a, **k: (None, ["corpus: missing (path does not exist)"])
    )

    with pytest.raises(FileNotFoundError) as excinfo:
        pipeline.canary_run(project_root=REPO_ROOT, output_dir=out)

    message = str(excinfo.value)
    assert "DR_FUNDUS_ROOT" in message
    assert "corpus" in message.lower()
    assert calls == []


def test_canary_weight_consent_defaults_off():
    params = inspect.signature(pipeline.canary_run).parameters
    assert params["allow_weight_download"].default is False
    assert params["train"].default is False
    assert params["confirm"].default is False
    # The published canary shape: 10 train + 5 validation per band, no test rows.
    source = inspect.getsource(pipeline.canary_run)
    assert "train_per_class=10" in source
    assert "val_per_class=5" in source
    assert "test_per_class=0" in source
    assert "decode_per_class=1" in source
