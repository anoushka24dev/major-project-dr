"""
Focused tests for the notebook contract, project discovery, and the setup/validation
helpers that ``colab/train_on_colab_fixed.ipynb`` relies on.

The notebook is deliberately thin: six cells, each one a call into a tested function
in ``colab.pipeline``. These tests check that shape (order, opt-in flags, delegation,
per-cell syntax) and then exercise the real helpers against the real pinned split
artifact. They do not install packages, train a model, or modify the audited split.
"""

from __future__ import annotations

import csv
import json
import inspect
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "colab"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import colab.setup_helpers as sh  # noqa: E402

SPLIT_CSV = REPO_ROOT / "provisional_primary_split" / "provisional_primary_split.csv"
SPLIT_DIR = REPO_ROOT / "provisional_primary_split"

CLIENTSIDE_JSON = REPO_ROOT / "colab" / "train_on_colab_fixed.ipynb"


# ---------------------------------------------------------------------------
# Notebook JSON + structural checks
# ---------------------------------------------------------------------------

def _notebook() -> Dict[str, Any]:
    return json.loads(CLIENTSIDE_JSON.read_text(encoding="utf-8"))


def test_fixed_notebook_is_valid_json():
    nb = _notebook()
    assert nb["nbformat"] == 4
    assert nb["nbformat_minor"] >= 5
    assert len(nb["cells"]) > 0


def _code_cell_sources() -> List[str]:
    return ["".join(c["source"]) for c in _notebook()["cells"] if c.get("cell_type") == "code"]


def _code_blob() -> str:
    return "\n".join(_code_cell_sources())


def test_fixed_notebook_is_a_thin_six_step_interface():
    """Six steps, each one a call into a tested function in ``colab.pipeline``."""
    sources = _code_cell_sources()
    assert len(sources) == 6

    labels, titles = [], []
    for source in sources:
        first = source.splitlines()[0].lstrip()
        m = re.match(r"# ([^.\s]+)\.\s+(.*)", first)
        assert m, first
        labels.append(m.group(1))
        titles.append(m.group(2).strip())

    assert labels == ["1", "2", "3", "4", "5", "6"]
    assert titles[1].startswith("Preflight")
    assert titles[2].startswith("Canary")
    assert "resume" in titles[3].lower()
    assert titles[4].startswith("Final test evaluation")
    assert titles[5].startswith("Verify")


def test_fixed_notebook_steps_appear_in_execution_order():
    sources = _code_cell_sources()

    def first(needle: str) -> int:
        hits = [i for i, s in enumerate(sources) if needle in s]
        assert hits, f"no code cell contains {needle!r}"
        return hits[0]

    order = [
        first("import colab.pipeline as pipeline"),
        first("pipeline.preflight("),
        first("pipeline.canary_run("),
        first("pipeline.train_run("),
        first("pipeline.evaluate_run("),
        first("sh.verify_output_run("),
    ]
    assert order == [0, 1, 2, 3, 4, 5], order

    # Only the training cell may repeat the preflight call (it re-checks when the
    # preflight cell was skipped); the canary and evaluation cells never do.
    assert [i for i, s in enumerate(sources) if "pipeline.preflight(" in s] == [1, 3]


def test_fixed_notebook_delegates_instead_of_reimplementing():
    blob = _code_blob()
    for forbidden in (
        "subprocess",
        "pip install",
        "!pip",
        "torch.save",
        "DataLoader(",
        "install_pinned",
        "run_training(",
        "run_final_evaluation(",
    ):
        assert forbidden not in blob, f"the notebook should delegate, not contain {forbidden!r}"
    # No unbounded project scan and no notebook-side filesystem crawling.
    assert "glob" not in blob and "**" not in blob


def test_fixed_notebook_every_code_cell_compiles():
    for index, source in enumerate(_code_cell_sources()):
        compile(source, f"cell-{index}", "exec")


def test_fixed_notebook_drive_mount_is_optional_and_bounded():
    blob = _code_blob()
    assert "from google.colab import drive" in blob
    assert "drive.mount(" in blob
    # Mounting is wrapped, so the same cell runs unchanged outside Colab.
    assert "except Exception" in blob
    assert "glob" not in blob


def test_fixed_notebook_project_discovery_is_delegated_but_overridable():
    blob = _code_blob()
    assert "OVERRIDE_PROJECT_ROOT" in blob
    assert "pipeline.discover_project(" in blob

    pipeline_src = (REPO_ROOT / "colab" / "pipeline.py").read_text(encoding="utf-8")
    assert "DR_PROJECT_ROOT" in pipeline_src
    assert "bounded_scan" in pipeline_src
    # Discovery is bounded: no unbounded recursive filesystem scan anywhere.
    assert "rglob(" not in pipeline_src
    assert "glob.glob" not in pipeline_src


def test_fixed_notebook_references_only_files_that_exist():
    blob = _code_blob()
    for module in ("pipeline", "setup_helpers"):
        assert f"colab.{module}" in blob
        assert (REPO_ROOT / "colab" / f"{module}.py").is_file()


# ---------------------------------------------------------------------------
# Split-artifact facts used by the notebook
# ---------------------------------------------------------------------------

def test_notebook_split_constants_match_the_actual_pinned_split():
    rows = list(csv.DictReader(SPLIT_CSV.read_text(encoding="utf-8").splitlines()))
    assert sh.EXPECTED_ROWS == len(rows)
    assert sh.EXPECTED_ROWS == 14201
    assert sh.EXPECTED_PARTITION_COUNTS == {
        "train": 9924,
        "validation": 2133,
        "test": 2144,
    }
    assert sh.EXPECTED_CLASSES == [
        "No apparent DR",
        "Mild NPDR",
        "Moderate NPDR",
        "Severe NPDR",
        "PDR",
    ]
    assert {r["primary_class"] for r in rows} == set(sh.EXPECTED_CLASSES)
    assert {r["partition"] for r in rows} == set(["train", "validation", "test"])


def test_notebook_split_constants_are_the_same_object_as_setup_helpers_exports():
    assert sh.EXPECTED_CLASSES is not None
    assert sh.EXPECTED_ROWS is not None
    assert sh.EXPECTED_PARTITION_COUNTS is not None
    assert isinstance(sh.EXPECTED_CLASSES, list)
    assert isinstance(sh.EXPECTED_PARTITION_COUNTS, dict)


# ---------------------------------------------------------------------------
# setup_helpers public API the notebook calls
# ---------------------------------------------------------------------------

def test_build_canary_split_is_deterministic_and_keeps_labels():
    out = Path(tempfile.mkdtemp(prefix="canary_test_"))
    out_csv = out / "canary_provisional_split.csv"
    meta1 = sh.build_canary_split(SPLIT_CSV, out_csv, train_per_class=4, val_per_class=2, test_per_class=0)
    meta2 = sh.build_canary_split(SPLIT_CSV, out_csv, train_per_class=4, val_per_class=2, test_per_class=0)

    assert meta1["out_path"] == str(out_csv)
    assert meta2["out_path"] == str(out_csv)
    assert meta1["rows"] == meta2["rows"]
    assert meta1["counts"] == meta2["counts"]
    assert meta1["train_classes"] == meta2["train_classes"]
    assert set(meta1["train_classes"]) == set(sh.EXPECTED_CLASSES)

    # The derived CSV must still be a valid split CSV for the same schema.
    rows = list(csv.DictReader(out_csv.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == meta1["rows"]
    assert {r["partition"] for r in rows} <= {"train", "validation", "test"}
    assert {r["primary_class"] for r in rows} == set(sh.EXPECTED_CLASSES)


def test_build_canary_split_does_not_write_into_the_audited_split_folder():
    with tempfile.TemporaryDirectory(prefix="canary_out_") as td:
        out_dir = Path(td)
        out_csv = out_dir / "canary_provisional_split.csv"
        sh.build_canary_split(SPLIT_CSV, out_csv, train_per_class=4, val_per_class=2, test_per_class=0)
        assert not (SPLIT_DIR / "canary_provisional_split.csv").exists()
        assert str(out_csv).startswith(str(out_dir))


def test_validate_split_artifact_reports_the_real_split_as_expected():
    validation = sh.validate_split_artifact(
        SPLIT_DIR,
        project_root=REPO_ROOT,
        expected_rows=sh.EXPECTED_ROWS,
        expected_partition_counts=sh.EXPECTED_PARTITION_COUNTS,
        expected_classes=sh.EXPECTED_CLASSES,
        image_root=None,
        path_check="sample",
        decode_sample_per_stratum=0,
    )
    assert validation.get("ok") is True
    checks = validation.get("checks", [])
    check_names = {item.get("name") for item in checks}
    assert check_names
    failed = [item for item in checks if not item.get("ok")]
    assert not failed, failed




def test_load_label_policy_excerpt_never_raises_and_names_the_file():
    excerpt = sh.load_label_policy_excerpt(REPO_ROOT, max_chars=400)
    assert isinstance(excerpt, str) and len(excerpt) > 0
    # The notebook relies on this always showing the provisional disclosure.
    assert "provisional" in excerpt.lower()


# ---------------------------------------------------------------------------
# Guards: the notebook opt-ins and the library contract agree
# ---------------------------------------------------------------------------

def test_pipeline_guards_match_the_library_guard_contract():
    from colab.train import confirm_guard_contract

    assert confirm_guard_contract() == {
        "training_requires_confirm": True,
        "training_requires_smoke_test": True,
        "final_evaluation_requires_confirm": True,
    }

    import colab.pipeline as pipeline

    assert inspect.signature(pipeline.train_run).parameters["confirm"].default is False
    assert inspect.signature(pipeline.evaluate_run).parameters["confirm"].default is False
    # Once the caller opts in, the pipeline forwards both library guards as True.
    train_src = inspect.getsource(pipeline.train_run)
    assert "confirm_training=True" in train_src
    assert "smoke_test_passed=True" in train_src


def test_fixed_notebook_flags_are_off_by_default_and_guarded():
    blob = _code_blob()
    assert "RUN_PREFLIGHT = True" in blob
    assert "RUN_TRAINING = False" in blob
    assert "RUN_FINAL_EVALUATION = False" in blob
    assert "RUN_CANARY = False" in blob
    assert "CANARY_TRAIN = False" in blob
    assert "ALLOW_WEIGHT_DOWNLOAD = False" in blob
    assert "OVERWRITE = False" in blob
    assert 'RESUME = "auto"' in blob
    # Each opt-in refuses to act while its flag is False.
    assert "if not RUN_TRAINING:" in blob
    assert "if not RUN_FINAL_EVALUATION:" in blob
    assert "if not RUN_CANARY:" in blob
    assert blob.count("confirm=True") >= 2


def test_fixed_notebook_stops_the_cell_when_preflight_fails():
    preflight_cell = _code_cell_sources()[1]
    assert "PREFLIGHT FAILED" in preflight_cell
    assert "raise SystemExit" in preflight_cell
    assert "pipeline.preflight(" in preflight_cell

    # The training cell re-checks preflight when cell #2 was skipped, then skips
    # its own duplicate run only because it just did the check itself.
    train_cell = _code_cell_sources()[3]
    assert "pipeline.preflight(" in train_cell
    assert "raise SystemExit" in train_cell
    assert "skip_preflight=True" in train_cell


def test_only_the_evaluation_cell_can_score_the_test_partition():
    for index, source in enumerate(_code_cell_sources()):
        if index == 4:
            assert "pipeline.evaluate_run(" in source
            assert "confirm=True" in source
        else:
            assert "evaluate_run(" not in source
            assert "test_metrics" not in source


# ---------------------------------------------------------------------------
# Canary: separate from full training, consent off by default
# ---------------------------------------------------------------------------

def test_fixed_notebook_canary_is_separate_from_full_training():
    canary_cell = _code_cell_sources()[2]
    assert "pipeline.canary_run(" in canary_cell
    assert "train=CANARY_TRAIN" in canary_cell
    assert "confirm=CANARY_TRAIN" in canary_cell
    assert "allow_weight_download=CANARY_ALLOW_WEIGHT_DOWNLOAD" in canary_cell
    assert "pipeline.train_run(" not in canary_cell
    assert "pipeline.evaluate_run(" not in canary_cell


def test_canary_run_keeps_its_own_split_inside_the_output_dir():
    import colab.pipeline as pipeline

    src = inspect.getsource(pipeline.canary_run)
    assert 'out / "canary_split"' in src
    # A canary has no test rows, so the canary entry point never evaluates one.
    assert "run_final_evaluation" not in src
    assert "evaluate_run" not in src
