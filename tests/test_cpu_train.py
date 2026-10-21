"""Tests for the standalone CPU entry point: ``colab.cpu_train``.

Contract pinned here:

* the module never imports ``colab.pipeline`` (and never imports torch at
  module scope, so ``preflight`` can still report a broken environment);
* ``preflight`` reports every failure at once instead of dying on the first;
* the device is always forced to CPU; training is refused without ``--confirm``;
* there is no ``--overwrite`` path - existing runs are continued via ``--resume``;
* the canary split is derived into its own output folder and never touches the
  audited split artifact;
* the test partition is never passed to ``run_training`` (the engine only ever
  receives train/validation rows; nothing here even filters for "test").

No training runs and no weights are downloaded: ``run_training`` and the smoke
gate are monkeypatched, and preflight is exercised read-only.
"""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "colab"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from colab import cpu_train  # noqa: E402

CPU_TRAIN_PY = REPO_ROOT / "colab" / "cpu_train.py"
SPLIT_DIR = REPO_ROOT / "provisional_primary_split"
SPLIT_CSV = SPLIT_DIR / "provisional_primary_split.csv"


def _source() -> str:
    return CPU_TRAIN_PY.read_text(encoding="utf-8")


def _module_level_imports(tree: ast.Module):
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            yield node.module or ""


# ---------------------------------------------------------------------------
# Import isolation
# ---------------------------------------------------------------------------

def test_cpu_train_never_imports_colab_pipeline():
    tree = ast.parse(_source())
    names = list(_module_level_imports(tree))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    offenders = [n for n in names if "pipeline" in n.split(".")]
    assert offenders == [], f"cpu_train must not import colab.pipeline, found: {offenders}"


def test_cpu_train_never_imports_torch_at_module_scope():
    """A broken torch must not prevent ``preflight`` from reporting the failure."""
    tree = ast.parse(_source())
    top = [n for n in _module_level_imports(tree)]
    assert not any(name.split(".")[0] in {"torch", "torchvision", "numpy"} for name in top)


def test_cli_usage_strings_match_the_module():
    text = _source()
    assert "python -m colab.cpu_train" in text
    # no accidental overwrite flag anywhere in the parser
    assert '"--overwrite"' not in text


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def test_preflight_reports_all_failures_together(tmp_path):
    """Missing split + missing corpus must both appear in ONE report."""
    report = cpu_train.preflight(
        dataset_root=tmp_path / "no_such_corpus",
        split_dir=tmp_path / "no_such_split",
    )
    assert report["ok"] is False
    assert "split_dir" in report["failures"]
    assert "dataset_root" in report["failures"]
    # every check carries the same shape
    for c in report["checks"]:
        assert set(c) == {"name", "status", "ok", "detail"}
        assert c["status"] in {"pass", "fail", "skip"}
    assert report["device"] == "cpu"


@pytest.mark.skipif(not SPLIT_CSV.is_file(), reason="pinned split CSV absent")
def test_preflight_validates_the_real_split_artifact(tmp_path):
    """On the real corpus, the audited counts must be verified read-only."""
    report = cpu_train.preflight(split_dir=SPLIT_DIR, path_check="sample")
    by_name = {c["name"]: c for c in report["checks"]}
    assert by_name["row_count"]["status"] == "pass"
    assert "14201" in by_name["row_count"]["detail"]
    assert by_name["partition_counts_match_audited_split"]["status"] == "pass"
    assert by_name["no_group_leakage_across_partitions"]["status"] == "pass"
    # overall verdict may still be False here (torchvision/weights may be absent)
    assert isinstance(report["ok"], bool)


def test_discovery_finds_nothing_without_the_real_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("DR_SPLIT_DIR", str(tmp_path))
    monkeypatch.setenv("DR_FUNDUS_ROOT", str(tmp_path))
    # explicit None + empty env must simply return candidates, never raise
    assert cpu_train.discover_split_dir(tmp_path / "missing") is None
    assert cpu_train.discover_dataset_root(tmp_path / "missing") is None


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def test_train_refuses_without_confirm(tmp_path, monkeypatch, capsys):
    def _explode(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("run_training must not be reached without --confirm")

    import colab.train as train_mod

    monkeypatch.setattr(train_mod, "run_training", _explode)
    rc = cpu_train.main(["train", "--output-dir", str(tmp_path / "out")])
    assert rc == cpu_train.EXIT_REFUSED
    assert not (tmp_path / "out").exists()
    assert "needs --confirm" in capsys.readouterr().err


def test_canary_train_refuses_without_confirm(tmp_path):
    rc = cpu_train.main(["canary", "--train", "--output-dir", str(tmp_path / "out")])
    assert rc == cpu_train.EXIT_REFUSED
    assert not (tmp_path / "out").exists()


def test_train_forces_cpu_and_passes_the_guard_flags(tmp_path, monkeypatch):
    """cmd_train must hand run_training a CPU device and every guard flag."""
    import colab.train as train_mod

    captured = {}

    def _fake_run_training(**kwargs):
        captured.update(kwargs)
        return {
            "best_epoch": 0,
            "best_val_loss": 0.5,
            "output_dir": kwargs["output_dir"],
            "run_summary_path": kwargs["output_dir"] / "run_summary.json",
            "device": str(kwargs["device"]),
        }

    monkeypatch.setattr(train_mod, "run_training", _fake_run_training)
    monkeypatch.setattr(
        cpu_train, "preflight",
        lambda **k: {
            "ok": True, "checks": [], "failures": [],
            "paths": {"split_dir": str(SPLIT_DIR), "dataset_root": str(REPO_ROOT)},
        },
    )
    monkeypatch.setattr(
        cpu_train, "run_smoke_gate", lambda **k: {"ok": True, "tail": []}
    )

    out = tmp_path / "run_out"
    rc = cpu_train.main(["train", "--confirm", "--output-dir", str(out)])
    assert rc == cpu_train.EXIT_OK
    assert captured["device"].type == "cpu"
    assert captured["confirm_training"] is True
    assert captured["smoke_test_passed"] is True
    assert captured["overwrite"] is False
    assert captured["resume"] is False
    assert captured["allow_weight_download"] is False
    assert captured["split_dir"] == SPLIT_DIR
    # epochs/patience defaults are real values, not placeholders
    assert captured["epochs"] == 10 and captured["patience"] == 3


def test_smoke_gate_failure_blocks_training(tmp_path, monkeypatch, capsys):
    import colab.train as train_mod

    def _explode(*a, **k):  # pragma: no cover
        raise AssertionError("training must not start when the smoke gate fails")

    monkeypatch.setattr(train_mod, "run_training", _explode)
    monkeypatch.setattr(
        cpu_train, "preflight",
        lambda **k: {
            "ok": True, "checks": [], "failures": [],
            "paths": {"split_dir": str(SPLIT_DIR), "dataset_root": str(REPO_ROOT)},
        },
    )
    monkeypatch.setattr(
        cpu_train, "run_smoke_gate",
        lambda **k: {"ok": False, "command": "pytest smoke", "tail": ["boom"]},
    )
    rc = cpu_train.main(["train", "--confirm", "--output-dir", str(tmp_path / "o")])
    assert rc == cpu_train.EXIT_FAIL
    assert "smoke gate FAILED" in capsys.readouterr().err


def test_output_dir_guards_protect_the_split_and_dataset(tmp_path):
    with pytest.raises(ValueError, match="split directory"):
        cpu_train._assert_safe_output_dir(SPLIT_DIR / "some_run", SPLIT_DIR, tmp_path)
    with pytest.raises(ValueError, match="project root"):
        cpu_train._assert_safe_output_dir(REPO_ROOT, SPLIT_DIR, tmp_path)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    with pytest.raises(ValueError, match="dataset image root"):
        cpu_train._assert_safe_output_dir(corpus / "nested", tmp_path / "split", corpus)
    # a sibling under the repo is fine
    cpu_train._assert_safe_output_dir(REPO_ROOT / "cpu_runs" / "full", SPLIT_DIR, tmp_path)


# ---------------------------------------------------------------------------
# Canary split derivation
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.skipif(not SPLIT_CSV.is_file(), reason="pinned split CSV absent")
def test_canary_is_derived_outside_the_audited_split(tmp_path):
    import colab.setup_helpers as sh

    before = _sha256(SPLIT_CSV)
    out_dir = tmp_path / "canary_run"
    canary_dir, validation = cpu_train.build_and_validate_canary(
        SPLIT_DIR, out_dir, cpu_train.discover_dataset_root() or tmp_path
    )
    after = _sha256(SPLIT_CSV)
    assert before == after, "the audited split CSV was modified"

    assert canary_dir == out_dir / "canary_split"
    assert SPLIT_DIR not in canary_dir.parents
    assert validation["rows"] == 75
    assert validation["partition_counts"] == {"train": 50, "validation": 25}
    if cpu_train.discover_dataset_root() is not None:
        assert validation["ok"] is True, [
            c for c in validation["checks"] if not c["ok"]
        ]
    # the canary CSV itself is where load_split_csv will read it
    assert (canary_dir / sh.SPLIT_CSV_NAME).is_file()
