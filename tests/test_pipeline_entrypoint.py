"""Tests for the single entry point: ``colab.pipeline``.

These cover what the notebook now relies on and nothing else:

* project discovery and output-directory guards;
* ``preflight`` reporting every failure at once, and passing on a complete project;
* the CLI contract (exit codes and the actionable ``[BLOCKED]`` message).

No training runs, no weights are downloaded, and nothing in the real project or the
audited split is written to. The "complete project" case uses the real pinned split
plus a synthetic corpus materialized under ``tmp_path``.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "colab"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import colab.pipeline as pipeline  # noqa: E402
import colab.setup_helpers as sh  # noqa: E402

SPLIT_DIR = REPO_ROOT / "provisional_primary_split"
SPLIT_CSV = SPLIT_DIR / "provisional_primary_split.csv"


# ---------------------------------------------------------------------------
# Project discovery
# ---------------------------------------------------------------------------

def test_discover_project_finds_the_real_project_root():
    assert pipeline.discover_project() == REPO_ROOT
    assert pipeline.discover_project(str(REPO_ROOT)) == REPO_ROOT


def test_discover_project_honours_the_environment_variable(monkeypatch):
    monkeypatch.setenv("DR_PROJECT_ROOT", str(REPO_ROOT))
    monkeypatch.chdir(Path(os.environ.get("TEMP") or "/tmp"))
    assert pipeline.discover_project() == REPO_ROOT


def test_discover_project_fails_closed_on_a_wrong_explicit_root(tmp_path):
    """A wrong override must error, not silently resolve some other checkout."""
    empty = tmp_path / "not_the_project"
    empty.mkdir()
    with pytest.raises(FileNotFoundError) as excinfo:
        pipeline.discover_project(str(empty))
    message = str(excinfo.value)
    assert str(empty) in message
    assert "not a project root" in message
    # It names what is missing, so the fix is obvious.
    for marker in sh.PROJECT_MARKERS[:2]:
        assert marker in message


def test_project_markers_are_the_real_files_on_disk():
    assert sh.PROJECT_MARKERS, "the markers must not be empty"
    for marker in sh.PROJECT_MARKERS:
        assert (REPO_ROOT / marker).exists(), f"project marker {marker} does not exist"


# ---------------------------------------------------------------------------
# Output-directory safety
# ---------------------------------------------------------------------------

def test_output_dir_guard_refuses_the_project_and_its_audited_folders():
    with pytest.raises(ValueError, match="Refusing to use"):
        pipeline.ensure_safe_output_dir(REPO_ROOT, REPO_ROOT)
    for name in pipeline.PROTECTED_DIRS:
        with pytest.raises(ValueError, match="Refusing to use"):
            pipeline.ensure_safe_output_dir(REPO_ROOT / name, REPO_ROOT)


def test_output_dir_guard_refuses_a_parent_of_the_project():
    with pytest.raises(ValueError, match="lives inside it"):
        pipeline.ensure_safe_output_dir(REPO_ROOT.parent, REPO_ROOT)


def test_output_dir_guard_allows_a_dedicated_folder():
    chosen = REPO_ROOT / "dr_baseline_outputs"
    assert pipeline.ensure_safe_output_dir(chosen, REPO_ROOT) == chosen


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def _synthetic_corpus(root: Path) -> int:
    """Materialize every image path in the pinned split under ``root``."""
    from PIL import Image

    rows = list(csv.DictReader(SPLIT_CSV.read_text(encoding="utf-8").splitlines()))
    for row in rows:
        path = root / row["image_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 12), (7, 8, 9)).save(path)
    return len(rows)


def _check(report, name):
    return next(item for item in report["checks"] if item["name"] == name)


def _satisfied_runtime():
    """A runtime report in which every pinned dependency is present and new enough."""
    return {
        "python_version": "3.11.0",
        "executable": "python",
        "packages": {
            dist: {"present": True, "version": ".".join(str(part) for part in minimum), "error": None}
            for dist, minimum in sh.MINIMUM_VERSIONS.items()
        }
        | {"pytest": {"present": True, "version": "8.0.0", "error": None}},
        "cuda_available": False,
        "cuda_device_name": None,
    }


def test_preflight_passes_on_a_complete_project(tmp_path, monkeypatch):
    """The real split, all 14,201 paths present, every stratum decodable.

    The runtime probe is stubbed to a satisfied environment: torchvision is not
    installed in every environment this suite runs in, and this test is about
    preflight's aggregation, not about the machine it happens to run on. The real
    probe's behaviour is covered by ``test_preflight_reports_the_real_dependencies``.
    """
    monkeypatch.setattr(sh, "inspect_runtime_environment", _satisfied_runtime)
    out = tmp_path / "outputs"
    corpus = tmp_path / "corpus"
    written = _synthetic_corpus(corpus)
    assert written == sh.EXPECTED_ROWS == 14201

    report = pipeline.preflight(
        REPO_ROOT,
        dataset_root=corpus,
        output_dir=out,
        check_all_images=True,
        decode_per_stratum=1,
        run_smoke=False,
    )

    assert report["ok"] is True, report["failures"]
    assert report["failures"] == []
    assert report["num_rows"] == 14201
    assert _check(report, "all_image_paths_exist")["status"] == "pass"
    assert _check(report, "sample_images_decode")["status"] == "pass"
    assert _check(report, "split_artifact")["status"] == "pass"
    assert "9924" in _check(report, "split_artifact")["detail"]
    # Preflight may create the output dir, but never a run inside it.
    assert not (out / "run_manifest.json").exists()


def test_preflight_reports_the_real_dependencies_truthfully(tmp_path):
    """Whatever this machine has, the dependency verdicts must match the probe."""
    report = pipeline.preflight(
        REPO_ROOT, output_dir=tmp_path / "out", check_all_images=False, run_smoke=False
    )
    runtime = sh.inspect_runtime_environment()

    for dist, minimum in sh.MINIMUM_VERSIONS.items():
        entry = runtime["packages"][dist]
        status = _check(report, f"dependency_{dist}")["status"]
        if not entry["present"]:
            assert status == "fail", f"{dist} is absent but preflight said {status}"
        elif not sh.meets_minimum(str(entry["version"]), minimum):
            assert status == "fail"
        else:
            assert status == "pass"

    # Preflight's overall verdict is exactly "no failures", never a softened pass.
    assert report["ok"] is (report["failures"] == [])


def test_preflight_reports_every_unresolved_artifact_together(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sh, "resolve_data_root", lambda *a, **k: (None, ["nowhere (does not exist)"])
    )
    report = pipeline.preflight(REPO_ROOT, output_dir=tmp_path / "out", run_smoke=False)

    assert report["ok"] is False
    # Both image checks fail closed, and the corpus root is named once.
    for name in ("dataset_root", "all_image_paths_exist", "sample_images_decode"):
        assert name in report["failures"], report["failures"]
    for item in report["checks"]:
        if item["status"] == "fail":
            assert item["hint"], f"failure {item['name']} has no actionable hint"
    # Everything is reported at once, not one failure per run.
    assert len(report["failures"]) >= 3
    json.dumps(report)  # JSON-serializable for logging


def test_preflight_flags_a_missing_image_instead_of_failing_later(tmp_path):
    corpus = tmp_path / "corpus"
    _synthetic_corpus(corpus)
    rows = list(csv.DictReader(SPLIT_CSV.read_text(encoding="utf-8").splitlines()))
    victim = corpus / rows[0]["image_path"]
    victim.unlink()

    report = pipeline.preflight(
        REPO_ROOT, dataset_root=corpus, output_dir=tmp_path / "out", run_smoke=False
    )

    assert report["ok"] is False
    detail = _check(report, "all_image_paths_exist")["detail"]
    assert "1 of 14201" in detail
    assert str(victim) in detail
    assert _check(report, "all_image_paths_exist")["hint"]


def test_preflight_flags_an_undecodable_image(tmp_path):
    corpus = tmp_path / "corpus"
    _synthetic_corpus(corpus)
    rows = list(csv.DictReader(SPLIT_CSV.read_text(encoding="utf-8").splitlines()))
    victim = next(row for row in rows if row["partition"] == "validation")
    (corpus / victim["image_path"]).write_bytes(b"not an image")

    report = pipeline.preflight(
        REPO_ROOT,
        dataset_root=corpus,
        output_dir=tmp_path / "out",
        check_all_images=False,
        decode_per_stratum=2,
        run_smoke=False,
    )

    assert report["ok"] is False
    assert "sample_images_decode" in report["failures"]


def test_preflight_treats_a_failing_smoke_suite_as_fatal(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    _synthetic_corpus(corpus)
    monkeypatch.setattr(
        sh,
        "run_smoke_test",
        lambda *a, **k: {
            "status": "failed",
            "exit_code": 1,
            "summary": {"failed": 1},
            "failure_class": {"evidence": "FAILED tests/test_colab_pipeline.py::test_x"},
        },
    )

    report = pipeline.preflight(
        REPO_ROOT,
        dataset_root=corpus,
        output_dir=tmp_path / "out",
        check_all_images=False,
        decode_per_stratum=1,
        run_smoke=True,
    )

    assert report["ok"] is False
    assert "smoke_test" in report["failures"]
    assert "exit 1" in _check(report, "smoke_test")["detail"]


def test_preflight_does_not_write_into_the_project(tmp_path):
    def _snapshot():
        return {
            str(path.relative_to(REPO_ROOT)): path.stat().st_mtime_ns
            for path in REPO_ROOT.rglob("*")
            if path.is_file() and "dr_baseline_outputs" not in str(path)
            and "__pycache__" not in str(path) and ".pytest_cache" not in str(path)
        }

    before = _snapshot()
    pipeline.preflight(REPO_ROOT, output_dir=tmp_path / "out", run_smoke=False)
    after = _snapshot()
    assert after == before


def test_print_preflight_names_every_failure(capsys):
    report = {
        "command": "preflight",
        "ok": False,
        "project_root": "root",
        "split_csv": "csv",
        "dataset_root": None,
        "output_dir": "out",
        "checks": [
            {"name": "dataset_root", "status": "fail", "detail": "missing", "hint": "set it"},
            {"name": "device", "status": "warn", "detail": "cpu", "hint": ""},
        ],
        "failures": ["dataset_root"],
        "warnings": ["device"],
    }
    pipeline.print_preflight(report)
    output = capsys.readouterr().out
    assert "PREFLIGHT FAILED" in output
    assert "dataset_root" in output
    assert "-> set it" in output


# ---------------------------------------------------------------------------
# CLI contract (the commands the notebook mirrors)
# ---------------------------------------------------------------------------

def _cli(*args, env_extra=None):
    env = dict(os.environ)
    env.pop("DR_FUNDUS_ROOT", None)
    env.pop("DR_PROJECT_ROOT", None)
    env.pop("DR_OUTPUT_DIR", None)
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-m", "colab.pipeline", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=env,
    )


def test_cli_preflight_exits_non_zero_and_names_the_missing_corpus(tmp_path):
    result = _cli("preflight", "--no-smoke", "--skip-image-scan", "--output-dir", str(tmp_path))
    assert result.returncode == pipeline.EXIT_PREFLIGHT_FAILED
    assert "PREFLIGHT FAILED" in result.stdout
    assert "dataset_root" in result.stdout
    assert "DR_FUNDUS_ROOT" in result.stdout


def test_cli_train_without_confirm_is_blocked_with_a_clear_message():
    result = _cli("train")
    assert result.returncode == pipeline.EXIT_BLOCKED
    assert "[BLOCKED]" in result.stdout
    assert "--confirm" in result.stdout


def test_cli_evaluate_without_confirm_is_blocked():
    result = _cli("evaluate")
    assert result.returncode == pipeline.EXIT_BLOCKED
    assert "[BLOCKED]" in result.stdout
    assert "--confirm" in result.stdout


def test_cli_status_on_an_unused_output_dir(tmp_path):
    result = _cli("status", "--output-dir", str(tmp_path / "nothing_here"))
    assert result.returncode == pipeline.EXIT_OK
    assert "RUN STATUS" in result.stdout
    assert "resumable" in result.stdout


@pytest.mark.parametrize("json_first", [True, False])
def test_cli_json_status_is_machine_readable(tmp_path, json_first):
    args = ("--json", "status") if json_first else ("status", "--json")
    result = _cli(*args, "--output-dir", str(tmp_path / "nothing_here"))
    assert result.returncode == pipeline.EXIT_OK
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    assert payload["resumable"] is False
    assert payload["status"] is None


# ---------------------------------------------------------------------------
# Status / resume planning
# ---------------------------------------------------------------------------

def _fake_run(out: Path, *, epochs: int, best_epoch: int = 0, best_val_loss: float = 0.5):
    """Write the smallest set of artifacts that make a directory a resumable run."""
    from colab.train import initial_training_state, save_checkpoint

    import torch

    out.mkdir(parents=True, exist_ok=True)
    config = {"task": "primary_five_band_provisional", "resume_supported": True}
    state = initial_training_state(epoch=epochs - 1, best_val_loss=best_val_loss, best_epoch=best_epoch)
    state["train_history"] = [{"epoch": i, "loss": 1.0} for i in range(epochs)]
    state["val_history"] = [{"epoch": i, "loss": 1.0} for i in range(epochs)]
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for epoch in range(epochs):
        save_checkpoint(out, state, model, optimizer, config, epoch, is_best=epoch == best_epoch)
    (out / "run_manifest.json").write_text(
        json.dumps({"status": "in_progress", "config": config, "train_epochs": epochs}),
        encoding="utf-8",
    )
    return out


def test_run_state_describes_an_interrupted_run(tmp_path):
    out = _fake_run(tmp_path / "run", epochs=2)
    state = pipeline.run_state(out)

    assert state["exists"] is True
    assert state["resumable"] is True
    assert state["epochs_trained"] == 2
    assert state["latest_checkpoint"].endswith("epoch_0001")
    assert "resume" in state["reason"]


def test_run_state_on_an_empty_directory_says_so(tmp_path):
    state = pipeline.run_state(tmp_path / "missing")
    assert state["exists"] is False
    assert state["resumable"] is False
    assert "does not exist" in state["reason"]


# ---------------------------------------------------------------------------
# Persistent logs (a disconnect must not take the log with it)
# ---------------------------------------------------------------------------

def test_train_run_mirrors_its_output_into_a_log_beside_the_artifacts(tmp_path, monkeypatch):
    """stdout/stderr are mirrored into <output_dir>/logs, so a dead runtime keeps them."""
    import colab.train as train_mod

    out = tmp_path / "outputs"
    corpus = tmp_path / "corpus"
    (corpus / "11.Normal Fundus").mkdir(parents=True)
    calls = []

    def _stub(**kwargs):
        calls.append(kwargs)
        print("epoch_end stanza from the stubbed loop")
        return {
            "best_epoch": 0,
            "best_val_loss": 0.25,
            "final_state": {"train_history": [{"epoch": 0}]},
            "device": "cpu",
            "best_checkpoint_dir": Path(kwargs["output_dir"]) / "best_checkpoint",
        }

    monkeypatch.setattr(train_mod, "run_training", _stub)

    result = pipeline.train_run(
        project_root=REPO_ROOT,
        dataset_root=corpus,
        output_dir=out,
        epochs=1,
        confirm=True,
        skip_preflight=True,
    )

    assert calls and calls[0]["confirm_training"] is True
    log_path = Path(result["log_path"])
    assert log_path.is_file()
    assert log_path.parent == out / "logs"
    text = log_path.read_text(encoding="utf-8")
    assert "[pipeline] output_dir" in text
    assert "epoch_end stanza from the stubbed loop" in text
    assert result["ok"] is True


def test_evaluate_run_mirrors_its_output_into_a_log(tmp_path, monkeypatch):
    import colab.train as train_mod

    out = tmp_path / "run"
    (out / "best_checkpoint").mkdir(parents=True)
    (out / "best_checkpoint" / "model.pt").write_bytes(b"stub")
    (out / "run_manifest.json").write_text(
        json.dumps({"status": "completed", "train_epochs": 1}), encoding="utf-8"
    )
    corpus = tmp_path / "corpus"
    (corpus / "11.Normal Fundus").mkdir(parents=True)

    def _stub(**kwargs):
        print("final_evaluation_done (stub)")
        return {
            "metrics": {
                "num_samples": 3,
                "accuracy": 0.5,
                "macro_precision": 0.4,
                "macro_recall": 0.4,
                "macro_f1": 0.4,
                "confusion_matrix": [[1, 0], [0, 2]],
                "provisional_note": "provisional",
            },
            "output_path": str(kwargs["output_dir"] / "final_test_evaluation" / "final_test_metrics.json"),
        }

    monkeypatch.setattr(train_mod, "run_final_evaluation", _stub)

    result = pipeline.evaluate_run(
        project_root=REPO_ROOT, dataset_root=corpus, output_dir=out, confirm=True
    )

    log_path = Path(result["log_path"])
    assert log_path.parent == out / "logs"
    assert "final_evaluation_done (stub)" in log_path.read_text(encoding="utf-8")
    assert result["metrics"]["num_samples"] == 3


def test_evaluate_run_refuses_an_unfinished_run(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "run_manifest.json").write_text(
        json.dumps({"status": "in_progress"}), encoding="utf-8"
    )
    corpus = tmp_path / "corpus"
    (corpus / "11.Normal Fundus").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="not 'completed'"):
        pipeline.evaluate_run(
            project_root=REPO_ROOT, dataset_root=corpus, output_dir=out, confirm=True
        )


# ---------------------------------------------------------------------------
# The audited artifacts this work is not allowed to change
# ---------------------------------------------------------------------------

def test_the_audited_split_is_still_the_audited_split():
    from collections import Counter

    from dataset.labels import PRIMARY_MAPPING

    rows = list(csv.DictReader(SPLIT_CSV.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == sh.EXPECTED_ROWS == 14201
    assert dict(Counter(row["partition"] for row in rows)) == sh.EXPECTED_PARTITION_COUNTS
    assert sorted({row["primary_class"] for row in rows}) == sorted(sh.EXPECTED_CLASSES)

    # The mapping is five-band and every band it produces is one the split uses.
    assert sorted(set(PRIMARY_MAPPING.values())) == sorted(sh.EXPECTED_CLASSES)

    # No derived canary was written into the audited folder.
    assert not list(SPLIT_DIR.glob("canary*"))
    assert sorted(path.name for path in SPLIT_DIR.iterdir() if path.is_file()) == [
        "provisional_primary_split.csv",
        "provisional_primary_split_report.md",
    ]
