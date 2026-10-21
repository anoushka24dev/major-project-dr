"""Single command-line entry point for the diabetic-retinopathy baseline.

Everything the notebook used to do through chained global state lives here as a
plain function, so the notebook is only a thin wrapper and the same commands work
from a terminal, from ``bash``/``!`` in Colab, and from tests.

Commands
--------
``preflight``  Check the whole environment and every artifact at once: project
               markers, dependency versions, the pinned 14,201-row split, all
               14,201 image paths, a decode sample per class per partition,
               group leakage, weight-cache state, device, and output-dir
               writability. Reports **all** failures together and exits non-zero.
``canary``     Build the 75-row derived canary split, validate it with the
               canary-specific validator, and optionally train it.
``train``      Start or resume a run. ``--resume`` continues the existing run in
               the output directory (model + optimizer + early-stopping state).
``evaluate``   Score the held-out test partition once, from the validation-selected
               checkpoint. This is the only command that touches test rows.
``status``     Report what is in an output directory and whether it can resume.

Examples
--------
    python -m colab.pipeline preflight
    python -m colab.pipeline train --confirm
    python -m colab.pipeline train --confirm --resume
    python -m colab.pipeline evaluate --confirm
    python -m colab.pipeline canary --train --allow-weight-download
    python -m colab.pipeline status

Guards are unchanged: ``train`` and ``evaluate`` require ``--confirm``, and
training still refuses to start unless ``--allow-weight-download`` is given when
the ImageNet weights are not already cached. Nothing here downloads weights,
deletes artifacts, regenerates the split, or changes the label mapping.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TextIO

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "colab")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import colab.setup_helpers as sh  # noqa: E402

#: Directories that must never be used as a run output directory.
PROTECTED_DIRS = ("dataset", "colab", "tests", "audit_output", "provisional_primary_split")

#: Exit codes, so a shell/CI script can branch on the failure class.
EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_PREFLIGHT_FAILED = 2
EXIT_BLOCKED = 3


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

class _Tee:
    """Write to several streams at once (console + log file)."""

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            try:
                stream.write(data)
            except Exception:  # pragma: no cover - a dead console must not kill a run
                pass
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            try:
                stream.flush()
            except Exception:  # pragma: no cover
                pass


def _json_default(value: Any) -> str:
    return str(value)


def _say(message: str = "") -> None:
    print(message)


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def discover_project(explicit: Optional[str] = None) -> Path:
    """Return the project root, or raise with the exact paths that were checked.

    Resolution order: explicit argument, ``DR_PROJECT_ROOT``, the current working
    directory, ``/content/major_project``, then a bounded scan of a mounted Drive.
    """
    # An explicit path is an instruction, not a hint: if it is wrong, say so instead
    # of silently training from a different checkout.
    if explicit:
        chosen = Path(explicit).expanduser()
        missing = sh.missing_markers(chosen)
        if missing:
            raise FileNotFoundError(
                f"The project root you passed is not a project root: {chosen}\n"
                f"Missing: {', '.join(missing)}\n"
                "Point it at the folder that contains dataset/ and colab/."
            )
        return chosen.resolve()

    candidates: List[Path] = []
    env = (os.environ.get("DR_PROJECT_ROOT") or "").strip()
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(Path.cwd())
    candidates.append(Path("/content/major_project"))
    for mount in (Path("/content/drive/MyDrive"), Path("/drive/MyDrive")):
        if mount.is_dir():
            candidates.append(mount / "major_project")
            candidates.append(mount)

    checked: List[str] = []
    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:  # pragma: no cover - defensive
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.is_dir():
            checked.append(f"{resolved} (does not exist)")
            continue
        missing = sh.missing_markers(resolved)
        if not missing:
            return resolved
        checked.append(f"{resolved} (missing {', '.join(missing[:3])})")

    # Last resort: a bounded scan, never an unbounded Drive crawl.
    for base in (Path("/content"), Path.cwd(), Path("/content/drive/MyDrive")):
        if not base.is_dir():
            continue
        scan = sh.bounded_scan(base, marker=sh.BOOTSTRAP_MARKER)
        for match in scan["matches"]:
            if not sh.missing_markers(match):
                return match

    raise FileNotFoundError(
        "Could not locate the project root. Set DR_PROJECT_ROOT (or pass --project-root) "
        "to the folder that contains dataset/ and colab/.\nChecked:\n  - "
        + "\n  - ".join(checked)
    )


def default_output_dir(project_root: Path) -> Path:
    """Prefer Drive (persistent across Colab disconnects), else the project folder."""
    env = (os.environ.get("DR_OUTPUT_DIR") or "").strip()
    if env:
        return Path(env).expanduser()
    for mount in (Path("/content/drive/MyDrive"), Path("/drive/MyDrive")):
        if mount.is_dir():
            return mount / "dr_baseline_outputs"
    return Path(project_root) / "dr_baseline_outputs"


def ensure_safe_output_dir(output_dir: Path, project_root: Path) -> Path:
    """Refuse an output directory that would let ``overwrite`` delete project files."""
    out = Path(output_dir).expanduser()
    root = Path(project_root).resolve()
    resolved = out.resolve() if out.exists() else out.absolute()
    protected = {root} | {(root / name).resolve() for name in PROTECTED_DIRS}
    if resolved in protected:
        raise ValueError(
            f"Refusing to use {out} as the output directory: it is the project root or one of "
            f"its audited folders ({', '.join(PROTECTED_DIRS)}). Pick a separate folder such as "
            f"{root / 'dr_baseline_outputs'}."
        )
    if resolved in root.parents:
        raise ValueError(
            f"Refusing to use {out} as the output directory: the project root {root} lives inside "
            "it, so an overwrite would delete the project. An output folder inside the project "
            f"(for example {root / 'dr_baseline_outputs'}) or elsewhere is fine."
        )
    return out


def _check(report: Dict[str, Any], name: str, status: str, detail: str, hint: str = "") -> None:
    report["checks"].append({"name": name, "status": status, "detail": detail, "hint": hint})
    if status == "fail":
        report["failures"].append(name)
    elif status == "warn":
        report["warnings"].append(name)


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def preflight(
    project_root: Optional[Path | str] = None,
    *,
    dataset_root: Optional[Path | str] = None,
    output_dir: Optional[Path | str] = None,
    check_all_images: bool = True,
    decode_per_stratum: int = 2,
    run_smoke: bool = True,
    smoke_timeout: int = 1800,
) -> Dict[str, Any]:
    """Check the entire environment and every artifact, reporting all failures.

    Never trains, never downloads weights, never writes inside the project. The
    only side effect is running the smoke suite as a subprocess when
    ``run_smoke=True``.

    Returns a JSON-serializable report with ``ok``, ``checks``, ``failures`` and
    ``warnings``. ``ok`` is False when any check failed; a failed preflight means
    training would fail too, so callers should stop.
    """
    report: Dict[str, Any] = {
        "command": "preflight",
        "ok": False,
        "checks": [],
        "failures": [],
        "warnings": [],
        "project_root": None,
        "split_csv": None,
        "dataset_root": None,
        "timestamp": _now_stamp(),
    }

    # 1. Project ------------------------------------------------------------
    try:
        root = discover_project(str(project_root) if project_root else None)
    except FileNotFoundError as exc:
        _check(report, "project_discovery", "fail", str(exc),
               "Set DR_PROJECT_ROOT to the folder containing dataset/ and colab/.")
        return report
    report["project_root"] = str(root)
    _check(report, "project_discovery", "pass", str(root))

    # 2. Runtime / dependencies --------------------------------------------
    try:
        runtime = sh.inspect_runtime_environment()
    except Exception as exc:  # pragma: no cover - defensive
        _check(report, "runtime_inspection", "fail", f"{type(exc).__name__}: {exc}")
        runtime = {"packages": {}, "cuda_available": False, "python_version": "?"}

    _check(report, "python_version", "pass", str(runtime.get("python_version")))
    for dist, minimum in sh.MINIMUM_VERSIONS.items():
        entry = (runtime.get("packages") or {}).get(dist, {})
        if not entry.get("present"):
            status = "fail" if dist in ("torch", "torchvision", "numpy", "Pillow") else "warn"
            _check(
                report,
                f"dependency_{dist}",
                status,
                f"not importable ({entry.get('error')})",
                "python -m pip install -r colab/requirements-colab.txt"
                if dist in ("torch", "torchvision")
                else f"python -m pip install {dist}",
            )
        elif not sh.meets_minimum(str(entry.get("version")), minimum):
            _check(
                report,
                f"dependency_{dist}",
                "fail",
                f"{entry.get('version')} is older than the required {'.'.join(map(str, minimum))}",
                "python -m pip install -r colab/requirements-colab.txt",
            )
        else:
            _check(report, f"dependency_{dist}", "pass", str(entry.get("version")))
    if (runtime.get("packages") or {}).get("pytest", {}).get("present"):
        _check(report, "dependency_pytest", "pass",
               str(runtime["packages"]["pytest"].get("version")))
    else:
        _check(report, "dependency_pytest", "warn", "not importable; the smoke gate cannot run",
               "python -m pip install pytest")

    report["device"] = {
        "cuda_available": bool(runtime.get("cuda_available")),
        "cuda_device_name": runtime.get("cuda_device_name"),
    }
    _check(
        report,
        "device",
        "pass" if runtime.get("cuda_available") else "warn",
        f"cuda_available={bool(runtime.get('cuda_available'))}"
        + (f" ({runtime.get('cuda_device_name')})" if runtime.get("cuda_device_name") else "")
        + ("; training will run on CPU" if not runtime.get("cuda_available") else ""),
        "" if runtime.get("cuda_available") else
        "Runtime > Change runtime type > GPU for a CUDA run (CPU still works, just slower).",
    )

    # 3. Split artifact (strict, unchanged validator) -----------------------
    split_dir = root / sh.SPLIT_DIR_NAME
    report["split_csv"] = str(split_dir / sh.SPLIT_CSV_NAME)
    split_validation = sh.validate_split_artifact(
        split_dir,
        project_root=root,
        expected_rows=sh.EXPECTED_ROWS,
        expected_partition_counts=sh.EXPECTED_PARTITION_COUNTS,
        expected_classes=sh.EXPECTED_CLASSES,
        image_root=None,
        path_check="off",
        decode_sample_per_stratum=0,
    )
    for item in split_validation.get("checks", []):
        if item["status"] == "fail":
            _check(report, f"split:{item['name']}", "fail", item["detail"],
                   "The pinned split is authoritative; restore it rather than regenerating it.")
        elif item["status"] == "skip":
            continue
    if split_validation.get("ok"):
        _check(
            report,
            "split_artifact",
            "pass",
            f"{split_validation['rows']} rows, partitions {split_validation['partition_counts']}",
        )

    # 4. Corpus ------------------------------------------------------------
    root_hint = dataset_root or (os.environ.get(sh.DATASET_ENV_VAR) or "").strip() or None
    data_root, checked_paths = sh.resolve_data_root(root, explicit=root_hint)
    report["dataset_root"] = str(data_root) if data_root else None
    report["dataset_root_checked"] = checked_paths
    if data_root is None:
        _check(report, "dataset_root", "fail", "the fundus corpus was not found",
               f"Set {sh.DATASET_ENV_VAR} to the folder that holds the source-label subfolders "
               "(e.g. '11.Normal Fundus/'). Checked: " + " | ".join(checked_paths))
    else:
        _check(report, "dataset_root", "pass", str(data_root))

    # 5 + 6 + 7. Images, decoding, leakage ---------------------------------
    if data_root is not None:
        rows = list(_load_split_rows(split_dir))
        report["num_rows"] = len(rows)

        missing: List[str] = []
        examined = 0
        for row in rows:
            path = data_root / row["image_path"]
            if check_all_images or examined < 400:
                examined += 1
                if not path.is_file():
                    missing.append(str(path))
        _check(
            report,
            "all_image_paths_exist" if check_all_images else "image_paths_sampled",
            "fail" if missing else "pass",
            f"all {len(rows)} image paths exist under {data_root}" if not missing
            else (
                f"{len(missing)} of {len(rows)} image paths are missing, first: "
                + "; ".join(missing[:3])
            ),
            f"Point {sh.DATASET_ENV_VAR} at the folder that actually contains these images."
            if missing else "",
        )

        decode = sh.validate_split_artifact(
            split_dir,
            project_root=root,
            expected_rows=sh.EXPECTED_ROWS,
            expected_partition_counts=sh.EXPECTED_PARTITION_COUNTS,
            expected_classes=sh.EXPECTED_CLASSES,
            image_root=data_root,
            path_check="off",
            decode_sample_per_stratum=max(1, int(decode_per_stratum)),
            run_protocol_validation=False,
        )
        decode_check = next(
            (c for c in decode.get("checks", []) if c["name"] == "sample_images_decode"), None
        )
        if decode_check is None:
            _check(report, "sample_images_decode", "warn", "decoder did not report")
        else:
            _check(
                report,
                "sample_images_decode",
                "pass" if decode_check["ok"] else "fail",
                decode_check["detail"],
                "A file that will not decode has to be replaced, not skipped." if not decode_check["ok"] else "",
            )
    else:
        _check(report, "all_image_paths_exist", "fail",
               "skipped because the corpus root is unresolved (failing closed)",
               f"Set {sh.DATASET_ENV_VAR}.")
        _check(report, "sample_images_decode", "fail",
               "skipped because the corpus root is unresolved (failing closed)",
               f"Set {sh.DATASET_ENV_VAR}.")

    # 8. Eligibility manifest ---------------------------------------------
    manifest_path = root / sh.ELIGIBILITY_MANIFEST_RELPATH
    if manifest_path.is_file():
        _check(report, "eligibility_manifest", "pass", str(manifest_path))
    else:
        _check(report, "eligibility_manifest", "warn", f"missing {manifest_path}",
               "Upload the full project including audit_output/ so protocol re-validation can run.")

    # 9. Weights cache (informational: consent can override) ----------------
    try:
        from colab_helpers import pretrained_weights_available, weights_cache_path

        cached = pretrained_weights_available()
        report["weights_cached"] = cached
        _check(
            report,
            "imagenet_weights",
            "pass" if cached else "warn",
            f"cached={cached} at {weights_cache_path()}",
            "" if cached else
            "Pass --allow-weight-download to train (one-time ~100 MB), or pre-seed the torch cache.",
        )
    except Exception as exc:
        report["weights_cached"] = None
        _check(report, "imagenet_weights", "warn", f"could not determine: {exc}")

    # 10. Output directory -------------------------------------------------
    try:
        out = ensure_safe_output_dir(
            Path(output_dir) if output_dir else default_output_dir(root), root
        )
        created_here = not out.exists()
        out.mkdir(parents=True, exist_ok=True)
        probe = out / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        if created_here and not any(out.iterdir()):
            # Preflight must not leave artifacts behind; only training creates runs.
            out.rmdir()
        report["output_dir"] = str(out)
        _check(report, "output_dir_writable", "pass", str(out))
    except Exception as exc:
        _check(report, "output_dir_writable", "fail", f"{type(exc).__name__}: {exc}",
               "Pick a writable, non-protected output directory with --output-dir.")

    # 11. Smoke suite ------------------------------------------------------
    if run_smoke:
        smoke = sh.run_smoke_test(root, env_overrides={"DR_SPLIT_DIR": str(split_dir)},
                                  timeout=smoke_timeout)
        report["smoke"] = {
            "status": smoke.get("status"),
            "exit_code": smoke.get("exit_code"),
            "summary": smoke.get("summary"),
        }
        if smoke.get("status") == "failed":
            _check(report, "smoke_test", "fail",
                   f"exit {smoke.get('exit_code')}: {(smoke.get('failure_class') or {}).get('evidence', '')}",
                   "Read the pytest output in the smoke summary and fix the cause.")
        elif smoke.get("status") == "passed_with_skips":
            _check(report, "smoke_test", "warn",
                   "passed with skipped image checks (corpus checks are covered above)")
        else:
            _check(report, "smoke_test", "pass", "all smoke checks ran and passed")

    report["ok"] = not report["failures"]
    return report


def _load_split_rows(split_dir: Path):
    import csv

    with (Path(split_dir) / sh.SPLIT_CSV_NAME).open(newline="", encoding="utf-8") as fh:
        yield from csv.DictReader(fh)


def print_preflight(report: Dict[str, Any]) -> None:
    """Human-readable preflight table."""
    _say("")
    _say("=" * 78)
    _say("PREFLIGHT")
    _say("=" * 78)
    _say(f"project     : {report.get('project_root')}")
    _say(f"split CSV   : {report.get('split_csv')}")
    _say(f"dataset root: {report.get('dataset_root')}")
    _say(f"output dir  : {report.get('output_dir')}")
    _say("")
    for item in report["checks"]:
        label = {"pass": "PASS", "fail": "FAIL", "warn": "WARN", "skip": "SKIP"}[item["status"]]
        _say(f"  [{label}] {item['name']}: {item['detail']}")
        if item["status"] == "fail" and item.get("hint"):
            _say(f"         -> {item['hint']}")
    _say("")
    if report["ok"]:
        _say(f"PREFLIGHT OK — {len(report['checks'])} checks, "
             f"{len(report['warnings'])} warning(s).")
    else:
        _say(f"PREFLIGHT FAILED — {len(report['failures'])} failure(s): "
             + ", ".join(report["failures"]))
        _say("Fix every failure above; nothing below them is worth running until they pass.")


# ---------------------------------------------------------------------------
# Run state / status
# ---------------------------------------------------------------------------

def run_state(output_dir: Path | str) -> Dict[str, Any]:
    """Describe an output directory: status, epochs, resume eligibility."""
    import colab.train as train_mod

    out = Path(output_dir).expanduser()
    state: Dict[str, Any] = {
        "output_dir": str(out),
        "exists": out.is_dir(),
        "status": None,
        "epochs_trained": 0,
        "best_epoch": None,
        "best_val_loss": None,
        "latest_checkpoint": None,
        "resumable": False,
        "reason": "",
        "has_test_metrics": (out / "final_test_evaluation" / "final_test_metrics.json").is_file(),
    }
    if not state["exists"]:
        state["reason"] = "output directory does not exist yet"
        return state

    manifest = train_mod._read_json(out / train_mod.RUN_MANIFEST_NAME) or {}
    state["status"] = manifest.get("status")
    state["best_epoch"] = manifest.get("best_epoch")
    state["best_val_loss"] = manifest.get("best_val_loss")
    state["epochs_trained"] = int(manifest.get("train_epochs") or 0)

    epoch_dir, last_epoch = train_mod.latest_epoch_checkpoint(out)
    state["latest_checkpoint"] = str(epoch_dir) if epoch_dir else None
    if last_epoch is not None:
        state["epochs_trained"] = max(state["epochs_trained"], int(last_epoch) + 1)

    if last_epoch is None:
        state["reason"] = "no complete checkpoints/epoch_XXXX/model.pt yet"
    elif state["status"] == "completed":
        state["resumable"] = True
        state["reason"] = (
            f"completed after {state['epochs_trained']} epoch(s); resumable only to extend "
            "(pass --epochs greater than that)"
        )
    else:
        state["resumable"] = True
        state["reason"] = f"interrupted/ongoing at epoch {last_epoch}; --resume continues here"
    return state


def print_status(state: Dict[str, Any]) -> None:
    _say("")
    _say("=" * 78)
    _say("RUN STATUS")
    _say("=" * 78)
    for key in ("output_dir", "status", "epochs_trained", "best_epoch", "best_val_loss",
                "latest_checkpoint", "resumable", "reason", "has_test_metrics"):
        _say(f"  {key:18}: {state.get(key)}")
    if not state["resumable"]:
        _say("")
        _say("Nothing to resume here; start a run with: python -m colab.pipeline train --confirm")


# ---------------------------------------------------------------------------
# Training / evaluation / canary
# ---------------------------------------------------------------------------

def _prepare(
    project_root: Optional[Path | str],
    dataset_root: Optional[Path | str],
    output_dir: Optional[Path | str],
) -> tuple[Path, Path, Path, Path]:
    root = discover_project(str(project_root) if project_root else None)
    data_root, checked = sh.resolve_data_root(
        root, explicit=dataset_root or (os.environ.get(sh.DATASET_ENV_VAR) or "").strip() or None
    )
    if data_root is None:
        raise FileNotFoundError(
            f"Could not find the fundus corpus. Set {sh.DATASET_ENV_VAR}.\nChecked:\n  - "
            + "\n  - ".join(checked)
        )
    split_dir = root / sh.SPLIT_DIR_NAME
    if not (split_dir / sh.SPLIT_CSV_NAME).is_file():
        raise FileNotFoundError(
            f"Pinned split CSV missing at {split_dir / sh.SPLIT_CSV_NAME}. Restore it; do not "
            "regenerate the audited split."
        )
    out = ensure_safe_output_dir(Path(output_dir) if output_dir else default_output_dir(root), root)
    return root, split_dir, data_root, out


def _with_log(out: Path, name: str):
    """Context manager: mirror stdout/stderr into <out>/logs/<name>.log."""
    log_dir = Path(out) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}_{_now_stamp()}.log"
    handle = log_path.open("a", encoding="utf-8")
    tee = _Tee(sys.stdout, handle)
    err = _Tee(sys.stderr, handle)
    return log_path, contextlib.redirect_stdout(tee), contextlib.redirect_stderr(err), handle


def train_run(
    *,
    project_root: Optional[Path | str] = None,
    dataset_root: Optional[Path | str] = None,
    output_dir: Optional[Path | str] = None,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 3,
    image_size: int = 224,
    seed: int = 12345,
    allow_weight_download: bool = False,
    resume: bool | str = False,
    overwrite: bool = False,
    confirm: bool = False,
    skip_preflight: bool = False,
) -> Dict[str, Any]:
    """Start or resume a training run, with logs written beside the artifacts.

    ``resume`` accepts ``False`` (always fresh), ``True`` (must resume an existing
    run), or ``"auto"`` (resume when ``output_dir`` holds a resumable run,
    otherwise start fresh). ``"auto"`` is what the notebook uses, so a runtime
    disconnect only needs the cell re-run -- no decision, no lost epochs.
    """
    if not confirm:
        raise PermissionError(
            "Training is guarded. Re-run with --confirm to start or resume a run."
        )

    root, split_dir, data_root, out = _prepare(project_root, dataset_root, output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not skip_preflight:
        report = preflight(
            root, dataset_root=data_root, output_dir=out, run_smoke=True,
            check_all_images=False,
        )
        print_preflight(report)
        if not report["ok"]:
            raise RuntimeError(
                "Preflight failed, so training was not started. Fix: "
                + ", ".join(report["failures"])
            )

    import colab.train as train_mod

    before = run_state(out)
    auto = resume == "auto"
    use_resume = bool(before["resumable"]) if auto else bool(resume)
    if resume is True and not before["resumable"]:
        raise FileNotFoundError(
            f"--resume was requested but {out} has nothing to resume ({before['reason']}).\n"
            "Drop --resume to start a fresh run."
        )

    log_path, out_ctx, err_ctx, handle = _with_log(out, "train")
    started = time.time()
    try:
        with out_ctx, err_ctx:
            _say(f"[pipeline] split_dir  : {split_dir}")
            _say(f"[pipeline] dataset    : {data_root}")
            _say(f"[pipeline] output_dir : {out}")
            _say(f"[pipeline] resume     : {use_resume}"
                 + (f" (from {before['latest_checkpoint']})" if use_resume else ""))
            _say(f"[pipeline] log file   : {log_path}")
            result = train_mod.run_training(
                split_dir=split_dir,
                dataset_root=data_root,
                output_dir=out,
                epochs=int(epochs),
                batch_size=int(batch_size),
                lr=float(lr),
                weight_decay=float(weight_decay),
                patience=int(patience),
                image_size=int(image_size),
                seed=int(seed),
                confirm_training=True,
                smoke_test_passed=True,
                allow_weight_download=bool(allow_weight_download),
                overwrite=bool(overwrite),
                resume=use_resume,
            )
    except KeyboardInterrupt:
        _say("")
        _say("[pipeline] interrupted. Checkpoints stay on disk; continue with:")
        _say(f"    python -m colab.pipeline train --confirm --resume --output-dir '{out}'")
        raise
    finally:
        handle.close()

    after = run_state(out)
    return {
        "command": "train",
        "ok": True,
        "output_dir": str(out),
        "log_path": str(log_path),
        "seconds": round(time.time() - started, 1),
        "resumed": use_resume,
        "best_epoch": result["best_epoch"],
        "best_val_loss": result["best_val_loss"],
        "epochs_trained": len(result["final_state"]["train_history"]),
        "device": result["device"],
        "best_checkpoint_dir": str(result["best_checkpoint_dir"]),
        "state": after,
        "next": [
            f"python -m colab.pipeline status --output-dir '{out}'",
            f"python -m colab.pipeline evaluate --confirm --output-dir '{out}'",
        ],
    }


def evaluate_run(
    *,
    project_root: Optional[Path | str] = None,
    dataset_root: Optional[Path | str] = None,
    output_dir: Optional[Path | str] = None,
    batch_size: int = 32,
    image_size: int = 224,
    confirm: bool = False,
) -> Dict[str, Any]:
    """Score the held-out test partition once, from the best checkpoint."""
    if not confirm:
        raise PermissionError(
            "Final test evaluation is guarded. Re-run with --confirm to score the test partition."
        )

    root, split_dir, data_root, out = _prepare(project_root, dataset_root, output_dir)
    import colab.train as train_mod

    state = run_state(out)
    if state["status"] != "completed":
        raise RuntimeError(
            f"Refusing to evaluate: the run at {out} is {state['status']!r}, not 'completed'.\n"
            "Finish training first (the test partition is never scored from a partial run)."
        )
    best_dir = out / train_mod.BEST_CHECKPOINT_DIR_NAME
    if not (best_dir / "model.pt").is_file():
        raise FileNotFoundError(
            f"Best checkpoint missing: {best_dir / 'model.pt'}. Training did not select a best epoch."
        )

    eval_path = out / train_mod.TEST_EVAL_DIR_NAME / "final_test_metrics.json"
    if eval_path.is_file():
        _say(f"[pipeline] NOTE: overwriting the previous test metrics at {eval_path}")

    log_path, out_ctx, err_ctx, handle = _with_log(out, "evaluate")
    try:
        with out_ctx, err_ctx:
            _say(f"[pipeline] held-out evaluation from {best_dir}")
            _say(f"[pipeline] log file: {log_path}")
            result = train_mod.run_final_evaluation(
                split_dir=split_dir,
                dataset_root=data_root,
                output_dir=out,
                checkpoint_dir=best_dir,
                batch_size=int(batch_size),
                image_size=int(image_size),
                confirm_final_evaluation=True,
            )
    finally:
        handle.close()

    metrics = result["metrics"]
    return {
        "command": "evaluate",
        "ok": True,
        "output_dir": str(out),
        "log_path": str(log_path),
        "output_path": result["output_path"],
        "metrics": {
            "num_samples": metrics["num_samples"],
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "confusion_matrix": metrics["confusion_matrix"],
        },
        "checkpoint": str(best_dir),
        "provisional_note": metrics["provisional_note"],
    }


def canary_run(
    *,
    project_root: Optional[Path | str] = None,
    dataset_root: Optional[Path | str] = None,
    output_dir: Optional[Path | str] = None,
    epochs: int = 2,
    batch_size: int = 8,
    seed: int = 12345,
    allow_weight_download: bool = False,
    train: bool = False,
    confirm: bool = False,
) -> Dict[str, Any]:
    """Build and validate the 75-row canary split; optionally train it."""
    root, _, data_root, out = _prepare(project_root, dataset_root, output_dir)

    split_dir = out / "canary_split"
    split_dir.mkdir(parents=True, exist_ok=True)
    csv_path = split_dir / sh.SPLIT_CSV_NAME
    meta = sh.build_canary_split(
        root / sh.SPLIT_DIR_NAME / sh.SPLIT_CSV_NAME,
        csv_path,
        train_per_class=10,
        val_per_class=5,
        test_per_class=0,
    )
    if Path(meta["out_path"]) != csv_path or not csv_path.is_file():
        raise RuntimeError(
            f"Canary writer/reader path mismatch: wrote {meta['out_path']}, expected {csv_path}."
        )

    validation = sh.validate_canary_split(
        split_dir,
        dataset_root=data_root,
        expected_train=50,
        expected_validation=25,
        expected_test=0,
        train_per_class=10,
        val_per_class=5,
        expected_classes=sh.EXPECTED_CLASSES,
        decode_per_class=1,
    )
    for item in validation["checks"]:
        label = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}[item["status"]]
        _say(f"  [{label}] {item['name']}: {item['detail']}")
    if not validation["ok"]:
        raise RuntimeError(
            "Canary split failed validation: "
            + ", ".join(c["name"] for c in validation["checks"] if not c["ok"])
        )

    result: Dict[str, Any] = {
        "command": "canary",
        "ok": True,
        "canary_split_dir": str(split_dir),
        "rows": meta["rows"],
        "partition_counts": meta["counts"],
        "validated": True,
        "trained": False,
    }
    if not train:
        _say("[pipeline] canary validated. Add --train to run 2 epochs on it.")
        return result

    if not confirm:
        raise PermissionError("Pass --confirm with --train to run the canary epochs.")
    if not allow_weight_download:
        from colab_helpers import pretrained_weights_available, weights_cache_path

        if not pretrained_weights_available():
            raise FileNotFoundError(
                "Canary training blocked: ImageNet ResNet-50 weights are not cached at "
                f"{weights_cache_path()} and --allow-weight-download was not given."
            )

    import colab.train as train_mod

    log_path, out_ctx, err_ctx, handle = _with_log(out, "canary")
    try:
        with out_ctx, err_ctx:
            _say(f"[pipeline] canary training ({epochs} epochs), log: {log_path}")
            trained = train_mod.run_training(
                split_dir=split_dir,
                dataset_root=data_root,
                output_dir=out,
                epochs=int(epochs),
                batch_size=int(batch_size),
                seed=int(seed),
                patience=1,
                image_size=224,
                confirm_training=True,
                smoke_test_passed=True,
                allow_weight_download=bool(allow_weight_download),
                overwrite=True,
            )
    finally:
        handle.close()

    result.update(
        trained=True,
        log_path=str(log_path),
        best_epoch=trained["best_epoch"],
        best_val_loss=trained["best_val_loss"],
        note=(
            "Canary has no test partition; it proves the pipeline, not held-out metrics. "
            "The next real run must use a separate --output-dir or --overwrite."
        ),
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m colab.pipeline",
        description="Preflight, train/resume, evaluate, or inspect the DR baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Typical sequence:\n"
            "  1) python -m colab.pipeline preflight\n"
            "  2) python -m colab.pipeline train --confirm\n"
            "  3) python -m colab.pipeline train --confirm --resume     # after a disconnect\n"
            "  4) python -m colab.pipeline evaluate --confirm\n"
        ),
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--project-root", default=None, help="project folder (else auto-discover)")
    common.add_argument("--dataset-root", default=None,
                        help=f"corpus folder (else ${sh.DATASET_ENV_VAR} or auto-discover)")
    common.add_argument("--output-dir", default=None, help="run output folder (persistent)")
    # Also accepted after the subcommand (``status --json``), where SUPPRESS keeps
    # the top-level value when the subcommand does not pass it.
    common.add_argument(
        "--json", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preflight", parents=[common], help="check everything at once")
    p.add_argument("--decode-per-stratum", type=int, default=2)
    p.add_argument("--skip-image-scan", action="store_true",
                   help="sample paths instead of checking all 14,201")
    p.add_argument("--no-smoke", action="store_true", help="skip the pytest smoke suite")

    t = sub.add_parser("train", parents=[common], help="start or resume training")
    t.add_argument("--confirm", action="store_true", help="required opt-in")
    t.add_argument("--resume", action="store_true", help="continue the run in --output-dir")
    t.add_argument("--auto-resume", action="store_true",
                   help="resume if --output-dir holds a resumable run, else start fresh")
    t.add_argument("--overwrite", action="store_true", help="discard the previous run (fresh start)")
    t.add_argument("--allow-weight-download", action="store_true",
                   help="consent to the one-time ImageNet weight download")
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--patience", type=int, default=3)
    t.add_argument("--image-size", type=int, default=224)
    t.add_argument("--seed", type=int, default=12345)
    t.add_argument("--skip-preflight", action="store_true",
                   help="skip the pre-training preflight (not recommended)")

    e = sub.add_parser("evaluate", parents=[common], help="held-out test evaluation")
    e.add_argument("--confirm", action="store_true", help="required opt-in")
    e.add_argument("--batch-size", type=int, default=32)
    e.add_argument("--image-size", type=int, default=224)

    c = sub.add_parser("canary", parents=[common], help="build/validate/train the canary split")
    c.add_argument("--train", action="store_true", help="train the canary after validating it")
    c.add_argument("--confirm", action="store_true", help="required with --train")
    c.add_argument("--allow-weight-download", action="store_true")
    c.add_argument("--epochs", type=int, default=2)
    c.add_argument("--batch-size", type=int, default=8)
    c.add_argument("--seed", type=int, default=12345)

    s = sub.add_parser("status", parents=[common], help="inspect an output directory")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)

    def emit(payload: Dict[str, Any]) -> None:
        if args.json:
            print(json.dumps(payload, indent=2, default=_json_default))

    try:
        if args.command == "preflight":
            report = preflight(
                args.project_root,
                dataset_root=args.dataset_root,
                output_dir=args.output_dir,
                check_all_images=not args.skip_image_scan,
                decode_per_stratum=args.decode_per_stratum,
                run_smoke=not args.no_smoke,
            )
            print_preflight(report)
            emit(report)
            return EXIT_OK if report["ok"] else EXIT_PREFLIGHT_FAILED

        if args.command == "status":
            state = run_state(
                Path(args.output_dir).expanduser()
                if args.output_dir
                else default_output_dir(discover_project(args.project_root))
            )
            print_status(state)
            emit(state)
            return EXIT_OK

        if args.command == "train":
            result = train_run(
                project_root=args.project_root,
                dataset_root=args.dataset_root,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
                image_size=args.image_size,
                seed=args.seed,
                allow_weight_download=args.allow_weight_download,
                resume="auto" if args.auto_resume else args.resume,
                overwrite=args.overwrite,
                confirm=args.confirm,
                skip_preflight=args.skip_preflight,
            )
            print_status(result["state"])
            emit(result)
            return EXIT_OK

        if args.command == "evaluate":
            result = evaluate_run(
                project_root=args.project_root,
                dataset_root=args.dataset_root,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                image_size=args.image_size,
                confirm=args.confirm,
            )
            print(json.dumps(result["metrics"], indent=2, default=_json_default))
            emit(result)
            return EXIT_OK

        if args.command == "canary":
            result = canary_run(
                project_root=args.project_root,
                dataset_root=args.dataset_root,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.seed,
                allow_weight_download=args.allow_weight_download,
                train=args.train,
                confirm=args.confirm,
            )
            emit(result)
            return EXIT_OK

        raise AssertionError(f"unhandled command {args.command!r}")

    except (FileNotFoundError, PermissionError, ValueError, RuntimeError) as exc:
        print("")
        print(f"[BLOCKED] {exc}")
        print("")
        return EXIT_BLOCKED
    except KeyboardInterrupt:
        print("")
        print("[INTERRUPTED] progress is checkpointed; resume with the command printed above.")
        return 130
    except Exception as exc:  # pragma: no cover - unexpected
        print("")
        print(f"[ERROR] unexpected {type(exc).__name__}: {exc}")
        print("Re-run with --json for context, or run: python -m colab.pipeline preflight")
        return EXIT_UNEXPECTED


if __name__ == "__main__":
    raise SystemExit(main())
