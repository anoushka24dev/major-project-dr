# cell -----------------------------------------------------------------------
# source: colab\cpu_train.py
"""Standalone CPU training entry point for the frozen ResNet-50 DR baseline.

This module deliberately does NOT import ``colab.pipeline``. It depends only on
the standard library at import time, so ``preflight`` can report a broken
environment (missing torch, missing torchvision, missing weights) instead of
dying with an import traceback. Torch and the training engine
(``colab.train``) are imported lazily, only after preflight passes and the user
has confirmed.

What it reuses (never rewrites):
- the audited split artifact ``provisional_primary_split/provisional_primary_split.csv``
  (14,201 rows; train 9,924 / validation 2,133 / test 2,144) - read-only;
- ``dataset.labels.PRIMARY_MAPPING`` five-band mapping via
  ``colab_helpers.BAND_ORDER`` and ``setup_helpers.validate_split_artifact``;
- the existing resize-and-pad + ImageNet normalization preprocessing and the
  frozen-ResNet-50-with-trainable-head model in ``colab_helpers``;
- the guarded training/checkpoint/resume engine in ``colab.train.run_training``
  (optimizer state, best-epoch bookkeeping and early-stopping counter are
  restored on resume; checkpoints are written atomically).

Guarantees (each pinned by tests/test_cpu_train.py):
1. Device is ALWAYS ``torch.device("cpu")``. CUDA is never required and never
   selected, even when available.
2. Nothing is downloaded unless ``--allow-weight-download`` is passed; the
   ImageNet ResNet-50 cache is checked first.
3. The test partition is never loaded, scored, or evaluated by any command
   here. Final test evaluation is a separate step (``colab.pipeline
   evaluate`` on Colab, or ``colab.train.run_final_evaluation``).
4. Nothing is deleted or overwritten: there is no ``--overwrite`` flag. An
   existing run can only be continued with ``--resume``.
5. Both training paths are gated: a full preflight must pass, the offline
   smoke test (``colab/smoke_test.py``) must exit 0, and ``--confirm`` must be
   given. The canary additionally builds a deterministic 75-row derived split
   (10 train / 5 validation per class, 0 test) in its own output folder.

Usage (from the project root):

    python -m colab.cpu_train preflight
    python -m colab.cpu_train canary --train --confirm --allow-weight-download
    python -m colab.cpu_train train  --confirm
    python -m colab.cpu_train train  --confirm --resume     # after a disconnect

(Equivalently ``python colab/cpu_train.py ...``; both work.)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

COLAB_DIR = Path(__file__).resolve().parent
REPO_ROOT = COLAB_DIR.parent
for _p in (COLAB_DIR, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DATASET_DIR_NAME = "Diabetic Retinopathy and Retinal Diseases Fundus I"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "cpu_runs"
CANARY_TRAIN_PER_CLASS = 10
CANARY_VAL_PER_CLASS = 5
SMOKE_TIMEOUT_SECONDS = 900

# Exit codes: 0 = ok, 1 = a check/gate/training failed, 2 = guard/usage refusal.
EXIT_OK = 0
EXIT_FAIL = 1
EXIT_REFUSED = 2


# ---------------------------------------------------------------------------
# Discovery (never raises; returns None when something is genuinely absent)
# ---------------------------------------------------------------------------

def discover_split_dir(explicit: Optional[Path] = None) -> Optional[Path]:
    """Locate the directory holding the pinned split CSV."""
    if explicit is not None:
        return Path(explicit) if Path(explicit).is_dir() else None
    env = os.environ.get("DR_SPLIT_DIR", "").strip()
    if env and Path(env).is_dir():
        return Path(env)
    for candidate in (
        REPO_ROOT / "provisional_primary_split",
        Path("/content/major_project/provisional_primary_split"),
        Path("/content/drive/MyDrive") / "major_project" / "provisional_primary_split",
    ):
        if candidate.is_dir():
            return candidate
    return None


def discover_dataset_root(explicit: Optional[Path] = None) -> Optional[Path]:
    """Locate the image corpus root (the folder containing e.g. ``11.Normal Fundus/``)."""
    if explicit is not None:
        return Path(explicit) if Path(explicit).is_dir() else None
    for env_name in ("DR_FUNDUS_ROOT", "DR_SMOKE_IMAGE_ROOT"):
        env = os.environ.get(env_name, "").strip()
        if env and Path(env).is_dir():
            return Path(env)
    for candidate in (
        Path("/content/drive/MyDrive") / DATASET_DIR_NAME,
        Path.home() / "Downloads" / DATASET_DIR_NAME,
        REPO_ROOT.parent / DATASET_DIR_NAME,
        Path("/content") / DATASET_DIR_NAME,
    ):
        if candidate.is_dir():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def _add(checks: List[Dict[str, Any]], name: str, ok: bool, detail: str,
         *, status: Optional[str] = None) -> None:
    status = status or ("pass" if ok else "fail")
    checks.append(
        {"name": name, "status": status, "ok": status in ("pass", "skip"), "detail": detail}
    )


def preflight(
    *,
    dataset_root: Optional[Path] = None,
    split_dir: Optional[Path] = None,
    allow_weight_download: bool = False,
    path_check: str = "all",
) -> Dict[str, Any]:
    """Check the complete CPU-training environment. Read-only; reports ALL failures.

    Checks, each reported individually:
    1. python, 2. torch importable, 3. torchvision importable, 4. cpu device,
    5. pretrained ResNet-50 weights cached (or download permitted),
    6. split dir + CSV, 7. the full audited split validation
    (rows/counts/classes/mapping/leakage/protocol/image paths/decoding),
    8. dataset image root.
    """
    checks: List[Dict[str, Any]] = []
    _add(checks, "python", True,
         f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} at {sys.executable}")

    torch_ok = False
    try:
        import torch

        torch_ok = True
        build = getattr(getattr(torch, "version", None), "cuda", None)
        _add(checks, "torch", True,
             f"torch {torch.__version__} (CUDA build: {'yes, ' + str(build) if build else 'no'})")
    except Exception as exc:
        _add(checks, "torch", False,
             f"{type(exc).__name__}: {exc}. Install torch first, e.g. "
             "`pip install torch` or `pip install -r colab/requirements-colab.txt` on Colab.")

    try:
        import torchvision

        _add(checks, "torchvision", True, f"torchvision {torchvision.__version__}")
    except Exception as exc:
        _add(checks, "torchvision", False,
             f"{type(exc).__name__}: {exc}. The frozen ResNet-50 needs torchvision "
             "matching the installed torch (e.g. `pip install torchvision`), or "
             "`pip install -r colab/requirements-colab.txt` on a Colab runtime.")

    if torch_ok:
        import torch

        try:
            cpu = torch.device("cpu")
            cuda_seen = bool(torch.cuda.is_available())
            _add(checks, "device_cpu", True,
                 f"device forced to {cpu}; "
                 + ("a CUDA device is visible but is intentionally NOT used"
                    if cuda_seen else "no CUDA device visible"))
        except Exception as exc:  # pragma: no cover - broken torch builds
            _add(checks, "device_cpu", False, f"{type(exc).__name__}: {exc}")
    else:
        _add(checks, "device_cpu", False, "skipped: torch is not importable")

    try:
        from colab_helpers import pretrained_weights_available, weights_cache_path

        cache = weights_cache_path()
        if pretrained_weights_available():
            _add(checks, "pretrained_weights", True, f"cached locally at {cache}")
        elif allow_weight_download:
            _add(checks, "pretrained_weights", True,
                 f"NOT cached at {cache}; download of one ImageNet resnet50 .pth "
                 "is permitted by --allow-weight-download (one-time)")
        else:
            _add(checks, "pretrained_weights", False,
                 f"ImageNet ResNet-50 weights are not cached at {cache}. Pre-seed the "
                 "cache, or pass --allow-weight-download to permit the one-time download.")
    except Exception as exc:
        _add(checks, "pretrained_weights", False, f"{type(exc).__name__}: {exc}")

    resolved_split = discover_split_dir(split_dir)
    if resolved_split is None:
        _add(checks, "split_dir", False,
             f"split directory not found (looked at DR_SPLIT_DIR, {REPO_ROOT / 'provisional_primary_split'}, "
             "/content/...). Set DR_SPLIT_DIR or pass --split-dir.")
        csv_path = None
    else:
        csv_path = resolved_split / "provisional_primary_split.csv"
        _add(checks, "split_dir", csv_path.is_file(),
             str(csv_path) if csv_path.is_file() else f"CSV missing: {csv_path}")

    resolved_root = discover_dataset_root(dataset_root)
    if resolved_root is None:
        _add(checks, "dataset_root", False,
             f"image corpus not found (looked at DR_FUNDUS_ROOT, DR_SMOKE_IMAGE_ROOT, "
             f"~/Downloads/{DATASET_DIR_NAME}, {REPO_ROOT.parent / DATASET_DIR_NAME}). "
             "Set DR_FUNDUS_ROOT or pass --dataset-root.")
    else:
        _add(checks, "dataset_root", True, str(resolved_root))

    if csv_path is not None and csv_path.is_file():
        try:
            from colab import setup_helpers as sh

            artifact = sh.validate_split_artifact(
                resolved_split,
                project_root=REPO_ROOT,
                image_root=resolved_root,
                path_check=path_check,
                decode_sample_per_stratum=1,
            )
            checks.extend(artifact["checks"])
        except Exception as exc:
            _add(checks, "split_artifact_validation", False,
                 f"{type(exc).__name__}: {exc}")

    failures = [c["name"] for c in checks if not c["ok"]]
    return {
        "command": "preflight",
        "ok": not failures and bool(checks),
        "checks": checks,
        "failures": failures,
        "paths": {
            "project_root": str(REPO_ROOT),
            "split_dir": str(resolved_split) if resolved_split else None,
            "dataset_root": str(resolved_root) if resolved_root else None,
        },
        "device": "cpu",
        "allow_weight_download": bool(allow_weight_download),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def print_preflight(report: Dict[str, Any]) -> None:
    for c in report["checks"]:
        label = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}[c["status"]]
        print(f"  [{label}] {c['name']}: {c['detail']}")
    total = len(report["checks"])
    fails = len(report["failures"])
    skips = sum(1 for c in report["checks"] if c["status"] == "skip")
    verdict = "PASSED" if report["ok"] else "FAILED"
    print(f"preflight {verdict}: {total} checks, {fails} failed, {skips} skipped")
    if report["failures"]:
        print("failing checks: " + ", ".join(report["failures"]))


# ---------------------------------------------------------------------------
# Smoke gate (the offline pytest suite; no installs, no downloads, no training)
# ---------------------------------------------------------------------------

def _smoke_command() -> List[str]:
    return [sys.executable, "-m", "pytest", str(COLAB_DIR / "smoke_test.py"), "-q"]


def run_smoke_gate(*, timeout: int = SMOKE_TIMEOUT_SECONDS) -> Dict[str, Any]:
    """Run ``colab/smoke_test.py`` and return its outcome. Never raises."""
    cmd = _smoke_command()
    try:
        proc = subprocess.run(
            cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=timeout
        )
    except FileNotFoundError as exc:
        return {"ok": False, "command": " ".join(cmd),
                "tail": [f"{type(exc).__name__}: {exc} (is pytest installed?)"]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "command": " ".join(cmd),
                "tail": [f"smoke test timed out after {timeout}s"]}
    tail = (proc.stdout + "\n" + proc.stderr).strip().splitlines()[-20:]
    return {"ok": proc.returncode == 0, "returncode": proc.returncode,
            "command": " ".join(cmd), "tail": tail}


# ---------------------------------------------------------------------------
# Shared guards for the training commands
# ---------------------------------------------------------------------------

def _assert_safe_output_dir(out: Path, split_dir: Path, dataset_root: Path) -> None:
    """Refuse an output directory that overlaps the split or the dataset."""
    o = out.expanduser().resolve()
    protected = [
        (split_dir, "split directory"),
        (dataset_root, "dataset image root"),
        (REPO_ROOT / "audit_output", "audit_output"),
    ]
    if o == REPO_ROOT.resolve():
        raise ValueError("Output directory must not be the project root; pass an explicit --output-dir.")
    for path, name in protected:
        p = Path(path).expanduser().resolve()
        if o == p or p in o.parents or o in p.parents:
            raise ValueError(
                f"Output directory {o} overlaps the {name} at {p}. Choose a different "
                "--output-dir; audited artifacts must never receive run files."
            )


_ERROR_HINTS = {
    FileExistsError: (
        "A previous run already exists at that output directory. Nothing was "
        "overwritten. Rerun with --resume to continue it, or pass a fresh --output-dir."
    ),
    FileNotFoundError: (
        "Something required is missing (weights, images, or split CSV). Run "
        "`python -m colab.cpu_train preflight` for the full report."
    ),
    PermissionError: "A confirmation guard refused the action; read the message above.",
}


def _report_training_error(exc: BaseException) -> None:
    print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
    hint = _ERROR_HINTS.get(type(exc))
    if hint:
        print(f"hint: {hint}", file=sys.stderr)


def _load_engine():
    """Import torch + colab.train lazily, with actionable errors."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover - preflight already caught this
        raise RuntimeError(
            f"torch is not importable ({type(exc).__name__}: {exc}); run preflight first."
        ) from exc
    try:
        import colab.train as train_mod  # noqa: WPS433 (lazy on purpose)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"the training engine colab.train could not be imported "
            f"({type(exc).__name__}: {exc})."
        ) from exc
    return torch, train_mod


def _gate_and_run(train_mod, torch, *, run_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Smoke gate, then run_training on an explicitly forced CPU device."""
    gate = run_smoke_gate()
    if not gate["ok"]:
        print("smoke gate FAILED (" + gate["command"] + "); tail:", file=sys.stderr)
        for line in gate["tail"]:
            print("  " + line, file=sys.stderr)
        raise PermissionError(
            "Refusing to train: `python -m pytest colab/smoke_test.py` did not exit 0."
        )
    print("smoke gate passed (colab/smoke_test.py exited 0).")
    run_kwargs["device"] = torch.device("cpu")
    run_kwargs["smoke_test_passed"] = True
    return train_mod.run_training(**run_kwargs)


def _print_run_result(result: Dict[str, Any]) -> None:
    print(
        f"training complete on {result['device']}: best epoch {result['best_epoch']} "
        f"(val loss {result['best_val_loss']:.4f})"
    )
    print(f"output dir: {result['output_dir']}")
    print(f"run summary: {result['run_summary_path']}")
    if result.get("resumed_from"):
        print(f"resumed from: {result['resumed_from']['checkpoint_dir']}")
    print("note: test partition not evaluated (that is a separate, explicit step).")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_preflight(args: argparse.Namespace) -> int:
    report = preflight(
        dataset_root=args.dataset_root,
        split_dir=args.split_dir,
        allow_weight_download=args.allow_weight_download,
        path_check=args.path_check,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_preflight(report)
    return EXIT_OK if report["ok"] else EXIT_FAIL


def build_and_validate_canary(
    split_dir: Path, out_dir: Path, dataset_root: Path
) -> Tuple[Path, Dict[str, Any]]:
    """Build the deterministic 75-row canary split and validate it.

    The canary CSV is written only under ``out_dir`` - never inside the audited
    ``provisional_primary_split/`` folder - and keeps every row's original
    partition and label. ``test_per_class=0`` means no test row is even read.
    """
    from colab import setup_helpers as sh

    canary_dir = out_dir / "canary_split"
    canary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = canary_dir / sh.SPLIT_CSV_NAME
    meta = sh.build_canary_split(
        split_dir / sh.SPLIT_CSV_NAME,
        csv_path,
        train_per_class=CANARY_TRAIN_PER_CLASS,
        val_per_class=CANARY_VAL_PER_CLASS,
        test_per_class=0,
    )
    if Path(meta["out_path"]) != csv_path or not csv_path.is_file():
        raise RuntimeError(
            f"Canary writer/reader path mismatch: wrote {meta['out_path']}, expected {csv_path}."
        )
    validation = sh.validate_canary_split(
        canary_dir,
        dataset_root=dataset_root,
        expected_train=CANARY_TRAIN_PER_CLASS * 5,
        expected_validation=CANARY_VAL_PER_CLASS * 5,
        expected_test=0,
        train_per_class=CANARY_TRAIN_PER_CLASS,
        val_per_class=CANARY_VAL_PER_CLASS,
        expected_classes=sh.EXPECTED_CLASSES,
        decode_per_class=1,
    )
    return canary_dir, validation


def cmd_canary(args: argparse.Namespace) -> int:
    if args.train and not args.confirm:
        print(
            "refused: canary training needs --confirm, e.g.\n"
            "  python -m colab.cpu_train canary --train --confirm --allow-weight-download",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    report = preflight(
        dataset_root=args.dataset_root,
        split_dir=args.split_dir,
        allow_weight_download=args.allow_weight_download,
        path_check="sample",
    )
    if not args.json:
        print_preflight(report)
    if not report["ok"] or report["paths"]["split_dir"] is None or report["paths"]["dataset_root"] is None:
        print("preflight failed; the canary cannot run.", file=sys.stderr)
        return EXIT_FAIL

    out_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT / "canary"
    split_dir = Path(report["paths"]["split_dir"])
    dataset_root = Path(report["paths"]["dataset_root"])
    _assert_safe_output_dir(out_dir, split_dir, dataset_root)

    canary_dir, validation = build_and_validate_canary(split_dir, out_dir, dataset_root)
    if not args.json:
        for item in validation["checks"]:
            label = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}[item["status"]]
            print(f"  [{label}] canary/{item['name']}: {item['detail']}")
    if not validation["ok"]:
        failed = [c["name"] for c in validation["checks"] if not c["ok"]]
        print("canary split failed validation: " + ", ".join(failed), file=sys.stderr)
        return EXIT_FAIL

    result: Dict[str, Any] = {
        "command": "canary",
        "ok": True,
        "canary_split_dir": str(canary_dir),
        "rows": validation["rows"],
        "partition_counts": validation["partition_counts"],
        "trained": False,
    }
    if not args.train:
        print(f"canary split validated ({result['rows']} rows, no test rows). "
              "Add --train --confirm to run 2 epochs on it.")
        if args.json:
            print(json.dumps(result, indent=2))
        return EXIT_OK

    torch, train_mod = _load_engine()
    try:
        trained = _gate_and_run(
            train_mod,
            torch,
            run_kwargs=dict(
                split_dir=canary_dir,
                dataset_root=dataset_root,
                output_dir=out_dir,
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                seed=int(args.seed),
                patience=1,
                image_size=224,
                confirm_training=True,
                allow_weight_download=bool(args.allow_weight_download),
                overwrite=False,
                resume=bool(args.resume),
            ),
        )
    except (PermissionError, FileNotFoundError, FileExistsError, ValueError) as exc:
        _report_training_error(exc)
        return EXIT_FAIL
    result.update(trained=True, best_epoch=trained["best_epoch"],
                  best_val_loss=trained["best_val_loss"], output_dir=str(trained["output_dir"]))
    if not args.json:
        _print_run_result(trained)
    else:
        print(json.dumps(result, indent=2))
    return EXIT_OK


def cmd_train(args: argparse.Namespace) -> int:
    if not args.confirm:
        print(
            "refused: full training needs --confirm, e.g.\n"
            "  python -m colab.cpu_train train --confirm\n"
            "(run `python -m colab.cpu_train preflight` and the canary first)",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    report = preflight(
        dataset_root=args.dataset_root,
        split_dir=args.split_dir,
        allow_weight_download=args.allow_weight_download,
        path_check="all",
    )
    if not args.json:
        print_preflight(report)
    if not report["ok"] or report["paths"]["split_dir"] is None or report["paths"]["dataset_root"] is None:
        print("preflight failed; training not started.", file=sys.stderr)
        return EXIT_FAIL

    out_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT / "full"
    split_dir = Path(report["paths"]["split_dir"])
    dataset_root = Path(report["paths"]["dataset_root"])
    try:
        _assert_safe_output_dir(out_dir, split_dir, dataset_root)
    except ValueError as exc:
        _report_training_error(exc)
        return EXIT_REFUSED

    torch, train_mod = _load_engine()
    try:
        trained = _gate_and_run(
            train_mod,
            torch,
            run_kwargs=dict(
                split_dir=split_dir,
                dataset_root=dataset_root,
                output_dir=out_dir,
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
                patience=int(args.patience),
                image_size=224,
                seed=int(args.seed),
                confirm_training=True,
                allow_weight_download=bool(args.allow_weight_download),
                overwrite=False,
                resume=bool(args.resume),
            ),
        )
    except (PermissionError, FileNotFoundError, FileExistsError, ValueError) as exc:
        _report_training_error(exc)
        return EXIT_FAIL
    if not args.json:
        _print_run_result(trained)
    else:
        print(json.dumps({
            "command": "train",
            "ok": True,
            "best_epoch": trained["best_epoch"],
            "best_val_loss": trained["best_val_loss"],
            "output_dir": str(trained["output_dir"]),
            "device": trained["device"],
        }, indent=2))
    return EXIT_OK


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, default=None,
                        help="image corpus root (default: DR_FUNDUS_ROOT / auto-discovery)")
    parser.add_argument("--split-dir", type=Path, default=None,
                        help="directory with the pinned split CSV (default: repo copy)")
    parser.add_argument("--allow-weight-download", action="store_true",
                        help="permit the one-time ImageNet ResNet-50 weight download")
    parser.add_argument("--json", action="store_true", help="print a machine-readable report")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m colab.cpu_train",
        description="Standalone CPU-only preflight/canary/training for the DR baseline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preflight", help="check torch/torchvision/weights/split/images; report all failures")
    _add_common(p)
    p.add_argument("--path-check", choices=["all", "sample", "off"], default="all",
                   help="how thoroughly to check image paths (default: all 14,201)")
    p.set_defaults(func=cmd_preflight)

    c = sub.add_parser("canary", help="build+validate the tiny derived split; optionally train it")
    _add_common(c)
    c.add_argument("--train", action="store_true", help="actually train on the 75-row canary")
    c.add_argument("--confirm", action="store_true", help="explicit opt-in required for --train")
    c.add_argument("--resume", action="store_true", help="continue an interrupted canary run")
    c.add_argument("--epochs", type=int, default=2)
    c.add_argument("--batch-size", type=int, default=8)
    c.add_argument("--seed", type=int, default=12345)
    c.add_argument("--output-dir", type=Path, default=None,
                   help="default: <repo>/cpu_runs/canary")
    c.set_defaults(func=cmd_canary)

    t = sub.add_parser("train", help="full training on the audited train/validation split")
    _add_common(t)
    t.add_argument("--confirm", action="store_true", help="explicit opt-in; required")
    t.add_argument("--resume", action="store_true", help="continue an interrupted run")
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--batch-size", type=int, default=16, help="CPU default: 16")
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--patience", type=int, default=3)
    t.add_argument("--seed", type=int, default=12345)
    t.add_argument("--output-dir", type=Path, default=None,
                   help="default: <repo>/cpu_runs/full")
    t.set_defaults(func=cmd_train)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

# end cell -------------------------------------------------------------------
