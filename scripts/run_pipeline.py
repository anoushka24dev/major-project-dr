#!/usr/bin/env python
"""Console launcher used by RUN.bat.

Why this exists
---------------
Windows batch quoting is fragile once an argument contains a space (for example
``--fundus-root "C:\\My Data\\Fundus"``) and cmd silently loses exit codes when a
command is piped through PowerShell. Handling the arguments here instead means:

* RUN.bat passes its arguments through verbatim with ``%*``;
* the child process is launched with ``subprocess``/direct calls, so quoting is
  Python's problem, not batch's;
* the exit code is real and propagates to the caller;
* console output is mirrored to ``logs/RUN_console.log`` without a pipeline
  masking the status.

Three launcher-only flags are handled here and never reach the orchestrator:
``--no-setup`` (used by RUN.bat), ``--tests`` (runs the pytest suite) and ``--smoke``
(a dataset-free check of the configuration, the five-class model and TTA averaging).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = PROJECT_ROOT / "logs" / "RUN_console.log"

#: Printed by ``--help`` when the project's third-party dependencies are not
#: installed yet (a bare machine, before the first environment setup). It must stay
#: dependency-free: asking a half-installed project for its usage should never fail.
_LAUNCHER_USAGE = """\
usage: RUN.bat [options]

Modes
  (no arguments)              full pipeline: download -> prepare -> quality ->
                              train -> evaluate -> calibration -> report
  --download-only             download and verify datasets only
  --prepare-only              manifests and quality only
  --train-only                training only
  --evaluate-only             evaluation, explainability, calibration, report
  --report-only               rebuild the report from existing outputs
  --mode {combined,external,download,evaluate,full,prepare,report,train}

Dataset locations (each also sets a DR_*_ROOT environment variable)
  --fundus-root PATH          existing Mendeley corpus (13 class folders)
  --mmrdr-root PATH           existing MMRDR copy
  --aptos-root PATH           existing APTOS 2019 copy
  --ddr-root PATH             existing DDR copy
  --idrid-root PATH           existing IDRiD copy

Other options
  --datasets NAME [NAME ...]  acquire only these sources
                              (mendeley mmrdr aptos ddr idrid)
  --experiments EXP [EXP ...] E0..E6 and final_candidate, or "all"
  --cv-folds N                grouped stratified N-fold CV, then retrain
  --resume                    continue interrupted training
  --force-retrain             ignore existing checkpoints
  --skip-download             inspect datasets without transferring bytes
  --config PATH               use an alternative config.yaml
  --validate-config           validate the configuration and exit
  --json                      print the execution summary as JSON
  --limit-quality N           assess only the first N images in quality
  --no-setup                  do not run the environment setup step
  --tests                     run the pipeline test suite instead
  --smoke                     quick dataset-free check (config, model, TTA)
  -h, --help                  show this help

Exit codes: 0 success | 1 a stage failed | 2 configuration/environment
            3 data blocked pending manual dataset access
"""


class _Tee:
    """Write to the console and to the run log at the same time."""

    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            try:
                stream.write(data)
                stream.flush()
            except UnicodeEncodeError:
                encoding = getattr(stream, "encoding", None) or "utf-8"
                stream.write(data.encode(encoding, errors="replace").decode(encoding, errors="replace"))
                stream.flush()
            except Exception:
                pass
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass


def print_usage() -> int:
    """Print usage, with or without the project's third-party dependencies.

    The authoritative list comes from the orchestrator's own parser when the stack is
    installed. On a bare machine the launcher's dependency-free summary is printed, so
    ``RUN.bat --help`` never fails just because ``.venv`` has not been built yet.
    """
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.pipeline import build_parser  # needs PyYAML through src.config

        build_parser().print_help()
        return 0
    except Exception:
        print(_LAUNCHER_USAGE)
        print(
            "Note: the project dependencies are not installed yet, so this is the "
            "launcher summary rather than the parser-generated help. Run RUN.bat "
            "without --help once to create .venv and install them."
        )
        return 0


def run_smoke_checks() -> int:
    """Dataset-free smoke test: configuration, five-class model, TTA averaging.

    Nothing is downloaded, trained or written. This proves the installed code can load
    and validate the configuration, build the ResNet-50 with five outputs, run a
    forward pass, and average TTA class probabilities into a valid probability matrix.
    It is what ``RUN.bat --smoke`` reports.
    """
    print("SMOKE: configuration, model output shape and TTA probability averaging")
    print("       (no datasets, no downloads, no training, no checkpoint written)")
    failures: list[str] = []
    try:
        import numpy as np
        import torch

        sys.path.insert(0, str(PROJECT_ROOT))

        from src.config import load_config
        from src.models import build_resnet50
        from src.tta import DEFAULT_VIEWS, build_views, mean_probabilities

        cfg = load_config()
        class_names = list(cfg.classes)
        print(f"  OK   config loaded and validated: {cfg.source}")
        print(f"  OK   taxonomy: {len(class_names)} classes {class_names}")

        model = build_resnet50(
            num_classes=len(class_names), pretrained=False, allow_weight_download=False
        )
        model.eval()
        batch = torch.zeros(2, 3, 64, 64)
        with torch.no_grad():
            logits = model(batch)
        if tuple(logits.shape) != (2, len(class_names)):
            raise RuntimeError(
                f"model returned {tuple(logits.shape)}, expected (2, {len(class_names)})"
            )
        print(f"  OK   ResNet-50 forward pass: logits {tuple(logits.shape)}")

        views = build_views(DEFAULT_VIEWS, include_original=True)
        per_view = []
        with torch.no_grad():
            for name, apply_view in views:
                if name == "identity":
                    per_view.append(torch.softmax(logits, dim=1).double().numpy())
                else:
                    per_view.append(
                        torch.softmax(model(apply_view(batch)), dim=1).double().numpy()
                    )
        averaged = mean_probabilities(per_view, num_classes=len(class_names))
        if averaged.shape != (2, len(class_names)):
            raise RuntimeError(f"TTA produced {averaged.shape}, expected (2, {len(class_names)})")
        if not np.allclose(averaged.sum(axis=1), 1.0, atol=1e-6):
            raise RuntimeError("averaged probabilities do not sum to 1")
        print(f"  OK   TTA averaged {len(views)} views: {[name for name, _ in views]}")
        print(f"  OK   averaged probabilities {averaged.shape}, every row sums to 1")
    except Exception as exc:  # noqa: BLE001 - report, never pretend
        failures.append(f"{type(exc).__name__}: {exc}")
        print(f"  FAIL {failures[-1]}")
        import traceback as _traceback

        _traceback.print_exc()

    if failures:
        print("SMOKE: FAILED")
        return 1
    print("SMOKE: PASSED")
    return 0


def main(argv: list[str]) -> int:
    args = [argument for argument in argv if argument != "--no-setup"]
    run_tests = "--tests" in args
    run_smoke = "--smoke" in args
    args = [argument for argument in args if argument not in {"--tests", "--smoke"}]

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log_handle = LOG_FILE.open("a", encoding="utf-8", errors="replace")
    log_handle.write("\n" + "=" * 70 + "\n")
    log_handle.write(time.strftime("RUN started %Y-%m-%d %H:%M:%S\n"))
    log_handle.write("command: " + " ".join([sys.executable, "-m", "src.pipeline", *args]) + "\n")
    log_handle.write("=" * 70 + "\n")
    log_handle.flush()

    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(original_stdout, log_handle)
    sys.stderr = _Tee(original_stderr, log_handle)

    try:
        if any(flag in args for flag in ("-h", "--help")):
            return print_usage()

        if run_smoke:
            return run_smoke_checks()

        if run_tests:
            print("Running the test suite (python -m pytest tests -q)")
            completed = subprocess.run(
                [sys.executable, "-m", "pytest", "tests", "-q"],
                cwd=str(PROJECT_ROOT),
            )
            return completed.returncode

        sys.path.insert(0, str(PROJECT_ROOT))
        from src.pipeline import main as pipeline_main

        return int(pipeline_main(args))
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
        log_handle.write(time.strftime("RUN finished %Y-%m-%d %H:%M:%S\n"))
        log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
