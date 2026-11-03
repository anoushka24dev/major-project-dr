#!/usr/bin/env python
"""Create the project virtual environment and install compatible dependencies.

Strategy, in order
------------------
1. Check the Python interpreter: 64-bit and >= 3.9 are required by current
   PyTorch wheels.
2. Create ``.venv`` with the *same* interpreter that ran this script, unless it
   already exists or ``--no-venv`` is given.
3. Install PyTorch **separately**, choosing an index URL from detected hardware:
   an NVIDIA GPU with a sufficiently new driver gets a CUDA build, otherwise the
   CPU build. The pinned versions in the old repository requirements file are not
   reused blindly, because they are not a verified-compatible set for arbitrary
   Python versions.
4. Install the remaining dependencies from ``requirements.txt``.
5. Verify every import inside the target interpreter, re-check
   ``torch.cuda.is_available()`` (a present GPU is not proof that CUDA works), and
   write ``outputs/reports/environment.json``.

The script is idempotent: it skips work that is already done and never needs
administrator rights. Nothing is installed into the system interpreter when
``.venv`` is used.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = PROJECT_ROOT / ".venv"
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
LOG_DIR = PROJECT_ROOT / "logs"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

MIN_PYTHON = (3, 9)
#: Driver versions (Linux/Windows equivalents) below which CUDA 12.1 wheels are
#: not expected to initialise. Checked against nvidia-smi output only as a hint.
MIN_DRIVER_FOR_CU121 = 525

REQUIRED_MODULES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "pandas": "pandas",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "yaml": "PyYAML",
    "scipy": "scipy",
    "tqdm": "tqdm",
}

VERIFY_SCRIPT = r"""
import json, platform, sys
report = {"python": platform.python_version(), "executable": sys.executable,
          "platform": platform.platform(), "modules": {}}
for module in ["torch","torchvision","numpy","pandas","PIL","sklearn","matplotlib","yaml","scipy","tqdm"]:
    try:
        imported = __import__(module)
        report["modules"][module] = {"ok": True,
                                     "version": getattr(imported, "__version__", "unknown")}
    except Exception as exc:
        report["modules"][module] = {"ok": False, "error": type(exc).__name__ + ": " + str(exc)}
try:
    import torch
    report["torch_cuda_build"] = getattr(torch.version, "cuda", None)
    report["cuda_available"] = bool(torch.cuda.is_available())
    report["device_count"] = int(torch.cuda.device_count()) if report["cuda_available"] else 0
    report["device_names"] = [torch.cuda.get_device_name(i) for i in range(report["device_count"])]
except Exception as exc:
    report["cuda_available"] = False
    report["cuda_error"] = str(exc)
print("__VERIFY_JSON__" + json.dumps(report))
"""


def log(message: str = "") -> None:
    print(message, flush=True)


def _venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def detect_python() -> Tuple[bool, List[str]]:
    """Return ``(ok, problems)`` for the interpreter running this script."""
    problems: List[str] = []
    if sys.version_info < MIN_PYTHON:
        problems.append(
            f"Python {sys.version_info.major}.{sys.version_info.minor} is too old; "
            f"install Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer (64-bit)."
        )
    if sys.maxsize <= 2**32:
        problems.append("32-bit Python detected; PyTorch wheels require 64-bit Python.")
    if sys.version_info >= (3, 14):
        problems.append(
            f"Python {sys.version_info.major}.{sys.version_info.minor} is very new; not every "
            "PyTorch release publishes wheels for it. If the install fails, use Python 3.11 or 3.12."
        )
    return (not problems), problems


def detect_nvidia() -> Dict[str, Any]:
    """Query nvidia-smi if present (hardware presence only)."""
    info: Dict[str, Any] = {"present": False, "names": [], "driver": None, "driver_major": None}
    executable = shutil.which("nvidia-smi")
    if not executable:
        return info
    try:
        completed = subprocess.run(
            [executable, "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        if completed.returncode != 0:
            return info
        for line in completed.stdout.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if parts and parts[0]:
                info["names"].append(parts[0])
            if len(parts) > 1 and parts[1]:
                info["driver"] = parts[1]
                try:
                    info["driver_major"] = int(parts[1].split(".")[0])
                except ValueError:
                    pass
        info["present"] = bool(info["names"])
    except Exception:
        pass
    return info


def run_command(command: List[str], *, logger=log, label: str = "") -> Tuple[int, str]:
    """Run a subprocess, streaming output and returning ``(code, combined_output)``."""
    if label:
        logger(f"  $ {label}")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(PROJECT_ROOT),
        )
    except FileNotFoundError as exc:
        return 127, str(exc)
    lines: List[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line)
        logger(line.rstrip())
    process.wait()
    return process.returncode, "".join(lines)


def module_available(python: Path, module: str) -> bool:
    completed = subprocess.run(
        [str(python), "-c", f"import {module}"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    return completed.returncode == 0


def choose_torch_index(nvidia: Dict[str, Any]) -> Tuple[str, str]:
    """Return ``(index_url, reason)`` for the PyTorch install."""
    if not nvidia["present"]:
        return (
            "https://download.pytorch.org/whl/cpu",
            "no NVIDIA GPU detected, so the CPU build is installed",
        )
    driver = nvidia.get("driver_major")
    if driver is not None and driver < MIN_DRIVER_FOR_CU121:
        return (
            "https://download.pytorch.org/whl/cpu",
            f"NVIDIA GPU present but the driver ({nvidia['driver']}) is older than the "
            f"CUDA 12.1 wheels require (~{MIN_DRIVER_FOR_CU121}); installed the CPU build. "
            "Update the NVIDIA driver, delete .venv and re-run to get GPU support.",
        )
    return (
        "https://download.pytorch.org/whl/cu121",
        f"NVIDIA GPU detected ({', '.join(nvidia['names'])}, driver {nvidia['driver']}); "
        "installed a CUDA 12.1 build",
    )


def verify(python: Path) -> Dict[str, Any]:
    """Import-check the target interpreter and return its report."""
    completed = subprocess.run(
        [str(python), "-c", VERIFY_SCRIPT],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=300,
    )
    for line in completed.stdout.splitlines():
        if line.startswith("__VERIFY_JSON__"):
            return json.loads(line[len("__VERIFY_JSON__"):])
    return {
        "error": "verification script produced no result",
        "stdout": completed.stdout[-2000:],
        "stderr": completed.stderr[-2000:],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create .venv and install dependencies.")
    parser.add_argument("--no-venv", action="store_true",
                        help="use the current interpreter instead of creating .venv")
    parser.add_argument("--check-only", action="store_true",
                        help="only report the environment; install nothing")
    parser.add_argument("--reinstall-torch", action="store_true",
                        help="reinstall PyTorch even if it already imports")
    parser.add_argument("--cpu-only", action="store_true",
                        help="force the CPU PyTorch build even if a GPU is present")
    parser.add_argument("--quiet", action="store_true", help="less output from pip")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log("ENVIRONMENT SETUP")
    log("=" * 70)
    log(f"project root : {PROJECT_ROOT}")
    log(f"interpreter  : {sys.executable}")
    log(f"python       : {sys.version.split()[0]}")
    log(f"platform     : {platform.platform()}")

    ok, problems = detect_python()
    if problems:
        log("")
        log("PYTHON PROBLEMS DETECTED:")
        for problem in problems:
            log(f"  - {problem}")
        log("")
        log("Install a 64-bit Python 3.11 or 3.12 from https://www.python.org/downloads/")
        log("and re-run RUN.bat. The pipeline is not started on an unsupported interpreter.")
        return 2

    nvidia = detect_nvidia()
    log(f"nvidia-smi   : {'present' if nvidia['present'] else 'not found'}"
        + (f" ({', '.join(nvidia['names'])}, driver {nvidia['driver']})" if nvidia["present"] else ""))

    target_python = Path(sys.executable)
    created_venv = False
    if not args.no_venv:
        candidate = _venv_python(VENV_DIR)
        if candidate.is_file():
            log(f"virtualenv   : reusing {VENV_DIR}")
            target_python = candidate
        else:
            log(f"virtualenv   : creating {VENV_DIR}")
            if not args.check_only:
                code, _output = run_command(
                    [sys.executable, "-m", "venv", str(VENV_DIR)],
                    logger=log, label="python -m venv .venv",
                )
                if code != 0 or not candidate.is_file():
                    log("")
                    log("Could not create the virtual environment. On Debian/Ubuntu install")
                    log("python3-venv; on Windows repair the Python installation.")
                    return 2
                created_venv = True
                target_python = candidate
    else:
        log("virtualenv   : skipped (--no-venv); installing into the current interpreter")

    log(f"target python: {target_python}")

    if args.check_only:
        report = verify(target_python)
        _print_verification(report)
        return 0

    upgrade = [str(target_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"]
    quiet = ["-q"] if args.quiet else []
    run_command(upgrade + quiet, logger=log, label="pip install --upgrade pip setuptools wheel")

    # --- PyTorch, hardware-aware -----------------------------------------
    torch_ok = module_available(target_python, "torch") and module_available(target_python, "torchvision")
    if torch_ok and not args.reinstall_torch:
        log("pytorch      : already importable, leaving it as installed")
    else:
        if args.cpu_only:
            index_url, reason = "https://download.pytorch.org/whl/cpu", "forced CPU build (--cpu-only)"
        else:
            index_url, reason = choose_torch_index(nvidia)
        log(f"pytorch      : {reason}")
        log(f"pytorch      : index {index_url}")
        command = [
            str(target_python), "-m", "pip", "install", "--upgrade",
            "--index-url", index_url, "torch", "torchvision",
        ] + quiet
        code, output = run_command(command, logger=log, label="pip install torch torchvision")
        if code != 0:
            log("")
            log("CUDA/CPU wheel install failed. Retrying with the plain PyPI CPU-compatible wheels.")
            fallback = [str(target_python), "-m", "pip", "install", "--upgrade", "torch", "torchvision"] + quiet
            code, output = run_command(fallback, logger=log, label="pip install torch torchvision (PyPI)")
            if code != 0:
                log("")
                log("PyTorch could not be installed. See the pip output above.")
                return 1

    # --- remaining dependencies ------------------------------------------
    if REQUIREMENTS.is_file():
        log(f"requirements : {REQUIREMENTS.name}")
        code, output = run_command(
            [str(target_python), "-m", "pip", "install", "-r", str(REQUIREMENTS)] + quiet,
            logger=log, label="pip install -r requirements.txt",
        )
        if code != 0:
            log("Dependency installation failed. See the pip output above.")
            return 1
    else:
        log(f"requirements : {REQUIREMENTS.name} not found; skipping")

    # --- verification -----------------------------------------------------
    report = verify(target_python)
    failed = _print_verification(report)
    report.update(
        {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "venv": str(VENV_DIR) if not args.no_venv else None,
            "venv_created": created_venv,
            "nvidia": nvidia,
            "seconds": round(time.time() - started, 1),
        }
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "environment.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    if failed:
        log("")
        log("These required packages are still not importable: " + ", ".join(failed))
        return 1

    log("")
    if report.get("cuda_available"):
        log("GPU SUPPORT READY: torch.cuda.is_available() is True.")
    elif nvidia["present"]:
        log("NOTE: an NVIDIA GPU is present but PyTorch cannot use CUDA.")
        log("      Training will run on the CPU. Update the NVIDIA driver, delete .venv,")
        log("      and re-run RUN.bat to try the GPU build again.")
    else:
        log("CPU MODE: no usable GPU, so training will run on the CPU.")
        log("         This works but is slower; reduce epochs or image size if needed.")
    return 0


def _print_verification(report: Dict[str, Any]) -> List[str]:
    """Print the verification table; return names of failed modules."""
    if "error" in report:
        log(f"verification FAILED: {report['error']}")
        return list(REQUIRED_MODULES)
    failed: List[str] = []
    log("")
    log("VERIFICATION")
    for module, label in REQUIRED_MODULES.items():
        entry = (report.get("modules") or {}).get(module) or {}
        if entry.get("ok"):
            log(f"  OK    {label:14s} {entry.get('version', '')}")
        else:
            log(f"  FAIL  {label:14s} {entry.get('error', 'not importable')}")
            failed.append(label)
    log(f"  torch CUDA build : {report.get('torch_cuda_build') or 'CPU build'}")
    log(f"  CUDA available   : {report.get('cuda_available')}")
    if report.get("device_names"):
        log(f"  devices          : {', '.join(report['device_names'])}")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
