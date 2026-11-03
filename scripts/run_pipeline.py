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

Two launcher-only flags are handled here and never reach the orchestrator:
``--no-setup`` (used by RUN.bat) and ``--tests`` (runs the pytest suite).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = PROJECT_ROOT / "logs" / "RUN_console.log"


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


def main(argv: list[str]) -> int:
    args = [argument for argument in argv if argument != "--no-setup"]
    run_tests = "--tests" in args
    args = [argument for argument in args if argument != "--tests"]

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
