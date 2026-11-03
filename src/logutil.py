"""Logging and machine-readable event recording for the pipeline.

Every stage writes a human-readable log and a JSON-lines event stream so the
final report and the exit code can be derived from recorded facts rather than
from console scraping.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

LOGGER_NAME = "drpipeline"
_LEVEL = logging.INFO


class _Utf8StreamHandler(logging.StreamHandler):
    """Console handler that never raises on a Windows code-page mismatch."""

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - OS dependent
        try:
            super().emit(record)
        except (UnicodeEncodeError, UnicodeDecodeError):
            try:
                text = self.format(record)
                encoding = getattr(self.stream, "encoding", None) or "utf-8"
                payload = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
                self.stream.write(payload + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)


def get_logger(log_file: Optional[Path] = None) -> logging.Logger:
    """Return the pipeline logger, optionally also writing to ``log_file``."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(_LEVEL)
    logger.propagate = False

    if not any(isinstance(h, _Utf8StreamHandler) for h in logger.handlers):
        console = _Utf8StreamHandler(stream=sys.stdout)
        console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(console)

    if log_file is not None:
        target = Path(log_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        resolved = str(target.resolve())
        already = any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", None) == resolved
            for h in logger.handlers
        )
        if not already:
            file_handler = logging.FileHandler(target, encoding="utf-8")
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            )
            logger.addHandler(file_handler)

    return logger


def reset_logger() -> None:
    """Detach all handlers (used by tests to keep output clean)."""
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        try:
            handler.close()
        finally:
            logger.removeHandler(handler)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class EventLog:
    """Append-only JSON-lines event log for one pipeline run."""

    def __init__(self, path: Path, logger: Optional[logging.Logger] = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logger or get_logger()

    def write(self, event: str, **fields: Any) -> Dict[str, Any]:
        record: Dict[str, Any] = {"ts": utc_now(), "event": event}
        record.update(fields)
        payload = json.dumps(record, default=str, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
        return record


class StageTimer:
    """Context manager that records a stage's duration and outcome."""

    def __init__(self, name: str, logger: logging.Logger) -> None:
        self.name = name
        self.logger = logger
        self.started = 0.0
        self.duration = 0.0
        self.error: Optional[str] = None
        self.status = "running"

    def __enter__(self) -> "StageTimer":
        self.started = time.perf_counter()
        self.logger.info("=== STAGE %s: start ===", self.name)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.duration = time.perf_counter() - self.started
        if exc is not None:
            self.status = "failed"
            self.error = f"{exc_type.__name__}: {exc}" if exc_type else str(exc)
            self.logger.error("=== STAGE %s: FAILED after %.1fs: %s ===", self.name, self.duration, self.error)
            return False
        self.logger.info("=== STAGE %s: done in %.1fs ===", self.name, self.duration)
        return False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.name,
            "status": self.status,
            "seconds": round(self.duration, 3),
            "error": self.error,
        }


@contextmanager
def stage(name: str, logger: logging.Logger, events: Optional[EventLog] = None) -> Iterator[StageTimer]:
    """Run a block as a named stage and emit start/finish events."""
    timer = StageTimer(name, logger)
    if events is not None:
        events.write("stage_start", stage=name)
    with timer:
        yield timer
    if events is not None:
        events.write(
            "stage_end",
            stage=name,
            status=timer.status,
            seconds=round(timer.duration, 3),
            error=timer.error,
        )
