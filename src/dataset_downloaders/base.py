"""Shared download machinery: resumable transfer, verification, safe extraction.

Guarantees
----------
* Interrupted transfers resume with an HTTP ``Range`` request where the server
  supports it, and restart cleanly when it does not.
* Checksums are verified when the provider publishes one.
* Archives are extracted with an explicit directory-traversal guard; no member
  may resolve outside the destination root.
* Nothing is deleted on failure; partial files are kept with a ``.part`` suffix.
* Every operation is appended to a download manifest and the log.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

USER_AGENT = "dr-trustworthy-pipeline/1.0 (research; +https://github.com/anoushka24dev/major-project-dr)"

MANIFEST_FILENAME = "download_manifest.json"


class DownloadError(RuntimeError):
    """Raised when a transfer cannot be completed or verified."""


class UnsafeArchiveError(RuntimeError):
    """Raised when an archive attempts directory traversal or absolute paths."""


@dataclass
class DownloadResult:
    """Outcome of one provider's acquisition stage."""

    source: str
    status: str
    detail: str
    root: Optional[Path] = None
    files: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    #: statuses
    OK = "ok"
    OK_LOCAL = "ok_local_copy"
    OK_ALREADY_PRESENT = "ok_already_present"
    BLOCKED = "blocked"
    FAILED = "failed"
    SKIPPED = "skipped"

    @property
    def usable(self) -> bool:
        """True when the data for this source is present and validated."""
        return self.status in {
            DownloadResult.OK,
            DownloadResult.OK_LOCAL,
            DownloadResult.OK_ALREADY_PRESENT,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "status": self.status,
            "usable": self.usable,
            "detail": self.detail,
            "root": str(self.root) if self.root else None,
            "files": list(self.files),
            "requirements": list(self.requirements),
            "metadata": self.metadata,
        }


class DownloadStatus:
    """String constants kept importable from the package root."""

    OK = DownloadResult.OK
    OK_LOCAL = DownloadResult.OK_LOCAL
    OK_ALREADY_PRESENT = DownloadResult.OK_ALREADY_PRESENT
    BLOCKED = DownloadResult.BLOCKED
    FAILED = DownloadResult.FAILED
    SKIPPED = DownloadResult.SKIPPED


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def http_json(url: str, *, timeout: float = 60.0, retries: int = 3, logger=None) -> Any:
    """GET a JSON document with retries."""
    last_error: Optional[Exception] = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # network, JSON, HTTP
            last_error = exc
            if logger:
                logger.warning("JSON request failed (attempt %d/%d) %s: %s", attempt, retries, url, exc)
            if attempt < retries:
                time.sleep(min(2 ** attempt, 10))
    raise DownloadError(f"Could not fetch JSON metadata from {url}: {last_error}")


def _sha256_of(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def download_file(
    url: str,
    destination: Path,
    *,
    expected_sha256: Optional[str] = None,
    expected_size: Optional[int] = None,
    chunk_size: int = 1024 * 1024,
    timeout: float = 120.0,
    resume: bool = True,
    max_retries: int = 5,
    backoff: float = 3.0,
    logger=None,
    progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
) -> Path:
    """Download ``url`` to ``destination``, resuming and verifying as configured.

    Returns the path to the completed file. Raises :class:`DownloadError` when the
    transfer or the verification fails.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        if expected_size is not None and destination.stat().st_size == expected_size:
            if not expected_sha256 or _sha256_of(destination) == expected_sha256.lower():
                if logger:
                    logger.info("already downloaded and verified: %s", destination.name)
                return destination
        if expected_size is None and expected_sha256 is None:
            if logger:
                logger.info("already present (unverified, no metadata): %s", destination.name)
            return destination

    part = destination.with_suffix(destination.suffix + ".part")
    last_error: Optional[Exception] = None

    for attempt in range(1, max(1, max_retries) + 1):
        existing = part.stat().st_size if part.exists() else 0
        if existing and not resume:
            part.unlink()
            existing = 0
        if expected_size is not None and existing > expected_size:
            part.unlink()
            existing = 0
        if expected_size is not None and existing == expected_size:
            break

        headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
        if existing:
            headers["Range"] = f"bytes={existing}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", response.getcode())
                if existing and status != 206:
                    # Server ignored the Range header: start over cleanly.
                    if logger:
                        logger.info("server ignored Range for %s; restarting transfer", destination.name)
                    part.unlink(missing_ok=True)
                    existing = 0
                total_header = response.headers.get("Content-Length")
                total = int(total_header) + existing if total_header else expected_size
                mode = "ab" if existing and status == 206 else "wb"
                written = existing
                with part.open(mode) as handle:
                    while True:
                        block = response.read(chunk_size)
                        if not block:
                            break
                        handle.write(block)
                        written += len(block)
                        if progress_callback:
                            progress_callback(written, total)
            if expected_size is not None and part.stat().st_size != expected_size:
                raise DownloadError(
                    f"size mismatch for {destination.name}: got {part.stat().st_size}, "
                    f"expected {expected_size}"
                )
            break
        except Exception as exc:
            last_error = exc
            if logger:
                logger.warning(
                    "transfer attempt %d/%d failed for %s: %s",
                    attempt, max_retries, destination.name, exc,
                )
            if attempt < max_retries:
                time.sleep(min(backoff * attempt, 30))
    else:
        raise DownloadError(f"Failed to download {url} after {max_retries} attempts: {last_error}")

    if expected_sha256:
        actual = _sha256_of(part)
        if actual.lower() != expected_sha256.lower():
            part.unlink(missing_ok=True)
            raise DownloadError(
                f"checksum mismatch for {destination.name}: got {actual}, expected {expected_sha256}"
            )
        if logger:
            logger.info("sha256 verified for %s", destination.name)

    os.replace(part, destination)
    return destination


# ---------------------------------------------------------------------------
# Safe extraction
# ---------------------------------------------------------------------------

def _is_within(base: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _guard_zip_members(archive: zipfile.ZipFile, destination: Path) -> None:
    for member in archive.namelist():
        name = member.replace("\\", "/")
        if name.startswith("/") or (len(name) > 1 and name[1] == ":"):
            raise UnsafeArchiveError(f"archive member has an absolute path: {member!r}")
        if ".." in Path(name).parts:
            raise UnsafeArchiveError(f"archive member attempts traversal: {member!r}")
        target = destination / name
        if not _is_within(destination, target):
            raise UnsafeArchiveError(f"archive member escapes destination: {member!r}")


def safe_extract_zip(
    archive_path: Path,
    destination: Path,
    *,
    logger=None,
    member_filter: Optional[Callable[[str], bool]] = None,
) -> Path:
    """Extract a zip after validating every member path."""
    archive_path = Path(archive_path)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            _guard_zip_members(archive, destination)
            members = archive.namelist()
            if member_filter is not None:
                members = [m for m in members if member_filter(m)]
            for member in members:
                archive.extract(member, destination)
    except zipfile.BadZipFile as exc:
        raise DownloadError(f"not a valid zip archive: {archive_path} ({exc})") from exc
    if logger:
        logger.info("extracted %s -> %s", archive_path.name, destination)
    return destination


def safe_extract_tar(archive_path: Path, destination: Path, *, logger=None) -> Path:
    """Extract a tar/tar.gz after validating every member path."""
    archive_path = Path(archive_path)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive_path) as archive:
            for member in archive.getmembers():
                name = member.name.replace("\\", "/")
                if name.startswith("/") or ".." in Path(name).parts:
                    raise UnsafeArchiveError(f"archive member attempts traversal: {member.name!r}")
                target = destination / name
                if not _is_within(destination, target):
                    raise UnsafeArchiveError(f"archive member escapes destination: {member.name!r}")
            archive.extractall(destination)
    except tarfile.TarError as exc:
        raise DownloadError(f"not a valid tar archive: {archive_path} ({exc})") from exc
    if logger:
        logger.info("extracted %s -> %s", archive_path.name, destination)
    return destination


def extract_archive(archive_path: Path, destination: Path, *, logger=None) -> Path:
    """Dispatch to the correct safe extractor by file extension."""
    name = Path(archive_path).name.lower()
    if name.endswith(".zip"):
        return safe_extract_zip(archive_path, destination, logger=logger)
    if name.endswith((".tar.gz", ".tgz", ".tar")):
        return safe_extract_tar(archive_path, destination, logger=logger)
    raise DownloadError(
        f"unsupported archive type: {archive_path.name}. "
        "Supported: .zip, .tar, .tar.gz, .tgz"
    )


def join_split_zip(
    parts: Sequence[Path],
    target: Path,
    *,
    logger=None,
    chunk_size: int = 8 * 1024 * 1024,
) -> Path:
    """Byte-concatenate ``.zip.001``-style parts into a single zip.

    Figshare publishes multi-gigabyte archives as a plain byte split; concatenating
    the parts in numeric order reconstructs the original archive exactly. The
    parts are validated to be contiguous before merging.
    """
    parts = [Path(p) for p in parts]
    if not parts:
        raise DownloadError("no archive parts supplied")
    indexes = []
    for part in parts:
        suffix = part.name.rsplit(".", 1)[-1]
        if not suffix.isdigit():
            raise DownloadError(f"unexpected archive part name: {part.name}")
        indexes.append(int(suffix))
    indexes_sorted = sorted(indexes)
    if indexes_sorted != list(range(indexes_sorted[0], indexes_sorted[0] + len(indexes_sorted))):
        raise DownloadError(f"archive parts are not contiguous: {indexes_sorted}")
    missing = [p for p in parts if not p.exists()]
    if missing:
        raise DownloadError(f"archive parts missing on disk: {[p.name for p in missing]}")

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    with partial.open("wb") as out:
        for part in sorted(parts, key=lambda p: int(p.name.rsplit(".", 1)[-1])):
            with part.open("rb") as handle:
                shutil.copyfileobj(handle, out, chunk_size)
            if logger:
                logger.info("joined %s", part.name)
    os.replace(partial, target)
    if logger:
        logger.info("reconstructed %s (%.1f GB)", target.name, target.stat().st_size / 1024**3)
    return target


# ---------------------------------------------------------------------------
# Space checks
# ---------------------------------------------------------------------------

def free_space_bytes(path: Path) -> Optional[int]:
    try:
        target = Path(path)
        while not target.exists() and target.parent != target:
            target = target.parent
        return shutil.disk_usage(str(target)).free
    except Exception:
        return None


def require_free_space(path: Path, needed_bytes: int, *, logger=None, multiplier: float = 1.1) -> None:
    """Raise a clear error when the disk cannot hold the download."""
    free = free_space_bytes(path)
    if free is None:
        return
    required = int(needed_bytes * multiplier)
    if free < required:
        raise DownloadError(
            f"not enough disk space at {path}: need ~{required / 1024**3:.1f} GB, "
            f"have {free / 1024**3:.1f} GB"
        )


# ---------------------------------------------------------------------------
# Download manifest
# ---------------------------------------------------------------------------

class DownloadManifest:
    """Append-only JSON record of every download/extraction operation."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: List[Dict[str, Any]] = []
        if self.path.exists():
            try:
                existing = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(existing, dict) and isinstance(existing.get("entries"), list):
                    self._entries = existing["entries"]
            except Exception:
                self._entries = []

    def add(self, operation: str, **fields: Any) -> Dict[str, Any]:
        entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "operation": operation}
        entry.update({k: (str(v) if isinstance(v, Path) else v) for k, v in fields.items()})
        self._entries.append(entry)
        self.flush()
        return entry

    def flush(self) -> Path:
        payload = {
            "manifest_version": 1,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "entries": self._entries,
        }
        self.path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
        return self.path

    @property
    def entries(self) -> List[Dict[str, Any]]:
        return list(self._entries)


# ---------------------------------------------------------------------------
# Tree validation helpers
# ---------------------------------------------------------------------------

def count_files(root: Path, suffixes: Iterable[str] = (".jpg", ".jpeg", ".png", ".tif", ".tiff")) -> int:
    suffixes = {s.lower() for s in suffixes}
    if not Path(root).exists():
        return 0
    total = 0
    for path in Path(root).rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            total += 1
    return total


def find_dirs_named(root: Path, name: str, *, max_depth: int = 4) -> List[Path]:
    """Find directories with an exact name, bounded in depth."""
    root = Path(root)
    if not root.exists():
        return []
    out: List[Path] = []
    stack = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        if depth > max_depth:
            continue
        try:
            for child in current.iterdir():
                if child.is_dir():
                    if child.name == name:
                        out.append(child)
                    else:
                        stack.append((child, depth + 1))
        except OSError:
            continue
    return sorted(out)
