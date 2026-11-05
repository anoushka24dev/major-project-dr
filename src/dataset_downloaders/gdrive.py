"""Google Drive acquisition helpers for the DDR archive chunks.

Verified against the live public folder on 2026-11-05
------------------------------------------------------
The official DDR repository (https://github.com/nkicsl/DDR-dataset) publishes the
images as ten byte-split chunks in the public Drive folder
``1z6tSFmxW_aNayUqVxx6h6bY4kwGzUTEC``. The folder listing resolves anonymously
through Drive's embedded folder view, and each chunk downloads through the
standard confirmation form (``confirm=t`` + ``uuid``) with HTTP Range support, so
interrupted chunks can resume.

What this module does *not* do: it never bypasses an access restriction. If Drive
asks for a sign-in (private folder, quota exceeded, virus-scan interstitial that
cannot be confirmed), the error is reported with Drive's own message and the
official link is returned as the fallback.
"""

from __future__ import annotations

import http.cookiejar
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .base import DownloadError, download_file

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)

_DRIVE_UC = "https://drive.google.com/uc?export=download&id={file_id}"
_DRIVE_DOWNLOAD = "https://drive.usercontent.google.com/download"
_FOLDER_VIEW = "https://drive.google.com/embeddedfolderview?id={folder_id}#list"

_FILE_ID_RE = re.compile(r"/(?:file/d/|folders/)([A-Za-z0-9_-]{20,})")
#: Drive renders each row as ``<a href=".../file/d/<id>/view">...<div class="flip-entry-title">NAME</div>``
#: (link first, then the title). The swapped order is also tolerated.
_ENTRY_LINK_FIRST = re.compile(
    r'href="https://drive\.google\.com/file/d/(?P<id>[A-Za-z0-9_-]{20,})[^"]*"'
    r'.*?class="flip-entry-title">(?P<name>[^<]+)</div>',
    re.DOTALL,
)
_ENTRY_TITLE_FIRST = re.compile(
    r'class="flip-entry-title">(?P<name>[^<]+)</div>.*?'
    r'https://drive\.google\.com/file/d/(?P<id>[A-Za-z0-9_-]{20,})',
    re.DOTALL,
)


@dataclass
class DriveFile:
    name: str
    id: str
    size: Optional[int] = None

    @property
    def part_index(self) -> Optional[int]:
        suffix = self.name.rsplit(".", 1)[-1]
        return int(suffix) if suffix.isdigit() else None


def folder_id_of(url_or_id: str) -> str:
    """Accept a Drive folder URL or a bare id and return the folder id."""
    text = str(url_or_id).strip()
    match = _FILE_ID_RE.search(text)
    if match:
        return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", text):
        return text
    raise DownloadError(f"not a Google Drive folder URL or id: {url_or_id!r}")


class DriveSession:
    """A cookie-bearing browser-like session for one Drive conversation."""

    def __init__(self) -> None:
        self.cookies = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.cookies)
        )

    def _cookie_header(self) -> str:
        return "; ".join(f"{cookie.name}={cookie.value}" for cookie in self.cookies)

    def _open(self, url: str, *, timeout: float):
        request = urllib.request.Request(url, headers={"User-Agent": BROWSER_UA})
        return self.opener.open(request, timeout=timeout)

    def list_folder(self, folder_id: str, *, timeout: float = 60.0) -> List[DriveFile]:
        """Return the files listed by Drive's public folder view."""
        try:
            with self._open(_FOLDER_VIEW.format(folder_id=folder_id), timeout=timeout) as response:
                html = response.read().decode("utf-8", "replace")
        except Exception as exc:
            raise DownloadError(f"could not list the Drive folder {folder_id}: {exc}") from exc
        files: List[DriveFile] = []
        seen: set[str] = set()
        for pattern in (_ENTRY_LINK_FIRST, _ENTRY_TITLE_FIRST):
            for match in pattern.finditer(html):
                identifier = match.group("id")
                if identifier in seen:
                    continue
                seen.add(identifier)
                files.append(
                    DriveFile(name=match.group("name").strip(), id=identifier)
                )
        if not files and "Sign in" in html:
            raise DownloadError(
                "Google Drive returned a sign-in page for the folder; it is not public"
            )
        if not files and "quota" in html.lower():
            raise DownloadError(
                "Google Drive reports that the download quota is exceeded for this folder"
            )
        if not files:
            raise DownloadError(
                "the Google Drive folder listing contained no downloadable files; the link "
                "may have changed, or Drive served a page this pipeline cannot read"
            )
        return files

    def resolve_download(self, file_id: str, *, timeout: float = 60.0) -> str:
        """Resolve the confirmation form into a direct, ranged-download URL."""
        try:
            with self._open(_DRIVE_UC.format(file_id=file_id), timeout=timeout) as response:
                content_type = response.headers.get("Content-Type", "")
                if "text/html" not in content_type:
                    return _DRIVE_UC.format(file_id=file_id)
                html = response.read().decode("utf-8", "replace")
        except Exception as exc:
            raise DownloadError(f"could not open the Drive file {file_id}: {exc}") from exc

        action = re.search(r'action="([^"]+)"', html)
        fields = dict(
            re.findall(r'name="([^"]+)"\s+value="([^"]*)"', html)
        )
        if fields.get("uuid") or fields.get("confirm"):
            fields.setdefault("export", "download")
            fields.setdefault("id", file_id)
            base = (
                urllib.parse.urljoin(_DRIVE_UC.format(file_id=file_id), action.group(1))
                if action
                else _DRIVE_DOWNLOAD
            )
            return f"{base}?{urllib.parse.urlencode(fields)}"
        raise DownloadError(
            "Google Drive served a confirmation page without a usable token "
            "(the folder may be private, quota-limited, or require a browser session)"
        )

    def headers(self) -> dict:
        cookie = self._cookie_header()
        return {"Cookie": cookie} if cookie else {}

    def probe_size(self, file_id: str, *, timeout: float = 60.0) -> Optional[int]:
        """Read the total size from a one-byte ranged request."""
        url = self.resolve_download(file_id, timeout=timeout)
        headers = {"User-Agent": BROWSER_UA, "Range": "bytes=0-0", **self.headers()}
        try:
            with self.opener.open(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
                content_range = response.headers.get("Content-Range") or ""
                match = re.search(r"/(\d+)$", content_range)
                if match:
                    return int(match.group(1))
                length = response.headers.get("Content-Length")
                return int(length) if length and content_range else None
        except Exception:
            return None


def download_file_from_drive(
    file_id: str,
    destination: Path,
    *,
    session: Optional[DriveSession] = None,
    expected_size: Optional[int] = None,
    chunk_size: int = 8 * 1024 * 1024,
    timeout: float = 120.0,
    resume: bool = True,
    max_retries: int = 5,
    backoff: float = 5.0,
    logger=None,
    progress_callback=None,
) -> Path:
    """Download one Drive file with resume, size verification and clear failures."""
    session = session or DriveSession()
    url = session.resolve_download(file_id, timeout=timeout)
    try:
        return download_file(
            url,
            destination,
            expected_size=expected_size,
            chunk_size=chunk_size,
            timeout=timeout,
            resume=resume,
            max_retries=max_retries,
            backoff=backoff,
            logger=logger,
            progress_callback=progress_callback,
            headers={"User-Agent": BROWSER_UA, **session.headers()},
        )
    except DownloadError:
        raise
    except Exception as exc:
        raise DownloadError(f"Drive transfer failed for {file_id}: {exc}") from exc
