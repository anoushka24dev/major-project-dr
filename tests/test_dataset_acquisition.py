"""Acquisition mechanics: resume, verification, safety and honest blocking."""

from __future__ import annotations

import hashlib
import http.server
import socket
import subprocess
import sys
import threading
import urllib.parse
import zipfile
from pathlib import Path

import pytest

from src.dataset_downloaders import base, gdrive
from src.dataset_downloaders.base import (
    DownloadError,
    UnsafeArchiveError,
    download_file,
    join_split_zip,
    safe_extract_zip,
    verify_zip,
)

PAYLOAD = bytes(range(256)) * 40


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    requests = []
    payload = PAYLOAD

    def log_message(self, *args):  # keep test output clean
        pass

    def _send_bytes(self, body: bytes, content_type: str = "application/octet-stream"):
        start = 0
        status = 200
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            spec = range_header.split("=", 1)[1]
            begin = int(spec.split("-", 1)[0] or 0)
            start = begin
            status = 206
        chunk = body[start:]
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(chunk)))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
        self.end_headers()
        self.wfile.write(chunk)

    def do_GET(self):
        _Handler.requests.append((self.path, self.headers.get("Range")))
        route = urllib.parse.urlparse(self.path).path
        if route == "/file.bin":
            return self._send_bytes(_Handler.payload)
        if route == "/interstitial":
            html = (
                '<html><body><form action="/download" method="get">'
                '<input type="hidden" name="id" value="FILEID">'
                '<input type="hidden" name="export" value="download">'
                '<input type="hidden" name="confirm" value="t">'
                '<input type="hidden" name="uuid" value="abc-123">'
                "</form></body></html>"
            )
            return self._send_bytes(html.encode(), "text/html; charset=utf-8")
        if route == "/download":
            return self._send_bytes(_Handler.payload)
        if route == "/folder":
            html = (
                '<a href="https://drive.google.com/file/d/AAAAAAAAAAAAAAAAAAAAAAAAAAA/view">'
                '<div class="flip-entry-title">DDR-dataset.zip.001</div></a>'
                '<a href="https://drive.google.com/file/d/BBBBBBBBBBBBBBBBBBBBBBBBBBB/view">'
                '<div class="flip-entry-title">DDR-dataset.zip.002</div></a>'
            )
            return self._send_bytes(html.encode(), "text/html; charset=utf-8")
        if route == "/bad.zip":
            return self._send_bytes(b"this is not a zip file")
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()


@pytest.fixture()
def server():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    _Handler.requests = []
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()
    httpd.server_close()


def test_download_success_and_checksum(tmp_path, server):
    destination = tmp_path / "file.bin"
    digest = hashlib.sha256(PAYLOAD).hexdigest()
    path = download_file(f"{server}/file.bin", destination, expected_sha256=digest,
                         expected_size=len(PAYLOAD))
    assert path.read_bytes() == PAYLOAD
    assert not (tmp_path / "file.bin.part").exists()


def test_checksum_mismatch_is_a_failure(tmp_path, server):
    with pytest.raises(DownloadError, match="checksum mismatch"):
        download_file(f"{server}/file.bin", tmp_path / "file.bin",
                      expected_sha256="0" * 64)
    assert not (tmp_path / "file.bin").exists()


def test_interrupted_download_resumes_with_range(tmp_path, server):
    destination = tmp_path / "file.bin"
    part = destination.with_suffix(".bin.part")
    half = len(PAYLOAD) // 2
    part.write_bytes(PAYLOAD[:half])

    path = download_file(f"{server}/file.bin", destination, expected_size=len(PAYLOAD))
    assert path.read_bytes() == PAYLOAD
    ranges = [rng for _path, rng in _Handler.requests if rng]
    assert ranges and ranges[0] == f"bytes={half}-"


def test_already_downloaded_file_is_not_transferred_again(tmp_path, server):
    destination = tmp_path / "file.bin"
    download_file(f"{server}/file.bin", destination, expected_size=len(PAYLOAD))
    requests_before = len(_Handler.requests)
    download_file(f"{server}/file.bin", destination, expected_size=len(PAYLOAD))
    assert len(_Handler.requests) == requests_before


def test_invalid_archive_is_detected(tmp_path, server):
    archive = tmp_path / "bad.zip"
    download_file(f"{server}/bad.zip", archive)
    with pytest.raises(DownloadError, match="not a valid zip"):
        verify_zip(archive)


def test_archive_traversal_is_refused(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "x")
    with pytest.raises(UnsafeArchiveError):
        safe_extract_zip(archive, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_split_archive_join_and_contiguity(tmp_path):
    parts = []
    data = b"0123456789abcdef"
    for index, chunk in enumerate([data[:5], data[5:10], data[10:]], start=1):
        part = tmp_path / f"archive.zip.{index:03d}"
        part.write_bytes(chunk)
        parts.append(part)
    merged = join_split_zip(parts, tmp_path / "archive.zip")
    assert merged.read_bytes() == data

    with pytest.raises(DownloadError, match="not contiguous"):
        join_split_zip([parts[0], parts[2]], tmp_path / "other.zip")


def test_gdrive_folder_listing_and_ranged_download(tmp_path, server, monkeypatch):
    monkeypatch.setattr(gdrive, "_FOLDER_VIEW", server + "/folder?id={folder_id}#list")
    monkeypatch.setattr(gdrive, "_DRIVE_UC", server + "/interstitial?id={file_id}")
    session = gdrive.DriveSession()
    files = session.list_folder("FOLDERID")
    assert [item.name for item in files] == ["DDR-dataset.zip.001", "DDR-dataset.zip.002"]
    assert files[0].part_index == 1

    resolved = session.resolve_download("FILEID")
    assert "confirm=t" in resolved and "uuid=abc-123" in resolved

    target = tmp_path / "chunk.001"
    path = gdrive.download_file_from_drive("FILEID", target, session=session,
                                           expected_size=len(PAYLOAD), max_retries=1)
    assert path.read_bytes() == PAYLOAD


def test_gdrive_blocks_a_sign_in_page(monkeypatch, server):
    monkeypatch.setattr(gdrive, "_FOLDER_VIEW", server + "/file.bin?id={folder_id}")
    session = gdrive.DriveSession()
    with pytest.raises(DownloadError, match="no downloadable files"):
        session.list_folder("FOLDERID")


def _cfg(tmp_path):
    from src.config import load_config

    cfg = load_config(None, root=Path.cwd())
    cfg._data.setdefault("paths", {})["aptos_dir"] = str(tmp_path / "aptos")
    cfg._data.setdefault("paths", {})["ddr_dir"] = str(tmp_path / "ddr")
    cfg._data.setdefault("paths", {})["cache_dir"] = str(tmp_path / "raw")
    cfg._data.setdefault("paths", {})["mendeley_dir"] = str(tmp_path / "mendeley")
    cfg._data.setdefault("paths", {})["mmrdr_dir"] = str(tmp_path / "mmrdr")
    cfg._data.setdefault("paths", {})["idrid_dir"] = str(tmp_path / "idrid")
    cfg._data.setdefault("paths", {})["outputs_dir"] = str(tmp_path / "outputs")
    return cfg


def test_aptos_without_credentials_is_blocked_and_creates_nothing(tmp_path, monkeypatch):
    from src.dataset_downloaders import aptos

    monkeypatch.setattr(aptos, "_kaggle_cli_available", lambda: False)
    monkeypatch.setattr(aptos, "_kaggle_credentials_present", lambda: False)
    cfg = _cfg(tmp_path)
    manifest = base.DownloadManifest(tmp_path / "manifest.json")
    result = aptos.run(cfg, _NullLogger(), manifest, skip_download=False)
    assert result.status == "blocked"
    assert any("Kaggle" in item for item in result.requirements)
    assert not (tmp_path / "aptos").exists() or not any((tmp_path / "aptos").iterdir())


def test_mendeley_empty_listing_is_reported_not_fabricated(tmp_path, monkeypatch):
    from src.dataset_downloaders import mendeley

    monkeypatch.setattr(mendeley, "_metadata", lambda cfg, logger: {"name": "x", "size": 0})
    monkeypatch.setattr(mendeley, "_file_listing", lambda cfg, logger: [])
    cfg = _cfg(tmp_path)
    manifest = base.DownloadManifest(tmp_path / "manifest.json")
    result = mendeley.run(cfg, _NullLogger(), manifest, fundus_root=None)
    assert result.status == "blocked"
    assert result.usable is False
    assert "no downloadable files" in result.detail.lower()
    target = Path(cfg.get("paths.mendeley_dir"))
    assert not target.exists() or not any(target.rglob("*.jpg"))


def test_provenance_guard_refuses_and_writes_nothing():
    script = Path("scripts/prepare_auxiliary_datasets.py")
    before = set(Path("datasets").rglob("*")) if Path("datasets").exists() else set()
    proc = subprocess.run(
        [sys.executable, str(script), "--all"], capture_output=True, text=True
    )
    assert proc.returncode == 2
    assert "PROVENANCE BLOCKED" in (proc.stdout + proc.stderr)
    after = set(Path("datasets").rglob("*")) if Path("datasets").exists() else set()
    assert after == before


class _NullLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass
