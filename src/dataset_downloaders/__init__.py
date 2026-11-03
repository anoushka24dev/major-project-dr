"""Dataset acquisition modules.

Each provider module exposes a ``run(...)`` function returning a
:class:`src.dataset_downloaders.base.DownloadResult`. The shared machinery —
resumable HTTP transfer, checksum verification, safe archive extraction,
download manifest and logging — lives in :mod:`base`.

Access policy: automated download is only ever attempted against endpoints that
the provider serves anonymously. Registration-gated or licence-gated material is
reported, never bypassed.
"""

from .base import DownloadResult, DownloadStatus  # noqa: F401

__all__ = ["DownloadResult", "DownloadStatus"]
