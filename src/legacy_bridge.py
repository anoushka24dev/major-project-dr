"""Adapters onto the repository's pre-existing, already-tested modules.

The legacy modules live under ``colab/`` and use flat imports
(``from colab_helpers import ...``), which means they are only importable when
``colab/`` itself is on ``sys.path``. Rather than editing working, tested code,
the unified pipeline goes through this single adapter so the coupling is visible
in one place.

Nothing here reimplements legacy logic; it only locates and imports it.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import List, Optional

#: Roots whose ``colab/`` directory is already on ``sys.path``. Tracked per root
#: rather than with one global flag: callers may legitimately use different
#: project roots (tests use temporary ones), and one root's setup must never
#: suppress another's.
_PREPARED_ROOTS: set = set()


def _ensure_paths(root: Path) -> None:
    """Put ``root`` and ``root/colab`` on ``sys.path`` (idempotent per root)."""
    resolved = str(Path(root).resolve())
    if resolved in _PREPARED_ROOTS:
        return
    for entry in (str((Path(resolved) / "colab")), resolved):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    _PREPARED_ROOTS.add(resolved)


def load_gradcam(root: Path) -> ModuleType:
    """Return the legacy ``colab/gradcam`` module.

    Provides ``GradCAM``, ``resolve_cam_layer``, ``layer_facts``,
    ``render_overlay``, ``fundus_mask`` and ``attention_stats``.
    """
    _ensure_paths(root)
    return importlib.import_module("gradcam")


def load_colab_helpers(root: Path) -> ModuleType:
    """Return the legacy ``colab/colab_helpers`` module (preprocessing helpers)."""
    _ensure_paths(root)
    return importlib.import_module("colab_helpers")


def load_decoding():
    """Return the repository's ``dataset.decoding`` module."""
    return importlib.import_module("dataset.decoding")


def legacy_modules_available(root: Path) -> dict:
    """Report which legacy modules can be imported, without raising."""
    availability = {}
    for name, loader in (
        ("colab.gradcam", lambda: load_gradcam(root)),
        ("colab.colab_helpers", lambda: load_colab_helpers(root)),
        ("dataset.decoding", load_decoding),
    ):
        try:
            loader()
            availability[name] = True
        except Exception as exc:  # pragma: no cover - reported, not raised
            availability[name] = f"{type(exc).__name__}: {exc}"
    return availability
