#!/usr/bin/env python
"""Dataset download manager.

Downloads everything the providers legitimately serve anonymously, verifies it,
extracts it safely, and reports precisely what a human must still supply.

Usage
-----
    python scripts/download_datasets.py                # all three sources
    python scripts/download_datasets.py --only mmrdr   # one source
    python scripts/download_datasets.py --skip-download

Exit codes
----------
0  every source that is *required for the requested stages* is usable
1  a source failed (checksum, extraction, provider error)
3  one or more sources are blocked pending manual access

Exit code 3 is deliberate: it is not a crash, it is "the pipeline cannot obtain
this data without a human", and the report says which one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ConfigError, load_config  # noqa: E402
from src.dataset_downloaders import DownloadResult  # noqa: E402
from src.dataset_downloaders.base import MANIFEST_FILENAME, DownloadManifest  # noqa: E402
from src.dataset_downloaders import idrid as idrid_mod  # noqa: E402
from src.dataset_downloaders import mendeley as mendeley_mod  # noqa: E402
from src.dataset_downloaders import mmrdr as mmrdr_mod  # noqa: E402
from src.logutil import get_logger  # noqa: E402

SOURCES = ("mendeley", "mmrdr", "idrid")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and validate the project datasets.")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--project-root", default=None, help="repository root")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=SOURCES,
        default=list(SOURCES),
        help="restrict the run to specific sources",
    )
    parser.add_argument("--fundus-root", default=None, help="pre-existing Mendeley corpus root")
    parser.add_argument("--idrid-root", default=None, help="pre-existing IDRiD root")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="inspect and validate only; never transfer bytes",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable summary on stdout")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config, root=args.project_root)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2

    logs_dir = cfg.path("paths.logs_dir", create=True)
    logger = get_logger(logs_dir / "download_datasets.log")
    manifest_path = cfg.outputs_dir("manifests") / MANIFEST_FILENAME
    manifest = DownloadManifest(manifest_path)

    results: Dict[str, DownloadResult] = {}
    fundus_root = Path(args.fundus_root).resolve() if args.fundus_root else cfg.fundus_root()

    if "mendeley" in args.only:
        try:
            results["mendeley"] = mendeley_mod.run(
                cfg, logger, manifest, fundus_root=fundus_root
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("mendeley acquisition crashed")
            results["mendeley"] = DownloadResult(
                source="mendeley", status=DownloadResult.FAILED,
                detail=f"{type(exc).__name__}: {exc}",
            )

    if "mmrdr" in args.only:
        try:
            results["mmrdr"] = mmrdr_mod.run(
                cfg, logger, manifest, skip_download=bool(args.skip_download)
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("mmrdr acquisition crashed")
            results["mmrdr"] = DownloadResult(
                source="mmrdr", status=DownloadResult.FAILED,
                detail=f"{type(exc).__name__}: {exc}",
            )

    if "idrid" in args.only:
        try:
            results["idrid"] = idrid_mod.run(
                cfg, logger, manifest,
                root=Path(args.idrid_root).resolve() if args.idrid_root else None,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("idrid validation crashed")
            results["idrid"] = DownloadResult(
                source="idrid", status=DownloadResult.FAILED,
                detail=f"{type(exc).__name__}: {exc}",
            )

    manifest.add(
        "summary",
        statuses={name: res.status for name, res in results.items()},
        usable=[name for name, res in results.items() if res.usable],
    )

    summary = {
        "sources": {name: res.as_dict() for name, res in results.items()},
        "manifest": str(manifest_path),
    }
    if args.json:
        print(json.dumps(summary, indent=1, default=str))
    else:
        print()
        print("=" * 74)
        print("DATASET ACQUISITION SUMMARY")
        print("=" * 74)
        for name, res in results.items():
            print(f"\n{name.upper():9s} status={res.status}")
            print(f"  {res.detail}")
            if res.root:
                print(f"  location: {res.root}")
            if res.requirements:
                print("  what you must do next:")
                for line in res.requirements:
                    print(f"    {line}")
        print()
        print(f"machine-readable manifest: {manifest_path}")

    if any(res.status == DownloadResult.FAILED for res in results.values()):
        return 1
    if any(res.status == DownloadResult.BLOCKED for res in results.values()):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
