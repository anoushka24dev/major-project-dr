#!/usr/bin/env python3
"""
Generate: a Kaggle-notebook-ready copy of the project code in colab/kaggle_notebook_docs/
so you only copy-paste each block once, and the file sizes are human-browsable before
you paste them into Kaggle cells.

This generator is run LOCALLY in your project (no Kaggle dependency at all). It uses the
*real* project files — no rewriting, no summarizing, no trimming — only:
- reformatting each module into a single notebook cell block headed by its own file name,
- splitting the blocks into independent files so each one fits in one paste,
- and cloning the audited split CSV faithfully (byte-identical) as a self-contained
  `provisional_primary_split.csv` so a Kaggle session can run `preflight -p` without the
  local dataset bundle.

What is NOT produced here:
- any trained model, any checkpoint, any download of ImageNet weights, any test-partition
  evaluation.
- any modification of the audited split or label mapping in this project.

What the Kaggle cell concatenation does:
- the `preflight -p` cell is self-contained (the split file is in that cell's directory)
- the `canary` and `train` cells import from colab.* as normal inside the same IPython
  session, so after pasting the self-contained cells they behave like the laptop entry
  point, just driven from a notebook instead of a CLI.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
import os

PROJECT_DIR = Path(__file__).resolve().parent.parent  # .. from docs -> repo root    OUT_DIR = PROJECT_DIR / "colab" / "kaggle_notebook_docs"
TOC_FILE = OUT_DIR / "README_split.md"
print(f"OUT_DIR = {OUT_DIR}")
# ---------------------------------------------------------------------------
# Public, auditable numbers
# ---------------------------------------------------------------------------
SCAN_DIRS = [
    PROJECT_DIR / "colab",
    PROJECT_DIR / "colab" / "kaggle_notebook_docs",
    PROJECT_DIR / "dataset",
    PROJECT_DIR / "provisional_primary_split",
]
SCAN_PATTERNS = ["*.py", "*.ipynb", "*.md", "provisional_primary_split.csv"]
AUDITED_SPLIT = PROJECT_DIR / "provisional_primary_split" / "provisional_primary_split.csv"

DB_FILE = OUT_DIR / "kaggle_notebook_docs.json"


def scan_public_files(project_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for base in SCAN_DIRS:
        if not base.exists():
            continue
        try:
            for pat in SCAN_PATTERNS:
                for path in sorted(base.glob(pat)):
                    if path.is_file():
                        rel = path.relative_to(project_dir)
                        stat = path.stat()
                        size_kb = round(stat.st_size / 1024, 2)
                        sha = hashlib.sha256(path.read_bytes()).hexdigest()[:12].upper()
                        entry: dict[str, object] = {
                            "path": str(rel).replace("\\", "/"),
                            "size_kb": size_kb,
                            "sha12": sha,
                            "scan_root": str(base.relative_to(project_dir)).replace("\\", "/"),
                            "scanned": True,
                        }
                        if path == AUDITED_SPLIT:
                            entry["audited_split"] = True
                        rows.append(entry)
        except OSError:
            continue
    return rows


def scan_nested_dirs(project_dir: Path) -> dict:
    result: dict[str, object] = {
        "colab/kaggle_notebook_docs": [],
        "depths": {},
        "notebooks": [],
    }
    base = PROJECT_DIR / "colab" / "kaggle_notebook_docs"
    for root, dirs, files in os.walk(str(base)):
        depth = Path(root).relative_to(base).parts
        suffix = Path(root).relative_to(base).as_posix() or "."
        result["depths"][suffix] = len(depth)
        for name in files:
            p = Path(root) / name
            rel = p.relative_to(base).as_posix()
            result["colab/kaggle_notebook_docs"].append(rel)
            if name.endswith((".ipynb", ".md", ".csv")):
                result["notebooks"].append(rel)
    result["notebooks"] = sorted(set(result["notebooks"]))[:40]
    return result


def read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"\r\n", "\n", text)
    return text


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def copy_audited_split(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    public = scan_public_files(PROJECT_DIR)
    nested = scan_nested_dirs(PROJECT_DIR)    print(f"scanned {len(public)} public entries")    if not public:
        print("WARNING: no public entries found")
        print(f"SCAN_DIRS:")
        for d in SCAN_DIRS:
            print(f"  {d} exists={d.exists()} is_dir={d.is_dir()}")
            if d.is_dir():
                try:
                    print(f"    children: {[c.name for c in d.iterdir()][:20]}")
                except OSError as e:
                    print(f"    error: {e}")
        print(f"AUDITED_SPLIT: {AUDITED_SPLIT} exists={AUDITED_SPLIT.is_file()}")
        print(f"OUT_DIR: {OUT_DIR}")
        print(f"PROJECT_DIR: {PROJECT_DIR}")
print(f"TOC_FILE: {TOC_FILE}")
print(f"DB_FILE: {DB_FILE}")
print(f"AUDITED_SPLIT: {AUDITED_SPLIT} exists={AUDITED_SPLIT.is_file()}")
print(f"OUT_DIR: {OUT_DIR}")        print(f"PROJECT_DIR: {PROJECT_DIR}")
print(f"SCAN_DIRS:")
for d in SCAN_DIRS:
    print(f"  {d} exists={d.exists()} is_dir={d.is_dir()}")
    if d.is_dir():
        try:
            print(f"    children: {[c.name for c in d.iterdir()][:20]}")
        except OSError as e:
            print(f"    error iterating: {e}")

print(f"TOC_FILE: {TOC_FILE}")
print(f"DB_FILE: {DB_FILE}")
for d in SCAN_DIRS:
    print(f"SCAN_DIRS item: {d} exists={d.exists()} is_dir={d.is_dir()}")
    if d.is_dir():
        try:
            print(f"  children: {[c.name for c in d.iterdir()][:20]}")
        except OSError as e:
            print(f"  error: {e}")

    pairs: list[dict[str, str]] = []
    for entry in public:
        path = PROJECT_DIR / entry["path"]
        if not path.is_file():
            continue
        name = path.name
        stem = name.replace(".py", "").replace(".ipynb", "").replace(".md", "").replace(".csv", "")
        text = read_text(path)
        out_name = f"{stem}_cell.txt" if name.endswith(('.py', '.ipynb', '.md')) else f"{stem}_copy.csv"
        dest = OUT_DIR / out_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith(('.py', '.ipynb', '.md')):
            dest.write_text(f"# cell: {stem}\n{text}", encoding="utf-8")
        elif name == "provisional_primary_split.csv":
            copy_audited_split(path, dest)
        else:
            shutil.copyfile(path, dest)
        out_rel = str(dest.relative_to(OUT_DIR)).replace("\\", "/")
        src_rel = entry["path"].replace("\\", "/")
        pairs.append({"in": f"{src_rel}  ({entry['size_kb']} KB, sha12={entry['sha12']})", "out": out_rel})

    toc: dict[str, object] = {
        "generated": "use this directory as the source for Kaggle cells",
        "audited_split": str(AUDITED_SPLIT.relative_to(PROJECT_DIR)).replace("\\", "/"),
        "count": len(public),
        "nested": nested,
        "pairs": pairs,
    }
    write_json(DB_FILE, toc)
    print(f"toc written to {DB_FILE}")
    for line in json.dumps(toc, indent=2, ensure_ascii=False).splitlines():
        print(line)
    print("")


if __name__ == "__main__":
    main()
