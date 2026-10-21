#!/usr/bin/env python3
"""
Local generator: produces one copy-paste-ready text file per project module so a Kaggle
notebook can run the CPU DR baseline without rewriting anything.

What it does
- copies each real source file verbatim into `colab/kaggle_notebook_docs/`, prefixed by a
  numeric cell name so the paste order is fixed and the file size is browsable before pasting;
- copies the audited split CSV byte-identical into the same folder so the `preflight` cell can
  run self-contained (Kaggle can instead mount the real path with DR_SPLIT_DIR);
- writes `kaggle_notebook_docs/README_split.md` with the cell order, sizes, and exact copy-paste
  instructions.

What it does NOT do
- modify, rewrite, summarize, or trim any real source;
- download weights, train, checkpoint, or evaluate the test partition;
- touch the audited split or label mapping in this project.

Kaggle cell concatenation behavior (expected)
- paste cells in the numeric order below;
- the `01_preflight` cell is self-contained: it reads `provisional_primary_split.csv` from the
  same folder unless DR_SPLIT_DIR is set;
- the `canary` and `train` cells still import from `colab.*` normally inside the same IPython
  session, so after pasting they behave like the laptop entry point, driven from notebook code
  instead of a CLI.
"""
from __future__ import annotations

import json
import shutil
import re as _re
from pathlib import Path
import hashlib

PROJECT = Path('.').resolve()
OUT = PROJECT / 'colab' / 'kaggle_notebook_docs'
OUT.mkdir(parents=True, exist_ok=True)

SOURCES: dict[str, str] = {
    'colab/cpu_train.py':          '01_preflight',
    'colab/colab_helpers.py':      '02_helpers',
    'colab/setup_helpers.py':      '03_setup',
    'colab/train.py':              '04_engine',
    'colab/smoke_test.py':         '05_smoke',
    'dataset/labels.py':           '06_labels',
    'dataset/decoding.py':         '07_decoding',
    'dataset/__init__.py':         '08_dataset_init',
    'dataset/split_validation.py': '09_split_valid',
}

SPLIT = PROJECT / 'provisional_primary_split' / 'provisional_primary_split.csv'
OUT_SPLIT = OUT / 'provisional_primary_split.csv'

HDR = "# cell -----------------------------------------------------------------------\n"
FOOTER = "\n# end cell -------------------------------------------------------------------\n"


def copy_text(src: Path, stem: str) -> dict:
    body = src.read_text(encoding='utf-8')
    body = _re.sub(r'\r\n', '\n', body)
    out = OUT / f'{stem}.py'
    out.write_text(HDR + f'# source: {src.relative_to(PROJECT)}\n' + body + FOOTER, encoding='utf-8')
    sha = hashlib.sha256(src.read_bytes()).hexdigest()[:12].upper()
    size_kb = round(src.stat().st_size / 1024, 2)
    return {'in': str(src.relative_to(PROJECT)).replace('\\', '/'), 'out': out.name, 'size_kb': size_kb, 'sha12': sha}


def copy_csv(src: Path) -> dict:
    out = OUT / 'provisional_primary_split.csv'
    shutil.copyfile(src, out)
    sha = hashlib.sha256(src.read_bytes()).hexdigest()[:12].upper()
    size_kb = round(src.stat().st_size / 1024, 2)
    return {'in': str(src.relative_to(PROJECT)).replace('\\', '/'), 'out': out.name, 'size_kb': size_kb, 'sha12': sha, 'audited': True}


def dimensional(path):
    raise NotImplementedError('stub removed during cleanup')


def main() -> None:
    import re as _re
    pairs: list[dict] = []
    for src_rel, stem in SOURCES.items():
        src = PROJECT / src_rel
        info = copy_text(src, stem)
        info['cell'] = stem
        pairs.append(info)
    split_info = copy_csv(SPLIT)
    split_info['cell'] = 'provisional_primary_split.csv'
    pairs.append(split_info)

    lines = []
    lines.append('# Kaggle cell concatenation for the frozen ResNet-50 DR baseline')
    lines.append('')
    lines.append('Paste these cells in the numeric order below inside one Kaggle notebook.')
    lines.append('After pasting, the `01_preflight` cell is self-contained (it can run with just')
    lines.append('the CSV in the same folder), and the canary/train cells import from `colab.*`')
    lines.append('inside the same IPython session.')
    lines.append('')
    lines.append('How to paste: copy the entire contents of each cell file and paste it into one')
    lines.append('Kaggle code cell, in order. Do not split a cell file across multiple Kaggle cells.')
    lines.append('')
    lines.append(f'| cell | file | size (KB) | sha12 | notes |')
    lines.append('| --- | --- | --- | --- | --- |')
    for p in pairs:
        note = 'audited split CSV' if p.get('audited') else ('module' if p['cell'].startswith('0') else 'asset')
        lines.append(f"| {p['cell']} | {p['out']} | {p['size_kb']} | {p['sha12']} | {note} |")
    lines.append('')
    lines.append('After pasting:')
    lines.append('')
    lines.append('Cell: preflight (offline, self-contained)')
    lines.append('```python')
    lines.append('# self-contained: reads provisional_primary_split.csv from the same folder')
    lines.append('import os, sys')
    lines.append('from pathlib import Path')
    lines.append('_here = Path().absolute()')
    lines.append('sys.path.insert(0, str(_here.parent.parent))      # repo root (colab/ + dataset/)')
    lines.append('os.environ["DR_SPLIT_DIR"] = str(_here)            # local copy of the audited split')
    lines.append('from colab.cpu_train import preflight, print_preflight')
    lines.append('r = preflight(path_check="sample")')
    lines.append('print_preflight(r)')
    lines.append('print("preflight ok:", r["ok"])')
    lines.append('```')
    lines.append('')
    lines.append('Cell: canary + train with confirmation (CPU only, no test evaluation)')
    lines.append('```python')
    lines.append('import os')
    lines.append('from pathlib import Path')
    lines.append('_here = Path().absolute()')
    lines.append('sys.path.insert(0, str(_here.parent.parent))')
    lines.append('os.environ["DR_SPLIT_DIR"] = str(_here)')
    lines.append('import colab.cpu_train as cpu')
    lines.append('parser = cpu.build_parser()')
    lines.append('# canary: 75 rows, 0 test rows, 2 epochs, confirm required')
    lines.append('cpu.main(["canary", "--train", "--confirm", "--allow-weight-download", "--output-dir", str(_here / "canary_run")])')
    lines.append('```')
    lines.append('')
    lines.append('Cell: full training (CPU only; resume with --resume on reruns)')
    lines.append('```python')
    lines.append('import os')
    lines.append('from pathlib import Path')
    lines.append('_here = Path().absolute()')
    lines.append('sys.path.insert(0, str(_here.parent.parent))')
    lines.append('os.environ["DR_SPLIT_DIR"] = str(_here)')
    lines.append('import colab.cpu_train as cpu')
    lines.append('parser = cpu.build_parser()')
    lines.append('# full: audited train/validation split, 10 epochs, confirm required')
    lines.append('cpu.main(["train", "--confirm", "--output-dir", str(_here.parent / "cpu_runs" / "full")])')
    lines.append('```')
    lines.append('')
    lines.append('Notes')
    lines.append('- `--allow-weight-download` is required only once per runtime; after the ImageNet')
    lines.append('  ResNet-50 weights are cached, remove it.')
    lines.append('- There is no `--overwrite` flag in this entry point; an existing run can only be')
    lines.append('  continued with `--resume`.')
    lines.append('- No test-partition evaluation happens here; that is a separate explicit step.')
    lines.append('')
    readme = OUT / 'README_split.md'
    readme.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    toc = {
        'generated': 'use colab/kaggle_notebook_docs/ as the source for Kaggle cells',
        'audited_split': str(SPLIT.relative_to(PROJECT)).replace('\\', '/'),
        'count': len(pairs),
        'pairs': pairs,
    }
    (OUT / 'kaggle_notebook_docs.json').write_text(json.dumps(toc, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f"wrote {len(pairs)} cell files + README_split.md")
    print(f"audited split: {SPLIT} -> {OUT_SPLIT.name}")
    for p in pairs:
        print(f"  {p['cell']:28s} -> {p['out']:34s} {p['size_kb']:7.2f}KB sha12={p['sha12']}")


if __name__ == '__main__':
    main()
