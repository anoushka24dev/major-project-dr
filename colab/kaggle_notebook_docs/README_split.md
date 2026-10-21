# Kaggle cell concatenation for the frozen ResNet-50 DR baseline

Paste these cells in the numeric order below inside one Kaggle notebook.
After pasting, the `01_preflight` cell is self-contained (it can run with just
the CSV in the same folder), and the canary/train cells import from `colab.*`
inside the same IPython session.

Kaggle-specific notes:
- The dataset folder and split CSV are normally posted as a Kaggle Dataset; point the
  cells at that mount with `DR_SPLIT_DIR` instead of the local CSV copy if you prefer.
  The copied  CSV is only for the offline `preflight` cell when nothing is mounted.
- The engine is CPU only and refuses to train without `--confirm`. In Kaggle, paste the
  notebook code that calls `cpu.main([...])` with `--confirm` written literally in the cell.
- The entry point has no `--overwrite` flag; an existing run can only be continued with
  `--resume`.
- The CSV copy in this folder is byte-identical to the audited split and is never written back
  anywhere.
- Kaggle sessions are ephemeral: `--output-dir` should point at the notebook’s writable folder
  (e.g. `/kaggle/working/cpu_runs/full`) so checkpoints survive notebook edits.

How to paste: copy the entire contents of each cell file and paste it into one
Kaggle code cell, in order. Do not split a cell file across multiple Kaggle cells.

| cell | file | size (KB) | sha12 | notes |
| --- | --- | --- | --- | --- |
| 01_preflight | 01_preflight.py | 26.18 | B8906EA4222C | module |
| 02_helpers | 02_helpers.py | 12.6 | 491F46A9E1CC | module |
| 03_setup | 03_setup.py | 68.24 | 4416427D1E8D | module |
| 04_engine | 04_engine.py | 58.3 | DEB45A6446DC | module |
| 05_smoke | 05_smoke.py | 17.21 | B2CE47729225 | module |
| 06_labels | 06_labels.py | 3.38 | DBDDC00D94C1 | module |
| 07_decoding | 07_decoding.py | 2.14 | EAEDDBC3AEEA | module |
| 08_dataset_init | 08_dataset_init.py | 0.22 | 23AD3CB0FF44 | module |
| 09_split_valid | 09_split_valid.py | 7.1 | 674F17F54A4D | module |
| provisional_primary_split.csv | provisional_primary_split.csv | 2155.5 | E73F8AEE0482 | audited split CSV |

After pasting:

Cell: preflight (offline, self-contained)
```python
# self-contained: reads provisional_primary_split.csv from the same folder
import os, sys
from pathlib import Path
_here = Path().absolute()
sys.path.insert(0, str(_here.parent.parent))      # repo root (colab/ + dataset/)
os.environ["DR_SPLIT_DIR"] = str(_here)            # local copy of the audited split
from colab.cpu_train import preflight, print_preflight
r = preflight(path_check="sample")
print_preflight(r)
print("preflight ok:", r["ok"])
```

Cell: canary + train with confirmation (CPU only, no test evaluation)
```python
import os
from pathlib import Path
_here = Path().absolute()
sys.path.insert(0, str(_here.parent.parent))
os.environ["DR_SPLIT_DIR"] = str(_here)
import colab.cpu_train as cpu
parser = cpu.build_parser()
# canary: 75 rows, 0 test rows, 2 epochs, confirm required
cpu.main(["canary", "--train", "--confirm", "--allow-weight-download", "--output-dir", str(_here / "canary_run")])
```

Cell: full training (CPU only; resume with --resume on reruns)
```python
import os
from pathlib import Path
_here = Path().absolute()
sys.path.insert(0, str(_here.parent.parent))
os.environ["DR_SPLIT_DIR"] = str(_here)
import colab.cpu_train as cpu
parser = cpu.build_parser()
# full: audited train/validation split, 10 epochs, confirm required
cpu.main(["train", "--confirm", "--output-dir", str(_here.parent / "cpu_runs" / "full")])
```

Notes
- `--allow-weight-download` is required only once per runtime; after the ImageNet
  ResNet-50 weights are cached, remove it.
- There is no `--overwrite` flag in this entry point; an existing run can only be
  continued with `--resume`.
- No test-partition evaluation happens here; that is a separate explicit step.

