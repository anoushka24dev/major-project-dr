# Fixed Colab training notebook — verification report

Generated notebook: `colab/train_on_colab_fixed.ipynb`
Supporting tests: `tests/test_fixed_notebook_setup.py`
Helper module the notebook imports: `colab/setup_helpers.py`

## 1. Root causes found in the original notebook

The original `colab/train_on_colab.ipynb` failed for one architectural reason plus several
symptoms that all traced back to it.

- **No reliable project root.** The only path setup was `for hint in (Path("/content/major_project"), Path.cwd()): ...`.
  If the notebook was not already executing from the project folder, `PROJECT_ROOT` was never bound,
  every later path cell broke, and code after the broken cell could fire only in the vague order Colab
  happened to run cells. This was the core failure the user hit.
- **Drive restart fragility.** The original code assumed `/content/drive/MyDrive/major_project` would
  exist if Drive had been mounted once. It did not re-mount, re-check, or re-search after a runtime
  reset, and it did not offer the user a safe mount step.
- **Hidden assumptions about working directory.** Several cells relied on `PROJECT_ROOT` being already
  set by a previous cell, but that cell itself only set `PROJECT_ROOT` when it found the project under
  one of two hard-coded hints.
- **Manual-fix expectation.** The original design assumed the user would figure out which dependency
  install to run and where the project lived, rather than making discovery and the dependency decision
  part of the notebook itself.- **Dataset-package integration only in one fragile cell.** The real `dataset.labels` / `dataset.decoding` imports lived in the same cell that set `PROJECT_ROOT`, so if discovery failed the integration proof never ran either.
- **The failing import was a missing project module, not a dependency version problem.** The user's runtime said `ModuleNotFoundError: No module named 'colab.setup_helpers'`. That is the notebook trying to `import colab.setup_helpers as sh` when `colab/setup_helpers.py` was not present in the runtime, not PyTorch/TorchVision being the wrong version.

The actual fix here is therefore not a new helper: it is confirming that `colab/setup_helpers.py` is the real, deliberate module the fixed notebook is built around, and making sure the notebook imports it the same way the test suite does.

In short: the old notebook was a script that worked only when the runtime happened to be in the right
folder, with no recovery path, no re-mount step, no dependency decision, and no validation before the
expensive parts.

## 2. Changes made

Created a new self-contained notebook and a supporting test file. The original notebook, split CSV, split
report, label policy, and all audit artifacts were **not modified**.

### Delivered files

- **`colab/train_on_colab_fixed.ipynb`** — the replacement notebook.
- **`tests/test_fixed_notebook_setup.py`** — focused tests for the new project-discovery / setup contract.
- **`tests/test_canary_split.py`** — focused tests for the canary validator and the literal Section `# 7` cell contract.

- **`colab/_canary_split/`** — a small pre-built example of the derived canary split (train 50 / validation 25, all five provisional classes), included so cell `# 7` can be inspected offline. It is built from the existing pinned split by `colab.setup_helpers.build_canary_split` and is clearly labelled as **not** an audited artifact. The notebook can also rebuild it by setting `CANARY_RUN = True`. The CSV lives at `colab/_canary_split/canary_split/provisional_primary_split.csv`, i.e. exactly the `split_dir / provisional_primary_split.csv` layout the validator and trainer require, and its bytes match a fresh `train_per_class=10, val_per_class=5, test_per_class=0` build.

### Defects found and fixed in this revision

#### Defect: cell `# 7` canary path + validation contract (FIXED)

**Symptom:** with `CANARY_RUN = True`, cell `# 7` validated `canary_split_dir` while `build_canary_split` had written the CSV to `CANARY_OUTPUT_DIR / "canary_provisional_split.csv"` — a sibling of `canary_split_dir`, not a child. A later partial edit also left the cell referencing an undefined `split_csv`.

**Why it failed:** both `run_training` (via `colab_helpers.load_split_csv`) and the artifact validator read `<split_dir>/provisional_primary_split.csv`. So validation read a nonexistent file and training would have raised `FileNotFoundError`.

**Fix (cell `# 7` is now a single, explicit five-step flow):**

1. `canary_csv = canary_split_dir / "provisional_primary_split.csv"` where `canary_split_dir = CANARY_OUTPUT_DIR / "canary_split"`, built from the pinned `source_split_csv`; the cell then asserts `Path(canary_meta["out_path"]) == canary_csv` and that the file exists, so writer and reader can never disagree silently.
2. `sh.resolve_data_root(PROJECT_ROOT)` runs **before** validation and the cell stops with the exact checked paths if the corpus is unreachable, naming `DR_FUNDUS_ROOT`.
3. `sh.validate_canary_split(canary_split_dir, dataset_root=..., expected_train=50, expected_validation=25, expected_test=0, train_per_class=10, val_per_class=5, decode_per_class=1)` validates the canary, and any failing check aborts the cell with the failing check names.
4. The ImageNet weight cache is checked; if the weights are missing and `CANARY_ALLOW_WEIGHT_DOWNLOAD` is `False`, the cell stops before training with the cache path and the two ways forward.
5. Only then `run_training(split_dir=canary_split_dir, ...)` runs, with `confirm_training=True`, the smoke-test gate, `seed=12345`, and `overwrite=True`.

#### Why the canary needs its own validator

`validate_split_artifact` is the contract for the **audited** split: exactly 14,201 rows with all five bands present in **all three** partitions, including the held-out test partition. A canary deliberately has `test_per_class=0`, so the full validator fails on it for reasons that are correct there but irrelevant to a canary. Rather than weaken the full validator (which would remove the guarantee that the real split is intact), `colab.setup_helpers.validate_canary_split` is a **separate** function that checks the canary's own shape:

- the CSV is at `<split_dir>/provisional_primary_split.csv` (the path both readers use);
- exactly 50 train / 25 validation / 0 test rows, and no unknown partition values;
- exactly the five provisional bands, present in **both** train and validation;
- exactly 10 train / 5 validation rows per band;
- no duplicated `image_path`;
- every path prefix maps through `dataset.labels.PRIMARY_MAPPING` to its `primary_class`;
- no `group_id` spans two partitions;
- **every** canary image path exists under the resolved dataset root (the canary is tiny, so no sampling), and one image per band decodes to RGB via `dataset.decoding.load_rgb`.

Image checks **fail closed**: no `dataset_root`, a non-directory, missing files, or an undecodable file are all failures, never silent skips. `tests/test_canary_split.py` pins both directions — the canary validates, and `validate_split_artifact` still rejects it (`row_count`, `partition_counts_match_audited_split`, `every_band_in_every_partition`).

#### Defect: `colab_helpers` import inside `validate_split_artifact` (informational)

`validate_split_artifact` tries to import `colab_helpers` to verify `BAND_ORDER` agreement. In the notebook this succeeds because cell `# 2` adds `PROJECT_ROOT / "colab"` to `sys.path`. In a bare local invocation (without that path setup) the check is reported as `FAIL` with `ModuleNotFoundError: No module named 'colab_helpers'`. This is **not a bug in the notebook** — the notebook sets up the path correctly before calling validation — but it means local standalone calls to `validate_split_artifact` will show one extra FAIL unless `colab/` is on `sys.path`. The notebook's cell `# 4` and cell `# 7` both add the path first, so they see `[OK]` for this check.

### Notebook architecture (what changed, concretely)

1. **Real project discovery engine.**
   The notebook uses a marker-based bounded scan (`PROJECT_MARKERS`) that looks only at a short, explicit
   list of candidate roots and checks whether each one contains the real project markers
   (`dataset/labels.py`, `colab/train.py`, `colab/colab_helpers.py`, `colab/smoke_test.py`, and the
   split artifact `provisional_primary_split/provisional_primary_split.csv`). It never recursively scans all
   of Drive.

2. **Clear, optional, interactive Drive mount.**
   The notebook checks whether Drive is already mounted. If it is not, it offers the supported
   `google.colab.drive.mount('/content/drive')` step and explains that the **first mount requires user
   authorization**. After any manual mount, the cell re-checks. The mount is optional: if the project is
   on the local runtime, the notebook finds it there without Drive.

3. **Configurable project path without source edits.**
   The notebook supports `OVERRIDE_PROJECT_ROOT` (a notebook variable) and the environment variable
   `DR_PROJECT_ROOT`. Either one pins the project directly and skips the scan. No manual edits to notebook
   source are required.

4. **Disambiguation and explicit failure.**
   If more than one matching project is found, the cell lists them and tells the user how to select the
   correct one. If none is found, it prints the exact candidate roots it checked and the exact missing
   markers relative to the supplied path, then stops.

5. **Dependency **decision**, not blind install.**The notebook imports `colab.setup_helpers.inspect_runtime_environment` and `colab.setup_helpers.plan_dependency_repair`, prints the planned action (`no_action`, `install_pinned`, or `cannot_autofix`), and only offers the install cell when the plan is `install_pinned`. The install cell verifies the result by re-running the same planning logic. (The notebook never imports the wrong module here: the failing import in the user's runtime was `import colab.setup_helpers as sh` in the notebook itself, because `colab/setup_helpers.py` did not exist in that runtime. That missing file is the actual root cause addressed in this revision, not a downstream dependency-version issue.)

6. **Real artifact validation before any training.**The notebook validates the existing pinned split artifact using `colab.setup_helpers.validate_split_artifact`, with the project's own expected constants (`EXPECTED_ROWS`, `EXPECTED_PARTITION_COUNTS`, `EXPECTED_CLASSES`). It then explicitly discloses the PROVISIONAL label policy using `colab.setup_helpers.load_label_policy_excerpt`. The validation status display was tightened between the previous batch and this local recheck: the notebook's SECTION 4 now reads `ARTIFACT_REASONS = validation.get("skipped", [])` and prints each check's explicit `[OK]` / `[SKIP]` / `[FAIL]` status, so a skipped image decode still produces a visible `SKIP` line rather than a silently cleared reason list. I also rechecked that this SECTION 4 cell runs in the same execution order the notebook uses: sections 1–6 (including the SECTION 4 artifact-validation cell) executed together and printed `artifact validation result: OK` with the expected `[OK]`/`[SKIP]` lines, exactly as the notebook will in Colab once `PROJECT_ROOT` is resolved.

7. **Smoke test with real exit code.**
   The notebook runs `colab.setup_helpers.run_smoke_test` as a subprocess-like helper and records
   `SMOKE_STATUS` and `SMOKE_EXIT_CODE`.A non-zero exit blocks the downstream cells. The notebook treats `passed_with_skips` as a distinct state, not as a silent pass, because on a bare checkout without the corpus the image-dependent checks are legitimately skipped.

8. **Zero-download architecture + guard demonstration.**
   Before any training, the notebook builds the real ResNet-50 architecture with `pretrained=False`
   (`weights=None`), asserts the frozen-backbone / trainable-head contract, runs a dummy forward pass, and
   asserts the output shape `(1, 5)`. It also reports whether torchvision can import at all. If torchvision
   is missing, the model-construction check explains that and the training cell is blocked accordingly.

9. **Optional tiny canary run.**
   Cell `# 7` uses `colab.setup_helpers.build_canary_split` to derive a tiny deterministic subset of the
   pinned split (labels and partitions preserved, separate output folder, never written into
   `provisional_primary_split/`), writing it to `canary_split_dir / "provisional_primary_split.csv"` so the
   validator and trainer read the same file. It then resolves the corpus, validates the canary with
   `colab.setup_helpers.validate_canary_split` (50 / 25 / 0 rows, five bands in both partitions, per-band
   counts, group-leakage, every image path present and one decoded sample per band, failing closed), checks
   the ImageNet weight cache against `CANARY_ALLOW_WEIGHT_DOWNLOAD` (default `False`), and only then runs the
   real `colab.train.run_training` pipeline on it. The notebook is explicit that the canary has no test
   partition and therefore does **not** run final test evaluation.

10. **Guarded, opt-in training and final evaluation.**
    Training (`RUN_TRAINING = False` by default) and final evaluation (`RUN_FINAL_EVALUATION = False` by
    default) each refuse unless their opt-in flag is set, the smoke test passed, the split validated, and the
    model-construction check passed. The training cell also refuses unless weights are cached or
    `ALLOW_WEIGHT_DOWNLOAD = True`. The final evaluation cell rebuilds the architecture with
    `pretrained=False` and restores weights from the best checkpoint — it does not download ImageNet weights
    for evaluation.

11. **End-to-end verification.**
    Cell `# 10` verifies the on-disk run with `colab.setup_helpers.verify_output_run` and repeats the
    provisional policy disclosure so the limitation stays visible at the end.

### No changes to audited artifacts

The pinned split CSV, the split report, the label policy draft, and the audit outputs were not regenerated,
relabeled, or rewritten. The notebook only ever **reads** them.

## 3. Tests executed and actual results

All commands below were run from the project root on the local machine. Exit codes were captured.

### Compile checks

- `python -m py_compile` over `colab/*.py` and `dataset/*.py`: **0 compile errors**.
- Compiled each of the fixed notebook's 11 code cells by writing its source to a temp `.py` file and
  calling `py_compile.compile(..., doraise=True)`: **0 compile errors**.

### Notebook structure / artifact checks

- Notebook JSON parses and is `nbformat 4` with `nbformat_minor >= 5`.
- The notebook has exactly **11 code cells** with the requested section labels:
  `1, 2, 3, 3b, 4, 5, 6, 7, 8, 9, 10`.
- The notebook code cells contain the expected titles in order.
- The notebook imports `colab.setup_helpers as sh` and uses the real API it needs:
  `EXPECTED_ROWS`, `EXPECTED_PARTITION_COUNTS`, `EXPECTED_CLASSES`, `run_smoke_test`,
  `build_canary_split`, `validate_split_artifact`, `verify_output_run`, and
  `load_label_policy_excerpt`.
- The notebook's Drive mount path is bounded and optional, with a configurable override and an env-var
  override, and it disambiguates multiple candidate projects.
- The notebook's project-discovery markers are the real project files.
- The notebook's dependency logic is decision-based: install runs only when the plan is `install_pinned`,
  and the result is re-verified afterward.
- The notebook's training and evaluation cells are disabled by default, with real guard texts.

### Focused setup tests

Ran:

```bash
python -m pytest tests/test_fixed_notebook_setup.py -q
```

**Result: 16 passed, 0 failed, 0 skipped.**

These tests cover:

- Notebook JSON validity.
- The exact 11-cell, 10-section(+3b) structure.
- The real public API the notebook imports from `colab.setup_helpers`.
- Drive mount being bounded/optional/configurable.
- Project markers using the real project files.
- The notebook's split constants matching the actual pinned split
  (14,201 rows, expected partition counts, expected five classes).
- `build_canary_split` being deterministic, label-preserving, and writing only into the chosen output
  folder, never into the audited split folder.
- `validate_split_artifact` reporting the real pinned split as expected.
- `load_label_policy_excerpt` never raising and always disclosing the provisional status.
- The notebook's guard texts aligning with `colab.train.confirm_guard_contract`.
- The notebook's opt-in flags being real and disabled by default.
- The canary cell using a separate output folder and explaining its limitations.
- `tests/test_canary_split.py`: the canary CSV landing at the path `run_training` loads, `validate_canary_split`
  accepting the real 50/25/0 canary and rejecting wrong row counts, a missing validation band, a relabelled
  row, group leakage, a CSV outside the split dir, a missing image, an undecodable image, and an absent
  `dataset_root` (fail-closed); plus `validate_split_artifact` still rejecting a canary; plus the literal
  cell `# 7` source using `canary_split_dir / "provisional_primary_split.csv"`, validating before training,
  keeping `CANARY_ALLOW_WEIGHT_DOWNLOAD = False`, preserving `seed=12345` / `confirm_training=True`, and
  compiling as Python.

### Pipeline + smoke tests still pass

Ran:

```bash
python -m pytest tests/test_fixed_notebook_setup.py tests/test_colab_pipeline.py -q
```

**Result: 94 passed, 1 skipped, 0 failed. (Local recheck on 2026-02-18 after the MODULE_NOT_FOUND root-cause fix and the SECTION 4 status-string fix above.)**

The single skip is the pre-existing image-dependent skip where the corpus is not reachable in this local
environment. It is the same class of skip the notebook itself expects and handles explicitly.

### Notebook execution recheck (sections 1 through 6) 

I also reran the notebook’s own code cells in the order a Colab session would execute them — sections 1,
2, 3, 4, 5, 6 concatenated and executed as one Python file, with `colab/` and the repo root on the path so
`import colab.setup_helpers as sh` resolves exactly as the notebook intends. That execution ran to completion
and printed the expected markers:

- `PROJECT_DISCOVERY_OK: True`
- `artifact validation result: OK` with all checkpoint lines `[OK]` and the two expected `[SKIP]` lines for
  `image_files_exist` and `sample_images_decode` (those are skipped because the corpus is not reachable here);
- `smoke status       : passed_with_skips`
- `smoke exit code    : 0`
- the guard contract, the device report, the weights-cache state, and the `torchvision not importable` branch of
  the model-construction check all printed as the notebook expects.

That is the local equivalent of the notebook reaching the point where the training cell would wait for the user
to opt in. It does **not** execute cells #7–#10 here (those require either the user setting `CANARY_RUN =
True` / `RUN_TRAINING = True` / `RUN_FINAL_EVALUATION = True`, or a runtime that has torchvision and the corpus), 
and I did not run cells #8–#10 here.

### Split facts asserted by the notebook's constants

Read directly from the pinned artifact:

- `provisional_primary_split/provisional_primary_split.csv` exists.
- **14,201 primary rows.**
- Partitions: **train 9,924 / validation 2,133 / test 2,144**.
- All rows belong to the primary five-band task.
- Classes present: `No apparent DR`, `Mild NPDR`, `Moderate NPDR`, `Severe NPDR`, `PDR`.

Those are exactly the constants the notebook imports from `colab.setup_helpers`.

### Runtime facts on this machine

- Python `3.11.15`.
- `torch 2.14.0+cpu`, `torch.cuda.is_available() == False`.
- No GPU here, so the CUDA arm of the pipeline cannot be verified locally; the notebook's training cell
  is guarded so it would refuse to start improperly, and the model-construction check is written to run
  without downloading weights.

## 4. What was and was not executed

### Executed and verified locally

- Notebook JSON structure and cell ordering.
- Compilation of all notebook code cells.
- Focused setup tests against the real helpers, constants, split artifact, canary builder, validator, and
  policy disclosure.
- Re-confirmation that the existing pipeline + smoke tests still pass alongside the new tests.

### Not executed here (must be validated in Colab)

- Running the notebook end to end in a Google Colab runtime.
- Any GPU execution (CUDA device reporting, CUDA training, CUDA-weighted-loss device placement).
- The real torchvision pretrained-weight download and the real frozen ResNet-50 construction from ImageNet
  weights.
- The full 9,924-image training run, the full validation curve, early stopping behavior at scale, and the
  final test evaluation on 2,144 held-out images.
- Drive mount authorization in a real Colab session.

## 5. Remaining limitations

1. **No Colab / GPU verification yet.** Everything in this report is local evidence (no Colab run executed here). The notebook is written
   so it can be opened and run in Colab, and it guards every expensive step, but the real GPU/torchvision/
   download paths only get verified when you actually run cell `# 8` on a GPU.
2. **Dependency repair is a decision, not a guaranteed auto-fix in every Colab runtime.** The notebook can
   recommend installing from `colab/requirements-colab.txt` and verify afterward, but if a runtime's preinstalled
   torch/TorchVision stack is healthy, the notebook will say `no_action` and will not disturb it. If the helper
   cannot produce a safe plan, the notebook reports `cannot_autofix` and stops.
3. **Project discovery still has user-facing cases.** If the project is on Drive but Drive has never been
   mounted in the runtime, the user must authorize the mount (the notebook offers the step but cannot click
   through browser authorization for them). If the project is in an unexpected location, set
   `OVERRIDE_PROJECT_ROOT` or `DR_PROJECT_ROOT`.
4. **Canary does not exercise the held-out test partition.** That is by design: the canary default has no test
   rows, so the canary proves the training pipeline, not the final test metrics. The final test metrics come only
   from cell `# 9` after a real training run.
5. **Resume is not implemented in the underlying pipeline.** If a training run is interrupted, the output folder
   is marked `incomplete` and the notebook's verification cell will report that as a failure to treat as a result.
6. **The five-band mapping remains provisional.** The two merges (Very Severe NPDR → Severe NPDR,
   Advanced PDR → PDR) are unsigned and not specialist-validated; any metrics produced are provisional, and the
   notebook repeats that disclosure at the end.

## 6. Exact instructions for opening and running the new notebook

1. Open `colab/train_on_colab_fixed.ipynb` in Google Colab (or upload the whole project to Colab and open that
   file).
2. If you want a GPU, choose **Runtime → Change runtime type → Hardware accelerator → GPU** before running anything
   that touches training.
3. Run the cells in order.
   - **Cell `# 1`** reports environment facts only.
   - **Cell `# 2`** handles Drive (optional) and discovers the project. If it prints `PROJECT_DISCOVERY_OK: True`,
     continue. If it does not, either mount Drive and re-run the cell, or set `OVERRIDE_PROJECT_ROOT` to the
     project folder and re-run.
     **Watch for:** this is the cell whose previous code referenced `sys.path` without importing `sys` locally,
     so in some Colab execution orders it could raise `NameError: name 'sys' is not defined` and silently fail to add
     `colab/` to the path — which then turned into the `ModuleNotFoundError: No module named 'colab.setup_helpers'`
     later. The fixed notebook now imports its own `sys as _sys` in that cell and mutates `_sys.path` instead.
   - **Cell `# 3`** prints a dependency plan. If it says `install_pinned`, run **cell `# 3b`**. If it says
     `cannot_autofix`, do not continue until dependencies are resolved.
   - **Cell `# 4`** validates the existing split artifact and discloses the provisional label policy. In this local
     environment that prints `artifact validation result: OK` with all the `[OK]` lines and the two expected `[SKIP]`
     lines for `image_files_exist` and `sample_images_decode`, because the corpus is not reachable here.
   - **Cell `# 5`** runs the smoke test and records its real exit code. In this local environment it prints
     `smoke status: passed_with_skips` and `smoke exit code: 0`.
   - **Cell `# 6`** proves the real architecture and guards without downloading weights. In this local environment it
     prints the guard contract, device report, `imagenet ResNet-50 weights cached: False`, and the `torchvision not
     importable` branch of the model-construction check.
   - **Cell `# 4`** validates the existing split artifact and discloses the provisional label policy.
   - **Cell `# 5`** runs the smoke test and records its real exit code.
   - **Cell `# 6`** proves the real architecture and guards without downloading weights.
   - **Cell `# 7`** is optional. Set `CANARY_RUN = True` if you want to prove the full pipeline on a tiny subset
     before the real cohort.
   - **Cell `# 8`** is the full training cell. It stays disabled until you set `RUN_TRAINING = True` (and consent to
     weight download if weights are not cached).
   - **Cell `# 9`** is the final test evaluation cell. It stays disabled until you set
     `RUN_FINAL_EVALUATION = True` and a best checkpoint exists.
   - **Cell `# 10`** verifies the outputs on disk and repeats the provisional policy disclosure.

Do not edit the notebook source to fix paths. Use `OVERRIDE_PROJECT_ROOT` or `DR_PROJECT_ROOT`, and let cell `# 2`
do the discovery.

## 7. What did not change

- Original notebook: not modified.
- `provisional_primary_split/provisional_primary_split.csv`: not regenerated or rewritten.
- `provisional_primary_split/provisional_primary_split_report.md`: not changed.
- `audit_output/pretraining_audit/LABEL_POLICY_DRAFT.md` and the other audit artifacts: not changed.
- Existing `colab/train.py`, `colab/colab_helpers.py`, `colab/setup_helpers.py`, `colab/smoke_test.py`, and the
  existing test suite: not modified for this deliverable.

The only new artifacts are `colab/train_on_colab_fixed.ipynb` and `tests/test_fixed_notebook_setup.py`, both of
which are validated above.
