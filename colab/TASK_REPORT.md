# Colab training pipeline — implementation and verification report

Scope: turn the existing Colab scaffold into a minimal, executable, safely guarded frozen
ResNet-50 training pipeline for the **provisional** primary five-band task.

**No model was trained. No GPU was used. No Colab runtime was launched. No package was
installed and no pretrained weights were downloaded.**

Read `colab/COLAB_EXECUTION_CHECKLIST.md` for the step-by-step Colab run and the evidence to check
at each cell.

---

## Validation pass — defects found and fixed (second session)

Every artifact was re-inspected and the suites re-run. The code paths that were previously only
statically reviewed were then *executed* locally, with the model and ImageNet weights stubbed and a
handful of real images, so the training and evaluation loops now have executed coverage here.

| # | Defect | Evidence | Fix | Covering test |
| --- | --- | --- | --- | --- |
| A | The weighted `CrossEntropyLoss` held its class weights on CPU while training could run on CUDA — a guaranteed `RuntimeError` on the first training step, which moving only the model does not fix | Reproduced locally with the `meta` device, which raises the same error class without a GPU: `RuntimeError: Tensor on device cpu is not on the expected device meta!` | `make_weighted_loss_fn(..., device=...)` moves the weight buffer onto the training device, and `run_training` passes `device=device` | `test_cpu_class_weights_fail_against_logits_on_another_device`, `test_weighted_loss_defaults_to_cpu_and_can_be_moved_by_to`, plus the device assertion in the mini run |
| B | `colab/train_on_colab.ipynb` did not compile: the skipped-smoke warning contained a string literal split across a raw newline | `compile()` of that code cell raised `SyntaxError`; Colab would have failed mid-session, after the path cells | The warning is three adjacent literals, one per line | `test_notebook_code_cells_compile` |
| C | `run_final_evaluation` rebuilt ResNet-50 through the pretrained path, so it demanded (and could have downloaded) ImageNet weights that evaluation never uses | The builder raised the cached-weights `FileNotFoundError` even though a valid best checkpoint was present | Evaluation builds the architecture with `pretrained=False` (`weights=None`) and relies on strict `load_state_dict` key matching | `test_mini_final_evaluation_scores_only_the_test_partition` (asserts `pretrained is False`), `test_build_frozen_resnet50_without_weights_never_enters_the_pretrained_path` |
| D | With `overwrite=True`, the previous run was deleted *before* the cheap validation steps, so a bad dataset or missing weights destroyed the run it was asked to replace | Read of the old control flow; the clear ran before dataset and model construction | The destructive clear is deferred until dataset, class weights, and model construction have all succeeded, and `owns_output_dir` stops a failed start from stamping `incomplete` onto a run this process never owned | `test_overwrite_does_not_destroy_a_previous_run_when_validation_fails`, `test_failed_start_does_not_relabel_a_previous_run` |
| E | `colab/requirements-colab.txt` pinned `torch==2.6.0+cu121` / `torchvision==0.21.0+cu121`, versions that exist nowhere, and named no wheel index | `pip install --dry-run torch==2.6.0+cu121` → `No matching distribution found`; the CUDA 12.1 index ends at `2.5.1+cu121` | Pinned the real CUDA 12.4 pair (`torch==2.6.0+cu124`, `torchvision==0.21.0+cu124`) and added `--extra-index-url https://download.pytorch.org/whl/cu124` to the file, since `+cu124` tags live on PyTorch's index, not PyPI | `pip install --dry-run -r colab/requirements-colab.txt` now reports `Would install numpy-2.2.1 pillow-12.2.0 torch-2.6.0+cu124 torchvision-0.21.0+cu124` |

The missing-torchvision guard was also extracted into `require_torchvision()` so both entry points
report it identically, and in `run_final_evaluation` that check now runs *after* the path and
checkpoint checks, so the more actionable error ("run training first") wins.

---

## 1. Exact files created or modified

### Created (new files)
| Path | Purpose |
| --- | --- |
| `colab/train.py` | Guarded frozen-ResNet-50 training + separate final-evaluation implementation |
| `colab/smoke_test.py` | pytest smoke suite that gates training (must exit 0) |
| `colab/TASK_REPORT.md` | This report |
| `colab/COLAB_EXECUTION_CHECKLIST.md` | Per-cell Colab run order and the evidence to check before each next step |
| `tests/test_colab_pipeline.py` | TDD test file authored for this task (see §3 note) |

### Modified (existing files, all inside `colab/`)
| Path | Change |
| --- | --- |
| `colab/colab_helpers.py` | Cleaned up; delegates image decoding to `dataset.decoding.load_rgb`; added `weights_cache_path`, `pretrained_weights_available`, `assert_head_is_trainable`, `inspect_model_structure`, `preprocessing_inference_factory`, and a padded content box on `_resize_and_pad` |
| `colab/train_on_colab.ipynb` | Added the visible `dataset/` integration cell; smoke gate now runs pytest and reads the real exit code; added the guard-contract demo cell; training and final-evaluation cells are guarded and off by default; output paths documented |
| `colab/README.md` | Updated status, cell flow, smoke-test check list, and the guard/output/interruption contract to match the implementation |
| `colab/requirements-colab.txt` | Repaired during the validation pass: the `+cu121` pins did not exist and no wheel index was given, so a plain `pip install -r` could not resolve (see defect E above) |

### Deleted
| Path | Reason |
| --- | --- |
| `colab/test.js` | Stray scratch file left by an interrupted edit; not part of the project |

### Not modified
Nothing outside `colab/` changed except the test file for this task. Verified by modification
time on 2026-10-14: `dataset/*` (2026-10-13 / 17:20), `audit_output/pretraining_audit/
PROVISIONAL_ELIGIBILITY_PRIMARY.csv` (2026-10-13 15:27), `provisional_primary_split/
provisional_primary_split.csv` and its report (2026-10-14 17:33) are all untouched by this
session, which started at 18:11. There is no git repository here, so this is a timestamp-based
check rather than a diff.

The split CSV was not regenerated, reshuffled, or rewritten. The provisional label mapping and
its documented limitations are preserved; `colab_helpers.BAND_ORDER` and the notebook both
assert agreement with `dataset.labels.PRIMARY_MAPPING`.

---

## 2. Task-by-task outcome

### Task 1 — Integrate the existing dataset package
- Inspected the real interfaces: `dataset/labels.py` (`PRIMARY_MAPPING`, `primary_task_label`,
  `is_non_dr_source`, `is_primary_eligible_source`) and `dataset/decoding.py` (`load_rgb`).
- `colab_helpers.load_rgb` is now a thin delegation to `dataset.decoding.load_rgb`, and
  `colab.train.PrimarySplitDataset` decodes every image through it — no parallel decoder.
- The notebook cell 0 **visibly** imports both modules and asserts
  `set(PRIMARY_MAPPING.values()) == set(BAND_ORDER)`, then prints both provisional merges.
- The smoke suite exercises the integration directly: import checks, label-mapping agreement,
  delegation check, and a real decode through `load_rgb`.
- No function was invented and no existing module was modified; direct integration was
  compatible, so nothing had to be worked around.

### Task 2 — Guarded frozen-ResNet-50 training pipeline
Implemented in `colab/train.py` (produced/trained from the notebook):
- `torchvision` ResNet-50 with `ResNet50_Weights.IMAGENET1K_V1`, backbone frozen, new
  `nn.Linear(2048, 5)` head; `assert_head_is_trainable` fails loudly if that contract breaks.
- The existing split CSV is loaded unchanged.
- Preprocessing is deterministic aspect-ratio-preserving resize-and-pad + ImageNet
  normalization; the content box is recorded so tests can prove no cropping occurs.
  No center crop and no random crop anywhere.
- Class weights are inverse-frequency over **training rows only**; `run_training` computes them
  from `filter_by_partition(rows, "train")` and the checkpoint records them.
- Training iterates the train loader only; validation is evaluated every epoch.
- Best checkpoint and early stopping are driven by a single rule, `update_best_tracking`, which
  takes validation loss only and cannot see test data by construction.
- `run_final_evaluation` is a separate, explicitly invoked function; train/val/test overlap is
  asserted before it runs.
- Reported metrics: accuracy, macro precision, macro recall, macro F1, per-class
  precision/recall/F1/support, and a square integer confusion matrix, keyed by the accurate
  provisional band names, plus a provisional-status note.

### Task 3 — Persistent outputs and interruptions
- Output directory is user-configurable; the notebook defaults it to
  `/content/drive/MyDrive/dr_baseline_outputs` when Drive is mounted.
- Files written: `run_meta.json` (full config incl. class weights, device and runtime report),
  `run_manifest.json` (status `in_progress` → `completed`, or `incomplete`),
  `run_summary.json` (per-epoch train/val history + final validation metrics),
  `checkpoints/epoch_XXXX/{model.pt,training_state.json}` (with optimizer state),
  `best_checkpoint/`, and `final_test_evaluation/final_test_metrics.json`.
- Overwrite behaviour is explicit: a run refuses to start if the directory already holds a run;
  `overwrite=True` is required to replace it deliberately.
- Interruption behaviour is explicit: **resume is not implemented**, the run is marked
  `incomplete` and stamped with `TRAINING_INCOMPLETE.txt`, and must be restarted in a fresh
  directory. Per-epoch checkpoints and optimizer state are still saved. `run_summary.json` is
  written only on genuine completion, so an incomplete run cannot masquerade as a completed one.
- Checkpoint compatibility is validated before use: `checkpoint_compat_report` rejects legacy
  metadata, a different model, or a head size that does not match the current band order, and
  flags missing weights.

### Task 4 — Safety guards
- Training is disabled by default: `run_training` raises `PermissionError` unless
  `confirm_training=True`, and again unless `smoke_test_passed=True`.
- The notebook gate is real: it runs `python -m pytest colab/smoke_test.py -q` and requires exit
  code 0. The previous version invoked the module directly, which would have exited 0 without
  running a single test — that was a genuine hole and it is fixed.
- Final test evaluation is disabled by default and requires `confirm_final_evaluation=True`.
- Clear failures were added for: missing split dir, missing dataset root, empty train/val
  partitions, test-row leakage, missing torchvision, uncached pretrained weights, unknown bands,
  a missing or incompatible checkpoint, and a pre-existing output directory.
- No automatic installs or downloads: weights must already be cached, or
  `allow_weight_download=True` must be passed explicitly; dependency installation only happens if
  you run the optional install cell.
- Device reporting reflects reality: `device_report()` reports CUDA only when
  `torch.cuda.is_available()`, and includes the real device name when CUDA exists.
- Verified by execution: with a passing smoke test and the opt-in flags off, both guarded
  notebook cells raise `SystemExit(0)` before importing the training entry points.

### Task 5 — TDD and verification
Covered in `tests/test_colab_pipeline.py` (65 tests) and `colab/smoke_test.py` (24 tests):
partition filtering and coverage, class weights from training labels only (including an
invariance check against val/test counts), the training path never touching test rows,
validation-based checkpoint selection and the early-stopping contract, guards preventing
accidental execution, output collision/overwrite protection, incomplete-run marking, checkpoint
metadata and compatibility checks, resize-and-pad behaviour, and label-mapping integration.

---

## 3. Tests run and actual results

All commands were run from the project root on CPU. `EXIT=` is the real pytest exit status.

| Command | Result |
| --- | --- |
| `DR_SMOKE_IMAGE_ROOT="…/Diabetic Retinopathy and Retinal Diseases Fundus I" python -m pytest colab/smoke_test.py -q` | **24 passed** (exit 0), 9.4 s — no skips; the real decode and dataset-item checks ran |
| `DR_SMOKE_IMAGE_ROOT="…" python -m pytest tests/test_colab_pipeline.py -q` | **78 passed, 1 skipped** (exit 0), 24.8 s |
| `DR_SMOKE_IMAGE_ROOT="…" python -m pytest tests/ -q` | **173 passed, 1 skipped** (exit 0), 152.1 s |
| `DR_SMOKE_IMAGE_ROOT="…" python -m pytest tests/ colab/smoke_test.py -q` | **197 passed, 1 skipped** (exit 0), 141.7 s — 0 failures. The single skip is the "weights not cached" check, skipped because torchvision is absent here so the torchvision precondition fires first (see §5) |
| `pip install --dry-run -r colab/requirements-colab.txt` | `Would install numpy-2.2.1 pillow-12.2.0 torch-2.6.0+cu124 torchvision-0.21.0+cu124` — the repaired pin set resolves |
| `pip install --dry-run torch==2.6.0+cu121` (the old pin) | `No matching distribution found` — the defect that was fixed |
| Notebook guard execution probe (both guarded cells, opt-in off) | Both stopped with `SystemExit(0)`; the entry-point imports sit after the guard |

Exit codes above were captured with `set -o pipefail`, so they are pytest's status rather than the
status of the output filter.

Earlier in the session the same command timed out at 120 s and 180 s. The cause was the new test
file iterating and decoding a full partition (9,924 images) several times and doing an O(n²)
CSV lookup per row — roughly 141M comparisons. That was fixed by sampling a few rows for the
image-dependent checks and indexing the CSV once, not by deleting assertions. The suite now runs
in under 7 s and the whole repository suite in under a minute.

---

## 4. Does the full existing test suite pass?

**Yes.** `python -m pytest tests/ colab/smoke_test.py -q` → **197 passed, 0 failed, 1 skipped**
(exit code 0, via `pipefail`) with the image corpus reachable — 95 pre-existing tests in `tests/`
plus the 78 pipeline tests and the 24 smoke tests. Without a corpus reachable, the image-dependent
checks report themselves as skipped rather than passing silently.

Note on file scope: `tests/test_colab_pipeline.py` is a **new file authored for this task**, not a
pre-existing project file. I rewrote it in this pass because the earlier draft contained dead code
(`if False else`), a vacuous conditional assertion, and the O(n²) slowdown above. Every other file
outside `colab/` is unchanged.

---

## 5. What is executable vs. still incomplete

### Executable and verified locally
- Split loading, partition filtering, and the pairwise-disjointness/coverage invariants
- Dataset construction and item decoding through `dataset.decoding.load_rgb`, with ImageNet
  normalization and correct label indexing
- Resize-and-pad preprocessing: exact content box, determinism, no cropping
- Class-weight computation from training rows only, including refusal on a missing band
- `train_epoch` / `evaluate_epoch` — exercised for real in tests with a tiny model and synthetic
  batches (real forward and backward pass)
- **The full `run_training` loop end to end** on real images: two epochs over a five-row training
  slice and a three-row validation slice, with the real dataset, preprocessing, weighting, loop,
  checkpointing, and manifest bookkeeping, and only the ImageNet weights/model stubbed. Asserts
  that nothing from the test partition is decoded, that the class weights equal the training-only
  computation, that the weighted loss was placed on the training device, that the saved best
  checkpoint is the epoch selection picked, and that the manifest ends at `completed`
- **The full `run_final_evaluation` path** against a real on-disk best checkpoint of the stub
  architecture: metrics and `final_test_metrics.json` are produced, `num_samples` matches the test
  slice, every decoded path is a test path, and the architecture was rebuilt with `pretrained=False`
- Metrics: accuracy, macro precision/recall/F1, per-class metrics with support, square integer
  confusion matrix, JSON-serializable
- Validation-based selection rule and its early-stopping behaviour
- Checkpoint save (epoch + best), metadata shape, and compatibility rejection paths
- Output-collision protection, run manifest statuses, incomplete-run marking, completed-run
  summary that clears the marker
- All training/evaluation guards
- Notebook structural validity, guard placement, and file references

### Implemented but not executed here (needs a real Colab run)
- `run_training` over all 9,924 training images at 224×224 for a full schedule, and the real
  validation curve, early stopping, and best-checkpoint selection that come out of it
- `run_final_evaluation` over all 2,144 held-out images using weights from a real training run
- `build_frozen_resnet50` with actual ImageNet weights — **torchvision itself is not installed in
  this local environment**, so the real `models.resnet50` call has never run here. What *was*
  executed is the helper's own logic against a stand-in `torchvision` module: `pretrained=False`
  passes `weights=None`, the absent-weights path raises before the constructor and therefore
  cannot download, and `allow_download=True` with cached weights is the only route to the
  ImageNet enum. `colab.train` and `colab_helpers` import cleanly here, and the entry points fail
  with an explicit RuntimeError naming `colab/requirements-colab.txt` rather than crashing obscurely.
- Any CUDA execution: this machine reports `torch.cuda.is_available() == False`, so only the CPU
  device path has been exercised. Defect A was reproduced with the `meta` device as the stand-in.

### Explicitly not done
- No training run, no GPU use, no Colab runtime, no dependency installation
- No resume implementation (documented as restart-required instead)

---

## 6. Locally tested vs. requires Colab validation

**Tested locally (CPU, this machine):** everything listed as "executable and verified locally"
above, including real image decoding from the local corpus, the two-epoch mini `run_training` run,
and the mini `run_final_evaluation` run.

**Requires Colab validation (not verified here):**
- Anything that goes through real torchvision: this machine has torch 2.14.0+cpu but **no
  torchvision**, so `models.resnet50` has never actually run here. `ensure_training_preconditions`
  correctly refuses with a clear message, and the helper logic itself was executed against a
  stand-in module, but the real construction path is unverified until cell `# 4` prints
  `MODEL_CONSTRUCTION_OK : True`.
- Downloading/loading `ResNet50_Weights.IMAGENET1K_V1` and building the real frozen ResNet-50
- Real training over 9,924 training images, per-epoch validation, early stopping, and best
  checkpoint selection at 224×224
- Final test evaluation over the 2,144 held-out images
- CUDA device reporting and CUDA execution on an actual GPU runtime. In particular, defect A was
  reproduced with the `meta` device and fixed, but the CUDA arm of that fix has not been run on
  hardware.
- Drive mount resolution and the `/content/major_project` fallbacks in a real Colab environment
- Whether Colab's preinstalled torch/torchvision satisfy the pinned stack. The repaired pins were
  verified resolvable for CPython 3.12/3.13 on `linux_x86_64` (and end-to-end through
  `pip install -r`), but no Colab runtime has executed the install

---

## 7. Unresolved issues and limitations

1. **No Colab/GPU verification.** Everything above is local CPU evidence. Colab compatibility and
   GPU execution remain unverified, and nothing in this report claims otherwise.
2. **Resume is not implemented.** Interrupted runs must restart; the checkpoint state needed for a
   future resume is written but untested.
3. **Pretrained weights are never fetched automatically**, so the first Colab run needs either a
   cached copy or `allow_weight_download=True`. The actual download path is untested here.
4. **The `DR_SMOKE_IMAGE_ROOT` variable is a convenience for local runs.** On Colab the corpus is
   discovered through `DR_FUNDUS_ROOT`, Drive, or the `/content/major_project` sibling layout.
5. **No git repository exists**, so "existing files unchanged" rests on file modification times
   rather than a diff.
6. **torchvision is not installed in this local environment**, so the model-building path — the
   single most important piece of the training pipeline — is unverified locally. It must be
   validated in Colab (cell `# 4` is the cheapest place to confirm it: it builds the architecture
   with `weights=None` and needs no download).
   A stand-in `torchvision` module was used to execute the helper's own branching, which is weaker
   evidence than a real import.
7. **The pinned dependency stack was repaired but not installed anywhere.** `torch==2.6.0+cu124` /
   `torchvision==0.21.0+cu124` resolve for CPython 3.12 and 3.13 on `linux_x86_64`, but installing
   them would also move numpy/Pillow on a healthy runtime, so the notebook only installs them when
   the stack is genuinely missing. If the runtime is healthy, skip cell `# 1`.
8. **Local verification of the requirements file used `pip --dry-run` on Windows plus
   platform-emulated resolution**, not a real Linux install: the `+cu124` wheels carry the
   `linux_x86_64` tag while numpy/Pillow carry `manylinux*`, so no single local command could
   resolve both groups at once. The combined `-r` resolution succeeded natively, which is the same
   code path the notebook uses.
7. **The image corpus is required for the strongest checks.** Without it, the image-dependent
   smoke and pipeline checks self-report as skipped; the remaining checks still run.
9. **The five-band mapping remains provisional.** Two merges (Very Severe NPDR → Severe NPDR,
   Advanced PDR → PDR) are unsigned, and 503 Zenodo-derived images lack verified patient linkage.
   Any metrics produced by this pipeline are provisional and not clinically valid.

---

## 8. Suggested next steps

- Run the notebook in Colab on a GPU following `colab/COLAB_EXECUTION_CHECKLIST.md`, and record the
  actual device report, the per-epoch timings, and the validation curve. This is the only way the
  real torchvision, ImageNet-weight, CUDA, and full-partition paths get verified — including the
  CUDA arm of the defect-A fix.
- Implement validated resume (load `run_meta.json` + the last epoch checkpoint, re-verify the
  config and the checkpoint compatibility report, refuse a mismatched run) and test it.
- The CPU mini-run fixture asked for in the previous version of this list now exists: `run_training`
  and `run_final_evaluation` are both exercised end to end on real images with the model stubbed.
  The remaining gap is a fixture that exercises the real `build_frozen_resnet50`, which needs
  torchvision and therefore cannot run in this environment.
- Wire the audit checkpoint reconciliation artifact into the pipeline so a run's provenance ties
  back to the audit outputs.
