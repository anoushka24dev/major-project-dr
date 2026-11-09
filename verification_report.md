# Verification report

Date: 2026-11-09
Repository: `major-project-dr` (Windows 11 host, project folder on OneDrive)
Scope: the five remaining gaps — RUN.bat, optional TTA, E0–E6 integration, dataset-root
detection, and this report. **No full training, no checkpoint or manifest was
overwritten, and no performance number is claimed anywhere below.**

Status vocabulary used throughout:

| Status | Meaning |
| --- | --- |
| **Implemented** | The code exists and is wired into a real code path. |
| **Integrated** | It is reachable from the CLI/pipeline, not only from a unit test. |
| **Tested** | An automated or scripted check exercises it (synthetic data allowed). |
| **Executed on real data** | It actually ran against a real dataset. Not reached in this environment: no Mendeley, MMRDR, APTOS, DDR or IDRiD tree exists here. |

---

## 1. Previous verification results (carried forward)

| Item | Result from the earlier pass |
| --- | --- |
| Code-level review of E0–E6, CV, training, evaluation, reporting | Traced through the code; consistent. |
| `tests/test_unified_pipeline.py` | 54 passed. |
| `RUN.bat` end-to-end | **Not verified.** Every `cmd.exe /c "RUN.bat …"` returned exit 124 with no batch output. |
| Test-time augmentation | **Missing** (no TTA code, config, tests or artefacts). |
| Dataset-root detection | Verified only by reading `src/config.py`; never exercised through the CLI. |
| `verification_report.md` | Did not exist. |

---

## 2. Root cause of the RUN.bat exit code 124

The timeout was **not** a defect in the batch file. In this agent's shell, `cmd.exe` is
started interactively, so it prints the Windows banner, waits on stdin, and is killed by
`timeout` — which is exactly the observed output and the 124 status:

```
$ timeout 20 cmd.exe /c "echo HELLO_FROM_CMD"
Microsoft Windows [Version 10.0.26300.9818]
(c) Microsoft Corporation. All rights reserved.
C:\Users\anous\OneDrive\Desktop\major_project>          <- waits forever
exit 124
```

The same command through PowerShell reaches real `cmd.exe`:
`powershell -NoProfile -Command "cmd.exe /c 'echo CMD_VIA_PS'"` → `CMD_VIA_PS`, `rc=0`.

All RUN.bat results in section 4 were therefore produced by executing the **actual batch
file** through PowerShell (`& .\RUN.bat …`), which hands the file to `cmd.exe` — not by
calling `python scripts/run_pipeline.py`.

---

## 3. Issues found and fixed in this pass

| # | Issue | Fix | Status |
| --- | --- | --- | --- |
| 1 | `RUN.bat --help` ran the full environment setup first (minutes of pip before any usage text). | `--help`/`-h` now skip setup, alongside `--no-setup`. | Implemented, Integrated, Tested |
| 2 | On a machine without the dependencies, `--help` crashed with `ImportError: PyYAML is required…` and exit 1 (observed with the system Python 3.13). | `scripts/run_pipeline.py` handles help itself: parser-generated usage when the stack is installed, a dependency-free summary otherwise; always exit 0. | Implemented, Integrated, Tested |
| 3 | After `--help`/`--tests`, the launcher still printed `Success. See outputs\reports\final_report.html` — a false success claim for a run that produced no report. | Informational runs (`--help`, `-h`, `--tests`) print `Finished. Informational run…` instead, and the `Results:`/`Metrics:` lines are suppressed. | Implemented, Integrated, Tested |
| 4 | `RUN.bat --smoke` did not exist (`unrecognized arguments: --smoke`). | Launcher-level `--smoke`: loads and validates the config, builds the five-class ResNet-50, runs a forward pass, and averages TTA probabilities on synthetic tensors. No data, no training. | Implemented, Integrated, Tested |
| 5 | No `--mmrdr-root`; MMRDR was pinned to `paths.mmrdr_dir`. | `--mmrdr-root` → `DR_MMRDR_ROOT` → `Config.mmrdr_root()`, used by the MMRDR adapter and the dataset inventory. | Implemented, Integrated, Tested |
| 6 | Four copy-pasted `os.environ[...] = …` blocks in `run_pipeline`, and a malformed path surfaced later as a mysterious stage failure. | Single `apply_root_overrides(args)` over `ROOT_OVERRIDE_FLAGS`; an unusable path raises `ConfigError` (exit 2). A path that does not exist is **not** an error — the acquisition stage reports it honestly. | Implemented, Integrated, Tested |
| 7 | `src/pipeline.py` called `run_prepare(..., dataset_mode=…)` but `src/prepare.py::run_prepare` had no such parameter → **every run that reached the prepare stage died with `TypeError`**. | `run_prepare` accepts `dataset_mode`; `combined` proceeds, `external` stops with an explicit `PrepareError` instead of building a manifest that would put held-out sources in the training pool. | Implemented, Integrated, Tested (unit) |
| 8 | Test-time augmentation did not exist. | `src/tta.py`, `inference.tta` config, pipeline hook, evaluation integration, 23 tests. | Implemented, Integrated, Tested |

---

## 4. RUN.bat test results (real batch execution)

Driver: PowerShell `& .\RUN.bat <args>` with `DR_NO_PAUSE=1`, `DR_NO_OPEN=1`, and
`DR_PYTHON` set to an interpreter that has the dependencies (needed only because the
`.venv` on this machine is a partial install — see §8).

| Case | Command | Exit | Observed |
| --- | --- | --- | --- |
| Help | `RUN.bat --help` | **0** | `[1/4] Locating Python…` → `[2/4] Environment setup skipped (--no-setup or --help was requested).` → usage → `[4/4] Finished with exit code 0.` + `Finished. Informational run (--help/--tests/--smoke/--validate-config): no pipeline report was produced.` No `Success.` claim. |
| Smoke | `RUN.bat --no-setup --smoke` | **0** | `SMOKE: configuration, model output shape and TTA probability averaging` … `OK ResNet-50 forward pass: logits (2, 5)` … `OK TTA averaged 6 views` … `OK averaged probabilities (2, 5), every row sums to 1` … `SMOKE: PASSED`; no `Success.` claim (informational run). |
| Config validation | `RUN.bat --no-setup --validate-config` | **0** | `Configuration OK: …\config\config.yaml`, five classes, 4 lesion channels, manifest version, seed; no `Success.` claim (informational run). |
| Bad argument | `RUN.bat --no-setup --mode bogus` | **2** | argparse: `argument --mode: invalid choice: 'bogus'`; launcher prints `Configuration or environment problem…`. |
| Blocked data | `RUN.bat --no-setup --download-only --skip-download` | **3** | `mendeley -> blocked`, `aptos -> blocked`, `ddr -> skipped`, `idrid -> blocked`; `PIPELINE DOWNLOAD — exit code 3`; launcher prints the exit-3 guidance (not success). |
| Failed stage | (earlier run on the system Python without PyYAML) | **1** | Traceback + `A pipeline stage failed…` guidance. Exit code propagated; no success message. |
| Space in path | `RUN.bat --no-setup --fundus-root "…\datasets\My Data\Fundus Corpus" --download-only --skip-download --datasets mendeley` | **3** | Path arrived as one argument; `mendeley -> blocked`; the manifest records the exact root (see §5). |

Required properties:

| Property | Result |
| --- | --- |
| `--help` | Verified, exit 0 (also on a dependency-free interpreter). |
| `--smoke` | Verified, exit 0, `SMOKE: PASSED`. |
| Argument forwarding | Verified: `logs/RUN_console.log` records the forwarded argv for every case, e.g. `command: …python.exe -m src.pipeline --fundus-root C:\…\My Data\Fundus Corpus --download-only --skip-download --datasets mendeley`; multi-value lists (`--datasets`, `--experiments`) are forwarded unchanged. |
| Python detection | Verified: falls back to `py -3` → `C:\Users\anous\AppData\Local\Programs\Python\Python313\python.exe` (3.13.6, 64-bit) on this machine, and honours `DR_PYTHON`. |
| Virtual-environment selection | Verified: `%~dp0.venv\Scripts\python.exe` is preferred over the base interpreter whenever it exists (observed: a partial `.venv` took precedence over `DR_PYTHON` and the run failed honestly with the missing-module error). |
| Error-code propagation | Verified for 0, 1, 2 and 3 (`set "RC=%ERRORLEVEL%"` … `endlocal & exit /b %RC%`). |
| Paths with spaces | Verified end-to-end for all five `--*-root` flags. |
| No false success | Verified: `Success.` is printed only for a non-informational exit 0. `--help`, `--tests`, `--smoke` and `--validate-config` print `Finished. Informational run…`; a failing stage prints the failing-stage guidance. |

Not verified: a run of the **setup** step to completion. `--smoke` without `--no-setup`
started `python -m venv .venv` and the pip install (correct behaviour), but the install was
interrupted deliberately; see §8 for the resulting `.venv` note.

---

## 5. Dataset-root detection

Mechanism: `--fundus-root` / `--mmrdr-root` / `--aptos-root` / `--ddr-root` /
`--idrid-root` → `apply_root_overrides()` → `DR_FUNDUS_ROOT` / `DR_MMRDR_ROOT` /
`DR_APTOS_ROOT` / `DR_DDR_ROOT` / `DR_IDRID_ROOT` → `Config.fundus_root()` /
`mmrdr_root()` / `aptos_root()` / `ddr_root()` / `idrid_root()`.

Executed through `RUN.bat` (all with `--download-only --skip-download --datasets <src>`
and roots that do not exist):

| Flag | Root supplied | Exit | Adapter result |
| --- | --- | --- | --- |
| `--fundus-root` | `…\datasets\My Data\Fundus Corpus` | 3 | `mendeley -> blocked` |
| `--mmrdr-root` | `…\datasets\My Data\MMRDR copy` | 3 | `mmrdr -> skipped` (transfer disabled by request) |
| `--aptos-root` | `…\datasets\My Data\APTOS 2019` | 3 | `aptos -> blocked` + `APTOS requires the Kaggle API; Kaggle CLI not found on PATH; no valid Kaggle API token at C:\Users\anous\.kaggle\kaggle.json.` |
| `--ddr-root` | `…\datasets\My Data\DDR set` | 3 | `ddr -> skipped` (transfer disabled by request) |
| `--idrid-root` | `…\datasets\My Data\IDRiD copy` | 3 | `idrid -> blocked` + `IDRiD root does not exist yet: …\My Data\IDRiD copy` |

Evidence that the spaced paths survived intact — entries in
`outputs/manifests/download_manifest.json`:

```
source=aptos  root=C:\Users\anous\OneDrive\Desktop\major_project\datasets\My Data\APTOS 2019
source=ddr    root=C:\Users\anous\OneDrive\Desktop\major_project\datasets\My Data\DDR set
source=idrid  root=C:\Users\anous\OneDrive\Desktop\major_project\datasets\My Data\IDRiD copy
```

Automated checks (`tests/test_dataset_roots.py`, 19 passed):

* every flag is registered and parses;
* each flag sets its environment variable and the matching `Config` reader returns the
  resolved path — including a path with spaces;
* an unset `--fundus-root` stays unset (`None`), not silently defaulted;
* an empty value (`--fundus-root ""`) is ignored and sets no variable;
* a missing directory is accepted, reported by `dataset_inventory` with `images: 0`, and
  **no** placeholder CSV/manifest is created there;
* a malformed path (embedded NUL) raises `ConfigError` naming the flag and the dataset;
* missing credentials are reported, not invented: `_kaggle_credentials_present()` is
  `False` with a clean `USERPROFILE`, and `authenticated_sources()` reports
  `configured: False` until `DR_IDRID_URLS`/`DR_IDRID_COOKIE` are set;
* every `--*-root` flag is documented in `RUN.bat`.

No placeholder dataset and no fake manifest was created at any point.

---

## 6. Test-time augmentation (optional)

**Design** (`src/tta.py`): each view is a medically defensible perturbation of the already
preprocessed tensor.

| View | What it is | Why it is admissible |
| --- | --- | --- |
| `identity` | unmodified image | the model's ordinary answer is always part of the average |
| `hflip` | horizontal mirror | the retinal mosaic is near-symmetric for DR presence/severity |
| `rotate_p10` / `rotate_m10` | ±10° rotation, black-pixel fill | inside the 10–15° range the training augmentation already uses |
| `brightness_p10` / `brightness_m10` | ±10% exposure gain | global illumination change, computed exactly in normalised space |

Excluded on purpose: vertical flip, cropping, large rotations, channel permutation —
none is anatomically neutral for fundus photographs. (The training pipeline already
refuses `vertical_flip`/`random_crop`; TTA follows the same rule.)

**Combination:** class probabilities are averaged, never labels. `mean_probabilities()`
validates 2-D shape, a consistent class count, finite values, non-negativity, equal
shapes across views, then returns the arithmetic mean renormalised per row. A tie-break
on a label vote is impossible because no vote is taken.

**Ordinary inference stays available and is the default:** `predict(views=None)` is
exactly one forward pass per batch; `inference.tta.enabled` is `false` in the shipped
config, so nothing changes until it is switched on.

**Validation comparison, never test:** after `train_classification` (and after CV, since
the hook sits in the stage loop) and before any stage that opens the test partition, the
pipeline compares ordinary inference with TTA on `outputs/manifests/validation_manifest.csv`
using the published checkpoint. The result — both metric sets, the views, the delta and the
selection — is written to `outputs/tta/tta_selection.json`. A tie keeps ordinary
inference. A failure here is recorded in the training summary and never fails training.

**Application:** `evaluate_classification` reads that file; if (and only if) `tta` was
selected it averages the same views on the test partition and records
`metrics["inference"] = {mode, views, reduction, selection_file, selected_on, note}` plus
`inference_mode`/`inference_views` in the stage summary. `logits` always come from the
unmodified image, so temperature scaling stays a property of the model.

| Status | Evidence |
| --- | --- |
| Implemented | `src/tta.py`; `inference.tta` in `config/config.yaml`; hook in `src/pipeline.py`; use in `src/evaluate_classification.py`. |
| Integrated | Reachable from `RUN.bat` (full/train/evaluate modes) with no new CLI flag; `--validate-config` accepts the new section; `--smoke` exercises the averaging. |
| Tested | `tests/test_tta.py`: 23 passed — arithmetic mean, no label vote, class order, five-class output shape, rows sum to 1, ordinary = one pass per batch, TTA = one pass per view per batch, logits from the unmodified image, view geometry/exposure, rejection of unknown views and unsupported reductions/metrics, saved selection read back, disabled TTA ignores a saved selection. |
| Executed on real data | **No.** No validation manifest and no trained checkpoint exist here, so no TTA accuracy number exists. **No improvement is claimed.** |

To enable on the GPU laptop: set `inference.tta.enabled: true` in `config/config.yaml`
(everything else has a default). The selected configuration then comes from the validation
comparison, not from a hard-coded choice.

---

## 7. E0–E6 and final_candidate

Every experiment is a config override driven through the single engine
(`src/experiment_engine.py`), so the data pipeline, checkpoint/fingerprint machinery and
metrics are identical across experiments.

| ID | Loss | Learning rates | Augmentation | Scheduler | Fine-tuning | Checkpoint | Selection metric |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | `cross_entropy` (class weighting per config) | all parameters 3.0e-4 | `default` (hflip, ±10°, brightness/contrast 0.08) | cosine + 1 warm-up epoch | single-stage | `runs/experiments/E0` → `best_resnet50.pt` | `val_macro_f1` (max) |
| E1 | `weighted_cross_entropy` (inverse frequency from the **training fold**) | all parameters 3.0e-4 | `default` | cosine | single-stage | `runs/experiments/E1` | `val_macro_f1` |
| E2 | `weighted_cross_entropy` | head 3.0e-4, backbone 3.0e-5 | `default` | cosine | two-stage: head warm-up for 3 epochs, then `unfreeze_all` | `runs/experiments/E2` | `val_macro_f1` |
| E3 | `weighted_cross_entropy` | head 3.0e-4, backbone 3.0e-5 | `retinal` (hflip, ±15°, brightness/contrast 0.12, colour jitter 0.06) | cosine | two-stage (3) | `runs/experiments/E3` | `val_macro_f1` |
| E4 | `focal` (γ=2.0, α=0.25 from config) | head 3.0e-4, backbone 3.0e-5 | `retinal` | cosine | two-stage (3) | `runs/experiments/E4` | `val_macro_f1` |
| E5 | `weighted_cross_entropy` + validation-only class-offset thresholds | head 3.0e-4, backbone 3.0e-5 | `retinal` | cosine | two-stage (3) | `runs/experiments/E5` + `posthoc.json` | `val_macro_f1` for the model; offsets fitted on validation |
| E6 | `weighted_cross_entropy` + ECE / reliability / entropy / temperature | head 3.0e-4, backbone 3.0e-5 | `retinal` | cosine | two-stage (3) | `runs/experiments/E6` + `posthoc.json` | `val_macro_f1`; temperature fitted on validation |
| `final_candidate` | inherits the best of E0–E6 by validation macro-F1 (falls back to E0) | as inherited | as inherited | cosine | as inherited; retrained on the development pool | own directory; published by byte copy with `checkpoint_provenance.json` | `val_macro_f1`; evaluated **once** on the held-out test set |

| Status | Evidence |
| --- | --- |
| Implemented | `src/experiments.py` (`EXPERIMENT_CONFIG_OVERRIDES`, `AUGMENTATION_VARIANTS`, `final_candidate_config`, `select_best_by_validation`), `src/experiment_engine.py` (`build_loss_fn`, `FocalLoss`, two-stage unfreeze, `_maybe_fit_posthoc`). |
| Integrated | `--experiments E0 … final_candidate|all`; comparison JSON/CSV written to `outputs/experiments/`; published checkpoint is what the evaluation stage loads. |
| Tested | `tests/test_experiments_integration.py`, 9 passed: all eight ids defined; each declares a complete recipe with all seven required facts and a distinct checkpoint path; the three loss variants build `CrossEntropyLoss`/`CrossEntropyLoss`/`FocalLoss` with γ and α from config; `final_candidate` inherits the selected experiment and falls back to E0; selection ignores failed/incomplete experiments; the comparison rows carry loss/augmentation/freeze epochs/checkpoint/metric; `external`/`all`/unknown selection rules; augmentation is applied to training data only (evaluation dataset has hflip off, rotation 0). |
| Executed on real data | **No.** No dataset is present, so no experiment has been trained and no comparison table exists. |

---

## 8. Test suite results

New tests in this pass: **51 passed** (`test_tta.py` 23, `test_dataset_roots.py` 19,
`test_experiments_integration.py` 9).

Existing files, each run on its own (the monolithic `pytest tests -q` run stalls — see
below):

| File | Result |
| --- | --- |
| test_canary_split | 19 passed |
| test_class_counts | 7 passed |
| test_colab_pipeline | 88 passed, 2 skipped |
| test_cpu_train | 12 passed |
| test_cross_validation | 14 passed |
| test_dataset_acquisition | 12 passed |
| test_eligibility | 13 passed |
| test_fixed_notebook_setup | 20 passed |
| test_gradcam | 19 passed |
| test_grouping | 9 passed |
| test_image_decoding | 11 passed |
| test_label_mapping | 11 passed |
| test_manifest | 10 passed |
| test_split_validation | 12 passed |
| test_unified_pipeline | 54 passed |
| test_dataset_modes | 16 passed, **4 failed**, 1 deselected (hangs) |
| test_pipeline_entrypoint | 11 passed, then **hangs** (no result for the remainder) |
| test_split_generation | **did not finish** within the 60 s/file budget |

Known failures, all pre-existing and outside this pass's changes:

1. `test_combined_mode_*`, `test_external_mode_*` (3 tests): they expect the prepare stage
   to be dataset-mode aware (`development_sources` held out, an "External-validation mode"
   message). `prepare.py` does not implement that. Previously this surfaced as
   `TypeError: run_prepare() got an unexpected keyword argument 'dataset_mode'` — fixed to
   an explicit `PrepareError`; the feature itself is **not** implemented (see §9).
2. `test_aptos_and_ddr_keep_provenance_and_labels`: expects `summary["sources"]["aptos"]`;
   the prepare summary uses the key `aptos2019`.
3. Hangs (network filesystem/browser work, not related to the changed modules): 
   `test_dataset_modes::test_ddr_missing_copy_is_blocked_not_faked` (DDR adapter attempts a
   Google-Drive request) and `test_pipeline_entrypoint` after the image-decode preflight
   tests. `test_split_generation` also exceeds 60 s.

Environment notes:

* pytest's third-party autoload plugin `pytest_python_requests` is broken here
  (`trafaret has no attribute 'StrBool'`), so the suite was run with
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
* A `.venv` was created at 17:28 by an interrupted `RUN.bat --smoke` (torch and PyYAML not
  yet installed) and has been **removed**, restoring the pre-test state; `RUN.bat` rebuilds
  it on the next normal run. The system interpreter on this machine is Python 3.13.6,
  whose `py -3` resolution has no PyYAML — the launcher now needs no dependencies for
  `--help`, and the setup step is what installs them.

---

## 9. Remaining unverified features

| Item | Why it is unverified | What would close it |
| --- | --- | --- |
| Training, CV and the E0–E6 comparison | No dataset root exists here. | Run the command in §10 on the GPU laptop. |
| TTA's actual effect | Requires a trained checkpoint plus the validation manifest. | Enable `inference.tta.enabled`, then compare `outputs/tta/tta_selection.json`. |
| `dataset_mode=external` in prepare | Not implemented: the manifests would include held-out sources. Now fails loudly instead of crashing. | Implement development-source filtering in `prepare.py` and re-enable the three failing tests. |
| `summary["sources"]["aptos"]` key naming | Pre-existing test/code mismatch (`aptos2019`). | Align the key or the test. |
| Complete `RUN.bat` setup step (venv + pip to completion) | Interrupted deliberately (multi-GB torch download). | Run `RUN.bat` once on the laptop without `--no-setup`. |
| `test_split_generation`, `test_pipeline_entrypoint`, one DDR test | Exceed the 60 s/file budget or hang on network/decode work. | Run the suite on a machine with a local dataset copy and no hanging network calls. |
| Segmentation, Grad-CAM, calibration and cross-dataset stages on real data | Depend on the datasets and on a trained classifier. | Same run as above. |

---

## 10. Command to run on the GPU laptop

From the repository root, with the datasets already downloaded (adjust the paths; quotes
are required where a path contains spaces):

```bat
RUN.bat --help
RUN.bat --smoke
RUN.bat --mode combined --experiments all --cv-folds 5 ^
        --fundus-root "D:\Datasets\Mendeley" ^
        --mmrdr-root   "D:\Datasets\MMRDR" ^
        --aptos-root   "D:\Datasets\APTOS2019" ^
        --ddr-root     "D:\Datasets\DDR" ^
        --idrid-root   "D:\Datasets\IDRiD"
```

To include the optional TTA comparison, set `inference.tta.enabled: true` in
`config/config.yaml` first (or run `--experiments all` without CV and add it afterwards);
the selection is then reported in `outputs/tta/tta_selection.json`.

Expect exit code 0 on success, 3 if a dataset is still blocked (the message names the
exact step and the file `outputs/manifests/download_manifest.json` records it), 2 for a
configuration problem, 1 if a stage failed.
