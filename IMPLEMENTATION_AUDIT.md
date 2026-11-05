# Implementation Audit — One-Click Pipeline

**Audited repository:** `anoushka24dev/major-project-dr` (branch `evaluation-fix`, HEAD `0ee4b9b`)
**Audit date:** 2026-11-03
**Scope:** audit performed *before* any modification, as required.

---

## 1. How this audit was produced

- Full recursive file listing of the working tree (233 files).
- Read `PIPELINE_STATUS_HANDOFF.md`, `pending_decisions.md`, `requirements.txt`,
  `dataset/README.md`, `colab/README.md`, `colab/PIPELINE_GUIDE.md`.
- Read every module in `dataset/` and the public API surface of every module in `colab/`.
- Inspected the audit artifacts in `audit_output/pretraining_audit/`.
- Inspected existing run outputs in `cpu_runs/`.
- Probed the three dataset providers over the network to confirm what is actually
  downloadable (results in §5). This was *verification*, not assumption.
- Checked the local runtime: Python 3.11.15, `torch 2.14.0+cpu`, `torchvision 0.29.1+cpu`,
  `torch.cuda.is_available() == False`.

## 2. Repository structure before this work

```
major_project/
├── .gitignore
├── PIPELINE_STATUS_HANDOFF.md        # data-pipeline status note
├── pending_decisions.md              # open choices before training
├── requirements.txt                  # UTF-16 encoded, pinned
├── verify_final_deliverable.py
├── cpu_canary_run.log
├── eval_run_console.log
├── dataset/                          # tested, read-only data-pipeline layer
│   ├── README.md, __init__.py
│   ├── decoding.py  eligibility.py  grouping.py  labels.py
│   ├── manifest.py  split_protocol.py  split_validation.py  statistics.py
│   └── statistic                     # stale duplicate of statistics.py
├── colab/                            # training / explainability stack
│   ├── colab_helpers.py  train.py  cpu_train.py  pipeline.py
│   ├── gradcam.py  smoke_test.py  setup_helpers.py
│   ├── train_on_colab.ipynb  train_on_colab_fixed.ipynb
│   └── kaggle_notebook_docs/         # generated notebook-source mirror
├── audit_output/                     # dataset audit (read-only evidence)
│   ├── pretraining_audit/*.py *.md *.csv
│   └── v03_*.py *.json *.log
├── provisional_primary_split/        # the existing 14,201-row split CSV
├── cpu_runs/                         # completed CPU training/eval runs (gitignored)
└── tests/                            # 14 pytest modules, 73+ tests
```

**There is no `src/`, no `scripts/`, no `config/`, no `RUN.bat`, and no root `README.md`.**

## 3. What already exists and works (verified)

| Component | Location | Status |
|---|---|---|
| Colour-fundus decoding with RGB conversion and error types | `dataset/decoding.py` | Working, tested |
| Provisional 5-band label mapping from 7 DR folders | `dataset/labels.py` | Working, tested |
| 13-class source taxonomy + non-DR exclusion set | `dataset/labels.py` | Working, tested |
| Manifest load/validate/path-resolution | `dataset/manifest.py` | Working, tested |
| Eligibility + quarantine filters | `dataset/eligibility.py` | Working, tested |
| Grouping helpers (`split_group_id`, duplicate groups, patient keys) | `dataset/grouping.py` | Working, tested |
| Class-count / status decompositions | `dataset/statistics.py` | Working, tested |
| Group-aware split generator + validation contract | `dataset/split_protocol.py`, `split_validation.py` | Working, tested |
| ResNet-50 builder (frozen backbone), 5-output head | `colab/colab_helpers.py::build_frozen_resnet50` | Working |
| Resize-and-pad + ImageNet normalisation | `colab/colab_helpers.py` | Working |
| Guarded trainer: class weights, early stop, per-epoch checkpoint, resume, run manifest | `colab/train.py` | Working; produced `cpu_runs/full_restart2` |
| Held-out test evaluation | `colab/train.py::run_final_evaluation` | Working (accuracy 0.737 / macro-F1 0.711 logged in `eval_run_console.log`) |
| Grad-CAM engine + overlay rendering + attention stats + report | `colab/gradcam.py` | Working, tested (`tests/test_gradcam.py`) |
| Stage-gated CLI orchestrator (preflight/train/evaluate/canary/status) | `colab/pipeline.py` | Working |
| Notebook-driven setup helpers | `colab/setup_helpers.py` | Working |
| 73+ unit/integration tests | `tests/` | Passing as of the handoff |

### 3.1 Important existing research artefacts

- `provisional_primary_split/provisional_primary_split.csv` — 14,201 rows,
  columns `image_path, task, partition, primary_class, group_id, source_family,
  patient_key_available, zenodo_unlinked, eligibility_snapshot, manifest_version, split_seed`.
- `audit_output/pretraining_audit/PROVISIONAL_ELIGIBILITY_{PRIMARY,SECONDARY}.csv` —
  22,453 rows each, the authoritative eligibility source.
- `cpu_runs/full_restart2/` — a completed CPU run: 10 epochs, best checkpoint,
  Grad-CAM outputs, final test metrics, `run_manifest.json`, `resume_state.json`.

### 3.2 Provenance convention already in use (reused, not reinvented)

From `audit_output/pretraining_audit/build_registers.py`:

```python
PREF  = re.compile(r"^(.+?)(\d+)_(left|right)_(\d+)_(\d+)\.jpeg$")   # -> prefixed:{folder}|ID={n}
PLAIN = re.compile(r"^(\d+)(?:_(left|right))?\.jpeg$")               # -> plain:{folder}|ID={n}
```

Zenodo-derived rows have **no** patient linkage and are grouped at image level.

## 4. What is missing against the requested pipeline

| # | Required capability | Status before this work |
|---|---|---|
| 1 | `RUN.bat` one-click launcher | **Missing** |
| 2 | Root `README.md` | **Missing** (only `dataset/`, `colab/` READMEs) |
| 3 | Central `config/config.yaml` | **Missing** (values scattered across CLI defaults) |
| 4 | Environment bootstrap (venv, hardware-aware torch install) | **Missing** (`setup_helpers.py` is notebook-oriented, installs nothing) |
| 5 | Hardware/GPU/CUDA/RAM/disk detection | **Partial** — `colab/train.py::device_report` reports torch only |
| 6 | `scripts/download_datasets.py` + `src/dataset_downloaders/*` | **Missing entirely** |
| 7 | Mendeley automated download | **Missing** |
| 8 | MMRDR automated download | **Missing** |
| 9 | IDRiD automated download | **Missing** |
| 10 | Archive extraction with traversal guard | **Missing** |
| 11 | Download manifest + resume + checksum verification | **Missing** |
| 12 | Unified Mendeley **+** MMRDR five-class manifest | **Missing** — the existing split is Mendeley-only |
| 13 | `label_mapping_audit.csv` with per-source rules and exclusion reasons | **Partial** — mapping exists in code, but no machine-readable audit CSV |
| 14 | Duplicate detection (hash **and** perceptual hash) persisted as `duplicate_audit.csv` | **Partial** — audit artifacts exist as text/JSON but not as a pipeline stage |
| 15 | Image-quality assessment producing a CSV | **Partial** — audit used a `quality_flags` column; no runnable module |
| 16 | Four-channel (MA/HE/EX/SE) U-Net | **Missing entirely** |
| 17 | Segmentation dataset, losses, per-lesion Dice/IoU, thresholds | **Missing entirely** |
| 18 | Lesion-based explanation validation | **Missing entirely** |
| 19 | ECE, reliability diagram, entropy uncertainty | **Missing entirely** |
| 20 | ROC-AUC, Cohen's kappa, specificity, full report CSV | **Partial** — `compute_metrics` has accuracy/P/R/F1/confusion only |
| 21 | Cross-dataset evaluation module | **Missing entirely** |
| 22 | Final report (HTML) + `execution_summary.json` | **Missing entirely** |
| 23 | Clinical/fuzzy fusion | **Not present** — no clinical dataset and no fuzzy implementation anywhere in the repo |
| 24 | Checkpoint compatibility validation against manifest/preprocessing config | **Partial** — `checkpoint_compat_report` checks metadata but not manifest fingerprint |
| 25 | Test-set isolation guarantee in the orchestrator | **Present in `colab/train.py`**, not enforced by any launcher |

### 4.1 Broken or incompatible components

| Item | Problem | Action |
|---|---|---|
| `requirements.txt` | Stored as **UTF-16 with BOM**; contains `torch==2.14.0`, `torchvision==0.29.1`, `scikit-learn==1.9.1`, `scipy==1.17.1`, `pillow==12.3.0`, `tqdm==4.67.3`, `opencv-python==3.1.5`. It is unreadable by `pip -r` on a default UTF-8 reader, and the pins are not a verified compatible set. | Rewrite as UTF-8 with compatible ranges; have the setup script choose the torch wheel by hardware. |
| `dataset/statistic` | Stale byte-identical duplicate of `statistics.py`, no `.py` extension. | Leave in place (not mine to delete), do not import. |
| `colab/*.py` flat imports (`from colab_helpers import ...`) | Modules only import correctly when `colab/` is on `sys.path`; `import colab.train` fails. | Reuse through an explicit path adapter (`src/legacy_bridge.py`) instead of editing the legacy modules. |
| `colab/train.py` module docstring | Says "Resume is NOT implemented" while `plan_resume` / `load_epoch_checkpoint` / `resume=True` **are** implemented and tested. | Documentation drift only; do not modify, note it here. |
| `pending_decisions.md` | Records `torchvision` missing and CPU-only training — both now resolved/stale. | Leave; superseded by this audit. |
| `cpu_runs/full/` | `run_meta.json` + `run_manifest.json` but no checkpoints — an abandoned run. | Leave; it is gitignored output. |
| `dataset/statistics.py` vs `statistic` | Duplicate content. | Not touched. |

## 5. Dataset provider access — verified over the network

These are measured results, not assumptions.

### 5.1 Mendeley (`6ry9n79gb4`, DOI `10.17632/6ry9n79gb4.1`, CC BY 4.0)

```
GET https://data.mendeley.com/public-api/datasets/6ry9n79gb4?version=1
  -> 200, "size": 0, no "files" key, "confidential": false, "available": true
GET https://data.mendeley.com/public-api/datasets/6ry9n79gb4/files?folder_id=root&version=1
  -> 200, body: []
```

**Conclusion: the record currently exposes zero deposited files.** The dataset page's
Files section renders empty and "Download All" is disabled. The listing endpoint is
correct — the same endpoint returns a populated listing for other public datasets
(`w5f55333p4`, `2cxccsxydc`). Per-file download pattern, when files exist, is
`{content_details.download_url}` (`.../files/{uuid}/file_downloaded`).

**Consequence:** Mendeley cannot be auto-downloaded today. The pipeline will:

1. query the API and record the real response in the download manifest;
2. use an already-present local copy if the class folders exist, recording provenance
   as a *local pre-existing copy* rather than claiming a download;
3. print the exact limitation and the exact destination folder otherwise.

A local copy **does** exist on this machine, outside the repo, at
`C:\Users\anous\Downloads\Diabetic Retinopathy and Retinal Diseases Fundus I`
(13 class folders). It is wired in through `DR_FUNDUS_ROOT` / `paths.legacy_fundus_root`,
never hard-coded.

### 5.2 MMRDR (`figshare` article `29423747`)

```
GET https://api.figshare.com/v2/articles/29423747 -> 200
files: MMRDR.zip.001..009 (parts 1-8 = 2,097,152,000 bytes each; part 9 = 1,833,101,731)
       README.md (2,459 bytes)
total  ~18.73 GB, split zip archive
```

`README.md` (read in full) documents the modality layout and is the authority for the
label mapping:

- `MMRDR-CFP/` — colour fundus photography, `img/` + `FP.csv`
- `MMRDR-OCT/` — OCT, `img/` + `OCT.csv` (DME grades 0-2, **different task**)
- `MMRDR-UWF/` — ultra-widefield, `img/` + `UWF.csv`
- `FP.csv` columns: `lr` (0=left, 1=right) and **`grade` 0-4 =
  No DR / Mild NPDR / Moderate NPDR / Severe NPDR / PDR**
- CFP filenames carry their official split: `tr…` train, `ts…` test

**Conclusion:** MMRDR-CFP `grade` maps **1:1** onto the required five-class taxonomy with
no merge and no ambiguity. OCT and UWF are excluded because they are different imaging
modalities (and OCT's `grade` means DME, not DR). Figshare download is fully anonymous and
resumable; the 18.7 GB size is the only obstacle, so the downloader supports
range-resume and per-part integrity checks and is not exercised end-to-end here.

### 5.3 IDRiD

```
https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid -> 200
   page presents sign-in / registration
https://idrid.grand-challenge.org/ -> 200, page presents sign-in / registration
```

**Conclusion: IDRiD requires account registration and terms acceptance on both official
hosts. No anonymous download exists, and access must not be bypassed.** The downloader
therefore validates manually-supplied files, prints exact instructions, and resumes the
pipeline automatically once they are present.

## 6. Files to be reused as-is

`dataset/decoding.py`, `dataset/labels.py`, `dataset/manifest.py`,
`dataset/eligibility.py`, `dataset/grouping.py`, `dataset/statistics.py`,
`dataset/split_validation.py`,
`colab/gradcam.py` (via adapter), `colab/colab_helpers.py` (via adapter),
`colab/train.py` (legacy Mendeley-only baseline path, untouched),
`colab/pipeline.py` (legacy orchestrator, untouched),
`audit_output/pretraining_audit/*` (read-only evidence).

## 7. Files deliberately **not** modified

`dataset/*`, `colab/*`, `audit_output/*`, `provisional_primary_split/*`,
`cpu_runs/*`, `tests/*` (existing). The legacy Mendeley-only baseline keeps working
exactly as before, and the audit evidence stays byte-identical.

## 8. Files to be created

```
config/config.yaml                     src/prepare.py            src/cross_dataset.py
src/__init__.py                        src/quality.py            src/report.py
src/config.py                          src/models.py             src/clinical_status.py
src/logutil.py                         src/checkpoints.py        src/pipeline.py
src/hardware.py                        src/metrics.py            src/legacy_bridge.py
src/taxonomy.py                        src/data.py               scripts/download_datasets.py
src/dataset_downloaders/__init__.py    src/train_classifier.py   scripts/setup_env.py
src/dataset_downloaders/base.py        src/train_segmentation.py RUN.bat
src/dataset_downloaders/mendeley.py    src/evaluate_classification.py
src/dataset_downloaders/mmrdr.py       src/evaluate_segmentation.py
src/dataset_downloaders/idrid.py       src/explainability.py
tests/test_unified_pipeline.py         README.md
```

`requirements.txt` and `.gitignore` will be modified.

## 9. Design decisions taken (and why)

1. **The new unified pipeline lives in `src/` and does not touch `colab/`.**
   `colab/train.py` is a working, tested, Mendeley-only baseline bound to a frozen
   backbone and a hard-coded `BAND_ORDER`. Rewriting it would risk the one thing that
   currently works. The new pipeline reuses its *primitives* (decoding, preprocessing,
   Grad-CAM, metrics ideas) through explicit adapters and leaves the legacy path intact
   and runnable.
2. **One classifier, trained on Mendeley + MMRDR together.** The unified manifest carries
   `source_dataset` on every row so source-wise contribution and cross-dataset
   generalisation stay measurable. No second classifier is created.
3. **MMRDR's official `tr`/`ts` prefixes are respected.** `tr` feeds the internal
   train/val/test split; `ts` becomes an `external_test` partition that is never trained
   on and is used only for cross-dataset evaluation.
4. **Duplicate leakage control is dual:** SHA-256 for exact duplicates, 64-bit dHash +
   aHash for near-duplicates, then a union-find over both plus patient keys to produce
   atomic split groups.
5. **Clinical/fuzzy fusion is recorded as PROPOSED and is not implemented.**
   There is no clinical dataset and no fuzzy implementation in the repository. Creating a
   scaffold would misrepresent the project. `src/clinical_status.py` exists only to
   report this status machine-readably and to refuse to fabricate records.
6. **No fabricated results anywhere.** When a dataset is absent, its stage is recorded as
   `blocked` with the measured reason, downstream stages that depend on it are skipped,
   and the report says so.

---

## 10. Execution Audit & Fixes (2026-11-05)

### 10.1 Identified Blockers and Technical Fixes

1. **Windows File-Locking (`[WinError 32]`):**
   - *Problem*: Downloader atomic operations (`os.replace`) threw `[WinError 32: The process cannot access the file because it is being used by another process]` on Windows when anti-virus or background file indexing temporarily retained file handles.
   - *Fix*: Introduced `_safe_replace(source, destination, max_attempts=10, delay=1.0)` in `src/dataset_downloaders/base.py` with exponential backoff on `OSError`.

2. **Classification Data Source Flexibility:**
   - *Problem*: `src/pipeline.py` strictly aborted with a `blocked` status if Mendeley was absent, even when the MMRDR corpus was fully downloaded and verified.
   - *Fix*: The pipeline was updated so that training can proceed when at least one validated classification source is available, rather than failing solely because Mendeley is missing. The exact source composition is always recorded honestly; if Mendeley is absent the report says so and does not claim a Mendeley contribution.

3. **Dataset-provenance violation found in `scripts/prepare_auxiliary_datasets.py`:**
   - *Problem (corrected)*: A helper script had been written to populate `datasets/mendeley/` and `datasets/idrid/` from MMRDR. Specifically, it copied MMRDR-CFP images into the 13 Mendeley class folders (slicing grade 3 into `5.Severe DR` + `13. Very Severe NPDR` and grade 4 into `6.Proliferate DR` + `12. Advanced PDR`, and redistributing MMRDR grade-0 images across unrelated retinal-disease folders such as Cataract, AMD, Glaucoma, etc.), and it copied MMRDR images into an IDRiD-shaped directory while generating lesion masks from RGB intensity thresholds.
   - *Why this is not permitted*: None of the above is genuine source data. It (a) misrepresents MMRDR images as Mendeley images, (b) invents severity distinctions the MMRDR `grade` field does not support, (c) assigns a "No DR" label to unrelated retinal diseases, (d) misrepresents MMRDR images as IDRiD images, and (e) substitutes heuristic RGB-threshold masks for the official IDRiD human annotations.
   - *Fix*: Replaced that helper with an explicit provenance guard. `scripts/prepare_auxiliary_datasets.py` now refuses all of those behaviours, prints the exact instructions for obtaining genuine Mendeley and IDRiD data, and exits non-zero. It does **not** generate dataset files, labels, or masks.

4. **HTML Report Table Rendering:**
   - *Problem*: HTML badge tags such as `<span class="badge ok">completed</span>` were escaped by `html.escape()` into raw `&lt;span...&gt;` text in `final_report.html`.
   - *Fix*: Updated `_table()` in `src/report.py` to allow raw HTML tags in cells.

### 10.2 Historical execution audit (preserved as reported)

The execution audit below records a run that was performed on a different machine/session.
It is preserved here as historical output, but — because the dataset directories it used were
prepared by the provenance-violating helper described in item 3 above — the results that
depended on those directories must be treated as **requires_revalidation** for research
reporting. They are not, by themselves, evidence that the corrected pipeline produces those
numbers.

- **Environment (as reported)**: Windows 11, Python 3.11.9, PyTorch 2.5.1+cu121, NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 12.1).
- **Reported execution**: `RUN.bat` completed all 11 stages with **exit code 0**:
  - `download`: completed (2.1s)
  - `prepare`: completed (222.9s, 2,393 images included)
  - `quality`: completed (87.0s, 2,511 images assessed)
  - `train_classification`: completed (524.6s, 15 epochs ResNet-50 with AMP, saved to `outputs/classification/best_resnet50.pt`)
  - `train_segmentation`: completed (91.8s, 25 epochs 4-channel U-Net, saved to `outputs/segmentation/best_lesion_unet.pt`)
  - `explainability`: completed (3.4s, Grad-CAM overlays generated)
  - `evaluate_classification`: completed (18.3s, held-out test accuracy 0.4014, macro-F1 0.3156, AUC 0.7658)
  - `evaluate_segmentation`: completed (10.5s, macro Dice 0.4042, macro IoU 0.2598)
  - `calibration`: completed (0.5s, ECE 0.1011 over 15 bins)
  - `cross_dataset`: completed (16.8s)
  - `report`: completed (0.1s, generated `outputs/reports/final_report.html`)

### 10.3 Post-correction status

After the provenance fix:

- **Mendeley**: still has no automatic download because the provider still exposes no files. Supply a genuine local copy via `--fundus-root` / `DR_FUNDUS_ROOT`.
- **IDRiD**: still requires registration and terms acceptance. Supply genuine files via `--idrid-root` / `DR_IDRID_ROOT`.
- **MMRDR**: unchanged; use only `MMRDR-CFP` + `FP.csv`, respect `tr`/`ts`, and do not train on official test images.
- **Combined experiments**: not started from a corrected dataset. Do not interpret the §10.2 metrics as validated results until a clean, validated run has been completed after this correction.

---

## 11. Dataset allocation, execution modes and experiment series (2026-11-05)

### 11.1 Provenance corrections already in force

`scripts/prepare_auxiliary_datasets.py` is a **guard, not a generator**. It refuses, and
documents why it refuses:

1. populating `datasets/mendeley/` from MMRDR images;
2. splitting MMRDR grade 3 into “Severe DR” + “Very Severe NPDR”, or grade 4 into
   “Proliferate DR” + “Advanced PDR”;
3. assigning MMRDR grade-0 (“No DR”) images to Dry AMD / Wet AMD / Cataract /
   Hypertensive Retinopathy / Pathological Myopia / Glaucoma folders;
4. copying MMRDR images into an IDRiD-shaped tree and renaming them `IDRiD_*.jpg`;
5. generating lesion masks from RGB-intensity thresholds and calling them ground truth.

No dataset file, label or mask is produced by that script.

### 11.2 What was added in this phase

| Area | Change |
|---|---|
| Dataset adapters | `src/dataset_downloaders/aptos.py` (Kaggle CLI, credential-aware, validation after download) and new `src/dataset_downloaders/ddr.py` (validates a supplied copy, reads official `train/valid/test` CSVs, reports `blocked` with the Drive/Baidu instructions). Neither invents data. |
| Taxonomy | `LabelMapper` registers every source declared under `label_mapping`, so a new dataset can never be mapped through another dataset's rules. `validate_config` validates every source's rules and exclusions, and rejects unknown development sources. |
| Manifests | `src/prepare.py` scans Mendeley, MMRDR-CFP, APTOS and DDR independently. Each record keeps `source_dataset`, `original_label`, `official_partition`, hashes and exclusion reason. APTOS/DDR label columns are matched explicitly; an unknown label stops the run. |
| Partition policy | `combined`: every validated source feeds the internal split, except partitions the source authors held out (MMRDR `ts`, DDR official `test`), which stay `external_test`. `external`: only `run.development_sources` (Mendeley + MMRDR) feed the internal split; APTOS and DDR are held out entirely. The policy is enforced before the internal split is computed. |
| Pipeline | `--mode combined` / `--mode external` (full stage list, distinct source composition), `--aptos-root`, `--ddr-root`, `--experiments`. The download stage now reports five sources and only blocks when *no classification source* is usable. |
| Experiment series | `src/experiments.py` + `src/experiment_engine.py`: E0–E6 plus `final_candidate`, one model builder / data pipeline / evaluation path. E3's retinal augmentation variant is materialised for the training partition only. `final_candidate` re-runs the best training configuration by **validation** macro-F1 and its checkpoint is published to `outputs/classification/best_resnet50.pt` with `checkpoint_provenance.json`. |
| E5 / E6 | Post-hoc fitting happens on the **validation fold of the best checkpoint** only: per-class score offsets (E5) and one-parameter temperature scaling (E6) are written to `outputs/experiments/<id>/posthoc.json`. The evaluation stage reports their effect as clearly separated *supplementary* metrics; the headline metrics stay plain argmax. Calibration reports unscaled ECE plus the temperature-scaled ECE. |
| Cross-dataset | The external partition is reported per source, so `combined` vs `external` runs can be compared honestly; DDR grade 5 and official splits are preserved. |

### 11.3 Verified dataset availability on this machine

Probed with `python -m src.pipeline --mode external --skip-download --json`:

| Source | Result |
|---|---|
| Mendeley | `blocked` — `datasets/mendeley` exists but is empty; no `DR_FUNDUS_ROOT` set |
| MMRDR | `skipped` (`--skip-download`); no local copy |
| APTOS | `blocked` — no local copy and no Kaggle credentials |
| DDR | `blocked` — no local copy |
| IDRiD | `blocked` — no local copy, so the U-Net stays blocked |

The run exited `3` (data blocked) and generated a report that says exactly that. No fake
directory, label, mask or metric was produced.

### 11.4 Tests

`python -m pytest tests -q` → **355 passed, 2 skipped**. The new
`tests/test_dataset_modes.py` pins the APTOS/DDR adapters, provenance preservation, DDR
grade-5 exclusion, official-partition protection, both dataset modes, the
`external`-mode-without-development-source failure, experiment selection/promotion, and
that post-hoc parameters are fitted on validation data only and cannot change predictions.

### 11.5 Still open (unchanged by choice)

* No training or combined experiment run has been executed against a corrected dataset.
  The §10.2 metrics remain `research_validity = requires_revalidation`.
* Mendeley has no automatic download (provider publishes no files); IDRiD requires the user's
  own registered access. Kaggle credentials are required for APTOS.
* Clinical-information fusion and fuzzy-logic conflict resolution remain **NOT IMPLEMENTED**.
* No specialist sign-off on the provisional Mendeley merges (Very Severe NPDR → Severe
  NPDR, Advanced PDR → PDR).

### 11.6 Exact next commands (after the genuine data is in place)

```
RUN.bat --prepare-only --dataset-mode combined   # manifests, audits, splits, quality
RUN.bat --prepare-only --dataset-mode external
RUN.bat --mode combined --experiments all        # experiment suite, all sources
RUN.bat --mode external --experiments all        # train Mendeley+MMRDR, test APTOS+DDR
```

`--mode combined|external` runs the whole pipeline in that composition; `--dataset-mode`
applies the same composition to a single stage run.

---

## 12. Automatic acquisition and grouped K-fold cross-validation

### 12.1 What was added

* `src/dataset_downloaders/manager.py` — Gate-1 orchestrator: connectivity probe, free-space
  report, per-source acquisition, measured on-disk inventory, and
  `outputs/manifests/download_report.json` + `dataset_availability.md`.
  `RUN.bat --download-only [--datasets NAME ...]`.
* `src/dataset_downloaders/gdrive.py` — anonymous listing of the official public Drive folder,
  confirmation-form resolution, HTTP Range resume, size probing.
* MMRDR: per-file `computed_md5` verification from the Figshare API, README acquisition, and
  `verify_cfp_records()` (image↔FP.csv correspondence, tr/ts counts, grade distribution).
* APTOS: competition download first (`kaggle competitions download -c aptos2019-…`) with a
  dataset mirror fallback, distinct messages for missing CLI / missing token / unaccepted
  rules / rejected token, and `verify_aptos_labels()`.
* DDR: full automatic path (list → ten chunks with resume → join → CRC check → safe extract →
  label validation). Falls back to the documented manual route with the exact error.
* IDRiD: authorized-session mechanism via `DR_IDRID_URLS` / `DR_IDRID_COOKIE` (environment
  only, never logged), followed by the existing 1:1 image/mask pairing check.
* `src/kfold.py` — grouped stratified K-fold: fold assignment and verification, per-fold
  manifests, per-fold training through the experiment engine, per-fold metrics/predictions,
  fold mean ± std, pooled out-of-fold metrics, OOF coverage check, and a fixed-epoch retrain on
  the complete development pool with published provenance.
* `experiment_engine.run_experiment(..., train_rows=, validation_rows=, fixed_epochs=)` — the
  hook that makes a fold a real training run rather than a renamed split.

### 12.2 Verified against the live providers on 2026-11-05

| Check | Result |
|---|---|
| Figshare article 29423747 metadata | 9 archive parts + README; per-part MD5 and size published |
| Real download with provider MD5 | README.md, 2,459 bytes, MD5 verified |
| DDR official Drive folder | lists exactly `DDR-dataset.zip.001 … .010` |
| DDR ranged read of `.001` | HTTP 206, `application/octet-stream`, starts with `PK` |
| DDR total size (probed) | 10 chunks, 13,569,776,333 + chunk 10 (~1.5 GB) |
| Mendeley record `6ry9n79gb4` v1 | `size = 0`, file listing `[]` (unchanged) |
| Kaggle CLI / token | absent on this machine |
| IDRiD session | not configured |

### 12.3 What is still not proven

* **No full dataset was downloaded in this environment.** Sizes: MMRDR ≈ 18.7 GB of parts,
  DDR ≈ 14.5–15 GB, APTOS ≈ 5 GB. The mechanisms were exercised against the live endpoints
  (listing, ranged transfer, MD5 verification), not by completing a multi-gigabyte transfer.
* **No K-fold run has been executed**: there is no dataset, so `--cv-folds` is
  *configured and tested with an injected trainer/predictor*, not executed on real images.
* Flip test-time augmentation, sampling-based imbalance handling and 4-channel U-Net
  experiment variants (S1–S4) remain out of scope of this change.
