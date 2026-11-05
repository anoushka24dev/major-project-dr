# A Multimodal Explainable Agentic AI Framework for Trustworthy Diabetic Retinopathy Diagnosis from Fundus Images

Research software for five-class diabetic-retinopathy (DR) grading from colour fundus
photographs, with lesion segmentation, Grad-CAM explainability, calibration and
uncertainty reporting, and honest cross-dataset evaluation.

> **This is research software, not a medical device.** It makes no diagnostic claim,
> has no clinical validation and has not been reviewed by a clinician. The five-class
> label mapping is provisional. Predictions flagged as uncertain must be reviewed by a
> human; they are not confirmed diagnoses.

---

## Quick start

```
1. Clone the repository
2. Double-click  RUN.bat
3. Wait for it to finish
4. Open  outputs/reports/final_report.html
```

That is the whole workflow. `RUN.bat` prepares the Python environment, detects the
hardware, acquires the datasets, trains or reuses both models, evaluates them,
generates explainability output and calibration metrics, and writes the report.

The first run also creates `.venv/` and installs dependencies, which takes several
minutes. Later runs reuse it.

`RUN.bat --download-only` acquires **every dataset it can without your credentials**:
MMRDR (Figshare) and DDR (official Google Drive folder) in full, plus APTOS once a Kaggle
token and the competition rules are in place. Downloads are idempotent and resumable; a
second run transfers nothing that is already verified. The acquisition report is written to
`outputs/manifests/dataset_availability.md`.

**Mendeley and IDRiD cannot be fetched anonymously** (the reasons are documented below, in
*Dataset access*). Supply them as genuine local copies, or give IDRiD an authorized session:

```
RUN.bat --fundus-root "PATH"   # genuine Mendeley corpus (13 class folders)
RUN.bat --idrid-root  "PATH"   # genuine IDRiD images + official lesion masks
RUN.bat --aptos-root  "PATH"   # APTOS 2019 (Kaggle account required)
RUN.bat --ddr-root    "PATH"   # DDR (if the automatic Drive transfer is unavailable)
set DR_IDRID_URLS=<archive urls>   + set DR_IDRID_COOKIE=<session cookie>  (optional)
```

or set `DR_FUNDUS_ROOT`, `DR_IDRID_ROOT`, `DR_APTOS_ROOT`, `DR_DDR_ROOT`.

There is **no auxiliary dataset generator**. `scripts/prepare_auxiliary_datasets.py` is a
provenance guard that refuses to fabricate data — it never builds a Mendeley folder set or a
set of "ground-truth" masks from another dataset. A missing dataset makes its stage `blocked`;
it is never replaced by look-alike data.

---

## What the pipeline actually does

| # | Stage | What it does | Output |
|---|---|---|---|
| 1 | `download` | Queries the provider APIs, downloads what is publicly available, verifies checksums, extracts safely | `datasets/`, `outputs/manifests/download_manifest.json` |
| 2 | `prepare` | Builds **one** unified five-class manifest from Mendeley + MMRDR + APTOS + DDR, audits label mapping, detects duplicates, creates leakage-safe splits | `outputs/manifests/*.csv`, `dataset_summary.json` |
| 3 | `quality` | Flags blurred / under- and over-exposed / unreadable images | `outputs/manifests/image_quality.csv` |
| 4 | `train_classification` | Trains (or resumes) a five-class ResNet-50 on the combined data | `outputs/classification/best_resnet50.pt` |
| 5 | `train_segmentation` | Trains a four-channel lesion U-Net on IDRiD | `outputs/segmentation/best_lesion_unet.pt` |
| 6 | `explainability` | Grad-CAM maps and lesion-overlap validation | `outputs/explainability/` |
| 7 | `evaluate_classification` | Full held-out test metrics | `outputs/classification/classification_metrics.json` |
| 8 | `evaluate_segmentation` | Per-lesion Dice/IoU and overlays | `outputs/segmentation/segmentation_metrics.json` |
| 9 | `calibration` | ECE, reliability diagram, entropy, review flags | `outputs/evaluation/calibration_results.json` |
| 10 | `cross_dataset` | Per-source and external-partition comparison | `outputs/evaluation/cross_dataset_results.json` |
| 11 | `report` | Self-contained HTML report + metrics table | `outputs/reports/final_report.html` |

A stage whose input data does not exist is recorded as **blocked** with the measured
reason; stages that depend on it are **skipped**. A stage that raises is **failed**, and
the run exits non-zero. Nothing is ever reported as successful when it did not run.

---

## Implementation status

| Component | Status |
|---|---|
| Five-class ResNet-50 on combined Mendeley + MMRDR | **Implemented** |
| Four-channel lesion U-Net (MA, HE, EX, SE) | **Implemented** |
| Grad-CAM explainability (reuses the existing engine) | **Implemented** |
| Lesion-based explanation validation | **Implemented** (requires IDRiD) |
| Image-quality assessment | **Implemented** (advisory) |
| Uncertainty (confidence + entropy) and review flags | **Implemented** |
| Expected Calibration Error + reliability diagram | **Implemented** |
| Classification / segmentation metrics | **Implemented** |
| Cross-dataset evaluation | **Implemented** (needs both sources present) |
| Checkpoint save / resume / compatibility fingerprinting | **Implemented** |
| Dataset acquisition | **Implemented**, with documented provider limits |
| Clinical-information fusion | **NOT IMPLEMENTED — PROPOSED** |
| Fuzzy-logic conflict resolution | **NOT IMPLEMENTED — PROPOSED** |

The clinical/fuzzy component is deliberately absent rather than stubbed:
`src/clinical_status.py` records exactly what is missing and why, and the pipeline
refuses to fabricate clinical records. Adding a placeholder would have implied a
clinical capability and a validation that do not exist.

---

## Datasets

### 1. Mendeley Data — `6ry9n79gb4` (DOI `10.17632/6ry9n79gb4.1`, CC BY 4.0)

Purpose: ResNet-50 classification training.

13 class folders, 22,453 fundus images. Seven folders are DR severity grades; the other
six are different retinal diseases and are **excluded** with a recorded reason.

### 2. MMRDR — Figshare article `29423747`

Purpose: ResNet-50 classification training (combined with Mendeley).

The article publishes a ~18.7 GB split zip (`MMRDR.zip.001` … `.009`). Inside are three
modalities:

| Subset | Used? | Why |
|---|---|---|
| `MMRDR-CFP` | **Yes** | Colour fundus photographs; `FP.csv` grade 0–4 is exactly our five classes |
| `MMRDR-OCT` | **No** | Different modality; its grade column is DME severity (0–2), a different task |
| `MMRDR-UWF` | **No** | Different modality (ultra-widefield), not an ordinary fundus photograph |

MMRDR already ships an official train/test split via the `tr…` / `ts…` filename prefixes.
The pipeline respects it: `tr` images feed the internal train/validation/test split, and
`ts` images become an `external_test` partition that is **never trained on** and is used
only for cross-dataset evaluation.

### 3. IDRiD — IEEE DataPort / Grand Challenge

Purpose: lesion segmentation and annotation-based explanation validation.

Provides the fundus images and the four per-lesion annotation sets (MA, HE, EX, SE).

### 4. APTOS 2019 — Kaggle `aptos2019-blindness-detection`

Purpose: genuine five-grade DR classification; combined training source and an
independent external test set in external-validation mode.

Labels come from the official `train.csv` (`id_code`, `diagnosis` 0–4), mapped 1:1 onto the
five classes, verbatim. APTOS publishes only a labelled training file, so the pipeline does
**not** claim any APTOS image is an official test image: the labelled images either feed the
internal split (combined mode) or are held out entirely (external mode).

Access requires a Kaggle account, accepted dataset terms and an API token at
`~/.kaggle/kaggle.json`. The pipeline checks for the token and the `kaggle` CLI, downloads
with the CLI when both exist, and otherwise reports `blocked` with the exact steps. It never
writes or stores a token.

### 5. DDR — OIA-DDR (`nkicsl/DDR-dataset`)

Purpose: genuine DR classification; combined training source and an independent external
test set in external-validation mode.

Images are distributed as ten byte-split zip chunks on Google Drive / Baidu Drive, so they
cannot be fetched anonymously over HTTP. The loader validates a locally supplied copy:
`DR_grading/train.csv`, `valid.csv`, `test.csv` plus the images. Grades 0–4 map 1:1; the
official grade 5 (**ungradable**) is excluded with a recorded reason and is never mapped to
No DR. The official `train`/`valid`/`test` partition of every image is preserved, and the
official `test` images stay in the external partition — they are never trained on.

Licence: CC BY-NC-SA 4.0 (non-commercial research use only).

### Dataset access — what is and is not automatic

These were verified against the live provider APIs, not assumed.

| Dataset | Automatic download | Reality |
|---|---|---|
| MMRDR | **Yes** | Figshare serves it anonymously (article 29423747). The API's per-file `computed_md5` and size are verified during a resumable transfer; the nine split parts are byte-concatenated, the zip's CRCs are checked, extraction is traversal-guarded, and FP.csv is then matched against the CFP images while the official `tr`/`ts` prefixes are preserved. |
| APTOS | **Only with Kaggle credentials** | `kaggle competitions download -c aptos2019-blindness-detection` is attempted first (dataset mirror as a fallback). Missing CLI, missing `kaggle.json`, unaccepted competition rules and rejected tokens each produce a distinct message. After extraction every `train.csv` id is matched to an image. Otherwise `blocked`, with instructions. |
| Mendeley | **No** | The public API currently reports `size = 0` and an **empty file list** for record `6ry9n79gb4` v1 (re-checked 2026-11-05). There is nothing to download. The pipeline records the real API response, never substitutes another dataset, and uses an existing local copy if one is configured. |
| DDR | **Yes** | The ten `DDR-dataset.zip.0NN` chunks in the official public Google Drive folder are listed anonymously, fetched through Drive's confirmation form with HTTP Range resume, joined, CRC-verified, extracted and label-checked. If Drive refuses, the exact error and the official link are reported. |
| IDRiD | **Only with an authorized session** | IEEE DataPort requires signing in. The pipeline accepts the resolved archive URLs and session Cookie in `DR_IDRID_URLS` / `DR_IDRID_COOKIE` (environment only, redacted everywhere) and then verifies 1:1 image/mask pairing. No password is stored, and access restrictions are not bypassed. |

**What you must do for Mendeley.** If you already have the corpus, point the pipeline at
it — the folder that directly contains the 13 class folders:

```
RUN.bat --fundus-root "C:\path\to\Diabetic Retinopathy and Retinal Diseases Fundus I"
```

or set the environment variable `DR_FUNDUS_ROOT`. The pipeline validates the folder
before using it and labels its provenance honestly as a *local pre-existing copy*, not as
a download.

**What you must do for IDRiD.**

1. Register at <https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid>
   (mirror: <https://idrid.grand-challenge.org/>) and accept the terms.
2. Download `1. Original Images` and `2. All Segmentation Groundtruths`.
3. Extract so the numbered folders sit directly inside `datasets/idrid/`:

```
datasets/idrid/
├── 1. Original Images/
│   └── a. Training Set/            IDRiD_01.jpg …
└── 2. All Segmentation Groundtruths/
    └── a. Training Set/
        ├── 1. Microaneurysms/      IDRiD_01_MA.tif …
        ├── 2. Haemorrhages/        IDRiD_01_HE.tif …
        ├── 3. Hard Exudates/       IDRiD_01_EX.tif …
        └── 4. Soft Exudates/       IDRiD_01_SE.tif …
```

4. Run `RUN.bat --resume`. Pairing is validated first: every image must have exactly one
   mask per lesion channel, and mismatches stop the segmentation stage with a clear
   message rather than being silently tolerated.

You can also use `--idrid-root "PATH"` or `DR_IDRID_ROOT`.

**What you must do for APTOS and DDR.**

* APTOS: create a Kaggle API token (`Account → Create New API Token`), save it to
  `~/.kaggle/kaggle.json`, install the `kaggle` CLI, accept the **competition rules** at
  <https://www.kaggle.com/c/aptos2019-blindness-detection/rules>, then
  `RUN.bat --download-only`. Or point the pipeline at an existing extracted copy with
  `--aptos-root "PATH"` / `DR_APTOS_ROOT`.
* DDR: nothing to do when the Google Drive transfer succeeds
  (`RUN.bat --download-only --datasets ddr`). If Drive blocks it, download the ten chunks
  from <https://github.com/nkicsl/DDR-dataset>, recombine
  (`cat DDR-dataset.zip.0* > DDR-dataset.zip`), extract, then `--ddr-root "PATH"` /
  `DR_DDR_ROOT`.

**No auxiliary dataset generator is provided.**

A script once existed at `scripts/prepare_auxiliary_datasets.py` that claimed to generate
“verified ground-truth” IDRiD lesion masks (`IDRiD_01_MA.tif` …) and to populate the 13
Mendeley class folders from MMRDR images. That was not genuine source data: it misrepresented
MMRDR images as Mendeley and IDRiD images, invented severity distinctions the MMRDR `grade`
field does not support, assigned “No DR” images to unrelated retinal diseases, and substituted
RGB-threshold heuristics for the official IDRiD human annotations. It has been replaced by an
explicit provenance guard that refuses all of those behaviours and points at the genuine-data
instructions below.

Do not try to recreate Mendeley or IDRiD data from another dataset. When a dataset is missing,
mark its stage blocked and supply the real data.

---

## The five classes

| Numeric label | Class (report wording) | Canonical name in code/CSV |
|---|---|---|
| 0 | No Apparent DR | `No apparent DR` |
| 1 | Mild NPDR | `Mild NPDR` |
| 2 | Moderate NPDR | `Moderate NPDR` |
| 3 | Severe NPDR | `Severe NPDR` |
| 4 | Proliferative DR (PDR) | `PDR` |

Canonical names match the repository's existing `dataset/labels.py` and the existing
`provisional_primary_split.csv`, so the legacy baseline and this pipeline stay
interchangeable. Display wording is used only in reports.

### How labels were mapped (audited)

* **Mendeley** — `11.Normal Fundus` → 0, `3.Mild DR` → 1, `4.Moderate DR` → 2,
  `5.Severe DR` → 3, `6.Proliferate DR` → 4. Two **provisional merges**:
  `13. Very Severe NPDR` → 3 and `12. Advanced PDR` → 4.
* **MMRDR** — `grade` 0–4 maps 1:1, verbatim, to the same five classes. No merge.
* **APTOS** — `diagnosis` 0–4 maps 1:1, verbatim. No merge.
* **DDR** — `diagnosis` 0–4 maps 1:1, verbatim; grade 5 (ungradable) is excluded.
* **Non-DR Mendeley folders** (Dry AMD, Wet AMD, Cataract, Hypertensive Retinopathy,
  Pathological Myopia, Glaucoma) are excluded with a written reason. They are never
  mapped to a DR grade.
* An **unknown label stops the pipeline.** Nothing is silently defaulted.

Every decision, plus the number of images affected, is written to
`outputs/manifests/label_mapping_audit.csv`. Excluded images remain in
`outputs/manifests/all_images.csv` with `included = no` and a reason — nothing is deleted.

---

## Models

### ResNet-50 (five classes)

ImageNet-pretrained ResNet-50, final layer replaced with five outputs. Trained on the
combined manifest with cross-entropy, optional inverse-frequency class weights,
AdamW/SGD, linear warm-up + cosine decay, optional AMP, gradient clipping, early stopping
and best-checkpoint selection on validation macro-F1.

* One classifier for **both** sources. No per-dataset model is created.
* Checkpoint selection and early stopping use **validation data only**.
* The test partition is opened once, after training, by the evaluation stage.
* Every checkpoint embeds a **fingerprint** (architecture, class order, image size,
  manifest version). A checkpoint whose fingerprint no longer matches is never reused or
  resumed; the mismatch is reported instead.

### Four-channel U-Net (lesion segmentation)

Encoder–decoder U-Net with four independent output channels in the fixed order
**MA, HE, EX, SE**. Channels are independent sigmoids, not a softmax — lesions co-occur.

Loss is BCE + soft Dice. Per-lesion decision thresholds are tuned **on validation data
only** and then frozen for the test partition. Metrics are reported per lesion plus macro.

### Preprocessing

Both paths resize with aspect-ratio preservation and pad to the square input — **no
centre crop and no random crop**, so a lesion near the retinal border can never be
removed. Images are converted to RGB and ImageNet-normalised.

Training augmentation (training partition only) is deliberately mild: horizontal flip, a
small rotation (±10°), and small brightness/contrast jitter. Vertical flips and random
crops are refused by the code even if enabled in configuration, because they change
clinically meaningful lesion appearance.

---

## Explainability

Grad-CAM is computed at the final convolutional block of ResNet-50 (`layer4[-1]`), reusing
the repository's existing `colab/gradcam.py` engine through an explicit adapter. For each
selected test image the pipeline writes the original image, the heatmap, the overlay, and
a row of prediction metadata.

**Interpretation limits.** Grad-CAM shows which regions contributed to the predicted
logit. It is *not* proof that the model used clinically valid evidence, and the coarse
7×7 map must not be read as a lesion localisation.

### Lesion-based explanation validation

When IDRiD is available, the Grad-CAM map for the predicted class is min-max normalised,
binarised at a configured percentile, and compared (Dice and IoU) against the union of the
four lesion masks and against each mask separately. Masks are transformed with the same
scale-and-pad geometry as the model input.

Two Dice numbers exist in this project and they are **not comparable**:

| Quantity | Where | Meaning |
|---|---|---|
| Lesion segmentation Dice | `outputs/segmentation/segmentation_metrics.json` | U-Net predicted masks vs IDRiD annotations |
| Explanation-overlap Dice | `outputs/explainability/explanation_validation.json` | binarised Grad-CAM vs IDRiD annotations |

They are reported separately and are never averaged together.

---

## Calibration, uncertainty and human review

* **ECE** — equal-width binning over the max softmax probability into 15 bins (the top bin
  includes 1.0). The strategy is named in the output JSON so the number is reproducible.
* **Reliability diagram** — accuracy against mean confidence, plus a confidence histogram.
* **Entropy** — predictive entropy in nats (max `ln 5 ≈ 1.609`).
* **Review flag** — a prediction below the confidence threshold or above the entropy
  threshold is flagged for human review, and the flag itself is reported, never presented
  as a diagnosis.

---

## Cross-dataset evaluation

Two separate things are reported:

1. **Source-wise performance on the internal test partition** — the unified test split
   contains both sources, so splitting the metrics by source shows whether one source is
   systematically harder.
2. **MMRDR's own official test partition** — never trained on. This is the honest
   cross-dataset number.

If only one source is present, no cross-dataset claim is made; the file says so.

---

## Installation and hardware

You normally do not run any of this by hand — `RUN.bat` does it. What it does:

1. Checks that Python is 64-bit and ≥ 3.9 (3.11 or 3.12 recommended).
2. Creates `.venv` with the same interpreter.
3. Installs **PyTorch separately**, choosing an index from detected hardware:
   * NVIDIA GPU + driver ≥ 525 → CUDA 12.1 build
   * otherwise → CPU build
4. Installs the remaining dependencies from `requirements.txt`.
5. Verifies every import **inside** the target interpreter and re-checks
   `torch.cuda.is_available()` — a physically present GPU is not treated as proof that
   CUDA works.
6. Records the exact installed versions in `outputs/reports/environment.json`.

`requirements.txt` deliberately does **not** pin PyTorch. The previous file pinned
`torch==2.14.0` / `torchvision==0.29.1` / `scikit-learn==1.9.1`, which is not a
verified-compatible set for arbitrary Python versions, and it was stored as UTF-16, which
`pip -r` cannot read. It is now UTF-8 with compatible ranges.

### GPU vs CPU

| | GPU (CUDA) | CPU |
|---|---|---|
| Works | Yes | Yes |
| Speed | Much faster | Slow but functional |
| Mixed precision | Enabled automatically | Disabled (fp32) |
| Batch size | Auto-sized from VRAM | Auto-sized from RAM |

Batch sizes and worker counts adapt to the detected hardware and are **recorded in the run
metadata**, so a change made for hardware reasons is never invisible. Methodology (class
mapping, split policy, loss, metrics) is never changed for hardware reasons.

If CUDA is unavailable the pipeline says so explicitly and offers CPU execution rather
than failing.

---

## Running it

```
RUN.bat                        full pipeline (default)
RUN.bat --resume               resume interrupted training
RUN.bat --download-only        acquire + verify every missing dataset
RUN.bat --download-only --datasets mmrdr ddr   only those sources
RUN.bat --prepare-only         manifests and quality only
RUN.bat --train-only           training only
RUN.bat --train-only --cv-folds 5   grouped stratified K-fold + final retrain
RUN.bat --evaluate-only        evaluation + report only
RUN.bat --report-only          rebuild the report from existing outputs
RUN.bat --force-retrain        ignore existing checkpoints and train from scratch
RUN.bat --skip-download        validate datasets without transferring bytes
RUN.bat --fundus-root "PATH"   use an existing Mendeley corpus
RUN.bat --idrid-root "PATH"    use an existing IDRiD copy
RUN.bat --aptos-root "PATH"    use an existing APTOS 2019 copy
RUN.bat --ddr-root "PATH"      use an existing DDR copy
RUN.bat --dataset-mode external  apply a dataset composition to a single stage run
RUN.bat --config "PATH"        use an alternative config/config.yaml
RUN.bat --no-setup             skip environment setup (CI / already-installed)
RUN.bat --tests                run the test suite
RUN.bat --help                 show usage
```

### Two ways of combining the datasets

`--mode combined` and `--mode external` select *which sources may train the classifier*.
They run the whole stage list; they do not skip stages.

| Command | Training sources | Held-out evaluation |
|---|---|---|
| `RUN.bat --mode combined` | every source that is present and passed validation (Mendeley + MMRDR + APTOS + DDR) | internal test split, plus the source authors' own test partitions (MMRDR `ts`, DDR official `test`) |
| `RUN.bat --mode external` | `run.development_sources` only (Mendeley + MMRDR) | every other source, held out entirely (APTOS + DDR), plus MMRDR `ts` |

Both modes record their exact source composition in `dataset_summary.json` and
`execution_summary.json`. If a requested source is unavailable the run says so and, when
the configured minimum-source policy still allows it, proceeds with fewer sources — it
never substitutes another dataset and never renames one mode as the other.

External test images are excluded *before* the internal split is computed, so they cannot
influence training, preprocessing choices, model selection, thresholds or calibration.

### The ResNet-50 experiment series (E0–E6)

```
RUN.bat --mode combined --experiments all
RUN.bat --mode external --experiments all
RUN.bat --mode combined --experiments "E0 E4"     # a subset
```

| Experiment | Configuration |
|---|---|
| E0 | Baseline ResNet-50 (cross-entropy) |
| E1 | Class-weighted cross-entropy (weights from the training fold only) |
| E2 | E1 + two-stage fine-tuning (head warm-up, then unfreeze) |
| E3 | E2 + retinal augmentation variant (training partition only) |
| E4 | E2 + multiclass focal loss — a controlled alternative, not a stack |
| E5 | Best training config + validation-selected class thresholds |
| E6 | Best training config + calibration/uncertainty evaluation |
| `final_candidate` | Re-runs the best training configuration (by validation macro-F1) and publishes its checkpoint for evaluation |

E5 and E6 fit their parameters on the **validation fold of the best checkpoint only** and
write `outputs/experiments/<id>/posthoc.json`. The evaluation stage reports the effect of
those thresholds / that temperature as *supplementary* metrics next to the unchanged
headline metrics, and the calibration stage reports unscaled and temperature-scaled ECE
side by side.

**Mapping to the research plan's E1–E6.** The identifiers above predate the research plan
and are kept so historical results stay comparable; their *meaning* is not silently
redefined.

| Research-plan experiment | Implemented as | Note |
|---|---|---|
| E1 Baseline ResNet-50 | `E0` | identical configuration |
| E2 Class imbalance | `E1` (weighted CE), `E4` (focal loss) | sampling-based balancing is **not implemented**; only one technique per run |
| E3 Augmentation | `E2` + `E3` | `E3` is the retinal augmentation variant; `E2` is the same config without it |
| E4 Inference / model configuration | single-image inference only | flip TTA is **not implemented** |
| E5 Calibration and thresholds | `E5` (thresholds), `E6` (temperature, ECE/Brier/NLL) | fitted on validation only; reported as supplementary |
| E6 Comparison and selection | `final_candidate` + `experiment_comparison.csv` | selection by validation macro-F1, minority recall shown alongside |

### Grouped stratified K-fold cross-validation

```
RUN.bat --train-only --cv-folds 5            # the full protocol below
RUN.bat --train-only --cv-folds 5 --dataset-mode external
```

* Folds are assigned over the **development pool only** (`train_manifest.csv` +
  `validation_manifest.csv`). A row whose `split` is `test` or `external_test` makes the run
  refuse to start.
* Assignment is `StratifiedGroupKFold` on `group_id`, stratified by the five-class label,
  with `run.cv.seed` (default 12345). A group is never split across a fold boundary; this
  is verified after assignment, not assumed.
* If group ids are missing, or a class has fewer groups than folds, the run reports
  **blocked** with the measured reason. The ungrouped fallback exists but must be enabled
  explicitly with `run.cv.allow_ungrouped_fallback` and is recorded in the summary.
* Each fold trains a fresh model through the same engine as E0–E6 (same builder, loss,
  augmentation, checkpoint format), selects on its own validation macro-F1, and writes
  `outputs/experiments/cv_baseline/fold_N/{best_model.pt,config.yaml,history.csv,metrics.json,predictions.csv}`.
* Results are aggregated **two ways**: fold-level mean ± sample standard deviation, and
  pooled out-of-fold predictions (`oof_predictions.csv`) where each image is predicted by
  the one fold that held it out. The pooled estimate is the headline number.
* After the folds, the model is retrained on the complete development pool with a fixed
  epoch budget (median best epoch across folds, `cv_final`), and only then published as
  `outputs/classification/best_resnet50.pt` with `checkpoint_provenance.json`. No test or
  external-test data is involved at any point.

Outputs: `outputs/experiments/cv_baseline/{cv_summary.json, fold_metrics.csv,
 oof_predictions.csv}`, `outputs/manifests/folds/cv_baseline/fold_N/*` (membership
manifests and class distributions).

Every experiment reuses one model builder, one data pipeline, one augmentation policy and
one evaluation path. Each writes `outputs/experiments/<id>/` and its own checkpoint under
`checkpoints/<id>/`; the selected checkpoint is copied to
`outputs/classification/best_resnet50.pt` together with `checkpoint_provenance.json`,
which names the experiment it came from. E5/E6 artefacts are produced by the calibration
and evaluation stages from the promoted checkpoint.

Exit codes: `0` success · `1` a stage failed · `2` environment or configuration problem ·
`3` a dataset is blocked pending manual access.

The underlying orchestrator can be called directly:

```
python -m src.pipeline --mode full
python -m src.pipeline --mode train --resume
python -m src.pipeline --validate-config
```

---

## Outputs

```
outputs/
├── manifests/
│   ├── all_images.csv                 every discovered image, included or not
│   ├── train_manifest.csv             unified five-class manifest per partition
│   ├── validation_manifest.csv
│   ├── test_manifest.csv
│   ├── external_test_manifest.csv     MMRDR official test images (if present)
│   ├── dataset_summary.json           counts, provenance, limitation notes
│   ├── label_mapping_audit.csv        per-label decision + rule + image count
│   ├── duplicate_audit.csv            exact and perceptual duplicate pairs
│   ├── idrid_pairs.csv                verified image/mask pairing
│   ├── image_quality.csv              per-image quality flags
│   └── download_manifest.json         every provider call and transfer
├── classification/
│   ├── best_resnet50.pt
│   ├── training_history.csv
│   ├── training_summary.json
│   ├── classification_metrics.json
│   ├── classification_report.csv
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── test_predictions.csv
│   └── test_probabilities.npz
├── segmentation/
│   ├── best_lesion_unet.pt
│   ├── training_history.csv
│   ├── segmentation_summary.json
│   ├── segmentation_metrics.json
│   └── lesion_predictions/            probability tensors + overlay PNGs
├── explainability/
│   ├── gradcam/                       original, heatmap, overlay per image + CSV
│   └── explanation_validation.json
├── evaluation/
│   ├── calibration_results.json
│   ├── reliability_diagram.png
│   ├── cross_dataset_results.json
│   ├── quality_summary.json
│   └── metrics_summary.csv
└── reports/
    ├── final_report.html
    ├── execution_summary.json
    └── environment.json
```

Datasets, checkpoints, outputs and `.venv` are gitignored and never committed.

---

## Resuming interrupted training

Every epoch writes a `last_checkpoint` (model + optimizer + scheduler + history) and, when
it improves, a `best_checkpoint`. On resume the pipeline:

1. loads the newest `last` checkpoint,
2. verifies its fingerprint against the current configuration and manifest version,
3. refuses to resume if anything that changes the meaning of the weights has changed,
4. continues from the next epoch.

```
RUN.bat --resume
```

A completed run is never silently overwritten: an existing run directory is moved to
`*_superseded_<timestamp>` first. If training is interrupted, a `TRAINING_INCOMPLETE.txt`
marker is written into the run directory.

---

## Configuration

Everything is centralised in `config/config.yaml`:

* dataset IDs, versions and API endpoints
* download and extraction directories
* image sizes, augmentation policy
* the five-class taxonomy and lesion channel order
* the complete label-mapping rules, with a written justification per label
* optimiser, learning rate, weight decay, schedulers, patience, seeds
* checkpoint and output paths
* evaluation, calibration and Grad-CAM settings
* download behaviour and checkpoint-reuse policy

Validate it before running:

```
python -m src.pipeline --validate-config
```

The validator rejects an inconsistent taxonomy (wrong class count, wrong lesion channel
order, mappings outside the class range), split ratios that do not sum to 1, a label that
is both mapped and excluded, and unknown stage names.

---

## Relationship to the existing code

This pipeline was built **on top of** the existing repository, not instead of it.

**Reused as-is:** `dataset/decoding.py`, `dataset/labels.py`, `dataset/manifest.py`,
`dataset/eligibility.py`, `dataset/grouping.py`, `dataset/statistics.py`,
`dataset/split_validation.py`, the patient-identifier conventions from
`audit_output/pretraining_audit/build_registers.py`, and the Grad-CAM engine in
`colab/gradcam.py` (via `src/legacy_bridge.py`).

**Left untouched and still runnable:** the original Mendeley-only baseline
(`colab/train.py`, `colab/pipeline.py`, `colab/cpu_train.py`, the notebooks) and every
audit artefact under `audit_output/`. The legacy path is bound to a frozen backbone and a
hard-coded class order; the unified pipeline is a separate, more general implementation.
`tests/test_unified_pipeline.py` asserts that both ResNet-50 builders agree on the head
shape, and that the configured Mendeley mapping matches `dataset/labels.py` exactly.

**New:** everything under `src/`, `scripts/`, `config/`, `RUN.bat`, this README and
`IMPLEMENTATION_AUDIT.md`.

`IMPLEMENTATION_AUDIT.md` records the full pre-change audit: what existed, what worked,
what was missing, what was broken, and why each design decision was made.

---

## Testing

```
RUN.bat --tests
# or
python -m pytest tests -q
```

The new suite (`tests/test_unified_pipeline.py`, 54 tests) covers configuration
validation, label mapping and its audit, five-class output shape, image and mask loading,
image/mask pairing, duplicate detection, split-leakage prevention, manifest creation,
model forward passes, GPU/CPU detection, checkpoint round-trip and fingerprint
invalidation, metric calculations (Dice, IoU, ECE, kappa, specificity, AUC
unavailability), download verification, archive traversal rejection, pipeline failure
handling, and an end-to-end smoke test.

Where real datasets are unavailable, tests build **synthetic fixtures strictly for
software testing**. No metric produced by a synthetic fixture is ever reported as a
research result.

---

## Common problems

| Symptom | Cause | Fix |
|---|---|---|
| `Python was not found` | Python missing or not on `PATH` | Install 64-bit Python 3.11/3.12 and tick "Add python.exe to PATH" |
| `Environment could not be prepared` | No internet on first run, 32-bit Python, or no disk space | Check the pip output; PyTorch needs ~2.5 GB |
| `CUDA available: False` despite an NVIDIA GPU | CPU-only torch build or an old driver | Update the NVIDIA driver, delete `.venv`, re-run `RUN.bat` |
| The wrong Python is picked up | `py -3` resolves to another installation | `set DR_PYTHON=C:\Python311\python.exe`, then run `RUN.bat` |
| `Out of memory` during training | Batch size too large for the GPU | Lower `classification.batch_size` in `config/config.yaml` |
| Exit code `3` | A dataset needs manual access | Read the printed instructions; then `RUN.bat --resume` |
| `Mendeley exposes no downloadable files` | Provider-side: the record currently publishes no files | Use `--fundus-root` or set `DR_FUNDUS_ROOT` |
| `IDRiD … requires registration` | Provider requires an account | Register, download, extract, then `RUN.bat --resume` |
| `LEAKAGE DETECTED` | A duplicate or patient group would span partitions | This is a hard stop by design; inspect `duplicate_audit.csv` |
| `Checkpoint is NOT reusable` | Config, class order or manifest version changed | Correct, expected behaviour. `RUN.bat --force-retrain` to retrain |
| `no MMRDR-CFP directory found` | MMRDR not downloaded | `RUN.bat --download-only`, or place MMRDR manually |
| Stage `blocked` in the report | Its input data does not exist | That is the honest state; supply the data and re-run |

---

## Research-integrity commitments

The pipeline enforces the following in code, not just in prose:

1. No dataset label is invented; unknown labels halt the run.
2. No download is claimed unless the files exist and validate.
3. No metric is fabricated; unavailable metrics are reported as unavailable with a reason.
4. A model is never described as trained unless a checkpoint was actually written.
5. The test partition never influences checkpoint selection, thresholds or early stopping.
6. Imaging modalities are never mixed (MMRDR OCT and UWF are excluded; only CFP is used).
7. Incompatible label definitions are never silently merged; merges are recorded as
   merges and a label conflict quarantines the affected images.
8. Synthetic clinical metadata is never presented as real; no clinical records exist here.
9. No clinical validation is claimed anywhere.
10. Implemented and proposed components are separated explicitly (see
    *Implementation status* and `outputs/reports/final_report.html`).
11. Every image keeps its source dataset identifier end to end.
12. Reproducibility: fixed seeds, saved manifests, recorded configurations, fingerprinted
    checkpoints.
13. Unavailable components and incomplete experiments are reported plainly in the final
    report and in `execution_summary.json`.

### Known limitations

* The five-class mapping is **provisional**: two Mendeley bands are merged, and no
  specialist has signed it off.
* Patient identifiers are derived only from documented filename conventions and are
  unavailable for MMRDR-CFP. Leakage control for those images rests on exact and
  perceptual duplicate grouping. Patient-independent evaluation is claimed only for the
  subset with a derived identifier.
* Patient identifiers are folder-scoped, matching the project's existing policy.
  Cross-folder patient identity is reported as a count, not resolved.
* Image-quality thresholds are heuristics, not a validated gradability score, and are
  advisory only.
* **No dataset or trained checkpoint is present in this repository snapshot.** `datasets/`
  contains an empty `mendeley/` folder, and no `best_resnet50.pt` or
  `best_lesion_unet.pt` exists. Any earlier claim to the contrary described a machine on
  which the data had been downloaded by hand; it does not describe this checkout. Run
  `RUN.bat --download-only` and re-check `outputs/manifests/dataset_availability.md`
  before believing any count.
* Grad-CAM is coarse (7×7 for a 224 px input) and is not a lesion localisation.

---

## License and attribution

The datasets keep their own licences: Mendeley Data CC BY 4.0, MMRDR per its Figshare
article, IDRiD requires registration and acceptance of its terms. Respect those terms when
publishing anything derived from them.
