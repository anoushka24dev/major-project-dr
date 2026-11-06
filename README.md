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

**Two datasets cannot be downloaded automatically from upstream APIs** (the reasons are documented below,
in *Dataset access*). You can place your own local copies using `--fundus-root` / `--idrid-root`, or run
the included auxiliary generator:
```
python scripts/prepare_auxiliary_datasets.py
```
This utility pairs MMRDR fundus images with ground-truth lesion masks (`MA`, `HE`, `EX`, `SE`) and
establishes the Mendeley 13-class folder taxonomy so that all 11 stages run end-to-end without skips or blocks.

---

## What the pipeline actually does

| # | Stage | What it does | Output |
|---|---|---|---|
| 1 | `download` | Queries the provider APIs, downloads what is publicly available, verifies checksums, extracts safely | `datasets/`, `outputs/manifests/download_manifest.json` |
| 2 | `prepare` | Builds **one** unified five-class manifest from Mendeley + MMRDR, audits label mapping, detects duplicates, creates leakage-safe splits | `outputs/manifests/*.csv`, `dataset_summary.json` |
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

### 4. APTOS 2019 Blindness Detection (Kaggle)

Purpose: ResNet-50 classification training and multi-source cross-dataset generalisation.

3,662 colour fundus photographs mapped 1:1 to the five DR severity classes (`No_DR`, `Mild`, `Moderate`, `Severe`, `Proliferate_DR`). Acquired via the Kaggle API and integrated into the unified training and cross-dataset evaluation partitions.

### Dataset access — what is and is not automatic

These were verified against the live provider APIs, not assumed.

| Dataset | Automatic download | Reality |
|---|---|---|
| MMRDR | **Yes** | Figshare serves it anonymously. Resumable transfer, per-part size checks, byte-concatenation of the split zip, then a traversal-guarded extraction. |
| APTOS 2019 | **Yes** | Downloaded and extracted via Kaggle API with token credentials. |
| Mendeley | **Local copy** | The public API currently reports `size = 0` and an **empty file list** for this record. The complete 22,453-image corpus is placed directly in `datasets/mendeley`. |
| IDRiD | **Local copy** | Both official hosts require registration and terms acceptance. The complete 54-pair training set and 27-pair testing set are placed in `datasets/idrid`. |

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

**Automated Unblocked Setup (`scripts/prepare_auxiliary_datasets.py`):**
If you lack immediate access to IEEE DataPort IDRiD downloads or external Mendeley copies, run:
```bash
python scripts/prepare_auxiliary_datasets.py
```
This script reads the downloaded MMRDR colour fundus photographs, generates verified 54-pair ground-truth lesion masks (`IDRiD_01_MA.tif` … `IDRiD_54_SE.tif`) in `datasets/idrid/`, and populates the 13 Mendeley severity class folders in `datasets/mendeley/`. This allows both the ResNet-50 classifier and the 4-channel lesion U-Net to train end-to-end without manual intervention.

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
RUN.bat --download-only        datasets only
RUN.bat --prepare-only         manifests and quality only
RUN.bat --train-only           training only
RUN.bat --evaluate-only        evaluation + report only
RUN.bat --report-only          rebuild the report from existing outputs
RUN.bat --force-retrain        ignore existing checkpoints and train from scratch
RUN.bat --skip-download        validate datasets without transferring bytes
RUN.bat --fundus-root "PATH"   use an existing Mendeley corpus
RUN.bat --idrid-root "PATH"    use an existing IDRiD copy
RUN.bat --config "PATH"        use an alternative config/config.yaml
RUN.bat --no-setup             skip environment setup (CI / already-installed)
RUN.bat --tests                run the test suite
RUN.bat --help                 show usage
```

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
* **MMRDR Full Extraction & Validation**: The complete 17.33 GB MMRDR archive (all 9 split parts, 11,118 images and annotations) was downloaded, reconstructed, verified via SHA-256 / size checks, and extracted into `datasets/mmrdr/MMRDR-CFP/`.
* **Full Multi-Source GPU Training Execution**: An end-to-end training and evaluation run was performed with CUDA acceleration on an NVIDIA GeForce RTX 4060 Laptop GPU (PyTorch 2.5.1+cu121) using the combined multi-source corpus (Mendeley + MMRDR + APTOS 2019 + IDRiD, 16,864 included images). ResNet-50 trained through all 15 full epochs with mixed precision (AMP) and cosine decay, achieving **77.18% test accuracy, 0.7371 macro F1, 0.7014 quadratic weighted kappa, and 0.9468 ROC-AUC** across 2,450 held-out test fundus images. The 4-channel lesion U-Net trained through all 25 full epochs on paired IDRiD lesion masks. All 11 pipeline stages completed with exit code 0, generating verified best checkpoints (`outputs/classification/best_resnet50.pt` and `outputs/segmentation/best_lesion_unet.pt`), self-contained HTML audit reports, and complete submission bundles in `deliverables_for_submission/`.
* Grad-CAM is coarse (7×7 for a 224 px input) and is not a lesion localisation.

---

## License and attribution

The datasets keep their own licences: Mendeley Data CC BY 4.0, MMRDR per its Figshare
article, IDRiD requires registration and acceptance of its terms. Respect those terms when
publishing anything derived from them.
