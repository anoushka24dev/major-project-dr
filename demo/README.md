# `demo/` - 50-training-image / 10-test-image ResNet-50 experiment

A small, self-contained companion experiment to the main pipeline in this repository.

It answers one question: **given only 50 training images and 10 held-out test images, what does
this repository's ImageNet-pretrained ResNet-50 actually score on the five-class DR taxonomy?**

It reuses the repository's own ResNet-50 builder, preprocessing and metric functions, so it is a
sanity check of the same code path the main pipeline uses, not a separate implementation.

> **Scope warning.** 50 training images cannot represent the variability of a real DR screening
> population, and 10 test images give very noisy estimates (one image = 10 percentage points).
> Nothing in this folder is a clinical validation, and the numbers must not be presented as
> full-dataset performance.

---

## 1. Files

| File | Purpose |
| --- | --- |
| `config.json` | Single source of truth: seed, class taxonomy, label mapping, per-class sampling quotas, model and training settings |
| `prepare_data.py` | Locates and inspects the dataset ZIP, extracts it safely, verifies and hashes every image, detects duplicates, maps labels, samples the 50/10 split, writes manifests |
| `train.py` | Trains ResNet-50 on the 50 training images, saves the best checkpoint, history, config and plots |
| `evaluate.py` | Scores the 10 held-out test images **once** with the best checkpoint and writes every metric artefact |
| `requirements-demo.txt` | Python dependencies |
| `manifests/` | Generated CSV/JSON provenance: inventory, duplicates, exact selected filenames and labels |
| `model/` | Generated checkpoint (`best_model.pt`, `last_model.pt`) - not committed (see `.gitignore`) |
| `results/` | Generated results: Markdown report, metrics, confusion matrix, per-class chart, history, curve, predictions |

Start reading the outcomes here: **`results/DEMO_RESULTS.md`**.

---

## 2. Dataset and label mapping

Source archive: *Dataset from fundus images for the study of diabetic retinopathy*, V03
(`Dataset from fundus images for the study of diabetic retinopathy_V03.zip`).

The archive holds seven DR-severity folders plus one annotation spreadsheet. The spreadsheet is a
spreadsheet, not an image, so it is never counted as a sample. Labels come from the archive's own
folder names; nothing is guessed and no label is silently reassigned.

| Archive folder | Images | Mapped label | Provenance |
| --- | --- | --- | --- |
| `1. No DR signs` | 711 | 0 - No apparent DR | verbatim |
| `2. Mild (or early) NPDR` | 6 | 1 - Mild NPDR | verbatim |
| `3. Moderate NPDR` | 110 | 2 - Moderate NPDR | verbatim |
| `4. Severe NPDR` | 210 | 3 - Severe NPDR | verbatim |
| `5. Very Severe NPDR` | 139 | 3 - Severe NPDR | **provisional merge**, same policy the repository already applies to `13. Very Severe NPDR` |
| `6. PDR` | 116 | 4 - PDR | verbatim |
| `7. Advanced PDR` | 145 | 4 - PDR | **provisional merge**, same policy the repository already applies to `12. Advanced PDR` |

Usable images: **1437** (711 / 6 / 110 / 349 / 261 after the provisional merges). Corrupt images
found: **0**. Exact duplicate files found: **0**.

The two merges are the repository's existing provisional policy in `config/config.yaml`
(`label_mapping.mendeley`), not a new mapping invented for this experiment. Both are recorded in
`config.json` and repeated per row in the manifests so a reader can always see that a file's
mapped class is a merge rather than a verbatim source grade.

---

## 3. The 50 / 10 split

Seeded (`12345`), class-aware, without replacement, drawn from the deduplicated class pools.
Test images are drawn first, then the validation carve-out, then the fit set, so no image can
appear twice.

| Mapped class | Fit | Validation | Test | Train total |
| --- | --- | --- | --- | --- |
| 0 - No apparent DR | 11 | 2 | 3 | 13 |
| 1 - Mild NPDR | 4 | 1 | 1 | 5 |
| 2 - Moderate NPDR | 9 | 2 | 2 | 11 |
| 3 - Severe NPDR | 10 | 2 | 2 | 12 |
| 4 - PDR | 7 | 2 | 2 | 9 |
| **Total** | **41** | **9** | **10** | **50** |

* The 50 "training images" are the fit + validation rows. `manifests/selected_train.csv` holds all
  50 with a `used_for` column (`fit` or `validation`).
* The 9 validation images exist only to select the checkpoint and stop training early. They are
  carved out of the 50 training images and never come from the test set.
* Mild NPDR has only 6 images in the archive, so 1 image is left unused. No image is duplicated
  to fill a quota.
* `manifests/selected_test.csv` holds the 10 test images. `train.py` never opens it.

---

## 4. Model and preprocessing

* **Architecture**: ResNet-50, `src/models.py::build_resnet50`, 5 output classes
  (25,583,621 parameters).
* **Initialisation**: ImageNet `IMAGENET1K_V1` weights; the 1000-class head is replaced by a
  fresh 5-class linear layer.
* **Preprocessing** (identical for training and test): decode to RGB with
  `dataset/decoding.py`, aspect-ratio-preserving resize-and-pad to 224x224 with
  `colab/colab_helpers.py::_resize_and_pad`, then ImageNet mean/std normalisation via
  `src/data.py`. No crop of any kind, so border lesions are never cut away.
* **Augmentation** (training rows only, `src/data.py::ClassifierDataset` driven by
  `config/config.yaml`): horizontal flip, rotation +/- 10 degrees, brightness and contrast
  +/- 0.08. Vertical flips and random crops are disabled on purpose because they are not neutral
  for fundus anatomy.

---

## 5. Training

| Item | Value |
| --- | --- |
| Optimiser | AdamW |
| Loss | Cross-entropy with inverse-frequency class weights computed from the 41 fit images only |
| Stage 1 `head_warmup` | 8 epochs, backbone frozen, only `fc` trained, lr 1e-3, weight decay 1e-4 |
| Stage 2 `finetune_top_blocks` | up to 30 epochs, `layer3` + `layer4` + `fc` trained, lr 1e-4 (cosine decay), weight decay 1e-4 |
| Batch size | 8 |
| Class-balanced sampling | disabled in the kept run; tried and rejected on validation (see section 7) |
| BatchNorm | kept in eval mode; 41 images per epoch cannot estimate batch statistics safely |
| Checkpoint selection | best validation macro-F1, ties broken by lower validation loss |
| Early stopping | patience 8 epochs |
| Device | detected automatically (CUDA when usable, otherwise CPU) |
| Determinism | `random`, NumPy and PyTorch seeds fixed; seeded DataLoader generator |

An interrupted run writes `model/TRAINING_INCOMPLETE.txt` and a resumable `model/last_model.pt`;
re-run `python demo/train.py --resume` to continue.

---

## 6. How to run it

From the repository root:

```bash
# 1. Locate the ZIP, extract, verify, sample 50/10, write manifests
python demo/prepare_data.py
#    ...or point at the archive explicitly:
#    python demo/prepare_data.py --zip "C:/path/to/Dataset from fundus images ...V03.zip"

# 2. Train (a few minutes on GPU, roughly 15-30 minutes on CPU)
python demo/train.py

# 3. Score the 10 held-out test images (refuses to run twice)
python demo/evaluate.py
```

`prepare_data.py` reads the archive but never modifies, moves or deletes it. Extracted images go to
`datasets/demo_fundus_v03/`, which is already covered by `.gitignore`.

If a dependency is missing:

```bash
python -m pip install -r demo/requirements-demo.txt
```

The demo needs the repository's own modules as well (`src/`, `dataset/`, `colab/`), so it must be
run from a checkout of this repository, not copied out of it.

---

## 7. Results

Everything is in `results/`:

| File | Contents |
| --- | --- |
| `DEMO_RESULTS.md` | Readable report: setup, all metrics, per-class table, confusion matrix, per-image predictions, interpretation, limitations |
| `metrics.json` | Machine-readable metrics, per-class statistics, confusion matrix, ROC-AUC (or the reason it is unavailable), self-check block |
| `classification_report.csv` | Per-class and macro/weighted precision, recall, F1, support |
| `confusion_matrix.png` | Confusion matrix over the 10 test images |
| `per_class_performance.png` | Per-class precision/recall/F1 bar chart |
| `test_predictions.csv` | One row per test image: truth, prediction, confidence, correctness, full class probabilities |
| `training_history.csv` | Per-epoch loss, accuracy, validation macro-F1, learning rate, epoch time |
| `training_curve.png` | Training/validation loss and accuracy curves, stage boundary marked |
| `training_config.json` | The configuration actually used, plus resolved device, library versions and class weights |
| `model_metadata.json` | Architecture, class mapping, seed, image size, selected file lists, best epoch |

The headline accuracy in `DEMO_RESULTS.md` was computed from `test_predictions.csv` and checked
against `metrics.json` by `evaluate.py` itself (`self_check` block), so the report cannot disagree
with the saved predictions.

### What the included run actually produced

| Metric | Value |
| --- | --- |
| Accuracy | **50.0%** (5/10 test images) |
| Macro precision / recall / F1 | 0.275 / 0.400 / 0.309 |
| Weighted precision / recall / F1 | 0.313 / 0.500 / 0.364 |
| Cohen's kappa | 0.306 |
| ROC-AUC (macro, one-vs-rest) | 0.749 |

This is a weak result and the confusion matrix in `results/DEMO_RESULTS.md` shows why: the model
correctly recognised 3/3 *No apparent DR* and 2/2 *Severe NPDR* images, but assigned *No apparent
DR* to every other test image. In other words it learned a strong prior towards the majority
class rather than a usable five-grade decision boundary, even though its ranking signal is
decent (macro ROC-AUC 0.749). With 4 training images of Mild NPDR that is not surprising, and
it is reported as it is rather than tuned away.

### How this checkpoint was chosen (validation only)

Three training configurations were compared **using only the 9-image validation split carved out
of the 50 training images**. The test partition was never involved in any of these decisions.

| Attempt | Configuration | Best validation macro-F1 | Outcome |
| --- | --- | --- | --- |
| kept | head warm-up 8 epochs (lr 1e-3), then `layer3`+`layer4`+`fc` for up to 30 epochs (lr 1e-4) | 0.567 | **selected** |
| rejected | higher learning rates (lr 3e-4 body / 1.5e-3 head) | collapsed (train accuracy fell to 0.10) | diverged |
| rejected | class-balanced sampling, 6 + 20 epochs | 0.422 | worse on validation |

The kept configuration is the one stored in `config.json`, so re-running reproduces it.

**ROC-AUC** is reported only when it is mathematically defined. A class with a single outcome in
the test set makes one-vs-rest AUC undefined for that class; `metrics.json` records `null` and
`DEMO_RESULTS.md` prints "not available" instead of a number.

---

## 8. Boundaries respected by this experiment

* The 10 test images are not used for training, validation, augmentation statistics, class
  weights, checkpoint selection, early stopping or threshold tuning.
* `evaluate.py` refuses to overwrite an existing `metrics.json`, so the test set is scored once.
  Retraining from scratch and re-scoring is possible with `--force`, and is documented rather
  than silent.
* No metric is hardcoded or estimated. If the experiment did not run, the report says so instead
  of showing placeholder numbers.
* No model binary, dataset file or private local path is committed.
