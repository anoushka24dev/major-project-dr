# Iterative Accuracy Optimization: Methodology & Measured Metrics

This document records every engineering, preprocessing, architectural, and optimization method applied to improve five-class Diabetic Retinopathy (`No apparent DR`, `Mild NPDR`, `Moderate NPDR`, `Severe NPDR`, `PDR`) classification accuracy on the unified multi-source dataset (`mendeley` + `mmrdr` + `aptos2019`, 16,864 included images split strictly by patient/duplicate groups into `11,453` train / `2,457` validation / `2,450` held-out test, plus `504` official MMRDR `ts` external test images).

All checkpoint selection and hyperparameter choices use **only** `train_manifest.csv` and `validation_manifest.csv`. `test_manifest.csv` and `external_test_manifest.csv` remain strictly held out for final evaluation.

---

## Summary Leaderboard (Held-Out Test Partition, n = 2,450)

| Iteration | Architecture & Resolution | Key Methods Changed | Best Val Acc | Held-Out Test Acc | Macro F1 | Cohen's Kappa | Macro ROC-AUC | External MMRDR Acc (n = 504) | Status |
|---|---|---|---|---|---|---|---|---|---|
| **Baseline** (`bf283f4`) | `resnet50` @ `224x224` | `class_weighting: true`, AdamW `lr=3e-4`, `wd=1e-4`, 15 epochs | `0.7721` | `0.7788` (`1908/2450`) | `0.7517` | `0.7110` | `0.9456` | `0.7679` | Completed |
| **Iteration 1** | `convnext_small` @ `384x384` | Dark-border autocrop + disk cache, `class_weighting: false`, Mixup (`0.2`) + CutMix (`0.4`, `mix_prob=0.35`), unscaled `lr=1.5e-4` | `0.7428` (Epoch 2) | — (stopped at Epoch 5) | — | — | — | — | Stopped early (`val_loss` diverged `0.7152 -> 0.7729`) |
| **Iteration 2** | `efficientnet_v2_s` @ `384x384` + 4-view TTA | **Differential LR** (`backbone=7.5e-5`, `head=3e-4`), CutMix/Mixup removed (`mix_prob=0.0`), `label_smoothing=0.05`, `class_weighting: false`, 4-view TTA | `0.8063` (no TTA) / `0.8091` (4-TTA) | `0.8082` (`+2.94%`) | `0.7632` | `0.7434` | **`0.9518`** | `0.7917` | Completed & evaluated |
| **Iteration 3** | `efficientnet_v2_s` @ `512x512` (progressive warm-start) + 4-view TTA | Warm-start from Iteration 2 (`effv2_s_384.pt`), `512x512` resolution, **Model EMA** (`ema_decay=0.5`), `backbone=2.4e-5`, `head=6e-5`, `label_smoothing=0.08` | **`0.8140`** (Epoch 6) | **`0.8110`** (`+3.22%`) | **`0.7738`** | **`0.7471`** | `0.9454` | **`0.8214`** (`+5.35%`) | Completed & evaluated |
| **Iteration 4** | `convnext_tiny` @ `384x384` + 4-view TTA | Cross-family **ConvNeXt-Tiny** (`27.8M`), `backbone=4.5e-5`, `head=3e-4`, `wd=2e-2`, **Model EMA** (`ema_decay=0.5`), `label_smoothing=0.08`, 15 epochs | `0.8095` (Epoch 8) | `0.7935` (`+1.47%`) | `0.7571` | `0.7258` | `0.9413` | `0.7996` (`+3.17%`) | Completed & evaluated (kept for cross-family ensemble; higher `Mild NPDR` recall `0.5801`) |
| **Iteration 4b** | **3-Model Ensemble** (`effv2_s_512` `0.40` + `effv2_s_384` `0.45` + `convnext_tiny_384` `0.15`) + 4-view TTA | Cross-family & multi-resolution probability blending tuned strictly on `validation_manifest.csv` (`0.9805` val top-2 acc) | **`0.8189`** | **`0.8167`** (`+3.79%`) | **`0.7784`** | **`0.7550`** | **`0.9568`** | `0.8095` (`0.9446` AUC) | Completed & evaluated |
| **Iteration 5** | `efficientnet_v2_s` @ `384x384` (`Ben Graham` + `GeM2d` + `OrdinalRetinalLoss`) | `4*(I-Blur(I))+128` local contrast cache (`_bg`), `GeM2d(p=3.0)`, adjacent-grade smoothing `0.10` + expected-grade regression `0.12` | `0.7692` (Ep 5) / `0.7843` (4-TTA) | — (rejected via validation) | — | — | — | — | Stopped early at Ep 7 (`val_loss 0.6975 -> 0.8000`; degraded ensemble) |
| **Iteration 6** | `convnext_tiny` @ `512x512` (progressive warm-start) + 4-view TTA | Warm-start from `convnext_tiny_384.pt`, `512x512` Raw RGB, `OrdinalRetinalLoss` (`ordinal_smoothing=true`, `0.08`, aux `0.06`), `backbone=2.625e-5`, `head=7.5e-5`, `ema=0.5` | `0.8026` (no TTA) / `0.8099` (4-TTA) | `0.8000` (`+2.12%`) | `0.7645` | `0.7334` | `0.9457` | `0.8135` (`0.6825` F1) | Completed & evaluated (improved `convnext_tiny` +`0.65%` test / +`1.39%` ext) |
| **Iteration 6b** | **4-Model Ensemble** (`effv2_s_512` `0.25` + `effv2_s_384` `0.30` + `convnext_tiny_384` `0.05` + `convnext_tiny_512` `0.35`) + 4-TTA | Multi-scale CNN ensemble including `convnext_tiny_512` | **`0.8201`** | `0.8114` (`+3.26%`) | `0.7735` | `0.7479` | **`0.9588`** | `0.8135` (**`0.9448` AUC**) | Completed & evaluated (best AUC `0.9588` & lowest ECE `0.0263`) |

---

## Detailed Explanation of Every Method Used

### 1. Retinal Dark-Border Autocrop (`_crop_dark_borders` in `src/data.py`)
- **What it does**: Computes the tight bounding box of non-dark fundus pixels (`max(R,G,B) > 12`) with a 1% safety margin before aspect-ratio-preserving resize-and-pad (`_resize_and_pad`). Guarded so images without meaningful dark margins (`>= 95%` active area) or degenerate masks remain untouched.
- **Why used**: Prior dataset analysis showed wide black scanner margins around circular fundi (e.g., ~40% wasted border area in `aptos2019`). Trimming empty black padding enlarges the actual retinal surface and lesion pixels at the target input resolution.

### 2. Deterministic Preprocessed Disk Caching (`precache_classification_images` in `src/data.py`)
- **What it does**: Decodes raw source images (`1024px–2658px`), applies dark-border autocrop and resize-and-pad once using `ThreadPoolExecutor`, and stores high-quality (`quality=95, subsampling=0`) images in `cache/cls_{size}_crop{0|1}/`.
- **Why used**: In the baseline run, CPU JPEG decoding and LANCZOS resizing took `~215–293s/epoch`. Caching moves the pipeline to **100% GPU utilization** on the RTX 4060 Laptop GPU (`8 GB` VRAM), cutting epoch time by ~2x–3x (`~118s/epoch` for `efficientnet_v2_s @ 384px`) without changing the exact image transforms.

### 3. Resolution Scaling (`224 -> 384 -> 512`)
- **What it does**: Increases network input grid from `224x224` (`50,176` px) to `384x384` (`147,456` px, ~2.9x pixels) and `512x512` (`262,144` px, ~5.2x pixels).
- **Why used**: In the baseline confusion matrix, **85.8% of all 542 test errors (`465/542`)** occurred at adjacent fine-lesion boundaries (`Mild <-> Moderate` and `Severe <-> PDR`), while `No apparent DR` already exceeded `97.9%` recall. Separating `Mild NPDR` from `Moderate NPDR` depends on detecting 1–3 pixel microaneurysms (`MA`) and small dot haemorrhages (`HE`) that vanish at `224x224`.

### 4. Modern Pretrained Backbones (`build_classifier` in `src/models.py`)
- **What it does**: Extends `src/models.py` beyond `resnet50` (`torchvision` IMAGENET1K_V1) to support `efficientnet_v2_s`, `efficientnet_v2_m`, `convnext_tiny`, `convnext_small`, `convnext_base`, `swin_v2_t`, `swin_v2_s`, `swin_v2_b`, `resnet50_v2`, and `resnet101`, while preserving `build_resnet50` verbatim for backward compatibility and updating `explainability.py` / `evaluate_classification.py` / `cross_dataset.py` to read architecture metadata from checkpoint fingerprints.
- **Why used**: `efficientnet_v2_s` (`20.2M` parameters) is natively engineered for `384px+` inputs via fused-MBConv + progressive stages and achieved `108.9s/epoch` in GPU benchmarking (nearly twice as fast per epoch as `convnext_small`'s `183.2s/epoch`).

### 5. Differential Learning Rates (`backbone_lr_mult` in `src/train_classifier.py`)
- **What it does**: Splits optimizer parameter groups into pretrained backbone tensors (`lr * backbone_lr_mult`) and newly initialized classifier head tensors (`lr`), alongside cosine decay + linear warmup.
- **Why used**: In **Iteration 1** (`convnext_small`), applying an unscaled `lr=1.5e-4` immediately across the newly unfrozen backbone caused validation loss to climb from Epoch 2 onward (`0.7152 -> 0.7729`). Setting `backbone_lr_mult: 0.25` in **Iteration 2** (`backbone=7.5e-5`, `head=3e-4`) enabled smooth fine-tuning without destroying pretrained filters (`val_acc` jumped to `0.7778` on the first unfrozen epoch and `0.8063` by Epoch 4).

### 6. Disabling CutMix/Mixup for Fine Retinal Lesion Preservation
- **What we observed**: Enabling CutMix (`alpha=0.4`) + Mixup (`alpha=0.2`, `mix_prob=0.35`) in **Iteration 1** degraded validation performance. Unlike coarse natural-image categories, Diabetic Retinopathy grading depends on discrete threshold lesions (`MA`, `HE`, neovascularization). Pasting patches or linear pixel blending dilutes small microaneurysms and creates mismatched soft targets. Setting `mix_prob: 0.0` and `random_erasing: 0.0` while using fundus-safe geometric/photometric augmentations (`horizontal_flip`, `rotation_degrees: 15`, `scale_jitter: 0.06`, `translate_frac: 0.02`, mild `brightness/contrast/saturation`) raised validation accuracy immediately.

### 7. Class Weighting Removal & Direct Accuracy Objective (`class_weights_for_mode`)
- **What it does**: Replaced mandatory `1 / count` inverse-frequency class weighting (`0.578` weight on `Moderate NPDR` vs `1.356` on `Mild NPDR`) with configurable `class_weighting` (`false`, `"sqrt"`, or `true`) and set `monitor: "val_accuracy"`.
- **Why used**: Heavy downweighting of `Moderate NPDR` (`34.6%` of dataset) pushed `216` true Moderate cases into `Mild NPDR` in the baseline model. Optimizing unweighted cross-entropy with `label_smoothing: 0.05` (Iteration 2) / `0.08` (Iteration 3) aligns training directly with overall classification accuracy while preventing extreme overconfidence on borderline grades.

### 8. Multi-View Test-Time Augmentation (`predict_probs_with_tta` in `src/models.py`)
- **What it does**: During `evaluate_classification` and `cross_dataset` (`evaluation.tta: "multi"`), averages softmax probability distributions across 4 fundus-preserving views: original image, horizontal flip, 94% center zoom (cropped 3% outer padding and bilinearly resized back), and flipped 94% center zoom.
- **Why used**: Reduces single-view orientation and framing variance at inference time cleanly on GPU without touching ground truth (`+0.28%` verified accuracy lift on validation: `0.8063 -> 0.8091`).

### 9. Progressive Resolution Warm-Start & Model EMA (Iteration 3, Completed)
- **What it does**:
  - `init_from_checkpoint`: Initializes `512x512` training from the best `384x384` checkpoint (`runs/models/effv2_s_384.pt`) with reduced learning rates (`backbone=2.4e-5`, `head=6e-5`).
  - `ema_decay: 0.5`: Maintains an Exponential Moving Average shadow copy of weights across epochs after unfreezing so noisy borderline batches do not destabilize validation accuracy (`Iteration 2` peaked at Epoch 4 before oscillating between `0.771–0.801` as training accuracy reached `0.9907`).
- **Observed Impact**: Validation accuracy rose to `0.8140` (`val_macroF1` reached `0.7844` at Epoch 8). On held-out test, accuracy improved to `0.8110`, macro F1 rose to `0.7738`, and cross-domain generalization on `mmrdr` surged (`0.8111 -> 0.8296` internal MMRDR test; `0.7917 -> 0.8214` and macro F1 `0.5769 -> 0.6887` on the 504-image official MMRDR external test set).

### 10. Cross-Family Architecture Diversity (`convnext_tiny`, Iteration 4, Completed)
- **What it does**: Trains a modern ConvNet (`convnext_tiny`, `7x7` depthwise convolutions, LayerNorm, inverted bottlenecks) from `384x384` cached fundus images using conservative differential learning rates (`backbone=4.5e-5`, `head=3e-4`, `weight_decay=2e-2`), `label_smoothing=0.08`, and `ema_decay=0.5`.
- **Observed Impact**: Adding architectural diversity outside the EfficientNetV2 family for multi-model probability ensembling. While standalone held-out test accuracy (`0.7935`, `1944/2450`) sits below `efficientnet_v2_s @ 512px` (`0.8110`), `convnext_tiny` achieves substantially higher `Mild NPDR` recall (`0.5801` vs `0.5083`) and strong `PDR` detection on external MMRDR (`0.9180` recall, `0.9916` ROC-AUC), making its errors complementary.

### 11. Cross-Family Multi-Resolution Probability Ensembling (Iteration 4b, Completed)
- **What it does**: Averages calibrated 4-view TTA probability distributions across `effv2_s_512` (`0.40`), `effv2_s_384` (`0.45`), and `convnext_tiny_384` (`0.15`), with weights chosen strictly via grid search on `validation_manifest.csv`.
- **Why used**: Combines `512px` micro-lesion detail, `384px` multi-scale receptive field stability, and `ConvNeXt` depthwise spatial priors. Noticeable gains appear in both calibration (`ECE: 0.0451 -> 0.0292`), discrimination (`Macro ROC-AUC: 0.9454 -> 0.9568`), and overall held-out accuracy (`0.8110 -> 0.8167`).
- **Diagnostic Note on Error Floor**: On `validation_manifest.csv`, this ensemble reaches **`98.05%` top-2 accuracy** and **`98.78%` ±1 grade accuracy**, and even on the `2,092/2,457` validation cases where all 3 models **unanimously agree**, exact-label agreement is `86.57%` (`cross_label_conflict_pairs = 815,246` and `879` cross-folder patient ID collisions documented by the dataset manifest audit). Closing remaining clean signal errors therefore requires stronger micro-lesion visual enhancement (Ben Graham local contrast normalization + GeM spatial pooling + Transformer diversity).

### 12. Ben Graham Local Contrast, GeM Pooling & Ordinal Loss Ablation (Iteration 5, Rejected via Validation)
- **What it tested**:
  - **Ben Graham local contrast normalization (`_apply_contrast_enhancement` in `src/data.py`)**: Computes `4 * (I - GaussianBlur(I)) + 128` masked smoothly inside the non-dark retinal interior into a separate `cache/cls_384_crop1_bg` store.
  - **Generalized Mean Pooling (`GeM2d(p=3.0)` in `src/models.py`)**: Replaces `AdaptiveAvgPool2d(1)` so small high-response lesion patches concentrate rather than diluting across 144 grid cells.
  - **`OrdinalRetinalLoss` (`src/train_classifier.py`)**: Adjacent-grade soft target smoothing (`label_smoothing=0.10`) plus SmoothL1 expected-grade regression (`ordinal_loss_weight=0.12`).
- **Observed Impact & Why Rejected**: On `efficientnet_v2_s @ 384px`, validation peaked at Epoch 5 (`0.7692` single-view / **`0.7843`** with 4-TTA, `0.7455` macro F1, `0.7134` kappa), while training accuracy climbed rapidly to `0.9343` and validation loss spiked `0.6975 -> 0.8000` by Epoch 7. Adding this checkpoint to the Iteration 4b ensemble on validation (`weights 0.05..0.20`) degraded accuracy (`0.8189 -> 0.8168–0.8185`). Aggressive `4x` high-pass subtraction discards genuine macular exudate/haemorrhage luminance cues while amplifying JPEG/camera texture noise. We keep `GeM2d` and `OrdinalRetinalLoss` in the codebase as clean modular options, revert preprocessing to Raw RGB + autocrop (`enhance_contrast: "none"`), and move to progressive `512px` `ConvNeXt` + `OrdinalRetinalLoss`.

### 13. Progressive 512px ConvNeXt-Tiny & OrdinalRetinalLoss (Iteration 6, Completed)
- **What it does**: Warm-starts `convnext_tiny` from `runs/models/convnext_tiny_384.pt` at `512x512` on pre-cached Raw-RGB images (`cache/cls_512_crop1`) using `OrdinalRetinalLoss(label_smoothing=0.08, ordinal_smoothing=True, ordinal_loss_weight=0.06)`, differential learning rates (`backbone=2.625e-5`, `head=7.5e-5`), `ema_decay=0.5`, `batch_size=12`, and `grad_accum_steps=2` (early stopped cleanly at Epoch 6).
- **Observed Impact**: Relative to Iteration 4 (`convnext_tiny @ 384px`), held-out test accuracy improved **`0.7935 -> 0.8000`**, macro F1 improved **`0.7571 -> 0.7645`**, Cohen's Kappa rose **`0.7258 -> 0.7334`**, `PDR` ROC-AUC climbed to **`0.9715`**, and official MMRDR external test accuracy rose **`0.7996 -> 0.8135`** (macro F1 **`0.6413 -> 0.6825`**). On validation (`4-TTA`), it achieved `0.8099` (`0.7768` macro F1) and pushed our 4-model cross-family validation ensemble to **`0.8201`** (`0.7893` macro F1, `0.7597` kappa).

---

## Iteration 6 Per-Class & Per-Source Held-Out Test Metrics (`convnext_tiny` @ `512x512`, 4-TTA)

### Overall Held-Out Test (`outputs/classification/classification_metrics.json`, n = 2,450)
- **Accuracy**: `0.8000` (`80.00%`, `1,960 / 2,450`)
- **Macro Precision / Recall / F1**: `0.7772` / `0.7549` / `0.7645`
- **Weighted F1**: `0.7982`
- **Cohen's Kappa**: `0.7334`
- **Macro ROC-AUC**: `0.9457`
- **Expected Calibration Error (ECE, 15 bins)**: `0.0458` (`9.0%` flagged for review under uncertainty thresholds)

| Class | Support | Precision | Recall (Sensitivity) | Specificity | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| `No apparent DR` | 667 | `0.9865` | `0.9835` | `0.9950` | `0.9850` | `0.9987` |
| `Mild NPDR` | 362 | `0.5801` | `0.5304` | `0.9334` | `0.5541` | `0.8770` |
| `Moderate NPDR` | 848 | `0.7576` | `0.8255` | `0.8602` | `0.7901` | `0.9210` |
| `Severe NPDR` | 299 | `0.7525` | `0.7525` | `0.9656` | `0.7525` | `0.9604` |
| `PDR` | 274 | `0.8095` | `0.6825` | `0.9798` | `0.7406` | `0.9715` |

### Breakdown by Source Dataset (`outputs/evaluation/cross_dataset_results.json`)
- **Internal Test — Mendeley** (`n = 1,965`): Accuracy `0.7969`, Macro F1 `0.7720`, Cohen's Kappa `0.7302`
- **Internal Test — MMRDR** (`n = 270`): Accuracy `0.8259`, Macro F1 `0.7013`, Cohen's Kappa `0.7300`
- **Internal Test — APTOS 2019** (`n = 215`): Accuracy `0.7953`, Macro F1 `0.6328`, Cohen's Kappa `0.7009`
- **External Official MMRDR `ts`** (`n = 504`): Accuracy `0.8135`, Macro F1 `0.6825`, Cohen's Kappa `0.7105`, Macro ROC-AUC `0.9286`

---

## Iteration 6b 4-Model Ensemble Held-Out Test Metrics (`0.25 effv2_s_512 + 0.30 effv2_s_384 + 0.05 convnext_tiny_384 + 0.35 convnext_tiny_512`)
- **Validation Accuracy / Macro F1 / Kappa**: **`0.8201`** / **`0.7893`** / **`0.7597`**
- **Held-Out Test Accuracy** (`n = 2,450`): `0.8114` (`81.14%`, `1,988 / 2,450`)
- **Held-Out Test Macro F1 / Kappa / Macro ROC-AUC**: `0.7735` / `0.7479` / **`0.9588`** (highest ROC-AUC achieved)
- **Expected Calibration Error (ECE, 15 bins)**: **`0.0263`** (lowest calibration error; `13.0%` flagged for review)
- **Source Breakdown**: Mendeley `0.8122` | MMRDR internal `0.8148` | APTOS `0.8000` | External MMRDR `ts` `0.8135` (`0.9448` ROC-AUC)
- **Next Observation**: While 4-model blending boosted ROC-AUC (`0.9588`) and calibration (`ECE 0.0263`), higher `convnext_tiny_512` weight slightly softened `Mild <-> Moderate` boundary decisions on test vs Iteration 4b (`0.8167`). Adding a **Vision Transformer (`swin_v2_t`)** in Iteration 7 introduces self-attention inductive bias outside the CNN family.

---

## Iteration 4b Per-Class & Per-Source Held-Out Test Metrics (3-Model Ensemble, 4-TTA)

### Overall Held-Out Test (`outputs/classification/classification_metrics.json`, n = 2,450)
- **Accuracy**: `0.8167` (`81.67%`, `2,001 / 2,450`)
- **Macro Precision / Recall / F1**: `0.7947` / `0.7680` / `0.7784`
- **Weighted F1**: `0.8128`
- **Cohen's Kappa**: `0.7550`
- **Macro ROC-AUC**: `0.9568`
- **Expected Calibration Error (ECE, 15 bins)**: `0.0292` (`13.3%` flagged for review under uncertainty thresholds)

| Class | Support | Precision | Recall (Sensitivity) | Specificity | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| `No apparent DR` | 667 | `0.9851` | `0.9895` | `0.9944` | `0.9873` | `0.9997` |
| `Mild NPDR` | 362 | `0.6416` | `0.5193` | `0.9497` | `0.5740` | `0.9166` |
| `Moderate NPDR` | 848 | `0.7711` | `0.8620` | `0.8645` | `0.8140` | `0.9421` |
| `Severe NPDR` | 299 | `0.7540` | `0.7793` | `0.9647` | `0.7664` | `0.9712` |
| `PDR` | 274 | `0.8217` | `0.6898` | `0.9812` | `0.7500` | `0.9545` |

### Breakdown by Source Dataset (`outputs/evaluation/cross_dataset_results.json`)
- **Internal Test — Mendeley** (`n = 1,965`): Accuracy `0.8158`, Macro F1 `0.7883`, Cohen's Kappa `0.7545`
- **Internal Test — MMRDR** (`n = 270`): Accuracy `0.8222`, Macro F1 `0.6341`, Cohen's Kappa `0.7198`
- **Internal Test — APTOS 2019** (`n = 215`): Accuracy `0.8186`, Macro F1 `0.6834`, Cohen's Kappa `0.7355`
- **External Official MMRDR `ts`** (`n = 504`): Accuracy `0.8095`, Macro F1 `0.6431`, Cohen's Kappa `0.6949`, Macro ROC-AUC `0.9446`

---

## Iteration 4 Per-Class & Per-Source Held-Out Test Metrics (`convnext_tiny` @ `384x384`, 4-TTA)

### Overall Held-Out Test (`outputs/classification/classification_metrics.json`, n = 2,450)
- **Accuracy**: `0.7935` (`79.35%`, `1,944 / 2,450`)
- **Macro Precision / Recall / F1**: `0.7651` / `0.7506` / `0.7571`
- **Weighted F1**: `0.7929`
- **Cohen's Kappa**: `0.7258`
- **Macro ROC-AUC**: `0.9413`
- **Expected Calibration Error (ECE, 15 bins)**: `0.0433` (`10.7%` flagged for review under uncertainty thresholds)

| Class | Support | Precision | Recall (Sensitivity) | Specificity | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| `No apparent DR` | 667 | `0.9791` | `0.9850` | `0.9921` | `0.9821` | `0.9971` |
| `Mild NPDR` | 362 | `0.5722` | `0.5801` | `0.9248` | `0.5761` | `0.8996` |
| `Moderate NPDR` | 848 | `0.7698` | `0.8007` | `0.8733` | `0.7850` | `0.9233` |
| `Severe NPDR` | 299 | `0.7388` | `0.7191` | `0.9647` | `0.7288` | `0.9547` |
| `PDR` | 274 | `0.7657` | `0.6679` | `0.9743` | `0.7135` | `0.9316` |

### Breakdown by Source Dataset (`outputs/evaluation/cross_dataset_results.json`)
- **Internal Test — Mendeley** (`n = 1,965`): Accuracy `0.7903`, Macro F1 `0.7655`, Cohen's Kappa `0.7230`
- **Internal Test — MMRDR** (`n = 270`): Accuracy `0.8148`, Macro F1 `0.6764`, Cohen's Kappa `0.7096`
- **Internal Test — APTOS 2019** (`n = 215`): Accuracy `0.7953`, Macro F1 `0.6181`, Cohen's Kappa `0.7001`
- **External Official MMRDR `ts`** (`n = 504`): Accuracy `0.7996`, Macro F1 `0.6413`, Cohen's Kappa `0.6866`, Macro ROC-AUC `0.9295`

---

## Iteration 3 Per-Class & Per-Source Held-Out Test Metrics (`efficientnet_v2_s` @ `512x512`, 4-TTA)

### Overall Held-Out Test (`outputs/classification/classification_metrics.json`, n = 2,450)
- **Accuracy**: `0.8110` (`81.10%`)
- **Macro Precision / Recall / F1**: `0.7977` / `0.7613` / `0.7738`
- **Weighted F1**: `0.8072`
- **Cohen's Kappa**: `0.7471`
- **Macro ROC-AUC**: `0.9454`
- **Expected Calibration Error (ECE, 15 bins)**: `0.0451` (`10.6%` flagged for review under uncertainty thresholds)

| Class | Support | Precision | Recall (Sensitivity) | Specificity | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| `No apparent DR` | 667 | `0.9851` | `0.9880` | `0.9944` | `0.9865` | `0.9985` |
| `Mild NPDR` | 362 | `0.6073` | `0.5083` | `0.9430` | `0.5534` | `0.9050` |
| `Moderate NPDR` | 848 | `0.7600` | `0.8550` | `0.8571` | `0.8047` | `0.9324` |
| `Severe NPDR` | 299 | `0.7423` | `0.8094` | `0.9609` | `0.7744` | `0.9655` |
| `PDR` | 274 | `0.8939` | `0.6460` | `0.9903` | `0.7500` | `0.9255` |

### Breakdown by Source Dataset (`outputs/evaluation/cross_dataset_results.json`)
- **Internal Test — Mendeley** (`n = 1,965`): Accuracy `0.8076`, Macro F1 `0.7801`, Cohen's Kappa `0.7431`
- **Internal Test — MMRDR** (`n = 270`): Accuracy `0.8296`, Macro F1 `0.6705`, Cohen's Kappa `0.7335`
- **Internal Test — APTOS 2019** (`n = 215`): Accuracy `0.8186`, Macro F1 `0.6844`, Cohen's Kappa `0.7364`
- **External Official MMRDR `ts`** (`n = 504`): Accuracy `0.8214`, Macro F1 `0.6887`, Cohen's Kappa `0.7179`

---

## Iteration 2 Per-Class & Per-Source Held-Out Test Metrics (`efficientnet_v2_s` @ `384x384`, 4-TTA)

### Overall Held-Out Test (`outputs/classification/classification_metrics.json`, n = 2,450)
- **Accuracy**: `0.8082` (`80.82%`)
- **Macro Precision / Recall / F1**: `0.7775` / `0.7551` / `0.7632`
- **Weighted F1**: `0.8026`
- **Cohen's Kappa**: `0.7434`
- **Macro ROC-AUC**: `0.9518`
- **Expected Calibration Error (ECE, 15 bins)**: `0.0257` (`16.1%` flagged for review under uncertainty thresholds)

| Class | Support | Precision | Recall (Sensitivity) | Specificity | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| `No apparent DR` | 667 | `0.9793` | `0.9910` | `0.9921` | `0.9851` | `0.9996` |
| `Mild NPDR` | 362 | `0.6444` | `0.4807` | `0.9540` | `0.5506` | `0.9086` |
| `Moderate NPDR` | 848 | `0.7681` | `0.8632` | `0.8620` | `0.8129` | `0.9387` |
| `Severe NPDR` | 299 | `0.7552` | `0.7324` | `0.9670` | `0.7436` | `0.9571` |
| `PDR` | 274 | `0.7405` | `0.7080` | `0.9688` | `0.7239` | `0.9549` |

### Breakdown by Source Dataset (`outputs/evaluation/cross_dataset_results.json`)
- **Internal Test — Mendeley** (`n = 1,965`): Accuracy `0.8071`, Macro F1 `0.7754`, Cohen's Kappa `0.7427`, Macro ROC-AUC `0.9519`
- **Internal Test — MMRDR** (`n = 270`): Accuracy `0.8111`, Macro F1 `0.5728`, Cohen's Kappa `0.7011`, Macro ROC-AUC `0.9443`
- **Internal Test — APTOS 2019** (`n = 215`): Accuracy `0.8140`, Macro F1 `0.6735`, Cohen's Kappa `0.7290`, Macro ROC-AUC `0.9402`
- **External Official MMRDR `ts`** (`n = 504`): Accuracy `0.7917`, Macro F1 `0.5769`, Cohen's Kappa `0.6640`

### Error Concentration Remaining (`470` errors / `2,450` test images)
- `Mild NPDR <-> Moderate NPDR`: `179 + 94 = 273` errors (**58.1%** of all errors)
- `Severe NPDR <-> PDR`: `57 + 62 = 119` errors (**25.3%** of all errors)
- All other class pairs combined: `78` errors (**16.6%** of all errors, only `3.18%` of test images)

