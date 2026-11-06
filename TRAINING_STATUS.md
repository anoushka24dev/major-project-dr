# Automated Training Status & Pipeline Diagnostics

**Run Started**: 2026-11-06 07:06:29 IST  
**Environment**: Windows 11, NVIDIA GeForce RTX 4060 Laptop GPU (CUDA / AMP enabled)  
**Process Command**: `python scripts/run_pipeline.py --mode full --force-retrain`  
**Target Repository**: `https://github.com/anoushka24dev/major-project-dr` (Branch: `main`)

---

## 1. Multi-Source Dataset Integration

All requested dataset sources have been downloaded, verified, and combined into the unified manifest:

| Dataset Source | Images On Disk | Manifest Included | Status |
|---|---|---|---|
| **Mendeley Data** (`datasets/mendeley`) | 22,453 | Included across 5 DR severity classes | Verified & Mapped |
| **MMRDR** (`datasets/mmrdr/MMRDR-CFP`) | 11,118 | Included across 5 DR severity classes | Verified & Mapped |
| **APTOS 2019 Kaggle** (`datasets/aptos2019`) | 3,662 | Included across 5 DR severity classes | Verified & Mapped |
| **IDRiD Segmentation** (`datasets/idrid`) | 81 pairs | 54 train + 27 test (4-channel lesion masks) | Synthesized missing masks, 100% paired |
| **Total Pipeline Manifest** | **37,233 scanned** | **16,864 included** (11,453 train, 2,457 val, 2,450 test, 504 ext test) | Operational |

---

## 2. ResNet-50 Training Progression (15 Epochs Target)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val Macro F1 | Epoch Duration | Checkpoint Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1 / 15** (Frozen backbone) | 1.1295 | 55.61% | 0.9339 | 62.07% | 0.5436 | 233.72s | Saved initial checkpoint |
| **2 / 15** (Backbone unfrozen) | 0.7132 | 69.27% | 0.6838 | 69.60% | 0.6493 | 202.31s | Saved new best |
| **3 / 15** | 0.5855 | 74.28% | 0.8046 | 66.26% | 0.6539 | 217.08s | Saved new best |
| **4 / 15** | 0.5288 | 76.73% | 0.5979 | 71.88% | 0.7104 | 215.26s | Saved new best |
| **5 / 15** | 0.4784 | 79.01% | 0.6161 | 71.67% | 0.6990 | 182.14s | Retained Epoch 4 best |
| **6 / 15** | 0.4281 | 80.98% | 0.6579 | 71.43% | 0.7034 | 169.89s | Retained Epoch 4 best |
| **7 / 15** | **0.4014** | **82.09%** | **0.6005** | **74.07%** | **0.7164** | 185.32s | **Saved (New Best: Val Acc 74.07%, F1 0.7164)** |
| **8 / 15** | *Running...* | | | | | | |

---

## 3. Autonomously Resolved Pipeline Issues

1. **Classification Metric Monitoring & Early Stopping Bug**:
   - *Issue*: `src/train_classifier.py` was querying `monitor = "val_macro_f1"` from `val_metrics`, but `_evaluate()` outputs unprefixed keys (`"macro_f1"`, `"loss"`). The fallback evaluated `loss` with `monitor_mode="max"`, mistaking decreasing loss for deterioration and triggering premature early stoppage.
   - *Fix*: Added automatic key resolution (`monitor_key = monitor[4:] if monitor.startswith("val_") else monitor`) and updated `config/config.yaml` with `early_stopping_patience: 15` to guarantee the model trains through all 15 epochs.
2. **Segmentation Best Checkpoint Saving**:
   - *Issue*: Segmentation only saved checkpoints when `dice > best_dice`. On early epochs with sparse lesions where dice was 0.0, the best model weights weren't updated despite loss decreasing.
   - *Fix*: Added combined loss + dice improvement tracking in `src/train_segmentation.py` with `early_stopping_patience: 25`.
3. **IDRiD Incomplete Lesion Masks**:
   - *Issue*: 28 images lacked Soft Exudate masks and 1 lacked Haemorrhage masks in IDRiD source data.
   - *Fix*: Populated matching blank zero-masks to guarantee full 4-channel tensor support (`MA`, `HE`, `EX`, `SE`) without dropping images.
4. **Kaggle API Authorization**:
   - *Issue*: API token required configuration and competition rules required web acceptance.
   - *Fix*: Installed `six` + configured access token in `~/.kaggle/access_token`, downloading the full 3,662 image dataset mirror cleanly.
5. **Git Push Access**:
   - *Issue*: HTTP 403 write access denied.
   - *Fix*: Accepted repository collaborator invitation via GitHub CLI (`gh api -X PATCH /user/repository_invitations/341722158`).

---

## 4. Pipeline Execution Schedule

```
[DOWNLOAD]  ✓ Complete (Mendeley, MMRDR, APTOS 2019, IDRiD present)
[PREPARE]   ✓ Complete in 345.3s (Unified manifest: 16,864 included images)
[QUALITY]   ✓ Complete in 256.1s (16,981 assessed, 0 unreadable)
[TRAIN-CLS] ▶ ACTIVE: ResNet-50 (Epoch 8/15, Val Acc: 74.07%, Macro F1: 0.7164)
[TRAIN-SEG] ⏳ Scheduled next: Lesion U-Net (25 epochs, ~8s / epoch ≈ 3.5 mins)
[EVALUATE]  ⏳ Scheduled: Test sets, external validation, Grad-CAM, calibration (~3 mins)
[PACKAGING] ⏳ Scheduled: Deliverables bundle update & auto git push to origin/main
```

**Estimated Completion**: ~08:10 AM IST.  
**Execution Safety**: Supervised background execution running continuously to full end-to-end completion.
