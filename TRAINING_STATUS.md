# Complete Pipeline Run & Benchmark Summary

**Execution Status**: COMPLETED (Exit Code 0)  
**Finished**: 2026-11-12 02:21 IST  
**Hardware Used**: NVIDIA GeForce RTX 4060 Laptop GPU (CUDA + AMP Enabled)  
**Target Repository**: `https://github.com/anoushka24dev/major-project-dr` (Branch: `main`)

---

## 1. Multi-Source Dataset Integration

All data sources were unified, deduplicated, and split without cross-partition contamination:

| Source | Raw Images | Included in Manifest | Role & Mapping |
|---|:---:|:---:|---|
| **Mendeley Data** | 22,453 | 13,104 | 5-class severity grades (Normal, Mild, Moderate, Severe, PDR) |
| **MMRDR-CFP** | 11,118 | 2,314 | 5-class color fundus photographs + official test partition |
| **APTOS 2019 (Kaggle)** | 3,662 | 1,446 | 5-class fundus images (Grades 0 to 4) |
| **IDRiD Segmentation** | 54 paired sets | 54 paired sets (38/8/8) | 4-channel pixel lesion masks (MA, HE, EX, SE) |
| **Unified Manifest** | **37,233 scanned** | **16,864 included** | **11,453 train, 2,457 val, 2,450 test, 504 ext test** |

---

## 2. Final Diagnostic Performance

### A. 5-Class Disease Grading (ResNet-50)
- **Architecture**: Pretrained ResNet-50 (Cosine learning rate schedule with 1-epoch warmup, AdamW optimizer, AMP)
- **Total Epochs Trained**: 15 Full Epochs (best validation macro F1 `0.7451` at epoch 13)
- **Held-Out Test Set**: 2,450 images

| Metric | Result | Clinical & Methodological Meaning |
|---|:---:|---|
| **Accuracy** | **77.88%** | Multi-class overall diagnosis accuracy |
| **Macro F1 Score** | **0.7517** | Balanced metric across rare and severe stages |
| **Weighted F1 Score** | **0.7823** | Support-weighted classification F1 |
| **Cohen's Kappa ($\kappa$)** | **0.7110** | Substantial multi-class inter-rater agreement |
| **Macro ROC-AUC (One-vs-Rest)** | **0.9456 (94.6%)** | Robust separation across all 5 DR stages |
| **Expected Calibration Error (ECE)** | **0.1297** | 15 equal-width confidence bins; `6.82%` (167/2,450) flagged for review |
| **External MMRDR Test (504 imgs)** | **76.79% acc / 0.9220 AUC** | Official external partition (`0.6420` macro F1, `0.6433` kappa) |
| **Model Weights** | `outputs/classification/best_resnet50.pt` | Verified & saved in deliverables |

### B. 4-Channel Retinal Lesion Segmentation (U-Net)
- **Architecture**: 4-level U-Net (`UNet4`, 32 base channels, 7,763,140 parameters, `512x512`)
- **Target Lesions**: Microaneurysms (MA), Haemorrhages (HE), Hard Exudates (EX), Soft Exudates (SE)
- **Total Epochs Trained**: 25 Full Epochs (retrained after fixing palette `P` red-mask luminance decoding & aspect-ratio padding in [`resize_and_pad_mask`](src/data.py))
- **Best Validation / Test Results**: Best validation macro Dice **`0.0362`** (epoch 6); held-out test (`8` IDRiD images) macro Dice **`0.0074`**, macro IoU **`0.0037`** (`HE: 0.0177`, `EX: 0.0102`, `MA: 0.0016`, `SE: 2.77e-10`)
- **Model Weights**: `outputs/segmentation/best_lesion_unet.pt` | Verified & saved in deliverables

### C. Explainability & Trustworthiness
- **Grad-CAM**: Generated activation maps for 15 examples (`outputs/explainability/gradcam/`) from `layer4.2` (`Bottleneck`) and evaluated lesion overlap against 8 held-out external IDRiD pairs (`mean union Dice 0.0778`, `mean union IoU 0.0413`).
- **Calibration & Review Policy**: Reliability diagram saved in `outputs/evaluation/reliability_diagram.png`; low-confidence / high-entropy predictions (`6.82%`) routed for human review (`43.71%` accuracy when flagged vs `80.38%` when not flagged).
- **Cross-Dataset Generalization**: Per-source and official external MMRDR evaluation saved in `outputs/evaluation/cross_dataset_results.json`.

---

## 3. Submission Deliverables

All deliverables have been packaged into [`deliverables_for_submission/`](deliverables_for_submission/):
1. **`model_checkpoints.zip`** (346.5 MB):
   - `best_resnet50.pt` (282.7 MB — ResNet-50 5-class classification)
   - `best_lesion_unet.pt` (93.3 MB — 4-channel lesion U-Net)
2. **`evaluation_reports_and_metrics.zip`** (20.6 MB):
   - Comprehensive metrics, confusion matrices, ROC curves, Grad-CAM overlays, and manifest audits.
3. **`final_report.html`**:
   - Interactive self-contained audit and benchmark report.
4. **`execution_summary.json`**:
   - Machine-readable stage-by-stage audit with exit code 0.
5. **`metrics_summary.csv`**:
   - Tabular summary of all test evaluations.
