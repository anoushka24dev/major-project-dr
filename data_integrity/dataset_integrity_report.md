# Dataset integrity report

- Policy: `data-integrity-v1-2026-11-09`
- Generated: 2026-11-09T13:36:41Z after 158.5 s
- Clean manifests: `cleaned_classification_manifest.csv`, `cleaned_segmentation_manifest.csv` (stamped, digest in `.clean_manifest.json`)
- Invalidated artifacts: `artifact_status.json`

## Status

| Area | Status | Evidence |
|---|---|---|
| mendeley provenance | IMPLEMENTED, NOT FULLY VERIFIED | 13 class folder(s) with images; filename families {'prefixed': 11097, 'plain': 10599, 'zenodo_numbered': 757}. The corpus publishes no per-file checksum, so provenance is established from the verbatim folder structure and filename conventions, not cryptographically. |
| mmrdr provenance | BLOCKED | MMRDR-CFP/img not found under C:\Users\anous\OneDrive\Desktop\major_project\datasets\mmrdr |
| aptos2019 provenance | BLOCKED | no APTOS 2019 tree at None |
| ddr provenance | BLOCKED | no DDR tree at C:\Users\anous\OneDrive\Desktop\major_project\datasets\ddr |
| idrid provenance | BLOCKED | no IDRiD tree (registration-gated download not performed) |
| Fabricated masks removed | BLOCKED | 0 fabricated mask(s) rejected; 0 foreign image(s); notes: no official IDRiD training images were found |
| Duplicate audit | FIXED AND VERIFIED | 16 exact SHA-256 group(s) covering 33 file(s); 969 perceptual pair(s) covering 1392 file(s) |
| Patient-level leakage audit | FIXED AND VERIFIED | 7482 patient/eye key(s), 8562 split group(s), 1475 isolation problem(s) |
| Clean manifests | FIXED AND VERIFIED | 15894 included row(s), 7996 exclusion row(s), digest stamped |
| Clean splits | BLOCKED | test: 2168, train: 10120, validation: 2169 |
| Automated tests | IMPLEMENTED, NOT FULLY VERIFIED | `tests/test_data_integrity.py` (see the test run recorded in the handover) |
| Old checkpoints invalidated | FIXED AND VERIFIED | 30 artifact(s) registered as invalid in `artifact_status.json` |

## Counts

- images discovered: **23890**
- images included in the clean cohort: **15894**
- images excluded: **7996**

| Exclusion reason | Images |
|---|---:|
| non-DR ocular condition (glaucoma); not a DR severity grade | 1878 |
| non-DR ocular condition (cataract); not a DR severity grade | 1505 |
| non-DR retinal disease (dry age-related macular degeneration); not a DR severity grade | 1360 |
| different disease entity (hypertensive retinopathy); not DR severity | 1344 |
| non-DR retinal disease (pathological myopia); not a DR severity grade | 1265 |
| non-DR retinal disease (wet age-related macular degeneration); not a DR severity grade | 587 |
| cross_label_duplicate_conflict: a duplicate or patient-linked image carries a different five-class label | 44 |
| near_uniform_image:no_retinal_structure | 13 |

### Per source

| Source | On disk | Included | Status |
|---|---:|---:|---|
| mendeley | 22453 | 14457 | LIKELY |
| mmrdr | 0 | 0 | BLOCKED |
| aptos2019 | 0 | 0 | NOT AVAILABLE |
| ddr | 0 | 0 | NOT AVAILABLE |
| idrid | 0 | 0 | BLOCKED |
| zenodo_v03_reference | 1437 | 0 | VERIFIED |

### Per split and class

| Split | Total | No apparent DR | Mild NPDR | Moderate NPDR | Severe NPDR | PDR |
|---|---:|---:|---:|---:|---:|---:|
| test | 2168 | 471 | 355 | 708 | 21 | 10 |
| train | 10120 | 2121 | 1705 | 3511 | 66 | 80 |
| validation | 2169 | 466 | 364 | 715 | 21 | 24 |

## Cross-source overlap

No image is byte-identical across two dataset roots in this audit. This does **not** prove that no foreign image exists — datasets that were not present cannot be compared.

## Segmentation

- status: `BLOCKED`
- official images: 0
- masks per channel: BLOCKED
- 1:1 pairing: no
- fabricated masks rejected: 0
- foreign images rejected: 0
- no official IDRiD training images were found

No segmentation manifest rows were written. IDRiD is **not** replaced by any other dataset: the official images and MA/HE/EX/SE annotations must be supplied (registration required) before segmentation training is allowed.

## Dataset acquisition failures

- **mmrdr**: BLOCKED — MMRDR-CFP/img not found under C:\Users\anous\OneDrive\Desktop\major_project\datasets\mmrdr
- **aptos2019**: NOT AVAILABLE — no APTOS 2019 tree at None
- **ddr**: NOT AVAILABLE — no DDR tree at C:\Users\anous\OneDrive\Desktop\major_project\datasets\ddr
- **idrid**: BLOCKED — no IDRiD tree (registration-gated download not performed)

## How to reproduce

```
python scripts/audit_datasets.py \
    --fundus-root "C:\Users\anous\Downloads\Diabetic Retinopathy and Retinal Diseases Fundus I" \
    --mmrdr-root "C:\Users\anous\OneDrive\Desktop\major_project\datasets\mmrdr" \
    # --aptos-root PATH   (aptos2019: not supplied)
    --ddr-root "C:\Users\anous\OneDrive\Desktop\major_project\datasets\ddr" \
    --idrid-root "C:\Users\anous\OneDrive\Desktop\major_project\datasets\idrid" \
```
