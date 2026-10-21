# Dataset Audit Report
## "A Multimodal Explainable Agentic AI Framework for Trustworthy Diabetic Retinopathy Diagnosis from Fundus Images" — Phase 1 dataset audit

- **Audited folder:** `C:\Users\anous\Downloads\Diabetic Retinopathy and Retinal Diseases Fundus I`
- **Date:** 12 October 2026
- **Scope:** audit only — no files were renamed, moved, deleted, modified, or generated inside the dataset.
- **Evidence tags:** **[V]** = verified evidence (reproducible measurement or official documentation) · **[S]** = automated suspicion requiring human review · **[U]** = unresolved question.

---

## 1. Executive summary

The dataset contains **22,453 readable JPEG images in 13 class folders that exactly match your expected category list**. Every file decoded successfully, every folder in the Mendeley zip extracted (22,453 = 22,453 entries) **[V]**, and the collection is confirmed to be a merge of **three public sources**, one of which is your Zenodo record — **all 757 Zenodo v0.2 images are already byte-identically embedded** in this dataset **[V]**. A follow-up comparison against the newer **Zenodo v0.3** (1,437 images) shows the dataset used exactly v0.2, that 12 of the 757 were later dropped by Zenodo, that 680 v0.3 images are new content the dataset lacks, and that the cross-label duplication **originates inside the Zenodo source itself** (§8) **[V]**.

However, the audit found issues that should be resolved or at least consciously acknowledged **before** any train/val/test split:

1. **236 files (1.05%) participate in duplicate relationships** (16 exact-duplicate groups; 205 files in high-confidence near-duplicate pairs) — 204 of them inside the 757-file Zenodo subset **[V]**.
2. **73 files appear (near-)identically under two different class labels** — e.g., the same pixels are labelled both *Normal Fundus* and *Advanced PDR* — verified by pixel-level comparison (MAE ≈ 0, correlation = 1.000) **[V]**. This is label noise, and it will leak across any random split.
3. **A subset of images is near-blank or has lost colour**: 10 flat mid-grey images, 64 desaturated images, 111 low-variance images, 231 very-dark exposures **[V/S]**.
4. **Patient identifiers exist but are partial**: two of the three sources support patient-level grouping; the Zenodo 757 carry no recoverable patient/laterality linkage — confirmed against both published Zenodo versions and the Data in Brief paper (anonymized image numbers, no patient IDs) **[V]**.
5. **Class imbalance is 49.7 : 1** (Moderate DR 5,368 vs Very Severe NPDR 108), and the two smallest classes are single-source, single-resolution subsets that already contain duplicates **[V]**.
6. **Source artifacts are label-correlated** (resolution, grey padding, black-border framing differ by source, and sources map to specific classes) — a serious shortcut-learning risk for a "trustworthy AI" claim **[V]**.

**Verdict: the dataset is usable for Phase 1, but not split-ready as-is.** See §10 for the precise conditions.

---

## 2. Class counts and percentages

Exact original folder names preserved (note the **space after the dot** in `12.` and `13.`) **[V]**:

| # | Folder name (verbatim) | Images | % of total |
|---|---|---:|---:|
| 4 | `4.Moderate DR` | 5,368 | 23.91 % |
| 11 | `11.Normal Fundus` | 3,061 | 13.63 % |
| 3 | `3.Mild DR` | 2,441 | 10.87 % |
| 5 | `5.Severe DR` | 1,918 | 8.54 % |
| 10 | `10.Glaucoma` | 1,878 | 8.36 % |
| 7 | `7.Cataract` | 1,505 | 6.70 % |
| 6 | `6.Proliferate DR` | 1,504 | 6.70 % |
| 1 | `1.Dry AMD` | 1,360 | 6.06 % |
| 8 | `8.Hypertensive Retinopathy` | 1,344 | 5.99 % |
| 9 | `9.Pathological Myopia` | 1,265 | 5.63 % |
| 2 | `2.Wet AMD` | 587 | 2.61 % |
| 12 | `12. Advanced PDR` | 114 | 0.51 % |
| 13 | `13. Very Severe NPDR` | 108 | 0.48 % |
| | **Total** | **22,453** | **100 %** |

- **Largest:** `4.Moderate DR` (5,368) · **Smallest:** `13. Very Severe NPDR` (108) · **Ratio 49.7 : 1** **[V]**
- All 13 expected categories present; **no missing, unexpected, or merged folders** **[V]**
- **No nested directories** inside any class folder; **no non-image files** (no README, labels CSV, or license file inside the dataset) **[V]**

---

## 3. Total number of images

- **22,453 files**, all JPEG: 21,696 `.jpeg` + 757 `.jpg` **[V]**
- The Mendeley zip contains exactly **22,453 entries** → extraction is complete **[V]**
- The Mendeley record's description states **22,454** images → **off-by-one discrepancy between the published count and the actual archive** **[V]** (cause unknown **[U]**)
- Top-level folder name appears truncated (`…Fundus I`); the zip's internal top-level name is identical, so the truncation is upstream in the Mendeley archive itself, not an extraction error **[V]**

---

## 4. Image-integrity findings

**Readability**
- **22,453 / 22,453 decoded successfully; 0 corrupt or unreadable files** (double-checked with `verify()` + full load) **[V]**
- Format: 100 % JPEG · Colour mode: 100 % `RGB` at the codec level (no palette, greyscale-mode or CMYK files) **[V]**

**Dimensions**
- 241 distinct resolutions; range **336 × 314 … 2,124 × 2,056** **[V]**
- Family profiles (see §8 for why families matter):
  - `512 × 512`: 10,813 files (all class-prefixed files + all Normal-Fundus plain files)
  - `1024 × H`: mostly Mild/Moderate plain files
  - `2124 × 2056`: 756 of 757 Zenodo files — **one outlier `5.Severe DR/411.jpg` is 1,444 × 1,444** (re-saved/cropped variant) **[V]**
  - Three tiny files at **336 × 314**: `4.Moderate DR/39106_right.jpeg`, `5.Severe DR/Severe DR39106_left_0_4705.jpeg`, `5.Severe DR/Severe DR39106_left_0_6506.jpeg` (17–19 KB) — smallest in dataset; same patient ID 39106 across two folders **[V]**
- Median file size 118 KB; smallest 17 KB **[V]**

**Suspicious-image flags (heuristic; nothing removed)** — full list in [flagged.txt](audit_output/flagged.txt)

| Flag | Count | Assessment |
|---|---:|---|
| Very dark (`lum_mean < 30`) | 231 | 172/231 have reddish retinal centre → dark exposure but genuine fundus **[V]**; **58 have low-structure centres → review [S]** |
| Near-grayscale (saturation < 18) | 64 | Neutral mid-grey centres (e.g. `3.Mild DR/21149_left.jpeg` R=G=B≈127.9) → desaturated/non-standard colour; visually confirm **[S]** |
| Near-uniform (`lum_std < 12`) | 10 | Flat mid-grey, centre std 6–11 → **near-blank; strongest removal candidates after human confirmation** **[S]** |
| Low texture | 21 | Mix of dark-but-valid fundus (13) and low-structure centres (8, incl. `12. Advanced PDR/701–704,710`) **[S]** |
| Black border anomaly | 3 | `12. Advanced PDR/746.jpg`, `13. Very Severe NPDR/549,551.jpg` **[S]** |
| Low contrast overall (`lum_std < 20`) | 448 | Concentrated in Zenodo subset (348/757) — dark/flat exposures **[V]**, needs review for usability **[S]** |
| Grey #808080 canvas corners | 97.3 % of plain, 39.3 % of prefixed, 0 % of Zenodo | Systematic preprocessing artifact, not per-image corruption **[V]** |

- **Caveat [U]:** screenshot tooling was unavailable in this environment, so flagged images were characterised numerically (colour/structure statistics + ASCII colour maps), not by eye. Open [sample_review.html](audit_output/sample_review.html) to visually review 20 labelled samples yourself.

---

## 5. Duplicate findings

**Exact duplicates (SHA-256)** **[V]**
- **16 groups, 33 files** — full list in [exact_dup_groups.txt](audit_output/exact_dup_groups.txt)
- **All within a single class** (no exact cross-class duplicates)
- All 16 groups come from the Zenodo-derived `.jpg` files, and **the same 16 groups already exist inside the original Zenodo zip** (`1. No DR signs/18–20.jpg`; 10 pairs in `3. Moderate NPDR`; 5 pairs in `4. Severe NPDR`) → duplication originates in the source dataset, not in the merge **[V]**

**Near-duplicates (perceptual: dHash ≤ 4 **and** 256-bit aHash ≤ 6, pixel-verified on samples)** **[V]**
- **149 pairs / 205 unique files** in the strong tier; combined with exact dups: **236 files (1.05 % of dataset)** involved in some duplicate relationship — list in [near_dup_pairs.txt](audit_output/near_dup_pairs.txt)
- Distribution: **54 pairs cross class labels** (73 files), 82 pairs within class across different patient IDs, 13 pairs within the same patient **[V]**
- By source family: **Zenodo 204/757 files (27 % of that subset)**, prefixed 28, plain 4 **[V]**
- Pixel-level verification on random samples: MAE 0.00–6.0, correlation 0.92–1.000 → same underlying photograph, some re-encoded **[V]**
- Verified cross-class examples (in [strong_cross_class_pairs.txt](audit_output/strong_cross_class_pairs.txt)):
  - `11.Normal Fundus/110.jpg` ≡ `12. Advanced PDR/743.jpg` (MAE 0.00, r = 1.000)
  - `13. Very Severe NPDR/547.jpg` ≡ `5.Severe DR/443.jpg`
  - `4.Moderate DR/243.jpg` ≡ `6.Proliferate DR/577.jpg` (and 579, 580)
  - `1.Dry AMD/Dry AMD1280_left_0_8225.jpeg` ↔ `10.Glaucoma/Glaucoma1280_left_0_6908.jpeg` (r = 0.939, colour-shifted copy)
  - `10.Glaucoma/Glaucoma2041_left_0_8337.jpeg` ↔ `2.Wet AMD/Wet AMD2041_left_0_7926.jpeg` (r = 0.923)

**Not counted as duplicates [V/S]:** 28,815 looser cross-class pairs (dHash ≤ 4 only) were mostly *hash degeneracy among dark images*, not confirmed similarity — they are listed in [near_dup_pairs.txt](audit_output/near_dup_pairs.txt) with distances but were excluded from counts above.

**Limitations [U]:** thresholds are conservative — augmented/rotated/heavily re-processed copies (common in the prefixed family) can escape detection; a dedicated augmentation-aware check would be needed for completeness.

---

## 6. Patient-identity and leakage risks

The filenames reveal **three distinct identifier systems** (matching three sources) **[V]**:

| Family | Pattern | Count | Patient grouping possible? |
|---|---|---:|---|
| Prefixed | `<Class><ID>_<left\|right>_0_<n>.jpeg` | 11,097 | **Yes, within class**: group key = (class, numeric ID), covers both eyes (e.g. Dry AMD 92/135 IDs have L+R). 1,335 eye-groups hold ≥ 5 images (distribution 1/2/5/6/7/12 images per eye) **[V]** |
| Plain | `<ID>[_left\|right].jpeg` | 10,599 | **Yes, within folder**: L/R pairs share an ID (both-eye IDs: Normal 1,022, Mild 595, Moderate 1,996) **[V]** |
| Zenodo | `<N>.jpg` | 757 | **No**: IDs 1…757 are contiguous and disjoint per class (Normal 1–187, Mild 188–191, Moderate 192–271, Severe 272–447, VS-NPDR 448–555, PDR 556–643, AdvPDR 644–757), one file per number, **no laterality, no cross-file patient linkage** **[V] / [U]** |

**Leakage risks, ordered by severity:**

1. **Cross-label duplicate leakage [V]:** the 73 files that exist under two labels will land in different splits under any random split, giving near-perfect metric inflation and contradictory supervision.
2. **Same-patient both-eyes leakage [V]:** without group splitting, left/right eyes of one patient will straddle train/test in the prefixed (e.g. 700+ both-eye IDs) and plain families.
3. **Multiple photos per eye [V]:** prefixed eye-groups hold up to 12 photos (different photos of the same eye — within-group correlation only 0.07–0.82, i.e. genuine re-shoots, not crops). Random splitting *will* leak.
4. **Zenodo patient linkage unknown [U]:** 757 images cannot be grouped by patient; two eyes of one patient may exist as two numbers. Cannot be guaranteed patient-independent.
5. **Cross-folder ID collisions are NOT same-patient [V]:** 232 (ID, side) pairs appear in two folders (Normal↔Mild/Moderate): 0/232 share dimensions and 0/232 exceed correlation 0.7 even after flip/rotation tests → different images; the ID spaces of different folders are independent. (Prefixed cross-class ID sharing is ambiguous: 46/49 checked pairs are different images, but 2 verified pairs are the same image — see §5.)
6. **Source-artefact shortcut risk [V]:** resolution/framing/padding predict the source, and the source predicts parts of the label (all Advanced PDR and Very Severe NPDR are 2,124 × 2,056 black-bordered Zenodo files; all prefixed files are 512 × 512; 97 % of plain files carry grey canvas corners). A model can score well by learning *source style* instead of pathology — an evaluation-validity threat even without classic patient leakage.

**Is patient-independent splitting possible?** **Partially, yes**: for the 21,696 `.jpeg` files, group by `(folder, numeric ID)` (which unions both eyes and all sessions) and additionally cluster duplicate/near-duplicate files into the same group. For the 757 Zenodo files it is **not possible from the available information** — treat them as independent single images and state that limitation, or exclude them from patient-level claims **[U]**.

---

## 7. Label-definition concerns

- Folder names were verified against your list: **all 13 present, spelling preserved** — including the non-standard **`6.Proliferate DR`** (the Mendeley record itself writes *"Proliferative"*; the folder spelling "Proliferate" is a dataset-side typo) **[V]**.
- **No per-image ground truth ships with the dataset** (no labels CSV); folder membership *is* the label **[V]**. Folder names are **not** evidence of medical validity **[V]** — the Mendeley description provides **no definitions, no grading standard, and no per-source labelling protocol** **[V]**.
- **Definitions found in documentation/research [V]:**
  - **Very Severe NPDR** = ETDRS DRSS level **53E** — "≥ 2 level 53A–D characteristics" (i.e. accumulation beyond severe NPDR's 4-2-1 criteria). Recognised in **ETDRS**, *not* in the 5-grade ICDR scale.
  - **Advanced PDR** = ETDRS levels **81/85** — "view partially obscured by VH or PRH from NV, or macula-involving retinal detachment". Also *not* an ICDR grade.
  - **PDR** begins at ETDRS level 61; ICDR defines PDR simply as "neovascularisation and/or preretinal/vitreous hemorrhage".
  - Source: ASRS clinical practice guideline (ETDRS table), https://pmc.ncbi.nlm.nih.gov/articles/PMC8297841/ ; review https://pmc.ncbi.nlm.nih.gov/articles/PMC9800497/
- **Ambiguity/overlap flags (do not merge — recording only):**
  1. The 13-class taxonomy **mixes scales**: ICDR-style (Mild/Moderate/Severe/Proliferative) alongside ETDRS sub-grades (Very Severe NPDR, Advanced PDR) **[V]**.
  2. **`6.Proliferate DR` ⊃ `12. Advanced PDR` clinically**: "advanced PDR" is a *subset severity of PDR*, so the two folders are not mutually exclusive by definition; they are only separated because one source (Zenodo) exported its own 7-grade scheme (PDR 88 vs Advanced PDR 114) while other sources contributed a single "proliferative" grade that likely *includes* advanced eyes **[S]**.
  3. **`5.Severe DR` vs `13. Very Severe NPDR`**: same conceptual band (severe NPDR) split across folders only for one source; images from ICDR-graded sources cannot be "very severe" under their own protocol **[S]**.
  4. **Cross-source grade definitions differ** (Zenodo 7-class ETDRS-like; plain family almost certainly Kaggle/EyePACS 5-class; prefixed family unknown) → the same folder merges non-equivalent clinical criteria **[S/U]**.
  5. **Demonstrated label conflicts**: 73 files carry two different labels at once (§5) — direct evidence that folder labels are not reliably ground truth **[V]**.
  6. Folder names are not medically validated by any shipped annotation file; nothing in the archive confirms ophthalmologist review per image **[U]**.

---

## 8. Dataset provenance and potential overlap

**Aggregator [V]:** Mendeley Data DOI **10.17632/6ry9n79gb4.1** — *"Diabetic Retinopathy and Retinal Diseases Fundus Image Dataset"*, Md Samiul Islam & Israt Jahan, published 23 Sep 2025, CC BY 4.0. Description: *"22,454 high-quality color fundus images from three publicly available sources"* — **the three sources are not named** in the record (verified against both the record page and DataCite metadata).

**Source A — Zenodo, Castillo Benítez et al., *"Dataset from fundus images for the study of diabetic retinopathy"* [VERIFIED]:**
- Concept DOI **10.5281/zenodo.4532360**; CC BY 4.0; Paraguay (Hospital de Clínicas, Zeiss VISUCAM 500, 2124×2056); ETDRS-based 7-grade scheme; Data in Brief paper: PMC8257963.
- **Version 0.2** (record 4647952, 757 JPGs): **all 757 `.jpg` files in the Mendeley dataset are SHA-256 identical to all 757 images in your local v0.2 zip**, with **100 % per-file label agreement** under the folder renames No DR signs → `11.Normal Fundus`, Mild NPDR → `3.Mild DR`, Moderate NPDR → `4.Moderate DR`, Severe NPDR → `5.Severe DR`, Very Severe NPDR → `13. Very Severe NPDR`, PDR → `6.Proliferate DR`, Advanced PDR → `12. Advanced PDR` **[V]**.
- **Conclusion: your Zenodo v0.2 contribution is already fully inside the dataset — do not add it again; doing so would create 757 duplicates** **[V]**. Mendeley and Zenodo are therefore **not independent** datasets **[V]**.

**Zenodo version 0.3 (record 4891308) — the newer 1,437-image zip you supplied [VERIFIED]:**
- v0.3 (June 2021) grew the collection 757 → 1,437 images and **re-encoded every file**: **0 of 1,437 are byte-identical to the dataset**, yet pixel-level comparison shows **745 of the dataset's 757 `.jpg` are visually identical to a v0.3 image** (MAE ≈ 0.13–0.26/255, r ≥ 0.9997 — JPEG re-encode noise only) **[V]**.
- **12 dataset files have no v0.3 counterpart** (verified absent, best-candidate normalized correlation 0.81–0.95): `13. Very Severe NPDR/449–458.jpg` (a contiguous block of 10) and `5.Severe DR/445–446.jpg` — v0.3 appears to have dropped/replaced them **[V]** / why is unknown **[U]**.
- **680 of the 1,437 v0.3 images are NOT in the Mendeley dataset** (new content: No DR signs 493, Severe NPDR 54, Moderate NPDR 32, Advanced PDR 31, Very Severe NPDR 41, PDR 27, Mild 2) — Mendeley merged v0.2, not v0.3 **[V]**. Zero v0.3 image matches any of the 21,696 `.jpeg` files, so v0.3 adds nothing to sources B/C **[V]**.
- `Annotations of the classifications.xlsx` inside the v0.3 zip: 1,437 rows, **100 % consistent with the folder labels**, no duplicates, no patient/laterality column **[V]**.
- **v0.3 contains the label-noise problem itself**: 24 pixel-identical clusters put the *same image under two different class labels inside v0.3* (125 files; e.g. `1345.jpg`=Advanced PDR ≡ `1285.jpg`=PDR, `613.jpg`=No DR signs ≡ `1423.jpg`=Advanced PDR; strict proof via 44 shared dataset twins) — **the cross-label duplication originates in the Zenodo source, not in the Mendeley merge** **[V]**.
- Same-image label disagreements between v0.3 and the dataset: **100 verified pairs / 66 dataset files / 66 v0.3 images** (e.g. the image that v0.3 files as `PDR` the dataset files as `4.Moderate DR`, 18 pairs). 62 of the 66 dataset files were already flagged as cross-label duplicates inside the dataset; only 4 are new (`11.Normal Fundus/43.jpg`, `/165.jpg`, `4.Moderate DR/241.jpg`, `6.Proliferate DR/622.jpg`) **[V]**.
- **Patient linkage: v0.3 does NOT resolve it.** The Data in Brief paper states *"The 'Image' attribute is the number of the anonymized and renamed patient image"* and discloses no patient count; the workbook has no patient ID — image numbers are sequential per file, not patient codes **[V]**. The §6 limitation (757 images cannot be grouped by patient) **stands**.
- **Do not merge v0.3 into the dataset as-is**: it would add 680 genuinely new images but also 745 re-encoded duplicates of existing files plus its own cross-label conflicts; if used, take only the 680 new images after adjudicating their labels, and note the CC BY 4.0 attribution **[V]**.

**Source B — plain `ID_left/right.jpeg` files (10,599: Normal 2,874 / Mild 2,437 / Moderate 5,288) [SUSPICION]:**
- Filename style matches **Kaggle EyePACS "Diabetic Retinopathy Detection"** (documented example `1_left.jpeg`); class counts line up closely with EyePACS training labels (Mild 2,443 vs 2,437; Moderate 5,292 vs 5,288) but Normal would be a **subsample** (2,874 of 25,810) **[S]**. APTOS (3,662 files, different counts) does not fit **[V]**. Not confirmed by any documentation **[U]**.

**Source C — class-prefixed files (11,097 across 8 classes) [SUSPICION]:**
- Best lead: Kaggle **"Retinal Fundus Images"** (`kssanjaynithish03/retinal-fundus-images`), published per-class counts matching ours for Dry AMD 1,360, Wet AMD 587, Cataract 1,505, Hypertensive Retinopathy 1,344, Pathological Myopia 1,265, Glaucoma 1,878, Proliferative 1,416 — with small mismatches for Mild (2,438 vs 2,437) and Severe (1,791 vs 1,742) **[S]**. The exact filename template could not be confirmed from public pages **[U]**. The Mendeley "Eye Disease Image Dataset" (s9bfhswzjb) was checked and rejected (different classes/counts) **[V]**.

**Overlap within the merged dataset [V]:** proven at image level in §5 — overlap exists (236 files duplicated, 73 cross-label) **within** the merge; between the sources themselves no direct comparison was possible beyond Zenodo (sources B and C originals were not available locally to hash against) **[U]**.

---

## 9. Unresolved questions

1. **Identity of sources B and C** — only documented as "three publicly available sources". Suggest the simplest completion: download the two candidate public datasets (Kaggle EyePACS DR detection; Kaggle retinal-fundus-images) and hash-match — same method that proved Source A.
2. **The 22,454 vs 22,453 off-by-one** — which file the Mendeley record counts that the zip lacks (or a description typo).
3. **Patient linkage for the 757 Zenodo images** — **resolved as unrecoverable**: the paper confirms anonymized, renamed image numbers and no published patient count; neither v0.2 nor v0.3 carries patient/laterality metadata **[V]**.
4. **Whether prefixed IDs are globally unique** — 47/49 cross-class ID collisions are different images (suggesting per-class numbering), yet 2 verified cross-class pairs are the same image under a matching ID.
5. **Grading standard actually used per source** — no documentation ties any label to ETDRS/ICDR/Kaggle protocol; the Zenodo paper's own grading citation could not be retrieved (PMC blocked).
6. **The 300 flagged images** — human visual confirmation needed (start with [sample_review.html](audit_output/sample_review.html) and [flagged.txt](audit_output/flagged.txt)); screenshot automation was unavailable in this environment.
7. **Whether the plain/prefixed colour anomalies are augmentation, compression, or capture artefacts** (desaturated centres, grey canvases, hue-mixed interiors) — visual review required.
8. **License compatibility for publication** — Mendeley CC BY 4.0 and Zenodo CC BY 4.0 confirmed **[V]**; sources B/C licenses to confirm when identified.

---

## 10. Recommendation on readiness for the next phase

**Ready for Phase 1 (modelling work can begin) — but NOT split-ready as-is.** Concretely:

**Do before splitting [all audit-consistent, no re-audit needed]:**
1. **Never add the Zenodo v0.2 zip on top** — it is already 100 % inside (§8). **Do not add v0.3 wholesale either** — only its 680 genuinely new images, after label adjudication (§8).
2. **Human-review the ~300 flagged files** (10 near-blank + 64 desaturated + 58 low-structure dark = priority ≈ 132; the rest of the dark/low-contrast flags are probably valid fundus). Record keep/drop decisions; do not delete yet.
3. **Resolve the 73 cross-label duplicate files** — they need a single adjudicated label or exclusion; otherwise they poison both training and evaluation.
4. **Split with grouping**: group key = `(folder, numeric ID)` for the 21,696 `.jpeg` files, plus union-find over the 236 exact/near-duplicate files so no duplicate cluster and no patient crosses splits.
5. **State the Zenodo limitation openly** in your methodology: 757 images cannot be grouped by patient.
6. **Plan for source-shortcut evaluation**: report metrics per source family (or per resolution/framing bucket) so accuracy gains can be shown not to come from artefacts.

**Defer (as you requested) until after review of these counts:**
- Sampling/weighting strategy for the 49.7 : 1 imbalance (the two smallest classes are 114/108 images, single-source, and already contain duplicate/conflicting files — their validation metrics will be unstable; cross-validation will likely be needed).

**Environment limitations to know:** OpenCV was unavailable (Pillow + NumPy used); browser screenshots failed in this environment (visual review must be done by you via `sample_review.html`); perceptual-hash thresholds are conservative and may miss heavily augmented duplicates; all identifier analysis is filename-based because the archive ships no metadata file.

---

### Appendix — audit artifacts (all read-only outputs, in `audit_output/`)

| File | Contents |
|---|---|
| [image_stats.jsonl](audit_output/image_stats.jsonl) | Per-image record: SHA-256, dimensions, mode, luminance/saturation/border/edge stats, perceptual hashes (22,453 rows) |
| [flagged.txt](audit_output/flagged.txt) | 300 suspicious images with flags and metrics |
| [exact_dup_groups.txt](audit_output/exact_dup_groups.txt) | 16 exact-duplicate groups |
| [near_dup_pairs.txt](audit_output/near_dup_pairs.txt) | 74,586 candidate pairs with dHash/aHash distances |
| [strong_cross_class_pairs.txt](audit_output/strong_cross_class_pairs.txt) | 54 high-confidence cross-label duplicate pairs |
| [sample_review.html](audit_output/sample_review.html) | 20 labelled sample images for visual review |
| [v03_manifest.json](audit_output/v03_manifest.json) | Zenodo v0.3 per-image SHA-256 + annotations workbook dump |
| [v03_matches.json](audit_output/v03_matches.json) | 1,093 perceptual candidate pairs (v0.3 ↔ dataset) |
| [v03_pairs_verified.json](audit_output/v03_pairs_verified.json) | All 1,093 pairs pixel-verified (MAE, r, same-image flag) |
| [v03_internal_clusters.json](audit_output/v03_internal_clusters.json) | 117 same-image clusters inside v0.3 (24 cross-label) |
| [v03_compare.py](audit_output/v03_compare.py), [v03_perceptual.py](audit_output/v03_perceptual.py), [v03_pixel_verify.py](audit_output/v03_pixel_verify.py), [v03_final_quantify.py](audit_output/v03_final_quantify.py), [v03_internal.py](audit_output/v03_internal.py) | Reproducible v0.3 comparison scripts |
