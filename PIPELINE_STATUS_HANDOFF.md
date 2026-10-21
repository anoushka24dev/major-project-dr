# Diabetic Retinopathy Data Pipeline — Status Handoff

**Project:** A Multimodal Explainable Agentic AI Framework for Trustworthy Diabetic Retinopathy Diagnosis from Fundus Images  
**Stage:** Dataset preparation / data-pipeline (TDD layer)  
**Date:** 14 Oct 2026 (session resumed from 13 Oct checkpoint)  
**Workspace:** `C:\Users\anous\OneDrive\Desktop\major_project`

---

## 1. What was re-verified this session

The session was resumed after a disconnect. I did **not** start over. I re-established the working state and re-ran the checks that matter most, without touching the dataset or the audit artifacts.

- **Manifest + pipeline source files re-read:** `dataset/labels.py`, `dataset/eligibility.py`, `dataset/manifest.py`, `dataset/grouping.py`, `dataset/statistics.py`, `dataset/decoding.py`, `dataset/split_validation.py`, plus the audit build scripts and reports.
- **Pytest suite re-run:** `cd "C:\Users\anous\OneDrive\Desktop\major_project" && python -m pytest tests/ -q`
  - Result: **73 passed in 55.88s**, exit code 0.
  - No failures, no errors, no skips reported by that run.
- **Image-decode probe re-run:** `tests/test_image_decoding.py::test_real_image_smoke_test`
  - Result: **1 passed in 1.08s**, exit code 0.
  - This is a real-file decode probe against one actual image on disk, not only a synthetic fixture test.
- **Audit artifact file checks:** the two provisional manifests exist and are non-empty on disk:
  - `audit_output/pretraining_audit/PROVISIONAL_ELIGIBILITY_PRIMARY.csv` — 8,129,904 bytes, last written 13 Oct 2026 15:27
  - `audit_output/pretraining_audit/PROVISIONAL_ELIGIBILITY_SECONDARY.csv` — present, 7,721,474 bytes

I also re-read the reconciliation findings and the dataset audit report so the numbers below are grounded in the current files, not in earlier memory.

---

## 2. Current test-portrait lights

This is the table the tests currently assert against the manifests and pipeline helpers.

### 2.1 Manifest loading and integrity

| Test (file) | What it checks |
|---|---|
| `test_manifest.py::test_primary_manifest_loads` | Primary CSV loads as a list of 22,453 rows |
| `test_manifest.py::test_secondary_manifest_loads` | Secondary CSV loads as a list of 22,453 rows |
| `test_manifest.py::test_primary_has_required_columns` | Primary row carries the full required schema plus `_resolved` |
| `test_manifest.py::test_primary_no_duplicate_image_paths` | No `image_path` repeats in the primary manifest |
| `test_manifest.py::test_missing_required_column` | A CSV missing a required column raises `ManifestValidationError` |
| `test_manifest.py::test_missing_file_raises_when_require_files_exist` | Missing image paths raise `ImagePathDoesNotExistError` when flagged |
| `test_manifest.py::test_primary_missing_files_integration` | Every primary path resolves to an existing file on disk |
| `test_manifest.py::test_secondary_missing_files_integration` | Every secondary path resolves to an existing file on disk |
| `test_manifest.py::test_summarize_primary` | Primary summary: 22,453 rows; 14,201 `eligible_provisional` |
| `test_manifest.py::test_summarize_secondary` | Secondary summary: 22,453 rows; 22,127 `eligible_provisional` |

### 2.2 Label mapping and task taxonomy

| Test (file) | What it checks |
|---|---|
| `test_label_mapping.py::test_source_label_set_has_13_members` | 13 source labels in the verbatim taxonomy |
| `test_label_mapping.py::test_primary_mapping_keys_are_known_source_labels` | Primary mapping keys are the 7 DR-related source labels |
| `test_label_mapping.py::test_primary_mapping_values` | Primary mapping resolves to No apparent DR / Mild NPDR / Moderate NPDR / Severe NPDR / PDR, with the two Zenodo merges applied |
| `test_label_mapping.py::test_secondary_mapping_is_verbatim` | Secondary task label == original folder label for all 13 |
| `test_label_mapping.py::test_unknown_source_label_raises` | Unknown source labels raise on secondary mapping |
| `test_label_mapping.py::test_unknown_or_non_primary_label_raises` | Non-primary labels (e.g. non-DR folders) raise on primary mapping |
| `test_label_mapping.py::test_non_dr_labels` | Non-DR label set is exactly the 6 non-DR folders |
| `test_label_mapping.py::test_primary_eligible_source_labels` | Primary-eligible source label membership matches the mapping |
| `test_label_mapping.py::test_validate_source_labels_passes` | The known 13 labels pass validation |
| `test_label_mapping.py::test_validate_source_labels_fails` | A bogus label fails validation |
| `test_label_mapping.py::test_primary_band_counts_from_rows` | Band counting from already-labeled rows works deterministically |

### 2.3 Eligibility and quarantine

| Test (file) | What it checks |
|---|---|
| `test_eligibility.py::test_primary_eligible_count` | 14,201 primary-eligible rows |
| `test_eligibility.py::test_primary_eligible_are_only_eligible_provisional` | Every primary-eligible row is marked `eligible_provisional` |
| `test_eligibility.py::test_primary_excludes_non_dr` | Non-DR excluded rows = 7,926 + 9 + 4 (primary) |
| `test_eligibility.py::test_non_dr_excluded_values` | Those rows only use known excluded-non-DR eligibility values |
| `test_eligibility.py::test_primary_quality_quarantine_count` | 242 rows in `quarantined_quality_review` (primary) |
| `test_eligibility.py::test_primary_conflict_quarantine_count` | 22 rows in `quarantined_label_conflict` (primary) |
| `test_eligibility.py::test_primary_both_quarantine_count` | 49 rows in `quarantined_conflict_and_quality` (primary) |
| `test_eligibility.py::test_primary_quarantine_rows_are_excluded_from_eligible` | Quarantined rows do not pass the primary-eligible filter |
| `test_eligibility.py::test_primary_excluded_rows_remain_traceable` | Excluded rows = 22,453 − 14,201; they are preserved, not deleted |
| `test_eligibility.py::test_is_any_quarantined_flag_logic` | `is_any_quarantined` matches the two yes/no flag columns |
| `test_eligibility.py::test_secondary_eligible_count` | 22,127 secondary-eligible rows |
| `test_eligibility.py::test_secondary_excludes_quarantined` | Secondary quarantine buckets: 251 / 26 / 49 |
| `test_eligibility.py::test_secondary_eligible_are_only_eligible_provisional` | Secondary-eligible rows are only `eligible_provisional` |

### 2.4 Class-count reporting

| Test (file) | What it checks |
|---|---|
| `test_class_counts.py::test_primary_eligible_band_counts` | Primary eligible bands = No apparent DR 2,973 / Mild 2,436 / Moderate 5,303 / Severe 1,971 / PDR 1,518; total 14,201 |
| `test_class_counts.py::test_primary_eligible_class_counts_excludes_quarantined` | Quarantined rows are not counted into eligible bands |
| `test_class_counts.py::test_primary_status_counts` | Full primary status counts match the stored buckets exactly |
| `test_class_counts.py::test_primary_excluded_and_quarantined_counts` | Eligible / quarantined / excluded decomposition matches |
| `test_class_counts.py::test_secondary_eligible_class_counts` | Secondary eligible counts = 13 classes, total 22,127, with the expected band numbers |
| `test_class_counts.py::test_secondary_eligible_excludes_quarantined` | Secondary eligible counts match a recomputed eligible-only tally |
| `test_class_counts.py::test_secondary_eligible_class_counts_allows_zero_count_classes_flag` | `allow_zero_count_classes` control behaves as intended |

### 2.5 Grouping metadata

| Test (file) | What it checks |
|---|---|
| `test_grouping.py::test_rows_by_group_id` | `split_group_id` grouping covers every primary row |
| `test_grouping.py::test_rows_by_duplicate_group_id` | Duplicate-group grouping is non-empty and every group has ≥1 row |
| `test_grouping.py::test_known_group_ids` | Known duplicate group IDs are non-empty strings |
| `test_grouping.py::test_group_membership_map` | Duplicate-group membership map is non-empty |
| `test_grouping.py::test_group_membership_map_is_deterministic` | Membership map is stable across calls |
| `test_grouping.py::test_has_patient_group_key` | Patient-key detection matches expected yes/no cases |
| `test_grouping.py::test_zenodo_unlinked_rows` | 757 Zenodo rows have no patient group key |
| `test_grouping.py::test_summarize_grouping` | Grouping summary reports positive split/duplicate/zenodo counts |
| `test_grouping.py::test_grouping_summary_matches_manual_counts` | Summary counts agree with the direct grouping helpers |

### 2.6 Split-validation contract

| Test (file) | What it checks |
|---|---|
| `test_split_validation.py::test_valid_assignment` | A normal assignment validates cleanly |
| `test_split_validation.py::test_unknown_path_rejected` | Unknown paths raise `SplitValidationError` |
| `test_split_validation.py::test_double_assignment_rejected` | One image in two partitions raises |
| `test_split_validation.py::test_invalid_partition_rejected` | Unknown partition name raises |
| `test_split_validation.py::test_quarantined_assigned_rejected` | Quality-quarantined images cannot be assigned |
| `test_split_validation.py::test_conflict_quarantined_assigned_rejected` | Conflict-quarantined images cannot be assigned |
| `test_split_validation.py::test_both_quarantined_assigned_rejected` | Combined quarantine cannot be assigned |
| `test_split_validation.py::test_excluded_non_dr_assigned_rejected` | Excluded non-DR images cannot be assigned (primary) |
| `test_split_validation.py::test_duplicate_group_must_stay_in_one_partition` | Duplicate-connected images cannot be split across partitions |
| `test_split_validation.py::test_duplicate_group_in_same_partition_ok` | Duplicate-connected images in the same partition are allowed |
| `test_split_validation.py::test_report_partition_counts` | The validation report returns correct per-partition counts |
| `test_split_validation.py::test_group_integrity_only_applies_to_assigned_members` | Unassigned members of a group do not trigger a violation |

### 2.7 Image decoding

| Test (file) | What it checks |
|---|---|
| `test_image_decoding.py::test_load_rgb_returns_pil_image` | `load_rgb` returns a PIL image |
| `test_image_decoding.py::test_load_rgb_dimensions` | Dimensions are returned correctly |
| `test_image_decoding.py::test_load_rgb_grayscale_converts` | Grayscale input is converted to RGB |
| `test_image_decoding.py::test_load_rgb_rgba_converts` | RGBA input is converted to RGB |
| `test_image_decoding.py::test_load_rgb_jpeg_converts` | JPEG input is converted to RGB |
| `test_image_decoding.py::test_load_report` | `load_report` returns the expected metadata dict |
| `test_image_decoding.py::test_to_numpy_rgb_shape_and_dtype` | `to_numpy_rgb` returns `(H, W, 3)` uint8 |
| `test_image_decoding.py::test_decode_error_on_missing` | Missing files raise `DecodeError` |
| `test_image_decoding.py::test_decode_error_on_not_a_file` | Directory paths raise `DecodeError` |
| `test_image_decoding.py::test_decode_error_on_corrupt_bytes` | Corrupt bytes raise `DecodeError` |
| `test_image_decoding.py::test_real_image_smoke_test` | A real dataset image decodes and converts to RGB successfully |

---

## 3. Audit checkpoint that the tests are pinned to

These are the numbers encoded in the current provisional manifests and reaffirmed by the tests above.

### 3.1 Dataset-level facts (from audit, read-only)

- **22,453 readable JPEG images** across **13 class folders** that exactly match the expected category list.
- Every file decoded successfully in the earlier audit; **0 corrupt/unreadable files**.
- Three source families:
  - `prefixed`: 11,097 files (likely)
  - `plain`: 10,599 files (likely)
  - `zenodo_v02`: 757 files (confirmed, byte-identical to Zenodo record 4647952)
- Provenance: Mendeley Data DOI `10.17632/6ry9n79gb4.1`; the 757 Zenodo images are already fully inside the dataset and must not be added again.

### 3.2 Primary task (5-band DR grading), provisional manifest

- **14,201 eligible_provisional**
- Quarantined:
  - `quarantined_quality_review`: **242**
  - `quarantined_label_conflict`: **22**
  - `quarantined_conflict_and_quality`: **49**
- Excluded non-DR:
  - `excluded_non_dr_primary`: **7,926**
  - `excluded_non_dr_primary_and_quality_quarantine`: **9**
  - `excluded_non_dr_primary_and_label_conflict_quarantine`: **4**
- Total: **22,453** ✓
- **Post-merge primary bands (eligible only):**
  - No apparent DR: **2,973**
  - Mild NPDR: **2,436**
  - Moderate NPDR: **5,303**
  - Severe NPDR (Severe DR + Very Severe NPDR merged): **1,971**
  - PDR (Proliferate DR + Advanced PDR merged): **1,518**

### 3.3 Secondary task (13-class), provisional manifest

- **22,127 eligible_provisional**
- Quarantined:
  - `quarantined_quality_review`: **251**
  - `quarantined_label_conflict`: **26**
  - `quarantined_conflict_and_quality`: **49**
- Total: **22,453** ✓

### 3.4 Duplicate / conflict / quality-flag facts from audit

- **Exact duplicates:** 16 groups, 33 files (all Zenodo-derived `.jpg`, all within a single class).
- **Strong near-duplicate tier:** 149 pairs / 205 unique files (dHash ≤ 4 and aHash ≤ 6).
- **Cross-label conflict files:** 75 unique files in 26 connected components → quarantined from both tasks.
- **Quality-flagged images:** 300 files with non-empty `quality_flags` in the provenance manifest → quarantined pending human review.
- **False-equivalence note from reconciliation:** the old “300 vs 291” wording was wrong; 300 is the quality-flag count, 291 was the wrong sum. The true quarantine-union is 326 images (251 Q-only + 26 C-only + 49 both), and the manifests reconcile to 22,453.

### 3.5 Zenodo limitation

- **503 eligible Zenodo-derived images** in both manifests have **no patient linkage**; they are grouped at image level only (447 split groups among them, max group size 3).
- Any future evaluation must not claim patient independence for those images.

---

## 4. Read-only assurance

No code this session wrote to any of the following:

- the downloaded dataset folder
- `audit_output/pretraining_audit/PROVISIONAL_ELIGIBILITY_PRIMARY.csv`
- `audit_output/pretraining_audit/PROVISIONAL_ELIGIBILITY_SECONDARY.csv`
- `audit_output/pretraining_audit/PROVENANCE_MANIFEST.csv`
- `audit_output/pretraining_audit/LABEL_CONFLICT_REGISTER.csv`
- `audit_output/pretraining_audit/QUALITY_REVIEW_REGISTER.csv`
- `audit_output/image_stats.jsonl` or any other audit artifact

The rerun was a **read-and-rerun** of pytest and one image-decode probe. The manifests and dataset were only read.

---

## 5. Limitations and blockers

### 5.1 Things this stage explicitly does not yet do

- **No split has been created.** Split-group IDs exist in the manifests, but no train/val/test assignment exists.
- **No model has been trained.** There are no performance numbers to report.
- **No patient-independent split exists yet**, and one is not guaranteed for the Zenodo subset.

### 5.2 Open decisions recorded in the audit artifacts

- **Label policy is provisional (Option A, 2026-10-13).** The two merges — Very Severe NPDR → Severe NPDR, Advanced PDR → PDR — are evidence-tagged but not specialist-signed-off. The Normal Fundus wording “no DR visible according to the source label” is recorded; “healthy” must not be claimed.
- **300 quality-flagged images are quarantined pending human review.** None have been visually reviewed; the review register still says `visual_review_performed = no` for every row.
- **75 cross-label conflict files are quarantined** from both tasks; no adjudication has happened.
- **33 exact-duplicate files (16 groups):** keep-both-in-grouped-split vs single-copy exclusion is still an open decision.
- **Two “likely” provenance families** still have unresolved licence / source-identity questions for any future publication.

### 5.3 One concrete reconciliation finding still open

The reconciliation findings record **13 mis-bucketings in the primary manifest** — 9 quality-flagged images placed in an excluded-non-DR bucket instead of `quarantined_quality_review`, and 4 conflict-flagged images placed in an excluded-non-DR bucket instead of `quarantined_label_conflict`. Their quarantine flags are still visible, and the grand total still reconciles to 22,453, but the stored reason breakdown understates the true Q-only (251) and C-only (26) counts in the primary manifest. The two clean resolution options are:

1. accept the manifests as-is and document the understated reason counts explicitly, or
2. rebuild the manifests with the bucket logic corrected so every flagged image’s quarantine reason is named even when it is also non-DR excluded.

That decision affects presentation vs rebuild, not discovery of new images.

### 5.4 Environment quirks encountered this session

- The project’s `tests/` discovery was slow through one shell path, so I ran pytest once through `python -m pytest` and the single decode probe separately. The counts above are from an actual run, not from a dry run.
- I avoided re-running anything that would re-scan the full image corpus this session; the relevant evidence already exists in `audit_output/`.

---

## 6. Where the task stands against the original success criteria

The original task was: establish a small, tested, reproducible data-pipeline foundation with the audit checkpoint intact, without prematurely starting model training.

Against that:

- **Data integrity:** manifests load, have the required schema, have no duplicate paths, and resolve to real files on disk; pytest asserts all of that.
- **Label mapping:** the 13-class source taxonomy, the primary 5-band mapping (with the two provisional merges), the secondary verbatim mapping, and unknown/non-primary rejection are all encoded and tested.
- **Quarantine enforcement:** eligibility filters, quarantine flag logic, and the quarantined-row exclusion from eligible counts are implemented and tested.
- **Image decoding:** `load_rgb` / `to_numpy_rgb` / error paths are implemented and tested; a real image decode path was re-verified this session.
- **Duplicate-group handling:** duplicate-group membership and grouping summaries are implemented and tested; the split-validator treats duplicate-connected groups as atomic.
- **Split-validation rules:** unknown paths, double assignment, invalid partitions, quarantined/excluded assignment, and duplicate-group splitting are all enforced and tested.
- **Audit checkpoint intact:** 73 pytest tests pass, the real-image decode probe passes, and no manifest/dataset/audit file was modified.

So the current state is: **a tested, read-only data-pipeline layer sitting on top of an intact provisional audit checkpoint**, with the remaining work being decisions and reviews rather than more pipeline plumbing.

---

## 7. Suggested next steps

These are options, not instructions. Pick whichever matches what you want to do next.

- **[finish_pipeline_glue]** Add the small remaining pipeline glue and/or a single reproducibility runner that recomputes the key manifest counts from the CSV files and compares them to the audit checkpoint, so the “tests pass” story also has an explicit checkpoint-reconciliation artifact.
- **[resolve_primary_misbucketings]** Decide and act on the 13 primary-manifest mis-bucketings: either document them as-is in the audit narrative, or rebuild the manifests with corrected quarantine-bucketing logic and re-verify.
- **[plan_quality_review]** Turn the quality-review protocol into a concrete first tranche — for example, review the 10 near-uniform / 21 low-texture images plus a small control sample — and record outcomes in a reviewed copy of the register.
- **[freeze_split_policy]** Lock the split-readiness decisions (grouping policy, split ratio, Zenodo test cap, sealing plan) so the next step can be a real split-build step rather than another assessment.

---

*This handoff is a derived status note. It does not modify any manifest, dataset file, or audit artifact.*
