# Provisional Eligibility Manifests — README

**Files:** `PROVISIONAL_ELIGIBILITY_PRIMARY.csv` (5-band DR grading task) and `PROVISIONAL_ELIGIBILITY_SECONDARY.csv` (13-class task). Built by `build_provisional_manifests.py` (exit 0, 2026-10-13) from `PROVENANCE_MANIFEST.csv` + `LABEL_CONFLICT_REGISTER.csv`, implementing the Option A approval with its seven safeguards.

## Provisional ≠ approved final dataset

| A **provisional eligibility manifest** is… | An **approved final dataset** would require… |
|---|---|
| A bookkeeping snapshot of which images *may* enter an experiment under today's approved decisions | The same table after (a) the human quality review of the 300 quarantined files completes, (b) the 75 conflict files are adjudicated or permanently excluded, (c) the two provisional merges (VSNPDR→Severe, AdvPDR→PDR) get specialist/source-documentation sign-off, and (d) a versioned split is created and sealed |
| Rebuildable at any time; changing it changes nothing on disk | A frozen, hashed artifact that results are pinned to |
| Carries `manifest_status = PROVISIONAL - NOT AN APPROVED FINAL DATASET` on every row | Its own approval record and hash |
| **No train/val/test assignment exists in it** (a `split_group_id` is the grouping key for a *future* split, not a split) | Split partitions + sealing per AC-9 |

## Columns

`manifest_status`, `manifest_version`, `approval_reference` — provenance of the manifest itself.
`image_path`, `original_label` — verbatim dataset facts (never modified).
`task_label`, `task_label_basis` — primary: 5-band label with its evidence tag (incl. `PROVISIONAL_MERGE` markers and the required Normal-Fundus wording "no DR visible according to the source label"); secondary: original folder label preserved verbatim.
`source_family`, `source_status` — from the provenance manifest (confirmed / likely).
`patient_group_key` — `(folder, ID)` for `.jpeg` families; `NONE` for Zenodo `.jpg` (no linkage exists).
`split_group_id` — union-find of patient keys over duplicate-component edges (`SG-#####`); the unit that a future split must keep intact. Zenodo files begin as singletons; 447 groups (503 eligible files) remain pure-Zenodo, i.e. grouped at image level only.
`duplicate_group_id` — register component (`DUP-G###`) when the file has one.
`quarantine_label_conflict`, `quarantine_quality_flag` — independent yes/no flags (a file can carry both).
`eligibility` — exactly one of: `eligible_provisional`, `quarantined_label_conflict`, `quarantined_quality_review`, `quarantined_conflict_and_quality`, `excluded_non_dr_primary` (primary only).

## Reconciliation (verified at build time, recomputed after rebuild)

### Primary (5-band)
- eligible_provisional: **14,201**
- quarantined_quality_review (Q-only, in DR cohort): **251**
- quarantined_label_conflict (C-only, in DR cohort): **26**
- quarantined_conflict_and_quality (both flags, in DR cohort): **49**
- excluded_non_dr_primary (no quarantine flags): **7,926**
- excluded_non_dr_primary_and_quality_quarantine: **9**
- excluded_non_dr_primary_and_label_conflict_quarantine: **4**
- excluded_non_dr_primary_and_both_quarantines: **0**
- **total = 14,201 + 251 + 26 + 49 + 7,926 + 9 + 4 = 22,453**

Reconciliation interpretation:
- **Quarantine reason totals (union across the whole manifest): Q-only 251 + C-only 26 + both 49 = 326.**
- The **300** figure is a quality-flag count (non-empty `quality_flags` in `PROVENANCE_MANIFEST.csv`); it is NOT the same as the union of quarantine reasons.
- The old **242 + 49 = 291** figure was the wrong sum to compare against 300.
- The 9 quality-only and 4 conflict-only images are **still excluded from the DR cohort** (non-DR folders). The fix is that their quarantine flags and reasons now appear explicitly in the eligibility cell rather than being silently nested inside `excluded_non_dr_primary`.

### Secondary (13-class)
- eligible_provisional: **22,127**
- quarantined_quality_review: **251**
- quarantined_label_conflict: **26**
- quarantined_conflict_and_quality: **49**
- **total = 22,127 + 251 + 26 + 49 = 22,453**

### Cross-manifest consistency
- Both manifests have 22,453 rows; the two path-sets are identical; both are identical to `PROVENANCE_MANIFEST.csv`.
- For every image, `PROVENANCE_MANIFEST.csv.quality_flags` non-empty ⇔ `quarantine_quality_flag == yes` in both manifests (bit-for-bit, 0 mismatches).
- Secondary is unchanged by this rebuild and remains correct.
- Every image has exactly one eligibility row per task; no image was deleted, renamed, relabeled, or moved; quarantined files remain on disk.

## Rules encoded (approvals #3, #4, #5)

- All 75 cross-label conflict files are quarantined from **both** tasks (their labels are, by definition, the affected ones; per-file label-specific quarantine is possible later after adjudication).
- All 300 quality-flagged files are quarantined pending documented human review (`QUALITY_REVIEW_PROTOCOL.md`); flags are review triggers, not defect verdicts.
- Non-DR folders never appear as eligible in the primary manifest, but keep full rows (with quarantine flags still visible) so accounting reconciles to 22,453.
