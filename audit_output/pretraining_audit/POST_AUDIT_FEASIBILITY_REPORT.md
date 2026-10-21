# Post-Audit Class Feasibility Report

**Status: PROVISIONAL — final eligible counts cannot be computed until (a) the label policy is approved, (b) the 75 conflict files are adjudicated, and (c) the 300-file quality review completes.** No class weights or sampling strategies are chosen here (explicitly deferred).

## 1. Original 13-class counts (audited, verified against `image_stats.jsonl`)

Dry AMD 1,360 · Wet AMD 587 · Mild DR 2,441 · Moderate DR 5,368 · Severe DR 1,918 · Proliferate DR 1,504 · Cataract 1,505 · Hypertensive Retinopathy 1,344 · Pathological Myopia 1,265 · Glaucoma 1,878 · Normal Fundus 3,061 · Advanced PDR 114 · Very Severe NPDR 108 — **total 22,453** (Mendeley record says 22,454; discrepancy unresolved, see `PROVENANCE_POLICY.md`).

## 2. Impact of conflicts and quality flags per folder

| Folder | Total | Conflict-involved (quarantine) | Quality-flagged (quarantine) | Provisional eligible | Independent groups (eligible) |
|---|---:|---:|---:|---:|---:|
| 1.Dry AMD | 1,360 | 1 | 0 | 1,359 | 135 |
| 2.Wet AMD | 587 | 1 | 0 | 586 | 46 |
| 3.Mild DR | 2,441 | 0 | 5 | 2,436 | 1,842 |
| 4.Moderate DR | 5,368 | 20 | 60 | 5,303 | 3,338 |
| 5.Severe DR | 1,918 | 7 | 24 | 1,893 | 722 |
| 6.Proliferate DR | 1,504 | 21 | 57 | 1,443 | 481 |
| 7.Cataract | 1,505 | 0 | 9 | 1,496 | 208 |
| 8.Hypertensive Retinopathy | 1,344 | 0 | 0 | 1,344 | 103 |
| 9.Pathological Myopia | 1,265 | 0 | 0 | 1,265 | 166 |
| 10.Glaucoma | 1,878 | 1 | 0 | 1,876 | 204 |
| 11.Normal Fundus | 3,061 | 13 | 80 | 2,973 | 1,951 |
| 12. Advanced PDR | 114 | 5 | 38 | 75 | 75 (all Zenodo singletons) |
| 13. Very Severe NPDR | 108 | 5 | 27 | 78 | 78 (all Zenodo singletons) |

Quality flags and conflicts overlap on some files; the "provisional eligible" column subtracts the union (script of record: `build_registers.py`). **Quality-flag exclusions are provisional**: no image is excluded until the human review confirms it (protocol: `QUALITY_REVIEW_PROTOCOL.md`).

## 3. Under the recommended 5-band primary mapping

| Primary band | Provisional eligible | Groups | Zenodo singletons | Max:min imbalance |
|---|---:|---:|---:|---:|
| No apparent DR | 2,973 | 1,951 | 99 | |
| Mild NPDR | 2,436 | 1,842 | 4 | |
| Moderate NPDR | 5,303 | 3,338 | 53 | **2.2 : 1** (vs Moderate) |
| Severe NPDR | 1,971 | 800 | 235 | 2.7 : 1 |
| PDR | 1,518 | 556 | 112 | 3.5 : 1 |

The merge converts the 13-class problem's **49.7 : 1** ratio into a **≤ 3.5 : 1** problem — a materially more defensible basis for macro-averaged metrics.

## 4. What the imbalance still threatens (no mitigation chosen yet)

- **Secondary 13-class task**: Very Severe NPDR (78 eligible) and Advanced PDR (75) are single-source (Zenodo), ungroupable by patient, ~30–35 % quality-flagged, and contain verified label conflicts. Per-class sensitivity/precision for these two classes will have very wide confidence intervals; a few misclassified images move metrics by whole percentage points. Any headline claim must not rest on them; consider reporting them separately or as "exploratory".
- **Primary task**: after the merge the worst band ratio is 3.5 : 1 — tolerable, but Severe and PDR bands are heavily Zenodo-enriched (235 and 112 of their groups are Zenodo singletons), so band and source remain partially confounded → per-family breakdowns are mandatory in evaluation.
- Class weighting / resampling / focal losses are **deliberately undecided** (deferred per instructions until eligibility is final).

## 5. Missing information required for final eligible counts

1. Label-policy approval (which folders join which primary band; whether quarantined files return after adjudication).
2. Adjudication outcome for the 75 conflict files (specialist or source-author).
3. Human quality-review outcomes for the 300 flagged files (how many are truly unusable).
4. Resolution of 22,454 vs 22,453 (affects only bookkeeping, not feasibility).
5. Prefixed-family grading scheme (affects how "Mild/Moderate/Severe/Proliferate" map onto ICDR bands for ~11k images).
