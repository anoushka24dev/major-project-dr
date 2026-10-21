# LABEL POLICY DRAFT

# ⚠️ PROVISIONALLY APPROVED (Option A — 2026-10-13, recorded in §11) — STILL NOT AN APPROVED FINAL DATASET

The user approved **Option A** with seven safeguards (§11). The policy remains a DRAFT in the parts marked unresolved: **no mapping has been confirmed by a retinal specialist or source author; no per-image clinical adjudication has ever occurred; and nothing here authorizes training.** The provisional eligibility manifests built from these decisions (`PROVISIONAL_ELIGIBILITY_PRIMARY.csv` / `_SECONDARY.csv`) are explicitly **not** an approved final dataset — see `ELIGIBILITY_MANIFEST_README.md`.

Scope: **Primary task = DR severity grading** (5-band candidate, §6). **Secondary task = 13-class retinal-condition classification** (original taxonomy preserved). The two tasks keep separate manifests (§10).

## 1. What documentation actually exists (evidence base)

| Source | What it says | What it does NOT say |
|---|---|---|
| **Mendeley record** (DOI 10.17632/6ry9n79gb4.1; full description retrieved this cycle) | *"…22,454 high-quality color fundus images from three publicly available sources… categorized into 13 clinically significant classes, including various stages of DR (Mild, Moderate, Severe, Proliferative, Advanced PDR, Very Severe NPDR)…"* | Any grading standard (ICDR/ETDRS), any per-class clinical definition, any per-image ground truth, the identity of the three sources, who labelled the merged images |
| **Zenodo / Data in Brief** (Castillo Benítez et al.; source of the 757 `.jpg`) | Record: *"classification… done in 7 categories"* by expert ophthalmologists, with the 7 class names + counts; the paper frames DR per ETDRS (NPDR vs PDR) | Lesion-level definitions of "Very Severe NPDR" / "Advanced PDR" in the record description itself |
| **ICDR scale** (Wilkinson et al. 2003; AAO) | 5 bands: *No apparent retinopathy* ("no abnormalities"); *Mild NPDR* ("microaneurysms only"); *Moderate NPDR* ("more than just microaneurysms but less than severe"); *Severe NPDR* (4-2-1 rule, no PDR signs); *PDR* ("neovascularization and/or vitreous/preretinal hemorrhage") | "Very Severe NPDR" and "Advanced PDR" — **these are not ICDR grades** |
| **ETDRS DRSS** | Levels 10 / 20 / 35 / 43 / 47 / 53A–E / 61 / 65 / 71–75 / 81–85. **53E = two or more severe characteristics ("very severe NPDR")**; **81–85 = "Advanced PDR", fundus partially obscured**. Approximate crosswalk: 10→no DR; 20/35→mild; 43/47→moderate; 53A–E→severe; ≥61→PDR | A claim that any dataset here actually used ETDRS photography standards |
| **Kaggle EyePACS DR Detection** (likely origin of the `plain` family) | *"A clinician has rated… on a scale of 0 to 4: 0 - No DR, 1 - Mild, 2 - Moderate, 3 - Severe, 4 - Proliferative DR"* | Named use of ICDR or ETDRS; per-image lesion criteria |
| **Kaggle APTOS 2019** (candidate, not matching counts) | 0–4 grades consistent with ICDR band names | Explicit ICDR attribution (unverified this cycle) |

**Finding:** the 13-folder taxonomy is an **undocumented mixture** of at least two vocabularies — ICDR-style five bands (Mild/Moderate/Severe/Proliferative, as used by the likely Kaggle sources) and ETDRS sub-grades (Very Severe NPDR = 53E, Advanced PDR = 81/85, contributed only by Zenodo). Folder names are the labels; **no shipped file validates any label**.

## 2. Label dictionary — all 13 folders

Verification legend: **D** = documented in a source; **T** = supported by clinical taxonomy; **I** = inference; **U** = unknown/unverifiable.

| # | Exact folder name | n | Intended meaning | Definition source | Verified? | Primary eligible | Proposed primary mapping | Specialist needed? | Secondary 13-class treatment | Key risks |
|---|---|---:|---|---|---|---|---|---|---|---|
| 1 | `1.Dry AMD` | 1,360 | Dry age-related macular degeneration | Folder name only | I/U | **No** (non-DR) | excluded from primary cohort | No (for exclusion) | keep as class | source = likely-prefixed only; no clinical definition |
| 2 | `2.Wet AMD` | 587 | Wet AMD | Folder name only | I/U | No | excluded | No | keep | 1 cross-label conflict file quarantined |
| 3 | `3.Mild DR` | 2,441 | Mild NPDR | Folder name; matches EyePACS "1 - Mild" and ICDR mild | D(name)/T | **Yes** | → **Mild NPDR** | For image-level truth, yes (see §4) | keep | per-image grading unverifiable; 4 Zenodo files are genuine ETDRS "Mild (or early) NPDR" |
| 4 | `4.Moderate DR` | 5,368 | Moderate NPDR | Folder name; EyePACS "2 - Moderate" | D(name)/T | Yes | → **Moderate NPDR** | Yes | keep | largest class; 20 conflict files quarantined; likely subsampled from a bigger source |
| 5 | `5.Severe DR` | 1,918 | Severe NPDR | Folder name; EyePACS "3 - Severe" | D(name)/T | Yes | → **Severe NPDR** (merged with #12) | Yes | keep | 4-2-1 rule cannot be checked without lesions/annotations |
| 6 | `6.Proliferate DR` | 1,504 | Proliferative DR | Folder name (typo preserved; record writes "Proliferative"); EyePACS "4 - Proliferative DR" | D(name)/T | Yes | → **PDR** (merged with #13) | Yes | keep | clinically subsumes Advanced PDR → not mutually exclusive with #13 by definition |
| 7 | `7.Cataract` | 1,505 | Cataract | Folder name only | I/U | No | excluded | No | keep | media-opacity images may also show DR — overlap risk |
| 8 | `8.Hypertensive Retinopathy` | 1,344 | Hypertensive retinopathy | Folder name only | I/U | No | excluded | No | keep | no definition |
| 9 | `9.Pathological Myopia` | 1,265 | Pathological myopia | Folder name only | I/U | No | excluded | No | keep | no definition |
| 10 | `10.Glaucoma` | 1,878 | Glaucoma | Folder name only | I/U | No | excluded | No | keep | 1 cross-label conflict file quarantined |
| 11 | `11.Normal Fundus` | 3,061 | No visible DR / normal-appearing fundus | Folder name; Zenodo counterpart literally named *"No DR signs"* | D(name)/T | Yes | → **No apparent DR** | For "healthy" claims, yes | keep | "no visible DR in this image" ≠ clinically confirmed absence of disease (see §5) |
| 12 | `12. Advanced PDR` | 114 | Advanced PDR = ETDRS 81/85 | Zenodo 7-class scheme; ETDRS | D(T) | Yes | → **PDR** (merged with #6) | Recommended before merge is treated as fact | keep | Zenodo-only; all 114 are single-source, 38 quality-flagged |
| 13 | `13. Very Severe NPDR` | 108 | Very severe NPDR = ETDRS 53E (≥2 severe characteristics) | Zenodo 7-class scheme; ETDRS | D(T) | Yes | → **Severe NPDR** (merged with #5) | Recommended | keep | Zenodo-only; 27 quality-flagged; 12 of the 757 were later dropped by Zenodo in v0.3 — block `449–458` sits in this folder |

(Exact folder strings preserved, including the space in `12. Advanced PDR` and `13. Very Severe NPDR`, and the "Proliferate" spelling.)

## 3. Grading-scheme determination

- Zenodo `.jpg` (757): graded under an **ETDRS-derived 7-level scheme** by expert ophthalmologists — the source documentation supports this **[D]**, though per-image criteria are not published.
- `plain` family (10,599): **almost certainly a Kaggle 5-class (0–4) scheme** consistent with ICDR band names **[I, strong count + filename evidence]** — formally unverified **[U]**.
- `prefixed` family (11,097): scheme **unknown** **[U]**.
- Mendeley itself applied **no documented protocol** to the merge **[D-via-absence]**.
→ Verdict: the folders are **not** the output of one grading standard. Treat "folder label" as *source-specific weak supervision*, not ground truth (corroborated by the 75 duplicate-connected files carrying two labels at once).

## 4. Candidate 5-band primary scheme — mapping defensibility

| Candidate mapping | Source documentation | Clinical taxonomy | Inference | Verdict |
|---|---|---|---|---|
| Mild DR → Mild NPDR | name-level only | ICDR mild = "MA only" — consistent | — | **Defensible at band level**; per-image truth unverifiable |
| Moderate DR → Moderate NPDR | name-level only | ICDR moderate band — consistent | — | Defensible at band level |
| Severe DR + **Very Severe NPDR** → Severe NPDR | Severe: name-level. VSNPDR: Zenodo/ETDRS 53E | ETDRS 53A–E all fall inside ICDR severe NPDR; ICDR has no sixth band | — | **Taxonomy-supported merge**; still needs your approval because it changes the Zenodo-only class (see alternatives §8) |
| Proliferate DR + **Advanced PDR** → PDR | PDR: name-level. AdvPDR: Zenodo/ETDRS 81–85 | ETDRS ≥61 all inside ICDR PDR | — | Taxonomy-supported merge; also *necessary* because the two folders are not clinically disjoint |
| Normal Fundus → No apparent DR | Zenodo's own name "No DR signs" | ICDR "no apparent retinopathy" = "no abnormalities" | That the non-Zenodo sources meant the same thing | **Defensible with explicit wording**: label = *no DR visible in this image*, not *confirmed healthy* |
| 6 non-DR folders → primary | — | not DR grades | — | **Excluded from primary cohort**; a future "other / not eligible for DR grading" outcome is conceivable but **not defined here** and would need its own scope note |

Per-image lesion verification (4-2-1 rule, neovascularization) is **impossible** with the shipped data (no annotations, no metadata). All band assignments are therefore *label-level* only — a specialist review of a stratified sample is the only available upgrade path.

## 5. "No apparent DR" vs clinically confirmed absence

The policy wording must be: **"no retinopathy signs visible in this image per the source labeller"**. Sources likely used different photography protocols and single-image-per-visit grading; early DR can be missed in one field of view. Downstream write-ups must not call this cohort "healthy controls" without the qualifier.

## 6. Recommended primary-task label set (for approval)

`No apparent DR · Mild NPDR · Moderate NPDR · Severe NPDR · PDR` — with:
- Zenodo `Very Severe NPDR` merged into **Severe NPDR**; Zenodo `Advanced PDR` merged into **PDR** (both taxonomy-justified, both reversible);
- 75 duplicate-conflict files quarantined out of the affected bands (§7);
- 300 quality-flagged files quarantined pending the human review (Workstream C);
- non-DR folders excluded from the cohort entirely (they remain in the secondary task).

Provisional post-quarantine band sizes (before quality-review outcomes): No apparent DR 2,973 · Mild 2,436 · Moderate 5,303 · Severe 1,971 · PDR 1,518 (see `build_summary.json`).

## 7. Quarantine rule for label conflicts

Any file in a cross-label duplicate component is **excluded from the primary and secondary experiments until adjudicated** (75 files, 26 components; per-class breakdown in `build_summary.json`). Rationale: their supervision is self-contradictory and they leak across splits. This is an exclusion from *experiments*, not a deletion.

## 8. Alternatives and consequences (for your decision)

| Option | Consequence |
|---|---|
| **A (recommended): merge as §6** | 5 balanced-ish bands (≈2.2:1 max:min); removes the two single-source tiny classes; honest to ICDR; loses VSNPDR/AdvPDR granularity (kept in secondary task) |
| B: keep 7 DR grades (6 bands incl. VSNPDR and AdvPDR separately) | Preserves granularity but VSNPDR (78 eligible) and AdvPDR (75) are **Zenodo-only, ungroupable by patient, 30–35 % quality-flagged** → unstable metrics, indefensible per-band claims; also keeps the clinically overlapping PDR/Advanced-PDR split |
| C: merge only PDR+AdvPDR, keep VSNPDR separate | Middle ground; still leaves a 78-image, single-source, ungroupable band |

## 9. Decisions requiring YOUR approval (status after Option A)

| # | Decision | Status (2026-10-13) |
|---|---|---|
| 1 | Adopt the 5-band mapping in §6 | ✅ **approved provisionally** (Option A, safeguard #1) — VSNPDR→Severe and AdvPDR→PDR recorded as provisional merges with evidence tags; original labels preserved |
| 2 | "No apparent DR" wording rule (§5) | ✅ **approved** (safeguard #2): "no DR visible according to the source label"; never "healthy" |
| 3 | Quarantine (not delete) the 75 conflict files | ✅ **approved**: quarantined from **both** tasks where labels are affected (safeguard #3); no deletion/relabeling |
| 4 | Quality-flagged files stay quarantined until review | ✅ **approved**: all 300 quarantined pending documented human review (safeguard #4) |
| 5 | Non-DR folders excluded from primary cohort | ✅ **approved** (safeguard #5); retained in the secondary 13-class task |
| 6 | Specialist consult for the two merges | ⚠️ **requirement recorded** (safeguard #6): review needed before any clinical-validation claim; NOT yet performed |
| 7 | No Zenodo v0.2/v0.3 additions | ✅ **confirmed** (safeguard #7); enforced by provenance manifest policy |

Remaining approvals: §12.

## 10. Task separation

Primary-task manifest and secondary-task manifest are generated separately from `PROVENANCE_MANIFEST.csv`; a file's row may exist in both, but labels, quarantine states, and split assignments are stored per task. The secondary task keeps the 13 original folder labels verbatim (with conflict quarantines applied).

## 11. Approval record — Option A (2026-10-13)

The user approved Option A provisionally, subject to seven safeguards, all now encoded in `build_provisional_manifests.py` and materialized in `PROVISIONAL_ELIGIBILITY_PRIMARY.csv` / `PROVISIONAL_ELIGIBILITY_SECONDARY.csv`:

1. Five-band primary scheme; VSNPDR→Severe NPDR and AdvPDR→PDR **provisional** merges with documented evidence (ETDRS 53E and 81/85 fall inside the ICDR severe/PDR bands) **and** the explicit note that no per-image clinical adjudication exists; all original labels preserved (`task_label_basis` column carries `PROVISIONAL_MERGE` markers).
2. Normal Fundus wording: "no DR visible according to the source label"; these images must not be presented as proof of health.
3. The 75 cross-label conflict files are quarantined from both tasks; not deleted, not relabeled.
4. All 300 quality-flagged images are quarantined pending documented human review; flags are not defect verdicts.
5. Non-DR conditions excluded from the primary cohort; the original 13 categories are retained for the secondary experiment subject to its own eligibility rules.
6. The need for specialist/source-documentation review of the two merges is recorded (this §11 and the `PROVISIONAL_MERGE` tags); **no clinical validation is claimed** and none may be claimed until such review happens.
7. Confirmed: no Zenodo v0.2 or v0.3 images will be appended to the dataset.

## 12. Still requiring user approval (post-Option-A)

1. **Adjudication route for the 75 conflict files**: specialist review vs permanent exclusion (and whether any return after review).
2. **Execution and sign-off of the 300-image human quality review** (`QUALITY_REVIEW_PROTOCOL.md`), which will change eligibility counts.
3. **Specialist/source-documentation sign-off for the two provisional merges** before any publication-grade claim (safeguard #6 requirement — recorded, not satisfied).
4. **Handling of the 33 byte-identical same-label duplicate files** (16 exact groups): keep both copies grouped in one split (current behavior) vs single-copy exclusion — affects reported sample sizes.
5. **Split ratio and the Zenodo test-partition cap** (see `SPLIT_READINESS_REPORT.md` §3) — deferred until items 1–2 resolve.
6. **External-evaluation strategy and licence review** for the two "likely"-provenance families (publication gate).
