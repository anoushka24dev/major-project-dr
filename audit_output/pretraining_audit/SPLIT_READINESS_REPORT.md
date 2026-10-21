# Split-Readiness Report

**Status: ASSESSMENT ONLY — no split has been created (and none should be until the label policy and quality review are approved).**

## 1. Grouping policy (to be applied when a split is eventually built)

**Unit of separation = connected group**, computed with union-find over two edge types:

1. **Patient/eye key** (`.jpeg` families only):
   - `prefixed`: `(class folder, numeric ID)` — unions both eyes and all re-shoots of one patient (e.g. `Dry AMD1280_left_…` + `Dry AMD1280_right_…`).
   - `plain`: `(folder, numeric ID)` — unions left/right and repeats.
   - Cross-folder ID collisions were tested in the audit: 232/232 are different images → do **not** union across folders for these families (except where a duplicate edge forces it).
2. **Duplicate component** (`DUP-G***` in `LABEL_CONFLICT_REGISTER.csv`) — overrides patient keys: a duplicate edge between different patients (3 verified cases) unions those patients into one group.

**Zenodo `.jpg` (757 files)**: no patient/eye linkage exists (verified in both Zenodo releases; the Data in Brief paper confirms anonymized, renamed image numbers). Each eligible Zenodo file is a **singleton group**; only duplicate components bind them. Every write-up must state that patient-independence is **not guaranteed** for these 503 provisional-eligible files.

## 2. Guarantee tiers (be precise about what a future split can claim)

| Tier | Applies to | Guarantee |
|---|---|---|
| **Patient-independent separation** | 21,696 `.jpeg` files (prefixed + plain) | Same patient/eye group never crosses train/val/test |
| **Image / duplicate-cluster isolation** | all files incl. 757 Zenodo | No duplicate or near-duplicate component crosses splits |
| **Source-generalization claim** | none of the above | **Not achievable from internal splits.** Requires an independently sourced external dataset with confirmed provenance; provenance of the two `.jpeg` families is only "likely", so overlap with common external benchmarks (e.g. EyePACS/APTOS) cannot be excluded and must be tested before any external eval |

## 3. Post-quarantine group counts (provisional; = total − quality-flagged − conflict-involved)

Primary-task bands under the recommended 5-band mapping (from `build_summary.json`):

| Primary band | Images eligible | Independent groups | of which Zenodo singletons (ungroupable) | Multi-image groups | Both-eye groups |
|---|---:|---:|---:|---:|---:|
| No apparent DR | 2,973 | 1,951 | 99 | 1,022 | 1,022 |
| Mild NPDR | 2,436 | 1,842 | 4 | 594 | 594 |
| Moderate NPDR | 5,303 | 3,338 | 53 | 1,965 | 1,965 |
| Severe NPDR (Severe DR + Very Severe NPDR) | 1,971 | 800 | 235 | 561 | 305 |
| PDR (Proliferate DR + Advanced PDR) | 1,518 | 556 | 112 | 444 | 262 |
| **Total** | **14,201** | **8,487** | **503** | — | — |

Secondary-task 13-class counts: see `build_summary.json` (`primary_stats` + `nondr_stats`).

**Is a 70/15/15 split defensible now?** Numerically, the merged bands have 556–3,338 groups each — enough for a stratified *group* split with non-degenerate test sets. But eligibility is provisional: the 300-file quality review and any conflict adjudication will change these counts, and the label policy itself is unapproved. **Recommendation: fix ratios only after approvals; default proposal = 70/15/15 by group, stratified per primary band, with the constraint that Zenodo singletons may not exceed ~10 % of any test partition** (limits the ungroupable-bleed surface; exact cap is your call).

## 4. Leakage checks that must pass before any split is committed

1. No `patient_group_key` appears in two splits (automated).
2. No `DUP-G*` component appears in two splits (automated).
3. No quarantined file (conflict or unreviewed quality flag) appears in any split (automated).
4. Per-split source-family composition table produced and reviewed (automated, human-eyeballed) — a split whose test set is, e.g., mostly Zenodo while train is mostly prefixed would silently measure source transfer, not pathology.
5. Test set is created once, sealed, and used only in final evaluation; model selection and calibration use train/val only (process rule).
6. Split manifest records its seed, the group table hash, and the register/policy versions used (automated).

## 5. Prerequisites before creating the split

1. `LABEL_POLICY_DRAFT.md` §9 decisions approved (mapping, wording, quarantines).
2. Human quality review completed (or an explicit decision to quarantine all 300 until it is).
3. Conflict adjudication decision taken for the 75 files (exclude vs review).
4. Split group table regenerated from the final quarantine state.
5. (For any future external evaluation) provenance overlap test between candidate external data and the two "likely" families.
