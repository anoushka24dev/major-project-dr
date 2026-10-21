# Data-Pipeline Acceptance Criteria (TDD test plan)

**Status: PROPOSED — write these tests BEFORE any pipeline code or training. No implementation code exists yet by design.**

Notation: **[A]** = automatable now · **[H]** = requires human/specialist judgment · stage = pipeline stage the check protects.

## A. Criteria and what each proves (and does not)

| # | Criterion | Proves | Cannot prove | Stage | Type |
|---|---|---|---|---|---|
| AC-1 | Dataset ingest is read-only: byte-level hash of every source file unchanged before/after any run | No pipeline step modified the source dataset | That the files were correct to begin with | ingest | [A] |
| AC-2 | Manifest completeness: every source image has exactly one provenance row; manifest count == filesystem count == 22,453 | No image silently added/dropped/duplicated in bookkeeping | Label correctness | ingest | [A] |
| AC-3 | Traceable labels: every experiment row carries `original_label` (folder, verbatim) and, where the task defines one, `task_label` with mapping version | Label lineage is auditable; mappings are versioned | That the mapping is clinically right (needs §9 approvals + [H]) | label mapping | [A] |
| AC-4 | Exclusion accounting: every file absent from an experiment manifest is present in a quarantine/exclusion register with a machine-checkable reason code | No silent deletions; counts reconcile (eligible + quarantined + out-of-scope == total) | That each exclusion was *justified* ([H] review of reasons) | quarantine | [A] |
| AC-5 | Conflict representation: every cross-label component (26) and file (75) is present in the conflict register; none appears in any experiment split unless an adjudication record exists | Label conflicts cannot leak back in implicitly | Which label is correct ([H]) | quarantine | [A] |
| AC-6 | Duplicate isolation: no `DUP-G*` component straddles two splits | Near-duplicate leakage across splits | That the register caught *all* duplicates (threshold-limited; see [H] audit below) | split | [A] |
| AC-7 | Patient isolation: no `patient_group_key` straddles two splits for `.jpeg` families | Patient/eye leakage for the 21,696 groupable files | Zenodo patient independence (impossible — singleton groups; must be stated, not tested) | split | [A] |
| AC-8 | Quarantine isolation: no quarantined file appears in train/val/test | Quarantine is enforced end-to-end | — | split | [A] |
| AC-9 | Test-set sealing: test manifest hash is fixed at split time; evaluation code path asserts the hash; model-selection and calibration logs reference only train/val IDs | Test data cannot influence model selection/calibration | Human rule-breaking outside the tooling | train/eval | [A] + process |
| AC-10 | Task separation: primary and secondary manifests are separate artifacts with distinct schemas; a test asserts no task's `task_label` column contains the other's label set | Tasks cannot be accidentally collapsed | Scientific soundness of either task | label mapping | [A] |
| AC-11 | Synthetic-metadata integrity: every metadata row carries `synthetic=true` provenance; a test asserts no synthetic field name matches a real clinical field used downstream; documentation build fails if synthetic experiments are described without the disclaimer | Synthetic data cannot be mistaken for observed clinical data | Whether a human writes prose correctly ([H] editorial check) | metadata gen | [A] + [H] |
| AC-12 | Family accounting: per-split source-family table generated; test asserts every split reports family composition and flags test sets where any family < 5 % or > 80 % | Source-shortcut risk is visible in every experiment report | That the model actually used shortcuts ([H] analysis) | split/eval | [A] |
| AC-13 | Version pinning: every experiment manifest records (hash of) provenance manifest, conflict register, quality register, and label-policy version | Results are reproducible against a known decision state | — | all | [A] |
| AC-14 | Duplicate-detection completeness spot-check: an augmentation-aware pass (rotation/flip/scale) re-runs on a stratified sample and reports additional matches | Bounds the "conservative thresholds" limitation | Full completeness | dedup [re-run] | [A] |
| AC-15 | Label-mapping review: a stratified image sample per (folder × proposed band) is reviewed by a qualified reviewer before mappings are frozen | Human check that folder→band mapping is clinically plausible | Per-image label truth (needs lesion-level review) | label mapping | [H] |
| AC-16 | Quality-review completion: register rows reviewed == planned sample; every `exclude-with-reason` has a reason and reviewer ID | Quality exclusions are evidenced | That judgments were medically correct ([H]) | quarantine | [A] + [H] |
| AC-17 | Leakage canaries: intentionally inject a synthetic duplicate and a synthetic cross-split patient into a test fixture; pipeline must flag both | The leakage checks themselves work (guards against silent check failure) | — | split | [A] |

## B. Fixtures and TDD notes

- Golden fixtures already exist in `audit_output/`: known duplicate pairs (e.g. `11.Normal Fundus/110.jpg ↔ 12. Advanced PDR/743.jpg`), known non-duplicates with similar hashes (the 28,815 loose pairs), the 12 Zenodo files absent from v0.3, and the exact 22,453-row manifest.
- Suggested order: AC-1/2 (ingest) → AC-3/10 (labels) → AC-4/5/16 (quarantine) → AC-6/7/8/17 (split) → AC-9/12/13 (experiment plumbing) → AC-11 (metadata) → AC-14/15 (pre-freeze reviews).
- Each failing test must fail for the *reason* it names; suppressing a check (skip, xfail) requires a written justification in this file.

## C. Explicitly out of scope until approvals land

- No test asserts clinical correctness of any label (impossible without specialist review — AC-15 is the closest proxy and it is [H]).
- No test asserts patient independence for Zenodo files (impossible; the honest statement lives in `SPLIT_READINESS_REPORT.md` §1–2 and must appear in any resulting paper).
