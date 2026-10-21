# LABEL_CONFLICT_REGISTER — counting conventions and unresolved cases

**Status: REGISTER OF RECORD — dispositions are PROPOSALS only. No file has been deleted, moved, or relabeled.**

Source data: `audit_output/` artifacts from the read-only audit (`exact_dup_groups.txt`, `near_dup_pairs.txt`, `strong_cross_class_pairs.txt`, `image_stats.jsonl`, `v03_pairs_verified.json`). Built by `build_registers.py` (reproducible, read-only).

## 1. What the CSV contains

One row per **duplicate pair edge** (`LABEL_CONFLICT_REGISTER.csv`, 168 data rows), with:
group ID, relationship type, evidence, both paths and labels, label agreement,
patient-group keys, source families, Zenodo-v0.3 label evidence, visual-review status,
proposed disposition, and disposition reason.

## 2. Counting conventions (do not treat these numbers as additive)

| Unit | Count | Definition |
|---|---:|---|
| Pair rows | 168 | Each row = one duplicate relationship (edge) |
| — exact-duplicate pairs | 18 | All unordered pairs inside the 16 SHA-256 groups (one group of 3 → 3 pairs, fifteen groups of 2 → 15 pairs) |
| — near-duplicate pairs (strong tier) | 149 | dHash ≤ 4 **and** 256-bit aHash ≤ 6, non-byte-identical (the audit's documented threshold) |
| — audit-noted weak pair | 1 | `Glaucoma2041_left_0_8337 ↔ Wet AMD2041_left_0_7926` (dHash 1, aHash 12 → below automatic threshold; the original audit pixel-verified r = 0.923. Included so the register is complete w.r.t. every cross-label relationship the audit identified.) |
| Unique files involved | 238 | Files appearing in ≥ 1 edge (the original report's "236" counted exact + strong-tier only; the extra weak pair adds 2) |
| Connected duplicate groups (components) | 103 | Union-find over all edges (exact members unioned; near-dup edges unioned). Size distribution: 93×2 files, 2×3, 5×4, 1×6, 1×18 |
| Cross-label pair rows | 55 | Edges whose two folder labels differ (54 strong-tier + 1 weak) |
| Cross-label unique files | **75** | Files that carry a contradictory label relationship (the earlier audit's "73" = strong tier only) |
| Cross-label components | 26 | Groups containing ≥ 2 different labels (80 files; 5 files are label-consistent members inside otherwise-conflicted components) |

Overlap cautions:
- **Exact vs near-dup overlap**: exact-duplicate files can also appear in near-dup edges; the 238-file union already deduplicates this.
- **Quality flags vs conflicts**: the 300 quality-flagged files and the 75 conflict files overlap; per-class "provisional eligible" counts subtract the **union** of both sets (see `build_summary.json`).
- Strong-tier breakdown: within-class 95 pairs (zenodo 80, prefixed 13, plain 2); cross-class 54 pairs (zenodo 53, prefixed 1). Patient relationship of strong pairs: same patient/eye group 13, verifiably different patient 3, **unknown 133 (any pair touching a Zenodo `.jpg` — no patient linkage exists)**.
- **Correction to the earlier report**: its phrase "82 pairs within class across different patient IDs" should read **3 pairs across verifiably different patient IDs + 79–80 Zenodo within-class pairs with no patient linkage at all**. The 13 "same patient" pairs are confirmed.

## 3. Evidence tiers in the register

1. `exact_duplicate` — SHA-256 identity. Conclusive that pixels are identical.
2. `near_duplicate` — dual perceptual-hash agreement (dHash ≤ 4, aHash ≤ 6). The original audit pixel-verified samples (MAE 0.00–6.0, r 0.92–1.000); it did **not** pixel-verify every one of the 149 pairs individually.
3. `audit_noted_weak_pair` — single-pair, weaker evidence (see above).

Zenodo-v0.3 label evidence columns record the v0.3 folder labels of pixel-identical twins (MAE < 1/255, r > 0.99). **This evidence does not adjudicate conflicts**: Zenodo v0.3 itself contains the same image under two labels in 24 verified internal clusters, so a v0.3 label is an additional opinion, not ground truth.

## 4. Disposition framework (conservative, per task instructions)

| Situation | Proposed disposition | Reason |
|---|---|---|
| Duplicate pair, labels agree | `keep_together_same_split_group` | Same image + same label; safe to train on, but copies must never straddle splits. Single-copy exclusion is a *candidate* needing explicit approval (redundancy, not error). |
| Duplicate pair, labels conflict | `QUARANTINE_both_pending_adjudication` | Contradictory supervision; would inflate metrics under any random split. Quarantine ≠ delete: files stay on disk, are excluded from experiment manifests only after approval. |

Quarantine applies to both/all members of a cross-label component, not just the two files of one edge.

## 5. Unresolved cases (require specialist or source-author input)

1. **75 conflict files / 26 components have no adjudicable ground truth locally.** Neither folder majority, model prediction, nor Zenodo-v0.3 opinion may decide them (task rules + evidence above). Options for you to choose later: (a) exclude all 75 from both tasks; (b) exclude only from the task whose label is in question; (c) obtain specialist review of the underlying images.
2. **Non-Zenodo cross-class pairs are few but suspicious**: the prefixed pair `Dry AMD1280_left_0_8225 ↔ Glaucoma1280_left_0_6908` (same numeric ID, different class prefixes) suggests either an ID collision across prefixed classes or a misfiled image; two further verified same-image cross-class prefixed pairs were noted in the original audit (§5). Whether prefixed IDs are globally unique remains **unresolved**.
3. **Augmented copies may remain undetected** (thresholds are conservative; prefixed-family augmentations can escape dHash/aHash). A dedicated augmentation-aware pass would be needed before declaring the register exhaustive.

## 6. Files

- `LABEL_CONFLICT_REGISTER.csv` — the register (168 rows; UTF-8-BOM, opens in Excel).
- `build_registers.py`, `build_summary.json`, `build_log.txt` — reproduction and counts.
