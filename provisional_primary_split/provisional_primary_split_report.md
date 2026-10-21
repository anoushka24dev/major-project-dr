# Provisional Primary Five-Band Split Report

**Status:** PROVISIONAL -- not an approved final dataset; generated from `PROVISIONAL_ELIGIBILITY_PRIMARY.csv`.

## Provenance
- Manifest version: `provisional-v1-2026-10-13`
- Manifest path: `audit_output\pretraining_audit\PROVISIONAL_ELIGIBILITY_PRIMARY.csv`
- Manifest file SHA-256: `86c8517d796a6abdad03596a0d48ee7e000194b5e8a54b66597541728506d457`
- Approval reference: `Option A approved by user 2026-10-13 (7 safeguards)`
- Generated split CSV: `provisional_primary_split\provisional_primary_split.csv`
- Seed: `12345`
- Target fractions: `{'train': 0.7, 'validation': 0.15, 'test': 0.15}`
- Soft Zenodo test cap: `0.1`

## Group atomicity
- Total components allocated: `14143`
- Group-table SHA-256: `6d3f658998893ab4e35ea3631d84d6e50fe7028133936e5c84d33e7a2ed8707f`
- Validation result from `SplitValidator`: `True`

## Image counts

| Partition | Images | Ratio |
|-----------|--------|-------|
| train | 9924 | 0.6988 |
| validation | 2133 | 0.1502 |
| test | 2144 | 0.1510 |

## Per-class counts

| Band | Images | Groups | Train | Validation | Test |
|------|--------|--------|-------|------------|------|
| No apparent DR | 2973 | 2963 | 2078 | 447 | 448 |
| Mild NPDR | 2436 | 2436 | 1706 | 365 | 365 |
| Moderate NPDR | 5303 | 5284 | 3705 | 796 | 802 |
| Severe NPDR | 1971 | 1947 | 1373 | 297 | 301 |
| PDR | 1518 | 1513 | 1062 | 228 | 228 |

## Zenodo-derived images without verified patient linkage

- Total Zenodo-eligible images in this split: `503`

| Partition | Zenodo images | Pure-Zenodo singletons |
|-----------|---------------|------------------------|
| train | 342 | 271 |
| validation | 73 | 61 |
| test | 88 | 61 |

## Limitations
- Eligibility is provisional and is pinned to the manifest version and approval referenced above.
- 503 eligible Zenodo-derived images lack verified patient linkage; patient independence cannot be claimed for them.
- Within the duplicate-connected-component view, 393 Zenodo-eligible images are pure-Zenodo singletons and 110 are bound into mixed-source duplicate components.
- The Severe NPDR and PDR taxonomy merges remain provisional and unsigned.
- Human quality review and conflict adjudication are incomplete; quarantined and excluded images are not assigned.
- Internal splitting does not establish source or external generalization.

## Notes

- Ratios are approximate. Actual proportions are reported above; they are not claimed to be exact 70/15/15.
- Duplicate-connected components are the atomic unit for this split; no such component crosses partitions.
- Patient independence is **not** claimed for the Zenodo-derived images.
- The Severe NPDR and PDR taxonomy merges remain provisional and unsigned.
