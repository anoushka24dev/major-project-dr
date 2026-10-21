# dataset

This package is the deterministic, test-driven data-pipeline layer for the diabetic retinopathy capstone. It works entirely on the provisional audit artifacts under `audit_output/pretraining_audit/` and never rewrites them.

## What lives here

- `manifest.py` — load and validate the primary/secondary eligibility manifests, resolve image paths, detect duplicates and missing files.
- `labels.py` — deterministic mapping between the 13 source labels and the provisional primary bands, plus the verbatim secondary task.
- `eligibility.py` — quarantine and non-DR exclusion logic driven by the manifest `eligibility` column.
- `statistics.py` — eligible class counts, status counts, and separate excluded/quarantined decomposition for reconciliation.
- `grouping.py` — helpers for `split_group_id`, `duplicate_group_id`, patient-key presence, and zenodo unlinked rows.
- `decoding.py` — Pillow-based image loading, RGB conversion, and small metadata reports.

## How the manifests control eligibility

The authoritative eligibility is the `eligibility` column in the provisional CSV files. The pipeline does not recompute eligibility from folder names or image contents.

For the primary task:
- `eligible_provisional` rows enter the candidate eligible set.
- `quarantined_*` and `excluded_non_dr_*` rows are excluded and remain traceable.

For the secondary task:
- eligible rows are still `eligible_provisional` rows in the secondary manifest.
- original source labels are kept verbatim.

## Provisional five-band mapping

The primary task uses a provisional five-band mapping:

- No apparent DR — `11.Normal Fundus`
- Mild NPDR — `3.Mild DR`
- Moderate NPDR — `4.Moderate DR`
- Severe NPDR — `5.Severe DR` + `13. Very Severe NPDR`
- PDR — `6.Proliferate DR` + `12. Advanced PDR`

These merges are provisional. Original source labels are preserved elsewhere, and the code does not claim specialist validation.

## Duplicate groups vs verified patient groups

`duplicate_group_id` marks rows that the audit flagged as duplicate-connected. That is a grouping signal, not a verified patient identity. `split_group_id` is used where present, and some records have `patient_group_key` values. Records that lack a usable patient key are reported as unlinked rather than assumed to be patient-independent.

## Running the tests

From the project root:

```
python -m pytest -q tests/
```

Unit tests use temporary synthetic fixtures and do not depend on the real image dataset being present. To also decode a real eligible image from each class, run:

```
python -m pytest tests/test_image_decoding.py::test_real_image_smoke_test -q
```

The smoke test only runs if the verified image root exists.

## Optional real-image smoke test

The smoke test loads one eligible primary row from each of the 13 source classes from the real manifest, resolves its image path under the verified image root, and decodes it with Pillow. It is a small integration check on real files; it is not part of the default unit run.

## Out of scope

- Model architecture, training, augmentation, splits, synthetic metadata, Grad-CAM, calibration, uncertainty, and clinical claims.
- Modifying the existing audit artifacts, labels, or images.
- Generating the actual train/validation/test split here. The split validator only checks an externally supplied assignment.
