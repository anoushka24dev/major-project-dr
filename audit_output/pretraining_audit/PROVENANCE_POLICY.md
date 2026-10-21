# Provenance and Source-Family Policy

**Status: ACTIVE POLICY for all pre-training manifests.** Machine-readable companion: `PROVENANCE_MANIFEST.csv` (22,453 rows; one per image: source family, status, evidence, resolution, patient-group key, duplicate-group ID, quality flags, Zenodo-v0.3 relationship, SHA-256).

## 1. Status tiers (never promote without evidence)

| Tier | Meaning | Rule |
|---|---|---|
| **confirmed** | File-level evidence ties the image to a named source | May be cited as fact |
| **likely** | Filename/count patterns fit a candidate source; no file-level proof | Must be written as "likely"; cannot support provenance-dependent claims |
| **unknown** | No candidate | Treated as unattributed |

## 2. Per-family status (verified this cycle)

| Family | Files | Status | Evidence |
|---|---:|---|---|
| `zenodo_v02` (plain `N.jpg` in the 7 DR folders) | 757 | **confirmed** | All 757 re-hashed this cycle: **757/757 SHA-256 identical** to Castillo Benítez et al. Zenodo v0.2 (record 4647952, concept DOI 10.5281/zenodo.4532360), with 100 % folder-label agreement under the documented renames (No DR signs→Normal Fundus, PDR→Proliferate DR) |
| `prefixed` (`<Class><ID>_<side>_0_<n>.jpeg`) | 11,097 | **likely** | Class-prefixed filename template + per-class count match with Kaggle "retinal-fundus-images"-style collections; exact source and licence **not** documented anywhere **[unresolved]** |
| `plain` (`<ID>[_side].jpeg`) | 10,599 | **likely** | Kaggle EyePACS-style names; class counts close to EyePACS training labels (Mild 2,443 vs 2,437; Moderate 5,292 vs 5,288) but Normal appears subsampled; **not** hash-confirmed against EyePACS **[unresolved]** |
| unknown | 0 | — | Every file matched a family pattern |

## 3. Known issues that the policy must keep visible

1. **Count discrepancy**: Mendeley record states 22,454; local audit found 22,453 readable files and the zip contains exactly 22,453 entries. Unresolved — possibly a description typo; never "fix" it by adding a file.
2. **Zenodo v0.2 is fully contained** in the dataset (confirmed above). Adding the Zenodo zip again would create 757 duplicates. **Zenodo files must not be appended** (neither v0.2 nor v0.3).
3. **Zenodo v0.3 overlaps visually but is a different release**: 745 of the 757 are pixel-identical to v0.3 images (re-encoded); v0.3 adds 680 genuinely new images, dropped 12 of the original 757, and contains its own cross-label contradictions (24 verified internal clusters). **Do not merge v0.3.** Its labels may be used only as weak corroborating evidence (see conflict register).
4. **Source-family artefacts are label-correlated**: resolution (512×512 prefixed vs 2124×2056 Zenodo), grey #808080 canvas corners (97 % of plain), and black borders (Zenodo) predict family, and family membership correlates with class (e.g. all Advanced PDR and Very Severe NPDR are Zenodo). Any evaluation must therefore report **per-family metrics**; a random image-level split cannot demonstrate cross-source generalization.
5. **Licences**: Mendeley CC BY 4.0 and Zenodo CC BY 4.0 confirmed; the two likely-Kaggle families carry Kaggle's Open Database/competition terms — **licence compatibility for publication is unresolved** and must be settled before any public release of derivatives.

## 6. Manifest rules

- Every future experiment manifest must carry the `image_path`, `original_label`, `source_family`, `source_status`, `duplicate_group_id`, `patient_group_key`, and quarantine columns from `PROVENANCE_MANIFEST.csv`; nothing may enter an experiment without a row here.
- External images may only be added after a new provenance row with its own evidence and licence record; **no external images were added in this phase, and none may be added without approval**.
- Provenance status changes (e.g. a "likely" family gets hash-confirmed) are recorded by regenerating the manifest with a dated version suffix, never by in-place silent edits.
