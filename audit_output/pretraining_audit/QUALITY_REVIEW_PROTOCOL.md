# Image-Quality Human-Review Protocol

**Status: DRAFT PROTOCOL — no visual review has been performed yet.** The 300 flagged images were characterised **numerically only** (luminance/saturation/edge statistics and coarse ASCII colour maps). Browser screenshot automation failed in the audit environment; no image has been "seen" by a reviewer or a model of record. `QUALITY_REVIEW_REGISTER.csv` therefore records `visual_review_performed = no` for every row.

**Governing rule: no image is deleted or excluded on a numerical threshold alone.** Dark and low-contrast fundus images are frequently genuine and sometimes clinically the most informative; flags mark *review priority*, never automatic exclusion.

## 1. Flag definitions (as implemented in `analyze.py`, verified)

| Flag | Rule | Files | Prior audit reading |
|---|---|---:|---|
| `near_uniform_low_std` | luminance std < 12 (128×128) | 10 | Flat mid-grey; near-blank. **P1 priority.** |
| `near_grayscale` | mean saturation < 18 | 64 | Neutral grey centres (e.g. R=G=B≈128) → desaturated or non-standard colour. **P2.** |
| `low_texture` | edge energy < 1.2 | 21 | Mix of dark-but-valid fundus and low-structure centres. **P2.** |
| `very_dark` | luminance mean < 30 | 231 | 172/231 show reddish retinal centres → dark exposure, likely genuine. **P3.** |
| `mostly_black` / `mostly_white` / `black_border` / `white_border` / `tiny_dims` / `very_bright` | per `analyze.py` | few | Framing/size artefacts. **P3–P4.** |

Flags overlap: 329 flag instances on 300 files (register column `flags_overlap`). A broader population of 448 images with `lum_std < 20` exists dataset-wide (348 of them Zenodo) and is *not* all in the 300; the protocol's control sample (§3) covers it.

## 2. Stratified review sample (recommended)

Review order: P1 → P2 → P3 → P4, and within priority, mix source families so that family-specific artefacts are judged by family norms.

| Stratum | Population | Review scope |
|---|---|---|
| `near_uniform_low_std` | 10 | **All 10** (tiny; these decide the "near-blank" question once) |
| `near_grayscale` | 64 | All 64, grouped by family (prefixed vs plain) |
| `low_texture` | 21 | All 21 |
| `very_dark` × {zenodo, plain, prefixed} | 231 | ≥ 30 per family (or all if fewer), plus every file that also carries a second flag |
| border/size/bright flags | ≤ 15 | All |
| **Controls (unflagged)** | 22,153 | ≥ 5 per primary class sampled across families and resolutions, incl. 5 Zenodo `lum_std<20` images that were **not** flagged — calibrates false-positive rate |

Total minimum ≈ 150–180 images. Reviewers see images **side by side with family-matched unflagged controls**, never in isolation, so "dark" is judged relative to its own family (Zenodo images are systematically darker than prefixed 512×512 images).

## 3. Reviewer's decision form (one row per image, appended to the register's `review_outcome`)

| Field | Values |
|---|---|
| Retinal content visible? | yes / partial / no |
| Usable for training (any task)? | yes / yes-with-caution / no |
| Dominant problem (if any) | underexposure / desaturation / blur / crop-FOV / compression / corruption / none |
| Likely genuine acquisition difficulty? | yes / no / unsure |
| Recommended status | eligible / quarantine-retain / exclude-with-reason |
| Free text | — |

Suggested tool: open `sample_review.html` (already in `audit_output/`) and extend it to iterate the register strata; any image viewer is acceptable as long as decisions are transcribed back into a copy of `QUALITY_REVIEW_REGISTER.csv`.

## 4. When specialist (retinal) input is required

1. Any image judged `no` for usability **and** belonging to a DR-severity class with fewer than ~200 eligible images (VSNPDR, Advanced PDR) — exclusion changes class balance materially.
2. Any `low_texture`/`near_uniform` image that is the **only** member of its duplicate component or patient group.
3. Any case where "partial" content could still show pathology relevant to grading (e.g. visible haemorrhages but cropped optic disc).
4. Any disagreement between two reviewers (if a second reviewer is available; single-reviewer mode must be declared in the write-up).

## 5. Recording and auditability rules

- Decisions are written to a **copy** (`QUALITY_REVIEW_REGISTER_reviewed.csv`), never over the original register.
- Every `exclude-with-reason` entry must carry a free-text reason and, where applicable, the specialist's confirmation flag.
- The final report must state: how many of the 300 were actually reviewed, by whom (role), with what tool, and that the remainder were only numerically characterised.
- No excluded file is deleted from the dataset; exclusion is implemented only in the later approved experiment manifest.
