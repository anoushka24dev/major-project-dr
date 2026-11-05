"""
Auxiliary dataset *provisioning* is intentionally NOT implemented here.

This module exists only to document, in one place, the provenance rules this project
enforces and to refuse the behaviours that would violate them.

The project has already audited, in code, the following actions as **not permitted**:

1. Copying MMRDR images into a Mendeley-shaped directory and presenting them as the
   genuine Mendeley Data corpus.
2. Copying MMRDR images into an IDRiD-shaped directory and presenting them as IDRiD.
3. Renaming another dataset's images (`IDRiD_*.jpg`) to look like IDRiD.
4. Splitting MMRDR grade 3 into "Severe DR" + "Very Severe NPDR", or grade 4 into
   "Proliferate DR" + "Advanced PDR", without genuine independent labels.
5. Distributing MMRDR grade-0 ("No DR") images across unrelated retinal-disease folders
   (Dry AMD, Wet AMD, Cataract, Hypertensive Retinopathy, Pathological Myopia, Glaucoma)
   as if "No DR" established the presence of those diseases.
6. Generating U-Net ground-truth lesion masks from RGB intensity heuristics and calling
   them verified human annotations.

Any of the above produces contaminated data, fabricated labels, or pseudo-labels that are
not ground truth. The pipeline refuses all of them, and this script enforces that refusal
at the tooling level instead of quietly permitting it.

This script is deliberately NOT a workaround for missing upstream data. When a dataset is
not genuinely available, the correct behaviour is to mark that stage blocked and supply the
real data, not to construct replacement data from a different source.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _error(title: str, body: str) -> None:
    print(f"[PROVENANCE BLOCKED] {title}")
    print(body, file=sys.stderr)


def _instructions_mendeley() -> str:
    return (
        "\n"
        "Mendeley Data (6ry9n79gb4, CC BY 4.0) is the genuine source for the Mendeley\n"
        "classification cohort. As audited, the public API currently exposes no downloadable\n"
        "files for this record. The correct way to supply Mendeley is therefore a local\n"
        "pre-existing copy, never a reconstruction from another dataset:\n"
        "\n"
        "  1. Obtain the genuine Mendeley corpus yourself (the dataset page or your own\n"
        "     prior download).\n"
        "  2. Point the pipeline at the folder that directly contains the 13 class folders:\n"
        "       RUN.bat --fundus-root \"PATH\\to\\Diabetic Retinopathy and Retinal Diseases Fundus I\"\n"
        "     or set DR_FUNDUS_ROOT.\n"
        "  3. Run RUN.bat --resume. The pipeline validates the folder and records provenance\n"
        "     as 'local_preexisting', not as a download.\n"
        "\n"
        "Do not populate datasets/mendeley/ from MMRDR, APTOS, DDR, or any other source.\n"
    )


def _instructions_idrid() -> str:
    return (
        "\n"
        "IDRiD (IEEE DataPort / Grand Challenge) is the genuine source for lesion\n"
        "segmentation and annotation-based explanation validation. It requires registration\n"
        "and terms acceptance; that restriction is not bypassed.\n"
        "\n"
        "  1. Register at:\n"
        "       https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid\n"
        "     (mirror: https://idrid.grand-challenge.org/)\n"
        "     and accept the terms.\n"
        "  2. Download '1. Original Images' and '2. All Segmentation Groundtruths'.\n"
        "  3. Extract so the numbered folders sit directly under datasets/idrid/:\n"
        "\n"
        "       datasets/idrid/\n"
        "       ├── 1. Original Images/\n"
        "       │   └── a. Training Set/          IDRiD_01.jpg ...\n"
        "       └── 2. All Segmentation Groundtruths/\n"
        "           └── a. Training Set/\n"
        "               ├── 1. Microaneurysms/    IDRiD_01_MA.tif ...\n"
        "               ├── 2. Haemorrhages/      IDRiD_01_HE.tif ...\n"
        "               ├── 3. Hard Exudates/     IDRiD_01_EX.tif ...\n"
        "               └── 4. Soft Exudates/     IDRiD_01_SE.tif ...\n"
        "\n"
        "  4. Run RUN.bat --resume. The IDRiD loader validates genuine layout and 1:1\n"
        "     image/mask pairing before using it.\n"
        "\n"
        "You can also use --idrid-root \"PATH\" or DR_IDRID_ROOT.\n"
        "\n"
        "Do not copy MMRDR images into datasets/idrid/ and rename them as IDRiD.\n"
        "Do not generate lesion masks with RGB thresholds and treat them as ground truth.\n"
    )


def setup_idrid() -> None:
    """Refuse to construct a fake IDRiD corpus from MMRDR.

    The genuine segmentation stage requires real IDRiD images plus the four official
    human-annotated lesion mask sets (MA, HE, EX, SE). This helper rejects the shortcut
    of copying MMRDR images in and generating heuristic masks from RGB intensity.
    """
    _error(
        "IDRiD auxiliary provisioning is blocked",
        "This script no longer copies MMRDR images into datasets/idrid/ and no longer "
        "generates heuristic lesion masks. The previous behaviour created fake IDRiD "
        "directory entries and pseudo-labels, not ground truth.\n"
        + _instructions_idrid(),
    )


def setup_mendeley() -> None:
    """Refuse to reconstruct a Mendeley corpus from MMRDR.

    The genuine Mendeley classification cohort uses the real Mendeley class folders and
    image files. This helper rejects the shortcut of slicing MMRDR FP.csv by grade,
    copying MMRDR images into the 13 Mendeley folders, inventing unsupported severity
    subdivisions, and assigning No-DR images to unrelated retinal-disease folders.
    """
    _error(
        "Mendeley auxiliary provisioning is blocked",
        "This script no longer populates datasets/mendeley/ from MMRDR. The previous "
        "behaviour was not the genuine Mendeley dataset: it copied MMRDR images, invented "
        "finer severity labels (Very Severe NPDR / Advanced PDR) that the source does not "
        "support, and assigned No-DR images to unrelated retinal diseases.\n"
        + _instructions_mendeley(),
    )


if __name__ == "__main__":
    print("scripts/prepare_auxiliary_datasets.py is now a provenance guard, not a data generator.")
    print("It refuses to fabricate Mendeley or IDRiD data from MMRDR.\n")
    setup_idrid()
    setup_mendeley()
    sys.exit(2)
