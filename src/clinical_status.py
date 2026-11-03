"""Status of the clinical-information and fuzzy-logic fusion component.

**This component is PROPOSED, not implemented.**

The project title describes a *multimodal* framework in which image-based risk is
compared with clinical risk and a fuzzy-logic layer flags disagreements for human
review. That capability depends on two things this repository does not have:

1. a genuine clinical dataset (no patient records, no tabular clinical variables);
2. an implemented clinical risk model and fuzzy inference system.

Neither exists anywhere in the repository, and none of the three project datasets
(Mendeley fundus images, MMRDR, IDRiD) supplies clinical metadata suitable for
this purpose.

Therefore no module, no synthetic patient generator and no placeholder fuzzy
system is provided: adding one would make the project *look* complete while
implying a clinical capability and a validation that do not exist. What is
provided instead is this machine-readable status record, so the gap is visible in
the generated report and in ``execution_summary.json`` rather than buried in prose.

If this component is built later, it must not run as part of the image-only
pipeline, and it must never present synthetic clinical records as real data.
"""

from __future__ import annotations

from typing import Any, Dict

STATUS = "proposed"

STATUS_RECORD: Dict[str, Any] = {
    "component": "clinical_information_and_fuzzy_logic_fusion",
    "status": STATUS,
    "implemented": False,
    "reason": (
        "No genuine clinical-data source exists in the repository, and no clinical risk "
        "model or fuzzy inference system is implemented."
    ),
    "what_exists": [
        "Image-based five-class DR risk from ResNet-50 (implemented, evaluable).",
        "Calibration (ECE) and uncertainty flags that route uncertain predictions to human "
        "review (implemented).",
        "Evidence-based diagnostic reporting per image (predictions CSV, Grad-CAM).",
    ],
    "what_is_missing": [
        "A real clinical dataset with per-patient clinical variables.",
        "A validated clinical risk model.",
        "A fuzzy inference system that resolves image/clinical risk disagreement.",
        "Any clinical validation of the combined decision rule.",
    ],
    "must_not_be_claimed": [
        "clinical validation of any kind",
        "multimodal fusion results",
        "that synthetic patient records represent real patients",
        "that the human-review warning constitutes a diagnosis",
    ],
    "how_to_proceed": [
        "Obtain an ethically approved clinical dataset with documented variables.",
        "Implement and validate the clinical risk model separately from the image model.",
        "Only then implement the fuzzy conflict-resolution layer and validate it.",
        "Keep this status record updated alongside the implementation.",
    ],
}


def status_record() -> Dict[str, Any]:
    """Return a copy of the component status record."""
    import copy

    return copy.deepcopy(STATUS_RECORD)


def is_implemented() -> bool:
    return False


def refuse_synthetic_clinical_data(*_args: Any, **_kwargs: Any) -> None:
    """Explicit guard: the pipeline must never fabricate clinical records.

    Called by the orchestrator in place of a fusion stage. Raises so that a future
    caller cannot mistake this module for a working implementation.
    """
    raise NotImplementedError(
        "Clinical/fuzzy fusion is PROPOSED and not implemented. The pipeline does not "
        "fabricate clinical records. See src/clinical_status.py for the exact gap."
    )
