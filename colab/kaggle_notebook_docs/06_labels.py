# cell -----------------------------------------------------------------------
# source: dataset\labels.py
"""Label mapping for the provisional primary and secondary tasks.

This module is explicit and deterministic. Unknown source labels are rejected
rather than silently mapped to a default class.

Notes
-----
The primary mapping reflects the provisional approval recorded in
``audit_output/pretraining_audit/LABEL_POLICY_DRAFT.md``. The merges
(Very Severe NPDR -> Severe NPDR, Advanced PDR -> PDR) are provisional, and
original source labels are preserved elsewhere. This module does not imply
specialist validation.
"""
from __future__ import annotations

from typing import Dict, Iterable, Sequence

# Exact source labels present in the manifests (verbatim folder names).
SOURCE_LABELS: Sequence[str] = [
    "1.Dry AMD",
    "2.Wet AMD",
    "3.Mild DR",
    "4.Moderate DR",
    "5.Severe DR",
    "6.Proliferate DR",
    "7.Cataract",
    "8.Hypertensive Retinopathy",
    "9.Pathological Myopia",
    "10.Glaucoma",
    "11.Normal Fundus",
    "12. Advanced PDR",
    "13. Very Severe NPDR",
]

PRIMARY_MAPPING: Dict[str, str] = {
    "11.Normal Fundus": "No apparent DR",
    "3.Mild DR": "Mild NPDR",
    "4.Moderate DR": "Moderate NPDR",
    "5.Severe DR": "Severe NPDR",
    "13. Very Severe NPDR": "Severe NPDR",
    "6.Proliferate DR": "PDR",
    "12. Advanced PDR": "PDR",
}

NON_DR_LABELS: set[str] = {
    "1.Dry AMD",
    "2.Wet AMD",
    "7.Cataract",
    "8.Hypertensive Retinopathy",
    "9.Pathological Myopia",
    "10.Glaucoma",
}

PRIMARY_ELIGIBLE_SOURCE_LABELS: set[str] = set(PRIMARY_MAPPING.keys())


def primary_task_label(source_label: str) -> str:
    """Return the provisional primary-task label for a known source label.

    Raises
    ------
    ValueError
        If ``source_label`` is unknown or not part of the primary mapping.
    """
    if source_label not in PRIMARY_MAPPING:
        raise ValueError(f"Unknown or non-primary source label: {source_label!r}")
    return PRIMARY_MAPPING[source_label]


def secondary_task_label(source_label: str) -> str:
    """Return the secondary-task label (verbatim original source label).

    Raises
    ------
    ValueError
        If ``source_label`` is not one of the known 13 source labels.
    """
    if source_label not in SOURCE_LABELS:
        raise ValueError(f"Unknown source label: {source_label!r}")
    return source_label


def is_primary_eligible_source(source_label: str) -> bool:
    """Return True if the source label is eligible for the provisional primary task."""
    return source_label in PRIMARY_ELIGIBLE_SOURCE_LABELS


def is_non_dr_source(source_label: str) -> bool:
    """Return True if the source label is one of the non-DR conditions."""
    return source_label in NON_DR_LABELS


def validate_source_labels(labels: Iterable[str]) -> None:
    """Raise ValueError on the first unknown source label encountered."""
    for label in labels:
        if label not in SOURCE_LABELS:
            raise ValueError(f"Unknown source label encountered: {label!r}")


def primary_band_counts_from_rows(rows: Iterable[Dict[str, str]]) -> Dict[str, int]:
    """Count primary-task bands from manifest rows that already carry a task_label.

    This is intended for reconciliation checks against existing manifests. It
    expects ``task_label`` to already be populated (as in the eligibility
    manifests).
    """
    from collections import Counter

    return dict(
        Counter(row.get("task_label", "") for row in rows if row.get("task_label", ""))
    )

# end cell -------------------------------------------------------------------
