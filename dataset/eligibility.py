"""Eligibility and quarantine enforcement for the provisional manifests.

This module reads eligibility from the authoritative manifest columns and does
not recompute eligibility from folder names or image contents. It only filters
and reports.

Notes
-----
Primary eligibility is defined by ``eligibility == "eligible_provisional"``
in the primary manifest. Everything else is excluded or quarantined for the
primary task by that same authoritative field.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

PRIMARY_ELIGIBLE_VALUE = "eligible_provisional"

QUARANTINE_VALUES = {
    "quality": "quarantined_quality_review",
    "conflict": "quarantined_label_conflict",
    "both": "quarantined_conflict_and_quality",
}

EXCLUDED_NON_DR_VALUES: set[str] = {
    "excluded_non_dr_primary",
    "excluded_non_dr_primary_and_quality_quarantine",
    "excluded_non_dr_primary_and_label_conflict_quarantine",
    "excluded_non_dr_primary_and_both_quarantines",
}


def _norm_flag(value: str | None) -> str:
    if value is None:
        return "no"
    v = str(value).strip().lower()
    if v in {"", "none", "n/a", "na", "null"}:
        return "no"
    return "yes" if v == "yes" else "no"


def is_primary_eligible(row: Dict[str, str]) -> bool:
    """Return True if the row is eligible for the primary task according to the manifest."""
    return row.get("eligibility", "") == PRIMARY_ELIGIBLE_VALUE


def is_primary_quarantined(row: Dict[str, str]) -> bool:
    """Return True if the row is quarantined for any reason in the primary task."""
    value = row.get("eligibility", "")
    return value in {
        QUARANTINE_VALUES["quality"],
        QUARANTINE_VALUES["conflict"],
        QUARANTINE_VALUES["both"],
    }


def is_primary_excluded_non_dr(row: Dict[str, str]) -> bool:
    """Return True if the row is excluded from the primary task due to non-DR status."""
    return row.get("eligibility", "") in EXCLUDED_NON_DR_VALUES


def is_any_quarantined(row: Dict[str, str]) -> bool:
    """Return True if the row has any quarantine flag set in the manifest."""
    return (
        _norm_flag(row.get("quarantine_quality_flag", "no")) == "yes"
        or _norm_flag(row.get("quarantine_label_conflict", "no")) == "yes"
    )


def filter_primary_eligible(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    """Return only primary-eligible rows, preserving order."""
    return [row for row in rows if is_primary_eligible(row)]


def filter_secondary_eligible(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    """Return secondary-task eligible rows according to the secondary manifest."""
    return [row for row in rows if row.get("eligibility", "") == PRIMARY_ELIGIBLE_VALUE]


def traceable_excluded_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    """Return rows that are excluded/quarantined so they remain traceable rather than deleted."""
    return [row for row in rows if not is_primary_eligible(row)]


def eligible_summary(rows: Sequence[Dict[str, str]], *, label_fn=None) -> Dict[str, int]:
    """Summarize eligible rows by label.

    Parameters
    ----------
    rows:
        Manifest rows.
    label_fn:
        Optional callable returning the label to count. If None, uses
        ``task_label`` for primary-style summaries.
    """
    from collections import Counter

    if label_fn is None:
        label_fn = lambda r: r.get("task_label", "")

    return dict(Counter(label_fn(r) for r in rows if is_primary_eligible(r)))
