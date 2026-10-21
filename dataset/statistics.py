"""Class-count reporting for the provisional tasks.

This module produces counts from manifest rows without recomputing eligibility.
Counts are driven by the manifest's ``eligibility`` and ``task_label``/``original_label``
columns.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from .eligibility import is_primary_eligible


def primary_eligible_class_counts(rows: Sequence[Dict[str, str]]) -> Dict[str, int]:
    """Return counts of eligible primary images by task_label.

    Only rows with ``eligibility == "eligible_provisional"`` are counted.
    """
    from collections import Counter

    counts = Counter()
    for row in rows:
        if is_primary_eligible(row):
            label = row.get("task_label", "")
            if label:
                counts[label] += 1
    return dict(sorted(counts.items()))


def secondary_eligible_class_counts(
    rows: Sequence[Dict[str, str]], *, allow_zero_count_classes: bool = False
) -> Dict[str, int]:
    """Return counts of eligible secondary images by original_label.

    Only rows with ``eligibility == "eligible_provisional"`` are counted.

    Parameters
    ----------
    allow_zero_count_classes:
        If True, eligible classes with zero rows still appear with a count of 0.
        This is only used by reconciliation/smoke tests.
    """
    from collections import Counter

    from dataset.labels import SOURCE_LABELS

    counts = Counter()
    for row in rows:
        if row.get("eligibility", "") == "eligible_provisional":
            label = row.get("original_label", "")
            if label:
                counts[label] += 1
    if allow_zero_count_classes:
        result: Dict[str, int] = {}
        for label in SOURCE_LABELS:
            result[label] = counts.get(label, 0)
        return dict(sorted(result.items()))
    return dict(sorted(counts.items()))


def full_status_counts(rows: Sequence[Dict[str, str]]) -> Dict[str, int]:
    """Return counts of every eligibility status in the manifest.

    This is useful for reconciliation against the existing audit checkpoint.
    """
    from collections import Counter

    return dict(Counter(row.get("eligibility", "") for row in rows))


def excluded_and_quarantined_counts(rows: Sequence[Dict[str, str]]) -> Dict[str, int]:
    """Return separate excluded and quarantined counts for traceability.

    Returns
    -------
    A dict with keys for the eligible count, the quality-quarantined count,
    the conflict-quarantined count, the both-quarantined count, and the
    excluded-non-DR count (primary-manifest-style decomposition).
    """
    from collections import Counter

    counts: Dict[str, int] = Counter()
    for row in rows:
        status = row.get("eligibility", "")
        if status == "eligible_provisional":
            counts["eligible"] += 1
        elif status == "quarantined_quality_review":
            counts["quarantined_quality_review"] += 1
        elif status == "quarantined_label_conflict":
            counts["quarantined_label_conflict"] += 1
        elif status == "quarantined_conflict_and_quality":
            counts["quarantined_conflict_and_quality"] += 1
        elif status.startswith("excluded_non_dr_primary"):
            counts["excluded_non_dr_primary"] += 1
        else:
            counts["other_excluded"] += 1
    return dict(counts)
