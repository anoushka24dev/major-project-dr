"""Grouping metadata handling for the provisional manifests.

This module uses the manifest's existing grouping fields and does not infer
patient identity from filenames or proximity.

Important limitations
--------------------
- A duplicate-connected group is not automatically a verified patient.
- Some Zenodo-derived images have no recoverable patient ID.
- This module never claims patient-independent evaluation where patient
  identities are unavailable.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set

from .manifest import Record


def rows_by_group_id(rows: Sequence[Record]) -> Dict[str, List[Record]]:
    """Group rows by ``split_group_id``.

    Rows missing a split_group_id are grouped under the empty-string key.
    """
    from collections import defaultdict

    grouped: Dict[str, List[Record]] = defaultdict(list)
    for row in rows:
        grouped[row.get("split_group_id", "")].append(row)
    return dict(grouped)


def rows_by_duplicate_group_id(rows: Sequence[Record]) -> Dict[str, List[Record]]:
    """Group rows by ``duplicate_group_id`` where present.

    Rows without a duplicate_group_id are omitted from the returned mapping.
    """
    from collections import defaultdict

    grouped: Dict[str, List[Record]] = defaultdict(list)
    for row in rows:
        g = row.get("duplicate_group_id", "") or ""
        if g:
            grouped[g].append(row)
    return dict(grouped)


def known_group_ids(rows: Sequence[Record]) -> Set[str]:
    """Return the set of non-empty duplicate_group_id values present in the rows."""
    return {
        row.get("duplicate_group_id", "")
        for row in rows
        if row.get("duplicate_group_id")
    }


def group_membership_map(rows: Sequence[Record]) -> Dict[str, Set[str]]:
    """Return a map from duplicate_group_id to the set of image_path members.

    Only rows with a non-empty duplicate_group_id are included.
    """
    from collections import defaultdict

    members: Dict[str, Set[str]] = defaultdict(set)
    for row in rows:
        g = row.get("duplicate_group_id", "") or ""
        if g:
            members[g].add(row.get("image_path", ""))
    return {k: frozenset(v) for k, v in members.items()}


def has_patient_group_key(row: Record) -> bool:
    """Return True if the row has a non-empty, non-'none' patient_group_key."""
    key = (row.get("patient_group_key") or "").strip()
    if not key:
        return False
    return key.strip().lower() != "none"


def zenodo_unlinked_rows(rows: Sequence[Record]) -> List[Record]:
    """Return Zenodo rows that have no usable patient_group_key.

    These should be reported as image-level groupings only.
    """
    from collections import defaultdict

    out: List[Record] = []
    for row in rows:
        if (row.get("source_family") or "").lower() == "zenodo_v02" and not has_patient_group_key(row):
            out.append(row)
    return out


def summarize_grouping(rows: Sequence[Record]) -> Dict:
    """Return a small grouping summary for reporting."""
    from collections import Counter

    split_groups = rows_by_group_id(rows)
    dup_groups = rows_by_duplicate_group_id(rows)
    zenodo_unlinked = zenodo_unlinked_rows(rows)
    return {
        "split_group_count": len(split_groups),
        "duplicate_group_count": len(dup_groups),
        "duplicate_connected_rows": sum(len(v) for v in dup_groups.values()),
        "zenodo_unlinked_rows": len(zenodo_unlinked),
        "has_patient_key_rows": sum(1 for r in rows if has_patient_group_key(r)),
    }
