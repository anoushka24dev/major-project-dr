# cell -----------------------------------------------------------------------
# source: dataset\split_validation.py
"""Split-validation contract for future split assignments.

This module does NOT generate splits. It validates an externally supplied
split assignment against manifest membership, duplicate-group integrity, and
quarantine/exclusion rules.

Limitations
-----------
- Unknown image identifiers are rejected.
- A duplicate-connected group is validated as a group, not as a verified patient.
- Patient independence is not claimed for Zenodo rows lacking verified patient
  identity.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set

from .manifest import Record

ALLOWED_DEFAULT_PARTITIONS = {"train", "validation", "test"}


class SplitValidationError(ValueError):
    """Raised when a supplied split assignment violates validation rules."""


PartitionName = str  # e.g. "train", "validation", "test"    Assignment = Mapping[str, PartitionName] | list[tuple[str, PartitionName]]  # image_path -> partition


class SplitValidator:
    """Validates a split assignment against a manifest and grouping metadata."""

    def __init__(
        self,
        manifest_rows: Sequence[Record],
        *,
        allowed_partitions: Sequence[str] | None = None,
        exclude_quarantined: bool = True,
        exclude_non_dr_primary: bool = True,
    ):
        self._rows = list(manifest_rows)
        self._by_path = {row.get("image_path", ""): row for row in self._rows}
        self._allowed_partitions = (
            set(allowed_partitions) if allowed_partitions else ALLOWED_DEFAULT_PARTITIONS
        )
        self._exclude_quarantined = exclude_quarantined
        self._exclude_non_dr_primary = exclude_non_dr_primary
        self._group_members = _build_member_sets(self._rows)

    def validate(self, assignment: Assignment) -> Dict[str, Any]:
        """Validate an assignment.

        Returns a report dict. Raises ``SplitValidationError`` on the first
        violation encountered.
        """
        errors: List[str] = []
        warnings: List[str] = []

        raw_items = _as_items(assignment)
        self._check_unknown_paths_from_items(raw_items, errors)
        self._check_duplicate_keys_from_items(raw_items, errors)
        if errors:
            raise SplitValidationError("; ".join(errors))

        normalized = _normalize_assignment_from_items(raw_items)
        self._check_partitions_valid(normalized, errors)
        self._check_quarantine_and_exclusion(normalized, errors, warnings)
        self._check_group_integrity(normalized, errors)

        if errors:
            raise SplitValidationError("; ".join(errors))

        return {
            "total_assigned": len(normalized),
            "partition_counts": _partition_counts(normalized, self._allowed_partitions),
            "errors": errors,
            "warnings": warnings,
            "valid": True,
        }

    def _check_unknown_paths_from_items(self, items: list[tuple[str, PartitionName]], errors: List[str]) -> None:
        for path, _ in items:
            if path not in self._by_path:
                errors.append(f"Unknown image_path in assignment: {path!r}")

    def _check_duplicate_keys_from_items(self, items: list[tuple[str, PartitionName]], errors: List[str]) -> None:
        # Detect duplicate keys in the source assignment.
        seen: dict[str, int] = {}
        for idx, (path, _) in enumerate(items):
            earlier = seen.get(path)
            if earlier is not None:
                errors.append(f"Image assigned to more than one partition: {path!r}")
            else:
                seen[path] = idx

    def _check_partitions_valid(self, assignment: Mapping[str, PartitionName], errors: List[str]) -> None:
        for path, part in assignment.items():
            if part not in self._allowed_partitions:
                errors.append(f"Invalid partition for {path!r}: {part!r}")

    def _check_quarantine_and_exclusion(
        self, assignment: Mapping[str, PartitionName], errors: List[str], warnings: List[str]
    ) -> None:
        for path, part in assignment.items():
            row = self._by_path.get(path)
            if row is None:
                continue
            if self._exclude_quarantined and is_any_quarantined(row):
                errors.append(
                    f"Quarantined image assigned to {part!r}: {path!r}"
                )
            if self._exclude_non_dr_primary and is_primary_excluded_non_dr(row):
                errors.append(
                    f"Excluded non-DR image assigned to {part!r}: {path!r}"
                )

    def _check_group_integrity(self, assignment: Mapping[str, PartitionName], errors: List[str]) -> None:
        for group_id, members in self._group_members.items():
            assigned_in_assignment = [p for p in members if p in assignment]
            if not assigned_in_assignment:
                continue
            partitions = {assignment[p] for p in assigned_in_assignment}
            if len(partitions) > 1:
                errors.append(
                    f"Duplicate-connected group {group_id!r} split across partitions: {partitions}"
                )

    def _check_no_double_assignment(self, assignment: Mapping[str, PartitionName], errors: List[str]) -> None:
        # After normalization there is at most one entry per path, so this is
        # only a safety check for the normalized assignment.
        assigned_paths: list[str] = []
        for path, _ in assignment.items():
            if path in assigned_paths:
                errors.append(f"Image assigned to more than one partition: {path!r}")
            assigned_paths.append(path)


def is_any_quarantined(row: Record) -> bool:
    from .eligibility import is_any_quarantined as _qa

    return _qa(row)


def is_primary_excluded_non_dr(row: Record) -> bool:
    from .eligibility import is_primary_excluded_non_dr as _ex

    return _ex(row)


def _build_member_sets(rows: Sequence[Record]) -> Dict[str, Set[str]]:
    from collections import defaultdict

    members: Dict[str, Set[str]] = defaultdict(set)
    for row in rows:
        g = row.get("duplicate_group_id") or ""
        path = row.get("image_path", "")
        if g and path:
            members[g].add(path)
    return dict(members)


def _as_items(assignment: Assignment) -> list[tuple[str, PartitionName]]:
    """Return assignment items from the raw input.

    If the caller passes a list of tuples, use it directly so duplicate keys
    are visible. If the caller passes a mapping, use its items().
    """
    if isinstance(assignment, list):
        return list(assignment)
    return list(assignment.items())


def _normalize_assignment_from_items(items: list[tuple[str, PartitionName]]) -> Dict[str, PartitionName]:
    """Return a single-entry-per-path mapping, keeping the last value for each path."""
    normalized: Dict[str, PartitionName] = {}
    for path, part in items:
        normalized[path] = part
    return normalized


def _partition_counts(assignment: Mapping[str, PartitionName], allowed: Set[str]) -> Dict[str, int]:
    from collections import Counter

    counts: Dict[str, int] = {p: 0 for p in sorted(allowed)}
    for part in assignment.values():
        if part in counts:
            counts[part] += 1
    return counts

# end cell -------------------------------------------------------------------
