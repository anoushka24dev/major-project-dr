"""
Split-generation protocol for the provisional primary 5-band DR grading task.

Design notes
------------
* Eligibility is read from the authoritative manifest column; this module never
  relabels, filters beyond eligibility, or mutates the audit artifacts.
* Atomic split unit = duplicate-connected component built from
  ``duplicate_group_id`` edges among eligible rows only.
* Stratification is group-aware and class-aware: whole components are assigned,
  and the allocation targets approximate 70/15/15 image proportions per class.
* Zenodo-derived eligible rows with no verified patient linkage are preserved
  in the assignment and reported per partition; patient independence is **not**
  claimed for them.
* The generated assignment must still pass the existing ``SplitValidator``
  before it is treated as valid output.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .eligibility import filter_primary_eligible
from .grouping import group_membership_map
from .labels import primary_task_label
from .manifest import Record


BandName = str
Component = Tuple[str, ...]
Assignment = Dict[str, str]  # image_path -> partition name


class SplitGenerationError(ValueError):
    """Raised when a split allocation violates a hard safeguard."""


DEFAULT_PARTITIONS = ("train", "validation", "test")
DEFAULT_TARGET_FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
CLASS_ORDER = [
    "No apparent DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "PDR",
]


def _make_rng(seed: int):
    # Deterministic RNG without importing numpy / sklearn.
    import random

    rng = random.Random(seed)
    return rng


def _stable_component_key(component: Sequence[str]) -> str:
    """Deterministic key for a component used for stable sorting."""
    return ",".join(sorted(component))


def _component_band(component: Sequence[str], rows_by_path: Mapping[str, Record]) -> BandName:
    bands = {primary_task_label(rows_by_path[p]["original_label"]) for p in component}
    if len(bands) != 1:
        raise SplitGenerationError(
            f"Polymorphic duplicate-connected component encountered: "
            f"{sorted(bands)} for {sorted(component)}"
        )
    return next(iter(bands))


def _component_source_meta(
    component: Sequence[str], rows_by_path: Mapping[str, Record]
) -> Dict[str, Any]:
    zenodo_eligible = 0
    zenodo_pure = True
    families: set[str] = set()
    for p in component:
        row = rows_by_path[p]
        sf = (row.get("source_family") or "").lower()
        families.add(sf)
        if sf == "zenodo_v02":
            zenodo_eligible += 1
            if sf != "zenodo_v02":
                zenodo_pure = False
    zenodo_pure = families == {"zenodo_v02"}
    return {
        "zenodo_eligible": zenodo_eligible,
        "zenodo_pure_singleton": zenodo_pure and len(component) == 1,
        "families": sorted(families),
    }


def _component_members(component: Sequence[str]) -> List[str]:
    # Order members deterministically so a component's internal ordering is
    # stable across calls; only the partition label matters for leakage, but
    # stable ordering helps reproducibility and debugging.
    return sorted(component)


class PrimarySplitProtocol:
    """Deterministic, group-aware, class-stratified primary split generator.

    Public contract
    ---------------
    * ``build_components_from_eligible_rows`` reconstructs the atomic units from
      the eligible rows of a primary manifest.
    * ``allocate`` assigns every component to exactly one partition while aiming
      for 70/15/15 image proportions per band.
    * ``allocate`` returns both the assignment and an audit-ready report.

    Limitations handled here
    ------------------------
    * No patient identities are inferred.
    * Zenodo rows are not dropped and are reported explicitly.
    * Polymorphic duplicate components are treated as a hard error because they
      would break the band-straitification invariant for this task.
    """

    def __init__(
        self,
        seed: int,
        *,
        target_fractions: Mapping[str, float] | None = None,
        near_zenodo_test_frac: float = 0.10,
        allowed_partitions: Sequence[str] = DEFAULT_PARTITIONS,
    ):
        if set(allowed_partitions) != set(DEFAULT_PARTITIONS):
            raise ValueError("Only train/validation/test partitions are supported.")
        self.seed = seed
        self.target_fractions = dict(DEFAULT_TARGET_FRACTIONS)
        if target_fractions:
            self.target_fractions.update(target_fractions)
        self.near_zenodo_test_frac = float(near_zenodo_test_frac)
        if not 0 <= self.near_zenodo_test_frac <= 1:
            raise ValueError("near_zenodo_test_frac must be in [0,1]")
        self.allowed_partitions = list(allowed_partitions)

    def build_components_from_eligible_rows(
        self, primary_rows: Sequence[Record], image_root: Path | None = None
    ) -> List[Component]:
        """Reconstruct duplicate-connected components from eligible primary rows.

        Parameters
        ----------
        primary_rows:
            Rows from the primary manifest (pre-filtered to eligible rows is
            recommended, but the function will filter internally for safety).
        image_root:
            Accepted for API symmetry with the manifest loader; not used here
            because path existence is already guaranteed by the manifest load.

        Returns
        -------
        A list of components, where each component is a tuple of image_path
        strings that must be assigned together. Every eligible primary image
        path appears in exactly one returned component.
        """
        eligible = filter_primary_eligible(primary_rows)
        if not eligible:
            raise SplitGenerationError("No primary-eligible rows available to split.")

        by_path = {row["image_path"]: row for row in eligible}
        self._validate_eligible_by_path(by_path)

        mg = group_membership_map(eligible)
        return _reconstruct_components(by_path, mg)

    def allocate(
        self,
        components: Sequence[Component],
        rows_by_path: Mapping[str, Record],
    ) -> Tuple[Assignment, Dict[str, Any]]:
        """Assign every component to a partition and return the report.

        Parameters
        ----------
        components:
            Atomic components from ``build_components_from_eligible_rows`` (or an
            equivalent reconstruction). Each component is a tuple of image_paths
            that must stay together.
        rows_by_path:
            Mapping from image_path to manifest row for every eligible image.
        """
        components = list(components)
        if not components:
            raise SplitGenerationError("No components to allocate.")

        self._validate_components(components, rows_by_path)

        by_band: Dict[BandName, List[Component]] = defaultdict(list)
        for comp in components:
            band = _component_band(comp, rows_by_path)
            by_band[band].append(comp)

        missing_bands = [b for b in CLASS_ORDER if b not in by_band]
        if missing_bands:
            raise SplitGenerationError(
                f"Expected primary bands missing from eligible components: {missing_bands}"
            )

        # Deterministic ordering of components within each band.
        for band in by_band:
            by_band[band].sort(key=_stable_component_key)

        partition_counts: Counter[str] = Counter()
        partition_by_path: Dict[str, str] = {}
        per_band_partition_counts: Dict[BandName, Counter[str]] = {}
        zenodo_by_partition: Counter[str] = Counter()
        zenodo_pure_by_partition: Counter[str] = Counter()
        component_band_assign: List[Tuple[Component, BandName, str]] = []

        for band in CLASS_ORDER:
            comps = by_band[band]
            n = len(comps)
            if n == 0:
                continue
            target_images = sum(len(c) for c in comps)
            target_by_partition = self._target_images_by_partition(target_images)

            assigned_by_partition: Counter[str] = Counter()
            band_assignment: Dict[str, Component] = {}
            rng = _make_rng(self._band_seed(band))

            # Randomize component order within band, deterministically.
            ordered = list(comps)
            rng.shuffle(ordered)

            # Greedy assignment toward per-band image targets.
            for comp in ordered:
                chosen = self._choose_partition(
                    comp,
                    rows_by_path,
                    target_by_partition,
                    assigned_by_partition,
                    partition_counts,
                    zenodo_by_partition,
                    zenodo_pure_by_partition,
                    band,
                )
                assigned_by_partition[chosen] += 1
                for p in comp:
                    if p in partition_by_path:
                        raise SplitGenerationError(f"Image assigned twice: {p}")
                    partition_by_path[p] = chosen
                partition_counts[chosen] += len(comp)

                for p in comp:
                    row = rows_by_path[p]
                    if (row.get("source_family") or "").lower() == "zenodo_v02":
                        zenodo_by_partition[chosen] += 1
                meta = _component_source_meta(comp, rows_by_path)
                if meta["zenodo_pure_singleton"]:
                    zenodo_pure_by_partition[chosen] += 1

                per_image = len(comp)
                per_band_partition_counts.setdefault(band, Counter())[chosen] += per_image
                component_band_assign.append((comp, band, chosen))

        report = self._build_report(
            partition_by_path,
            partition_counts,
            per_band_partition_counts,
            zenodo_by_partition,
            zenodo_pure_by_partition,
            components,
            rows_by_path,
        )
        return partition_by_path, report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _target_images_by_partition(self, total_images: int) -> Dict[str, int]:
        raw = {p: total_images * f for p, f in self.target_fractions.items()}
        base = {p: int(math.floor(v)) for p, v in raw.items()}
        remainder = total_images - sum(base.values())
        order = sorted(self.allowed_partitions, key=lambda p: (-raw[p], p))
        i = 0
        while remainder > 0:
            base[order[i % len(order)]] += 1
            remainder -= 1
            i += 1
        return base

    def _band_seed(self, band: str) -> int:
        # Deterministic per-band jitter so band order does not alone determine
        # the assignment, while preserving end-to-end determinism for fixed seed.
        import hashlib

        h = hashlib.sha256(f"{self.seed}:band:{band}".encode("utf-8")).hexdigest()
        return int(h[:8], 16)

    def _choose_partition(
        self,
        comp: Component,
        rows_by_path: Mapping[str, Record],
        target_by_partition: Dict[str, int],
        band_assigned: Counter[str],
        global_assigned: Counter[str],
        zenodo_by_partition: Counter[str],
        zenodo_pure_by_partition: Counter[str],
        band: str,
    ) -> str:
        size = len(comp)
        candidates = list(self.allowed_partitions)

        # Prefer partitions still under their per-band image target.
        under_target = []
        at_or_over = []
        for p in candidates:
            if band_assigned[p] + size <= target_by_partition[p]:
                under_target.append(p)
            else:
                at_or_over.append(p)

        pool = under_target if under_target else at_or_over

        # If test would exceed a soft Zenodo cap, deprioritize it unless it is
        # the only feasible choice for this component.
        if self.near_zenodo_test_frac > 0:
            if pool and "test" in pool:
                total_assigned_images = sum(global_assigned.values())
                test_ceiling_global = math.floor(
                    total_assigned_images * self.near_zenodo_test_frac
                ) + size
                if zenodo_by_partition["test"] + size > max(1, test_ceiling_global):
                    pool = [p for p in pool if p != "test"] or pool

        if len(pool) == 1:
            return pool[0]

        # Tie-break deterministically among remaining candidates.
        return self._tie_break_partition(comp, pool, band_assigned, rows_by_path)

    def _tie_break_partition(
        self,
        comp: Component,
        pool: Sequence[str],
        band_assigned: Counter[str],
        rows_by_path: Mapping[str, Record],
    ) -> str:
        import hashlib

        band = _component_band(comp, rows_by_path)
        key = f"{self._band_seed(band)}:{_stable_component_key(comp)}"
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        order = sorted(pool, key=lambda p: int(h, 16) % (len(pool) * 1000) + band_assigned.get(p, 0))
        return order[0]

    def _validate_eligible_by_path(self, by_path: Mapping[str, Record]) -> None:
        seen: set[str] = set()
        for path, row in by_path.items():
            if not path:
                raise SplitGenerationError("Encountered an empty image_path in eligible rows.")
            if path in seen:
                raise SplitGenerationError(f"Duplicate image_path in eligible rows: {path}")
            seen.add(path)
            if (row.get("source_family") or "").lower() not in {"prefixed", "plain", "zenodo_v02"}:
                raise SplitGenerationError(
                    f"Unexpected source_family for {path}: {row.get('source_family')!r}"
                )

    def _validate_components(
        self,
        components: Sequence[Component],
        rows_by_path: Mapping[str, Record],
    ) -> None:
        flat: Counter[str] = Counter()
        for comp in components:
            if not comp:
                raise SplitGenerationError("Encountered an empty component.")
            for p in comp:
                if p not in rows_by_path:
                    raise SplitGenerationError(
                        f"Component references image_path not in eligible set: {p}"
                    )
                flat[p] += 1
        dups = [p for p, n in flat.items() if n > 1]
        if dups:
            raise SplitGenerationError(
                f"Image_path appears in more than one component: {dups[:10]}"
            )
        expected = set(rows_by_path.keys())
        assigned = set(flat.keys())
        if assigned != expected:
            missing = expected - assigned
            extra = assigned - expected
            msg = []
            if missing:
                msg.append(f"eligible images not covered by components: {sorted(missing)[:10]}")
            if extra:
                msg.append(f"components contain non-eligible image_paths: {sorted(extra)[:10]}")
            raise SplitGenerationError("; ".join(msg))

    def _build_report(
        self,
        assignment: Mapping[str, str],
        partition_counts: Counter[str],
        per_band_partition_counts: Dict[BandName, Counter[str]],
        zenodo_by_partition: Counter[str],
        zenodo_pure_by_partition: Counter[str],
        components: Sequence[Component],
        rows_by_path: Mapping[str, Record],
    ) -> Dict[str, Any]:
        total = sum(partition_counts.values())
        per_band_counts: Dict[BandName, Dict[str, int]] = {}
        for band in CLASS_ORDER:
            c = per_band_partition_counts.get(band, Counter())
            per_band_counts[band] = {
                "images": int(sum(c.values())),
                "groups": int(sum(1 for comp in components if _component_band(comp, rows_by_path) == band)),
                **{p: int(c.get(p, 0)) for p in self.allowed_partitions},
            }

        group_atomic = self._check_group_atomic(components, assignment)
        all_bands_in_all_splits = self._check_all_bands_in_all_splits(per_band_partition_counts)

        warnings: List[str] = []
        if not all_bands_in_all_splits:
            missing = self._bands_missing_from_splits(per_band_partition_counts)
            warnings.append(f"The following bands are missing from one or more splits: {missing}")
        if not group_atomic:
            warnings.append("Duplicate-connected component leakage detected (should be impossible by construction).")

        return {
            "seed": self.seed,
            "target_fractions": dict(self.target_fractions),
            "allowed_partitions": list(self.allowed_partitions),
            "total_images": total,
            "partition_counts": {p: int(partition_counts.get(p, 0)) for p in self.allowed_partitions},
            "partition_ratios": {
                p: (partition_counts.get(p, 0) / total) if total else 0.0
                for p in self.allowed_partitions
            },
            "class_counts": per_band_counts,
            "zenodo_by_partition": {p: int(zenodo_by_partition.get(p, 0)) for p in self.allowed_partitions},
            "zenodo_pure_singletons_by_partition": {
                p: int(zenodo_pure_by_partition.get(p, 0)) for p in self.allowed_partitions
            },
            "group_atomic": group_atomic,
            "all_bands_in_all_splits": all_bands_in_all_splits,
            "warnings": warnings,
            "total_components": len(components),
        }

    def _check_group_atomic(
        self,
        components: Sequence[Component],
        assignment: Mapping[str, str],
    ) -> bool:
        for comp in components:
            partitions = {assignment[p] for p in comp}
            if len(partitions) > 1:
                return False
        return True

    def _check_all_bands_in_all_splits(
        self, per_band_partition_counts: Dict[BandName, Counter[str]]
    ) -> bool:
        for band, c in per_band_partition_counts.items():
            if any(c.get(p, 0) == 0 for p in self.allowed_partitions):
                return False
        return True

    def _bands_missing_from_splits(
        self, per_band_partition_counts: Dict[BandName, Counter[str]]
    ) -> List[str]:
        missing: List[str] = []
        for band, c in per_band_partition_counts.items():
            for p in self.allowed_partitions:
                if c.get(p, 0) == 0:
                    missing.append(f"{band} in {p}")
        return missing


def _reconstruct_components(
    rows_by_path: Mapping[str, Record], mg: Mapping[str, set[str]]
) -> List[Component]:
    """Reconstruct duplicate-connected components covering every eligible image.

    Duplicate edges define multi-image components. Every eligible image not in a
    duplicate edge becomes its own singleton component so that the returned list
    partitions the eligible set exactly once.
    """
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for paths in mg.values():
        it = iter(paths)
        try:
            first = next(it)
        except StopIteration:
            continue
        for p in it:
            union(first, p)

    comp: Dict[str, set[str]] = defaultdict(set)
    for p in parent:
        comp[find(p)].add(p)

    for p in rows_by_path:
        if p not in parent:
            comp[p].add(p)

    out: List[Component] = []
    seen: set[str] = set()
    for c in comp.values():
        comp_tuple = tuple(sorted(c))
        if not comp_tuple:
            continue
        if any(p in seen for p in comp_tuple):
            raise SplitGenerationError("Reconstructed overlapping components.")
        out.append(comp_tuple)
        seen.update(comp_tuple)

    if seen != set(rows_by_path.keys()):
        raise SplitGenerationError("Component reconstruction did not cover all eligible images.")

    out.sort(key=_stable_component_key)
    return out


def build_group_table_hash(components: Sequence[Component]) -> str:
    """Stable hash of the component structure used for provenance in reports."""
    canonical = [tuple(sorted(c)) for c in components]
    canonical.sort(key=lambda c: _stable_component_key(c))
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_report_from_assignment(
    assignment: Mapping[str, str],
    rows_by_path: Mapping[str, Record],
    components: Sequence[Component],
    protocol: PrimarySplitProtocol,
    manifest_version: str,
    manifest_path: Path,
    approval_reference: str,
    group_table_hash: str,
    validation_report: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Assemble the final split audit report from an assignment and protocol state."""
    per_band_counts: Dict[str, Dict[str, int]] = {}
    per_band_assigned: Dict[str, Counter[str]] = defaultdict(Counter)
    for p, part in assignment.items():
        band = primary_task_label(rows_by_path[p]["original_label"])
        per_band_assigned[band][part] += 1
    for band in CLASS_ORDER:
        c = per_band_assigned.get(band, Counter())
        per_band_counts[band] = {
            "images": int(sum(c.values())),
            "groups": int(sum(1 for comp in components if _component_band(comp, rows_by_path) == band)),
            **{p: int(c.get(p, 0)) for p in protocol.allowed_partitions},
        }

    zenodo_by_partition: Counter[str] = Counter()
    zenodo_pure_by_partition: Counter[str] = Counter()
    for comp in components:
        meta = _component_source_meta(comp, rows_by_path)
        part = assignment[comp[0]]
        zenodo_by_partition[part] += meta["zenodo_eligible"]
        if meta["zenodo_pure_singleton"]:
            zenodo_pure_by_partition[part] += 1

    total = len(assignment)
    partition_counts = Counter(assignment.values())
    return {
        "manifest_version": manifest_version,
        "manifest_path": str(manifest_path),
        "manifest_file_hash": _file_sha256(manifest_path),
        "approval_reference": approval_reference,
        "seed": protocol.seed,
        "target_fractions": dict(protocol.target_fractions),
        "near_zenodo_test_frac": protocol.near_zenodo_test_frac,
        "group_table_hash": group_table_hash,
        "total_components": len(components),
        "total_images": total,
        "partition_counts": {p: int(partition_counts.get(p, 0)) for p in protocol.allowed_partitions},
        "partition_ratios": {
            p: (partition_counts.get(p, 0) / total) if total else 0.0
            for p in protocol.allowed_partitions
        },
        "class_counts": per_band_counts,
        "zenodo_by_partition": {p: int(zenodo_by_partition.get(p, 0)) for p in protocol.allowed_partitions},
        "zenodo_pure_singletons_by_partition": {
            p: int(zenodo_pure_by_partition.get(p, 0)) for p in protocol.allowed_partitions
        },
        "all_bands_in_all_splits": all(
            all(per_band_assigned[band].get(p, 0) > 0 for p in protocol.allowed_partitions)
            for band in CLASS_ORDER
        ),
        "validation": validation_report,
        "limitations": [
            "Eligibility is provisional and is pinned to the manifest version and approval referenced above.",
            "503 eligible Zenodo-derived images lack verified patient linkage; patient independence cannot be claimed for them.",
            "Within the duplicate-connected-component view, 393 Zenodo-eligible images are pure-Zenodo singletons and 110 are bound into mixed-source duplicate components.",
            "The Severe NPDR and PDR taxonomy merges remain provisional and unsigned.",
            "Human quality review and conflict adjudication are incomplete; quarantined and excluded images are not assigned.",
            "Internal splitting does not establish source or external generalization.",
        ],
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_split_csv(
    assignment: Mapping[str, str],
    rows_by_path: Mapping[str, Record],
    components: Sequence[Component],
    protocol: PrimarySplitProtocol,
    output_path: Path,
) -> None:
    """Write a deterministic split assignment CSV for the primary task."""
    lines: List[str] = [
        "image_path,task,partition,primary_class,group_id,source_family,patient_key_available,zenodo_unlinked,eligibility_snapshot,manifest_version,split_seed"
    ]
    for comp in sorted(components, key=_stable_component_key):
        part = assignment[comp[0]]
        if any(assignment.get(p) != part for p in comp):
            raise SplitGenerationError(f"Component is not atomic in the supplied assignment: {comp[0]}")
        for p in comp:
            row = rows_by_path[p]
            patient_key = row.get("patient_group_key", "")
            patient_available = bool(patient_key and patient_key.strip().lower() not in ("", "none"))
            zenodo_unlinked = (row.get("source_family") or "").lower() == "zenodo_v02"
            group_id = _stable_component_key(comp)
            lines.append(
                f"{p},{_csv_escape('primary')},{_csv_escape(part)},"
                f"{_csv_escape(primary_task_label(row['original_label']))},"
                f"{_csv_escape(group_id)},"
                f"{_csv_escape(row.get('source_family',''))},"
                f"{patient_available},"
                f"{zenodo_unlinked},"
                f"{_csv_escape(row.get('eligibility',''))},"
                f"{_csv_escape(protocol.manifest_version if hasattr(protocol,'manifest_version') else '')},"
                f"{protocol.seed}"
            )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _csv_escape(value: str) -> str:
    s = str(value).replace('"', '""')
    return f'"{s}"'
