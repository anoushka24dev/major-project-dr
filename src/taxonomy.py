"""The five-class taxonomy and the auditable per-dataset label mapping.

Every image's original label is preserved. The mapping from an original label to
a numeric class is declared in ``config/config.yaml`` together with the rule that
justifies it, and every decision — including exclusions — is written to
``outputs/manifests/label_mapping_audit.csv``.

Design rules enforced here
--------------------------
* An unknown label raises immediately. Nothing is silently defaulted.
* A label can never be both mapped and excluded.
* Excluded images are counted and reasoned, never deleted.
* A merge (e.g. Very Severe NPDR -> Severe NPDR) is recorded as a merge, so the
  report can distinguish verbatim grades from merged bands.
* Mendeley and MMRDR grades are mapped independently. Similarly named labels from
  two datasets are never assumed to mean the same thing; each has its own rule.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from .config import Config

DECISION_MAPPED = "mapped"
DECISION_MERGED = "mapped_merged"
DECISION_EXCLUDED = "excluded"

AUDIT_FILENAME = "label_mapping_audit.csv"
AUDIT_COLUMNS = (
    "source_dataset",
    "original_label",
    "mapped_label",
    "mapped_class_name",
    "image_count",
    "mapping_decision",
    "mapping_rule",
    "exclusion_reason",
)


class UnknownLabelError(ValueError):
    """Raised when a source label has no declared mapping or exclusion."""


class AmbiguousLabelError(ValueError):
    """Raised when a source label is declared both mapped and excluded."""


@dataclass
class MappingDecision:
    """The audited outcome of mapping one original label."""

    source_dataset: str
    original_label: str
    mapped_label: Optional[int]
    decision: str
    rule: str = ""
    exclusion_reason: str = ""
    image_count: int = 0

    @property
    def mapped_class_name(self) -> str:
        return "" if self.mapped_label is None else _CLASS_NAMES[self.mapped_label]

    def as_row(self) -> Dict[str, str]:
        return {
            "source_dataset": self.source_dataset,
            "original_label": self.original_label,
            "mapped_label": "" if self.mapped_label is None else str(self.mapped_label),
            "mapped_class_name": self.mapped_class_name,
            "image_count": str(self.image_count),
            "mapping_decision": self.decision,
            "mapping_rule": self.rule,
            "exclusion_reason": self.exclusion_reason,
        }


_CLASS_NAMES: List[str] = []


def set_class_names(names: Sequence[str]) -> None:
    """Bind the class-name list used for reporting."""
    global _CLASS_NAMES
    _CLASS_NAMES = [str(n) for n in names]


class LabelMapper:
    """Maps source labels to five-class indices using the configured rules."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.class_names = list(cfg.classes)
        set_class_names(self.class_names)

        self._rules: Dict[str, Dict[str, Dict]] = {}
        self._exclusions: Dict[str, Dict[str, str]] = {}
        # Every source declared in config label_mapping is registered, so a new
        # dataset cannot be silently mapped through another dataset's rules.
        mapping = cfg.get("label_mapping") or {}
        for source, section in mapping.items():
            if not isinstance(section, dict) or not section.get("rules"):
                continue
            name = str(source).lower()
            self._rules[name] = {str(k): dict(v) for k, v in section["rules"].items()}
            self._exclusions[name] = {
                str(k): str(v)
                for k, v in (section.get("exclude_non_dr") or {}).items()
            }
        for source in ("mendeley", "mmrdr"):
            self._rules.setdefault(source, {})
            self._exclusions.setdefault(source, {})

        self._counts: Dict[tuple[str, str], int] = {}
        self._decisions: Dict[tuple[str, str], MappingDecision] = {}

    # -- mapping ---------------------------------------------------------
    def resolve(self, source_dataset: str, original_label: str) -> MappingDecision:
        """Return the decision for one original label, raising when undeclared."""
        source = str(source_dataset).lower()
        if source not in self._rules:
            raise UnknownLabelError(
                f"No label-mapping rules are configured for source dataset {source_dataset!r}. "
                f"Known sources: {sorted(self._rules)}"
            )
        key = str(original_label)
        rule_entry = self._rules[source].get(key)
        exclusion = self._exclusions.get(source, {}).get(key)

        if rule_entry is not None and exclusion is not None:
            raise AmbiguousLabelError(
                f"{source}: label {key!r} is declared both as a DR grade and as a non-DR "
                "exclusion. Fix config.label_mapping before running."
            )

        if exclusion is not None:
            decision = MappingDecision(
                source_dataset=source,
                original_label=key,
                mapped_label=None,
                decision=DECISION_EXCLUDED,
                exclusion_reason=exclusion,
            )
            return decision

        if rule_entry is None:
            raise UnknownLabelError(
                f"{source}: label {key!r} has no declared mapping and no declared exclusion.\n"
                f"  Known mapped labels   : {sorted(self._rules[source])}\n"
                f"  Known excluded labels : {sorted(self._exclusions.get(source, {}))}\n"
                "Add an explicit rule or exclusion to config/config.yaml. "
                "The pipeline will not guess a clinical grade."
            )

        mapped = int(rule_entry["mapped_label"])
        rule_text = str(rule_entry.get("rule", "")).strip()
        is_merge = "merge" in rule_text.lower()
        decision = MappingDecision(
            source_dataset=source,
            original_label=key,
            mapped_label=mapped,
            decision=DECISION_MERGED if is_merge else DECISION_MAPPED,
            rule=rule_text,
        )
        return decision

    def record(self, decision: MappingDecision) -> None:
        """Accumulate an image count for a decision."""
        key = (decision.source_dataset, decision.original_label)
        self._decisions[key] = decision
        self._counts[key] = self._counts.get(key, 0) + 1

    def record_many(self, source_dataset: str, original_labels: Iterable[str]) -> None:
        for label in original_labels:
            self.record(self.resolve(source_dataset, label))

    # -- audit -----------------------------------------------------------
    def decisions(self) -> List[MappingDecision]:
        out: List[MappingDecision] = []
        for key, decision in self._decisions.items():
            resolved = MappingDecision(**{**decision.__dict__, "image_count": self._counts.get(key, 0)})
            out.append(resolved)
        out.sort(key=lambda d: (d.source_dataset, d.mapped_label if d.mapped_label is not None else 99, d.original_label))
        return out

    def counts_by_class(self, source_dataset: Optional[str] = None) -> Dict[str, int]:
        counts = {name: 0 for name in self.class_names}
        for (source, _label), count in self._counts.items():
            if source_dataset is not None and source != source_dataset.lower():
                continue
            decision = self._decisions[(source, _label)]
            if decision.mapped_label is not None:
                counts[self.class_names[decision.mapped_label]] += count
        return counts

    def excluded_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for (source, label), count in self._counts.items():
            decision = self._decisions[(source, label)]
            if decision.mapped_label is None:
                out[f"{source}:{label}"] = out.get(f"{source}:{label}", 0) + count
        return out

    def write_audit(self, path: Path) -> Path:
        """Write the mapping audit CSV."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = [decision.as_row() for decision in self.decisions()]
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(AUDIT_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)
        return target

    def require_all_classes_present(self, minimum_per_class: int = 1) -> List[str]:
        """Return a list of problems when a class has too few mapped images."""
        problems: List[str] = []
        counts = self.counts_by_class()
        for name, count in counts.items():
            if count < minimum_per_class:
                problems.append(
                    f"class {name!r} has {count} images (< {minimum_per_class} required)"
                )
        return problems


# ---------------------------------------------------------------------------
# Cross-check against the repository's pre-existing label policy
# ---------------------------------------------------------------------------

def cross_check_mendeley_policy(cfg: Config) -> Dict[str, object]:
    """Compare the config mapping with ``dataset.labels.PRIMARY_MAPPING``.

    The repository already contains an approved (provisional) Mendeley mapping
    and a test suite pinned to it. The unified pipeline must not diverge from it
    silently, so this comparison is executed and its result reported.

    The comparison is on the *mapped class index*, which is the real semantic
    contract; the human-readable class strings are additionally reported when
    they differ so a rename cannot hide a re-grade.
    """
    from dataset.labels import PRIMARY_MAPPING, NON_DR_LABELS

    # Index of each canonical band, from the repository's own band order.
    band_order: List[str] = list(cfg.classes)
    try:
        from .legacy_bridge import load_colab_helpers

        band_order = list(load_colab_helpers(cfg.root).BAND_ORDER)
    except Exception:  # pragma: no cover - legacy module absent
        pass

    classes = list(cfg.classes)
    configured_index: Dict[str, int] = {}
    for label, spec in (cfg.get("label_mapping.mendeley.rules", {}) or {}).items():
        configured_index[str(label)] = int(spec["mapped_label"])

    def _index_of(name: str) -> Optional[int]:
        if name in band_order:
            return band_order.index(name)
        if name in classes:
            return classes.index(name)
        return None

    disagreements: Dict[str, Dict[str, object]] = {}
    for label in set(configured_index) | set(PRIMARY_MAPPING):
        reference_name = PRIMARY_MAPPING.get(label)
        reference_index = _index_of(reference_name) if reference_name else None
        mine = configured_index.get(label)
        if mine != reference_index:
            disagreements[label] = {
                "config_mapped_index": mine,
                "config_mapped_class": None if mine is None else classes[mine],
                "dataset_labels_mapped_class": reference_name,
                "dataset_labels_mapped_index": reference_index,
            }

    excluded_configured = set(cfg.get("label_mapping.mendeley.exclude_non_dr", {}) or {})
    exclusion_mismatch = sorted(excluded_configured.symmetric_difference(NON_DR_LABELS))

    return {
        "consistent": not disagreements and not exclusion_mismatch,
        "mapping_disagreements": disagreements,
        "exclusion_mismatch": exclusion_mismatch,
    }
