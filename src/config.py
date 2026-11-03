"""Configuration loading, validation and path resolution.

All pipeline behaviour is driven by ``config/config.yaml``. This module is the
only place that knows how the YAML maps onto typed values and absolute paths,
so no other module needs to re-derive paths or scatter magic numbers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

try:  # PyYAML is a hard dependency of the pipeline, but keep the error readable.
    import yaml
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "PyYAML is required to read config/config.yaml. "
        "Run RUN.bat, or `python scripts/setup_env.py`."
    ) from exc

CONFIG_ENV_VAR = "DR_CONFIG"
FUNDUS_ROOT_ENV_VAR = "DR_FUNDUS_ROOT"
IDRID_ROOT_ENV_VAR = "DR_IDRID_ROOT"

DEFAULT_CONFIG_RELATIVE = Path("config") / "config.yaml"

REQUIRED_TOP_LEVEL = (
    "project",
    "paths",
    "datasets",
    "taxonomy",
    "label_mapping",
    "preprocessing",
    "split",
    "classification",
    "segmentation",
    "evaluation",
    "download",
    "run",
)

#: Placeholders expanded inside YAML string values.
_HOME_TOKENS = ("~", "$HOME", "%USERPROFILE%")


class ConfigError(ValueError):
    """Raised when the configuration file is missing or invalid."""


def project_root(start: Optional[Path | str] = None) -> Path:
    """Return the repository root.

    Resolution order: ``start`` if given, else the parent of this file's
    directory (``src/`` lives directly under the root), else the current working
    directory. Using ``__file__`` keeps the pipeline working when the launcher
    is double-clicked from any directory.
    """
    if start is not None:
        return Path(start).resolve()
    return Path(__file__).resolve().parent.parent


def default_config_path(root: Optional[Path | str] = None) -> Path:
    """Return the default config path, honouring ``DR_CONFIG``."""
    env = os.environ.get(CONFIG_ENV_VAR, "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (project_root(root) / DEFAULT_CONFIG_RELATIVE).resolve()


def _expand(value: str) -> str:
    text = str(value)
    if any(text.startswith(token) for token in _HOME_TOKENS):
        return str(Path(os.path.expandvars(text)).expanduser())
    return os.path.expandvars(text)


def _walk_expand(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _walk_expand(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_walk_expand(v) for v in node]
    if isinstance(node, str):
        return _expand(node)
    return node


class Config:
    """Immutable-ish wrapper around the parsed YAML with dotted access."""

    def __init__(self, data: Mapping[str, Any], *, root: Path, source: Path) -> None:
        self._data: Dict[str, Any] = dict(data)
        self.root = Path(root)
        self.source = Path(source)

    # -- access -----------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        """Return a nested value by dotted key, e.g. ``"split.ratios.train"``."""
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        """Return a nested value, raising :class:`ConfigError` when absent."""
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise ConfigError(f"Missing required configuration key: {dotted!r}")
        return value

    def section(self, name: str) -> Dict[str, Any]:
        value = self.get(name, {})
        if not isinstance(value, dict):
            raise ConfigError(f"Configuration section {name!r} must be a mapping.")
        return value

    def as_dict(self) -> Dict[str, Any]:
        import copy

        return copy.deepcopy(self._data)

    # -- derived paths ----------------------------------------------------
    def path(self, dotted: str, *, create: bool = False) -> Path:
        """Resolve a configured relative path against the project root."""
        raw = self.require(dotted)
        candidate = Path(str(raw))
        if not candidate.is_absolute():
            candidate = self.root / candidate
        candidate = candidate.resolve()
        if create:
            candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    def optional_path(self, dotted: str) -> Optional[Path]:
        """Return a resolved path, or None when unset/empty."""
        raw = self.get(dotted, "")
        if raw is None or str(raw).strip() == "":
            return None
        candidate = Path(str(raw))
        if not candidate.is_absolute():
            candidate = self.root / candidate
        return candidate.resolve()

    # -- convenience ------------------------------------------------------
    @property
    def classes(self) -> List[str]:
        return list(self.require("taxonomy.classes"))

    @property
    def display_names(self) -> List[str]:
        """Human-facing class wording used only in reports."""
        configured = self.get("taxonomy.display_names")
        classes = self.classes
        if not configured:
            return classes
        names = [str(x) for x in configured]
        if len(names) != len(classes):
            return classes
        return names

    def display_name(self, index: int) -> str:
        names = self.display_names
        if 0 <= index < len(names):
            return names[index]
        return f"class {index}"

    @property
    def lesion_channels(self) -> List[str]:
        return list(self.require("taxonomy.lesion_channels"))

    @property
    def seed(self) -> int:
        return int(self.require("project.seed"))

    @property
    def manifest_version(self) -> str:
        return str(self.require("project.manifest_version"))

    def outputs_dir(self, *parts: str, create: bool = True) -> Path:
        base = self.path("paths.outputs_dir", create=create)
        target = base.joinpath(*parts) if parts else base
        if create:
            target.mkdir(parents=True, exist_ok=True)
        return target

    def fundus_root(self) -> Optional[Path]:
        """Root of a pre-existing Mendeley corpus, if configured.

        Precedence: ``DR_FUNDUS_ROOT`` env var, then ``paths.legacy_fundus_root``.
        Never hard-coded to a personal machine path.
        """
        env = os.environ.get(FUNDUS_ROOT_ENV_VAR, "").strip()
        if env:
            return Path(env).expanduser().resolve()
        return self.optional_path("paths.legacy_fundus_root")

    def idrid_root(self) -> Path:
        env = os.environ.get(IDRID_ROOT_ENV_VAR, "").strip()
        if env:
            return Path(env).expanduser().resolve()
        configured = self.optional_path("paths.idrid_manual_root")
        if configured is not None:
            return configured
        return self.path("paths.idrid_dir")

    def effective_batch_size(self, stage: str = "classification", *, fallback: int = 8) -> int:
        value = self.get(f"{stage}.batch_size", "auto")
        if isinstance(value, str) and value.strip().lower() == "auto":
            return fallback
        return int(value)

    def effective_num_workers(self, stage: str = "classification") -> int:
        value = self.get(f"{stage}.num_workers", "auto")
        if isinstance(value, str) and value.strip().lower() == "auto":
            return 0 if os.name == "nt" else min(4, os.cpu_count() or 1)
        return max(0, int(value))

    def mixed_precision_setting(self, stage: str) -> str:
        value = str(self.get(f"{stage}.mixed_precision", "auto")).strip().lower()
        return value


def load_config(
    path: Optional[Path | str] = None,
    *,
    root: Optional[Path | str] = None,
    validate: bool = True,
) -> Config:
    """Load and (by default) validate the YAML configuration."""
    base = project_root(root)
    source = Path(path).expanduser().resolve() if path else default_config_path(base)
    if not source.is_file():
        raise ConfigError(
            f"Configuration file not found: {source}\n"
            f"Expected the repository's config/config.yaml. "
            f"Set {CONFIG_ENV_VAR} to override."
        )
    with source.open("r", encoding="utf-8") as handle:
        try:
            raw = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML in {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Configuration root must be a mapping, got {type(raw).__name__}.")
    cfg = Config(_walk_expand(raw), root=base, source=source)
    if validate:
        validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    """Fail fast on structural problems before any expensive work begins."""
    missing = [key for key in REQUIRED_TOP_LEVEL if key not in cfg.as_dict()]
    if missing:
        raise ConfigError(f"Configuration is missing top-level sections: {missing}")

    classes = cfg.classes
    if len(classes) != 5:
        raise ConfigError(
            f"The five-class taxonomy must contain exactly 5 classes, found {len(classes)}: {classes}"
        )
    if len(set(classes)) != len(classes):
        raise ConfigError(f"Duplicate class names in taxonomy: {classes}")

    channels = cfg.lesion_channels
    if [c.upper() for c in channels] != ["MA", "HE", "EX", "SE"]:
        raise ConfigError(
            "Lesion channel order must be MA, HE, EX, SE (the segmentation output order). "
            f"Found {channels}."
        )

    num_classes = int(cfg.require("classification.num_classes"))
    if num_classes != len(classes):
        raise ConfigError(
            f"classification.num_classes={num_classes} does not match the "
            f"{len(classes)} configured classes."
        )

    out_channels = int(cfg.require("segmentation.out_channels"))
    if out_channels != len(channels):
        raise ConfigError(
            f"segmentation.out_channels={out_channels} does not match the "
            f"{len(channels)} configured lesion channels."
        )

    ratios = cfg.section("split").get("ratios", {})
    for key in ("train", "validation", "test"):
        if key not in ratios:
            raise ConfigError(f"split.ratios is missing {key!r}.")
    total = float(sum(float(ratios[k]) for k in ("train", "validation", "test")))
    if abs(total - 1.0) > 1e-6:
        raise ConfigError(f"split.ratios must sum to 1.0, got {total}.")
    if float(ratios["train"]) <= 0 or float(ratios["validation"]) <= 0 or float(ratios["test"]) <= 0:
        raise ConfigError("Every split ratio must be > 0.")

    image_size = int(cfg.require("preprocessing.classification.image_size"))
    if image_size < 32:
        raise ConfigError(f"preprocessing.classification.image_size={image_size} is too small.")

    # Every mapped label must be a valid class index.
    for source_name in ("mendeley", "mmrdr"):
        rules = cfg.get(f"label_mapping.{source_name}.rules", {}) or {}
        if not rules:
            raise ConfigError(f"label_mapping.{source_name}.rules must not be empty.")
        for original, spec in rules.items():
            if not isinstance(spec, dict) or "mapped_label" not in spec:
                raise ConfigError(
                    f"label_mapping.{source_name}.rules[{original!r}] must provide 'mapped_label'."
                )
            mapped = int(spec["mapped_label"])
            if not 0 <= mapped < len(classes):
                raise ConfigError(
                    f"label_mapping.{source_name}.rules[{original!r}] maps to {mapped}, "
                    f"outside 0..{len(classes) - 1}."
                )

    excluded = cfg.get("label_mapping.mendeley.exclude_non_dr", {}) or {}
    overlap = set(excluded) & set(cfg.get("label_mapping.mendeley.rules", {}) or {})
    if overlap:
        raise ConfigError(
            f"These Mendeley folders are both mapped and excluded: {sorted(overlap)}. "
            "A label cannot be both a DR grade and a non-DR exclusion."
        )

    ece_bins = int(cfg.get("evaluation.calibration.bins", 15))
    if ece_bins < 2:
        raise ConfigError("evaluation.calibration.bins must be >= 2.")

    # Unknown stages in run.stages would silently never execute.
    known = set(pipeline_stage_names())
    unknown = [s for s in (cfg.get("run.stages") or []) if s not in known]
    if unknown:
        raise ConfigError(f"run.stages contains unknown stages: {unknown}. Known: {sorted(known)}")


def pipeline_stage_names() -> Sequence[str]:
    """Canonical ordered stage names used by the orchestrator."""
    return (
        "download",
        "prepare",
        "quality",
        "train_classification",
        "train_segmentation",
        "explainability",
        "evaluate_classification",
        "evaluate_segmentation",
        "calibration",
        "cross_dataset",
        "report",
    )
