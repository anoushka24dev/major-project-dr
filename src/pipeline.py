"""Central pipeline orchestrator.

One entry point for every stage, with explicit dependency gating:

* a stage whose input is missing is recorded as ``blocked`` with the measured
  reason, and its dependants are recorded as ``skipped``;
* a stage that raises is recorded as ``failed``, stops dependent stages, and makes
  the process exit non-zero — a failed training run can never be reported as
  successful;
* the final report is always written, including when earlier stages failed, so the
  reason is visible to the person who double-clicked the launcher.

Stage modes
-----------
``full``      download → prepare → quality → train → evaluate → report
``download``  dataset acquisition only
``prepare``   manifests and quality assessment only
``train``     the two training stages only
``evaluate``  evaluation, explainability, calibration, cross-dataset, report
``report``    rebuild the report from existing artefacts

Dataset-composition modes
-------------------------
``combined``  run the whole pipeline with one classifier trained on every validated
              classification source (Mendeley + MMRDR + APTOS + DDR).
``external``  run the whole pipeline training only on ``run.development_sources``
              (initially Mendeley + MMRDR) and evaluating on the held-out sources.

Both are recorded in the execution summary, and neither is inferred from whatever
happens to be on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .clinical_status import status_record
from .config import Config, ConfigError, load_config
from .logutil import EventLog, get_logger, stage as stage_context

FULL_STAGES: List[str] = [
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
]

MODES: Dict[str, List[str]] = {
    "full": FULL_STAGES,
    # Dataset-composition modes run the full stage list; they differ in which
    # sources are allowed to train the classifier (see src.prepare).
    "combined": FULL_STAGES,
    "external": FULL_STAGES,
    "download": ["download"],
    "prepare": ["prepare", "quality"],
    "train": ["train_classification", "train_segmentation"],
    "evaluate": [
        "explainability",
        "evaluate_classification",
        "evaluate_segmentation",
        "calibration",
        "cross_dataset",
        "report",
    ],
    "report": ["report"],
}

#: Stages that cannot run when an earlier stage did not produce its input.
DEPENDENCIES: Dict[str, Sequence[str]] = {
    "prepare": ("download",),
    "quality": ("prepare",),
    "train_classification": ("prepare",),
    "train_segmentation": (),
    "explainability": ("train_classification",),
    "evaluate_classification": ("train_classification",),
    "calibration": ("evaluate_classification",),
    "cross_dataset": ("evaluate_classification",),
    "report": (),
}

STAGE_LABELS = {
    "download": "Dataset acquisition",
    "prepare": "Unified manifest preparation",
    "quality": "Image quality assessment",
    "train_classification": "ResNet-50 training",
    "train_segmentation": "Lesion U-Net training",
    "explainability": "Grad-CAM explainability",
    "evaluate_classification": "Classification evaluation",
    "evaluate_segmentation": "Segmentation evaluation",
    "calibration": "Calibration and uncertainty",
    "cross_dataset": "Cross-dataset evaluation",
    "report": "Final report",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description="Diabetic-retinopathy research pipeline orchestrator.",
    )
    parser.add_argument("--mode", default=None, choices=sorted(MODES),
                        help="which part of the pipeline to run (default: full)")
    # Mode aliases, so the launcher can forward human-friendly flags unchanged.
    aliases = parser.add_argument_group("mode shortcuts")
    aliases.add_argument("--download-only", dest="mode_alias", action="store_const",
                         const="download", help="download and verify datasets only")
    aliases.add_argument("--prepare-only", dest="mode_alias", action="store_const",
                         const="prepare", help="manifests and quality only")
    aliases.add_argument("--train-only", dest="mode_alias", action="store_const",
                         const="train", help="training only")
    aliases.add_argument("--evaluate-only", dest="mode_alias", action="store_const",
                         const="evaluate", help="evaluation and report only")
    aliases.add_argument("--report-only", dest="mode_alias", action="store_const",
                         const="report", help="rebuild the report only")
    parser.add_argument("--no-setup", action="store_true",
                        help="accepted and ignored: the launcher handles environment setup")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--project-root", default=None, help="repository root")
    parser.add_argument("--fundus-root", default=None,
                        help="root of a pre-existing Mendeley corpus (13 class folders)")
    parser.add_argument("--idrid-root", default=None, help="root of a pre-existing IDRiD copy")
    parser.add_argument("--aptos-root", default=None,
                        help="root of a pre-existing APTOS 2019 copy")
    parser.add_argument("--ddr-root", default=None,
                        help="root of a pre-existing DDR copy")
    parser.add_argument("--dataset-mode", dest="dataset_mode", default=None,
                        choices=["combined", "external"],
                        help="which sources may train the classifier; combinable with any "
                             "stage mode, e.g. --prepare-only --dataset-mode external")
    parser.add_argument("--experiments", nargs="+", default=None, metavar="EXP",
                        help="run the ResNet-50 experiment series instead of one baseline "
                             "run: 'all', or a subset such as 'E0 E4 final_candidate'")
    parser.add_argument("--cv-folds", dest="cv_folds", type=int, default=None, metavar="N",
                        help="run grouped stratified N-fold cross-validation (default 5, from "
                             "run.cv.n_splits), then retrain on the full development pool")
    parser.add_argument("--resume", action="store_true",
                        help="continue interrupted training from the last checkpoint")
    parser.add_argument("--force-retrain", action="store_true",
                        help="ignore existing checkpoints and train from scratch")
    parser.add_argument("--skip-download", action="store_true",
                        help="inspect datasets without transferring bytes")
    parser.add_argument("--datasets", nargs="+", default=None, metavar="NAME",
                        help="acquire only these sources, e.g. --datasets mmrdr ddr")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    parser.add_argument("--validate-config", action="store_true",
                        help="validate config/config.yaml and exit")
    parser.add_argument("--limit-quality", type=int, default=None,
                        help="assess only the first N images in the quality stage")
    return parser


def _limitations(cfg: Config, summary: Dict[str, Any], stage_records: Dict[str, Dict[str, Any]]) -> List[str]:
    """Collect honest limitations from what actually happened."""
    items: List[str] = []

    if summary.get("dataset_mode") == "external":
        items.append(
            "External-validation mode: the classifier trains only on "
            + ", ".join(cfg.development_sources)
            + ". Every other source is evaluated as a held-out external test set and "
            "does not influence training, thresholds, calibration or model selection."
        )

    dataset = summary.get("dataset_summary") or {}
    if dataset.get("mode") == "partial":
        items.append(
            "Classification data is PARTIAL: "
            + str(dataset.get("combined_note", "not all sources were available"))
        )
    sources = (dataset.get("sources") or {})
    for name, payload in sources.items():
        if not payload.get("available"):
            items.append(f"{name} unavailable: {payload.get('reason', 'unknown reason')}")

    grouping = dataset.get("grouping") or {}
    if grouping.get("records_without_patient_id"):
        items.append(
            f"{grouping['records_without_patient_id']} included images have no derived patient "
            "identifier; leakage control for those relies on exact and perceptual duplicate "
            "grouping only."
        )
    if grouping.get("cross_folder_patient_id_collisions"):
        items.append(
            f"{grouping['cross_folder_patient_id_collisions']} patient identifiers appear in "
            "more than one class folder. The project's provisional policy scopes split groups "
            "per folder, so cross-folder patient identity is NOT resolved."
        )

    mapping = dataset.get("label_mapping_policy") or {}
    if mapping.get("provisional"):
        items.append(
            "The five-class label mapping is provisional: 'Very Severe NPDR' is merged into "
            "Severe NPDR and 'Advanced PDR' into PDR, and no specialist has signed it off."
        )

    for stage_name, record in stage_records.items():
        if record.get("status") == "blocked":
            items.append(f"Stage {stage_name} blocked: {record.get('reason') or record.get('detail')}")
        elif record.get("status") == "failed":
            items.append(f"Stage {stage_name} failed: {record.get('error')}")

    items.append(
        "Clinical-information fusion and fuzzy-logic reasoning are PROPOSED and not "
        "implemented; no clinical data exists in this repository and no synthetic clinical "
        "records were created."
    )
    items.append(
        "No clinical validation, regulatory clearance or diagnostic claim is made anywhere "
        "in this pipeline."
    )
    return items


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute the requested mode and return an execution summary."""
    cfg = load_config(args.config, root=args.project_root)

    if args.fundus_root:
        import os

        os.environ["DR_FUNDUS_ROOT"] = str(Path(args.fundus_root).resolve())
    if args.idrid_root:
        import os

        os.environ["DR_IDRID_ROOT"] = str(Path(args.idrid_root).resolve())
    if args.aptos_root:
        import os

        os.environ["DR_APTOS_ROOT"] = str(Path(args.aptos_root).resolve())
    if args.ddr_root:
        import os

        os.environ["DR_DDR_ROOT"] = str(Path(args.ddr_root).resolve())

    # Re-load so environment overrides take effect.
    cfg = load_config(args.config, root=args.project_root)

    # A dataset-composition mode on the command line overrides the configured one,
    # and is recorded in the run so downstream stages cannot disagree about it.
    if getattr(args, "dataset_mode", None):
        dataset_mode = args.dataset_mode
    elif args.mode in {"combined", "external"}:
        dataset_mode = args.mode
    else:
        dataset_mode = cfg.dataset_mode
    cfg._data.setdefault("run", {})["dataset_mode"] = dataset_mode

    logs_dir = cfg.path("paths.logs_dir", create=True)
    stamp = __import__("time").strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"pipeline_{stamp}_{args.mode}.log"
    logger = get_logger(log_path)
    events = EventLog(logs_dir / f"events_{stamp}_{args.mode}.jsonl", logger=logger)

    logger.info("=" * 70)
    logger.info("Diabetic-retinopathy pipeline v%s | mode=%s", __version__, args.mode)
    logger.info("dataset composition mode: %s", dataset_mode)
    logger.info("config: %s", cfg.source)
    logger.info("log:    %s", log_path)
    logger.info("=" * 70)
    events.write(
        "run_start", mode=args.mode, dataset_mode=dataset_mode,
        version=__version__, config=str(cfg.source),
    )

    stages = list(MODES[args.mode])
    stage_records: Dict[str, Dict[str, Any]] = {}
    outputs: Dict[str, Any] = {}
    failed = False

    for stage_name in stages:
        # Dependency gate. Only dependencies that are part of THIS mode are checked;
        # a dependency outside the mode is assumed to have been produced by an
        # earlier run (and if it was not, the stage itself fails with a clear message).
        blocked_by = [
            dependency for dependency in DEPENDENCIES.get(stage_name, ())
            if dependency in stages
            and (
                dependency not in outputs
                or stage_records.get(dependency, {}).get("status") in {"failed", "skipped"}
                or (outputs.get(dependency) or {}).get("status") == "blocked"
            )
        ]
        if blocked_by and stage_name != "report":
            record = {
                "status": "skipped",
                "reason": f"dependency not satisfied: {', '.join(blocked_by)}",
            }
            stage_records[stage_name] = record
            logger.warning("SKIP %s (%s)", stage_name, record["reason"])
            events.write("stage_skipped", stage=stage_name, reason=record["reason"])
            continue

        label = STAGE_LABELS.get(stage_name, stage_name)
        logger.info("")
        logger.info("--- %s ---", label)
        try:
            with stage_context(stage_name, logger, events) as timer:
                outputs[stage_name] = _run_stage(
                    stage_name, cfg, logger, events, args, dataset_mode
                )
                if isinstance(outputs[stage_name], dict):
                    stage_records[stage_name] = {
                        "status": outputs[stage_name].get("status", "completed"),
                        **{k: v for k, v in outputs[stage_name].items() if k != "status"},
                    }
                else:
                    stage_records[stage_name] = {"status": "completed"}
        except Exception as exc:
            logger.error("Stage %s failed: %s: %s", stage_name, type(exc).__name__, exc)
            logger.debug("%s", traceback.format_exc())
            stage_records[stage_name] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            events.write("stage_failed", stage=stage_name, error=str(exc))
            failed = True
            if bool(cfg.get("run.stop_on_failure", True)):
                continue

    logger.info("")
    logger.info("=" * 70)
    for stage_name in stages:
        record = stage_records.get(stage_name, {"status": "not_run"})
        logger.info("%-26s %s", stage_name, record.get("status"))

    data_blocked = any(
        record.get("status") in {"blocked", "failed"} for record in stage_records.values()
    )
    exit_code = 1 if failed else (3 if data_blocked else 0)

    summary: Dict[str, Any] = {
        "pipeline_version": __version__,
        "mode": args.mode,
        "dataset_mode": dataset_mode,
        "experiments_requested": list(args.experiments or []),
        "config_file": str(cfg.source),
        "log_file": str(log_path),
        "stages": stage_records,
        "outputs": {k: v for k, v in outputs.items() if isinstance(v, dict)},
        "exit_code": exit_code,
        "resume": bool(args.resume),
        "force_retrain": bool(args.force_retrain),
    }

    report_path = cfg.outputs_dir("reports") / "final_report.html"
    if "report" not in stages and report_path.is_file():
        summary["existing_report"] = str(report_path)

    # Assemble limitations from the recorded facts, then rebuild the report so it
    # includes this run's outcome (even a failed one).
    dataset_summary = json.loads(
        (cfg.outputs_dir("manifests") / "dataset_summary.json").read_text(encoding="utf-8")
    ) if (cfg.outputs_dir("manifests") / "dataset_summary.json").is_file() else None
    summary["dataset_summary"] = dataset_summary
    summary["limitations"] = _limitations(cfg, summary, stage_records)

    if report_path.parent.exists() or args.mode == "full":
        try:
            from .report import build_report

            report_result = build_report(cfg, logger, summary, events)
            summary["report"] = report_result
        except Exception as exc:
            logger.error("Could not write the final report: %s", exc)
            summary["report"] = {"status": "failed", "error": str(exc)}

    events.write("run_end", exit_code=exit_code, stages={
        name: record.get("status") for name, record in stage_records.items()
    })
    logger.info("exit code: %d", exit_code)
    logger.info("report: %s", report_path if report_path.is_file() else "not written")
    return summary


def _run_stage(
    stage_name: str,
    cfg: Config,
    logger,
    events,
    args,
    dataset_mode: str = "combined",
) -> Dict[str, Any]:
    """Dispatch one stage to its implementation module."""
    if stage_name == "download":
        from .dataset_downloaders.manager import run_download_stage

        result = run_download_stage(
            cfg,
            logger,
            events,
            skip_download=bool(args.skip_download),
            only=getattr(args, "datasets", None),
        )
        result["dataset_mode"] = dataset_mode
        return result

    if stage_name == "prepare":
        from .prepare import run_prepare

        return run_prepare(cfg, logger, events, dataset_mode=dataset_mode)

    if stage_name == "quality":
        from .quality import run_quality_assessment

        return run_quality_assessment(cfg, logger, events, limit=args.limit_quality)

    if stage_name == "train_classification":
        if getattr(args, "cv_folds", None):
            from .kfold import run_cv_protocol

            result = run_cv_protocol(
                cfg,
                logger,
                events,
                n_splits=int(args.cv_folds),
                resume=bool(args.resume),
                force_retrain=bool(args.force_retrain),
            )
            summary = {
                key: value
                for key, value in result.items()
                if key not in {"folds", "plan"}
            }
            return {
                "status": str(result.get("status", "failed")),
                "cross_validation": summary,
                "folds_completed": len(result.get("folds") or []),
                "published_checkpoint": result.get("published_checkpoint"),
                "detail": result.get("reason"),
            }

        requested = list(getattr(args, "experiments", None) or [])
        if requested:
            from .experiments import (
                promote_final_candidate,
                run_selected_experiments,
                write_experiment_comparison,
                write_experiment_csvs,
            )

            result = run_selected_experiments(
                cfg,
                logger,
                events,
                experiments=requested,
                resume=bool(args.resume),
                force_retrain=bool(args.force_retrain),
            )
            comparison_path = write_experiment_comparison(
                cfg, result["comparison"], result["individual"], result["selected"]
            )
            csv_paths = write_experiment_csvs(cfg, result["individual"])
            promoted = promote_final_candidate(
                cfg, result["individual"], result["selected"], logger=logger
            )
            failed = [
                eid for eid, summary in result["individual"].items()
                if summary.get("status") not in {"completed", "reused"}
            ]
            status = "failed" if promoted is None else "completed"
            return {
                "status": status,
                "experiments": result["selected"],
                "best_by_validation_macro_f1": result["best_by_validation_macro_f1"],
                "failed_experiments": failed,
                "comparison_json": str(comparison_path),
                "comparison_csv": str(csv_paths.get("experiment_comparison.csv", "")),
                "promoted_checkpoint": str(promoted) if promoted else None,
                "detail": (
                    None if promoted
                    else "no experiment produced a checkpoint, so no test-time evaluation "
                         "can run"
                ),
            }

        from .train_classifier import run_training

        return run_training(
            cfg, logger, events,
            force_retrain=bool(args.force_retrain),
            resume=bool(args.resume),
        )

    if stage_name == "train_segmentation":
        from .train_segmentation import run_training as run_seg_training

        return run_seg_training(
            cfg, logger, events,
            force_retrain=bool(args.force_retrain),
            resume=bool(args.resume),
        )

    if stage_name == "explainability":
        from .explainability import run_explainability

        return run_explainability(cfg, logger, events)

    if stage_name == "evaluate_classification":
        from .evaluate_classification import run_evaluation

        return run_evaluation(cfg, logger, events)

    if stage_name == "evaluate_segmentation":
        from .evaluate_segmentation import run_evaluation as run_seg_evaluation

        return run_seg_evaluation(cfg, logger, events)

    if stage_name == "calibration":
        from .calibration import run_calibration

        return run_calibration(cfg, logger, events)

    if stage_name == "cross_dataset":
        from .cross_dataset import run_cross_dataset

        return run_cross_dataset(cfg, logger, events)

    if stage_name == "report":
        from .report import build_report

        # The report stage is executed later with the full execution summary; this
        # placeholder keeps the stage list uniform.
        return {"status": "completed"}

    raise ValueError(f"Unknown stage: {stage_name}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    alias = getattr(args, "mode_alias", None)
    if alias:
        if args.mode and args.mode != alias:
            print(
                f"CONFIG ERROR: --mode {args.mode} conflicts with the mode shortcut "
                f"that selects {alias!r}.", file=sys.stderr,
            )
            return 2
        args.mode = alias
    if not args.mode:
        args.mode = "full"

    if args.validate_config:
        try:
            cfg = load_config(args.config, root=args.project_root)
        except ConfigError as exc:
            print(f"CONFIG INVALID: {exc}", file=sys.stderr)
            return 2
        print(f"Configuration OK: {cfg.source}")
        print(f"  classes          : {cfg.classes}")
        print(f"  lesion channels  : {cfg.lesion_channels}")
        print(f"  manifest version : {cfg.manifest_version}")
        print(f"  seed             : {cfg.seed}")
        return 0

    try:
        summary = run_pipeline(args)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    if args.json:
        print(json.dumps(summary, indent=1, default=str))
    else:
        print()
        print("=" * 70)
        print(f"PIPELINE {summary['mode'].upper()} — exit code {summary['exit_code']}")
        print("=" * 70)
        for name, record in summary["stages"].items():
            print(f"  {name:26s} {record.get('status')}"
                  + (f"  ({record.get('error')})" if record.get("error") else ""))
        report = summary.get("report") or {}
        if report.get("html_report"):
            print(f"\nReport: {report['html_report']}")
        limitations = summary.get("limitations") or []
        if limitations:
            print("\nLimitations:")
            for item in limitations:
                print(f"  - {item}")
    return int(summary.get("exit_code", 0))


if __name__ == "__main__":
    raise SystemExit(main())
