"""Final report generation.

Produces three artefacts from whatever the pipeline actually wrote:

* ``outputs/reports/final_report.html``  — human-readable, self-contained report
* ``outputs/reports/execution_summary.json`` — machine-readable run outcome
* ``outputs/evaluation/metrics_summary.csv`` — one flat table of headline metrics

Every value is read from a file produced by a stage. Any metric whose source file
is absent is rendered as "not available" with the reason, never as zero and never
as a plausible placeholder.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import __version__
from .clinical_status import status_record
from .config import Config
from .logutil import utc_now


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return None


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(headers: List[str], rows: List[List[Any]]) -> str:
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(
            f"<td>{cell}</td>" if isinstance(cell, str) and cell.startswith("<")
            else f"<td>{html.escape(_fmt(cell))}</td>"
            for cell in row
        ) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _section(title: str, inner: str, note: str = "") -> str:
    note_html = f'<p class="note">{html.escape(note)}</p>' if note else ""
    return f'<section><h2>{html.escape(title)}</h2>{note_html}{inner}</section>'


def _status_badge(status: str) -> str:
    css = {
        "completed": "ok",
        "reused": "ok",
        "ok": "ok",
        "ok_local_copy": "ok",
        "ok_already_present": "ok",
        "blocked": "warn",
        "skipped": "muted",
        "not_run": "muted",
        "proposed": "muted",
        "failed": "bad",
    }.get(str(status).lower(), "muted")
    return f'<span class="badge {css}">{html.escape(str(status))}</span>'


def collect_metrics(cfg: Config) -> Dict[str, Any]:
    """Gather every available metric file, tagging missing ones with a reason."""
    outputs = cfg.path("paths.outputs_dir")
    collected: Dict[str, Any] = {}

    classification = _read_json(outputs / "classification" / "classification_metrics.json")
    collected["classification"] = classification
    calibration = _read_json(outputs / "evaluation" / "calibration_results.json")
    collected["calibration"] = calibration
    segmentation = _read_json(outputs / "segmentation" / "segmentation_metrics.json")
    collected["segmentation"] = segmentation
    explanation = _read_json(outputs / "explainability" / "explanation_validation.json")
    collected["explanation"] = explanation
    cross = _read_json(outputs / "evaluation" / "cross_dataset_results.json")
    collected["cross_dataset"] = cross
    summary = _read_json(outputs / "manifests" / "dataset_summary.json")
    collected["dataset_summary"] = summary
    quality = _read_json(outputs / "evaluation" / "quality_summary.json")
    collected["quality"] = quality
    training = _read_json(outputs / "classification" / "training_summary.json")
    collected["classification_training"] = training
    seg_training = _read_json(outputs / "segmentation" / "segmentation_summary.json")
    collected["segmentation_training"] = seg_training
    return collected


def write_metrics_csv(path: Path, metrics: Dict[str, Any]) -> Path:
    """Flatten headline metrics into one CSV."""
    rows: List[Dict[str, Any]] = []

    def add(stage: str, metric: str, value: Any, note: str = "") -> None:
        rows.append({"stage": stage, "metric": metric, "value": value, "note": note})

    classification = metrics.get("classification")
    if classification:
        for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1",
                    "macro_specificity", "weighted_f1", "cohen_kappa", "num_samples"):
            add("classification_test", key, classification.get(key))
        auc = (classification.get("roc_auc") or {}).get("macro")
        add("classification_test", "roc_auc_macro", auc,
            "" if auc is not None else "undefined for this test set")
        for name, stats in (classification.get("per_class") or {}).items():
            add("classification_test", f"f1[{name}]", stats.get("f1"))
            add("classification_test", f"recall[{name}]", stats.get("recall"))
            add("classification_test", f"specificity[{name}]", stats.get("specificity"))
    else:
        add("classification_test", "status", "not available", "no classification_metrics.json")

    calibration = metrics.get("calibration")
    if calibration:
        add("calibration", "ece", (calibration.get("calibration") or {}).get("ece"),
            f"{(calibration.get('calibration') or {}).get('num_bins')} equal-width bins")
        flags = (calibration.get("uncertainty") or {}).get("review_flags") or {}
        add("calibration", "fraction_flagged_for_review", flags.get("fraction_flagged"))
    else:
        add("calibration", "status", "not available", "no calibration_results.json")

    segmentation = metrics.get("segmentation")
    if segmentation and segmentation.get("available", True) is not False:
        add("segmentation_test", "macro_dice", segmentation.get("macro_dice"))
        add("segmentation_test", "macro_iou", segmentation.get("macro_iou"))
        add("segmentation_test", "num_images", segmentation.get("num_images"))
        for name, stats in (segmentation.get("per_channel") or {}).items():
            add("segmentation_test", f"dice[{name}]", stats.get("dice"))
            add("segmentation_test", f"iou[{name}]", stats.get("iou"))
    else:
        reason = (segmentation or {}).get("reason", "no segmentation_metrics.json")
        add("segmentation_test", "status", "not available", reason)

    explanation = metrics.get("explanation")
    if explanation:
        overlap = explanation.get("lesion_overlap") or {}
        if overlap.get("available"):
            aggregate = overlap.get("aggregate") or {}
            add("explanation_validation", "mean_dice_vs_lesion_union",
                aggregate.get("mean_dice_vs_lesion_union"))
            add("explanation_validation", "mean_iou_vs_lesion_union",
                aggregate.get("mean_iou_vs_lesion_union"))
        else:
            add("explanation_validation", "status", "not available",
                overlap.get("reason", "IDRiD annotations unavailable"))
    else:
        add("explanation_validation", "status", "not available", "no explanation_validation.json")

    cross = metrics.get("cross_dataset")
    if cross:
        within = cross.get("within_internal_test") or {}
        for source, entry in within.items():
            if isinstance(entry, dict) and entry.get("available"):
                add(f"cross_dataset[{source}]", "accuracy", entry.get("accuracy"))
                add(f"cross_dataset[{source}]", "macro_f1", entry.get("macro_f1"))
        external = (cross.get("external_partition") or {}).get("metrics") or {}
        if external.get("available"):
            add("cross_dataset[mmrdr_official_test]", "accuracy", external.get("accuracy"))
            add("cross_dataset[mmrdr_official_test]", "macro_f1", external.get("macro_f1"))

    dataset = metrics.get("dataset_summary")
    if dataset:
        add("dataset", "mode", dataset.get("mode"))
        counts = dataset.get("counts") or {}
        add("dataset", "images_included", counts.get("images_included"))
        add("dataset", "images_excluded", counts.get("images_excluded"))
        for name, value in (counts.get("included_by_class") or {}).items():
            add("dataset", f"included[{name}]", value)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(rows)
    return target


def _render_html(cfg: Config, metrics: Dict[str, Any], execution: Dict[str, Any]) -> str:
    class_names = cfg.display_names
    parts: List[str] = []
    parts.append(
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Diabetic Retinopathy Pipeline Report</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; padding: 32px;
        color: #14202b; background: #f7f9fb; line-height: 1.5; }}
 .wrap {{ max-width: 1080px; margin: 0 auto; }}
 h1 {{ font-size: 24px; margin: 0 0 4px; }}
 h2 {{ font-size: 17px; margin: 28px 0 10px; border-bottom: 2px solid #dbe4ec; padding-bottom: 6px; }}
 section {{ background: #fff; border: 1px solid #e2e9f0; border-radius: 8px; padding: 16px 20px; margin: 16px 0; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 8px 0; }}
 th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #eef2f6; }}
 th {{ background: #f2f6fa; font-weight: 600; }}
 .badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }}
 .ok {{ background: #dcf5e4; color: #14663a; }}
 .warn {{ background: #fdf0d5; color: #7a5200; }}
 .bad {{ background: #fbdcda; color: #8a1c1c; }}
 .muted {{ background: #e9edf2; color: #4a5766; }}
 .note {{ color: #5a6672; font-size: 12px; margin: 4px 0 8px; }}
 .disclaimer {{ background: #fff6e5; border: 1px solid #f0d9a8; border-radius: 8px; padding: 14px 18px; }}
 code {{ background: #f2f6fa; padding: 1px 5px; border-radius: 4px; font-size: 12px; }}
 ul {{ margin: 6px 0 6px 18px; padding: 0; }}
</style></head><body><div class="wrap">"""
    )
    parts.append("<h1>Diabetic Retinopathy Pipeline — Final Report</h1>")
    parts.append(
        f'<p class="note">Generated {html.escape(utc_now())} · pipeline v{html.escape(__version__)} · '
        f"configuration <code>{html.escape(str(cfg.source))}</code> · "
        f"manifest version <code>{html.escape(cfg.manifest_version)}</code></p>"
    )

    parts.append(
        '<div class="disclaimer"><strong>Research software, not a medical device.</strong> '
        "Every number below comes from a pipeline run on the data actually present on this "
        "machine. The five-class label mapping is provisional (see the label-mapping audit), "
        "the classifier is research code, and no clinical validation has been performed. "
        "Uncertain predictions are flagged for human review and must not be read as "
        "confirmed diagnoses.</div>"
    )

    # Run outcome
    rows = [
        [stage, _status_badge(record.get("status", "unknown")),
         _fmt(record.get("seconds")), record.get("error") or ""]
        for stage, record in (execution.get("stages") or {}).items()
    ]
    has_blocked_or_skipped = any(
        record.get("status") in {"blocked", "skipped", "failed"}
        for record in (execution.get("stages") or {}).values()
    )
    outcome_note = (
        "A stage marked blocked means its input data do not exist; dependent stages were skipped rather than run on missing data."
        if has_blocked_or_skipped
        else "All pipeline stages completed successfully end-to-end without skips or blocks."
    )
    parts.append(
        _section(
            "Execution outcome",
            _table(["Stage", "Status", "Seconds", "Error"], rows) if rows else "<p>No stages recorded.</p>",
            outcome_note,
        )
    )

    # Datasets
    dataset = metrics.get("dataset_summary") or {}
    if dataset:
        counts = dataset.get("counts") or {}
        info_rows = [
            ["Mode", "combined (Mendeley + MMRDR)" if dataset.get("mode") == "combined"
             else f"partial — {dataset.get('mode')}"],
            ["Images discovered", counts.get("images_discovered")],
            ["Images included", counts.get("images_included")],
            ["Images excluded", counts.get("images_excluded")],
        ]
        for source, payload in (dataset.get("sources") or {}).items():
            info_rows.append(
                [f"Source: {source}",
                 payload.get("provenance") or payload.get("reason") or payload.get("available")]
            )
        parts.append(
            _section(
                "Datasets",
                _table(["Item", "Value"], info_rows)
                + _table(
                    ["Included class", "Images"],
                    [[name, value] for name, value in (counts.get("included_by_class") or {}).items()],
                ),
                dataset.get("combined_note", ""),
            )
        )
        splits = dataset.get("splits") or {}
        if splits:
            parts.append(
                _section(
                    "Splits (group-aware, leakage-checked)",
                    _table(
                        ["Partition", "Images", "Per class"],
                        [[name, payload.get("total"), json.dumps(payload.get("by_class"))]
                         for name, payload in splits.items()],
                    ),
                    (dataset.get("grouping") or {}).get("limitation", ""),
                )
            )

    # Classification
    classification = metrics.get("classification")
    if classification:
        header = [
            ["Accuracy", classification.get("accuracy")],
            ["Macro precision", classification.get("macro_precision")],
            ["Macro recall / sensitivity", classification.get("macro_recall")],
            ["Macro specificity", classification.get("macro_specificity")],
            ["Macro F1", classification.get("macro_f1")],
            ["Weighted F1", classification.get("weighted_f1")],
            ["Cohen's kappa", classification.get("cohen_kappa")],
            ["ROC-AUC (macro, one-vs-rest)", (classification.get("roc_auc") or {}).get("macro")],
            ["Test images", classification.get("num_samples")],
        ]
        per_class = [
            [name,
             _fmt(stats.get("precision")), _fmt(stats.get("recall")),
             _fmt(stats.get("specificity")), _fmt(stats.get("f1")),
             stats.get("support")]
            for name, stats in (classification.get("per_class") or {}).items()
        ]
        parts.append(
            _section(
                "Classification — held-out test set",
                _table(["Metric", "Value"], header)
                + _table(["Class", "Precision", "Recall", "Specificity", "F1", "Support"], per_class),
                classification.get("integrity_note", ""),
            )
        )
        matrix = classification.get("confusion_matrix") or []
        if matrix:
            parts.append(
                _section(
                    "Confusion matrix (rows = truth)",
                    _table(["True \\ Predicted"] + class_names,
                           [[class_names[i] if i < len(class_names) else f"class {i}"] +
                            [str(v) for v in row] for i, row in enumerate(matrix)]),
                )
            )

    # Segmentation
    segmentation = metrics.get("segmentation") or {}
    if segmentation.get("available") is not False and segmentation.get("per_channel"):
        parts.append(
            _section(
                "Lesion segmentation — held-out IDRiD pairs",
                _table(["Lesion", "Dice", "IoU"],
                       [[name, _fmt(stats.get("dice")), _fmt(stats.get("iou"))]
                        for name, stats in (segmentation.get("per_channel") or {}).items()]
                       + [["macro", _fmt(segmentation.get("macro_dice")),
                           _fmt(segmentation.get("macro_iou"))]]),
                segmentation.get("note", ""),
            )
        )
    else:
        reason = segmentation.get("reason") or (metrics.get("segmentation_training") or {}).get("reason")
        parts.append(
            _section(
                "Lesion segmentation",
                f'<p>{html.escape(str(reason or "No U-Net checkpoint or no IDRiD annotations."))}</p>'
                '<p class="note">No segmentation metric is reported because none could be '
                "computed from real annotations.</p>",
            )
        )

    # Explainability
    explanation = metrics.get("explanation") or {}
    overlap = explanation.get("lesion_overlap") or {}
    if overlap.get("available"):
        aggregate = overlap.get("aggregate") or {}
        parts.append(
            _section(
                "Explanation validation (Grad-CAM vs lesion annotations)",
                _table(["Measure", "Value"], [
                    ["Mean Dice vs lesion union", _fmt(aggregate.get("mean_dice_vs_lesion_union"))],
                    ["Mean IoU vs lesion union", _fmt(aggregate.get("mean_iou_vs_lesion_union"))],
                    ["Images compared", aggregate.get("num_images")],
                    ["CAM binarisation percentile",
                     (overlap.get("thresholding") or {}).get("binarise_percentile")],
                ]),
                explanation.get("interpretation_note", ""),
            )
        )
    else:
        parts.append(
            _section(
                "Explanation validation",
                f'<p>{html.escape(str(overlap.get("reason", "IDRiD annotations unavailable")))}</p>'
                '<p class="note">Grad-CAM visualisations may still exist; only the quantitative '
                "annotation comparison is unavailable.</p>",
            )
        )

    # Calibration
    calibration = metrics.get("calibration")
    if calibration:
        cal = calibration.get("calibration") or {}
        flags = (calibration.get("uncertainty") or {}).get("review_flags") or {}
        parts.append(
            _section(
                "Calibration and uncertainty (test set)",
                _table(["Metric", "Value"], [
                    ["Expected Calibration Error", _fmt(cal.get("ece"))],
                    ["Bins", cal.get("num_bins")],
                    ["Binning strategy", cal.get("binning_strategy")],
                    ["Predictions flagged for review", flags.get("either")],
                    ["Fraction flagged", _fmt(flags.get("fraction_flagged"))],
                    ["Mean confidence",
                     _fmt(((calibration.get("uncertainty") or {}).get("confidence") or {}).get("mean"))],
                    ["Mean entropy (nats)",
                     _fmt(((calibration.get("uncertainty") or {}).get("entropy") or {}).get("mean"))],
                ]),
                cal.get("binning_description", ""),
            )
        )

    # Cross dataset
    cross = metrics.get("cross_dataset")
    if cross:
        within = cross.get("within_internal_test") or {}
        rows = [
            [source, _fmt(entry.get("accuracy")), _fmt(entry.get("macro_f1")), entry.get("num_samples")]
            for source, entry in within.items()
            if isinstance(entry, dict) and entry.get("available")
        ]
        external = (cross.get("external_partition") or {}).get("metrics") or {}
        if external.get("available"):
            rows.append(["MMRDR official test (external)", _fmt(external.get("accuracy")),
                         _fmt(external.get("macro_f1")), external.get("num_samples")])
        shift = cross.get("domain_shift") or {}
        parts.append(
            _section(
                "Cross-dataset evaluation",
                _table(["Partition", "Accuracy", "Macro F1", "Images"], rows)
                if rows else "<p>No comparable partitions were available.</p>",
                shift.get("interpretation") or shift.get("reason", ""),
            )
        )

    # Quality
    quality = metrics.get("quality")
    if quality:
        parts.append(
            _section(
                "Image quality assessment",
                _table(["Metric", "Value"], [
                    ["Images assessed", quality.get("images_assessed")],
                    ["Unreadable", quality.get("unreadable")],
                    ["Flagged for review", quality.get("flagged_for_review")],
                    ["Flags", json.dumps(quality.get("flag_counts") or {})],
                ]),
                quality.get("advisory_note", ""),
            )
        )

    # Clinical status
    clinical = status_record()
    parts.append(
        _section(
            f"Clinical / fuzzy-logic fusion — {clinical['status'].upper()}",
            f"<p><strong>Not implemented.</strong> {html.escape(clinical['reason'])}</p>"
            "<p>What exists today:</p><ul>"
            + "".join(f"<li>{html.escape(item)}</li>" for item in clinical["what_exists"])
            + "</ul><p>What is missing:</p><ul>"
            + "".join(f"<li>{html.escape(item)}</li>" for item in clinical["what_is_missing"])
            + "</ul>",
            "No synthetic clinical records were created and no multimodal fusion result is "
            "claimed.",
        )
    )

    # Limitations
    limitations = execution.get("limitations") or []
    if limitations:
        parts.append(
            _section(
                "Limitations recorded during this run",
                "<ul>" + "".join(f"<li>{html.escape(str(item))}</li>" for item in limitations) + "</ul>",
            )
        )

    parts.append(
        '<p class="note">Report generated by <code>src/report.py</code>. '
        "Source artefacts are under <code>outputs/</code>.</p></div></body></html>"
    )
    return "".join(parts)


def build_report(
    cfg: Config,
    logger,
    execution: Dict[str, Any],
    events=None,
) -> Dict[str, Any]:
    """Write the final HTML report, execution summary and metrics CSV."""
    metrics = collect_metrics(cfg)
    reports_dir = cfg.outputs_dir("reports")
    evaluation_dir = cfg.outputs_dir("evaluation")

    metrics_csv = write_metrics_csv(evaluation_dir / "metrics_summary.csv", metrics)

    execution_summary = dict(execution)
    execution_summary["generated"] = utc_now()
    execution_summary["pipeline_version"] = __version__
    execution_summary["manifest_version"] = cfg.manifest_version
    execution_summary["config_file"] = str(cfg.source)
    execution_summary["metrics_summary_csv"] = str(metrics_csv)
    execution_summary["clinical_fusion_status"] = status_record()

    available = {
        "classification_metrics": metrics.get("classification") is not None,
        "calibration_results": metrics.get("calibration") is not None,
        "segmentation_metrics": bool((metrics.get("segmentation") or {}).get("per_channel")),
        "explanation_validation": metrics.get("explanation") is not None,
        "cross_dataset_results": metrics.get("cross_dataset") is not None,
        "dataset_summary": metrics.get("dataset_summary") is not None,
        "quality_summary": metrics.get("quality") is not None,
    }
    execution_summary["artefacts_present"] = available

    summary_path = reports_dir / "execution_summary.json"
    summary_path.write_text(json.dumps(execution_summary, indent=2, default=str), encoding="utf-8")

    html_text = _render_html(cfg, metrics, execution_summary)
    html_path = reports_dir / "final_report.html"
    html_path.write_text(html_text, encoding="utf-8")

    logger.info("report written to %s", html_path)
    if events is not None:
        events.write("report_done", html=str(html_path), summary=str(summary_path))
    return {
        "status": "completed",
        "html_report": str(html_path),
        "execution_summary": str(summary_path),
        "metrics_summary_csv": str(metrics_csv),
        "artefacts_present": available,
    }
