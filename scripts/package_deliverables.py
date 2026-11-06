"""Package and verify final deliverables for submission."""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
DELIVERABLES = ROOT / "deliverables_for_submission"
DELIVERABLES.mkdir(parents=True, exist_ok=True)


def package() -> None:
    print("Packaging deliverables...")

    # 1. Copy key reports directly
    for filename in ["final_report.html", "execution_summary.json"]:
        src = OUTPUTS / "reports" / filename
        if src.is_file():
            shutil.copyfile(src, DELIVERABLES / filename)
            print(f"Copied {filename} to {DELIVERABLES}")

    metrics_csv = OUTPUTS / "evaluation" / "metrics_summary.csv"
    if metrics_csv.is_file():
        shutil.copyfile(metrics_csv, DELIVERABLES / "metrics_summary.csv")
        print(f"Copied metrics_summary.csv to {DELIVERABLES}")

    # 2. Package model checkpoints
    ckpt_zip = DELIVERABLES / "model_checkpoints.zip"
    with zipfile.ZipFile(ckpt_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        resnet_pt = OUTPUTS / "classification" / "best_resnet50.pt"
        if resnet_pt.is_file():
            zf.write(resnet_pt, "best_resnet50.pt")
            print(f"Added best_resnet50.pt ({resnet_pt.stat().st_size / 1e6:.1f} MB)")
        unet_pt = OUTPUTS / "segmentation" / "best_lesion_unet.pt"
        if unet_pt.is_file():
            zf.write(unet_pt, "best_lesion_unet.pt")
            print(f"Added best_lesion_unet.pt ({unet_pt.stat().st_size / 1e6:.1f} MB)")
    print(f"Created {ckpt_zip} ({ckpt_zip.stat().st_size / 1e6:.1f} MB)")

    # 3. Package evaluation reports, figures, and metrics
    eval_zip = DELIVERABLES / "evaluation_reports_and_metrics.zip"
    with zipfile.ZipFile(eval_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for sub in ["reports", "evaluation", "explainability", "manifests"]:
            dir_path = OUTPUTS / sub
            if dir_path.is_dir():
                for f in dir_path.rglob("*"):
                    if f.is_file():
                        rel = f.relative_to(OUTPUTS)
                        zf.write(f, str(rel))
    print(f"Created {eval_zip} ({eval_zip.stat().st_size / 1e6:.1f} MB)")

    print("\nAll deliverables packaged successfully in deliverables_for_submission/")


if __name__ == "__main__":
    package()

