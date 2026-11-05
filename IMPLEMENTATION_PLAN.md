"""Scope guard for the DR pipeline improvement task.

Works on the existing repo at commit 1baa467 (branch evaluation-fix),
which already has: unified Mendeley+MMRDR five-class manifest, ResNet-50
training, four-channel IDRiD U-Net, Grad-CAM, calibration/ECE, cross-dataset
eval, reports, RUN.bat, README, audit, tests.

This task adds, on top of that existing basename:
- APTOS + DDR dataset adapters + Kaggle/README-based downloaders (download stage
  extended, never rewrites existing FTP/figshare/Mendeley downloaders).
- Two execution modes: combined (train on Mendeley+MMRDR+APTOS+DDR) and external
  (train only on Mendeley+MMRDR, evaluate on APTOS + DDR).
- A configurable experiment engine reusing ONE ResNet-50, ONE data pipeline,
  ONE evaluation path, with E0..E6 experiment variants and a final candidate.
- Experiment output layout under outputs/experiments/ plus checkpoints/E* dirs.
- Tests for the new adapters and experiment components.
- Documentation updates: README, IMPLEMENTATION_AUDIT, RUN.bat usage.
"""
