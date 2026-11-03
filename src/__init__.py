"""Unified, auditable diabetic-retinopathy research pipeline.

This package is the one-click executable layer requested for the project. It
reuses the already-tested ``dataset`` package for decoding/label policy and the
``colab.gradcam`` primitives for explainability, and adds the parts the
repository did not have: dataset acquisition, the combined Mendeley+MMRDR
five-class manifest, image-quality assessment, the four-channel lesion U-Net,
calibration/uncertainty, cross-dataset evaluation, reporting and orchestration.

Research-integrity stance
-------------------------
Nothing in this package fabricates data or metrics. When an input is missing,
the responsible stage is recorded as ``blocked`` with the measured reason and
dependent stages are skipped. The five-class label mapping is provisional and
is always described as such. The clinical/fuzzy fusion component is PROPOSED
and is not implemented (see :mod:`src.clinical_status`).
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
