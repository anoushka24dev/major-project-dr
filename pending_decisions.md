# Pending decisions for the training milestone

This file records the open choices before implementing the training pipeline. Nothing has been implemented yet.

## Stack decisions
1. **PyTorch vision dependency:** the current environment has `torch 2.14.0` (CPU) but `torchvision` is not importable. The proposed ResNet-50 baseline needs `torchvision`. Confirm whether to add it now, and whether via a pinned requirements file.
2. **timm:** not required for the first baseline. Proposed to skip it for now. Confirm or replace with another preference.
3. **Hardware:** training environment is CPU-only right now. Confirm whether GPU training is available/desired or whether the baseline should stay CPU-bound.

## Design decisions
4. **Augmentations:** proposed conservative preprocessing for the baseline. Horizontal flip and random resized crop on training data may or may not be appropriate for fundus images. Confirm the preferred augmentation policy.
5. **Freeze strategy:** proposed head-only training first as the cheapest reproducible baseline. Confirm or prefer a different starting strategy.
6. **Run management:** proposed a single scriptable baseline run vs a config-driven experiment runner. Confirm the preferred approach.

## Scope guardrails (already agreed)
- Do not change any existing dataset manifests, split assignments, source images, or audit artifacts.
- Do not change the provisional five-band label mapping.
- Do not implement the IDRiD evaluation yet.
- Do not build the agentic workflow yet.
- Do not begin full model training during this planning step.
- Do not install dependencies or overwrite files without first reporting what is needed.
- Preserve all known limitations, including the provisional taxonomy merges and missing verified patient linkage for 503 Zenodo-derived images.
