# Colab Baseline for Diabetic Retinopathy Training

This folder contains a provisional Colab-ready baseline for the primary five-band DR task.

**Status:** implemented and verified locally on CPU (including executed end-to-end mini runs with the model stubbed). **Not** yet executed in a Colab runtime and **not** GPU-verified.

Start with `colab/COLAB_EXECUTION_CHECKLIST.md`: it lists the run order, the evidence to check at each cell, and what to do when a check fails.

No model has been trained. Nothing in this folder claims GPU execution, Colab compatibility, or clinical validity.

## What this package contains

- `colab/requirements-colab.txt` — optional dependency pins for a Colab Python runtime
  (`torch==2.6.0+cu124`, `torchvision==0.21.0+cu124`, resolved through PyTorch's wheel index).
  Only install this when the runtime genuinely lacks torch/torchvision.
- `colab/colab_helpers.py` — path resolution, split loading, deterministic preprocessing, and a frozen ResNet-50 builder.
- `colab/smoke_test.py` — pytest smoke suite gating training: split integrity, dataset-package integration, real image decoding, class weights, metrics, checkpoint and overwrite guards.
- `colab/train.py` — minimal executable frozen ResNet-50 training pipeline with validation-based checkpoint selection, early stopping, output guards, and a separate final test-evaluation entry point.
- `colab/train_on_colab.ipynb` — notebook that imports the existing `dataset/` package, validates the split, runs the smoke test under pytest, and only then exposes guarded training and final-evaluation cells.
- `colab/COLAB_EXECUTION_CHECKLIST.md` — per-cell run order, expected evidence, and troubleshooting.
- `colab/TASK_REPORT.md` — implementation and verification report, including the defects found and
  fixed in the validation pass.
- `colab/README.md` — this file.

## What this baseline does and does not do

**It does:**
- Use the existing split artifact unchanged.
- Validate the split CSV and partition isolation before training.
- Decode a sample of real images from the split as part of smoke testing, through the existing `dataset.decoding.load_rgb` helper rather than a parallel implementation.
- Use aspect-ratio-preserving resize-and-pad preprocessing.
- Keep validation and test preprocessing deterministic.
- Build a frozen ResNet-50 with a new 5-class head.
- Compute class weights from the training labels only.
- Train only on the training partition.
- Evaluate on validation each epoch and select the best checkpoint by validation loss.
- Support early stopping.
- Keep the test partition for a separate final evaluation step.
- Save checkpoints, configuration, training history, and metrics to a persistent output folder.

**It does not:**
- Modify the existing dataset, split CSV, split report, manifest, or any audit artifact.
- Change the provisional five-band label mapping.
- Run a full training run locally here.
- Claim GPU execution unless CUDA is actually available in the runtime.
- Claim clinical validity from a baseline run.

## Before opening the notebook

1. Upload the project into the Colab runtime, or mount Google Drive that already contains it.
   - The notebook expects to find the `dataset/` package and the `provisional_primary_split/` folder.
2. Make the dataset root available to the runtime.
   - Either upload the fundus image folder into the runtime,
   - or mount Google Drive and place the dataset there,
   - or set the environment variable `DR_FUNDUS_ROOT` to the dataset folder.
3. Make the split directory available.
   - Either set `DR_SPLIT_DIR` to the folder that contains `provisional_primary_split/`,
   - or rely on the notebook's fallback paths when the project is uploaded to `/content/major_project`.

## Recommended runtime setup

1. Open `colab/train_on_colab.ipynb` in Google Colab.
2. Runtime → Change runtime type → Hardware accelerator → GPU.
3. Run the cells in order.

`torch` and `torchvision` are required to build the model. This pipeline was developed on a
machine that has `torch` but **no `torchvision`**, so the real `models.resnet50` call has never run
here; it must be validated in Colab, and cell `# 4` is the cheapest place to do it (it builds the
architecture with `weights=None` and downloads nothing). The builder's own logic was executed
locally against a stand-in `torchvision` module, which is weaker evidence than a real import.
Without torchvision the entry points refuse to start and say so explicitly rather than failing
obscurely.

## Cell flow

The notebook labels its own cells `# 0.` through `# 6.`, and those labels are what
`colab/COLAB_EXECUTION_CHECKLIST.md` refers to.

1. **`# 0.` Environment setup + dataset-package integration check** — adds the uploaded project to `sys.path`, prints Python/torch/torchvision state, imports `dataset.labels` and `dataset.decoding` explicitly, and asserts that `colab_helpers.BAND_ORDER` matches `dataset.labels.PRIMARY_MAPPING`.
2. **`# 1.` Optional dependency install** — installs only if torch/torchvision are genuinely missing, and only from `colab/requirements-colab.txt`; it never uninstalls Colab's preinstalled stack.
3. **`# 2.` Path resolution** — resolves the dataset root and split directory (env var, then Drive mount, then project-relative) and defaults the output directory to Drive when Drive is mounted.
4. **`# 3.` Smoke test** — runs `python -m pytest colab/smoke_test.py -q` as a subprocess and reads its exit code. Training cannot proceed unless it exits 0.
5. **`# 4.` Guard contract demo** — prints the guard contract, the *actual* device report, and whether the ImageNet weights are already cached, then builds the ResNet-50 architecture with `weights=None` to prove the frozen-backbone/5-class-head contract. Safe to run every time; trains nothing and downloads nothing.
6. **`# 5.` Training cell** — guarded by `RUN_TRAINING = False`; never touches the test partition.
7. **`# 6.` Final evaluation cell** — guarded by `RUN_FINAL_EVALUATION = False`; the only place the held-out test partition is scored.

## Environment variables you can set

- `DR_FUNDUS_ROOT` — dataset root folder.
- `DR_SPLIT_DIR` — folder containing `provisional_primary_split/`.
- `DR_OUTPUT_DIR` — output folder; defaults to `/content/drive/MyDrive/dr_baseline_outputs` when Drive is mounted.
- `DR_SMOKE_IMAGE_ROOT` — image folder used by the smoke tests and tests when it is not discoverable automatically.

If these are not set, the notebook attempts Drive-mounted and `/content/major_project` fallbacks.

## Smoke test checks

Run with `python -m pytest colab/smoke_test.py -q`. It must exit 0 and it never trains.

- The provisional five-band order is exactly the five expected band names.
- The split CSV loads, has the expected row count, and its partitions are exactly `train`/`validation`/`test`.
- Partitions are pairwise disjoint and together cover every row.
- `dataset.labels` imports and its `PRIMARY_MAPPING` values match `colab_helpers.BAND_ORDER`, including the two provisional merges.
- `colab_helpers.load_rgb` delegates to `dataset.decoding.load_rgb`, and a real image decodes through it (skipped with a reason if no corpus is reachable).
- A sampled dataset item is an ImageNet-normalized `3x64x64` float tensor whose label matches its band.
- Preprocessing keeps the aspect ratio, pads rather than crops, and reports its content box.
- Class weights are exactly inverse-frequency over the *training* rows and refuse a band with no examples.
- Metrics reports accuracy, macro precision/recall/F1, per-class metrics with support, a square integer confusion matrix, and the sample count.
- A checkpoint round-trip writes `model.pt`, `training_state.json`, and a `best_checkpoint/` copy with compatibility metadata.
- Overwrite protection refuses to reuse an output directory that already holds a run.
- Training refuses to start without the explicit opt-in, and refuses before the smoke test has passed.
- Final test evaluation is refused without an explicit request and without a checkpoint.
- The notebook is valid JSON, keeps both guards, and references only files that exist.

## Guard, output, and interruption contract

- Training is **disabled by default**. `run_training` requires `confirm_training=True` **and** `smoke_test_passed=True`.
- Final test evaluation is **disabled by default**. `run_final_evaluation` requires `confirm_final_evaluation=True`.
- `run_training` refuses to start if the output directory already contains a run. Pass `overwrite=True` to replace it deliberately; nothing is overwritten silently.
- `run_manifest.json` records the run status: `in_progress`, `completed`, or `incomplete`.
- Resume is **not implemented**. An interrupted run is marked with `TRAINING_INCOMPLETE.txt` and must be restarted in a fresh output directory. Per-epoch checkpoints and optimizer state are still written for a future resume implementation.
- Pretrained ImageNet weights are **not downloaded automatically**. Training refuses to start unless the weights are already cached, or `allow_weight_download=True` is passed explicitly.
- Device reporting reflects the real runtime: CUDA is reported only when `torch.cuda.is_available()` is true.

## Outputs to save

After a training run, save at least:
- best validation checkpoint
- config snapshot
- per-epoch metrics and final metrics as JSON
- training and validation history
- split provenance and the seed used

Save them to a persistent folder such as `/content/drive/MyDrive/dr_baseline_outputs`, then download that folder from Drive or the Colab runtime.

## Limitations to keep in mind

- The split is provisional and pinned to the existing split artifact.
- The five-band taxonomy merges remain provisional and unsigned.
- 503 Zenodo-derived images lack verified patient linkage; patient independence is not claimed.
- This is a methodological baseline, not a clinical validation.

## Local verification before uploading

The smoke suite runs without a GPU and without downloading weights:

```bash
python -m pytest colab/smoke_test.py -q
```

If the image corpus is not auto-discovered, point the smoke tests at it explicitly:

```bash
DR_SMOKE_IMAGE_ROOT="/path/to/Diabetic Retinopathy and Retinal Diseases Fundus I" \
  python -m pytest colab/smoke_test.py -q
```

The full suite also runs on CPU and downloads nothing:

```bash
DR_SMOKE_IMAGE_ROOT="/path/to/Diabetic Retinopathy and Retinal Diseases Fundus I" \
  python -m pytest tests/ colab/smoke_test.py -q
```

Image-dependent checks are *skipped with a stated reason* when no corpus is reachable; they are never silently weakened. Do not expect GPU checks to pass on a CPU-only machine.

That suite includes executed end-to-end mini runs: two full epochs of `run_training` over a
five-image training slice and a three-image validation slice of the real split, and a
`run_final_evaluation` pass over a three-image test slice, both with the ImageNet weights and the
model architecture stubbed. They assert that no test image is decoded during training, that the
class weights come from the training rows only, that the weighted loss sits on the training device,
and that every metric file and manifest status lands where it should.

To check the dependency pins resolve without installing anything:

```bash
pip install --dry-run -r colab/requirements-colab.txt
```
