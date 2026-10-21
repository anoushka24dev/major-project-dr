# Colab runbook — provisional frozen ResNet-50 baseline

The authoritative description of the pipeline, its defects, and its verification is
[`colab/PIPELINE_GUIDE.md`](PIPELINE_GUIDE.md). This file is only the click-by-click
runbook.

**Nothing in this file has been executed in Colab.** Locally, everything except real
GPU/real-corpus execution has been run (see the guide, §6).

---

## Before you start

- Runtime → Change runtime type → **GPU**.
- The project folder must contain `dataset/` and `colab/`. Set `DR_PROJECT_ROOT` (or the
  `OVERRIDE_PROJECT_ROOT` variable in cell `# 1`) if discovery cannot find it.
- Make the fundus corpus reachable: set `DR_FUNDUS_ROOT` to the folder that holds the
  source-label subfolders (e.g. `11.Normal Fundus/`).
- Outputs default to `<Drive>/dr_baseline_outputs` when Drive is mounted, so a
  disconnect does not lose the run. Override with `DR_OUTPUT_DIR`.

## Step 1 — Preflight (the only gate)

Notebook cell `# 2`, or `python -m colab.pipeline preflight`.

Passes when the last line is `PREFLIGHT OK`. It checks: project markers, Python/torch/
torchvision/numpy/Pillow versions, CUDA or CPU fallback, the pinned 14,201-row split via
the unchanged strict validator, all 14,201 image paths, a decode sample from every
(partition, class), the eligibility manifest, the weight cache, output-dir writability,
and the smoke suite's real exit code.

Stop if it ends `PREFLIGHT FAILED`: each `[FAIL]` line has a `->` fix. Do not run
training until it passes.

## Step 2 — Train

Notebook cell `# 4` with `RUN_TRAINING = True`, or:

```bash
python -m colab.pipeline train --confirm
python -m colab.pipeline train --confirm --allow-weight-download   # first run, weights not cached
```

Expected evidence: a `training_start` line with the device, then one `epoch_end` line per
epoch with `val_loss`, `val_accuracy`, `val_macro_f1`, `best_epoch`, and `patience_counter`.

## Step 3 — After a disconnect

Re-run the same cell (or command) once. The notebook uses `RESUME = "auto"`.

```bash
python -m colab.pipeline train --confirm --resume
```

Expected evidence: a `resume_start` line naming the checkpoint, the restored epoch, the
restored `best_val_loss`, and `start_epoch` > 0. Training continues from the next epoch;
nothing already trained is retrained and the best checkpoint is never overwritten by a
worse epoch.

Check state at any time:

```bash
python -m colab.pipeline status
```

## Step 4 — Evaluate once

Notebook cell `# 5` with `RUN_FINAL_EVALUATION = True`, or:

```bash
python -m colab.pipeline evaluate --confirm
```

It refuses unless `status` says `completed` and `best_checkpoint/model.pt` exists. Writes
`final_test_evaluation/final_test_metrics.json` (accuracy, macro precision/recall/F1,
per-class metrics, confusion matrix).

Do not re-tune against this number. Running this step again overwrites the metrics file.

## Step 5 — Verify the artifacts

Notebook cell `# 6`, or `python -m colab.pipeline status`.

Passes when `VERIFICATION_OK: True`: `run_meta.json`, `run_manifest.json` (`completed`),
`run_summary.json`, `best_checkpoint/model.pt`, the per-epoch checkpoints, no
`TRAINING_INCOMPLETE.txt`, and — if evaluation ran — the test metrics file.

## Optional — the canary

Notebook cell `# 3` with `RUN_CANARY = True` (add `CANARY_TRAIN = True` for 2 epochs), or:

```bash
python -m colab.pipeline canary
```

It builds the 75-row derived split at `<output_dir>/canary_split/provisional_primary_split.csv`
(50 / 25 / 0), validates it with the canary-specific validator (all failure modes closed),
and leaves the strict full-split validator untouched. A canary proves the plumbing; it has
no test partition and its numbers are not results.

## Stop conditions

| Symptom | Meaning | Next action |
| --- | --- | --- |
| `PREFLIGHT FAILED — dependency_torchvision` | torchvision missing/too old | `python -m pip install -r colab/requirements-colab.txt` |
| `dataset_root: the fundus corpus was not found` | corpus not reachable | set `DR_FUNDUS_ROOT` |
| `N of 14201 image paths are missing` | wrong corpus root or incomplete upload | point `DR_FUNDUS_ROOT` at the real folder |
| `training blocked: preflight failed` | the gate did its job | fix the preflight failures |
| `[BLOCKED] ... weights are not cached` | no consent to download | add `--allow-weight-download` |
| `[BLOCKED] ... already finished ... 'completed'` | asked to resume a finished run for no new epochs | pass a larger `--epochs`, or `--overwrite` for a fresh run |
