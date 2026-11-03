# Stabilized DR baseline pipeline — architecture, defects, verification

Status of this document: written from the code on disk, verified by the commands in
§6. It supersedes the cell-by-cell instructions in `colab/TASK_REPORT.md`,
`colab/FIXED_NOTEBOOK_REPORT.md` and the notes formerly in
`PIPELINE_STATUS_HANDOFF.md` (since removed; still readable in git history),
which describe the previous multi-cell notebook. The two `colab/` reports remain as history.

---

## 1. Architecture

One implementation, three thin ways to drive it.

```
                     colab/pipeline.py            <- the only entry point
                     (preflight, canary, train, evaluate, status)
                          |             |
      notebook cell  ----  |             |  ----  python -m colab.pipeline <command>
   (thin, 6 cells)         |             |
                           v             v
                  colab/train.py     colab/setup_helpers.py
                  (loop, checkpoints,  (project discovery, split +
                   resume, evaluation)  canary validators, smoke runner,
                          |             runtime inspection, output check)
                          v                    |
                  colab/colab_helpers.py       |
                  (datasets, preprocessing,    v
                   model, weights cache)   dataset/  (labels, decoding,
                          |                 split protocol, manifest)
                          v
                     torch / torchvision
```

Why it is shaped like this:

- **Every failure is reproducible outside the notebook.** Each step is a function in
  `colab/pipeline.py` and a subcommand, so a failure can be re-run in a terminal, in a
  test, or in a shell cell without re-executing earlier cells.
- **The notebook holds no logic.** It locates the project, mounts Drive (optionally),
  and calls four functions. There is no plan/install/validate logic in cells any more,
  so a cell cannot drift out of sync with the library.
- **Cell state is not part of the contract.** Only cell `# 1` sets `PROJECT_ROOT`; every
  later cell re-derives it (`try: PROJECT_ROOT / except NameError`) and the training
  cell re-runs preflight itself when cell `# 2` was skipped. Running cells out of order
  produces a real check, never a `NameError`.
- **The read-only boundary is explicit.** `pipeline.PROTECTED_DIRS` refuses an output
  directory that is the project root or one of its audited folders, and the only
  destructive operation (`clear_output_dir`) runs after cheap validation and only with
  an explicit `overwrite=True`.

### Guards (unchanged, deliberately)

| Gate | Where | Behaviour |
| --- | --- | --- |
| `RUN_TRAINING = False` | notebook cell 4 | the cell prints a message and does nothing |
| `confirm=True` | `pipeline.train_run` / `evaluate_run` | raises `PermissionError` without it |
| `smoke_test_passed=True` | `colab.train.run_training` | refuses to train unless forwarded as True (pipeline does this only after preflight) |
| weight consent | `allow_weight_download` | refuses when `resnet50-*.pth` is not cached and consent was not given |
| test isolation | `run_training` vs `run_final_evaluation` | training never loads test rows; only `evaluate` scores them, once |
| split integrity | `validate_split_artifact` | untouched; still requires 14,201 rows / 9,924 / 2,133 / 2,144 / all five bands |

---

## 2. The two commands that matter

```bash
# 1. Check everything before spending GPU time (exit 0 = safe to train).
python -m colab.pipeline preflight

# 2. Start training.
python -m colab.pipeline train --confirm

# 3. After a disconnect, continue from the last complete epoch.
python -m colab.pipeline train --confirm --resume

# 4. Score the held-out test partition once, from the best checkpoint.
python -m colab.pipeline evaluate --confirm
```

Useful extras:

```bash
python -m colab.pipeline status                     # what is in the output dir, is it resumable
python -m colab.pipeline canary                     # build + validate the 75-row canary
python -m colab.pipeline canary --train --confirm    # also run 2 epochs on it
python -m colab.pipeline train --confirm --allow-weight-download   # consent to the one-time ~100 MB
python -m colab.pipeline train --confirm --auto-resume             # resume if possible, else fresh
```

Exit codes: `0` ok, `2` preflight failed, `3` blocked by a guard or a missing
artifact, `130` interrupted. Add `--json` for machine-readable output.

Common flags: `--project-root`, `--dataset-root` (or `DR_FUNDUS_ROOT`), `--output-dir`
(or `DR_OUTPUT_DIR`; defaults to `<Drive>/dr_baseline_outputs` when Drive is mounted,
else `<project>/dr_baseline_outputs`).

### Notebook (the same four calls, in Colab)

| Cell | Do this | Passes when |
| --- | --- | --- |
| `# 1` Setup | run once | prints the project root, split dir, and output dir |
| `# 2` Preflight | run | ends with `PREFLIGHT OK`; on failure it stops with every FAIL and a `->` fix |
| `# 3` Canary | optional: `RUN_CANARY = True` | 75 rows, 50/25/0 validated; no test rows |
| `# 4` Train | set `RUN_TRAINING = True`, run | logs each epoch; after a disconnect, run it again (resume is automatic) |
| `# 5` Evaluate | set `RUN_FINAL_EVALUATION = True` | writes `final_test_evaluation/final_test_metrics.json` |
| `# 6` Verify | run | `VERIFICATION_OK: True` |

---

## 3. Defects found in this audit and what changed

| # | Defect | Evidence it was real | Fix |
| --- | --- | --- | --- |
| 1 | **Resume reset the best/early-stopping trackers.** After restoring state, the loop re-initialised `best_val_loss=inf`, `best_epoch=0`, `patience_counter=0`, so the first resumed epoch always looked like a new best and overwrote the real best checkpoint with worse weights, and early stopping restarted from zero. | Reading `run_training`'s loop initialisation against `state` | Initialise the loop from the restored state (`train.py`) |
| 2 | **Resume could never start.** `resolve_output_dir(..., overwrite=False)` ran before resume planning and raised `FileExistsError` for any directory that already held a run — i.e. for every resumable run. | `run_training` call order; a resume test failed immediately | Pass `overwrite or resume` to the guard; nothing is deleted, since the destructive path still keys off `overwrite` alone |
| 3 | **`run_manifest.json` hardcoded `resume_supported: False`** and never recorded where a run resumed from. | `write_run_manifest` | Derive it from the config and add `resumed_from` |
| 4 | **`TRAINING_INCOMPLETE.txt` told users to restart from scratch** and that resume did not exist. | `mark_incomplete` | Points at `python -m colab.pipeline train --confirm --resume` |
| 5 | **Discovery ignored a wrong explicit project root** and silently resolved a different checkout (from cwd or a bounded scan). | `discover_project`; a test now fails without the fix | An explicit path (or `DR_PROJECT_ROOT`) is validated and fails closed, naming the missing markers |
| 6 | **Two notebooks claimed to be the entry point.** `colab/train_on_colab.ipynb` (10 cells) and `train_on_colab_fixed.ipynb` (11 cells) both looked authoritative; the smoke suite tested the old one, so the documented entry point was effectively untested. | `smoke_test.NOTEBOOK_PATH` pointed at `train_on_colab.ipynb` | All tests now target `train_on_colab_fixed.ipynb`; the old notebook is left on disk but documented as superseded |
| 7 | **The notebook reimplemented the pipeline** (dependency plan, install, discovery, split validation, smoke, canary, training) across 11 cells, so every fix had to be made twice. | Notebook source | Notebook is 6 cells over `colab/pipeline.py`; tests assert the delegation |
| 8 | **No single preflight.** Environment, split, corpus, and smoke were checked in four different cells, each failing separately. | Notebook source | `pipeline.preflight` checks them all and reports every failure with a hint |
| 9 | **Missing image paths were only discovered mid-epoch**, after the model was built. | `preflight` now exits on the first bad path; test `test_preflight_flags_a_missing_image_instead_of_failing_later` | All 14,201 paths checked before training |
| 10 | **A missing corpus made the image checks silently skip** rather than fail. | `preflight` fails closed (`skipped because the corpus root is unresolved`) | `test_preflight_reports_every_unresolved_artifact_together` |
| 11 | **Unicode in printed policy text crashed the last cell** on a cp1252 console (`⚠` → `UnicodeEncodeError`). | Executing cell `# 6` on Windows | Cell `# 1` reconfigures stdout/stderr with `errors="replace"`; the full cell was executed afterwards |
| 12 | **No run log on disk.** The notebook's stdout died with the session. | Notebook source | Every command mirrors stdout/stderr into `<output_dir>/logs/<step>_<timestamp>.log` (tested by `test_train_run_mirrors_its_output_into_a_log_beside_the_artifacts`) |
| 13 | **Checkpoints were not written atomically.** A disconnect during `torch.save` could leave a truncated `model.pt` that `latest_epoch_checkpoint` counted as a valid resume point, so the next resume would fail on a corrupt file. | `save_checkpoint` wrote straight to `model.pt`; the resume logic only checks that the file exists | Writes now go to a temp file and are renamed (`_atomic_save`, `_atomic_copy`, `_atomic_write_json`); a failed write leaves the previous checkpoint untouched and no `.tmp` behind |

### Files changed

- `colab/train.py` — resume correctness (#1, #2, #3, #4) plus `plan_resume`, `load_epoch_checkpoint`, `write_resume_state`, `latest_epoch_checkpoint`.
- `colab/pipeline.py` — **new**: preflight, canary, train/resume, evaluate, status, CLI, logging, output-dir guards.
- `colab/train_on_colab_fixed.ipynb` — **rewritten**: 6 thin cells (setup, preflight, canary, train/resume, evaluate, verify).
- `colab/smoke_test.py` — notebook assertions now target the real entry point and its contract.
- `tests/test_pipeline_entrypoint.py` — **new**: preflight, discovery, output guards, CLI exit codes, status.
- `tests/test_colab_pipeline.py` — resume tests on a synthetic corpus; call-signature contract now checks `pipeline.py` against `train.py`; `resume_supported` expectation updated.
- `tests/test_canary_split.py` — canary contract moved from notebook text to `pipeline.canary_run` (same assertions, now against executed code).
- `tests/test_fixed_notebook_setup.py` — notebook contract rewritten for the 6-cell interface.

---

## 4. What is checked before training starts (`preflight`)

| Check | Fails closed? |
| --- | --- |
| project markers (`dataset/labels.py`, `colab/train.py`, split CSV, `colab/setup_helpers.py`) | yes |
| Python + torch / torchvision / numpy / Pillow versions against `MINIMUM_VERSIONS` | yes |
| pytest present (the smoke gate needs it) | warn |
| CUDA availability, with the CPU fallback stated | warn |
| pinned split via the unchanged strict validator: 14,201 rows, 9,924 / 2,133 / 2,144, five bands, protocol checks | yes |
| corpus root resolvable | yes (skips nothing) |
| **all 14,201 image paths exist** | yes |
| decode sample: 2 images per (partition, class) = 10 strata × 2 | yes |
| eligibility manifest present (for protocol re-validation) | warn |
| ImageNet weights cached | warn (`--allow-weight-download` overrides) |
| output dir writable and not protected | yes |
| smoke suite (`python -m pytest colab/smoke_test.py -q`), real exit code | yes |

A failed preflight is a non-zero exit and, in the notebook, a `SystemExit` before any
training work.

---

## 5. Resume semantics

Checkpoint layout, all inside the output directory:

```
run_meta.json                     config, resolved device, class weights
run_manifest.json                 status: in_progress | completed | incomplete
resume_state.json                 epoch count + best-loss/early-stopping bookkeeping
checkpoints/epoch_XXXX/model.pt   weights + optimizer state + config
checkpoints/epoch_XXXX/training_state.json
best_checkpoint/{model.pt,training_state.json}   validation-selected copy
run_summary.json                  per-epoch histories (written on completion)
logs/<step>_<timestamp>.log       mirrored stdout/stderr
TRAINING_INCOMPLETE.txt           present only for an interrupted run
```

Every write is temp-file-plus-rename, so a disconnect mid-write can never leave a
truncated `model.pt` that looks like a resume point.

`--resume` restores model weights, **optimizer state**, the epoch histories, the
best-validation bookkeeping, and the early-stopping counter, then trains from
`last_complete_epoch + 1` up to `--epochs`.

Refusals, each with the exact next step instead of a traceback:

- no run in the directory → `FileNotFoundError` naming the path;
- a run with no complete `epoch_XXXX/model.pt` → cannot resume, start with `--overwrite`;
- `best_epoch` selected but `best_checkpoint/model.pt` missing → the selected weights are gone;
- `completed` run asked for the same or fewer epochs → use `--epochs` greater than trained;
- `--resume` with `--overwrite` → contradictory, choose one.

---

## 6. Verification

Command:

```bash
python -m pytest colab/smoke_test.py tests/ -q
```

Result on this checkout: **271 passed, 3 skipped** (the 3 skips are the repo's
pre-existing image-dependent checks — no fundus corpus is present).

### What actually ran

| Area | How | Real or synthetic |
| --- | --- | --- |
| Split artifact, labels, grouping, manifest, eligibility | existing suites | **real** audited artifacts |
| Preflight happy path: 14,201 paths + 10 decode strata | `test_preflight_passes_on_a_complete_project` | real split + **synthetic** 16×12 images |
| Preflight failures: missing image, undecodable image, unresolved corpus, failing smoke | 4 tests | real split + synthetic corpus |
| Preflight dependency verdicts | `test_preflight_reports_the_real_dependencies_truthfully` | **real** environment (torchvision absent here) |
| CLI: exit codes, `[BLOCKED]` messages, `--json`, status | 6 subprocess tests | **real** CLI |
| Resume: continue, best-tracker preservation, optimizer restore, refusals, extension | 6 tests running the **real** training loop on a 75-row derived split | synthetic corpus, CPU, stubbed model + runtime |
| Canary: path, validation-before-training, weight consent, no test rows | 8 tests over `pipeline.canary_run` | real 14,201-row split + synthetic corpus |
| Notebook: 6 cells compile, order, opt-in defaults, delegation | 13 tests, plus a full in-order execution of all six cells | **real** notebook source; execution on this machine stops at preflight (no corpus, no torchvision) as designed |
| Persistent logs: `train` and `evaluate` mirror output into `<output_dir>/logs` | 2 tests over the real `train_run` / `evaluate_run` with the loop stubbed | synthetic project layout |
| Audited artifacts unchanged | `test_the_audited_split_is_still_the_audited_split`: 14,201 rows, 9,924 / 2,133 / 2,144, five bands, five-band mapping, no canary written into the audited folder | **real** artifacts |
| Atomic checkpoints: successful write, simulated mid-write crash, unusable epoch ignored on resume | 3 tests | synthetic tensors |
| Training on real data | not run (deliberately) | — |
| CUDA training | not run (no GPU here) | — |
| Resume after a real Colab disconnect | not run; simulated by writing the incomplete marker and resuming | — |
| Colab Drive mount | not run (no Colab here); the mount is wrapped and skipped locally | — |

The 3 skips: `test_mini_training_run_is_validation_selected_and_never_reads_test_images`,
`test_mini_run_stops_early_after_patience_is_exhausted`,
`test_mini_final_evaluation_scores_only_the_test_partition` (they need the real corpus;
the resume tests cover the same loop on synthetic images).

### Honest limitations

1. **No GPU run and no real-data run happened.** CUDA correctness rests on the code
   paths (`resolve_device`, `torch.cuda.manual_seed_all`, loss on the model's device)
   and on CPU execution, not on a GPU measurement.
2. **Resume was proven on a synthetic CPU run**, not by killing a real Colab session.
   The state that a disconnect leaves behind (checkpoints + `TRAINING_INCOMPLETE.txt`)
   was simulated exactly, and the loop was executed for real.
3. **`torchvision` is not installed in this environment**, so preflight here ends with
   `dependency_torchvision` failing. Colab ships torch and torchvision, but the first
   Colab run of `preflight` is the authoritative check.
4. **Drive persistence is a prediction.** `default_output_dir` prefers
   `/content/drive/MyDrive`, and cell `# 1` mounts Drive when it can, but neither was
   exercised on Colab.
5. **`colab/train_on_colab.ipynb` is still on disk** and no longer tested or
   referenced. Deleting it is a one-line cleanup this audit did not take on itself.
6. **`colab/_canary_test_outputs/canary_provisional_split.csv`** is a stale,
   unreferenced leftover from an older layout. Nothing reads it.
7. **The label mapping remains provisional and unsigned.** Nothing here is a clinical
   result, and `evaluate` prints that caveat with the metrics.

---

## 7. Minimal manual checklist

1. `Runtime → Change runtime type → GPU`.
2. Upload/clone the project so `dataset/` and `colab/` are together; mount Drive if the
   project lives there.
3. In a cell (or a terminal): `python -m colab.pipeline preflight`
   → must end `PREFLIGHT OK`. If not, every FAIL line names its fix; no other step is
   worth running yet.
4. `python -m colab.pipeline train --confirm` (add `--allow-weight-download` the first
   time). In the notebook: cell `# 1`, `# 2`, then `# 4` with `RUN_TRAINING = True`.
5. If the runtime disconnects: re-run the same training command, or cell `# 4` again.
   `--resume` is implicit in the notebook (`RESUME = "auto"`).
6. `python -m colab.pipeline evaluate --confirm` once, after `status` says `completed`.
7. `python -m colab.pipeline status` and cell `# 6` to confirm the artifacts.

That is 3 actions for a normal run (preflight, train, evaluate) and 1 action to
recover from a disconnect.
