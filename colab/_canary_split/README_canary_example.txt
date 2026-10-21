EXAMPLE ONLY — not an audited artifact.

This tiny derived split was built from the existing pinned split by colab.setup_helpers.build_canary_split
with the notebook defaults (train_per_class=10, val_per_class=5, test_per_class=0).

It is included so colab/train_on_colab_fixed.ipynb cell #7 can be inspected offline.

To rebuild it yourself, set CANARY_RUN = True in the notebook.

Derived split contents:
- 75 rows total
- train: 50
- validation: 25
- test: 0
- classes present: No apparent DR, Mild NPDR, Moderate NPDR, Severe NPDR, PDR

This folder is NOT part of the audited split. It lives under colab/, not under provisional_primary_split/.

NOTE: The CSV is at canary_split/provisional_primary_split.csv (not at the folder root),
because validate_canary_split() and run_training() (via colab_helpers.load_split_csv) both read
split_dir / "provisional_primary_split.csv". A canary written anywhere else validates an empty folder.

The non-image checks of colab.setup_helpers.validate_canary_split pass on this folder. The image checks
cannot pass here because the fundus corpus is not part of this repository; supply DR_FUNDUS_ROOT (or run
cell #7 in Colab) to exercise them.
