# Q1 — the training-hold-out calibration arm (requires a retrain)

`scripts/experiment_calibration.py` answers Q1 for the two calibration sources we
can study from the existing prediction dumps:

- **val-calib** — the official LP-PDBBind val split (CL2), held out from training.
- **test-oracle** — a random hold-out carved from the test split itself (same
  distribution as the eval points; the coverage upper bound).

The proposal's third comparator — a **random *training* hold-out** (CL1
distribution) — cannot be read off the current dumps, because the deep ensemble was
trained on the full CL1 train split (7,131 complexes). Residuals on any training
point are therefore in-sample (optimistically small) and would make the conformal
band invalid. A faithful training-hold-out arm needs calibration points that were
**never seen during training**, i.e. a retrain.

## How to produce it

1. Reserve a random calibration partition from CL1 train *before* fitting (e.g. 10%,
   ~713 complexes), fixed by `--cal-seed`, and train each ensemble seed on the
   remaining ~6,418.
2. Dump predictions on that held-out calibration partition alongside val/test.
3. Add a `train-holdout` arm to the Q1 table: calibrate on those out-of-sample
   training-distribution residuals, evaluate on test.

Expected result: because the CL1 calibration distribution differs from the CL2 test
distribution even more than val does, `train-holdout` should undercover at least as
badly as `val-calib` — reinforcing that calibration-set *distribution*, not sample
count, drives Q1, and motivating the weighted-conformal fix (Q3).

This is a GPU SLURM job (one retrain of the 10-seed ensemble on a reduced train
split). It is gated on an explicit go-ahead because it reserves queue time and fixes
an experimental-design choice (hold-out fraction and seed).
