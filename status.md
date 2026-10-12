# Status — pick-up-here

**As of:** 2026-10-12. **Final due:** 2026-12-08 15:00. **Schedule position:** ~6–7
weeks ahead (see `report.md` §8; the Q3 core finding, scheduled Nov 9–23, is done).

Read order for a fresh session: this file → `docs/results_summary.md` → `CLAUDE.md`
→ `report.md`. The results doc has the numbers; this file has the state and the
next move.

## Done (all three research questions have first-pass results)

- **Base model:** 10-seed deep ensemble, FFNN-ECIF reproduction. Checkpoints in
  `runs/deterministic/seed_*`. Best val RMSE ~1.60–1.62.
- **Predictions dumped** (`runs/predictions/*.npz`): val, test, peptides_nc (2,350),
  bdb (115). Columns: `det, ens_mean, ens_std, mc_mean, mc_std`.
- **Q1** (calibration-set choice): val vs test-oracle (`runs/predictions`) AND the
  train-holdout retrain arm (`runs/q1_holdout/`, reserved 713 CL1 complexes).
  Result: coverage tracks distance from test — train-holdout 0.694 < val 0.823 <
  oracle 0.899 @0.90. **Distribution, not size, drives coverage.**
- **Q2** (uncertainty-normalized): ensemble-std normalization recovers coverage with
  finite adaptive widths; weakens as shift deepens.
- **Q3** (weighted conformal): helps only under mild overlapping-support shift
  (val→test, AUC 0.78); under severe shift (peptides AUC 0.99, bdb AUC 1.00) bands
  blow up to +∞ — coverage "recovered" but vacuous. The headline limitation.

All experiment scripts are in `scripts/experiment_*.py`; reproduce commands are in
`docs/results_summary.md`. 30 fast tests pass, ruff clean.

## Not done

- **Robustness/CIs** — results are single-run. No multi-seed/bootstrap confidence
  intervals on the conformal numbers yet. (Recommended next; see below.)
- **Figures** — none yet.
- **Writeup** — not started (`paper:write` skill available).
- **BbB** — not started (cut-list item #1; now affordable given the buffer).
- **Selective-prediction analysis** — not started (cut-list item #2).

## Key caveats to carry forward

- Trainable counts are 7,131 / 947 / 2,160 (not 7,259 / 947 / 2,166) — 134 split
  complexes absent from PDBbind v2020.R1. Documented deviation, not a bug.
- Peptides-Holdout: 2,350 of 2,498 have structures on disk (148 missing).
- BDB2020+: 115 labeled rows (CSV `pKa` column is the target), not the proposal's
  111 — known discrepancy.
- Q3 weighted-conformal infinite widths are correct behavior, not a code bug: the
  near-perfect domain classifier makes likelihood ratios diverge (support barely
  overlaps), so the weighted quantile hits the +∞ atom. `runs/results/
  weighted_conformal.json` records `frac_infinite` / `finite_mean_width` per row.

## Recommended next move

**Harden before adding scope.** Run each conformal experiment over multiple
calibration seeds / ensemble bootstraps to get confidence intervals, so the Q1/Q2/Q3
tables are defensible, then build figures. BbB only after the core paper is solid.
`experiment_calibration.py` already averages over repeats; `experiment_weighted.py`
and `experiment_normalized.py` would need a `--repeats`/seed loop added.

## How to run things (HiPerGator, SLURM only — never login-node/CPU for training)

```
sbatch --array=0-9 scripts/slurm/train_ensemble.sh      # baseline ensemble
sbatch --array=0-9 scripts/slurm/train_q1_holdout.sh    # Q1 hold-out ensemble
sbatch scripts/slurm/featurize.sh                        # ECIF cache (CPU)
SPLITS="val test peptides_nc bdb" sbatch scripts/slurm/predict.sh
sbatch scripts/slurm/predict_q1.sh                       # train_holdout+val+test dumps
```
Account/qos `so589980.ucf`; GPU partition `hpg-turin`, CPU `hpg-default`.
Analyses (CPU, fine to run directly): the `scripts/experiment_*.py` commands in
`docs/results_summary.md`.

## Git

Clean on `main`. Latest: `docs: conformal-under-shift results summary`. `runs/` and
`data/` are gitignored — results live in `docs/results_summary.md`, not the repo tree.
Tag the Dec 8 commit `v0.1-course-submission`.
