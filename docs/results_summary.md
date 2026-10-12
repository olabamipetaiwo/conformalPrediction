# Conformal-under-shift results (as of 2026-10-12)

Base model: 10-seed deep ensemble of the FFNN-ECIF reproduction (best val RMSE
~1.60-1.62). Point predictor `ens_mean` unless noted. All numbers are reproducible
from the committed scripts over the prediction dumps in `runs/` (gitignored).

Reproduce:
```
python scripts/experiment_calibration.py --pred-dir runs/predictions          # Q1 (val vs oracle)
python scripts/experiment_calibration.py --pred-dir runs/q1_holdout/predictions # Q1 (+train-holdout)
python scripts/experiment_normalized.py  --cal val --eval test peptides_nc bdb  # Q2
python scripts/experiment_weighted.py     --cal val --eval test peptides_nc bdb  # Q3
```

## Shift is real and increases with dataset distance

Ensemble RMSE (pK units): test 1.6 → peptides-holdout **2.75** → BDB2020+ **3.10**.
Domain-classifier AUC (val vs eval, ECIF features): test **0.78**, peptides **0.99**,
BDB **1.00**. BDB/peptides are near-separable from the calibration distribution.

## Q1 — calibration-set choice (eval=test, 50 repeats, eval points held fixed)

| calibration source | cov @0.90 | cov @0.80 | mean width @0.90 |
|---|---|---|---|
| train-holdout (CL1, never trained on) | **0.694** | 0.584 | 4.06 |
| val split (CL2) | 0.823 | 0.716 | 5.61 |
| test-oracle (in-distribution hold-out) | **0.899** | 0.802 | 7.11 |

Coverage tracks distributional distance from the test set exactly:
CL1 training hold-out undercovers worst, the official val split is better but still
short, and only in-distribution calibration reaches nominal. **The calibration set's
distribution — not its size — drives coverage.** (train-holdout from the retrain that
reserved 713 CL1 complexes; see `docs/q1_training_holdout.md`.)

## Q2 — uncertainty-normalized scores (cal=val, target 0.90)

| variant | test | peptides | bdb |
|---|---|---|---|
| ensemble, plain | 0.836 | 0.678 | 0.504 |
| ensemble, std-normalized | 0.945 | 0.860 | 0.713 |
| MC-dropout, std-normalized | 0.870 | 0.837 | 0.435 |

Normalizing by ensemble disagreement recovers (and on test overshoots) coverage with
finite, adaptive widths — the disagreement carries shift-relevant signal. The recovery
shrinks as the shift deepens and never reaches nominal on BDB.

## Q3 — weighted conformal (cal=val, target 0.90)

| eval | domain AUC | unweighted cov | weighted cov | weighted width |
|---|---|---|---|---|
| test | 0.78 | 0.836 | 0.846 | finite (~5.7) |
| peptides | 0.99 | 0.678 | 0.958 | **mostly infinite** |
| bdb | 1.00 | 0.504 | 0.991 | **infinite** |

**Key finding (a limitation, not a win).** Weighted conformal helps only in the mild,
overlapping-support regime (val→test): a small, finite-width coverage gain, with a
residual gap attributable to concept shift (`y|x`) that density ratios cannot fix.
Under severe shift (peptides, BDB) the domain classifier is near-perfect, the
likelihood ratios diverge, and weighting "restores coverage" only by inflating most
intervals to **+∞** — a vacuous guarantee. This is the expected consequence of
Tibshirani et al.'s requirement that the calibration and target supports overlap
(bounded likelihood ratio); it fails exactly where the shift is largest.

## Takeaway for the writeup

The three questions compose into one narrative: split conformal's marginal guarantee
is only as good as the exchangeability it assumes (Q1); per-point uncertainty buys
back some adaptivity under moderate shift (Q2); and reweighting recovers coverage only
while the shift is mild enough that support overlaps, degenerating to infinite
intervals under the strong shifts that matter most (Q3).
