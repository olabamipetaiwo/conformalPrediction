"""Q1 -- does the calibration-set choice matter under distribution shift?

Split conformal only promises coverage when the calibration set is exchangeable
with the evaluation set. LP-PDBBind's val split (CL2) and test split (CL2) share a
class, but both differ from the CL1 training distribution. This isolates the effect
of *where the calibration data comes from* by holding the evaluation set fixed and
swapping only the calibration source:

  val-calib   -- calibrate on the official val split (CL2, held out from training)
  test-oracle -- calibrate on a random hold-out carved from test itself
                 (same distribution as the eval points; an upper bound)

Both are scored on the *same* remaining test points, averaged over random draws of
the test hold-out, so any coverage gap is attributable to the calibration source's
distribution, not to sample size or the eval points. A third arm -- a genuine random
*training* hold-out (CL1 distribution, out-of-sample) -- requires a retrain that
reserves a calibration partition and is run separately; see docs/q1_training_holdout.md.

Reads the prediction dumps (``runs/predictions``); CPU only.

  python scripts/experiment_calibration.py --eval test --alpha 0.1 0.2 --repeats 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cap6545.conformal import SplitConformal
from cap6545.evaluation import empirical_coverage, mean_width

PRED_DIR = Path("runs/predictions")
RESULTS_DIR = Path("runs/results")
PREDICTOR = "ens_mean"


def load(split: str) -> dict[str, np.ndarray]:
    path = PRED_DIR / f"{split}.npz"
    if not path.exists():
        raise SystemExit(f"missing {path}; run scripts/predict.py (SLURM) first.")
    return dict(np.load(path, allow_pickle=True))


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", default="val")
    parser.add_argument("--eval", default="test")
    parser.add_argument("--alpha", nargs="+", type=float, default=[0.1, 0.2])
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--cal-frac", type=float, default=0.3, help="fraction of eval split used as test-oracle cal"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    val = load(args.val)
    ev = load(args.eval)
    rng = np.random.default_rng(args.seed)
    n = ev["y"].size
    n_cal = round(args.cal_frac * n)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    header = f"{'calib':12} {'alpha':>5} {'target':>7} {'coverage(mean±std)':>22} {'mean_w':>8}"
    print(
        f"predictor={PREDICTOR}  eval={args.eval} (n={n})  test-oracle cal n≈{n_cal}  "
        f"repeats={args.repeats}"
    )
    print(header)
    print("-" * len(header))
    for alpha in args.alpha:
        # val-calib: fixed calibration on val, evaluated on the same random eval subsets
        # so the two arms are compared on identical eval points.
        val_sc = SplitConformal.calibrate(val["y"], val[PREDICTOR], alpha=alpha)
        val_cov, val_w, orc_cov, orc_w = [], [], [], []
        for _ in range(args.repeats):
            perm = rng.permutation(n)
            cal_idx, eval_idx = perm[:n_cal], perm[n_cal:]
            ye, pe = ev["y"][eval_idx], ev[PREDICTOR][eval_idx]

            lo, hi = val_sc.intervals(pe)
            val_cov.append(empirical_coverage(ye, lo, hi))
            val_w.append(mean_width(lo, hi))

            orc = SplitConformal.calibrate(
                ev["y"][cal_idx], ev[PREDICTOR][cal_idx], alpha=alpha
            )
            lo, hi = orc.intervals(pe)
            orc_cov.append(empirical_coverage(ye, lo, hi))
            orc_w.append(mean_width(lo, hi))

        for label, cov, wid in (
            ("val-calib", val_cov, val_w),
            ("test-oracle", orc_cov, orc_w),
        ):
            cs, ws = summarize(cov), summarize(wid)
            print(
                f"{label:12} {alpha:5.2f} {1 - alpha:7.2f} "
                f"{cs['mean']:9.4f} ± {cs['std']:.4f}      {ws['mean']:8.3f}"
            )
            rows.append(
                {
                    "calib": label,
                    "eval": args.eval,
                    "alpha": alpha,
                    "target_coverage": 1 - alpha,
                    "coverage_mean": cs["mean"],
                    "coverage_std": cs["std"],
                    "mean_width": ws["mean"],
                    "repeats": args.repeats,
                }
            )

    out = RESULTS_DIR / "calibration_choice.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
