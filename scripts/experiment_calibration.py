"""Q1 -- does the calibration-set choice matter under distribution shift?

Split conformal only promises coverage when the calibration set is exchangeable
with the evaluation set. LP-PDBBind's val split (CL2) and test split (CL2) share a
class, but both differ from the CL1 training distribution. This isolates the effect
of *where the calibration data comes from* by holding the evaluation set fixed and
swapping only the calibration source:

  val-calib     -- calibrate on the official val split (CL2, held out from training)
  train-holdout -- calibrate on a random CL1 *training* hold-out never trained on
                   (only available when ``--pred-dir`` points at the Q1 retrain dumps)
  test-oracle   -- calibrate on a random hold-out carved from test itself
                   (same distribution as the eval points; an upper bound)

Every arm is scored on the *same* remaining test points, averaged over random draws of
the test hold-out, so any coverage gap is attributable to the calibration source's
distribution, not to sample size or the eval points.

``--pred-dir`` selects which model's dumps to use. Point it at ``runs/predictions``
(baseline ensemble) for the val-vs-oracle comparison, or at
``runs/q1_holdout/predictions`` (the retrain that reserved a calibration partition)
to add the ``train-holdout`` arm; see docs/q1_training_holdout.md.

Reads the prediction dumps; CPU only.

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


def load(pred_dir: Path, split: str) -> dict[str, np.ndarray]:
    path = pred_dir / f"{split}.npz"
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
    parser.add_argument("--pred-dir", type=Path, default=PRED_DIR)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "calibration_choice.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    val = load(args.pred_dir, args.val)
    ev = load(args.pred_dir, args.eval)
    # Fixed-calibration arms: (label, calibrated SplitConformal keyed by alpha).
    fixed_sources = {"val-calib": val}
    holdout_path = args.pred_dir / "train_holdout.npz"
    if holdout_path.exists():
        fixed_sources["train-holdout"] = load(args.pred_dir, "train_holdout")

    rng = np.random.default_rng(args.seed)
    n = ev["y"].size
    n_cal = round(args.cal_frac * n)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    header = f"{'calib':13} {'alpha':>5} {'target':>7} {'coverage(mean±std)':>22} {'mean_w':>8}"
    print(
        f"predictor={PREDICTOR}  pred_dir={args.pred_dir}  eval={args.eval} (n={n})  "
        f"test-oracle cal n≈{n_cal}  repeats={args.repeats}"
    )
    print(header)
    print("-" * len(header))
    for alpha in args.alpha:
        fixed_sc = {
            label: SplitConformal.calibrate(src["y"], src[PREDICTOR], alpha=alpha)
            for label, src in fixed_sources.items()
        }
        cov: dict[str, list[float]] = {label: [] for label in fixed_sc}
        wid: dict[str, list[float]] = {label: [] for label in fixed_sc}
        cov["test-oracle"], wid["test-oracle"] = [], []
        for _ in range(args.repeats):
            perm = rng.permutation(n)
            cal_idx, eval_idx = perm[:n_cal], perm[n_cal:]
            ye, pe = ev["y"][eval_idx], ev[PREDICTOR][eval_idx]

            for label, sc in fixed_sc.items():
                lo, hi = sc.intervals(pe)
                cov[label].append(empirical_coverage(ye, lo, hi))
                wid[label].append(mean_width(lo, hi))

            orc = SplitConformal.calibrate(ev["y"][cal_idx], ev[PREDICTOR][cal_idx], alpha=alpha)
            lo, hi = orc.intervals(pe)
            cov["test-oracle"].append(empirical_coverage(ye, lo, hi))
            wid["test-oracle"].append(mean_width(lo, hi))

        order = [*fixed_sc.keys(), "test-oracle"]
        for label in order:
            cs, ws = summarize(cov[label]), summarize(wid[label])
            print(
                f"{label:13} {alpha:5.2f} {1 - alpha:7.2f} "
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

    args.out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
