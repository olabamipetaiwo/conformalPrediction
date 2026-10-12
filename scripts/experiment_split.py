"""Split conformal on real model predictions: calibrate on val, evaluate elsewhere.

Reads the per-split prediction dumps from ``runs/predictions`` and reports
coverage + width for plain split conformal (absolute-residual), for each point
predictor, at one or more miscoverage levels. The calibration split is held out
from training, so in-distribution coverage should land near ``1 - alpha``;
shift splits (e.g. bdb) are where coverage is expected to degrade.

  python scripts/experiment_split.py --cal val --eval test --alpha 0.1 0.2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cap6545.conformal import SplitConformal
from cap6545.evaluation import coverage_report

PRED_DIR = Path("runs/predictions")
RESULTS_DIR = Path("runs/results")
PREDICTORS = ("det", "ens_mean")


def load(split: str) -> dict[str, np.ndarray]:
    path = PRED_DIR / f"{split}.npz"
    if not path.exists():
        raise SystemExit(f"missing {path}; run scripts/predict.py (SLURM) first.")
    return dict(np.load(path, allow_pickle=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cal", default="val")
    parser.add_argument("--eval", nargs="+", default=["test"])
    parser.add_argument("--alpha", nargs="+", type=float, default=[0.1, 0.2])
    args = parser.parse_args()

    cal = load(args.cal)
    eval_sets = {name: load(name) for name in args.eval}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"calibration split: {args.cal} (n={cal['y'].size})\n")
    header = f"{'predictor':10} {'eval':6} {'alpha':>5} {'target':>7} {'coverage':>9} {'mean_w':>7}"
    print(header)
    print("-" * len(header))
    for predictor in PREDICTORS:
        for alpha in args.alpha:
            sc = SplitConformal.calibrate(cal["y"], cal[predictor], alpha=alpha)
            for name, ev in eval_sets.items():
                lo, hi = sc.intervals(ev[predictor])
                rep = coverage_report(ev["y"], lo, hi, alpha)
                print(
                    f"{predictor:10} {name:6} {alpha:5.2f} {rep.target_coverage:7.2f} "
                    f"{rep.empirical_coverage:9.4f} {rep.mean_width:7.3f}"
                )
                rows.append(
                    {"predictor": predictor, "cal": args.cal, "eval": name, **rep.as_dict()}
                )

    out = RESULTS_DIR / "split_conformal.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
