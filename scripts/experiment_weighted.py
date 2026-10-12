"""Q3 -- weighted conformal recovery under distribution shift.

Plain split conformal calibrated on the LP-PDBBind val split undercovers the
shifted test split. This runs weighted split conformal (Tibshirani et al. 2019):
a domain classifier on ECIF features estimates the likelihood ratio between the
calibration and evaluation inputs, and those weights are fed to the weighted
conformal quantile. The table contrasts unweighted vs weighted coverage/width.

Reads the per-split prediction dumps (``runs/predictions``) plus the cached ECIF
features (``data/processed/ecif``); CPU only, no GPU needed.

  python scripts/experiment_weighted.py --cal val --eval test --alpha 0.1 0.2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cap6545.conformal import (
    absolute_residual_scores,
    conformal_quantile,
    domain_density_ratios,
    weighted_conformal_quantile,
)
from cap6545.data.features import select_features
from cap6545.evaluation import empirical_coverage

PRED_DIR = Path("runs/predictions")
CACHE = Path("data/processed/ecif")
RESULTS_DIR = Path("runs/results")
PREDICTORS = ("det", "ens_mean")


def load_pred(split: str) -> dict[str, np.ndarray]:
    path = PRED_DIR / f"{split}.npz"
    if not path.exists():
        raise SystemExit(f"missing {path}; run scripts/predict.py (SLURM) first.")
    return dict(np.load(path, allow_pickle=True))


def load_features(ids: np.ndarray) -> np.ndarray:
    """Selected 790-dim ECIF features for ``ids``, in order, from the cache."""
    vecs = []
    for pdbid in ids:
        path = CACHE / f"{pdbid}.npy"
        if not path.exists():
            raise SystemExit(f"missing cached features {path}; featurize first.")
        vecs.append(np.load(path))
    return select_features(np.stack(vecs))


def width_summary(half: np.ndarray) -> dict[str, float]:
    """Full-width stats tolerant of infinite bands (robust median + inf fraction)."""
    w = 2.0 * half
    finite = w[np.isfinite(w)]
    return {
        "mean_width": float(np.mean(w)),  # inf if any band is infinite
        "median_width": float(np.median(w)),
        "finite_mean_width": float(np.mean(finite)) if finite.size else float("inf"),
        "frac_infinite": float(np.mean(~np.isfinite(w))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cal", default="val")
    parser.add_argument("--eval", nargs="+", default=["test"])
    parser.add_argument("--alpha", nargs="+", type=float, default=[0.1, 0.2])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cal = load_pred(args.cal)
    x_cal = load_features(cal["ids"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    header = (
        f"{'predictor':10} {'eval':6} {'method':10} {'alpha':>5} {'target':>7} "
        f"{'coverage':>9} {'med_w':>7} {'auc':>5}"
    )
    for eval_name in args.eval:
        ev = load_pred(eval_name)
        x_ev = load_features(ev["ids"])
        ratios = domain_density_ratios(x_cal, x_ev, seed=args.seed)
        print(f"\n== cal={args.cal} eval={eval_name}  domain-classifier AUC={ratios.auc:.3f} ==")
        print(header)
        print("-" * len(header))
        for predictor in PREDICTORS:
            scores = absolute_residual_scores(cal["y"], cal[predictor])
            for alpha in args.alpha:
                # unweighted split conformal (constant band)
                q = conformal_quantile(scores, alpha)
                half_u = np.full(ev["y"].size, q)
                # weighted split conformal (per-point band)
                half_w = weighted_conformal_quantile(
                    scores, ratios.cal_weights, ratios.test_weights, alpha
                )
                for method, half in (("unweighted", half_u), ("weighted", half_w)):
                    lo, hi = ev[predictor] - half, ev[predictor] + half
                    cov = empirical_coverage(ev["y"], lo, hi)
                    ws = width_summary(half)
                    print(
                        f"{predictor:10} {eval_name:6} {method:10} {alpha:5.2f} "
                        f"{1 - alpha:7.2f} {cov:9.4f} {ws['median_width']:7.3f} "
                        f"{ratios.auc:5.2f}"
                    )
                    rows.append(
                        {
                            "predictor": predictor,
                            "cal": args.cal,
                            "eval": eval_name,
                            "method": method,
                            "alpha": alpha,
                            "target_coverage": 1 - alpha,
                            "empirical_coverage": cov,
                            "domain_auc": ratios.auc,
                            "n": int(ev["y"].size),
                            **ws,
                        }
                    )

    out = RESULTS_DIR / "weighted_conformal.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
