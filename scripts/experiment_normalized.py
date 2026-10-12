"""Q2 -- uncertainty-normalized conformal scores.

Plain split conformal gives every test point the same band width. Normalizing the
nonconformity score by a per-point uncertainty (ensemble disagreement or MC-dropout
spread) lets the band widen where the model is unsure and tighten where it is
confident -- same marginal coverage, but adaptive width. This contrasts plain vs
ensemble-std-normalized vs MC-dropout-std-normalized bands, calibrating on val.

Reads the per-split prediction dumps (``runs/predictions``); CPU only.

  python scripts/experiment_normalized.py --cal val --eval test --alpha 0.1 0.2
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

# (label, prediction key, scale key or None for plain)
VARIANTS = (
    ("ens_plain", "ens_mean", None),
    ("ens_norm", "ens_mean", "ens_std"),
    ("mc_plain", "mc_mean", None),
    ("mc_norm", "mc_mean", "mc_std"),
)


def load(split: str) -> dict[str, np.ndarray]:
    path = PRED_DIR / f"{split}.npz"
    if not path.exists():
        raise SystemExit(f"missing {path}; run scripts/predict.py (SLURM) first.")
    return dict(np.load(path, allow_pickle=True))


def positive_scale(values: np.ndarray) -> np.ndarray:
    """Guard a std column against zeros so normalized scores stay finite."""
    v = np.asarray(values, dtype=float)
    floor = max(1e-6, 1e-3 * float(np.median(v[v > 0])) if np.any(v > 0) else 1e-6)
    return np.maximum(v, floor)


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
    header = (
        f"{'variant':10} {'eval':6} {'alpha':>5} {'target':>7} "
        f"{'coverage':>9} {'mean_w':>7} {'med_w':>7}"
    )
    print(f"calibration split: {args.cal} (n={cal['y'].size})")
    print(header)
    print("-" * len(header))
    for label, pred_key, scale_key in VARIANTS:
        for alpha in args.alpha:
            scale_cal = positive_scale(cal[scale_key]) if scale_key else None
            sc = SplitConformal.calibrate(cal["y"], cal[pred_key], alpha=alpha, scale_cal=scale_cal)
            for name, ev in eval_sets.items():
                scale_ev = positive_scale(ev[scale_key]) if scale_key else None
                lo, hi = sc.intervals(ev[pred_key], scale=scale_ev)
                rep = coverage_report(ev["y"], lo, hi, alpha)
                print(
                    f"{label:10} {name:6} {alpha:5.2f} {rep.target_coverage:7.2f} "
                    f"{rep.empirical_coverage:9.4f} {rep.mean_width:7.3f} {rep.median_width:7.3f}"
                )
                rows.append(
                    {"variant": label, "cal": args.cal, "eval": name, **rep.as_dict()}
                )

    out = RESULTS_DIR / "normalized_conformal.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
