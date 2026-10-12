"""Dump model predictions per split to disk, so conformal analysis needs no GPU.

Loads every deterministic/ensemble checkpoint under ``runs/deterministic/seed_*``
and writes, for each split, an ``.npz`` with the targets and three predictors:

  det        - single-model prediction (seed 0)
  ens_mean   - deep-ensemble mean over all seeds       (ens_std = disagreement)
  mc_mean    - MC-dropout mean over N passes (seed 0)   (mc_std  = dropout spread)

The ensemble/MC std columns are the uncertainty estimates the normalized
conformal variant consumes. Run as a (small) GPU SLURM job, not on a login node.

  python scripts/predict.py --splits val test --mc-samples 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from cap6545.data.features import Standardizer
from cap6545.data.splits import load_lp_pdbbind_splits
from cap6545.data.structures import index_structures
from cap6545.models.ffnn import FFNN, mc_dropout_predict
from cap6545.training.dataset import build_split_arrays
from cap6545.training.train import default_device

CKPT_GLOB = "runs/deterministic/seed_*/checkpoint.pt"
PDBBIND_ROOT = Path("data/raw/pdbbind")
CACHE = Path("data/processed/ecif")
OUT_DIR = Path("runs/predictions")


def load_model(ckpt_path: Path, device: torch.device) -> tuple[FFNN, Standardizer]:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = ck["config"]["model"]
    model = FFNN(
        in_features=m["in_features"],
        hidden_units=m["hidden_units"],
        dropout=m["dropout"],
        activation=m["activation"],
    ).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    scaler = Standardizer(
        mean=np.asarray(ck["standardizer_mean"]), std=np.asarray(ck["standardizer_std"])
    )
    return model, scaler


@torch.no_grad()
def predict(model: FFNN, x_std: np.ndarray, device: torch.device) -> np.ndarray:
    xt = torch.tensor(x_std, dtype=torch.float32, device=device)
    return model(xt).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--mc-samples", type=int, default=50)
    args = parser.parse_args()

    device = default_device()
    ckpts = sorted(Path().glob(CKPT_GLOB))
    if not ckpts:
        raise SystemExit(f"no checkpoints under {CKPT_GLOB}; train first.")
    print(f"device={device.type} | {len(ckpts)} checkpoints")
    models = [load_model(c, device) for c in ckpts]

    splits, _ = load_lp_pdbbind_splits()
    index = index_structures(PDBBIND_ROOT)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name in args.splits:
        arrays = build_split_arrays(splits[name], index, cache_dir=CACHE)
        x, y, ids = arrays.x, arrays.y, arrays.ids

        member_preds = np.stack(
            [predict(model, scaler.transform(x), device) for model, scaler in models]
        )
        ens_mean, ens_std = member_preds.mean(axis=0), member_preds.std(axis=0)

        model0, scaler0 = models[0]
        mc_mean, mc_std = mc_dropout_predict(
            model0,
            torch.tensor(scaler0.transform(x), dtype=torch.float32, device=device),
            samples=args.mc_samples,
        )

        out = OUT_DIR / f"{name}.npz"
        np.savez(
            out,
            ids=np.array(ids),
            y=y,
            det=member_preds[0],
            ens_mean=ens_mean,
            ens_std=ens_std,
            mc_mean=mc_mean.cpu().numpy(),
            mc_std=mc_std.cpu().numpy(),
        )
        rmse = float(np.sqrt(np.mean((ens_mean - y) ** 2)))
        print(f"[{name:5}] n={len(ids)} ensemble_rmse={rmse:.4f} -> {out}")


if __name__ == "__main__":
    main()
