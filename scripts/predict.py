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
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cap6545.data.bdb import load_bdb2020plus
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


def resolve_splits(names: list[str], holdout_ids: Path | None = None) -> dict[str, pd.DataFrame]:
    """Split frames for the requested names, drawing from the right data source.

    ``train``/``val``/``test``/``peptides_nc`` come from the LP-PDBBind loader;
    ``bdb`` from the BDB2020+ shift set; ``train_holdout`` is the LP train frame
    restricted to the Q1 calibration ids in ``holdout_ids``.
    """
    out: dict[str, pd.DataFrame] = {}
    known_lp = {"train", "val", "test", "peptides_nc"}
    need_lp = {n for n in names if n in known_lp} | ({"train"} if "train_holdout" in names else set())
    lp: dict[str, pd.DataFrame] = {}
    if need_lp:
        lp, _ = load_lp_pdbbind_splits()
    for n in names:
        if n == "bdb":
            out[n] = load_bdb2020plus()
        elif n == "train_holdout":
            if holdout_ids is None:
                raise SystemExit("train_holdout requested but --holdout-ids not given")
            ids = {i.lower() for i in json.loads(Path(holdout_ids).read_text())}
            out[n] = lp["train"].loc[lp["train"].index.isin(ids)]
        elif n in lp:
            out[n] = lp[n]
        else:
            raise SystemExit(f"unknown split {n!r}; known: {sorted(known_lp)} + ['bdb','train_holdout']")
    return out


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
    parser.add_argument("--ckpt-glob", default=CKPT_GLOB)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--holdout-ids", type=Path, default=None,
                        help="JSON of Q1 calibration ids, for the train_holdout split")
    args = parser.parse_args()

    device = default_device()
    ckpts = sorted(Path().glob(args.ckpt_glob))
    if not ckpts:
        raise SystemExit(f"no checkpoints under {args.ckpt_glob}; train first.")
    print(f"device={device.type} | {len(ckpts)} checkpoints")
    models = [load_model(c, device) for c in ckpts]

    splits = resolve_splits(args.splits, holdout_ids=args.holdout_ids)
    index = index_structures(PDBBIND_ROOT)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

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

        if arrays.missing:
            print(f"  [{name}] {len(arrays.missing)} ids lack structures; "
                  f"predicting on {len(arrays.ids)}.")
        out = out_dir / f"{name}.npz"
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
