"""Train the deterministic FFNN baseline for one seed.

A SLURM array job varies only ``--seed`` and ``--config``; all run parameters
live in the YAML config. Trains on the LP-PDBBind train split, validates on the
val split, and writes a checkpoint + history under ``<output_dir>/seed_<seed>/``.

  python scripts/train.py --config configs/deterministic.yaml --seed 0
"""

from __future__ import annotations

import argparse

from cap6545.data.splits import load_lp_pdbbind_splits
from cap6545.data.structures import index_structures
from cap6545.training.config import TrainConfig
from cap6545.training.dataset import build_split_arrays
from cap6545.training.train import default_device, save_checkpoint, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = TrainConfig.from_yaml(args.config)
    if args.seed is not None:
        config = config.model_copy(update={"seed": args.seed})

    splits, _ = load_lp_pdbbind_splits()
    index = index_structures(config.pdbbind_root)

    train = build_split_arrays(splits["train"], index, cache_dir=config.features_cache)
    val = build_split_arrays(splits["val"], index, cache_dir=config.features_cache)
    if not train.ids:
        raise SystemExit(
            f"no LP-PDBBind train structures under {config.pdbbind_root}; "
            "run scripts/download_pdbbind.py first."
        )
    for name, arrays in (("train", train), ("val", val)):
        if arrays.missing:
            print(f"WARNING: {name} missing {len(arrays.missing)} structures "
                  f"(e.g. {arrays.missing[:5]}); training on {len(arrays.ids)}.")

    device = default_device()
    print(f"device={device.type} | train={len(train.ids)} val={len(val.ids)} "
          f"| standardization={config.standardization}")

    result = train_model(config, train.x, train.y, val.x, val.y, device=device)
    save_checkpoint(result, config.run_dir(), config)

    last = result.history[-1]
    print(f"done seed={config.seed}: train_mse={last['train_mse']:.4f} "
          f"val_rmse={last.get('val_rmse', float('nan')):.4f} -> {config.run_dir()}")


if __name__ == "__main__":
    main()
