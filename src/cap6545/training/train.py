"""Training loop for the FFNN baseline.

The core, :func:`train_model`, takes selected-feature arrays so it is easy to
test and reuse; standardization is applied here according to the run config
(fit on the train split, or the authors' published statistics). The CLI assembles
arrays from a split and a cached-feature store, then calls it.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cap6545.data.features import Standardizer, published_standardizer
from cap6545.models.ffnn import FFNN
from cap6545.training.config import TrainConfig

_OPTIMIZERS = {
    "rmsprop": torch.optim.RMSprop,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}


def default_device() -> torch.device:
    """Use the GPU when one is visible (SLURM jobs), else CPU (e.g. login node)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainResult:
    model: FFNN
    standardizer: Standardizer
    history: list[dict[str, float]] = field(default_factory=list)


def _build_standardizer(mode: str, x_train: np.ndarray) -> Standardizer:
    if mode == "published":
        return published_standardizer()
    if mode == "fit":
        return Standardizer.fit(x_train)
    raise ValueError(f"unknown standardization mode {mode!r}")


def _rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def train_model(
    config: TrainConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    *,
    device: str | torch.device = "cpu",
) -> TrainResult:
    """Train an FFNN on selected-feature arrays and return model + scaler + history."""
    set_seed(config.seed)
    device = torch.device(device)

    scaler = _build_standardizer(config.standardization, x_train)
    xt = torch.tensor(scaler.transform(x_train), dtype=torch.float32)
    yt = torch.tensor(np.asarray(y_train), dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(xt, yt),
        batch_size=config.optim.batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = FFNN(
        in_features=config.model.in_features,
        hidden_units=config.model.hidden_units,
        dropout=config.model.dropout,
        activation=config.model.activation,
    ).to(device)
    optim_cls = _OPTIMIZERS[config.optim.optimizer]
    optimizer = optim_cls(
        model.parameters(), lr=config.optim.lr, weight_decay=config.optim.weight_decay
    )
    loss_fn = nn.MSELoss()

    has_val = x_val is not None and y_val is not None
    if has_val:
        xv = torch.from_numpy(scaler.transform(x_val)).float().to(device)
        yv = np.asarray(y_val)

    history: list[dict[str, float]] = []
    for epoch in range(config.optim.epochs):
        model.train()
        epoch_loss = 0.0
        n = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
            n += len(xb)
        record = {"epoch": epoch, "train_mse": epoch_loss / max(n, 1)}
        if has_val:
            model.eval()
            with torch.no_grad():
                record["val_rmse"] = _rmse(model(xv).cpu().numpy(), yv)
        history.append(record)

    return TrainResult(model=model, standardizer=scaler, history=history)


def save_checkpoint(result: TrainResult, run_dir: Path, config: TrainConfig) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": result.model.state_dict(),
            "standardizer_mean": result.standardizer.mean,
            "standardizer_std": result.standardizer.std,
            "config": config.model_dump(mode="json"),
        },
        run_dir / "checkpoint.pt",
    )
    (run_dir / "history.json").write_text(json.dumps(result.history, indent=2))
    (run_dir / "config.json").write_text(json.dumps(config.model_dump(mode="json"), indent=2))
