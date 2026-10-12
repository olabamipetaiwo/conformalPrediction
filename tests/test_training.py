"""Tests for the FFNN model and training loop.

The synthetic training test is fast and data-free. The end-to-end test featurizes
real BDB2020+ structures and skips when they are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from cap6545.models.ffnn import FFNN, mc_dropout_predict
from cap6545.training.config import ModelConfig, OptimConfig, TrainConfig
from cap6545.training.train import save_checkpoint, train_model

_BDB_DIR = Path("data/raw/pdbbind/BDB2020+")


def test_ffnn_forward_shape_and_default_architecture() -> None:
    model = FFNN()
    x = torch.randn(5, 790)
    out = model(x)
    assert out.shape == (5,)
    linears = [m for m in model.net if isinstance(m, torch.nn.Linear)]
    assert [lin.out_features for lin in linears] == [900, 300, 1]


def test_mc_dropout_predict_is_stochastic() -> None:
    torch.manual_seed(0)
    model = FFNN(in_features=16, hidden_units=(32,), dropout=0.5)
    x = torch.randn(8, 16)
    mean, std = mc_dropout_predict(model, x, samples=20)
    assert mean.shape == (8,)
    assert std.shape == (8,)
    assert (std > 0).all()  # dropout active -> nonzero predictive spread


def _synthetic(n: int, d: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    # Fixed weights so train/val share one linear relationship the model can
    # generalize to; only the samples and noise vary by seed.
    w = np.random.default_rng(0).normal(size=d)
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float64)
    y = x @ w + 0.01 * rng.normal(size=n)
    return x, y


def test_train_model_reduces_loss_on_synthetic_data() -> None:
    x_train, y_train = _synthetic(256, 8, seed=1)
    x_val, y_val = _synthetic(64, 8, seed=2)
    config = TrainConfig(
        seed=0,
        model=ModelConfig(in_features=8, hidden_units=(32, 16), dropout=0.0),
        optim=OptimConfig(optimizer="adam", lr=1e-2, weight_decay=0.0, batch_size=32, epochs=60),
    )
    result = train_model(config, x_train, y_train, x_val, y_val)
    first = result.history[0]["train_mse"]
    last = result.history[-1]["train_mse"]
    assert last < first * 0.5  # loss should drop substantially
    assert result.history[-1]["val_rmse"] < result.history[0]["val_rmse"]


def test_train_model_reproducible_with_seed() -> None:
    x_train, y_train = _synthetic(128, 8, seed=3)
    config = TrainConfig(
        seed=7,
        model=ModelConfig(in_features=8, hidden_units=(16,), dropout=0.0),
        optim=OptimConfig(optimizer="adam", lr=1e-2, weight_decay=0.0, batch_size=16, epochs=10),
    )
    a = train_model(config, x_train, y_train)
    b = train_model(config, x_train, y_train)
    assert a.history[-1]["train_mse"] == pytest.approx(b.history[-1]["train_mse"])


def test_config_from_yaml_roundtrip(tmp_path: Path) -> None:
    import yaml

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({"seed": 42, "optim": {"epochs": 5}}))
    config = TrainConfig.from_yaml(cfg_path)
    assert config.seed == 42
    assert config.optim.epochs == 5
    assert config.standardization == "fit"  # default preserved


@pytest.mark.skipif(
    not (_BDB_DIR / "BDB2020+.csv").exists(),
    reason="BDB2020+ structures not downloaded",
)
def test_end_to_end_train_on_bdb(tmp_path: Path) -> None:
    import pandas as pd

    from cap6545.data.features import select_features
    from cap6545.data.featurize import ComplexRecord, featurize_split

    meta = pd.read_csv(_BDB_DIR / "BDB2020+.csv").head(24)
    ds = _BDB_DIR / "dataset"
    records = [
        ComplexRecord(r.pdbid, ds / r.pdbid / "protein.pdb", ds / r.pdbid / "ligand.sdf")
        for r in meta.itertuples()
    ]
    _, full = featurize_split(records, cache_dir=tmp_path / "ecif")
    x = select_features(full).astype(np.float64)
    y = meta["pKa"].to_numpy(dtype=np.float64)

    config = TrainConfig(seed=0, optim=OptimConfig(epochs=3))
    result = train_model(config, x, y)
    assert len(result.history) == 3
    assert np.isfinite(result.history[-1]["train_mse"])

    save_checkpoint(result, tmp_path / "run", config)
    assert (tmp_path / "run" / "checkpoint.pt").exists()
    assert (tmp_path / "run" / "history.json").exists()


@pytest.mark.skipif(
    not (_BDB_DIR / "BDB2020+.csv").exists(),
    reason="BDB2020+ structures not downloaded",
)
def test_build_split_arrays_aligns_targets(tmp_path: Path) -> None:
    import pandas as pd

    from cap6545.data.structures import index_structures
    from cap6545.training.dataset import build_split_arrays

    meta = pd.read_csv(_BDB_DIR / "BDB2020+.csv").head(12)
    split_df = meta.set_index("pdbid")
    index = index_structures(_BDB_DIR / "dataset")

    arrays = build_split_arrays(
        split_df, index, target_col="pKa", cache_dir=tmp_path / "ecif"
    )
    assert arrays.x.shape == (len(arrays.ids), 790)
    assert arrays.y.shape == (len(arrays.ids),)
    assert arrays.missing == []
    # Targets must stay aligned to the featurized id order.
    lower = split_df.rename(index=str.lower)
    expected = np.array([lower.loc[i, "pKa"] for i in arrays.ids], dtype=np.float64)
    assert np.allclose(arrays.y, expected)
