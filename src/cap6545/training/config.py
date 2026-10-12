"""Run configuration for training, loaded from YAML and overridable per run.

A SLURM array job varies only ``--seed`` and ``--config``; everything that
differs between runs lives here, and no paths are hardcoded in training code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    in_features: int = 790
    hidden_units: tuple[int, ...] = (900, 300)
    dropout: float = 0.4
    activation: str = "elu"


class OptimConfig(BaseModel):
    optimizer: Literal["rmsprop", "adam", "adamw"] = "rmsprop"
    lr: float = 6.894932301255905e-05
    weight_decay: float = 0.004423364659297819
    batch_size: int = 32
    epochs: int = 300


class TrainConfig(BaseModel):
    """Full per-run config. Paths are relative to the project root unless absolute."""

    seed: int = 0
    # "fit" standardizes on our own train split (no leakage); "published" uses the
    # authors' vendored mean/std for verbatim reproduction.
    standardization: Literal["fit", "published"] = "fit"
    pdbbind_root: Path = Path("data/raw/pdbbind")
    features_cache: Path = Path("data/processed/ecif")
    output_dir: Path = Path("runs")
    model: ModelConfig = Field(default_factory=ModelConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> TrainConfig:
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        return cls.model_validate(data)

    def run_dir(self) -> Path:
        return self.output_dir / f"seed_{self.seed}"
