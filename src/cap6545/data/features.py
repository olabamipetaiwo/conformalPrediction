"""Feature selection and standardization on top of the full ECIF vector.

The published FFNN uses a fixed 790-feature subset of the 1540-dim ECIF vector,
standardized to zero mean / unit variance. The subset and the authors' training
statistics are vendored in ``Remained_features_statistics.csv`` (row 0 = mean,
row 1 = std), so their exact preprocessing can be reproduced.

For training on our own splits, prefer :meth:`Standardizer.fit` on the training
split to avoid leaking statistics across splits; use :func:`published_standardizer`
only when reproducing the authors' model verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from cap6545.data.ecif import ECIF_PAIR_NAMES

_STATS_PATH = Path(__file__).with_name("Remained_features_statistics.csv")


@lru_cache(maxsize=1)
def _stats_frame() -> pd.DataFrame:
    return pd.read_csv(_STATS_PATH)


@lru_cache(maxsize=1)
def selected_feature_names() -> tuple[str, ...]:
    """The 790 ECIF features used by the published model, in their column order."""
    return tuple(_stats_frame().columns)


@lru_cache(maxsize=1)
def _selection_index() -> np.ndarray:
    pos = {name: i for i, name in enumerate(ECIF_PAIR_NAMES)}
    return np.array([pos[name] for name in selected_feature_names()], dtype=np.int64)


def select_features(ecif: np.ndarray) -> np.ndarray:
    """Reduce full 1540-dim ECIF vectors to the selected 790, preserving order.

    Accepts a single vector ``(1540,)`` or a batch ``(n, 1540)``.
    """
    idx = _selection_index()
    if ecif.ndim == 1:
        return ecif[idx]
    return ecif[:, idx]


@dataclass(frozen=True)
class Standardizer:
    """Zero-mean/unit-variance scaler over the selected feature columns.

    ``std`` values of zero are replaced by one so constant columns map to zero
    rather than NaN, matching the authors' ``fillna(0)`` behavior.
    """

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, selected: np.ndarray) -> Standardizer:
        mean = selected.mean(axis=0)
        std = selected.std(axis=0, ddof=0)
        return cls(mean=mean, std=std)

    def transform(self, selected: np.ndarray) -> np.ndarray:
        safe_std = np.where(self.std == 0, 1.0, self.std)
        out = (selected - self.mean) / safe_std
        return np.nan_to_num(out, nan=0.0)


def published_standardizer() -> Standardizer:
    """The authors' training-set mean/std for the 790 selected features."""
    stats = _stats_frame()
    return Standardizer(
        mean=stats.iloc[0].to_numpy(dtype=np.float64),
        std=stats.iloc[1].to_numpy(dtype=np.float64),
    )
