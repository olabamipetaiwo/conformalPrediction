"""Assemble selected-feature matrices and targets for a split.

Bridges the splits loader, the structure index, and the cached featurizer:
given a split's metadata frame (indexed by pdbid, with a target column) and a
structure index, it featurizes the complexes that have structures, caches them,
and returns aligned ``(ids, X_selected, y)`` ready for :func:`train_model`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cap6545.data.features import select_features
from cap6545.data.featurize import DEFAULT_CACHE_DIR, featurize_split
from cap6545.data.structures import StructurePaths, records_for_ids


@dataclass
class SplitArrays:
    ids: list[str]
    x: np.ndarray  # (n, 790) selected features
    y: np.ndarray  # (n,) targets
    missing: list[str]  # ids with no structure available


def build_split_arrays(
    split_df: pd.DataFrame,
    index: dict[str, StructurePaths],
    *,
    target_col: str = "value",
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> SplitArrays:
    """Featurize a split and return aligned selected-feature arrays and targets."""
    records, missing = records_for_ids(list(split_df.index), index)
    ids, full = featurize_split(records, cache_dir=cache_dir)
    x = select_features(full).astype(np.float64)

    targets = split_df[target_col]
    targets.index = targets.index.astype(str).str.lower()
    y = targets.loc[ids].to_numpy(dtype=np.float64)
    return SplitArrays(ids=ids, x=x, y=y, missing=missing)
