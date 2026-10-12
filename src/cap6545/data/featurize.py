"""Cache ECIF features to disk, keyed by complex ID, so training never re-featurizes.

Each complex's full 1540-dim ECIF vector is stored as ``<cache_dir>/<pdbid>.npy``.
Featurizing a split returns an aligned ``(n, 1540)`` matrix and the list of IDs.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cap6545.data.ecif import ECIFFeaturizer, Featurizer
from cap6545.utils.paths import PROCESSED_DIR

DEFAULT_CACHE_DIR = PROCESSED_DIR / "ecif"


@dataclass(frozen=True)
class ComplexRecord:
    """Locate one complex's structure files."""

    pdbid: str
    protein_pdb: Path
    ligand_sdf: Path


def _cache_path(cache_dir: Path, pdbid: str) -> Path:
    return cache_dir / f"{pdbid}.npy"


def featurize_complex(
    record: ComplexRecord,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    featurizer: Featurizer | None = None,
    overwrite: bool = False,
) -> np.ndarray:
    """Return one complex's ECIF vector, reading from or writing to the cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, record.pdbid)
    if path.exists() and not overwrite:
        return np.load(path)

    featurizer = featurizer or ECIFFeaturizer()
    vec = featurizer.featurize(record.protein_pdb, record.ligand_sdf)
    np.save(path, vec)
    return vec


def featurize_split(
    records: Iterable[ComplexRecord],
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    featurizer: Featurizer | None = None,
    overwrite: bool = False,
) -> tuple[list[str], np.ndarray]:
    """Featurize many complexes, returning ``(ids, matrix)`` aligned by row.

    Complexes already cached are loaded, not recomputed (unless ``overwrite``).
    """
    featurizer = featurizer or ECIFFeaturizer()
    ids: list[str] = []
    rows: list[np.ndarray] = []
    for record in records:
        rows.append(
            featurize_complex(
                record, cache_dir=cache_dir, featurizer=featurizer, overwrite=overwrite
            )
        )
        ids.append(record.pdbid)
    matrix = np.vstack(rows) if rows else np.empty((0, len(featurizer.feature_names)))
    return ids, matrix
