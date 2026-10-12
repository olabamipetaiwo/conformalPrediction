"""Loader for the BDB2020+ shift set.

BDB2020+ ships as ``BDB2020+.csv`` (one row per complex) plus a ``dataset/<pdbid>/``
structure tree. The CSV's ``value`` column is the raw affinity; the model targets the
``-log`` affinity, so the ``pKa`` column is used and surfaced as ``value`` to match the
LP-PDBBind split frames consumed by :func:`build_split_arrays`.

Only the labeled rows (115) are returned, even though more structure directories exist
on disk; this is the documented 115-vs-111 discrepancy with the proposal.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BDB_CSV = Path("data/raw/pdbbind/BDB2020+/BDB2020+.csv")


def load_bdb2020plus(csv_path: Path | str = BDB_CSV) -> pd.DataFrame:
    """Return a BDB2020+ frame indexed by lowercase pdbid with a ``value`` (pKa) column."""
    df = pd.read_csv(csv_path)
    df["pdbid"] = df["pdbid"].astype(str).str.lower().str.strip()
    df = df.set_index("pdbid")
    df.index.name = "pdbid"
    df = df.loc[df["pKa"].notna()].copy()
    df["value"] = df["pKa"].astype(float)
    return df
