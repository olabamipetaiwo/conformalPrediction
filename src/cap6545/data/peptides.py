"""Loader for the Peptides-Holdout benchmark (Zenodo 8410136, ToolBoxSF study).

The CSV ships inside toolboxsf_benchmarks.zip at
toolboxsf_benchmarks/csv_files/peptides_holdout_test.csv and is extracted to
data/raw/ by scripts/download_peptides.py. Columns in the Zenodo CSV:
    key (pdbid), pk (-logKd/Ki), protein (path), ligand (path).

The covalent flag is NOT in this CSV; it lives in the LP-PDBBind table and is
applied by cap6545.data.splits when the two are joined (all 2,573 peptide PDB
IDs appear in the LP-PDBBind table).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cap6545.utils.paths import RAW_DIR

DEFAULT_PATH = RAW_DIR / "peptides_holdout_test.csv"
RAW_ROW_COUNT = 2573


def load_peptides_holdout(path: Path | str = DEFAULT_PATH) -> pd.DataFrame:
    """Load the raw Peptides-Holdout CSV, indexed by lowercase pdbid."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `python scripts/download_peptides.py` first."
        )

    df = pd.read_csv(path).rename(columns={"key": "pdbid"})
    df["pdbid"] = df["pdbid"].astype(str).str.lower().str.strip()

    if len(df) != RAW_ROW_COUNT:
        raise AssertionError(
            f"peptides CSV has {len(df)} rows; proposal commits to {RAW_ROW_COUNT}."
        )
    return df.set_index("pdbid")


def peptide_pdbids(path: Path | str = DEFAULT_PATH) -> set[str]:
    """Return the set of PDB IDs in the peptides CSV (lowercase, raw: no filtering)."""
    return set(load_peptides_holdout(path).index)
