"""Loader for LP-PDBBind splits with peptide-overlap removal.

Reproduces the filter from Rayka & Naghavi 2025: train on CL1 non-covalent, validate
and test on CL2 non-covalent, then remove any PDB IDs that appear in the
Peptides-Holdout benchmark. Expected counts after filtering are the ones committed
in the approved proposal.

LP-PDBBind table columns (as of commit cloned):
    pdbid (first unnamed column), header, smiles, category, seq, resolution, date,
    type, new_split (train/val/test/''), CL1, CL2, CL3 (bool), covalent (bool), value.

The Peptides-Holdout CSV from Zenodo has no covalent column, so the covalent flag
for peptide complexes is looked up here against the LP-PDBBind table (all 2,573
peptide PDB IDs appear in LP-PDBBind).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cap6545.data.peptides import peptide_pdbids
from cap6545.utils.paths import EXTERNAL_DIR

LP_PDBBIND_TABLE_CANDIDATES = (
    EXTERNAL_DIR / "LP-PDBBind" / "dataset" / "LP_PDBBind.csv",
    EXTERNAL_DIR / "LP-PDBBind" / "LP_PDBBind.csv",
)

EXPECTED_SPLITS = {
    "train": (7259, 0),
    "val": (947, 3),
    "test": (2166, 3),
}
EXPECTED_PEPTIDES_NC = (2498, 5)


@dataclass(frozen=True)
class SplitCounts:
    train: int
    val: int
    test: int
    peptides_nc: int

    def check(self) -> list[str]:
        errors: list[str] = []
        pairs: list[tuple[str, int, tuple[int, int]]] = [
            ("train", self.train, EXPECTED_SPLITS["train"]),
            ("val", self.val, EXPECTED_SPLITS["val"]),
            ("test", self.test, EXPECTED_SPLITS["test"]),
            ("peptides_nc", self.peptides_nc, EXPECTED_PEPTIDES_NC),
        ]
        for name, count, (expected, tol) in pairs:
            if abs(count - expected) > tol:
                errors.append(
                    f"{name}: got {count}, expected {expected} (tolerance {tol})"
                )
        return errors


def expected_counts() -> dict[str, tuple[int, int]]:
    return {**EXPECTED_SPLITS, "peptides_nc": EXPECTED_PEPTIDES_NC}


def _find_lp_pdbbind_table() -> Path:
    for candidate in LP_PDBBIND_TABLE_CANDIDATES:
        if candidate.exists():
            return candidate
    searched = "\n  ".join(str(c) for c in LP_PDBBIND_TABLE_CANDIDATES)
    raise FileNotFoundError(
        "LP-PDBBind table not found. Clone the repo into external/LP-PDBBind. "
        f"Searched:\n  {searched}"
    )


def _load_lp_pdbbind_table(path: Path | None = None) -> pd.DataFrame:
    path = path or _find_lp_pdbbind_table()
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str).str.lower().str.strip()
    df.index.name = "pdbid"
    for col in ("CL1", "CL2", "CL3", "covalent"):
        df[col] = df[col].astype(str).str.lower().isin({"true", "1", "t", "yes"})
    return df


def load_lp_pdbbind_splits(
    table_path: Path | str | None = None,
    *,
    peptide_overlap_removal: bool = True,
) -> tuple[dict[str, pd.DataFrame], SplitCounts]:
    """Return train/val/test/peptides_nc dataframes and their row counts.

    Train uses CL1 non-covalent; val and test use CL2 non-covalent. When
    ``peptide_overlap_removal`` is true, PDB IDs in the Peptides-Holdout benchmark
    (regardless of covalent status) are dropped from train/val/test.

    The ``peptides_nc`` split is the Peptides-Holdout rows whose LP-PDBBind
    ``covalent`` flag is false.
    """
    path = Path(table_path) if table_path is not None else _find_lp_pdbbind_table()
    df = _load_lp_pdbbind_table(path)

    non_covalent = df.loc[~df["covalent"]]
    train = non_covalent.loc[(non_covalent["new_split"] == "train") & non_covalent["CL1"]].copy()
    val = non_covalent.loc[(non_covalent["new_split"] == "val") & non_covalent["CL2"]].copy()
    test = non_covalent.loc[(non_covalent["new_split"] == "test") & non_covalent["CL2"]].copy()

    peptide_ids = peptide_pdbids()
    peptide_covalent = df.loc[df.index.isin(peptide_ids), "covalent"]
    peptides_nc_ids = set(peptide_covalent[~peptide_covalent].index)
    peptides_nc = df.loc[df.index.isin(peptides_nc_ids)].copy()

    if peptide_overlap_removal:
        train = train.loc[~train.index.isin(peptide_ids)]
        val = val.loc[~val.index.isin(peptide_ids)]
        test = test.loc[~test.index.isin(peptide_ids)]

    splits = {"train": train, "val": val, "test": test, "peptides_nc": peptides_nc}
    counts = SplitCounts(
        train=len(train), val=len(val), test=len(test), peptides_nc=len(peptides_nc)
    )
    return splits, counts
