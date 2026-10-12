"""ECIF (Extended Connectivity Interaction Features) for protein-ligand complexes.

Faithful reimplementation of Sanchez-Cruz et al. 2021 (DIFACQUIM/ECIF, MIT),
the featurization used by Rayka & Naghavi 2025 (via their REINDEER tool) for the
FFNN base model this project reproduces.

An ECIF atom type is a 6-field string:

    Element;ExplicitValence;#HeavyNeighbors;#H;Aromatic;InRing   e.g. ``C;4;1;3;0;0``

ECIF counts, for every (protein atom type, ligand atom type) pair, how many such
pairs fall within ``distance_cutoff`` (6.0 A). With 22 predefined protein types
and 70 ligand types that is a fixed 1540-dim count vector, ordered as
``itertools.product(protein_types, ligand_types)`` — identical to the column
order of the authors' ``data_instances_ecif.csv``.

Feature selection (to the 790 columns the published model uses) and
standardization live in :mod:`cap6545.data.features`, not here: this module only
produces the full, canonical ECIF vector.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.spatial.distance import cdist

_ATOM_KEYS_PATH = Path(__file__).with_name("PDB_Atom_Keys.csv")

# Predefined protein atom types (22) and ligand atom types (70) from the PDBbind
# 2016 refined set, verbatim from the reference implementation. Order is
# load-bearing: it fixes the ECIF column order.
ECIF_PROTEIN_ATOMS: tuple[str, ...] = (
    "C;4;1;3;0;0", "C;4;2;1;1;1", "C;4;2;2;0;0", "C;4;2;2;0;1",
    "C;4;3;0;0;0", "C;4;3;0;1;1", "C;4;3;1;0;0", "C;4;3;1;0;1",
    "C;5;3;0;0;0", "C;6;3;0;0;0", "N;3;1;2;0;0", "N;3;2;0;1;1",
    "N;3;2;1;0;0", "N;3;2;1;1;1", "N;3;3;0;0;1", "N;4;1;2;0;0",
    "N;4;1;3;0;0", "N;4;2;1;0;0", "O;2;1;0;0;0", "O;2;1;1;0;0",
    "S;2;1;1;0;0", "S;2;2;0;0;0",
)
ECIF_LIGAND_ATOMS: tuple[str, ...] = (
    "Br;1;1;0;0;0", "C;3;3;0;1;1", "C;4;1;1;0;0", "C;4;1;2;0;0",
    "C;4;1;3;0;0", "C;4;2;0;0;0", "C;4;2;1;0;0", "C;4;2;1;0;1",
    "C;4;2;1;1;1", "C;4;2;2;0;0", "C;4;2;2;0;1", "C;4;3;0;0;0",
    "C;4;3;0;0;1", "C;4;3;0;1;1", "C;4;3;1;0;0", "C;4;3;1;0;1",
    "C;4;4;0;0;0", "C;4;4;0;0;1", "C;5;3;0;0;0", "C;5;3;0;1;1",
    "C;6;3;0;0;0", "Cl;1;1;0;0;0", "F;1;1;0;0;0", "I;1;1;0;0;0",
    "N;3;1;0;0;0", "N;3;1;1;0;0", "N;3;1;2;0;0", "N;3;2;0;0;0",
    "N;3;2;0;0;1", "N;3;2;0;1;1", "N;3;2;1;0;0", "N;3;2;1;0;1",
    "N;3;2;1;1;1", "N;3;3;0;0;0", "N;3;3;0;0;1", "N;3;3;0;1;1",
    "N;4;1;2;0;0", "N;4;1;3;0;0", "N;4;2;1;0;0", "N;4;2;2;0;0",
    "N;4;2;2;0;1", "N;4;3;0;0;0", "N;4;3;0;0;1", "N;4;3;1;0;0",
    "N;4;3;1;0;1", "N;4;4;0;0;0", "N;4;4;0;0;1", "N;5;2;0;0;0",
    "N;5;3;0;0;0", "N;5;3;0;1;1", "O;2;1;0;0;0", "O;2;1;1;0;0",
    "O;2;2;0;0;0", "O;2;2;0;0;1", "O;2;2;0;1;1", "P;5;4;0;0;0",
    "P;6;4;0;0;0", "P;6;4;0;0;1", "P;7;4;0;0;0", "S;2;1;0;0;0",
    "S;2;1;1;0;0", "S;2;2;0;0;0", "S;2;2;0;0;1", "S;2;2;0;1;1",
    "S;3;3;0;0;0", "S;3;3;0;0;1", "S;4;3;0;0;0", "S;6;4;0;0;0",
    "S;6;4;0;0;1", "S;7;4;0;0;0",
)

#: The 1540 ECIF pair keys, ``"{protein}-{ligand}"``, in canonical column order.
ECIF_PAIR_NAMES: tuple[str, ...] = tuple(
    f"{protein}-{ligand}"
    for protein, ligand in product(ECIF_PROTEIN_ATOMS, ECIF_LIGAND_ATOMS)
)

DISTANCE_CUTOFF = 6.0


@lru_cache(maxsize=1)
def _atom_keys() -> dict[str, str]:
    """Map ``{resname}-{atomname}`` to an ECIF protein atom type."""
    df = pd.read_csv(_ATOM_KEYS_PATH)
    return dict(zip(df["PDB_ATOM"], df["ECIF_ATOM_TYPE"], strict=True))


def get_atom_type(atom: Chem.Atom) -> str:
    """Return the 6-field ECIF atom type string for an RDKit atom."""
    neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
    return ";".join(
        (
            atom.GetSymbol(),
            str(atom.GetValence(Chem.ValenceType.EXPLICIT)),
            str(sum(s != "H" for s in neighbors)),
            str(sum(s == "H" for s in neighbors)),
            str(int(atom.GetIsAromatic())),
            str(int(atom.IsInRing())),
        )
    )


def load_ligand_atoms(sdf_path: Path | str) -> pd.DataFrame:
    """Load a ligand SDF as a frame of non-H atoms with ECIF types and coords."""
    mol = Chem.MolFromMolFile(str(sdf_path), sanitize=False)
    if mol is None:
        raise ValueError(f"RDKit could not parse ligand SDF: {sdf_path}")
    mol.UpdatePropertyCache(strict=False)

    conf = mol.GetConformer()
    rows = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "H":
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        rows.append((atom.GetIdx(), get_atom_type(atom), pos.x, pos.y, pos.z))
    df = pd.DataFrame(rows, columns=["ATOM_INDEX", "ECIF_ATOM_TYPE", "X", "Y", "Z"])
    return df


def load_protein_atoms(pdb_path: Path | str) -> pd.DataFrame:
    """Load a protein PDB as a frame of non-H atoms with ECIF types and coords.

    ECIF types are assigned from the ``{resname}-{atomname}`` lookup table rather
    than from RDKit perception, which is unreliable on PDB-derived proteins.
    """
    keys = _atom_keys()
    rows = []
    with open(pdb_path) as fh:
        for line in fh:
            if line[:4] != "ATOM":
                continue
            name = line[12:16].replace(" ", "")
            is_h = (len(name) < 4 and name[0] == "H") or (
                len(name) == 4 and name[0] == "H" and name[1] == "H"
            )
            if is_h:
                continue
            pdb_atom = f"{line[17:20]}-{name}"
            ecif_type = keys.get(pdb_atom)
            if ecif_type is None:
                continue
            rows.append(
                (
                    int(line[6:11]),
                    ecif_type,
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
            )
    df = pd.DataFrame(rows, columns=["ATOM_INDEX", "ECIF_ATOM_TYPE", "X", "Y", "Z"])
    return df.sort_values("ATOM_INDEX").reset_index(drop=True)


def compute_ecif(
    protein_pdb: Path | str,
    ligand_sdf: Path | str,
    distance_cutoff: float = DISTANCE_CUTOFF,
) -> np.ndarray:
    """Return the 1540-dim ECIF count vector for one complex.

    Counts protein-ligand atom-type pairs within ``distance_cutoff`` angstroms.
    Pairs whose type is outside the predefined ECIF vocabulary are ignored, as in
    the reference implementation.
    """
    protein = load_protein_atoms(protein_pdb)
    ligand = load_ligand_atoms(ligand_sdf)

    # Prefilter protein atoms to a cubic box around the ligand (matches reference).
    for axis in ("X", "Y", "Z"):
        lo = ligand[axis].min() - distance_cutoff
        hi = ligand[axis].max() + distance_cutoff
        protein = protein[(protein[axis] > lo) & (protein[axis] < hi)]

    index = {name: i for i, name in enumerate(ECIF_PAIR_NAMES)}
    counts = np.zeros(len(ECIF_PAIR_NAMES), dtype=np.int64)
    if protein.empty or ligand.empty:
        return counts

    distances = cdist(
        protein[["X", "Y", "Z"]].to_numpy(),
        ligand[["X", "Y", "Z"]].to_numpy(),
        metric="euclidean",
    )
    p_types = protein["ECIF_ATOM_TYPE"].to_numpy()
    l_types = ligand["ECIF_ATOM_TYPE"].to_numpy()
    p_idx, l_idx = np.where(distances <= distance_cutoff)
    for pi, li in zip(p_types[p_idx], l_types[l_idx], strict=True):
        col = index.get(f"{pi}-{li}")
        if col is not None:
            counts[col] += 1
    return counts


@runtime_checkable
class Featurizer(Protocol):
    """Minimal structure-featurizer interface.

    Keeping featurization behind this protocol lets a second base model's
    featurizer (e.g. OnionNet-2, Pafnucy) slot in without touching training code.
    """

    @property
    def feature_names(self) -> tuple[str, ...]: ...

    def featurize(self, protein_pdb: Path | str, ligand_sdf: Path | str) -> np.ndarray: ...


class ECIFFeaturizer:
    """ECIF featurizer: produces the full 1540-dim count vector per complex."""

    def __init__(self, distance_cutoff: float = DISTANCE_CUTOFF) -> None:
        self.distance_cutoff = distance_cutoff

    @property
    def feature_names(self) -> tuple[str, ...]:
        return ECIF_PAIR_NAMES

    def featurize(self, protein_pdb: Path | str, ligand_sdf: Path | str) -> np.ndarray:
        return compute_ecif(protein_pdb, ligand_sdf, self.distance_cutoff)
