"""Locate prepared structure files for complexes across dataset layouts.

Different sources lay structures out differently:

* PDBbind v2020 extracts to ``<set>/<pdbid>/<pdbid>_protein.pdb`` (+ ``_ligand.sdf``),
  nested under ``refined-set/`` and ``v2020-other-PL/`` subdirectories.
* BDB2020+ uses ``dataset/<pdbid>/protein.pdb`` (+ ``ligand.sdf``).

:func:`index_structures` scans a root once and maps each lowercase pdbid to its
protein/ligand paths, so lookups during featurization are O(1) regardless of how
deep the files are nested.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cap6545.data.featurize import ComplexRecord


@dataclass(frozen=True)
class StructurePaths:
    protein_pdb: Path
    ligand_sdf: Path


def index_structures(root: Path | str) -> dict[str, StructurePaths]:
    """Map pdbid -> structure paths by scanning ``root`` for protein PDB files.

    Recognizes both ``<pdbid>_protein.pdb`` (PDBbind) and ``protein.pdb`` under a
    ``<pdbid>`` directory (BDB2020+). The ligand is the sibling ``*_ligand.sdf``
    or ``ligand.sdf``.
    """
    root = Path(root)
    index: dict[str, StructurePaths] = {}

    for protein in root.rglob("*protein.pdb"):
        name = protein.name
        pdbid = (
            name[: -len("_protein.pdb")]
            if name.endswith("_protein.pdb")
            else protein.parent.name
        ).lower()
        ligand = _sibling_ligand(protein)
        if ligand is not None:
            index.setdefault(pdbid, StructurePaths(protein, ligand))
    return index


def _sibling_ligand(protein: Path) -> Path | None:
    parent = protein.parent
    candidates = sorted(parent.glob("*ligand.sdf"))
    return candidates[0] if candidates else None


def records_for_ids(
    pdbids: list[str], index: dict[str, StructurePaths]
) -> tuple[list[ComplexRecord], list[str]]:
    """Build featurization records for the ids present in ``index``.

    Returns ``(records, missing_ids)`` so callers can report coverage gaps.
    """
    records: list[ComplexRecord] = []
    missing: list[str] = []
    for pdbid in pdbids:
        paths = index.get(pdbid.lower())
        if paths is None:
            missing.append(pdbid)
            continue
        records.append(ComplexRecord(pdbid.lower(), paths.protein_pdb, paths.ligand_sdf))
    return records, missing
