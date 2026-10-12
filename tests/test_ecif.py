"""Tests for the ECIF featurizer, feature selection, and standardization.

Pure-logic tests run anywhere. The golden-vector test against the reference
example structures is an integration test and skips when they are absent.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from cap6545.data.ecif import (
    ECIF_LIGAND_ATOMS,
    ECIF_PAIR_NAMES,
    ECIF_PROTEIN_ATOMS,
    ECIFFeaturizer,
    compute_ecif,
    get_atom_type,
)
from cap6545.data.features import (
    Standardizer,
    published_standardizer,
    select_features,
    selected_feature_names,
)

_EXAMPLE_DIR = Path("external/ECIF/Example_Structures")


def test_pair_vocabulary_is_1540_in_product_order() -> None:
    assert len(ECIF_PROTEIN_ATOMS) == 22
    assert len(ECIF_LIGAND_ATOMS) == 70
    assert len(ECIF_PAIR_NAMES) == 1540
    expected = tuple(
        f"{protein}-{ligand}"
        for protein, ligand in product(ECIF_PROTEIN_ATOMS, ECIF_LIGAND_ATOMS)
    )
    assert expected == ECIF_PAIR_NAMES
    assert len(set(ECIF_PAIR_NAMES)) == 1540


def test_get_atom_type_methyl_carbon() -> None:
    mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
    mol.UpdatePropertyCache(strict=False)
    carbon = next(a for a in mol.GetAtoms() if a.GetSymbol() == "C")
    # C, explicit valence 4, 1 heavy neighbor, 3 H, not aromatic, not in ring
    assert get_atom_type(carbon) == "C;4;1;3;0;0"


def test_get_atom_type_aromatic_ring_flags() -> None:
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    mol.UpdatePropertyCache(strict=False)
    carbon = next(a for a in mol.GetAtoms() if a.GetSymbol() == "C")
    fields = get_atom_type(carbon).split(";")
    assert fields[0] == "C"
    assert fields[4] == "1"  # aromatic
    assert fields[5] == "1"  # in ring


def test_selected_features_are_790_valid_subset() -> None:
    names = selected_feature_names()
    assert len(names) == 790
    assert set(names).issubset(set(ECIF_PAIR_NAMES))


def test_select_features_vector_and_batch() -> None:
    full = np.arange(len(ECIF_PAIR_NAMES), dtype=np.float64)
    sel_vec = select_features(full)
    assert sel_vec.shape == (790,)

    batch = np.stack([full, full + 1])
    sel_batch = select_features(batch)
    assert sel_batch.shape == (2, 790)
    # Row-wise selection must equal the single-vector path.
    assert np.array_equal(sel_batch[0], sel_vec)


def test_standardizer_fit_transform() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 790))
    scaler = Standardizer.fit(x)
    z = scaler.transform(x)
    assert z.shape == x.shape
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(z.std(axis=0), 1.0, atol=1e-6)


def test_standardizer_constant_column_is_zero_not_nan() -> None:
    x = np.ones((10, 790))
    scaler = Standardizer.fit(x)
    z = scaler.transform(x)
    assert not np.isnan(z).any()
    assert np.allclose(z, 0.0)


def test_published_standardizer_shapes() -> None:
    scaler = published_standardizer()
    assert scaler.mean.shape == (790,)
    assert scaler.std.shape == (790,)
    assert not np.any(scaler.std == 0)


@pytest.mark.skipif(
    not (_EXAMPLE_DIR / "1a0q_protein.pdb").exists(),
    reason="reference ECIF example structures not available",
)
def test_compute_ecif_matches_reference_golden() -> None:
    # Golden values verified bit-for-bit against DIFACQUIM/ECIF GetECIF (6.0 A).
    vec = compute_ecif(
        _EXAMPLE_DIR / "1a0q_protein.pdb",
        _EXAMPLE_DIR / "1a0q_ligandCD1.sdf",
    )
    assert vec.shape == (1540,)
    assert vec.dtype == np.int64
    assert int(vec.sum()) == 547
    assert int((vec > 0).sum()) == 137


@pytest.mark.skipif(
    not (_EXAMPLE_DIR / "1a0q_protein.pdb").exists(),
    reason="reference ECIF example structures not available",
)
def test_featurizer_interface() -> None:
    feat = ECIFFeaturizer()
    assert feat.feature_names == ECIF_PAIR_NAMES
    vec = feat.featurize(
        _EXAMPLE_DIR / "1a0q_protein.pdb",
        _EXAMPLE_DIR / "1a0q_ligandCD1.sdf",
    )
    assert vec.shape == (1540,)
