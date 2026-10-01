"""Smoke test: dataset counts match the approved proposal within tolerance.

Skipped when external data is missing so a fresh clone still passes `pytest`.
"""

from __future__ import annotations

import pytest

from cap6545.data import expected_counts, load_lp_pdbbind_splits, load_peptides_holdout
from cap6545.data.peptides import DEFAULT_PATH as PEPTIDES_PATH
from cap6545.data.splits import LP_PDBBIND_TABLE_CANDIDATES


def _lp_table_available() -> bool:
    return any(p.exists() for p in LP_PDBBIND_TABLE_CANDIDATES)


@pytest.mark.skipif(not PEPTIDES_PATH.exists(), reason="peptides CSV not downloaded")
def test_peptides_holdout_raw_count() -> None:
    df = load_peptides_holdout()
    assert len(df) == 2573


@pytest.mark.skipif(
    not _lp_table_available() or not PEPTIDES_PATH.exists(),
    reason="LP-PDBBind table or peptides CSV not available",
)
def test_lp_pdbbind_split_counts() -> None:
    splits, counts = load_lp_pdbbind_splits(peptide_overlap_removal=True)

    errors = counts.check()
    assert not errors, "\n".join(errors)

    for name, df in splits.items():
        exp, tol = expected_counts()[name]
        assert abs(len(df) - exp) <= tol, (name, len(df), exp, tol)
