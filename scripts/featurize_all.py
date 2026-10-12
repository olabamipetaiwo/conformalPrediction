"""Featurize all available complexes into the ECIF cache (data/processed/ecif).

Populates the per-complex .npy cache training reads from. Covers the three
LP-PDBBind splits (from the v2020.R1 P-L structures) and the BDB2020+ shift set.
Peptides-Holdout is skipped until its structures are downloaded.
"""

from __future__ import annotations

import time
from pathlib import Path

from cap6545.data.ecif import ECIFFeaturizer
from cap6545.data.featurize import DEFAULT_CACHE_DIR, featurize_complex
from cap6545.data.splits import load_lp_pdbbind_splits
from cap6545.data.structures import index_structures, records_for_ids

P_L = Path("data/raw/pdbbind/P-L")
BDB = Path("data/raw/pdbbind/BDB2020+")


def run(name: str, records, featurizer) -> None:
    ok, fails, t0 = 0, [], time.perf_counter()
    for r in records:
        try:
            featurize_complex(r, featurizer=featurizer)
            ok += 1
        except Exception as e:
            fails.append((r.pdbid, type(e).__name__, str(e)[:100]))
    dt = time.perf_counter() - t0
    print(f"[{name:6}] featurized={ok}  failed={len(fails)}  ({dt:.1f}s)")
    for pid, exc, msg in fails[:10]:
        print(f"         FAIL {pid}: {exc}: {msg}")


def main() -> None:
    feat = ECIFFeaturizer()
    print(f"cache dir: {DEFAULT_CACHE_DIR}")

    splits, _ = load_lp_pdbbind_splits()
    lp_idx = index_structures(P_L)
    print(f"P-L indexed: {len(lp_idx)} structures\n")
    for name in ("train", "val", "test", "peptides_nc"):
        ids = list(splits[name].index)
        records, missing = records_for_ids(ids, lp_idx)
        print(f"[{name:11}] labels={len(ids)}  resolved={len(records)}  no-structure={len(missing)}")
        run(name, records, feat)

    print()
    bdb_idx = index_structures(BDB)
    bdb_records, _ = records_for_ids(list(bdb_idx.keys()), bdb_idx)
    print(f"[bdb   ] indexed={len(bdb_idx)}")
    run("bdb", bdb_records, feat)

    cached = len(list(DEFAULT_CACHE_DIR.glob("*.npy")))
    print(f"\ntotal cached vectors: {cached}")


if __name__ == "__main__":
    main()
