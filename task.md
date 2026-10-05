Open items from todo.md
- PDBbind registration done (2026-10-05) — structure download unblocked.
- Peptide CSV + PDBbind structures not downloaded in-repo yet — the pilot numbers in the proposal came from scratchpad data.
- Parks et al. 2020 and Oršolić & Šmuc 2023 still unread.

What hasn't been built at all (vs. the approved proposal scope)
- ECIF feature extraction.
- FFNN model (deterministic baseline).
- Training loop + HiPerGator SLURM scripts.
- Deep ensemble, MC dropout, Bayes-by-Backprop variants.
- Split / normalized / weighted conformal implementations.
- Coverage + width + selective-prediction evaluation.

Timeline check — 9 weeks left. Per the proposal's cut list, if things slip, drop BbB first, then selective-prediction; keep Q1 (calibration-set choice) and Q3 (weighted conformal).

Want me to pick the next concrete thing to start — probably either (a) wiring up the real data download + asserting the 7,259 / 947 / 2,166 / 2,498 counts in test_splits.py, or (b) scaffolding the ECIF featurizer + deterministic FFNN?


---

Next concrete step is pulling the actual data into data/raw/ and writing it into the loader + tests so the asserted counts (7,259 / 947 / 2,166 / 2,498) are reproducible from a clean clone. Rough shape:

1. scripts/download_pdbbind.py — pulls the LP-PDBBind split files (CL1 train, CL2 val/test) and the structure archive into data/raw/pdbbind/ (gitignored). PDBbind credentials via env vars, not hardcoded.
2. scripts/download_peptides.py — already exists; verify it still works end-to-end and lands files in data/raw/peptides/.
3. Extend src/cap6545/data/splits.py to load the real files (not just whatever stub it has now), apply the non-covalent filter and the 134-peptide-overlap removal.
4. Flesh out tests/test_splits.py with the four count assertions (exact for train, tolerance 3/3/5 for val/test/peptides) — mark it as a slow/integration test so CI without data skips it cleanly.

Want me to start on #1 (the PDBbind download script), or do you want to run the download by hand and have me go straight to #3 (loader + assertions)?