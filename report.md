# Handoff Report — CAP 6545 Final Project

**Audience:** the Claude Code instance that will continue this work on HiPerGator.
**Author:** Claude instance running on the user's laptop.
**As of:** 2026-10-05.
**Weeks remaining to final:** ~9 (final due 2026-12-08 15:00; presentation week of 2026-12-01).

Read this file first. Then read `CLAUDE.md`, `Proposal/proposal.tex` (the approved single-column one — ignore `proposal_twocol.tex` and `proposal_paper.tex`), and `todo.md`. Everything else in the repo flows from those.

---

## 1. What this project is

**Course:** CAP 6545 Machine Learning Methods for Biomedical Data (UCF, Fall 2026, Dr. Haiyan Hu). Solo project. Compute: UF HiPerGator.

**Approved research question (locked by the proposal, approved 2026-10-01):**

> Do split conformal prediction intervals keep their coverage under distribution shift for structure-based protein-ligand binding affinity models?

**Base model:** FFNN on ECIF features, reproducing Rayka & Naghavi 2025's best model.

**Uncertainty variants to build:** deterministic, deep ensemble, MC dropout, Bayes-by-Backprop (BbB).
- BbB is the only one with reusable code from the authors' repo; the other three are from scratch.

**Conformal methods to build:**
- Split conformal with absolute-residual scores.
- Uncertainty-normalized conformal.
- Weighted conformal with domain-classifier density ratios.

**Research questions (as reported in the proposal):**
- **Q1** — calibration-set choice (validation set vs. random training hold-out).
- **Q2** — uncertainty-normalized nonconformity scores.
- **Q3** — weighted conformal recovery under shift.

**Cut list if behind schedule (in order):** drop BbB first → drop selective-prediction analysis. Keep Q1 and Q3 as the core; they are the paper.

---

## 2. Datasets & load-bearing counts

All counts are reconciled against the approved proposal and must be reproducible by the loader. If the loader produces different numbers beyond tolerance, **STOP and debug — do not proceed with training.**

| Split | Raw | After filters | Paper reports | Tolerance |
|---|---|---|---|---|
| LP-PDBBind CL1 train (non-covalent, minus 134 peptide-overlap) | 7,393 | **7,259** | 7,259 | exact |
| LP-PDBBind CL2 val (minus peptide overlap) | 958 | **947** | 944 | ±3 |
| LP-PDBBind CL2 test (minus peptide overlap) | 2,171 | **2,166** | 2,163 | ±3 |
| Peptides-Holdout (minus 75 covalent) | 2,573 | **2,498** | 2,503 | ±5 |
| BDB2020+ | — | **111** | 111 | exact |

**Data sources:**
- LP-PDBBind splits + structures: PDBbind registration (user confirmed registered on 2026-10-05).
- Peptides-Holdout: Zenodo 8410136, `toolboxsf_benchmarks.zip` → `csv_files/peptides_holdout_test.csv`.
- BDB2020+: from the LP-PDBBind repo (111 complexes).

**UPDATE 2026-10-12 — PDBbind structures downloaded.** Obtained the v2020.R1 "P-L" package (19,037 complexes, the reprocessed subset surviving into v2024) as `data/raw/pdbbind/P-L.tar.gz`, verified intact (`gzip -t`) and extracted to `data/raw/pdbbind/P-L/<year-range>/<pdbid>/`. The current `structures.py` indexer handles this layout unchanged (recursive `rglob`). Coverage vs. splits: train 7,131/7,259, val 947/947, test 2,160/2,166 — **134 split complexes (128 train, 6 test) are absent** because R1 dropped ~400 deprecated v2020 complexes (missing IDs in `data/raw/pdbbind/missing_from_R1.tsv`). Decision: accept the loss and document (see CLAUDE.md "Load-bearing data counts", layer 2). Operational trainable counts are 7,131 / 947 / 2,160. Not an exact Rayka & Naghavi reproduction.

---

## 3. Current repo state

**Branch:** `main`. One commit: `ee9386e dev::setup repo`. Nothing since 2026-10-01.

**Scaffolding exists; implementation does not.**

```
src/cap6545/
  data/
    __init__.py
    splits.py       (~124 lines — partial loader, verify against §2 counts)
    peptides.py     (~45 lines — peptide-set handling)
  conformal/        (empty __init__.py only)
  models/           (empty __init__.py only)
  training/         (empty __init__.py only)
  evaluation/       (empty __init__.py only)
  utils/
    paths.py
    __init__.py
tests/
  test_splits.py    (one test; needs to be fleshed out with the §2 count assertions)
scripts/
  download_peptides.py  (verify end-to-end)
data/
  raw/              (gitignored, nearly empty)
  processed/        (gitignored, empty)
Proposal/           (contains approved proposal.tex)
external/           (gitignored; LP-PDBBind + uncertainty_quantification repos go here)
notes/
  literature/
  workshop/         (untracked locally; see §7)
```

**What exists:** repo scaffolding, approved proposal, partial data-splits loader, peptide downloader, placeholder tests, `.gitignore`, `pyproject.toml`, `requirements.txt`.

**What does NOT exist yet (all required by the proposal):**
- Real PDBbind data in `data/raw/` (user just unblocked this by registering).
- ECIF feature extraction.
- FFNN model (deterministic baseline).
- Training loop + SLURM scripts.
- Deep ensemble / MC dropout / BbB implementations.
- Split / normalized / weighted conformal implementations.
- Coverage + width + selective-prediction evaluation.
- Full dataset-count assertions in tests.

---

## 4. Open blockers resolved / outstanding

**Resolved:**
- PDBbind registration — done 2026-10-05, structure download unblocked.

**Still open:**
- Read Parks et al. 2020 and Oršolić & Šmuc 2023 to confirm conformal prior art coverage.
- Peptide CSV + PDBbind structures not downloaded in-repo yet.

---

## 5. Immediate next steps (ordered)

The user wants to do all remaining work on HiPerGator — downloads, pipeline development, training. **Don't re-download anything locally.**

1. **HiPerGator setup**
   - Clone this repo into blue storage (not `$HOME` — PDBbind structures will blow past the home quota). Suggested: `/blue/<group>/<gatorlink>/cap6545/`.
   - Create a Python env matching `requirements.txt` (user to confirm whether venv + `module load python/3.11` or conda).
   - Confirm `.gitignore` keeps `data/raw/`, `data/processed/`, `external/` out of git.

2. **Download data on HiPerGator**
   - LP-PDBBind splits + structures into `data/raw/pdbbind/`.
   - Peptides-Holdout CSV into `data/raw/peptides/` (reuse `scripts/download_peptides.py`; verify it still works).
   - BDB2020+ (111 complexes) from the LP-PDBBind repo.

3. **Finish the data loader**
   - Extend `src/cap6545/data/splits.py` to produce the four splits from the raw files, applying the non-covalent filter and the 134-peptide-overlap removal.
   - Flesh out `tests/test_splits.py` with the four count assertions from §2 (exact for train & BDB2020+; tolerance 3/3/5 for val/test/peptides). Mark these as integration tests so a laptop CI without data skips cleanly.

4. **ECIF featurizer**
   - Implement in `src/cap6545/data/` (new module, e.g. `ecif.py`).
   - Cache features to `data/processed/` keyed by complex ID so training doesn't re-featurize.

5. **Deterministic FFNN baseline**
   - `src/cap6545/models/ffnn.py` — architecture per Rayka & Naghavi 2025.
   - `src/cap6545/training/train.py` — takes a config file (dataset path, seed, hyperparams). No hardcoded paths; a SLURM array job should vary only `--seed` / `--config`.
   - Smoke-test on an interactive HiPerGator node with a ~100-complex subset before launching batch jobs.

6. **Scale up on SLURM**
   - `scripts/slurm/train_deterministic.sh` — single-seed full training.
   - Then ensemble (array job over seeds), MC dropout, BbB (per cut list).

7. **Conformal + evaluation**
   - `src/cap6545/conformal/` — split, normalized, weighted (domain-classifier density ratios).
   - `src/cap6545/evaluation/` — coverage, width, selective prediction.
   - Address Q1 and Q3 first (per cut list); Q2 and BbB after if time permits.

---

## 6. Conventions to follow

From the user's global `CLAUDE.md` and this project's `CLAUDE.md`:

- **Python style:** current idiomatic Python — type hints, f-strings, `pathlib.Path` for all paths, no legacy patterns.
- **Config:** use the Pydantic-settings pattern for anything that varies by run (dataset path, seed, hyperparams). No hardcoded paths in training code.
- **Secrets:** PDBbind credentials via env vars, not hardcoded. `.env` is already in `.gitignore`.
- **Comments:** default to none. Only when the *why* is non-obvious.
- **Testing:** unit tests for pure logic; the dataset-count assertions are integration tests, mark them so they skip gracefully without data.
- **Git attribution:** end commits with `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>`. End PR descriptions with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- **No scope creep:** build only what the proposal commits to. The cut list is real — if Dec 8 is at risk, cut BbB then selective-prediction, don't rebalance scope elsewhere.

---

## 7. Post-Dec 8 workshop plan (context only — do not act on until after Dec 8)

`notes/workshop/` (untracked locally; carry over to HiPerGator) contains:
- `idea.md` — the user's workshop targeting rationale (COPA 2027 primary, ICLR MLDD parallel, Bioinformatics Letter fallback).
- `targeting_plan.md` — per-venue deltas needed on top of the Dec 8 deliverable.

Key implication for the Dec 8 code: write it so a second base model (OnionNet-2, Pafnucy) can be slotted in later without a rewrite — i.e. keep the featurizer and the model behind separable interfaces. Don't *build* a second model for Dec 8; just don't foreclose it.

Also: tag the Dec 8 commit (`v0.1-course-submission` or similar). Workshop extensions branch from that tag.

---

## 8. Timeline — realistic

Today: 2026-10-05. Final due 2026-12-08.

| Weeks | Target |
|---|---|
| Now → Oct 12 | HiPerGator env + repo + data downloaded + loader asserting counts |
| Oct 12 → Oct 19 | ECIF featurizer + deterministic FFNN smoke-tested |
| Oct 19 → Oct 26 | Full deterministic training + first ensemble run on SLURM |
| Oct 26 → Nov 9 | MC dropout + split/normalized conformal + Q1 analysis |
| Nov 9 → Nov 23 | Weighted conformal + Q3 analysis (the core finding) |
| Nov 23 → Dec 1 | BbB if on schedule; otherwise cut per §1 |
| Dec 1 → Dec 8 | Writeup, figures, code cleanup, tag + submit |

This has ~1 week of slack. Peptide-overlap bookkeeping bugs and SLURM queue waits will eat it.

---

## 9. Hand-off checklist

Before you (HiPerGator Claude) start work:

- [ ] Confirm you can read `Proposal/proposal.tex` and the two `CLAUDE.md` files.
- [ ] Confirm the project path on HiPerGator (`/blue/<group>/<gatorlink>/cap6545/`).
- [ ] Confirm the Python module + env strategy with the user.
- [ ] Ask the user to copy `notes/workshop/` over (untracked here; useful for keeping post-Dec 8 context nearby but not blocking).

Then start on §5 step 1.
