---
marp: true
theme: default
paginate: true
size: 16:9
header: "CAP 6545 Final Project — Progress Update"
footer: "Fall 2026 · Dr. Haiyan Hu"
style: |
  section { font-size: 24px; }
  h1 { color: #1a3c6e; }
  h2 { color: #1a3c6e; }
  table { font-size: 20px; }
  .small { font-size: 18px; color: #555; }
  strong { color: #b03a2e; }
---

# Conformal Prediction Under Distribution Shift

### for structure-based protein–ligand binding affinity

**Progress update — 4-minute checkpoint**

<br>

Solo project · UF HiPerGator · ~6 weeks ahead of schedule

<span class="small">Final due 2026-12-08 · Presentation week of 2026-12-01</span>

---

<!-- ===================== PART 1: PROBLEM (2 pts) ===================== -->

## 1 · The problem

**Research question (locked by approved proposal):**

> Do **split conformal prediction** intervals keep their **coverage** under
> **distribution shift** for structure-based protein–ligand binding affinity models?

- Conformal prediction gives a *distribution-free* coverage guarantee — but only under **exchangeability** (train/calibration/test drawn alike).
- Drug-discovery deployment **violates** that: new chemotypes, new targets, new assays.
- **Why it matters:** a binding-affinity model that reports "90% intervals" that actually cover 50% is worse than one that is honest about what it doesn't know.

---

## 1 · Shift is real, and it comes in degrees

We evaluate on a **ladder of increasing shift** away from the calibration data:

| Eval set | Domain-classifier AUC | Ensemble RMSE (pK) |
|---|---|---|
| CL2 **test** (in-family) | 0.78 | 1.6 |
| **Peptides**-Holdout | 0.99 | 2.75 |
| **BDB2020+** (new assays) | 1.00 | 3.10 |

<span class="small">AUC = how separable the eval set is from the calibration set using the same ECIF features. BDB/peptides are near-separable → severe covariate shift.</span>

---

<!-- ===================== PART 2: METHOD & IMPLEMENTATION (5 pts) ===================== -->

## 2 · Method — base model & data

**Base model:** FFNN on **ECIF** features — reproducing Rayka & Naghavi (2025).
`1540 ECIF → 790 selected → 790/900/300/1 FFNN`

**Data (LP-PDBBind, scaffold/family splits):**

| Split | Role | Trainable |
|---|---|---|
| CL1 train | fit model | 7,131 |
| CL2 val | **calibration** | 947 |
| CL2 test / Peptides / BDB2020+ | shifted eval | 2,160 / 2,350 / 115 |

<span class="small">Trainable counts reflect PDBbind **v2020.R1** structures (134 split complexes dropped by R1 — documented deviation, higher-quality structures, not an exact reproduction).</span>

---

## 2 · Method — uncertainty × conformal matrix

**Uncertainty models (implemented):**
- Deterministic FFNN
- **Deep ensemble** (10 seeds) — primary point predictor, val RMSE ~1.60–1.62
- MC dropout
- *Bayes-by-Backprop — deferred (cut-list item #1)*

**Conformal methods (implemented):**
- **Split** conformal — absolute-residual scores
- **Uncertainty-normalized** — residual ÷ ensemble disagreement
- **Weighted** — domain-classifier density ratios (Tibshirani et al.)

---

## 2 · Implementation

- **Clean, config-driven pipeline** on HiPerGator — `src/cap6545/{data,models,training,conformal,evaluation}`, Pydantic-settings configs, no hardcoded paths.
- **SLURM** GPU array jobs for the 10-seed ensemble; CPU jobs for ECIF featurization and all conformal analyses.
- **Reproducible:** every table below regenerates from committed `scripts/experiment_*.py` over cached prediction dumps.
- **Tested:** loader asserts the load-bearing split counts; 30 fast tests pass, ruff clean.

<span class="small">Featurizer and model kept behind separable interfaces so a second base model can slot in later.</span>

---

<!-- ===================== PART 3: RESULTS (2 pts) ===================== -->

## 3 · Results — three questions, one narrative

**Q1 — Does the calibration set choice matter?** (coverage @ nominal 0.90, eval=test)

| Calibration source | Coverage @0.90 |
|---|---|
| Train hold-out (CL1) | 0.694 |
| Official val split (CL2) | 0.823 |
| Test-oracle (in-distribution) | 0.899 |

**→ Distribution, not size, drives coverage.** Coverage tracks distance to the test set; split conformal is only as good as its exchangeability assumption.

---

## 3 · Results — buying back coverage

**Q2 — Uncertainty-normalized scores** (cal=val, target 0.90):

| | test | peptides | bdb |
|---|---|---|---|
| ensemble, plain | 0.836 | 0.678 | 0.504 |
| ensemble, **std-normalized** | **0.945** | **0.860** | 0.713 |

Ensemble disagreement carries shift signal → recovers coverage with **finite, adaptive** widths. Weakens as shift deepens.

**Q3 — Weighted conformal** (the headline **limitation**): helps only under *mild, overlapping* shift (val→test). Under severe shift (peptides/BDB) likelihood ratios diverge → intervals inflate to **+∞** — coverage "recovered" but **vacuous**.

---

<!-- ===================== PART 4: PLAN TO FINISH (1 pt) ===================== -->

## 4 · Plan to finish

| When | Task |
|---|---|
| Now → Nov 9 | **Harden:** multi-seed / bootstrap confidence intervals on Q1–Q3 tables |
| | **Figures:** coverage-vs-shift, width distributions, Q3 ∞-width fraction |
| Nov 9 → Nov 23 | Draft paper (intro → method → results → discussion) |
| Nov 23 → Dec 1 | *Bayes-by-Backprop + selective prediction — only if ahead (cut-list)* |
| Dec 1 → Dec 8 | Polish, tag `v0.1-course-submission`, submit |

**Status:** all three research questions have first-pass results; ahead of the proposal timeline. Remaining risk is robustness/writeup, not the core findings.

---

## Summary

- **Problem:** conformal coverage guarantees assume exchangeability — drug discovery breaks it.
- **Method:** FFNN-ECIF base model × (ensemble, MC dropout) × (split, normalized, weighted) conformal, over a measured shift ladder.
- **Results:** calibration distribution drives coverage (Q1); uncertainty-normalization buys back adaptivity (Q2); reweighting recovers coverage only while support overlaps, else diverges (Q3).
- **Next:** confidence intervals + figures → writeup.

**Feedback welcome.**
