# Laghuvarapu, Lin, Sun 2023 — CoDrug: Conformal + Density Estimation Under Shift

**Full citation:** Laghuvarapu, S., Lin, Z., Sun, J. CoDrug: Conformal Drug Property
Prediction with Density Estimation under Covariate Shift. *Advances in Neural
Information Processing Systems* 36, 2023.

**Link:** https://proceedings.neurips.cc/paper_files/paper/2023/hash/7691484a7a35d5e2742279c1d926b778-Abstract-Conference.html

**Read on:**

## Setup

<!--
Weighted conformal prediction (per Tibshirani 2019) applied to drug property
prediction, with the density ratio estimated by a dedicated density estimator
rather than a domain classifier.
Key questions to answer:
- Which drug property tasks (regression vs. classification; which MoleculeNet subsets)?
- Which density estimator (normalizing flow? KDE in a learned embedding?)
- How do they report coverage AND width together?
-->

## Scope: what distribution-shift / conformal-prediction regime it covers

- Core method for Q3: weighted conformal with learned density ratios.
- Focused on 2D-descriptor / graph models — not structure-based.

## What it covers (relevant to this project)

- Reference implementation and evaluation protocol for weighted conformal in a
  drug-discovery setting; this project uses a domain-classifier variant of the
  same idea.
- Width-at-equal-coverage comparison; useful template for the Q2 analysis format.

## What it does NOT cover (the gap this project fills)

- Structure-based model (FFNN on ECIF features) — they use 2D-descriptor /
  graph nets.
- LP-PDBBind + BDB2020+ + peptide-ligand shift set.
- Comparison of calibration-set choice (random train hold-out vs. the LP-PDBBind
  validation split) — their calibration setup is fixed.
- Uncertainty-normalized conformal scores keyed on {deep ensemble, MC dropout, BbB}.

## Quotes to cite in the final report

## Open questions / things to double-check

- Do they use a density estimator or a domain classifier for the ratio?
  (The proposal commits to a domain classifier; CoDrug may do the opposite —
  worth being explicit about the difference in the final report.)
- How stable are their weight estimates on small shifted sets?
- Do they report any failure modes for weight instability?
