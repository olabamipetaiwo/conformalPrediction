# Parks et al. 2020 — Proteochemometric & Conformal PL-Affinity Models

**Full citation:** Parks, C., Gaieb, Z., Amaro, R. E. An Analysis of Proteochemometric and
Conformal Prediction Machine Learning Protein-Ligand Binding Affinity Models.
*Frontiers in Molecular Biosciences* 7:93, 2020.

**Link:** https://doi.org/10.3389/fmolb.2020.00093

**Read on:**

## Setup

<!--
What the paper actually does. Expected shape (verify when reading):
- Proteochemometric (PCM) RF/XGBoost regressors on protein + ligand descriptors.
- Split conformal (Mondrian?) wrappers for prediction intervals on affinity.
- Evaluates coverage on held-out external targets.
-->

## Scope: what distribution-shift / conformal-prediction regime it covers

<!--
This is the single closest prior work for Q1. Pin down:
- Is conformal applied to a structure-based model, or only descriptor-based?
- Which external datasets stand in for the shifted test set?
- Which calibration-set choice do they use (random hold-out vs. a specified val set)?
-->

## What it covers (relevant to this project)

- Conformal intervals for PL-affinity regression (keeps validity on one external target,
  loses it on another — proposal already cites this)

## What it does NOT cover (the gap this project fills)

- LP-PDBBind leak-free splits (did not exist in 2020)
- BDB2020+ benchmark (did not exist in 2020)
- Peptide-ligand shift set
- Structure-based FFNN-ECIF model (they use proteochemometric descriptors)
- Weighted conformal under covariate shift (Tibshirani 2019 came out same year;
  check if Parks reweights calibration data)
- Uncertainty-normalized conformal scores keyed on {deep ensemble, MC dropout, BbB}

## Quotes to cite in the final report

<!-- short direct quotes with page/section anchors -->

## Open questions / things to double-check

- Which exact proteochemometric descriptor set do they use?
- Do they report nonconformity score type (absolute residual vs. normalized)?
- Do they stratify by anything (target class, affinity range)?
