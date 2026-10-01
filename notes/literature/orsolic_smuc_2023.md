# Oršolić & Šmuc 2023 — Dynamic Applicability Domain / Local Conformal

**Full citation:** Oršolić, D. and Šmuc, T. Dynamic applicability domain (dAD):
compound-target binding affinity estimates with local conformal prediction.
*Bioinformatics* 39(8):btad465, 2023.

**Link:** https://doi.org/10.1093/bioinformatics/btad465

**Read on:**

## Setup

<!--
Local / Mondrian conformal prediction on kinase binding data.
Key questions to answer:
- Which kinase dataset exactly (ChEMBL subset? KIBA? Davis?)
- What is "dynamic applicability domain" operationally — kNN-conditional
  calibration? Mondrian conformal with target-class bins?
- Which base regressor?
-->

## Scope: what distribution-shift / conformal-prediction regime it covers

- Covers novel compounds and novel targets (relevant to Q1 and shift framing).
- Kinase-only — does NOT touch PDBbind or structure-based models.

## What it covers (relevant to this project)

- Local/Mondrian conformal framing for drug-affinity: a precedent for the
  calibration-set-choice question (Q1).
- Novel-compound and novel-target evaluation protocol — their experimental split
  is a template for how to describe "shift" quantitatively.

## What it does NOT cover (the gap this project fills)

- LP-PDBBind (uses kinase/ChEMBL data instead)
- Structure-based FFNN on ECIF features (uses compound-target descriptors)
- Peptide-ligand set
- Weighted conformal (Q3) via density-ratio reweighting
- Uncertainty-normalized conformal with {deep ensemble, MC dropout, BbB} (Q2)

## Quotes to cite in the final report

## Open questions / things to double-check

- Do they report coverage AND width, or only one of them?
- How do they estimate local calibration — kNN count, class membership, both?
- What is their nonconformity score definition?
