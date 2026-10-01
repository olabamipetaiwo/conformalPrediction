# CAP 6545 – Paper Critique To-Do

Course: Machine Learning Methods for Biomedical Data (UCF, Fall 2026, Dr. Haiyan Hu)

## Selected Papers

### Paper 1 — Category I (Bioinformatics with ML components, 2024+)
- **Title:** Accurate structure prediction of biomolecular interactions with AlphaFold 3
- **Authors:** Abramson et al.
- **Venue:** Nature, 2024
- **Link:** https://www.nature.com/articles/s41586-024-07487-w
- **Addendum:** https://www.nature.com/articles/s41586-024-08416-7
- **PubMed:** https://pubmed.ncbi.nlm.nih.gov/38718835/

### Paper 2 — Category II (ML, 2024+)
- **Title:** Simple and Effective Masked Diffusion Language Models (MDLM)
- **Authors:** Sahoo, Arriola, Gokaslan, Marroquin, Rush, Schiff, Chiu, Kuleshov
- **Venue:** NeurIPS 2024
- **Link:** https://neurips.cc/virtual/2024/poster/95622
- **Code:** https://github.com/kuleshov-group/mdlm

## Tasks

- [x] Get assignment constraints (one paper per category, 2024 or later)
- [x] Choose Paper 1 (Category I): AlphaFold 3
- [x] Choose Paper 2 (Category II): MDLM
- [ ] Confirm Nature link resolves and that both papers meet the instructor's criteria
- [ ] Read both papers in full
- [ ] Paper 1 critique: summary, strengths, weaknesses, open questions
- [ ] Paper 2 critique: summary, strengths, weaknesses, open questions
- [ ] Confirm critique length, format, and due date with the assignment text

## Final Project

- **Proposal due:** Tue Sep 29, 2026, 11:59pm (unlimited attempts on Canvas)
- **Final project presentations:** week of Dec 1 (last class Dec 3)
- **Final project due:** Dec 8, 3pm
- **Team:** solo, so no team contribution plan is needed
- **Compute:** HiPerGator (UF) GPUs
- **Alternative proposals (two-column layout):** `proposal_twocol.tex` (safe version: conformal coverage under shift) and `proposal_paper.tex` (publication-track version: conformal selection with FDR control and novelty-aware weights). Pick one before submitting on Sep 29.
- **Candidate ideas:**
  - Protein-ligand binding affinity prediction with uncertainty (builds on AlphaFold 3 and Boltz-2). Recommended.
  - HMM for sequence annotation (matches the course)
  - Masked-diffusion model for peptide or protein sequence generation (builds on MDLM)
  - Chest X-ray or pathology classification under domain shift
  - EHR risk prediction with EM imputation (MIMIC access is slow)
- **Status:** topic chosen and revised after a literature review: do conformal prediction intervals keep their coverage under distribution shift for binding affinity prediction on LP-PDBBind and BDB2020+ (proposal in `proposal.tex`). The earlier plan (comparing 5 uncertainty methods, plus Boltz-2) was dropped because a 2025 Scientific Reports paper already did the comparison, and Boltz-2 is preprint-only.

### Project tasks
- [x] Choose the topic
- [x] HiPerGator access (already have it)
- [ ] Register for PDBbind (free for academics) to get the structure files
- [x] Peptide set found: Zenodo 8410136, `toolboxsf_benchmarks.zip` -> `csv_files/peptides_holdout_test.csv` (2,573 rows; 2,498 after dropping covalent)
- [x] Split sizes reconciled: CL1 train (non-covalent) minus the 134 peptide-set overlaps = 7,259, matching the published count exactly; val/test after removing peptide-set overlaps = 947 / 2,166 (published 944 / 2,163, 3 apart each)
- [ ] Download the peptide CSV and PDBbind structures once registered (pilot data and scripts are in the session scratchpad only)
- [ ] Read Parks et al. 2020 and Oršolić and Šmuc 2023 in full to confirm what conformal work already covers
- [ ] Compile `proposal.tex` (Overleaf) and check it
- [ ] Write and submit the proposal (see `proposal.tex`, due Sep 29)

## Notes

- Critique points to explore for AlphaFold 3: limited code and data access at release, hallucination in disordered regions (see DisProtBench: https://arxiv.org/pdf/2507.02883), generalization beyond the training distribution.
- Critique points to explore for MDLM: the simplified Rao-Blackwellized objective, how fair the comparison to autoregressive baselines is, and the link to probabilistic inference from the course.
- Assignment due date: TBD
