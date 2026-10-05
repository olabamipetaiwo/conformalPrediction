# Workshop Targeting Plan — Post-Dec 8 Extension

**Last updated:** 2026-10-02
**Base artifact:** CAP 6545 final report (`Proposal/proposal.tex`), due Dec 8, 2026.

All deadlines are from the last confirmed cycle I have; verify current CFPs before planning around them.

## The Dec 8 Baseline (what the course project already gives us)

- FFNN-ECIF reproduction on LP-PDBBind (train/val/test), BDB2020+, Peptides-Holdout.
- Four UQ variants: deterministic, deep ensemble, MC dropout, Bayes by Backprop.
- Three conformal flavors: split, uncertainty-normalized, weighted (domain-classifier density ratio).
- Multi-seed runs; coverage + width + selective-prediction analysis.
- Leak-free data pipeline with asserted split counts (7259 / 947 / 2166 / 2498).

That is a complete workshop story *only* if Q3 shows a characterizable weighted-conformal failure. If Q3 is "it works, use validation calibration + weighting," the Dec 8 draft is a class project, not a workshop paper.

## Shared Add-Ons (useful for any venue)

- **Reproducibility pack**: pin `requirements.txt`, HiPerGator SLURM scripts, trained-model checkpoints, prediction CSVs, exact seeds. Zero added research; pure packaging.
- **One-page shift metric**: a scalar (e.g., Jensen-Shannon on ECIF, or classifier AUROC between train and shifted set) that monotonically tracks coverage loss across LP-Test / BDB2020+ / Peptides. Small analysis, large reviewer impact.

## Per-Venue Plan

### Tier 1a — COPA 2027 *(primary target)*

- **Format:** PMLR full paper, up to ~15pp.
- **Deadline:** historically mid-April. Submission ~5 months after Dec 8 — comfortable.
- **Why:** dedicated conformal venue; shift-under-validity is on-topic.
- **Delta needed:**
  - Q3 characterization: *when* does weighted conformal fail? Density-ratio instability vs. shift magnitude plot.
  - A second nonconformity score family (quantile or CQR) so the methods comparison isn't one-dimensional.
  - Theory-adjacent framing: tie the empirical failure back to the exchangeability assumption, not just a benchmark table.
- **Risk:** reviewers are conformal specialists; weak Q3 → desk-reject.
- **Go/no-go:** decide by Jan 15, 2027 based on Q3 result quality.

### Tier 1b — ICLR 2027 MLDD Workshop *(parallel target if it runs)*

- **Format:** 4–6pp short paper.
- **Deadline:** historically early-to-mid February. **Tight — 2 months after Dec 8.**
- **Why:** structure-based drug-discovery audience; forgives narrow benchmark focus; strongest fit for the ECIF framing.
- **Delta needed:**
  - A **second base model** (OnionNet-2 or Pafnucy, since Rayka's repo has prediction CSVs for them nearby) → "the finding generalizes beyond FFNN-ECIF."
  - Trimmed narrative: Q1 + Q3 only, drop BbB and selective prediction into appendix.
- **Risk:** workshop may not run in 2027 (ran 2022–2024, pattern unclear). Check CFP by Jan 10.
- **Go/no-go:** decide by Jan 20, 2027.

### Tier 2 — ICML 2027 WCB Workshop *(backup if MLDD doesn't run)*

- **Format:** 4pp.
- **Deadline:** historically mid-May.
- **Delta needed:** essentially the same as MLDD; audience is broader-computational-biology, so lean on the "leak-free splits + honest coverage reporting" angle rather than drug-discovery novelty.

### Tier 3 — Bioinformatics *Applications Note* / *Letter*

- **Format:** 2pp + reproducibility pack.
- **Deadline:** rolling.
- **Why:** indexed, DOI, often faster acceptance than a workshop. Rewards the leak-free setup more than methodological novelty.
- **Delta needed:** polished reproducibility pack + a single clear coverage-under-shift headline figure. No new experiments required.
- **Risk:** low. If Q3 is unremarkable, this is the fallback that still produces a citeable artifact.

## Decision Tree

1. Dec 8 — ship course report. Freeze code at a tagged commit.
2. Dec 15–Jan 10 — assess Q3: is weighted conformal failure characterizable?
3. **If yes** → COPA primary + MLDD short-paper parallel (same core result, different page budget).
4. **If no** → Bioinformatics Letter only. Don't force a workshop pivot.
5. Dec 15 — start Shared Add-Ons regardless (reproducibility pack is pure upside).
