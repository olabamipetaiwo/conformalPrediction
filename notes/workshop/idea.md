The approved version is a replication-plus-extension study. The novelty is "apply conformal (split / normalized / weighted) to an existing model on existing benchmarks that nobody has specifically combined." That's a real but thin gap. Realistic venue if you did ~20–30% more work on top of the course deliverable: a NeurIPS/ICLR workshop like ML4H, AIDrugX, or MLDD, or a short paper at Bioinformatics / JCIM. Three things would move it from "class project" to "submittable":

1. A second base model (Pafnucy, OnionNet) so the claim isn't FFNN-ECIF-specific.
2. A surprising finding — the strongest version is Q3's weighted conformal failing in a characterizable way (unstable density ratios on peptides). "It works" is a weaker story than "here's exactly when and why it breaks."
3. The abandoned proposal_paper.tex angle (conformal selection with FDR control + novelty-aware weights) is the version that was actually publication-tracked — you set it aside because it's harder and riskier, and that was the right call for Dec 8. But it's the one that would clear a workshop review cleanly.

My recommendation: ship the course project as scoped (safer, timeline fits), and if the Q3 result turns out interesting (unstable weights, coverage drift characterizable by shift magnitude), extend after Dec 8 into a workshop submission using the paper_tex variant as the structural outline. Workshop deadlines in Jan–Mar give you runway.


    Tier 1 — strongest fit:

- COPA 2027 — Conformal and Probabilistic Prediction with Applications (PMLR). The dedicated conformal venue; a paper about conformal validity under covariate shift lands here natively. Full papers up to ~15pp. Deadline historically mid-April, conference late August/September.
- ICLR 2027 MLDD Workshop — Machine Learning for Drug Discovery. Short papers (4–6pp). Deadline historically early-to-mid February. Best fit for the structure-based framing, if it runs again (ran 2022–2024; sometimes alternates).

Tier 2 — adjacent, viable:

- ICML 2027 WCB Workshop — Workshop on Computational Biology. 4pp. Deadline ~mid-May. Broader bioinformatics audience; reviewers appreciate honest negative-result framing.
- AAAI/ICML Trustworthy ML workshops — conformal is a trustworthy-ML staple. Timing fits if ICLR MLDD doesn't run; fit is weaker because the application side gets less love from the UQ-theory reviewers.

Tier 3 — short journal paper, often better than workshops:

- Bioinformatics – Applications Note (2pp) or Letter to the Editor. Rolling. If the Q3 result is clean, this format is specifically for exactly this kind of focused empirical finding, has a real DOI, and reviewers care about reproducibility — which plays to your leak-free setup.
- JCIM (ACS Journal of Chemical Information and Modeling). Longer form, structure-based audience, slower turnaround. Only worth it if you have a second base model and a diagnostic/methods angle.

My ranking for your situation: COPA → ICLR MLDD → Bioinformatics Letter. COPA because it's the native venue; MLDD because the pediatric audience of drug-discovery ML reviewers is forgiving of focused benchmarks; Bioinformatics because a clean 2-page letter is often a better ROI than a workshop poster.

If you want, I can draft a one-page targeting plan with the specific "what we'd need to add on top of the Dec 8 deliverable" per venue — e.g., for COPA you really need the Q3 charaatics you need reproducibility packaging more thannew results.