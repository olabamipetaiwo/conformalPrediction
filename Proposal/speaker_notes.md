# CAP 6545 Pre-Proposal: Speaker Notes

Rule: nothing here repeats what is on a slide. Each note adds the reasoning, an example, or a transition.
Target: about 3:20 spoken, leaving buffer inside the 3–4 minute limit. Roughly 150 words per minute.

---

## Slide 1: Title (~10 s)

Picture a chemist choosing which few compounds to synthesize out of thousands. The model says one of them binds tightly. The question I care about is how much that number can be trusted, and whether the answer changes when the protein is one the model has never seen.

---

## Slide 2: The problem (~35 s)

Think of a weather forecast tuned on Florida data and then used in Alaska. It still produces confident-looking numbers, and nothing warns you it has stopped being reliable.

That is the failure I'm interested in. A conformal interval that has lost its guarantee looks exactly like one that hasn't. So the risk is silent, and a chemist could spend synthesis budget on a bad bet without knowing.

The appeal of conformal prediction is that it needs no assumptions about the model. All of its assumptions sit on the data, and that is precisely the part drug discovery keeps violating.

---

## Slide 3: What is known, and the gap (~45 s)

I'm not going to walk through the citations. What matters is that each paper covers one piece of the puzzle, and none of them put the pieces together on clean data.

"Leak-free" is the important word. In standard PDBBind splits, near-duplicate complexes end up on both sides of the split. That makes the shift look smaller than it really is, and a coverage study on such data would be flattering itself.

On the pilot: I deliberately used a weak ligand-only model, so the effect can't be blamed on a fancy network. I also bootstrapped the drop, and the interval sits well below the target, so it isn't sampling noise.

The caveat is that this is a proxy. I'm not claiming the same numbers for the real model. The pilot only shows the question is worth asking.

---

## Slide 4: Research questions (~35 s)

Read the three questions as diagnose, treat, repair. Q1 asks whether the guarantee actually breaks. Q2 asks whether a cheap fix works, which is scaling each interval by how unsure the model is. Q3 asks whether a heavier fix works, which is reweighting the calibration data.

That is also why I have four network variants. They aren't a method comparison, since a 2025 paper already did that. They're four different ways of getting a per-prediction uncertainty to feed into Q2.

I kept the network small on purpose, so the effort goes into the conformal question and not into architecture tuning.

---

## Slide 5: Data and resources (~40 s)

The peptide set is a real shift, not just a different label. Its complexes have measurably weaker binding on average than the training set, so the model is being asked about a different part of the affinity range.

For the counts: I match the published training size, and the validation and test sets land within three complexes of theirs. I'll report the exact numbers I get rather than pretend they are identical.

On the small set: I'll be upfront that a pass there proves little. The peptide set is large enough to carry the statistical weight. The other small sets in the literature, the EGFR and Mpro ones, are too small for coverage claims, so I'll only describe them.

---

## Slide 6: Plan and risks (~35 s)

I won't read the schedule. The point is that the first three weeks carry the most risk. A wrong data pipeline would quietly invalidate every result after it.

So I've set a hard checkpoint: I don't move on to the conformal work until my baseline's error and correlation are close to the published values.

The cut order follows what matters most. Q1 and Q3 are the core, so everything else goes first. Thanksgiving week is deliberately light.

I'd especially value feedback on whether the two shifted test sets are enough, or whether you'd want a third kind of shift. Thank you.

---

## Q&A prep (not spoken; for the discussion period)

- **Why not Boltz-2 or other newer models?** Preprint only, and the project restricts its baselines to published work.
- **Why not a bigger or newer network?** The question is about conformal validity, not accuracy. A small, cheap model allows many seeds and repeated runs on HiPerGator.
- **What if there's no coverage drop on the real model?** That is a valid finding, and the pilot itself showed coverage holding on some subsets. It would still tell practitioners the guarantee is safe in that setting.
- **Is this novel?** Not as a method. It is an empirical extension, and the claim is limited to my literature search.
- **Why is weighted conformal risky?** It depends on estimating the density ratio between calibration and test data. That estimate can be unstable, and if so I report that as a result.
- **Honesty flag:** the to-do list shows you haven't yet read Parks et al. 2020 or Oršolić & Šmuc 2023 in full. If asked what they cover, describe only what the proposal says, or say you're finishing those reads this week.
