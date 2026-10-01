# CAP 6545 Pre-Proposal Presentation: Slide Content

Sep 24, 2026 · 3–4 minutes plus discussion · six slides, about 35–40 seconds each

---

## Slide 1: Title

Do Conformal Prediction Intervals Stay Valid Under Distribution Shift?
A leak-free study of protein–ligand binding affinity prediction

Taiwo Olabamipe · Computer Science · University of Central Florida
CAP 6545 · Fall 2026 · Dr. Haiyan Hu

Speaker notes (~15 s): My project is about knowing when to trust a drug-binding prediction, especially on targets the model has never seen.

---

## Slide 2: The problem

- Drug discovery needs to know how tightly a molecule binds a protein (binding affinity)
- Deep learning models predict it, but give one number with no confidence
- Conformal prediction wraps any model in an interval with a guarantee: e.g. 90% of true values fall inside
- The guarantee assumes calibration and test data look alike. In drug discovery, new targets and new chemistry break that assumption

Speaker notes (~40 s): A chemist needs to know when a prediction is reliable. Conformal prediction gives a guaranteed interval, but only if the test data resembles the calibration data. New drug targets usually don't.

---

## Slide 3: What is known, and the gap

Known
- Conformal prediction has been applied to binding affinity: valid on one external target, not on another (Parks et al., 2020)
- Weighted conformal prediction corrects covariate shift (Tibshirani et al., NeurIPS 2019); adapted to drug properties (CoDrug, NeurIPS 2023)
- Uncertainty for structure-based models degrades under shift on leak-free splits, but without conformal methods (Rayka & Naghavi, Sci. Rep. 2025)

Gap: no conformal evaluation I found on leak-free LP-PDBBind, BDB2020+, and a peptide-ligand shift set. This is an empirical extension, not a new method.

Pilot (weak proxy model, real LP-PDBBind data): test coverage at a 90% target was
- 90.3% calibrated on the LP validation set
- 82.3% calibrated on a random slice of training data
- 89.8% after normalizing by model uncertainty

Speaker notes (~50 s): Related work exists, so I'm not claiming a new method. What I found missing is a leak-free test on structure-based models. A quick pilot with a simple model shows the effect is real: the choice of calibration set moved coverage from 90% to 82%.

---

## Slide 4: Research questions

1. Q1: How does the calibration set (random training hold-out vs. LP-PDBBind validation) change coverage on the LP test set, BDB2020+, and the peptide set?
2. Q2: Does normalizing intervals by each method's uncertainty keep coverage at target under shift, while staying narrow?
3. Q3: Does weighted conformal prediction recover the coverage lost under shift?

Model: feed-forward network on ECIF features (Sánchez-Cruz et al., Bioinformatics 2021), with deterministic, deep ensemble, MC dropout, and Bayes-by-Backprop versions

Speaker notes (~40 s): Three questions. Does the choice of calibration set matter, does uncertainty-based normalizing help, and does reweighting recover lost coverage. Any answer is useful: either a warning or a recipe.

---

## Slide 5: Data and resources

| Set | Role | Size |
|---|---|---|
| LP-PDBBind (Li et al., J. Phys. Chem. B 2026) | train / validation / test | 7,259 / 947 / 2,166 |
| BDB2020+ | shifted test | 111 |
| Peptides-Holdout (Durant et al., Bioinformatics 2025) | shifted test | 2,498 |

- All public. I removed the 134 training complexes that overlap the peptide set, which reproduces the published training count of 7,259 exactly
- PDBbind structure files: free for academics after registration
- Compute: HiPerGator GPUs for ensembles and repeated seeds
- Limit: BDB2020+ is small, so coverage there is only detectable within about ±5.6 points

Speaker notes (~45 s): All data are public and I've already checked them. The peptide set overlapped with training, so I removed those complexes, which matches the published counts. BDB2020+ is small, so I'll say what it can and can't show.

---

## Slide 6: Plan and risks

| Weeks | Plan |
|---|---|
| Sep 29 – Oct 12 | Register for PDBbind, build splits, compute ECIF features |
| Oct 13 – Oct 26 | Train the baseline network, add ensemble, MC dropout, Bayes-by-Backprop |
| Oct 27 – Nov 9 | Split conformal intervals and uncertainty-normalized scores (Q1, Q2) |
| Nov 10 – Nov 23 | Weighted conformal (Q3), selective-prediction analysis, start report |
| Nov 24 – Dec 8 | Final presentation (Dec 1 week), report and code due Dec 8, 3 PM |

Risks and fallbacks
- Time: drop Bayes-by-Backprop first, then the selective-prediction analysis
- Weighted conformal may be unstable: report that as a finding
- Possible upgrade if results are interesting: conformal selection with false-discovery-rate control

Speaker notes (~40 s): The plan runs in two-week blocks. If I fall behind I have a defined order for what to cut. Thank you, I'd welcome questions on the calibration setup or the choice of shift sets.

---

## References (optional backup slide)

- Angelopoulos & Bates. *Conformal Prediction: A Gentle Introduction.* Found. Trends Mach. Learn. 16(4), 2023
- Tibshirani, Barber, Candès, Ramdas. *Conformal Prediction Under Covariate Shift.* NeurIPS 32, 2019
- Laghuvarapu, Lin, Sun. *CoDrug.* NeurIPS 36, 2023
- Parks, Gaieb, Amaro. Front. Mol. Biosci. 7:93, 2020
- Oršolić & Šmuc. Bioinformatics 39(8):btad465, 2023
- Rayka & Naghavi. Sci. Rep. 15:43156, 2025
- Li et al. *Leak Proof PDBBind.* J. Phys. Chem. B 130:730–740, 2026
- Durant et al. Bioinformatics 41(2):btaf040, 2025
- Sánchez-Cruz et al. Bioinformatics 37:1376–1382, 2021
