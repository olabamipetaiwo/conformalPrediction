# Conformal Prediction Intervals Under Distribution Shift for Protein-Ligand Binding Affinity

CAP 6545 final project (UCF, Fall 2026). See `Proposal/proposal.tex` for the approved scope.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Data

External datasets are not checked into git. To reproduce:

1. Register at https://www.pdbbind-plus.org.cn (academic, free) and download the structure files.
2. `python scripts/download_peptides.py` — fetches the Peptides-Holdout CSV from Zenodo 8410136.
3. `git clone https://github.com/THGLab/LP-PDBBind external/LP-PDBBind`
4. `git clone https://github.com/miladrayka/uncertainty_quantification external/uncertainty_quantification`

## Tests

```bash
pytest
```

The smoke test asserts the dataset counts from the proposal (7,259 / 947 / 2,166 / 2,498).
