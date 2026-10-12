#!/bin/bash
# Populate the ECIF feature cache for all splits, including the peptides shift set.
# CPU-only (ECIF featurization uses RDKit, no GPU); run before the predict job.
#   sbatch scripts/slurm/featurize.sh
#SBATCH --job-name=cap6545_feat
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=8
#SBATCH --mem=24gb
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm/%x_%j.out
#SBATCH --account=so589980.ucf
#SBATCH --qos=so589980.ucf

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs/slurm

module load python/3.11
source .venv/bin/activate

python scripts/featurize_all.py
