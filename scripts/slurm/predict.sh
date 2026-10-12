#!/bin/bash
# Dump per-split predictions (deterministic + ensemble + MC-dropout) to runs/predictions.
#   sbatch scripts/slurm/predict.sh
#SBATCH --job-name=cap6545_pred
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=00:30:00
#SBATCH --output=runs/slurm/%x_%j.out
#SBATCH --account=so589980.ucf
#SBATCH --qos=so589980.ucf

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs/slurm

module load python/3.11
source .venv/bin/activate

# SPLITS is intentionally unquoted so multiple split names word-split into argv.
python scripts/predict.py --splits ${SPLITS:-val test} --mc-samples "${MC_SAMPLES:-50}"
