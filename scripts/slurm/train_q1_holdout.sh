#!/bin/bash
# Q1 training-hold-out ensemble: one GPU task per seed via a SLURM array.
# Reserves a shared 10% calibration partition from CL1 train (never trained on).
#   sbatch --array=0-9 scripts/slurm/train_q1_holdout.sh
#SBATCH --job-name=cap6545_q1
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm/%x_%A_%a.out
#SBATCH --account=so589980.ucf
#SBATCH --qos=so589980.ucf

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs/slurm

module load python/3.11
source .venv/bin/activate

python scripts/train.py --config configs/q1_holdout.yaml --seed "$SLURM_ARRAY_TASK_ID"
