#!/bin/bash
# Deep-ensemble training: one GPU task per seed via a SLURM array.
#   sbatch --array=0-9 scripts/slurm/train_ensemble.sh
# The array index IS the seed, so each member is independently reproducible.
#SBATCH --job-name=cap6545_ens
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm/%x_%A_%a.out
##SBATCH --account=CHANGE_ME
##SBATCH --qos=CHANGE_ME

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs/slurm

module load python/3.11
source .venv/bin/activate

python scripts/train.py --config configs/deterministic.yaml --seed "$SLURM_ARRAY_TASK_ID"
