#!/bin/bash
# Single-seed deterministic FFNN training on one GPU.
#   sbatch scripts/slurm/train_deterministic.sh
# Set SBATCH_ACCOUNT / SBATCH_QOS in your env, or edit the two lines below.
#SBATCH --job-name=cap6545_det
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm/%x_%j.out
##SBATCH --account=CHANGE_ME
##SBATCH --qos=CHANGE_ME

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs/slurm

module load python/3.11
source .venv/bin/activate

python scripts/train.py --config configs/deterministic.yaml --seed "${SEED:-0}"
