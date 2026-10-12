#!/bin/bash
# Dump Q1 hold-out ensemble predictions (train_holdout + val + test) for the
# calibration-set-choice analysis. Run after the train_q1_holdout array finishes.
#   sbatch scripts/slurm/predict_q1.sh
#SBATCH --job-name=cap6545_predq1
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

python scripts/predict.py \
  --splits train_holdout val test \
  --ckpt-glob 'runs/q1_holdout/seed_*/checkpoint.pt' \
  --out-dir runs/q1_holdout/predictions \
  --holdout-ids runs/q1_holdout/cal_holdout_ids.json \
  --mc-samples "${MC_SAMPLES:-50}"
