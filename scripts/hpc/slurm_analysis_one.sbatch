#!/bin/bash
#SBATCH --job-name=ocl_lpfull
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=allgroups

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocl_survey || { echo "conda activate failed"; exit 1; }
export PYTHONPATH="$SLURM_SUBMIT_DIR"; export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"; STRIDE=${STRIDE:-1}

echo "===== $RUN (stride=$STRIDE) ====="
python scripts/analysis/compute_lp.py "$RUN" --stride $STRIDE --train-per-class 200 \
  || echo "LP FAILED: $RUN"
echo "DONE: $RUN"
