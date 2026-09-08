#!/bin/bash
#SBATCH --job-name=ocl_nc
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=allgroups

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocl_survey || { echo "conda activate failed"; exit 1; }
export PYTHONPATH="$SLURM_SUBMIT_DIR"; export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"; STRIDE=${STRIDE:-5}

echo "===== $RUN (stride=$STRIDE) ====="
python scripts/analysis/compute_nc.py "$RUN" --stride $STRIDE || echo "NC FAILED: $RUN"
echo "DONE: $RUN"
