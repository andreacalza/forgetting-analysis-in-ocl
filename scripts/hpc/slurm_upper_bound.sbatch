#!/bin/bash
#SBATCH --job-name=ocl_ub
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=allgroups

# Offline upper-bound classifier at each task boundary only (~11 per run). One
# joint LogisticRegression over all classes seen so far, not per-task -- the
# ceiling any (non-online) classifier could reach at that point in the stream.

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocl_survey || { echo "conda activate failed"; exit 1; }
export PYTHONPATH="$SLURM_SUBMIT_DIR"; export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"

echo "===== $RUN (upper bound, boundaries only) ====="
python scripts/analysis/compute_upper_bound.py "$RUN" --train-per-class 200 \
  || echo "UPPER-BOUND FAILED: $RUN"
echo "DONE: $RUN"
