#!/bin/bash
#SBATCH --job-name=ocl_jub
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --partition=allgroups

# Joint-training upper bound: retrains the multi-head architecture from scratch on the
# union of tasks 0..j, once per boundary, and reports accuracy / SNR / LP. Unlike every
# other analysis job in this repo this one TRAINS, so it needs a real time budget: ten
# trainings of growing size, the last one over the whole dataset.

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocl_survey || { echo "conda activate failed"; exit 1; }
export PYTHONPATH="$SLURM_SUBMIT_DIR"; export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"; EPOCHS=${EPOCHS:-50}; BS=${BS:-128}

echo "===== $RUN (joint upper bound, ${EPOCHS} epochs, batch ${BS}) ====="
python scripts/analysis/compute_joint_upper_bound.py "$RUN" \
  --epochs "$EPOCHS" --batch-size "$BS" \
  || echo "JOINT-UB FAILED: $RUN"
echo "DONE: $RUN"
