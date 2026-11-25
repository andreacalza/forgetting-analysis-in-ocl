#!/bin/bash
#SBATCH --job-name=ocl_lpsgd
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
cd "$SLURM_SUBMIT_DIR"; STRIDE=${STRIDE:-1}

echo "===== $RUN (stride=$STRIDE) ====="
python scripts/analysis/compute_lp_sgd.py "$RUN" --stride $STRIDE --train-per-class 200 || echo "LP-SGD FAILED: $RUN"
echo "DONE: $RUN"

# Usage, one job per run dir:
# for d in results_ep1/*/*/ results_ep3/*/*/ results_ep5/*/*/; do
#   sbatch --export=ALL,RUN="${d%/}" scripts/hpc/slurm_lp_sgd.sbatch
# done
