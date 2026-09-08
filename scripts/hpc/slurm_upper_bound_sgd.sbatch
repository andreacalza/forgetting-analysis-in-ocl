#!/bin/bash
#SBATCH --job-name=ocl_ubsgd
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=allgroups

# SGDClassifier twin of slurm_upper_bound.sbatch. Writes upper_bound_curve_sgd.csv
# and never touches the LogisticRegression results already in upper_bound_curve.csv.

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocl_survey || { echo "conda activate failed"; exit 1; }
export PYTHONPATH="$SLURM_SUBMIT_DIR"; export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"

echo "===== $RUN (upper bound SGD, boundaries only) ====="
python scripts/analysis/compute_upper_bound_sgd.py "$RUN" --train-per-class 200 \
  || echo "UPPER-BOUND-SGD FAILED: $RUN"
echo "DONE: $RUN"
