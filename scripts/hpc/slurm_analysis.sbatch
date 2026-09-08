#!/bin/bash
#SBATCH --job-name=ocl_analysis
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:TESLA-T4-16GB:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=allgroups

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocl_survey || { echo "conda activate failed"; exit 1; }
export PYTHONPATH="$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"

for d in results/naive_split_cifar100_10_*/0 results/er_split_cifar100_10_*/0; do
  [ -d "$d/checkpoints" ] || { echo "skip (no checkpoints): $d"; continue; }
  echo "===================== $d ====================="
  python scripts/analysis/compute_lp.py  "$d" --stride 5 --train-per-class 200 || echo "LP FAILED: $d"
  python scripts/analysis/compute_snr.py "$d" --stride 5 || echo "SNR FAILED: $d"
done
echo "ALL ANALYSIS DONE"
