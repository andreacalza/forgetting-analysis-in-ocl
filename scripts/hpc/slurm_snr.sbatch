#!/bin/bash
#SBATCH --job-name=ocl_snr
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=allgroups

# Per-run SNR analysis (twin of slurm_nc.sbatch). Writes <RUN>/snr_curve.csv.
# Usage: sbatch --export=ALL,RUN=results_ep50/er_split_cifar100_10_1000/0 scripts/hpc/slurm_snr.sbatch

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocl_survey || { echo "conda activate failed"; exit 1; }
export PYTHONPATH="$SLURM_SUBMIT_DIR"; export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"; STRIDE=${STRIDE:-5}
echo "===== $RUN (stride=$STRIDE) ====="
python scripts/analysis/compute_snr.py "$RUN" --stride $STRIDE || echo "SNR FAILED: $RUN"
echo "DONE: $RUN"
