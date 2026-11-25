#!/bin/bash
#SBATCH --job-name=ocl_alearn
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=allgroups

# a_learn / a_final / F_deep (SGD probe) per (u, buffer), mean±std over seeds.
# Reads only the existing lp_curve_sgd.csv files -> analysis_out_alearn/alearn_afinal.csv
# Usage: sbatch scripts/hpc/slurm_alearn_afinal.sbatch

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocl_survey || { echo "conda activate failed"; exit 1; }
export PYTHONPATH="$SLURM_SUBMIT_DIR" PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"
ROOTS="${ROOTS:-results_ep1 results_ep3 results_ep5 results_ep50}"
echo "roots: $ROOTS  on $(hostname)"

python scripts/analysis/compute_alearn_afinal.py $ROOTS --outdir analysis_out_alearn \
  || { echo "FAILED: compute_alearn_afinal"; exit 1; }
echo; echo "ALL DONE"
