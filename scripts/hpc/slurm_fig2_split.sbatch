#!/bin/bash
#SBATCH --job-name=ocl_fig2split
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=allgroups

# Fig. 2 split into two images with their own legends (same colour per buffer in both):
#   paper_figs/deep_by_task_u1-5.{png,pdf}  (u = 1, 3, 5)
#   paper_figs/deep_by_task_u50.{png,pdf}   (u = 50)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocl_survey || { echo "conda activate failed"; exit 1; }
export PYTHONPATH="$SLURM_SUBMIT_DIR" PYTHONUNBUFFERED=1 MPLBACKEND=Agg
cd "$SLURM_SUBMIT_DIR"
A=scripts/analysis

python $A/paper_fig_deep_by_task.py results_ep1 results_ep3 results_ep5 --no-suptitle --ylim=-17,9 \
  --out paper_figs/deep_by_task_u1-5 || echo "FAILED: u1-5"
python $A/paper_fig_deep_by_task.py results_ep50 --no-suptitle --ylim=-17,9 \
  --out paper_figs/deep_by_task_u50 || echo "FAILED: u50"
echo; echo "ALL DONE"
