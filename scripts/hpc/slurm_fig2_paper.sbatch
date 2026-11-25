#!/bin/bash
#SBATCH --job-name=ocl_fig2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=allgroups

# Paper Fig. 2: u = 1 and u = 50 side by side, legend bottom right in both panels.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocl_survey || { echo "conda activate failed"; exit 1; }
export PYTHONPATH="$SLURM_SUBMIT_DIR" PYTHONUNBUFFERED=1 MPLBACKEND=Agg
cd "$SLURM_SUBMIT_DIR"

python scripts/analysis/paper_fig_deep_by_task.py results_ep1 results_ep50 --no-suptitle \
  --fontsize 16 --xlabel "task index (order in the stream)" \
  --legend-in each --legend-loc "lower right" --legend-ncols 2 \
  --panel-width 5.8 --panel-height 3.8 \
  --out paper_figs/fig2_deep_by_task || echo "FAILED: fig2"
echo; echo "ALL DONE"
