#!/bin/bash
#SBATCH --job-name=ocl_paperfigs
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=allgroups

# Paper metrics + figures from the existing per-run CSVs (nothing is retrained):
#   1. F_shallow as in Eq. (1) of the paper  -> analysis_out_shallow/
#   2. Fig. 1 (shallow/deep vs buffer)        -> paper_figs/forgetting_vs_buffer.{png,pdf,csv}
#   3. Fig. 2 (deep forgetting by task)       -> paper_figs/deep_by_task.{png,pdf}
# Usage: sbatch scripts/hpc/slurm_paper_figs.sbatch

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocl_survey || { echo "conda activate failed"; exit 1; }
export PYTHONPATH="$SLURM_SUBMIT_DIR" PYTHONUNBUFFERED=1 MPLBACKEND=Agg
cd "$SLURM_SUBMIT_DIR"
A=scripts/analysis
ROOTS="${ROOTS:-results_ep1 results_ep3 results_ep5 results_ep50}"

for s in compute_shallow_paper.py paper_fig_forgetting_vs_buffer.py paper_fig_deep_by_task.py; do
  [ -f "$A/$s" ] || { echo "MISSING: $A/$s"; exit 1; }
done
echo "roots: $ROOTS"

echo; echo "===== 1/3 F_shallow (paper, Eq. 1) ====="
python $A/compute_shallow_paper.py $ROOTS --outdir analysis_out_shallow || echo "FAILED: compute_shallow_paper"

echo; echo "===== 2/3 Fig. 1: forgetting vs buffer ====="
python $A/paper_fig_forgetting_vs_buffer.py $ROOTS --out paper_figs/forgetting_vs_buffer || echo "FAILED: fig 1"

echo; echo "===== 3/3 Fig. 2: deep forgetting by task ====="
python $A/paper_fig_deep_by_task.py $ROOTS --out paper_figs/deep_by_task || echo "FAILED: fig 2"

echo; echo "===== F_shallow summary ====="
cat analysis_out_shallow/shallow_paper_summary.csv
echo; echo "ALL DONE"
