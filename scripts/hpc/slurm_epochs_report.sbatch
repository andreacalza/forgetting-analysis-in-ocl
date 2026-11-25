#!/bin/bash
#SBATCH --job-name=ocl_report
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=allgroups

# Final step of launch_epochs_pipeline.sh: tables + figures over REPORT_ROOTS
# (default results_ep1/3/5 + the new setting), so the new setting sits next to the
# old ones. Reads only the per-run CSVs and never deletes them. Deep forgetting and
# LP use the SGD probe (lp_curve_sgd.csv) for every epoch setting.
#
# Env (set by the launcher):
#   EPOCHS        the new epoch setting (e.g. 50)
#   RUN_NAMES     space-separated run folder names (one NC figure each)
#   REPORT_ROOTS  space-separated result roots
#   OUTDIR        output folder (default analysis_out_ep$EPOCHS)

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocl_survey || { echo "conda activate failed"; exit 1; }
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg

REPO="$SLURM_SUBMIT_DIR"
export PYTHONPATH="$REPO"
cd "$REPO"
A="$REPO/scripts/analysis"

: "${EPOCHS:?set EPOCHS}"
REPORT_ROOTS="${REPORT_ROOTS:-results_ep1 results_ep3 results_ep5 results_ep${EPOCHS}}"
OUTDIR="${OUTDIR:-analysis_out_ep${EPOCHS}}"
OUT="$REPO/$OUTDIR"
mkdir -p "$OUT/forgetting" "$OUT/nc_tables" "$OUT/nc_figures" "$OUT/heatmaps"

ROOTS=()
for r in $REPORT_ROOTS; do
  if [ -d "$r" ] && [ -n "$(find "$r" -name config.yaml -print -quit)" ]; then ROOTS+=("$r")
  else echo "WARNING: $r missing or empty, left out"; fi
done
ABS_ROOTS=("${ROOTS[@]/#/$REPO/}")
echo "roots: ${ROOTS[*]}"
echo "output: $OUT"

step() { echo; echo "===================== $* ====================="; }

step "1/5 master table + cross-epoch figures (analyze_epochs.py, SGD probe)"
python "$A/analyze_epochs.py" "${ROOTS[@]}" --outdir "$OUT" --probe sgd \
  || echo "FAILED: analyze_epochs"

step "2/5 per-task shallow/deep forgetting tables (forgetting_by_epochs.py, SGD probe)"
( cd "$OUT/forgetting" && python "$A/forgetting_by_epochs.py" "${ABS_ROOTS[@]}" --probe-file lp_curve_sgd.csv ) \
  || echo "FAILED: forgetting_by_epochs"

step "3/5 Neural Collapse tables (make_nc_tables.py)"
python "$A/make_nc_tables.py" "${ROOTS[@]}" --format csv --outdir "$OUT/nc_tables" \
  || echo "FAILED: make_nc_tables csv"
python "$A/make_nc_tables.py" "${ROOTS[@]}" --format latex --out "$OUT/nc_tables/nc_tables.tex" \
  || echo "FAILED: make_nc_tables latex"

step "4/5 Neural Collapse figures, one row per epoch setting (plot_nc.py --paper-grid)"
for name in $RUN_NAMES; do
  runs=()
  for r in "${ROOTS[@]}"; do
    [ -f "$r/$name/0/nc_curve.csv" ] && runs+=("$r/$name/0")
  done
  if [ ${#runs[@]} -eq 0 ]; then echo "skip $name: no nc_curve.csv"; continue; fi
  echo "$name: ${runs[*]}"
  python "$A/plot_nc.py" "${runs[@]}" --paper-grid \
    --out "$OUT/nc_figures/nc_paper_grid_${name}.png" || echo "FAILED: plot_nc $name"
done

step "5/5 per-task heatmaps of the new runs (plot_task_heatmap.py)"
for d in results_ep${EPOCHS}/*/*/; do
  d="${d%/}"
  [ -f "$d/config.yaml" ] || continue
  tag="$(basename "$(dirname "$d")")_seed$(basename "$d")"
  python "$A/plot_task_heatmap.py" "$d" --out "$OUT/heatmaps/${tag}.png" \
    || echo "FAILED: heatmap $d"
done

echo; echo "REPORT DONE -> $OUT"
