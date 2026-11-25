#!/bin/bash
# One-shot launcher for a new epoch setting of the TIL epoch sweep:
#   1. training        slurm_epochs_sweep.sbatch (same protocol as the 1/3/5-epoch runs)
#   2. per-run analyses, each starting as soon as ITS training run finishes:
#        LP (SGDClassifier)  slurm_lp_sgd.sbatch  -> lp_curve_sgd.csv
#        Neural Collapse     slurm_nc.sbatch      -> nc_curve.csv
#        SNR                 slurm_snr.sbatch     -> snr_curve.csv
#   3. tables + figures      slurm_epochs_report.sbatch, over results_ep1/3/5 + the new root
#
# Run ON THE LOGIN NODE:            bash scripts/hpc/launch_epochs_pipeline.sh

set -euo pipefail

# ----------------------------------------------------------------- settings ---
EPOCHS="${EPOCHS:-50}"
# strategy:mem_size, spelled exactly as in CONFIGS of slurm_epochs_sweep.sbatch
WANTED=(${WANTED:-naive:0 er:200 er:1000 er:10000 er:50000})
SEEDS_WANTED=(${SEEDS_WANTED:-0 1 2})

# Wall-clock limits: the 5-epoch runs used 12h (train) and 2h (each analysis);
# 50 epochs = 10x the steps. sbatch refuses a limit above the partition maximum
# right away, and the script then stops before submitting anything else.
TRAIN_TIME="${TRAIN_TIME:-5-00:00:00}"
LPSGD_TIME="${LPSGD_TIME:-20:00:00}"
NC_TIME="${NC_TIME:-20:00:00}"
SNR_TIME="${SNR_TIME:-20:00:00}"

# Checkpoint strides: same as for the earlier settings, so the tables stay comparable
LPSGD_STRIDE="${LPSGD_STRIDE:-1}"
NC_STRIDE="${NC_STRIDE:-5}"
SNR_STRIDE="${SNR_STRIDE:-5}"

# Result roots that go into the tables/figures (results_ep20 deliberately left out)
REPORT_ROOTS="${REPORT_ROOTS:-results_ep1 results_ep3 results_ep5 results_ep${EPOCHS}}"

DRY_RUN="${DRY_RUN:-0}"
YES="${YES:-0}"
# ------------------------------------------------------------------------------

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"                       # sbatch is submitted from here -> SLURM_SUBMIT_DIR
H=scripts/hpc
SWEEP=$H/slurm_epochs_sweep.sbatch
mkdir -p logs

# CONFIGS/SEEDS are read from the training script itself, so the array indices
# can never drift from what the job actually runs.
eval "$(grep -E '^(CONFIGS|SEEDS)=' "$SWEEP")"
NCFG=${#CONFIGS[@]}

dir_name() {   # strategy:mem -> run folder written by experiments/main.py
  local strat=${1%%:*} mem=${1##*:}
  [ "$strat" = naive ] && mem=2000   # naive keeps the experiment default mem_size
  echo "${strat}_split_cifar100_10_${mem}"
}

INDICES=(); RUNS=(); NAMES=()
for cfg in "${WANTED[@]}"; do
  ci=-1
  for i in "${!CONFIGS[@]}"; do [ "${CONFIGS[$i]}" = "$cfg" ] && ci=$i; done
  [ $ci -ge 0 ] || { echo "ERROR: '$cfg' not in CONFIGS of $SWEEP: ${CONFIGS[*]}"; exit 1; }
  NAMES+=("$(dir_name "$cfg")")
  for s in "${SEEDS_WANTED[@]}"; do
    si=-1
    for j in "${!SEEDS[@]}"; do [ "${SEEDS[$j]}" = "$s" ] && si=$j; done
    [ $si -ge 0 ] || { echo "ERROR: seed $s not in SEEDS of $SWEEP: ${SEEDS[*]}"; exit 1; }
    INDICES+=($((ci + NCFG * si)))
    RUNS+=("results_ep${EPOCHS}/$(dir_name "$cfg")/$s")
  done
done
ARRAY=$(IFS=,; echo "${INDICES[*]}")

echo "== epochs=$EPOCHS  runs=${#RUNS[@]}  array=$ARRAY"
for k in "${!RUNS[@]}"; do echo "   [${INDICES[$k]}] ${RUNS[$k]}"; done
echo "== report roots: $REPORT_ROOTS"

# ------------------------------------------------------------ sanity checks ---
if [ -d "results_ep${EPOCHS}" ] && [ -n "$(find "results_ep${EPOCHS}" -name config.yaml -print -quit)" ]; then
  echo "WARNING: results_ep${EPOCHS} already contains runs; they will be overwritten."
fi
for ref in results_ep5/${NAMES[0]}/0/checkpoints results_ep5/*/0/checkpoints; do
  if [ -d "$ref" ]; then       # checkpoints scale linearly with the epochs
    kb=$(du -sk "$ref" | cut -f1)
    est=$(( kb * EPOCHS / 5 * ${#RUNS[@]} / 1024 / 1024 ))
    echo "== disk: $ref = $((kb/1024)) MB -> ~${est} GB of checkpoints for ${#RUNS[@]} runs at $EPOCHS epochs"
    break
  fi
done
if [ "$DRY_RUN" != 1 ] && [ "$YES" != 1 ]; then
  read -r -p "Submit? [y/N] " ok; [ "$ok" = y ] || [ "$ok" = Y ] || { echo "aborted"; exit 0; }
fi

submit() {   # prints the job id (a fake one in dry-run)
  if [ "$DRY_RUN" = 1 ]; then
    echo "sbatch $*" >&2; echo "DRY$RANDOM"
  else
    sbatch --parsable "$@" | cut -d';' -f1
  fi
}

# ------------------------------------------------------------------ 1. train ---
TRAIN=$(submit --array="$ARRAY" --time="$TRAIN_TIME" \
        --export=ALL,EPOCHS="$EPOCHS" "$SWEEP")
echo "== training array: $TRAIN"

# ------------------------------------------------------ 2. per-run analyses ---
ANALYSIS=()
for k in "${!RUNS[@]}"; do
  run=${RUNS[$k]}
  # afterok on THIS run's array task; if that training fails, its analyses are
  # cancelled (kill-on-invalid-dep) instead of waiting in the queue forever
  dep=(--dependency="afterok:${TRAIN}_${INDICES[$k]}" --kill-on-invalid-dep=yes --exclude=dellcuda1)
  ANALYSIS+=("$(submit "${dep[@]}" --time="$LPSGD_TIME" --export=ALL,RUN="$run",STRIDE="$LPSGD_STRIDE" $H/slurm_lp_sgd.sbatch)")
  ANALYSIS+=("$(submit "${dep[@]}" --time="$NC_TIME"    --export=ALL,RUN="$run",STRIDE="$NC_STRIDE"    $H/slurm_nc.sbatch)")
  ANALYSIS+=("$(submit "${dep[@]}" --time="$SNR_TIME"   --export=ALL,RUN="$run",STRIDE="$SNR_STRIDE"   $H/slurm_snr.sbatch)")
done
echo "== analysis jobs: ${#ANALYSIS[@]}"

# ---------------------------------------------------- 3. tables + figures ---
# afterany: the report runs even if an analysis failed (the scripts skip what is missing)
DEP="afterany:$(IFS=:; echo "${ANALYSIS[*]}")"
REPORT=$(submit --dependency="$DEP" \
         --export=ALL,EPOCHS="$EPOCHS",RUN_NAMES="${NAMES[*]}",REPORT_ROOTS="$REPORT_ROOTS" \
         $H/slurm_epochs_report.sbatch)
echo "== report job: $REPORT  -> analysis_out_ep${EPOCHS}/"

echo
echo "Monitor with:  squeue -u \$USER"
echo "Cancel all:    scancel $TRAIN ${ANALYSIS[*]} $REPORT"
