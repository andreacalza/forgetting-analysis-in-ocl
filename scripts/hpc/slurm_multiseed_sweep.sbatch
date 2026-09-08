#!/bin/bash
#SBATCH --job-name=ocl_seed
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-23
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:TESLA-T4-16GB:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=allgroups

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocl_survey || { echo "conda activate failed"; exit 1; }
export PYTHONPATH="$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR/experiments"

CONFIGS=("naive:0" "er:200" "er:500" "er:1000" "er:2000" "er:5000" "er:10000" "er:50000")
SEEDS=(0 1 2)
NCFG=${#CONFIGS[@]}
CFG=${CONFIGS[$((SLURM_ARRAY_TASK_ID % NCFG))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / NCFG))]}
STRAT=${CFG%%:*}; MEM=${CFG##*:}
EXTRA=""; [ "$STRAT" = "er" ] && EXTRA="strategy.mem_size=$MEM"

echo "task=$SLURM_ARRAY_TASK_ID strategy=$STRAT mem=$MEM seed=$SEED on $(hostname)"
python main.py strategy=$STRAT experiment=split_cifar100_til evaluation=anytime \
  strategy.device=cuda $EXTRA experiment.seed=$SEED evaluation.eval_every=250 \
  benchmark.dataset_root=$HOME/.avalanche/data/cifar100 \
  evaluation.num_gpus=1 evaluation.parallel_evaluation=false \
  experiment.debug=false experiment.save_checkpoints=true
