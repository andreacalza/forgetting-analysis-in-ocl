#!/bin/bash
#SBATCH --job-name=ocl_naive
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:TESLA-T4-16GB:1
#SBATCH --account=thesis
#SBATCH --partition=allgroups

set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocl_survey

export PYTHONPATH="$SLURM_SUBMIT_DIR"
export HYDRA_FULL_ERROR=1
cd "$SLURM_SUBMIT_DIR/experiments"

echo "naive on $(hostname)"
python -c "import torch; print('GPU available:', torch.cuda.is_available())"

python main.py strategy=naive experiment=split_cifar100_til evaluation=anytime \
  strategy.device=cuda \
  benchmark.dataset_root=$HOME/.avalanche/data/cifar100 \
  evaluation.num_gpus=1 evaluation.parallel_evaluation=false \
  experiment.debug=false experiment.save_checkpoints=true
