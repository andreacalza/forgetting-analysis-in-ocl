#!/bin/bash
#SBATCH --job-name=ocl_er
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:TESLA-T4-16GB:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=allgroups

set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocl_survey

# buffer size: default 2000, override con --export=ALL,MEM_SIZE=<valore>
MEM_SIZE=${MEM_SIZE:-2000}

export PYTHONPATH="$SLURM_SUBMIT_DIR"
export HYDRA_FULL_ERROR=1
cd "$SLURM_SUBMIT_DIR/experiments"

echo "er mem_size=$MEM_SIZE on $(hostname)"
python -c "import torch; print('GPU available:', torch.cuda.is_available())"

python main.py strategy=er experiment=split_cifar100_til evaluation=anytime \
  strategy.device=cuda strategy.mem_size=$MEM_SIZE \
  benchmark.dataset_root=$HOME/.avalanche/data/cifar100 \
  evaluation.num_gpus=1 evaluation.parallel_evaluation=false \
  experiment.debug=false experiment.save_checkpoints=true
