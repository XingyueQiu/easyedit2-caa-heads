#!/bin/bash
#SBATCH --job-name=caa_00_train
#SBATCH --account=bgxm-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

PROJECT=/home/xqiu7/easyedit2-caa-heads
CONFIG=${1:-code/configs/config_toxicity-qwen15.yaml}

echo "=============================="
echo "JOB: 00_train_vectors"
echo "CONFIG: $CONFIG"
echo "NODE: $(hostname)"
echo "STARTED: $(date)"
echo "=============================="

module load pytorch-conda/2.8
conda activate base

cd $PROJECT
export CUDA_VISIBLE_DEVICES=0

python code/scripts/00_train_vectors.py \
    --config $CONFIG \
    --skip-existing

echo "FINISHED: $(date)"
