#!/bin/bash
#SBATCH --job-name=caa_03_baselines
#SBATCH --account=bgxm-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail

PROJECT=/home/xqiu7/easyedit2-caa-heads
LAYER=$SLURM_ARRAY_TASK_ID
CONFIG=${CONFIG:-code/configs/config_toxicity-qwen15.yaml}
EVAL=${EVAL:-toxigen}

echo "=============================="
echo "JOB: 03_baselines  layer=$LAYER"
echo "CONFIG: $CONFIG"
echo "NODE: $(hostname)"
echo "STARTED: $(date)"
echo "=============================="

module load pytorch-conda/2.8
conda activate base

cd $PROJECT
export CUDA_VISIBLE_DEVICES=0

python code/scripts/03_baselines.py \
    --layer $LAYER \
    --config $CONFIG \
    --eval $EVAL

echo "FINISHED: $(date)"
