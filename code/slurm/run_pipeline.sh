#!/bin/bash
# =============================================================================
# run_pipeline.sh — Master SLURM pipeline
# Usage: bash code/slurm/run_pipeline.sh qwen15
#        bash code/slurm/run_pipeline.sh qwen25
# =============================================================================

set -euo pipefail

MODEL=${1:-qwen15}
CONFIG="code/configs/config_toxicity-${MODEL}.yaml"
PROJECT=/home/xqiu7/easyedit2-caa-heads
SLURM_DIR=$PROJECT/code/slurm
EVAL=toxigen
N_STRATA=6
LAYERS_FILE=$PROJECT/code/results/.selected_layers_${MODEL}.txt

echo "=================================================="
echo "CAA PIPELINE: $MODEL"
echo "Config: $CONFIG"
echo "Started: $(date)"
echo "=================================================="

cd $PROJECT

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config not found: $CONFIG"
    exit 1
fi

mkdir -p code/results

# ── 00: Train vectors ─────────────────────────────────────────────────────────
JOB_00=$(sbatch --parsable $SLURM_DIR/00_train_vectors.sh $CONFIG)
echo "00_train_vectors: job $JOB_00"

# ── 01: Layer sweep ───────────────────────────────────────────────────────────
JOB_01=$(sbatch --parsable \
    --dependency=afterok:$JOB_00 \
    $SLURM_DIR/01_layer_sweep.sh $CONFIG)
echo "01_layer_sweep:   job $JOB_01  (after $JOB_00)"

# ── SELECT: Stratified layer selection (CPU job) ──────────────────────────────
JOB_SELECT=$(sbatch --parsable \
    --dependency=afterok:$JOB_01 \
    --job-name=caa_select_${MODEL} \
    --account=bgxm-delta-gpu \
    --partition=gpuA100x4 \
    --gres=gpu:0 \
    --cpus-per-task=2 \
    --mem=8G \
    --time=00:10:00 \
    --output=caa_select_${MODEL}_%j.out \
    --wrap="
        set -euo pipefail
        module load pytorch-conda/2.8
        conda activate base
        cd $PROJECT
        MODEL_NAME=\$(python -c \"
import yaml
with open('$CONFIG') as f: c = yaml.safe_load(f)
print(c.get('model_name_or_path','').split('/')[-1])
\")
        SWEEP=\$(ls code/results/\${MODEL_NAME}/toxicity/layer_sweep_*.json 2>/dev/null | sort | tail -1)
        if [ -z \"\$SWEEP\" ]; then
            echo 'ERROR: no layer_sweep JSON found'; exit 1
        fi
        echo \"Sweep file: \$SWEEP\"
        python code/scripts/stratified_select.py \\
            --sweep \"\$SWEEP\" \\
            --n-strata $N_STRATA \\
            --verbose > $LAYERS_FILE
        echo \"Selected layers: \$(cat $LAYERS_FILE)\"
    ")
echo "stratified_select: job $JOB_SELECT  (after $JOB_01)"

# ── COORD: Submit per-layer arrays ────────────────────────────────────────────
JOB_COORD=$(sbatch --parsable \
    --dependency=afterok:$JOB_SELECT \
    --job-name=caa_coord_${MODEL} \
    --account=bgxm-delta-gpu \
    --partition=gpuA100x4 \
    --gres=gpu:0 \
    --cpus-per-task=2 \
    --mem=8G \
    --time=00:15:00 \
    --output=caa_coord_${MODEL}_%j.out \
    --wrap="
        set -euo pipefail
        cd $PROJECT
        LAYERS=\$(cat $LAYERS_FILE)
        ARRAY_STR=\$(echo \$LAYERS | tr ' ' ',')
        echo \"Submitting arrays for layers: \$LAYERS\"

        JOB_03=\$(sbatch --parsable \\
            --array=\$ARRAY_STR \\
            --export=CONFIG=$CONFIG,EVAL=$EVAL \\
            $SLURM_DIR/03_baselines_array.sh)
        echo \"03_baselines array: \$JOB_03\"

        JOB_04=\$(sbatch --parsable \\
            --array=\$ARRAY_STR \\
            --dependency=afterok:\$JOB_03 \\
            --export=CONFIG=$CONFIG,EVAL=$EVAL \\
            $SLURM_DIR/04_greedy_array.sh)
        echo \"04_greedy array: \$JOB_04  (after \$JOB_03)\"

        JOB_05=\$(sbatch --parsable \\
            --array=\$ARRAY_STR \\
            --dependency=afterok:\$JOB_04 \\
            --export=CONFIG=$CONFIG,EVAL=$EVAL \\
            $SLURM_DIR/05_random_k_array.sh)
        echo \"05_random_k: \$JOB_05\"

        JOB_07=\$(sbatch --parsable \\
            --array=\$ARRAY_STR \\
            --dependency=afterok:\$JOB_04 \\
            --export=CONFIG=$CONFIG,EVAL=$EVAL \\
            $SLURM_DIR/07_per_input_array.sh)
        echo \"07_per_input: \$JOB_07\"

        JOB_08=\$(sbatch --parsable \\
            --array=\$ARRAY_STR \\
            --dependency=afterok:\$JOB_04 \\
            --export=CONFIG=$CONFIG,EVAL=$EVAL \\
            $SLURM_DIR/08_held_out_array.sh)
        echo \"08_held_out: \$JOB_08\"

        echo \"All arrays submitted. Monitor: squeue -u \$USER\"
    ")
echo "coordinator: job $JOB_COORD  (after $JOB_SELECT)"

echo ""
echo "=================================================="
echo "SUBMITTED: $MODEL"
echo "Chain: $JOB_00 -> $JOB_01 -> $JOB_SELECT -> $JOB_COORD -> arrays"
echo "Monitor: squeue -u \$USER"
echo "Cancel all: scancel $JOB_00 $JOB_01 $JOB_SELECT $JOB_COORD"
echo "=================================================="
