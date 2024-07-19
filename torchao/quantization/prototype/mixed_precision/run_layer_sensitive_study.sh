#!/bin/bash

GPUS=(0 1 2 3 4 5)

CONFIGS1=("2" "3" "4" "5" "6" "8")

PYTHON_SCRIPT="scripts/sensitivity_study.py"

for LAYER in {0..31}; do
  for i in "${!GPUS[@]}"; do
    GPU="${GPUS[$i]}"
    CONFIG1="${CONFIGS1[$i]}"

    LOG_FILE="Sensi_${LAYER}_${CONFIG1}.txt"

    CUDA_VISIBLE_DEVICES=$GPU python $PYTHON_SCRIPT --repo_id=checkpoints/meta-llama/Meta-Llama-3-8B --quantization=$CONFIG1 --layer=$LAYER &>"$LOG_FILE" &
  done

  wait
done

echo "All processes are complete."
