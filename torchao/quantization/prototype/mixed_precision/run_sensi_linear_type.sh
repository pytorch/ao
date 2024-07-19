#!/bin/bash

CONFIGS1=("q_proj" "k_proj" "v_proj" "o_proj" "gate_proj" "up_proj" "down_proj")
CONFIGS2=("2" "3" "4" "5" "6" "8")

PYTHON_SCRIPT="scripts/sensitivity_study.py"

GPUS=(0 1 2 3 4 5)

for i in "${!CONFIGS1[@]}"; do
  CONFIG1="${CONFIGS1[$i]}"

  for j in "${!CONFIGS2[@]}"; do
    CONFIG2="${CONFIGS2[$j]}"
    GPU="${GPUS[$j]}"

    LOG_FILE="Sensi_skipsensi_${CONFIG1}_${CONFIG2}.txt"

    CUDA_VISIBLE_DEVICES=$GPU python $PYTHON_SCRIPT --repo_id=checkpoints/meta-llama/Meta-Llama-3-8B --quantization=$CONFIG2 --linear_type=$CONFIG1 &>"$LOG_FILE" &
  done

  wait
done


echo "All processes are complete."
