#!/bin/bash

# List of GPUs to use
GPUS=(2 3 4 5 6 7)

# List of configuration files
#CONFIGS1=("int6wo" "int5wo" "int2wo" "int3wo" "int8wo" "int4wo" "None" "autoquant")
CONFIGS1=("2" "3" "4" "5" "6" "8")
#CONFIGS1=(8 8 8 8 8 4 4 16)
#CONFIGS2=(6 5 4 3 2 3 2 8)

#CONFIGS1=(16 16 16 16 16)
#CONFIGS2=(6 5 4 3 2)

PYTHON_SCRIPT="scripts/mx_eval.py"

for i in "${!GPUS[@]}"; do
  GPU="${GPUS[$i]}"
  CONFIG1="${CONFIGS1[$i]}"

  LOG_FILE="UNI_${CONFIG1}_SYM.txt"

  CUDA_VISIBLE_DEVICES=$GPU python $PYTHON_SCRIPT --repo_id=checkpoints/meta-llama/Meta-Llama-3-8B --quantization=$CONFIG1 --quant_sym=sym &>"$LOG_FILE" &
done

wait

for i in "${!GPUS[@]}"; do
  GPU="${GPUS[$i]}"
  CONFIG1="${CONFIGS1[$i]}"

  LOG_FILE="UNI_${CONFIG1}_ASYM.txt"

  CUDA_VISIBLE_DEVICES=$GPU python $PYTHON_SCRIPT --repo_id=checkpoints/meta-llama/Meta-Llama-3-8B --quantization=$CONFIG1 --quant_sym=asym &>"$LOG_FILE" &
done

wait

echo "All processes are complete."
