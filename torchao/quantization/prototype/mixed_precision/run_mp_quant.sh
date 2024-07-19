#!/bin/bash

# List of GPUs to use
GPUS=(0 1 2 3 4 5)

# List of configuration files
CONFIGS1=(8 8 8 8 8 16)
CONFIGS2=(6 5 4 3 2 8)

#CONFIGS1=(16 16 16 16 16)
#CONFIGS2=(6 5 4 3 2)

#CONFIGS1=(5 5 5 6 6 6 6 3)
#CONFIGS2=(4 3 2 5 4 3 2 2)

PYTHON_SCRIPT="scripts/mp_quant_eval.py"

for i in "${!GPUS[@]}"; do
  GPU="${GPUS[$i]}"
  CONFIG1="${CONFIGS1[$i]}"
  CONFIG2="${CONFIGS2[$i]}"

  LOG_FILE="MP_${CONFIG1}_${CONFIG2}.txt"

  CUDA_VISIBLE_DEVICES=$GPU python $PYTHON_SCRIPT --repo_id=checkpoints/meta-llama/Meta-Llama-3-8B --quantization=MP_llama3 --sensi_bit=$CONFIG1 --non_sensi_bit=$CONFIG2 &>"$LOG_FILE" &
done

wait

echo "All processes are complete."
