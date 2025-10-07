# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash
set -e
source eval_env_checks.sh
check_vllm

MODEL_ID_ARRAY=()
BATCH_SIZE_ARRAY=(1)  # default can be overwritten by user input
INPUT_LEN="256"      # default input length
OUTPUT_LEN="256"     # default output length
# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_ids)
      shift
      # Collect all subsequent arguments that are not another flag
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        MODEL_ID_ARRAY+=("$1")
        shift
      done
      ;;
    --batch_sizes)
      shift
      BATCH_SIZE_ARRAY=()
      # Collect all subsequent arguments that are not another flag
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        BATCH_SIZE_ARRAY+=("$1")
        shift
      done
      ;;
    --input_len)
      shift
      if [[ $# -eq 0 ]]; then
        echo "Error: --input_len requires a value"
        exit 1
      fi
      INPUT_LEN="$1"
      shift
      ;;
    --output_len)
      shift
      if [[ $# -eq 0 ]]; then
        echo "Error: --output_len requires a value"
        exit 1
      fi
      OUTPUT_LEN="$1"
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 --model_id <model_id> [--batch_sizes <batch_sizes>] [--input_len <input_len>] [--output_len <output_len>]"
      exit 1
      ;;
  esac
done
if [[ ${#MODEL_ID_ARRAY[@]} -eq 0 ]]; then
  echo "Error: --model_ids is required"
  echo "Usage: $0 --model_ids <model_id1> <model_id2> ... [--batch_sizes <batch_size1> <batch_size2> ...] [--input_len <input_len>] [--output_len <output_len>]"
  exit 1
fi
# Save the original directory
ORIG_DIR="$(pwd)"
# cd to VLLM_DIR
cd $VLLM_DIR
for MODEL_ID in "${MODEL_ID_ARRAY[@]}"; do
    echo "======================== Eval Latency $MODEL_ID ==========================="
    # Replace all '/' with '_'
    SAFE_MODEL_ID="${MODEL_ID//\//_}"
    # Loop over batch sizes and print (replace with your eval command)
    for BATCH_SIZE in "${BATCH_SIZE_ARRAY[@]}"; do
        OUTPUT_FILE="$ORIG_DIR/${SAFE_MODEL_ID}_latency_batch${BATCH_SIZE}_in${INPUT_LEN}_out${OUTPUT_LEN}.log"
        echo "Running latency eval for model $MODEL_ID with batch size $BATCH_SIZE with input length: $INPUT_LEN and output length: $OUTPUT_LEN"
        VLLM_DISABLE_COMPILE_CACHE=1 vllm bench latency --input-len $INPUT_LEN --output-len $OUTPUT_LEN --model $MODEL_ID --batch-size $BATCH_SIZE > "$OUTPUT_FILE" 2>&1
        echo "Latency eval result saved to $OUTPUT_FILE"
    done
    echo "======================== Eval Latency $MODEL_ID End ========================="
done

# cd back to original place
cd $ORIG_DIR
