# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash
set -e
source eval_env_checks.sh
check_lm_eval

MODEL_ID_ARRAY=()
TASK_ARRAY=("mmlu")  # default can be overwritten by user input
USE_CACHE=false      # default: do not use cache
# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_ids)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        MODEL_ID_ARRAY+=("$1")
        shift
      done
      ;;
    --tasks)
      shift
      TASK_ARRAY=()
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        TASK_ARRAY+=("$1")
        shift
      done
      ;;
    --use_cache)
      USE_CACHE=true
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 --model_id <model_id> [--tasks <tasks> (comma-separated, e.g. mmlu,arc_challenge, default mmlu)] [--use_cache]"
      exit 1
      ;;
  esac
done
if [[ ${#MODEL_ID_ARRAY[@]} -eq 0 ]]; then
  echo "Error: --model_ids is required"
  echo "Usage: $0 --model_ids <model_id1> <model_id2> ... [--tasks <task1> <task2> ...]"
  exit 1
fi
RESULTS_DIR="$(pwd)/quality_eval_results"
for MODEL_ID in "${MODEL_ID_ARRAY[@]}"; do
    # Replace all '/' with '_'
    SAFE_MODEL_ID="${MODEL_ID//\//_}"
    echo "======================== Eval Model Quality $MODLE_ID ======================"
    for TASK in "${TASK_ARRAY[@]}"; do
        OUTPUT_FILE="$(pwd)/${SAFE_MODEL_ID}_quality_${TASK}.log"
        EVAL_CACHE_DB_PREFIX="/tmp/${SAFE_MODEL_ID}_quality_${TASK}"
        mkdir -p "${EVAL_CACHE_DB_PREFIX}"
        echo "Running model quality (accuracy) evaluation for model $MODEL_ID on task $TASK"
        LM_EVAL_CMD="lm_eval \
            --model hf \
            --model_args pretrained=\"$MODEL_ID\" \
            --tasks \"$TASK\" \
            --device cuda:0 \
            --batch_size auto \
            --output_path \"$RESULTS_DIR\""

        if $USE_CACHE; then
            LM_EVAL_CMD="$LM_EVAL_CMD --use_cache \"$EVAL_CACHE_DB_PREFIX\""
        fi

        eval "$LM_EVAL_CMD" > "$OUTPUT_FILE" 2>&1
        echo "Quality eval output for task '$TASK' saved to $OUTPUT_FILE"
    done
    echo "======================== Eval Model Quality $MODEL_ID End =================="
done
