# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/eval_env_checks.sh"
check_lmms_eval

usage() {
  echo "Usage: $0 --model_id <model_id> --model_type <model_type> [--tasks <tasks> (comma-separated, e.g. mmlu,arc_challenge, default mmlu)] [--use_cache]"
  exit 1
}

MODEL_ID_ARRAY=()
MODEL_TYPE=""
TASK_ARRAY=("chartqa")  # default can be overwritten by user input
BATCH_SIZE=1
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
    --model_type)
      MODEL_TYPE="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
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
      usage
      exit 1
      ;;
  esac
done
if [[ ${#MODEL_ID_ARRAY[@]} -eq 0 ]]; then
  echo "Error: --model_ids is required"
  usage
  exit 1
fi
if [[ -z "$MODEL_TYPE" ]]; then
  echo "Error: --model_type is required"
  usage
  exit 1
fi
RESULTS_DIR="$(pwd)/mm_quality_eval_results"
for MODEL_ID in "${MODEL_ID_ARRAY[@]}"; do
    # Replace all '/' with '_'
    SAFE_MODEL_ID="${MODEL_ID//\//_}"
    echo "======================== Eval Multi-modal Model Quality $MODLE_ID ======================"
    for TASK in "${TASK_ARRAY[@]}"; do
        OUTPUT_FILE="$(pwd)/${SAFE_MODEL_ID}_mm_quality_${TASK}.log"
        EVAL_CACHE_DB_PREFIX="/tmp/${SAFE_MODEL_ID}_mm_quality_${TASK}"
        mkdir -p "${EVAL_CACHE_DB_PREFIX}"
        echo "Running multi-modal model quality (accuracy) evaluation for model $MODEL_ID on task $TASK"

        MAIN_PORT=12356
        LMMS_EVAL_CMD="accelerate launch \
           --main_process_port \"$MAIN_PORT\" \
           -m lmms_eval \
           --model \"$MODEL_TYPE\" \
           --model_args \"pretrained=$MODEL_ID\" \
           --tasks \"$TASK\" \
          --batch_size \"$BATCH_SIZE\" \
          --output_path \"$RESULTS_DIR\""

        if $USE_CACHE; then
            LMMS_EVAL_CMD="$LMMS_EVAL_CMD --use_cache \"$EVAL_CACHE_DB_PREFIX\""
        fi

        eval "$LMMS_EVAL_CMD" > "$OUTPUT_FILE" 2>&1
        echo "Quality eval output for task '$TASK' saved to $OUTPUT_FILE"
    done
    echo "======================== Eval Model Quality $MODEL_ID End =================="
done
