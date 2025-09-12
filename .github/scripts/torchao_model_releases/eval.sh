# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash
set -e
source eval_env_checks.sh

usage() {
  echo "Usage: $0 --eval_type <all|memory|latency|quality> --model_ids <model1> <model2> ... [--batch_sizes <batch_sizes>] [--tasks <tasks>]"
  echo "Defaults:"
  echo "  batch_sizes: 1 256"
  echo "  tasks: mmlu"
  exit 1
}
EVAL_TYPE=""
MODEL_ID_ARRAY=()
# these will be parsed in the other scripts
BATCH_SIZES="1 256"    # Default for latency eval
TASKS="mmlu"           # Default for quality eval
# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --eval_type)
      shift
      if [[ $# -eq 0 ]]; then
        echo "Error: --eval_type requires a value"
        exit 1
      fi
      EVAL_TYPE="$1"
      shift
      ;;
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
      if [[ $# -eq 0 ]]; then
        echo "Error: --batch_sizes requires a value"
        exit 1
      fi
      BATCH_SIZES="$1"
      shift
      ;;
    --tasks)
      shift
      if [[ $# -eq 0 ]]; then
        echo "Error: --tasks requires a value"
        exit 1
      fi
      TASKS="$1"
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      ;;
  esac
done
if [[ -z "$EVAL_TYPE" || ${#MODEL_ID_ARRAY[@]} -eq 0 ]]; then
  echo "Error: --eval_type and --model_ids are required"
  usage
fi

run_memory() {
  check_torch
  local model_id="$1"
  sh eval_memory.sh --model_ids "$model_id"
}
run_latency() {
  check_vllm
  local model_id="$1"
  sh eval_latency.sh --model_ids "$model_id" --batch_sizes $BATCH_SIZES
}
run_quality() {
  check_lm_eval
  local model_id="$1"
  sh eval_quality.sh --model_ids "$model_id" --tasks $TASKS
}
for MODEL_ID in "${MODEL_ID_ARRAY[@]}"; do
  case "$EVAL_TYPE" in
    memory)
      run_memory "$MODEL_ID"
      ;;
    latency)
      run_latency "$MODEL_ID"
      ;;
    quality)
      run_quality "$MODEL_ID"
      ;;
    all)
      run_memory "$MODEL_ID"
      run_latency "$MODEL_ID"
      run_quality "$MODEL_ID"
      ;;
    *)
      echo "Unknown eval_type: $EVAL_TYPE"
      echo "Valid types are: all, memory, latency, quality"
      exit 2
      ;;
  esac
done

# Run summarize_results.sh with MODEL_IDS if eval_type is "all"
if [[ "$EVAL_TYPE" == "all" ]]; then
  sh summarize_results.sh --model_ids "${MODEL_ID_ARRAY[@]}"
fi
