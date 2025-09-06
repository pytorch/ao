# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash
set -e
source eval_env_checks.sh
check_torch
MODEL_ID_ARRAY=()
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
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 --model_ids <model_id1> <model_id2> ..."
      exit 1
      ;;
  esac
done
if [[ ${#MODEL_ID_ARRAY[@]} -eq 0 ]]; then
  echo "Usage: $0 --model_ids <model_id1> <model_id2> ..."
  exit 1
fi
for MODEL_ID in "${MODEL_ID_ARRAY[@]}"; do
    # Replace all '/' with '_'
    SAFE_MODEL_ID="${MODEL_ID//\//_}"
    OUTPUT_FILE="$(pwd)/${SAFE_MODEL_ID}_memory.log"
    echo "======================== Eval Memory $MODEL_ID ============================"
    python eval_peak_memory_usage.py --model_id "$MODEL_ID" > "$OUTPUT_FILE" 2>&1
    echo "Evaluation complete. Output saved to $OUTPUT_FILE"
    echo "======================== Eval Memory $MODEL_ID End ========================"
done
