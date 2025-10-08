# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash
set -e
usage() {
  echo "Usage: $0 --model_ids <model_id1> <model_id2> ..."
  exit 1
}
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
      usage
      ;;
  esac
done
if [[ ${#MODEL_ID_ARRAY[@]} -eq 0 ]]; then
  echo "Error: --model_ids is required"
  usage
  exit 1
fi
for MODEL_ID in "${MODEL_ID_ARRAY[@]}"; do
   SAFE_MODEL_ID="${MODEL_ID//\//_}"
   OUTPUT_FILE="summary_results_${SAFE_MODEL_ID}.log"
   # Clear or create the output file
   > "$OUTPUT_FILE"

   {
    echo "===== Summary for model: $MODEL_ID ====="
    QUALITY_LOG_PATTERN="${SAFE_MODEL_ID}_quality_*.log"
    # Quality logs (multiple files, one per task)
    QUALITY_LOGS=( $QUALITY_LOG_PATTERN )
    if [ -e "${QUALITY_LOGS[0]}" ]; then
        for Q_LOG in "${QUALITY_LOGS[@]}"; do
            # find last appearance of pretrained={MODEL_ID} and
            # extract all lines after that
            PATTERN="pretrained=${MODEL_ID}"
            LAST_LINE=$(grep -n "$PATTERN" "$Q_LOG" | tail -1 | cut -d: -f1)
            if [ -n "$LAST_LINE" ]; then
                echo "--- Quality log: $Q_LOG (lines starting from $((LAST_LINE))) ---"
                tail -n +"$((LAST_LINE))" "$Q_LOG"
            else
                echo "Pattern not found in $Q_LOG"
            fi
      done
    else
      echo "--- No quality logs found matching pattern: $QUALITY_LOG_PATTERN"
    fi

    MM_QUALITY_LOG_PATTERN="${SAFE_MODEL_ID}_mm_quality_*.log"
    # Multi-modal Quality logs (multiple files, one per task)
    MM_QUALITY_LOGS=( $MM_QUALITY_LOG_PATTERN )
    if [ -e "${MM_QUALITY_LOGS[0]}" ]; then
        for Q_LOG in "${MM_QUALITY_LOGS[@]}"; do
            # find last appearance of pretrained={MODEL_ID} and
            # extract all lines after that
            PATTERN="pretrained=${MODEL_ID}"
            LAST_LINE=$(grep -n "$PATTERN" "$Q_LOG" | tail -1 | cut -d: -f1)
            if [ -n "$LAST_LINE" ]; then
                echo "--- Multi-modal Quality log: $Q_LOG (lines starting from $((LAST_LINE))) ---"
                tail -n +"$((LAST_LINE))" "$Q_LOG"
            else
                echo "Pattern not found in $Q_LOG"
            fi
      done
    else
      echo "--- No quality logs found matching pattern: $MM_QUALITY_LOG_PATTERN"
    fi

    MEMORY_LOG="${SAFE_MODEL_ID}_memory.log"
    if [ -f "$MEMORY_LOG" ]; then
      echo "--- Memory log (last 1 lines) ---"
      tail -n 1 "$MEMORY_LOG"
    else
      echo "--- Memory log not found: $MEMORY_LOG"
    fi

    LATENCY_LOG_PATTERN="${SAFE_MODEL_ID}_latency_batch*_in*_out*.log"
    LATENCY_LOGS=( $LATENCY_LOG_PATTERN )
    if [ -e "${LATENCY_LOGS[0]}" ]; then
      for LAT_LOG in "${LATENCY_LOGS[@]}"; do
        echo "--- Latency log: $LAT_LOG (last 7 lines) ---"
        tail -n 7 "$LAT_LOG"
      done
    else
      echo "--- No latency logs found matching pattern: $LATENCY_LOG_PATTERN"
    fi
    echo ""
    echo "===== End of Summary for model: $MODEL_ID ====="
  } >> "$OUTPUT_FILE"
  echo "Summary of results saved to $OUTPUT_FILE"
done
