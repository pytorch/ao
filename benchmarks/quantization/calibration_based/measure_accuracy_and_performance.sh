# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash

# measure_accuracy_and_performance.sh - Evaluate calibration-based quantization recipes (AWQ, SmoothQuant)
#
# Usage: ./measure_accuracy_and_performance.sh [RECIPE] [MODEL_ID] [LOG_FILE] [CALIB_TASKS] [CALIB_LIMIT]
#
# Arguments:
#   RECIPE         (optional) Recipe name or "all" (default: all)
#                  Valid recipes: awq_int4_weight_only, smoothquant_int8
#   MODEL_ID       (optional) HuggingFace model ID (default: meta-llama/Llama-3.1-8B)
#   LOG_FILE       (optional) Output log file path (default: benchmarks/data/calibration_accuracy_and_performance_log.txt)
#   CALIB_TASKS    (optional) Calibration tasks (default: wikitext)
#   CALIB_LIMIT    (optional) Calibration limit (default: 128)
#
# Environment Variables:
#   SKIP_MODEL_CREATE  If set to 1, skip creating quantized models (assumes models already exist)
#   SKIP_LM_EVAL       If set to 1, skip running lm_eval (only creates quantized models)
#
# Examples:
#   ./measure_accuracy_and_performance.sh                                # Run all calibration recipes with default model
#   ./measure_accuracy_and_performance.sh awq_int4_weight_only           # Run single recipe
#   ./measure_accuracy_and_performance.sh all meta-llama/Llama-3.2-1B    # Custom model with all recipes
#   ./measure_accuracy_and_performance.sh awq_int4_weight_only meta-llama/Llama-3.2-8B my_log.txt  # All custom args
#   SKIP_MODEL_CREATE=1 ./measure_accuracy_and_performance.sh all        # Skip model creation, only run eval
#   SKIP_LM_EVAL=1 ./measure_accuracy_and_performance.sh all             # Skip lm_eval, only create models

set -e

# Define all available calibration-based quantization recipes
QUANT_RECIPES_ALL=(
  "awq_int4_weight_only"
  "smoothquant_int8"
)

# Get recipe as first positional argument (optional, default: all)
RECIPE="${1:-all}"

# Get model_id as second positional argument (optional)
MODEL_ID="${2:-meta-llama/Llama-3.1-8B}"

# Get log file as third positional argument (optional)
LOG_FILE="${3:-benchmarks/data/calibration_based/accuracy_and_performance_log.txt}"

# Get calibration tasks as fourth positional argument (optional)
CALIB_TASKS="${4:-wikitext}"

# Get calibration limit as fifth positional argument (optional)
CALIB_LIMIT="${5:-128}"

# Select recipes based on argument
if [ "$RECIPE" = "all" ]; then
  QUANT_RECIPES=("${QUANT_RECIPES_ALL[@]}")
else
  # Check if it's a valid recipe name
  VALID_RECIPE=false
  for recipe in "${QUANT_RECIPES_ALL[@]}"; do
    if [ "$recipe" = "$RECIPE" ]; then
      VALID_RECIPE=true
      QUANT_RECIPES=("$RECIPE")
      break
    fi
  done

  if [ "$VALID_RECIPE" = false ]; then
    echo "Error: Invalid recipe name: '$RECIPE'"
    echo ""
    echo "Valid recipe names:"
    for recipe in "${QUANT_RECIPES_ALL[@]}"; do
      echo "  - $recipe"
    done
    exit 1
  fi
fi

mkdir -p "$(dirname "$LOG_FILE")"
rm -rf $LOG_FILE
touch $LOG_FILE

python -c "import torch; import torchao; print(f'{torch.__version__=}\n{torch.cuda.get_device_name()=}\n{torchao.__version__=}')" | tee -a "$LOG_FILE"

for quant_recipe in "${QUANT_RECIPES[@]}"; do

  echo | tee -a "$LOG_FILE"
  echo "processing quant_recipe $quant_recipe" | tee -a "$LOG_FILE"
  echo | tee -a "$LOG_FILE"

  OUTPUT_DIR="benchmarks/data/quantized_model/$MODEL_ID-$quant_recipe/"

  # create quantized model (unless skipped via environment variable)
  if [ "${SKIP_MODEL_CREATE:-0}" != "1" ]; then
    # Note: the -u flag is to prevent python from buffering stdout and stderr
    # and make the output log file be in chronological order
    rm -rf $OUTPUT_DIR

    python -u -m benchmarks.quantization.calibration_based.create_quantized_model \
      --model $MODEL_ID \
      --output_dir $OUTPUT_DIR \
      --recipe $quant_recipe \
      --calibration_tasks $CALIB_TASKS \
      --calibration_limit $CALIB_LIMIT \
      2>&1 | tee -a "$LOG_FILE"
  else
    echo "Skipping model creation (SKIP_MODEL_CREATE=1), using existing model at $OUTPUT_DIR" | tee -a "$LOG_FILE"
  fi

  # run eval (unless skipped via environment variable)
  if [ "${SKIP_LM_EVAL:-0}" != "1" ]; then
    lm_eval --model hf --model_args "pretrained=$OUTPUT_DIR" --tasks "wikitext,winogrande" --device "cuda:0" --batch_size 1 --output_path "$OUTPUT_DIR/lm_eval_outputs/" 2>&1 | tee -a "$LOG_FILE"
  else
    echo "Skipping lm_eval (SKIP_LM_EVAL=1)" | tee -a "$LOG_FILE"
  fi

done

benchmarks/quantization/parse_log.py $LOG_FILE
