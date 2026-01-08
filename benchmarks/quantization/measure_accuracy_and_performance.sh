# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash

# measure_accuracy_and_performance.sh - Evaluate quantization recipe accuracy
#
# Usage: ./measure_accuracy_and_performance.sh [TAG_OR_RECIPE] [MODEL_ID] [LOG_FILE]
#
# Arguments:
#   TAG_OR_RECIPE  (optional) Tag group, single recipe name, or "all" (default: all)
#                  Valid tags: all, h100, b200
#                  Valid recipes: None, float8_rowwise,
#                                 int4_groupwise_weight_float8_rowwise_activation,
#                                 int4_groupwise_hqq_weight_only,
#                                 int8_rowwise_weight_only, int8_rowwise,
#                                 awq_int4_weight_only, smoothquant_int8
#   MODEL_ID       (optional) HuggingFace model ID (default: meta-llama/Llama-3.1-8B)
#   LOG_FILE       (optional) Output log file path (default: benchmarks/data/measure_accuracy_and_performance_log.txt)
#
# Environment Variables:
#   SKIP_MODEL_CREATE  If set to 1, skip creating quantized models (assumes models already exist)
#   SKIP_LM_EVAL       If set to 1, skip running lm_eval (only creates quantized models)
#   SKIP_VLLM          If set to 1, skip running vllm performance benchmarking
#
# Examples:
#   ./measure_accuracy_and_performance.sh                    # Run all recipes with default model
#   ./measure_accuracy_and_performance.sh h100               # Run H100-compatible recipes only
#   ./measure_accuracy_and_performance.sh float8_rowwise     # Run single recipe
#   ./measure_accuracy_and_performance.sh h100 meta-llama/Llama-3.2-8B  # Custom model with H100 recipes
#   ./measure_accuracy_and_performance.sh int8_rowwise meta-llama/Llama-3.2-8B my_log.txt  # All custom args
#   SKIP_MODEL_CREATE=1 ./measure_accuracy_and_performance.sh h100  # Skip model creation, only run eval
#   SKIP_LM_EVAL=1 ./measure_accuracy_and_performance.sh h100  # Skip lm_eval, only create models
#   SKIP_VLLM=1 ./measure_accuracy_and_performance.sh h100  # Skip vllm benchmarking

set -e

# Define all available quantization recipes
QUANT_RECIPES_ALL=(
  # no quantization (baseline)
  "None"
  "float8_rowwise"
  "int4_groupwise_weight_float8_rowwise_activation"
  # calibration-based quantization
  "awq_int4_weight_only"
  "smoothquant_int8"
  # note: below only works on A100
  "int4_groupwise_hqq_weight_only"
  "int8_rowwise_weight_only"
  "int8_rowwise"
  "mxfp8"
  "nvfp4"
)

# Define B200-compatible recipes
QUANT_RECIPES_B200=(
  "None"
  "mxfp8"
  "nvfp4"
  "float8_rowwise"
)

# Define H100-compatible recipes (excludes A100-only recipes)
# Note: the int8 ones work but performance is not ideal
# TODO(future PR): add `int4_groupwise_weight_float8_rowwise_activation` here,
#   need to fix https://gist.github.com/vkuzo/6b128681b628744d445c553cdeac8a85
QUANT_RECIPES_H100=(
  "None"
  "float8_rowwise"
  "int8_rowwise_weight_only"
  "int8_rowwise"
)

# Define recipes that are known to be broken in vllm
VLLM_BROKEN_RECIPES=(
  # TODO(future PR): fix this
  # error: https://gist.github.com/vkuzo/5bf389079442bb9851ef315cdcb797b4
  "int8_rowwise"
  # TODO(future PR): fix this
  # error: https://gist.github.com/vkuzo/b15ec478ee0a04d274ddb46acfa6d209
  "mxfp8"
  # TODO(future PR): fix this
  # error: https://gist.github.com/namgyu-youn/dff3e22320b028b28f7d533727a88bb1
  "awq_int4_weight_only"
  # TODO(future PR): fix this (same issue in AWQ)
  # error: https://gist.github.com/namgyu-youn/0dca97ff669cfebfcb3af522ae10ea83
  "smoothquant_int8"
)

# TODO(future PR): add A100 and B200 tag groups

# Get tag/recipe as first positional argument (optional, default: all)
TAG_OR_RECIPE="${1:-all}"

# Get model_id as second positional argument (optional)
MODEL_ID="${2:-meta-llama/Llama-3.1-8B}"

# Get log file as third positional argument (optional)
LOG_FILE="${3:-benchmarks/data/measure_accuracy_and_performance_log.txt}"

# Select recipes based on tag or specific recipe
if [ "$TAG_OR_RECIPE" = "all" ]; then
  QUANT_RECIPES=("${QUANT_RECIPES_ALL[@]}")
elif [ "$TAG_OR_RECIPE" = "h100" ]; then
  QUANT_RECIPES=("${QUANT_RECIPES_H100[@]}")
elif [ "$TAG_OR_RECIPE" = "b200" ]; then
  QUANT_RECIPES=("${QUANT_RECIPES_B200[@]}")
else
  # Check if it's a valid recipe name
  VALID_RECIPE=false
  for recipe in "${QUANT_RECIPES_ALL[@]}"; do
    if [ "$recipe" = "$TAG_OR_RECIPE" ]; then
      VALID_RECIPE=true
      QUANT_RECIPES=("$TAG_OR_RECIPE")
      break
    fi
  done

  if [ "$VALID_RECIPE" = false ]; then
    echo "Error: Invalid tag or recipe name: '$TAG_OR_RECIPE'"
    echo ""
    echo "Valid tags:"
    echo "  - all"
    echo "  - h100"
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

python -c "import torch; import torchao; import vllm; print(f'{torch.__version__=}\n{torch.cuda.get_device_name()=}\n{torchao.__version__=}\n{vllm.__version__=}')" | tee -a "$LOG_FILE"

for quant_recipe in "${QUANT_RECIPES[@]}"; do

  echo | tee -a "$LOG_FILE"
  echo "processing quant_recipe $quant_recipe" | tee -a "$LOG_FILE"
  echo | tee -a "$LOG_FILE"

  OUTPUT_DIR="benchmarks/data/quantized_model/$MODEL_ID-$quant_recipe/"

  # create quantized model (unless skipped via environment variable)
  if [ "${SKIP_MODEL_CREATE:-0}" != "1" ]; then
    # Note: the -u flag is to prevent python from buffering stdout and stderr
    # and make the output log file be in chronological order
    rm -rf $OUTPUT_DIR && python -u benchmarks/quantization/create_quantized_model.py --model_id $MODEL_ID --output_dir $OUTPUT_DIR --quant_recipe_name $quant_recipe 2>&1 | tee -a "$LOG_FILE"
  else
    echo "Skipping model creation (SKIP_MODEL_CREATE=1), using existing model at $OUTPUT_DIR" | tee -a "$LOG_FILE"
  fi

  # run eval (unless skipped via environment variable)
  if [ "${SKIP_LM_EVAL:-0}" != "1" ]; then
    lm_eval --model hf --model_args "pretrained=$OUTPUT_DIR" --tasks "wikitext,winogrande" --device "cuda:0" --batch_size 1 --output_path "$OUTPUT_DIR/lm_eval_outputs/" 2>&1 | tee -a "$LOG_FILE"
  else
    echo "Skipping lm_eval (SKIP_LM_EVAL=1)" | tee -a "$LOG_FILE"
  fi

  # simple performance test (unless skipped via environment variable)
  if [ "${SKIP_VLLM:-0}" != "1" ]; then
    # Check if this recipe is known to be broken in vllm
    RECIPE_BROKEN_IN_VLLM=false
    for broken_recipe in "${VLLM_BROKEN_RECIPES[@]}"; do
      if [ "$quant_recipe" = "$broken_recipe" ]; then
        RECIPE_BROKEN_IN_VLLM=true
        break
      fi
    done

    if [ "$RECIPE_BROKEN_IN_VLLM" = true ]; then
      echo "Skipping vllm benchmarking for $quant_recipe (known to be broken in vllm)" | tee -a "$LOG_FILE"
    else
      # prefill
      PREFILL_ARGS="--num_prompts 32 --input_len 4096 --output_len 32 --max_model_len 4128"
      echo | tee -a "$LOG_FILE"
      echo "benchmarking vllm prefill performance with $PREFILL_ARGS" | tee -a "$LOG_FILE"
      echo | tee -a "$LOG_FILE"
      vllm bench throughput --model $OUTPUT_DIR --dtype bfloat16 $PREFILL_ARGS 2>&1 | tee -a "$LOG_FILE"

      # decode
      DECODE_ARGS="--num_prompts 128 --input_len 32 --output_len 2048 --max_model_len 2080"
      echo | tee -a "$LOG_FILE"
      echo "benchmarking vllm decode performance with $DECODE_ARGS"
      echo | tee -a "$LOG_FILE"
      vllm bench throughput --model $OUTPUT_DIR --dtype bfloat16 $DECODE_ARGS 2>&1 | tee -a "$LOG_FILE"
    fi
  else
    echo "Skipping vllm benchmarking (SKIP_VLLM=1)" | tee -a "$LOG_FILE"
  fi

done

benchmarks/quantization/parse_log.py $LOG_FILE
