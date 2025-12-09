#!/bin/bash

# measure_accuracy_and_performance.sh - Evaluate quantization recipe accuracy
#
# Usage: ./measure_accuracy_and_performance.sh [TAG_OR_RECIPE] [MODEL_ID] [LOG_FILE]
#
# Arguments:
#   TAG_OR_RECIPE  (optional) Tag group, single recipe name, or "all" (default: all)
#                  Valid tags: all, h100
#                  Valid recipes: None, float8_rowwise,
#                                 int4_groupwise_weight_float8_rowwise_activation,
#                                 int4_groupwise_hqq_weight_only,
#                                 int8_rowwise_weight_only, int8_rowwise
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
  # note: below only works on A100
  "int4_groupwise_hqq_weight_only"
  "int8_rowwise_weight_only"
  "int8_rowwise"
)

# Define H100-compatible recipes (excludes A100-only recipes)
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
  # current stack trace: https://gist.github.com/vkuzo/eed4894c5f3434e15d70b163e6077f60
  "float8_rowwise"
  # as of this PR, this recipe is still using AQT and CUDA graph capture time
  # in vLLM is really slow (>5 mins)
  # TODO(future PR): reenable this once we migrate this recipe off of AQT
  "int8_rowwise_weight_only"
  # TODO(future PR): fix this
  # error: https://gist.github.com/vkuzo/5bf389079442bb9851ef315cdcb797b4
  "int8_rowwise"
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

rm -rf $LOG_FILE
touch $LOG_FILE

for quant_recipe in "${QUANT_RECIPES[@]}"; do

  echo "processing $quant_recipe"

  OUTPUT_DIR="benchmarks/data/quantized_model/$MODEL_ID-$quant_recipe/"

  # create quantized model (unless skipped via environment variable)
  if [ "${SKIP_MODEL_CREATE:-0}" != "1" ]; then
    # Note: the -u flag is to prevent python from buffering stdout and stderr
    # and make the output log file be in chronological order
    rm -rf $OUTPUT_DIR && python -u benchmarks/quantization/create_quantized_model.py --model_id $MODEL_ID --output_dir $OUTPUT_DIR --quant_recipe_name $quant_recipe 2>&1 | tee -a "$LOG_FILE"
  else
    echo "Skipping model creation (SKIP_MODEL_CREATE=1), using existing model at $OUTPUT_DIR"
  fi

  # run eval (unless skipped via environment variable)
  if [ "${SKIP_LM_EVAL:-0}" != "1" ]; then
    lm_eval --model hf --model_args "pretrained=$OUTPUT_DIR" --tasks "wikitext,winogrande" --device "cuda:0" --batch_size auto --output_path "$OUTPUT_DIR/lm_eval_outputs/" 2>&1 | tee -a "$LOG_FILE"
  else
    echo "Skipping lm_eval (SKIP_LM_EVAL=1)"
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
      echo "Skipping vllm benchmarking for $quant_recipe (known to be broken in vllm)"
    else
      vllm bench latency --input_len 256 --output_len 256 --model $OUTPUT_DIR --batch_size 1 2>&1 | tee -a "$LOG_FILE"
    fi
  else
    echo "Skipping vllm benchmarking (SKIP_VLLM=1)"
  fi

done

# TODO(future PR): script to parse the log file instead of manual copy-paste
