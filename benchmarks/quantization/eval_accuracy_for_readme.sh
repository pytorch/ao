#!/bin/bash

# eval_accuracy_for_readme.sh - Evaluate quantization recipe accuracy
#
# Usage: ./eval_accuracy_for_readme.sh [TAG_OR_RECIPE] [MODEL_ID] [LOG_FILE]
#
# Arguments:
#   TAG_OR_RECIPE  (optional) Tag group, single recipe name, or "all" (default: all)
#                  Valid tags: all, h100
#                  Valid recipes: None, float8_rowwise,
#                                 int4_groupwise_weight_float8_rowwise_activation,
#                                 int4_groupwise_hqq_weight_only,
#                                 int8_rowwise_weight_only, int8_rowwise
#   MODEL_ID       (optional) HuggingFace model ID (default: meta-llama/Llama-3.1-8B)
#   LOG_FILE       (optional) Output log file path (default: benchmarks/data/eval_accuracy_for_readme_log.txt)
#
# Examples:
#   ./eval_accuracy_for_readme.sh                    # Run all recipes with default model
#   ./eval_accuracy_for_readme.sh h100               # Run H100-compatible recipes only
#   ./eval_accuracy_for_readme.sh float8_rowwise     # Run single recipe
#   ./eval_accuracy_for_readme.sh h100 meta-llama/Llama-3.2-8B  # Custom model with H100 recipes
#   ./eval_accuracy_for_readme.sh int8_rowwise meta-llama/Llama-3.2-8B my_log.txt  # All custom args

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

# TODO(future PR): add A100 and B200 tag groups

# Get tag/recipe as first positional argument (optional, default: all)
TAG_OR_RECIPE="${1:-all}"

# Get model_id as second positional argument (optional)
MODEL_ID="${2:-meta-llama/Llama-3.1-8B}"

# Get log file as third positional argument (optional)
LOG_FILE="${3:-benchmarks/data/eval_accuracy_for_readme_log.txt}"

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

rm $LOG_FILE
touch $LOG_FILE

for quant_recipe in "${QUANT_RECIPES[@]}"; do

  echo "processing $quant_recipe"
 
  OUTPUT_DIR="benchmarks/data/quantized_model/$MODEL_ID-$quant_recipe/"

  # create quantized model
  # Note: the -u flag is to prevent python from buffering stdout and stderr
  # and make the output log file be in chronological order
  rm -rf $OUTPUT_DIR && python -u benchmarks/quantization/create_quantized_model.py --model_id $MODEL_ID --output_dir $OUTPUT_DIR --quant_recipe_name $quant_recipe 2>&1 | tee -a "$LOG_FILE"

  # run eval
  lm_eval --model hf --model_args "pretrained=$OUTPUT_DIR" --tasks "wikitext,winogrande" --device "cuda:0" --batch_size auto --output_path "$OUTPUT_DIR/lm_eval_outputs/" 2>&1 | tee -a "$LOG_FILE"
done

# TODO(future PR): script to parse the log file instead of manual copy-paste
