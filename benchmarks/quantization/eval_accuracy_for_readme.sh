#!/bin/bash

set -e

# Get model_id as positional argument (optional)
MODEL_ID="${1:-meta-llama/Llama-3.1-8B}"

# Get log file as first positional argument (optional)
LOG_FILE="${2:-benchmarks/data/eval_accuracy_for_readme_log.txt}"

# Build the base command arguments
BASE_ARGS="--tasks wikitext winogrande"
if [[ -n "$MODEL_ID" ]]; then
  BASE_ARGS="--model_id $MODEL_ID $BASE_ARGS"
fi

# baseline
# note: the -u flag is to prevent python from buffering stdout and stderr
# and make the output log file be in chronological order
time python -u benchmarks/quantization/eval_accuracy_for_readme.py $BASE_ARGS 2>&1 | tee "$LOG_FILE"

# quantized recipes
# note:
# * `int4_groupwise_hqq_weight_float8_rowwise_activation` doesn't work with dtype_map auto: https://gist.github.com/vkuzo/6b128681b628744d445c553cdeac8a85
# * `int4_groupwise_hqq_weight_only` only works on A100
for quant_recipe in float8_rowwise int4_groupwise_weight_float8_rowwise_activation int4_groupwise_hqq_weight_only int8_rowwise_weight_only int8_rowwise; do
  time python -u benchmarks/quantization/eval_accuracy_for_readme.py $BASE_ARGS --quant_recipe_name $quant_recipe 2>&1 | tee -a "$LOG_FILE"
done

# TODO(future PR): script to parse the log file instead of manual copy-paste
