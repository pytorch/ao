#!/bin/bash

set -e

# Get model_id as first positional argument (optional)
MODEL_ID="${1:-meta-llama/Llama-3.1-8B}"

# Get log file as second positional argument (optional)
LOG_FILE="${2:-benchmarks/data/eval_accuracy_for_readme_log.txt}"
rm $LOG_FILE
touch $LOG_FILE

QUANT_RECIPES=(
  # no quantization (baseline)
  "None"
  "float8_rowwise"
  # note: below doesn't work with dtype_map auto: https://gist.github.com/vkuzo/6b128681b628744d445c553cdeac8a85
  "int4_groupwise_weight_float8_rowwise_activation"
  # note: below only works on A100
  "int4_groupwise_hqq_weight_only"
  "int8_rowwise_weight_only"
  "int8_rowwise"
)

for quant_recipe in "${QUANT_RECIPES[@]}"; do

  echo "processing $quant_recipe"
 
  OUTPUT_DIR="benchmarks/data/quantized_model/$MODEL_ID-$quant_recipe/"
  rm -rf $OUTPUT_DIR

  # create quantized model
  # Note: the -u flag is to prevent python from buffering stdout and stderr
  # and make the output log file be in chronological order
  python -u benchmarks/quantization/create_quantized_model.py --model_id $MODEL_ID --output_dir $OUTPUT_DIR --quant_recipe_name $quant_recipe 2>&1 | tee -a "$LOG_FILE"

  # run eval
  lm_eval --model hf --model_args "pretrained=$OUTPUT_DIR" --tasks "wikitext,winogrande" --device "cuda:0" --batch_size auto --output_path "$OUTPUT_DIR/lm_eval_outputs/" 2>&1 | tee -a "$LOG_FILE"
done

# TODO(future PR): script to parse the log file instead of manual copy-paste
