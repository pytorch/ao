# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash

# see README.md for instructions

# Default quantization options
default_quants=("FP8" "INT4" "INT8-INT4")
push_to_hub=""
push_to_user_id=""
populate_model_card_template=""
# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_id)
      model_id="$2"
      shift 2
      ;;
    --quants)
      shift
      quants=()
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        quants+=("$1")
        shift
      done
      ;;
     --push_to_hub)
      push_to_hub="--push_to_hub"
      shift
      ;;
     --push_to_user_id)
      push_to_user_id=("--push_to_user_id $2")
      shift 2
      ;;
     --populate_model_card_template)
      populate_model_card_template="--populate_model_card_template"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done
# Use default quants if none specified
if [[ -z "$model_id" ]]; then
  echo "Error: --model_id is required"
  echo "Usage: $0 --model_id <model_id> [--quants <quant1> [quant2 ...]] [--push_to_hub] [--push_to_user_id <push_to_user_id>] [--populate_model_card_template]"
  exit 1
fi
if [[ ${#quants[@]} -eq 0 ]]; then
  quants=("${default_quants[@]}")
fi
# Run the python command for each quantization option
for quant in "${quants[@]}"; do
  echo "Running: python quantize_and_upload.py --model_id $model_id --quant $quant $push_to_hub $push_to_user_id $populate_model_card_template"
  python quantize_and_upload.py --model_id "$model_id" --quant "$quant" $push_to_hub $push_to_user_id $populate_model_card_template
done
