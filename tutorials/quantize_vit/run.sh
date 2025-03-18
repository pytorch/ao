# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#!/bin/bash

# Run bfloat16 version
TORCH_LOGS='graph_breaks,recompiles' python run_vit_b.py

# Run dynamic quantized version
TORCH_LOGS='graph_breaks,recompiles' python run_vit_b_quant.py

# Store the output code for further inspection
echo "bfloat16 generated code lives in:"
TORCH_LOGS='output_code' python run_vit_b.py 2>&1 | grep "Output code written to: " | awk -F" " '{print $NF}'
echo "quantization generated code lives in:"
TORCH_LOGS='output_code' python run_vit_b_quant.py 2>&1 | grep "Output code written to: " | awk -F" " '{print $NF}'
