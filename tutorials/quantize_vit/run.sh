#!/bin/bash

# Run bfloat16 version
TORCH_LOGS='graph_breaks,recompiles' python run_vit_b.py

# Run dynamic quantized version
TORCH_LOGS='graph_breaks,recompiles' python run_vit_b_quant.py

# Store the output code for further inspection
TORCH_LOGS='output_code' python run_vit_b.py 2> bfloat16_code
TORCH_LOGS='output_code' python run_vit_b_quant.py 2> quant_code
