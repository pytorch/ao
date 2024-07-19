#!/bin/bash

PYTHON_SCRIPT="scripts/generate.py"
python $PYTHON_SCRIPT --checkpoint_path=checkpoints/meta-llama/Meta-Llama-3-8B/model.pth --compile --quantization=MP_llama3 --sensi_bit=5 --non_sensi_bit=4 --write_result mp_quant_benchmark_results.txt
python $PYTHON_SCRIPT --checkpoint_path=checkpoints/meta-llama/Meta-Llama-3-8B/model.pth --compile --quantization=MP_llama3 --sensi_bit=4 --non_sensi_bit=3 --write_result mp_quant_benchmark_results.txt
python $PYTHON_SCRIPT --checkpoint_path=checkpoints/meta-llama/Meta-Llama-3-8B/model.pth --compile --quantization=MP_llama3 --sensi_bit=5 --non_sensi_bit=3 --write_result mp_quant_benchmark_results.txt


echo "All processes are complete."
