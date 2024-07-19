#!/bin/bash
PYTHON_SCRIPT="scripts/generate.py"
python $PYTHON_SCRIPT --checkpoint_path=checkpoints/meta-llama/Meta-Llama-3-8B/model.pth --compile --quantization=int4wo --write_result uni_quant_benchmark_results.txt
python $PYTHON_SCRIPT --checkpoint_path=checkpoints/meta-llama/Meta-Llama-3-8B/model.pth --compile --quantization=int8wo --write_result uni_quant_benchmark_results.txt
python $PYTHON_SCRIPT --checkpoint_path=checkpoints/meta-llama/Meta-Llama-3-8B/model.pth --compile --quantization=2 --write_result uni_quant_benchmark_results.txt
python $PYTHON_SCRIPT --checkpoint_path=checkpoints/meta-llama/Meta-Llama-3-8B/model.pth --compile --quantization=3 --write_result uni_quant_benchmark_results.txt
python $PYTHON_SCRIPT --checkpoint_path=checkpoints/meta-llama/Meta-Llama-3-8B/model.pth --compile --quantization=5 --write_result uni_quant_benchmark_results.txt
python $PYTHON_SCRIPT --checkpoint_path=checkpoints/meta-llama/Meta-Llama-3-8B/model.pth --compile --quantization=6 --write_result uni_quant_benchmark_results.txt

echo "All processes are complete."
