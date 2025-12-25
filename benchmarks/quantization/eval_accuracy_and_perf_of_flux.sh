#!/bin/bash

# number of local GPUs to use for accuracy eval
NUM_GPUS=8

# float8 rowwise
# note: max-autotune performance is nearly identical to regular compile on b200, so skip it for now
time torchrun --nproc_per_node=$NUM_GPUS benchmarks/quantization/eval_accuracy_and_perf_of_flux.py --quant_config_str float8_rowwise --mode accuracy --use_deterministic_algorithms
time python -u benchmarks/quantization/eval_accuracy_and_perf_of_flux.py --quant_config_str float8_rowwise --mode aggregate_accuracy --num_gpus_used $NUM_GPUS
time python -u benchmarks/quantization/eval_accuracy_and_perf_of_flux.py --quant_config_str float8_rowwise --mode performance_hp --use_compile
time python -u benchmarks/quantization/eval_accuracy_and_perf_of_flux.py --quant_config_str float8_rowwise --mode performance_quant --use_compile

# mxfp8
time torchrun --nproc_per_node=$NUM_GPUS benchmarks/quantization/eval_accuracy_and_perf_of_flux.py --quant_config_str mxfp8 --mode accuracy --cache_baseline_images --use_deterministic_algorithms
time python -u benchmarks/quantization/eval_accuracy_and_perf_of_flux.py --quant_config_str mxfp8 --mode aggregate_accuracy --num_gpus_used $NUM_GPUS
# time python -u benchmarks/quantization/eval_accuracy_and_perf_of_flux.py --quant_config_str mxfp8 --mode performance_hp --use_compile
time python -u benchmarks/quantization/eval_accuracy_and_perf_of_flux.py --quant_config_str mxfp8 --mode performance_quant --use_compile

# nvfp4
# note: even though we are using a triton kernel for to_nvfp4 cast, we still need
# to enable compile for fast generation of the nvfp4 global scale
time torchrun --nproc_per_node=$NUM_GPUS benchmarks/quantization/eval_accuracy_and_perf_of_flux.py --quant_config_str nvfp4 --mode accuracy --cache_baseline_images --use_deterministic_algorithms
time python -u benchmarks/quantization/eval_accuracy_and_perf_of_flux.py --quant_config_str nvfp4 --mode aggregate_accuracy --num_gpus_used $NUM_GPUS
# time python -u benchmarks/quantization/eval_accuracy_and_perf_of_flux.py --quant_config_str nvfp4 --mode performance_hp --use_compile
time python -u benchmarks/quantization/eval_accuracy_and_perf_of_flux.py --quant_config_str nvfp4 --mode performance_quant --use_compile
