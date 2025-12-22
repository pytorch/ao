#!/bin/bash

# float8 rowwise
# note: max-autotune performance is nearly identical to regular compile on b200, so skip it for now
time python -u benchmarks/quantization/eval_accuracy_and_perf_of_flux.py --quant_config_str float8_rowwise --mode accuracy
time python -u benchmarks/quantization/eval_accuracy_and_perf_of_flux.py --quant_config_str float8_rowwise --mode performance --use_compile

# mxfp8
time python -u benchmarks/quantization/eval_accuracy_and_perf_of_flux.py --quant_config_str mxfp8 --mode accuracy --cache_baseline_images
time python -u benchmarks/quantization/eval_accuracy_and_perf_of_flux.py --quant_config_str mxfp8 --mode performance --use_compile

# nvfp4
# note: even though we are using a triton kernel for to_nvfp4 cast, we still need
# to enable compile for fast generation of the nvfp4 global scale
time python -u benchmarks/quantization/eval_accuracy_and_perf_of_flux.py --quant_config_str nvfp4 --mode accuracy --cache_baseline_images
time python -u benchmarks/quantization/eval_accuracy_and_perf_of_flux.py --quant_config_str nvfp4 --mode performance --use_compile
