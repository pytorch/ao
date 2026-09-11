# Blockwise FP8 Training Benchmarks

This directory contains benchmarking scripts for the blockwise FP8 quantization
and GEMM paths under `torchao.prototype.blockwise_fp8_training.kernels`.

## Linear Roofline Benchmark

The linear benchmark compares measured `Float8BlockwiseLinear` fwd/bwd speedup
against the shared blockwise FP8 roofline target:

```bash
python benchmarks/prototype/blockwise_fp8_training/bench_linear_roofline.py
```

What it reports:

- `b_bf16_e2e_s`: measured BF16 linear fwd/bwd time.
- `b_fp8_e2e_s`: measured `Float8BlockwiseLinear` fwd/bwd time.
- `b_fp8_e2e_spdp`: measured BF16 / FP8 speedup.
- `r_fp8_gemm_and_ovhd_spdp`: modeled blockwise FP8 roofline speedup.
- `b_fp8_e2e_spdp_ratio_of_r`: measured speedup as a ratio of modeled
  roofline speedup.

By default, it runs the DSV3 16B/671B FFN shapes with the scaled-mm backend.
Pass `--use_triton` to time the prototype Triton GEMM backend.

## Quantized Kernel Bandwidth Benchmark

The kernel-path bandwidth utility is:

```bash
python -m benchmarks.prototype.blockwise_fp8_training.benchmark_quant_kernel_bandwidth
```

To additionally validate Triton outputs against the Torch reference
implementations:

```bash
python -m benchmarks.prototype.blockwise_fp8_training.benchmark_quant_kernel_bandwidth --check-correctness
```

What it reports:

- `kernel_us`: measured runtime of the public quantization wrapper call
- `effective_logical_io_gbps`: logical tensor IO bytes divided by measured time
- `logical_io_vs_achievable_%`: `effective_logical_io_gbps / achievable_bandwidth_gbps`

Notes:

- The benchmark times the public wrapper functions in
  `torchao.prototype.blockwise_fp8_training.kernels`.
- `--check-correctness` runs the matching Torch reference path once per valid
  kernel and shape before reporting results. This adds overhead and is intended
  for validation, not headline timing runs.
- The bandwidth number uses the expected tensor IO footprint, not hardware DRAM
  counters.
- Peak bandwidth defaults to CUDA device properties. `--use-roofline-utils`
  switches to the static `roofline_utils` table.

### Methodology

- It times the public wrapper call, matching the style of the other benchmark
  scripts in this directory.
- It uses CUDA event timing and the median, via
  `benchmark_cuda_function_in_microseconds(...)` from
  [benchmarks/utils.py](/home/dev/ao/benchmarks/utils.py#L101).
- It validates unsupported shapes up front and skips them instead of silently
  measuring invalid configurations.

## Current H100 Results

Captured on 2026-03-20 with:

```bash
python -m benchmarks.prototype.blockwise_fp8_training.benchmark_quant_kernel_bandwidth
```

Environment:

- GPU: `NVIDIA H100 80GB HBM3`
- Peak bandwidth reference: `3352.3 GB/s`
- Peak bandwidth source: `cuda_device_properties`
- Achievable bandwidth reference: `3084.1 GB/s`
- Achievable bandwidth uses `92.0%` of peak bandwidth
- Achievable bandwidth source: `roofline_utils_pct_achievable_mem_bw`

### Per-shape Results
Tested with shapes 32768 and 131072 to reflect real world training:

| kernel | shape | kernel_us | effective_logical_io_gbps | logical_io_vs_achievable_% |
|---|---|---:|---:|---:|
| act_quant_transposed_lhs | 32768x4096 | 154.46 | 2633.9 | 85.4 |
| weight_quant_transposed_rhs | 32768x4096 | 150.53 | 2675.2 | 86.7 |
| act_quant_lhs | 32768x4096 | 150.86 | 2696.8 | 87.4 |
| act_quant_rhs | 32768x4096 | 148.70 | 2736.0 | 88.7 |
| weight_quant_rhs | 32768x4096 | 144.99 | 2777.3 | 90.1 |
| weight_quant_transposed_rhs | 131072x4096 | 581.89 | 2768.1 | 89.8 |
| act_quant_lhs | 131072x4096 | 586.98 | 2772.5 | 89.9 |
| act_quant_transposed_lhs | 131072x4096 | 581.47 | 2798.7 | 90.7 |
| act_quant_rhs | 131072x4096 | 562.56 | 2892.8 | 93.8 |
| weight_quant_rhs | 131072x4096 | 555.30 | 2900.7 | 94.1 |
