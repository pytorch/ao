# Blockwise FP8 Training Benchmarks

This directory contains benchmarking scripts for the blockwise FP8 quantization
and GEMM paths under `torchao.prototype.blockwise_fp8_training.kernels`.

## Quantized Kernel Bandwidth Benchmark

The kernel-path bandwidth utility is:

```bash
python -m benchmarks.prototype.blockwise_fp8_training.benchmark_quant_kernel_bandwidth
```

What it reports:

- `kernel_us`: measured runtime of the public quantization wrapper call
- `effective_logical_io_gbps`: logical tensor IO bytes divided by measured time
- `logical_io_vs_peak_%`: `effective_logical_io_gbps / peak_bandwidth_gbps`
- `logical_io_vs_achievable_%`: `effective_logical_io_gbps / achievable_bandwidth_gbps`

Notes:

- The benchmark times the public wrapper functions in
  `torchao.prototype.blockwise_fp8_training.kernels`.
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
- Achievable bandwidth source: `roofline_utils_pct_achievable_mem_bw`

### Per-shape Results
Tested with shapes 32768 and 131072 to reflect real world training:

| kernel | shape | kernel_us | effective_logical_io_gbps | logical_io_vs_peak_% | logical_io_vs_achievable_% |
|---|---|---:|---:|---:|---:|
| act_quant_transposed_lhs | 32768x4096 | 154.46 | 2633.9 | 78.6 | 85.4 |
| weight_quant_transposed_rhs | 32768x4096 | 150.53 | 2675.2 | 79.8 | 86.7 |
| act_quant_lhs | 32768x4096 | 150.86 | 2696.8 | 80.4 | 87.4 |
| act_quant_rhs | 32768x4096 | 148.70 | 2736.0 | 81.6 | 88.7 |
| weight_quant_rhs | 32768x4096 | 144.99 | 2777.3 | 82.8 | 90.1 |
| weight_quant_transposed_rhs | 131072x4096 | 581.89 | 2768.1 | 82.6 | 89.8 |
| act_quant_lhs | 131072x4096 | 586.98 | 2772.5 | 82.7 | 89.9 |
| act_quant_transposed_lhs | 131072x4096 | 581.47 | 2798.7 | 83.5 | 90.7 |
| act_quant_rhs | 131072x4096 | 562.56 | 2892.8 | 86.3 | 93.8 |
| weight_quant_rhs | 131072x4096 | 555.30 | 2900.7 | 86.5 | 94.1 |