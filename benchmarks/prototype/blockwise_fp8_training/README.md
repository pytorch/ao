# Blockwise FP8 Training Benchmarks

This directory contains benchmarking scripts for the blockwise FP8 quantization
wrappers and GEMM paths under
`torchao.prototype.blockwise_fp8_training.kernels`.

## Quant Wrapper Bandwidth Benchmark

The wrapper-path bandwidth utility is:

```bash
python -m benchmarks.prototype.blockwise_fp8_training.benchmark_quant_kernel_bandwidth
```

What it reports:

- `wrapper_us`: measured runtime of the Python wrapper call
- `effective_logical_io_gbps`: logical tensor IO bytes divided by measured time
- `logical_io_vs_peak_%`: `effective_logical_io_gbps / peak_bandwidth_gbps`
- `logical_io_vs_achievable_%`: `effective_logical_io_gbps / achievable_bandwidth_gbps`

Notes:

- The timing reflects the public wrapper path, including output allocation.
- The bandwidth number uses the expected tensor IO footprint, not hardware DRAM
  counters.
- Peak bandwidth defaults to CUDA device properties. `--use-roofline-utils`
  switches to the static `roofline_utils` table.

## Current H100 Results

Captured on 2026-03-19 with:

Environment:

- GPU: `NVIDIA H100 80GB HBM3`
- Peak bandwidth reference: `3352.3 GB/s`
- Peak bandwidth source: `cuda_device_properties`
- Achievable bandwidth reference: `3084.1 GB/s`
- Achievable bandwidth source: `roofline_utils_pct_achievable_mem_bw`

### Per-shape Results

| kernel | shape | wrapper_us | effective_logical_io_gbps | logical_io_vs_peak_% | logical_io_vs_achievable_% |
|---|---|---:|---:|---:|---:|
| act_quant_transposed_lhs | 32768x4096 | 154.62 | 2631.2 | 78.5 | 85.3 |
| weight_quant_transposed_rhs | 32768x4096 | 150.14 | 2682.0 | 80.0 | 87.0 |
| act_quant_lhs | 32768x4096 | 150.66 | 2700.5 | 80.6 | 87.6 |
| act_quant_rhs | 32768x4096 | 147.97 | 2749.6 | 82.0 | 89.2 |
| weight_quant_rhs | 32768x4096 | 144.22 | 2792.1 | 83.3 | 90.5 |
| act_quant_transposed_lhs | 131072x4096 | 590.37 | 2756.6 | 82.2 | 89.4 |
| weight_quant_transposed_rhs | 131072x4096 | 580.74 | 2773.6 | 82.7 | 89.9 |
| act_quant_lhs | 131072x4096 | 585.38 | 2780.1 | 82.9 | 90.1 |
| act_quant_rhs | 131072x4096 | 563.17 | 2889.7 | 86.2 | 93.7 |
| weight_quant_rhs | 131072x4096 | 556.48 | 2894.5 | 86.3 | 93.9 |

### Aggregate Results

| kernel | avg_effective_logical_io_gbps | avg_logical_io_vs_peak_% | worst_case_logical_io_vs_peak_% |
|---|---:|---:|---:|
| act_quant_transposed_lhs | 2693.9 | 80.4 | 78.5 |
| weight_quant_transposed_rhs | 2727.8 | 81.4 | 80.0 |
| act_quant_lhs | 2740.3 | 81.7 | 80.6 |
| act_quant_rhs | 2819.6 | 84.1 | 82.0 |
| weight_quant_rhs | 2843.3 | 84.8 | 83.3 |

The strongest wrapper-path result in this run was `weight_quant_rhs` at
`2894.5 GB/s` (`86.3%` of the peak reference). The weakest was
`act_quant_transposed_lhs` at `2631.2 GB/s` (`78.5%` of the peak reference).
