# Blockwise FP8 Training Benchmarks

This directory contains benchmarking scripts for the blockwise FP8 quantization
and GEMM paths under `torchao.prototype.blockwise_fp8_training.kernels`.

## Quantized Kernel Bandwidth Benchmark

The kernel-path bandwidth utility is:

```bash
python -m benchmarks.prototype.blockwise_fp8_training.benchmark_quant_kernel_bandwidth
```

What it reports:

- `kernel_us`: measured runtime of the preallocated-output kernel launch path
- `effective_logical_io_gbps`: logical tensor IO bytes divided by measured time
- `logical_io_vs_peak_%`: `effective_logical_io_gbps / peak_bandwidth_gbps`
- `logical_io_vs_achievable_%`: `effective_logical_io_gbps / achievable_bandwidth_gbps`

Notes:

- The benchmark preallocates `y` and `s` and times the Triton launch path.
- The benchmark verifies that the preallocated-output launch path matches the
  public wrapper outputs before timing.
- The bandwidth number uses the expected tensor IO footprint, not hardware DRAM
  counters.
- Peak bandwidth defaults to CUDA device properties. `--use-roofline-utils`
  switches to the static `roofline_utils` table.

### Methodology

This benchmark intentionally uses
`do_bench_triton(...)` from [autotuner.py](/home/dev/ao/torchao/kernel/autotuner.py#L16)
instead, because it flushes L2 between measured iterations. For a bandwidth
benchmark, warm-cache timings can overstate effective memory
bandwidth.

- It times the low-level kernel launch path, not the Python wrapper.
- It preallocates outputs so allocation time is not counted as kernel time.
- It runs an untimed reference check against the public wrapper before timing.
- It uses CUDA event timing and the median, via `do_bench_triton(...)`.
- It avoids `torch.cuda.empty_cache()`, which changes allocator state but does
  not flush L2.
- It validates unsupported shapes up front and skips them instead of silently
  measuring invalid configurations.

## Current H100 Results

Captured on 2026-03-19 with:

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
| act_quant_lhs | 32768x4096 | 668.10 | 609.0 | 18.2 | 19.7 |
| act_quant_rhs | 32768x4096 | 658.56 | 617.8 | 18.4 | 20.0 |
| act_quant_transposed_lhs | 32768x4096 | 655.14 | 621.0 | 18.5 | 20.1 |
| weight_quant_rhs | 32768x4096 | 283.49 | 1420.5 | 42.4 | 46.1 |
| weight_quant_transposed_rhs | 32768x4096 | 281.92 | 1428.4 | 42.6 | 46.3 |
| act_quant_lhs | 131072x4096 | 663.10 | 2454.2 | 73.2 | 79.6 |
| act_quant_rhs | 131072x4096 | 657.34 | 2475.7 | 73.9 | 80.3 |
| act_quant_transposed_lhs | 131072x4096 | 655.14 | 2484.0 | 74.1 | 80.5 |
| weight_quant_transposed_rhs | 131072x4096 | 583.20 | 2761.9 | 82.4 | 89.6 |
| weight_quant_rhs | 131072x4096 | 559.52 | 2878.8 | 85.9 | 93.3 |
