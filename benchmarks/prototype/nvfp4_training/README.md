# NVFP4 Training Benchmarks

This directory contains benchmarking scripts for the NVFP4 training kernels
under `torchao.prototype.mx_formats`.

## Hadamard Amax Benchmark

Benchmarks `triton_rht_amax` — the fused Randomized Hadamard Transform + amax
reduction kernel used in NVFP4 training.

```bash
python -m benchmarks.prototype.nvfp4_training.bench_hadamard_amax
```

What it reports:

- `time_us`: median kernel runtime in microseconds
- `gbps`: effective memory bandwidth (input read bytes / time)

### Methodology

- Sweeps M ∈ {128, 256, 1024, 8192} × N ∈ {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}
- Uses `benchmark_cuda_function_in_microseconds` from `benchmarks/utils.py`,
  which wraps `triton.testing.do_bench` and returns the median.
- Bandwidth is computed from input read bytes only (bfloat16 input, scalar output).
