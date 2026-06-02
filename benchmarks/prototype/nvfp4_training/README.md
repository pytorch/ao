# NVFP4 Training Benchmarks

This directory contains benchmarking scripts for the NVFP4 training kernels
under `torchao.prototype.moe_training.nvfp4_training`.

## Hadamard Amax Benchmark

Benchmarks `triton_rht_amax` — the fused Randomized Hadamard Transform + amax
reduction kernel used in NVFP4 training.

```bash
python -m benchmarks.prototype.nvfp4_training.bench_hadamard_amax
```

To run model-derived representative shapes:

```bash
python -m benchmarks.prototype.nvfp4_training.bench_hadamard_amax --shape-set representative-models
```

What it reports:

- `time_us`: median kernel runtime in microseconds
- `gbps`: effective memory bandwidth (input read bytes / time)

### Methodology

- Sweeps M ∈ {128, 256, 1024, 8192} × N ∈ {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}
- Uses `benchmark_cuda_function_in_microseconds` from `benchmarks/utils.py`,
  which wraps `triton.testing.do_bench` and returns the median.
- Bandwidth is computed from input read bytes only (bfloat16 input, scalar output).

### Representative Model Results

The following shapes are activation-input matrices `(M, N)` for representative
linear layers. `M` is `batch_size * sequence_length` except for the DeepSeek-V3
routed expert rows, where `M` is the average per-expert token count:
`4096 tokens * 8 experts per token / 256 routed experts = 128`.

Run environment: NVIDIA GB200, PyTorch 2.12.0a0, Triton 3.7.0.

| Model | Shape | M | N | time_us | gbps |
|---|---|---:|---:|---:|---:|
| Llama 3 8B | hidden-state input | 2048 | 4096 | 19.488 | 860.900 |
| Llama 3 8B | mlp.down input | 2048 | 14336 | 31.744 | 1849.810 |
| Llama 3 70B | hidden-state input | 2048 | 8192 | 25.600 | 1310.720 |
| Llama 3 70B | mlp.down input | 2048 | 28672 | 46.048 | 2550.390 |

## Hadamard Quantize Row+Col Benchmark

Benchmarks `triton_rht_quantize_row_col` — the fused RHT + NVFP4 columnwise quantization
kernel with rowwise quantization. Requires SM100 (Blackwell).

```bash
python -m benchmarks.prototype.nvfp4_training.bench_hadamard_quantize_row_col
```

To run model-derived representative shapes:

```bash
python -m benchmarks.prototype.nvfp4_training.bench_hadamard_quantize_row_col --shape-set representative-models
```

What it reports:

- `rounding`: `rtne` for round-to-nearest-even or `rs` for stochastic rounding
- `time_us`: median kernel-only runtime in microseconds
- `gbps`: effective memory bandwidth (input read + FP4 output + scale factor write bytes / time)

### Methodology

- Sweeps M ∈ {128, 256, 1024, 8192} × N ∈ {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}
- Runs both `stochastic_rounding=False` (`rtne`) and
  `stochastic_rounding=True` (`rs`) by default; use `--rounding rtne` or
  `--rounding rs` to benchmark one mode.
- Skips configurations that raise `NotImplementedError` (pre-SM100 hardware).
- Uses `benchmark_cuda_function_in_microseconds` from `benchmarks/utils.py`.
- Precomputes global amax values, the RHT matrix, output tensors, RS seed/offset
  tensors, and Triton allocator setup before timing; the timed region directly
  launches the Triton row+col quantization kernel.
- Bandwidth accounts for bfloat16 input read, columnwise FP4 + swizzled scale write,
  and rowwise FP4 + swizzled scale write.
- Device peak memory bandwidth is computed from CUDA device properties as
  `(memory_bus_width_bits / 8) * (memory_clock_rate_khz * 1e3) * 2`.

### Representative Model Results

The following shapes use the same representative model configurations as
`bench_hadamard_amax.py`.

Run environment: NVIDIA GB200, PyTorch 2.13.0a0+git1f19af4, Triton 3.7.0.
Peak memory bandwidth from CUDA device properties: 7928.1 GB/s.

| Model | Shape | M | N | Rounding | time_us | gbps |
|---|---|---:|---:|---|---:|---:|
| Llama 3 8B | hidden-state input | 2048 | 4096 | rtne | 31.072 | 843.666 |
| Llama 3 8B | mlp.down input | 2048 | 14336 | rtne | 65.856 | 1393.200 |
| Llama 3 70B | hidden-state input | 2048 | 8192 | rtne | 43.776 | 1197.660 |
| Llama 3 70B | mlp.down input | 2048 | 28672 | rtne | 117.296 | 1564.420 |
| Llama 3 8B | hidden-state input | 2048 | 4096 | rs | 40.672 | 644.532 |
| Llama 3 8B | mlp.down input | 2048 | 14336 | rs | 91.168 | 1006.390 |
| Llama 3 70B | hidden-state input | 2048 | 8192 | rs | 60.128 | 871.953 |
| Llama 3 70B | mlp.down input | 2048 | 28672 | rs | 166.624 | 1101.290 |

## 2D Quantize Benchmark

Benchmarks `triton_quantize_2d_weight` — the 2D 16x16 NVFP4 E2M1 weight
quantization kernel producing rowwise and colwise packed FP4 outputs with
swizzled scale factors. Requires SM100 (Blackwell).

```bash
python -m benchmarks.prototype.nvfp4_training.bench_quantize_2d
```

To run model-derived representative shapes:

```bash
python -m benchmarks.prototype.nvfp4_training.bench_quantize_2d --shape-set representative-models
```

What it reports:

- `time_us`: median kernel-only runtime in microseconds
- `gbps`: effective memory bandwidth (input read + rowwise/colwise FP4 output +
  rowwise/colwise scale factor write bytes / time)

### Methodology

- Sweeps M ∈ {128, 256, 1024, 8192} × N ∈ {256, 512, 1024, 2048, 4096, 8192, 16384, 32768}
- Skips on pre-SM100 hardware.
- Uses `benchmark_cuda_function_in_microseconds` from `benchmarks/utils.py`.
- Precomputes global amax values, output tensors, and Triton allocator setup
  before timing; the timed region directly launches the Triton 2D quantization
  kernel.
- Bandwidth accounts for bfloat16 input read, rowwise FP4 + swizzled scale
  writes, and colwise FP4 + swizzled scale writes.

### Representative Model Results

The following shapes use the same representative model configurations as
`bench_hadamard_amax.py`.

Run environment: NVIDIA GB200, PyTorch 2.13.0a0+git1f19af4, Triton 3.7.0.

| Model | Shape | M | N | time_us | gbps |
|---|---|---:|---:|---:|---:|
| Llama 3 8B | hidden-state input | 2048 | 4096 | 49.600 | 528.516 |
| Llama 3 8B | mlp.down input | 2048 | 14336 | 123.616 | 742.221 |
| Llama 3 70B | hidden-state input | 2048 | 8192 | 78.304 | 669.555 |
| Llama 3 70B | mlp.down input | 2048 | 28672 | 232.160 | 790.407 |
