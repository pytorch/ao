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

Benchmarks `triton_quantize_2d_weight` — the 2D NVFP4 E2M1 weight
quantization kernel (2D 16x16 block scaling) producing rowwise and colwise
packed FP4 outputs with swizzled scale factors. Requires SM100 (Blackwell).

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

## CuteDSL kernels — comparison vs Triton

Each `bench_*` script above runs **both backends** (Triton and CuteDSL) on the same shapes and
reports the speedup. The CuteDSL (`nvidia-cutlass-dsl`) kernels do the Randomized Hadamard Transform
on Blackwell tensor cores; they require SM100 and the same shape constraints as the Triton kernels,
with the addition that **M must be divisible by 256**.

```bash
python -m benchmarks.prototype.nvfp4_training.bench_hadamard_amax --shape-set representative-models
python -m benchmarks.prototype.nvfp4_training.bench_hadamard_quantize_row_col --shape-set representative-models
python -m benchmarks.prototype.nvfp4_training.bench_quantize_2d --shape-set representative-models
```

### Methodology

- Reports **device kernel time** (CUDA kernel self-time via `torch.profiler`, averaged over the
  timed loop; see `bench_utils.kernel_time_us`) for each backend, fed the same precomputed global
  amaxes. Device kernel time is used rather than wall-clock because NVFP4 training runs the linear
  under CUDA graphs / `torch.compile`, which amortizes host launch overhead.

Run environment: NVIDIA GB200, PyTorch 2.12.0a0, Triton 3.7.0, nvidia-cutlass-dsl 4.6.1.

### Hadamard Amax (`cutedsl_rht_amax` vs `triton_rht_amax`)

| Model | Shape | M | N | cutedsl_kernel_us | triton_kernel_us | speedup | cutedsl_gbps |
|---|---|---:|---:|---:|---:|---:|---:|
| Llama 3 8B | hidden-state input | 2048 | 4096 | 11.36 | 16.78 | 1.48x | 1477.4 |
| Llama 3 8B | mlp.down input | 2048 | 14336 | 20.99 | 27.41 | 1.31x | 2797.5 |
| Llama 3 70B | hidden-state input | 2048 | 8192 | 16.21 | 21.43 | 1.32x | 2070.2 |
| Llama 3 70B | mlp.down input | 2048 | 28672 | 38.03 | 42.92 | 1.13x | 3087.8 |

### Hadamard Quantize Row+Col (`cutedsl_rht_quantize_row_col` vs `triton_rht_quantize_row_col`, RTNE)

| Model | Shape | M | N | cutedsl_kernel_us | triton_kernel_us | speedup | cutedsl_gbps |
|---|---|---:|---:|---:|---:|---:|---:|
| Llama 3 8B | hidden-state input | 2048 | 4096 | 13.82 | 24.15 | 1.75x | 1897.4 |
| Llama 3 8B | mlp.down input | 2048 | 14336 | 35.11 | 60.47 | 1.72x | 2613.0 |
| Llama 3 70B | hidden-state input | 2048 | 8192 | 23.51 | 37.22 | 1.58x | 2229.6 |
| Llama 3 70B | mlp.down input | 2048 | 28672 | 67.97 | 113.71 | 1.67x | 2699.6 |

### 2D Weight Quantize (`cutedsl_weight_quantize_2d` vs `triton_weight_quantize_2d`, no RHT)

Both kernels emit 2D 16x16 weight block scaling.
Requires `out_features % 256 == 0`.

| Model | Weight | M (out) | N (in) | cutedsl_kernel_us | triton_kernel_us | speedup | cutedsl_gbps |
|---|---|---:|---:|---:|---:|---:|---:|
| Llama 3 8B | mlp.gate/up | 14336 | 4096 | 79.38 | 249.05 | 3.14x | 2311.6 |
| Llama 3 8B | mlp.down | 4096 | 14336 | 79.73 | 248.95 | 3.12x | 2301.5 |
| Llama 3 70B | mlp.gate/up | 28672 | 8192 | 291.66 | 948.65 | 3.25x | 2516.6 |
| Llama 3 70B | mlp.down | 8192 | 28672 | 292.05 | 948.51 | 3.25x | 2513.2 |

## Grouped (MoE) kernels

The grouped kernels are the expert-parallel analogs of the kernels above: one launch
covers every local expert instead of one launch per expert. They are Triton-only —
there is no CuteDSL grouped variant.

Shapes come from `deepseek_v3_shapes.py` (TorchTitan DeepSeek-V3 recipes). `E` is the
local expert count `experts / expert_parallel_degree`; `gate/up (w1/w3)` is
`(E, moe_hidden_dim, dim)` and `down (w2)` is `(E, dim, moe_hidden_dim)`.

| model | experts | EP degree | E (local) | dim | moe_hidden_dim |
|---|---:|---:|---:|---:|---:|
| debugmodel | 8 | 1 | 8 | 256 | 256 |
| 16B | 64 | 8 | 8 | 2048 | 1408 |
| 671B | 256 | 2 | 128 | 7168 | 2048 |

Run environment for every table in this section: NVIDIA GB200, PyTorch
2.15.0a0+git04a7716, Triton 3.8.0.

### Grouped Hadamard Amax

Benchmarks `_group_rht_amax_triton_kernel` — the grouped analog of `triton_rht_amax`,
corresponding to TransformerEngine's `nvte_group_hadamard_transform_amax_graph_safe`.
Over a row-concatenated packed activation tensor it produces, per group, the post-RHT
columnwise amax and the raw rowwise amax, without materializing the transform.

```bash
python -m benchmarks.prototype.nvfp4_training.bench_group_hadamard_amax
```

- Uses `benchmark_cuda_function_in_microseconds` (median `triton.testing.do_bench`).
- Bandwidth counts the bfloat16 input read plus the `2 * E` float32 amax writes.
- Launched directly at `num_warps=8, num_stages=3` with `SHAPE_REP=SAME_BOTH_DIMS`,
  bypassing the custom op so the timed region is the kernel alone.

| model | projection | E | M | N | time_us | gbps |
|---|---|---:|---:|---:|---:|---:|
| debugmodel | gate/up (w1/w3) | 8 | 256 | 256 | 10.368 | 101.1 |
| debugmodel | down (w2) | 8 | 256 | 256 | 10.656 | 98.4 |
| 16B | gate/up (w1/w3) | 8 | 1408 | 2048 | 35.264 | 1308.3 |
| 16B | down (w2) | 8 | 2048 | 1408 | 35.264 | 1308.3 |
| 671B | gate/up (w1/w3) | 128 | 2048 | 7168 | 2070.270 | 1815.3 |
| 671B | down (w2) | 128 | 7168 | 2048 | 2062.340 | 1822.3 |

### Grouped Hadamard Quantize Row+Col

Benchmarks `_group_rht_quantize_row_col_kernel` — the grouped analog of
`triton_rht_quantize_row_col`. Consumes the per-group amaxes produced above and writes
rowwise flat buffers plus columnwise per-group views over one flat columnwise buffer.

```bash
python -m benchmarks.prototype.nvfp4_training.bench_group_rht_quantize_row_col
```

Use `--rounding rtne` or `--rounding rs` to benchmark one mode; the default is both.

- Bandwidth accounts for the bfloat16 input read plus rowwise and columnwise FP4 codes
  and swizzled FP8 scales.
- `pct_peak_mem_bw` is against peak from CUDA device properties, 7928.1 GB/s here.
- Stochastic rounding takes caller-owned Philox state as single-element device views, so
  the `rs` rows include the graph-safe RNG path rather than a host-side generator.

Round-to-nearest-even (`rtne`):

| model | projection | E | M | N | time_us | gbps | pct_peak |
|---|---|---:|---:|---:|---:|---:|---:|
| debugmodel | gate/up (w1/w3) | 8 | 256 | 256 | 13.344 | 122.8 | 1.55 |
| debugmodel | down (w2) | 8 | 256 | 256 | 13.312 | 123.1 | 1.55 |
| 16B | gate/up (w1/w3) | 8 | 1408 | 2048 | 42.016 | 1715.8 | 21.64 |
| 16B | down (w2) | 8 | 2048 | 1408 | 42.592 | 1692.6 | 21.35 |
| 671B | gate/up (w1/w3) | 128 | 2048 | 7168 | 2621.470 | 2240.0 | 28.25 |
| 671B | down (w2) | 128 | 7168 | 2048 | 2600.700 | 2257.9 | 28.48 |

Stochastic rounding (`rs`):

| model | projection | E | M | N | time_us | gbps | pct_peak |
|---|---|---:|---:|---:|---:|---:|---:|
| debugmodel | gate/up (w1/w3) | 8 | 256 | 256 | 19.488 | 84.1 | 1.06 |
| debugmodel | down (w2) | 8 | 256 | 256 | 19.488 | 84.1 | 1.06 |
| 16B | gate/up (w1/w3) | 8 | 1408 | 2048 | 78.848 | 914.3 | 11.53 |
| 16B | down (w2) | 8 | 2048 | 1408 | 78.848 | 914.3 | 11.53 |
| 671B | gate/up (w1/w3) | 128 | 2048 | 7168 | 5467.660 | 1074.0 | 13.55 |
| 671B | down (w2) | 128 | 7168 | 2048 | 5458.270 | 1075.8 | 13.57 |

Stochastic rounding costs roughly 2x at every shape above the debug model.

### Grouped 2D Weight Quantize

Benchmarks `_group_weight_quantize_2d_kernel` — the grouped analog of
`triton_weight_quantize_2d`. Quantizes dense `(E, M, N)` BF16 expert weights with 2D
16x16 block scaling, emitting rowwise and columnwise (`W.T`) FP4 codes and swizzled
scale factors for every expert. Requires SM100.

```bash
python -m benchmarks.prototype.nvfp4_training.bench_group_quantize_2d
```

- Bandwidth accounts for the bfloat16 read plus both FP4 outputs and both scale writes.
- Launched at the shipped `BLOCK_M = BLOCK_N = 128` with output buffers and global
  amaxes precomputed, so the timed region is the kernel alone.

| model | projection | E | M | N | time_us | gbps |
|---|---|---:|---:|---:|---:|---:|
| debugmodel | gate/up (w1/w3) | 8 | 256 | 256 | 10.624 | 154.2 |
| debugmodel | down (w2) | 8 | 256 | 256 | 10.336 | 158.5 |
| 16B | gate/up (w1/w3) | 8 | 1408 | 2048 | 45.504 | 1584.3 |
| 16B | down (w2) | 8 | 2048 | 1408 | 45.504 | 1584.3 |
| 671B | gate/up (w1/w3) | 128 | 2048 | 7168 | 2745.340 | 2138.9 |
| 671B | down (w2) | 128 | 7168 | 2048 | 2722.020 | 2157.2 |

### Grouped Weight Amax

Benchmarks `triton_group_weight_amax` — the input-side twin of the grouped 2D weight
quantize, producing exactly the `(E,)` float32 amax that kernel consumes. Compared
against `torch.linalg.vector_norm(W, ord=inf, dim=(1, 2))`, which computes bit-exact
the same values; the kernel wins on memory-level parallelism, not on doing less work.

```bash
python -m benchmarks.prototype.nvfp4_training.bench_group_weight_amax
```

- Reports **device kernel time** (`bench_utils.kernel_time_us`, CUDA self-time via
  `torch.profiler`) rather than wall time, because the reduction is small enough at
  these shapes that host dispatch would dominate.
- Runs at `E = 4` rather than the model's local expert count: the target deployment is
  high expert parallelism, and the ranking inverts at large `E`.
- `kernel_time_us` profiles a hot loop over one buffer and does not flush L2, so shapes
  under L2 capacity read partly from cache and the absolute TB/s is optimistic. Both
  backends are pure-read reductions and lose the cache in the same proportion, so the
  speedup survives — 1.52x hot against 1.54x L2-flushed at 671B. Read the speedup
  column and treat bandwidth as an upper bound.

| model | projection | E | M | N | vector_norm_us | triton_us | speedup | triton_gbps |
|---|---|---:|---:|---:|---:|---:|---|---:|
| debugmodel | gate/up (w1/w3) | 4 | 256 | 256 | 9.100 | 3.850 | 2.36x | 136.2 |
| debugmodel | down (w2) | 4 | 256 | 256 | 9.071 | 3.936 | 2.30x | 133.2 |
| 16B | gate/up (w1/w3) | 4 | 1408 | 2048 | 11.744 | 5.961 | 1.97x | 3870.0 |
| 16B | down (w2) | 4 | 2048 | 1408 | 11.622 | 5.985 | 1.94x | 3854.2 |
| 671B | gate/up (w1/w3) | 4 | 2048 | 7168 | 34.086 | 22.809 | 1.49x | 5148.8 |
| 671B | down (w2) | 4 | 7168 | 2048 | 34.540 | 22.785 | 1.52x | 5154.2 |

`nvfp4_linear` uses this same op at `E = 1` on `W.unsqueeze(0)` — nothing in the kernel
is expert-specific beyond the `program_id(1)` base.
