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
on Blackwell tensor cores; they require SM100 and accept exactly the shapes the Triton kernels do.

The two backends produce **bitwise identical output** — FP4 codes and FP8 scale factors, RTNE and
stochastic rounding alike — so they are drop-in interchangeable and the choice is purely a
performance one.

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

Run environment: NVIDIA GB200, PyTorch 2.15.0a0+git04a7716, Triton 3.8.0,
nvidia-cutlass-dsl 4.5.2.

### Hadamard Amax (`cutedsl_rht_amax` vs `triton_rht_amax`)

| Model | Shape | M | N | cutedsl_kernel_us | triton_kernel_us | speedup | cutedsl_gbps |
|---|---|---:|---:|---:|---:|---:|---:|
| Llama 3 8B | hidden-state input | 2048 | 4096 | 11.25 | 16.35 | 1.45x | 1491.7 |
| Llama 3 8B | mlp.down input | 2048 | 14336 | 20.98 | 25.31 | 1.21x | 2799.0 |
| Llama 3 70B | hidden-state input | 2048 | 8192 | 16.14 | 20.02 | 1.24x | 2079.2 |
| Llama 3 70B | mlp.down input | 2048 | 28672 | 37.24 | 39.59 | 1.06x | 3153.8 |

### Hadamard Quantize Row+Col (`cutedsl_rht_quantize_row_col` vs `triton_rht_quantize_row_col`, RTNE)

| Model | Shape | M | N | cutedsl_kernel_us | triton_kernel_us | speedup | cutedsl_gbps |
|---|---|---:|---:|---:|---:|---:|---:|
| Llama 3 8B | hidden-state input | 2048 | 4096 | 14.76 | 23.77 | 1.61x | 1775.7 |
| Llama 3 8B | mlp.down input | 2048 | 14336 | 37.62 | 60.95 | 1.62x | 2438.9 |
| Llama 3 70B | hidden-state input | 2048 | 8192 | 25.29 | 37.50 | 1.48x | 2072.8 |
| Llama 3 70B | mlp.down input | 2048 | 28672 | 72.22 | 113.88 | 1.58x | 2540.7 |

### 2D Weight Quantize (`cutedsl_weight_quantize_2d` vs `triton_weight_quantize_2d`, no RHT)

Both kernels emit 2D 16x16 weight block scaling.
Requires `out_features % 128 == 0`.

| Model | Weight | M (out) | N (in) | cutedsl_kernel_us | triton_kernel_us | speedup | cutedsl_gbps |
|---|---|---:|---:|---:|---:|---:|---:|
| Llama 3 8B | mlp.gate/up | 14336 | 4096 | 78.89 | 144.58 | 1.83x | 2325.9 |
| Llama 3 8B | mlp.down | 4096 | 14336 | 78.80 | 144.49 | 1.83x | 2328.6 |
| Llama 3 70B | mlp.gate/up | 28672 | 8192 | 286.06 | 535.91 | 1.87x | 2565.9 |
| Llama 3 70B | mlp.down | 8192 | 28672 | 284.79 | 535.20 | 1.88x | 2577.3 |

## Grouped (MoE) kernels

The grouped kernels are the expert-parallel analogs of the kernels above: one launch
covers every local expert instead of one launch per expert. Both backends implement
each of them, except the weight amax, which is Triton-only.

Shapes come from `deepseek_v3_shapes.py` (TorchTitan DeepSeek-V3 recipes). `E` is the
local expert count `experts / expert_parallel_degree`; `gate/up (w1/w3)` is
`(E, moe_hidden_dim, dim)` and `down (w2)` is `(E, dim, moe_hidden_dim)`.

| model | experts | EP degree | E (local) | dim | moe_hidden_dim |
|---|---:|---:|---:|---:|---:|
| debugmodel | 8 | 1 | 8 | 256 | 256 |
| 16B | 64 | 8 | 8 | 2048 | 1408 |
| 671B | 256 | 2 | 128 | 7168 | 2048 |

Every benchmark below runs at `E = 4` rather than the model's local expert count: the
target deployment is high expert parallelism, so a rank holds a handful of experts and
the per-model M/N at small `E` is the representative shape. `--experts` overrides it
where the script takes one.

Run environment for every table in this section: NVIDIA GB200, PyTorch
2.15.0a0+git04a7716, Triton 3.8.0, nvidia-cutlass-dsl 4.5.2.

### CuteDSL kernels — comparison vs Triton

Each grouped `bench_*` script runs **both backends** on the same shapes and reports the
speedup. The two produce **bitwise identical output** — codes and scale factors, RTNE
and stochastic rounding alike — so the choice is purely a performance one.

```bash
python -m benchmarks.prototype.nvfp4_training.bench_group_hadamard_amax --experts 4
python -m benchmarks.prototype.nvfp4_training.bench_group_rht_quantize_row_col --experts 4
python -m benchmarks.prototype.nvfp4_training.bench_group_quantize_2d
```

#### Methodology

- Reports **device kernel time** (CUDA kernel self-time via `torch.profiler`, averaged
  over the timed loop; see `bench_utils.kernel_time_us`) for each backend, fed the same
  precomputed global amaxes. Device kernel time rather than wall clock because NVFP4
  training runs these under CUDA graphs / `torch.compile`, which amortizes host launch
  overhead. Earlier revisions of this file timed the raw Triton kernels with
  `do_bench`; those numbers are not comparable to these.
- Both backends go through their custom op, so the timed region is the same work.
- `pct_peak` is against peak from CUDA device properties, 7928.1 GB/s here.
- The CuteDSL grouped kernels cap at `MAX_GROUPS = 64` and report `n/a` above it.

The debug model is 256x256 at `E = 4` — sixteen 128x128 tiles, which cannot fill the
GPU. CuteDSL loses there because its persistent CLC scheduler has nothing to amortize;
that is the expected shape of the curve, not a regression.

#### Grouped Hadamard Amax (`cutedsl_group_rht_amax` vs `triton_group_rht_amax`)

The grouped analog of `rht_amax`, corresponding to TransformerEngine's
`nvte_group_hadamard_transform_amax_graph_safe`. Over a row-concatenated packed
activation tensor it produces, per group, the post-RHT columnwise amax and the raw
rowwise amax, without materializing the transform.

| model | projection | E | M | N | cutedsl_us | triton_us | speedup | cutedsl_gbps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| debugmodel | gate/up (w1/w3) | 4 | 256 | 256 | 11.22 | 8.84 | 0.79x | 46.7 |
| debugmodel | down (w2) | 4 | 256 | 256 | 11.20 | 8.83 | 0.79x | 46.8 |
| 16B | gate/up (w1/w3) | 4 | 1408 | 2048 | 10.16 | 18.18 | 1.79x | 2270.7 |
| 16B | down (w2) | 4 | 2048 | 1408 | 9.77 | 18.18 | 1.86x | 2362.2 |
| 671B | gate/up (w1/w3) | 4 | 2048 | 7168 | 22.99 | 39.71 | 1.73x | 5109.2 |
| 671B | down (w2) | 4 | 7168 | 2048 | 22.26 | 40.56 | 1.82x | 5276.9 |

#### Grouped Hadamard Quantize Row+Col

The grouped analog of `rht_quantize_row_col`. Consumes the per-group amaxes produced
above and writes rowwise flat buffers plus columnwise per-group views over one flat
columnwise buffer. Bandwidth counts the bfloat16 input read plus rowwise and columnwise
FP4 codes and swizzled FP8 scales. Use `--rounding rtne` or `--rounding rs` for one
mode; the default is both.

Round-to-nearest-even (`rtne`):

| model | projection | E | M | N | cutedsl_us | triton_us | speedup | cutedsl_gbps | pct_peak |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| debugmodel | gate/up (w1/w3) | 4 | 256 | 256 | 19.19 | 7.05 | 0.37x | 42.7 | 0.54 |
| debugmodel | down (w2) | 4 | 256 | 256 | 19.18 | 7.06 | 0.37x | 42.7 | 0.54 |
| 16B | gate/up (w1/w3) | 4 | 1408 | 2048 | 19.11 | 18.69 | 0.98x | 1885.8 | 23.79 |
| 16B | down (w2) | 4 | 2048 | 1408 | 17.95 | 18.68 | 1.04x | 2008.3 | 25.33 |
| 671B | gate/up (w1/w3) | 4 | 2048 | 7168 | 51.68 | 85.75 | 1.66x | 3551.0 | 44.79 |
| 671B | down (w2) | 4 | 7168 | 2048 | 52.77 | 85.03 | 1.61x | 3477.2 | 43.86 |

Stochastic rounding (`rs`):

| model | projection | E | M | N | cutedsl_us | triton_us | speedup | cutedsl_gbps | pct_peak |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| debugmodel | gate/up (w1/w3) | 4 | 256 | 256 | 42.45 | 11.73 | 0.28x | 19.3 | 0.24 |
| debugmodel | down (w2) | 4 | 256 | 256 | 42.48 | 11.77 | 0.28x | 19.3 | 0.24 |
| 16B | gate/up (w1/w3) | 4 | 1408 | 2048 | 41.50 | 36.36 | 0.88x | 868.5 | 10.95 |
| 16B | down (w2) | 4 | 2048 | 1408 | 40.97 | 36.38 | 0.89x | 879.8 | 11.10 |
| 671B | gate/up (w1/w3) | 4 | 2048 | 7168 | 127.50 | 164.17 | 1.29x | 1439.2 | 18.15 |
| 671B | down (w2) | 4 | 7168 | 2048 | 128.85 | 163.62 | 1.27x | 1424.1 | 17.96 |

Stochastic rounding costs roughly 2.5x RTNE on CuteDSL and 2x on Triton. Both run a
10-round Philox per random word and draw the identical stream; CuteDSL issues half as
many calls (Triton discards two of every four it computes), which is what keeps it
ahead at 671B despite the extra ALU work.

#### Grouped 2D Weight Quantize (`cutedsl_group_weight_quantize_2d` vs Triton)

Quantizes dense `(E, M, N)` BF16 expert weights with 2D 16x16 block scaling, emitting
rowwise and columnwise (`W.T`) FP4 codes and swizzled scale factors for every expert.
No RHT. Bandwidth accounts for the bfloat16 read plus both FP4 outputs and both scale
writes.

| model | projection | E | M (out) | N (in) | cutedsl_us | triton_us | speedup | cutedsl_gbps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| debugmodel | gate/up (w1/w3) | 4 | 256 | 256 | 9.02 | 5.97 | 0.66x | 90.8 |
| debugmodel | down (w2) | 4 | 256 | 256 | 9.02 | 5.98 | 0.66x | 90.8 |
| 16B | gate/up (w1/w3) | 4 | 1408 | 2048 | 21.39 | 23.18 | 1.08x | 1685.0 |
| 16B | down (w2) | 4 | 2048 | 1408 | 22.50 | 23.38 | 1.04x | 1601.8 |
| 671B | gate/up (w1/w3) | 4 | 2048 | 7168 | 81.00 | 99.41 | 1.23x | 2265.3 |
| 671B | down (w2) | 4 | 7168 | 2048 | 80.87 | 98.01 | 1.21x | 2269.1 |

The 16B `gate/up` row has `M = 1408`, which is `% 128` but not `% 256`, so it compiles
the 128-row CuteDSL supertile. That path runs near Triton parity rather than the
1.2x the 256-row shapes get: the no-MMA weight path has no per-supertile pipeline for
the shorter height to amortize.

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
- The ranking inverts at large `E`; see the `E = 4` note at the top of this section.
- `kernel_time_us` profiles a hot loop over one buffer and does not flush L2, so shapes
  under L2 capacity read partly from cache and the absolute TB/s is optimistic. Both
  backends are pure-read reductions and lose the cache in the same proportion, so the
  speedup survives — 1.52x hot against 1.54x L2-flushed at 671B. Read the speedup
  column and treat bandwidth as an upper bound.

| model | projection | E | M | N | vector_norm_us | triton_us | speedup | triton_gbps |
|---|---|---:|---:|---:|---:|---:|---|---:|
| debugmodel | gate/up (w1/w3) | 4 | 256 | 256 | 9.205 | 3.859 | 2.39x | 135.9 |
| debugmodel | down (w2) | 4 | 256 | 256 | 9.173 | 3.988 | 2.30x | 131.5 |
| 16B | gate/up (w1/w3) | 4 | 1408 | 2048 | 11.336 | 5.894 | 1.92x | 3913.6 |
| 16B | down (w2) | 4 | 2048 | 1408 | 11.340 | 5.928 | 1.91x | 3891.7 |
| 671B | gate/up (w1/w3) | 4 | 2048 | 7168 | 33.689 | 19.931 | 1.69x | 5892.4 |
| 671B | down (w2) | 4 | 7168 | 2048 | 33.776 | 19.998 | 1.69x | 5872.6 |

`nvfp4_linear` uses this same op at `E = 1` on `W.unsqueeze(0)` — nothing in the kernel
is expert-specific beyond the `program_id(1)` base.
