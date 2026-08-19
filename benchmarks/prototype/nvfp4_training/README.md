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

Under RTNE the two backends produce **bitwise identical output** — FP4 codes and FP8 scale
factors — so they are drop-in interchangeable and the choice is purely a performance one.
Under stochastic rounding the CuteDSL kernel draws one Philox counter per 16-element block
and consumes all four output words, rather than reproducing triton's per-packed-byte
stride, so its SR codes are a different, equally valid stream.

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

Run environment: NVIDIA GB200, CUDA 13.4, PyTorch 2.15.0a0+git0f3e7e2, Triton 3.8.0,
nvidia-cutlass-dsl 4.5.2. Every table in this section was re-measured together on that
environment, at the current tree, as the median of three full script passes; earlier
revisions of this file used a different torch build, so their absolute numbers are not
comparable to these on either backend.

### Hadamard Amax (`cutedsl_rht_amax` vs `triton_rht_amax`)

| Model | Shape | M | N | cutedsl_kernel_us | triton_kernel_us | speedup | cutedsl_gbps |
|---|---|---:|---:|---:|---:|---:|---:|
| Llama 3 8B | hidden-state input | 2048 | 4096 | 8.58 | 16.23 | 1.89x | 1955.9 |
| Llama 3 8B | mlp.down input | 2048 | 14336 | 13.29 | 25.39 | 1.91x | 4419.6 |
| Llama 3 70B | hidden-state input | 2048 | 8192 | 10.98 | 20.04 | 1.83x | 3056.7 |
| Llama 3 70B | mlp.down input | 2048 | 28672 | 24.76 | 39.72 | 1.60x | 4744.1 |

### Hadamard Quantize Row+Col (`cutedsl_rht_quantize_row_col` vs `triton_rht_quantize_row_col`, RTNE)

`--math all` emits the `exact` and `fast` rows for a shape from one invocation, so
`fast/exact` is each backend against its own exact time in the same run. Both backends
implement fast math and stay bitwise identical to each other and to TE in either mode, so
the choice is free at the same numerics; `fast` is the `nvfp4_linear` default.

| Model | Shape | M | N | math | cutedsl_kernel_us | triton_kernel_us | speedup | cutedsl_gbps | pct_peak_bw | cutedsl fast/exact | triton fast/exact |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Llama 3 8B | hidden-state input | 2048 | 4096 | exact | 13.89 | 25.30 | 1.82x | 1887.1 | 23.8 | — | — |
| Llama 3 8B | hidden-state input | 2048 | 4096 | fast | 8.87 | 22.21 | 2.50x | 2955.7 | 37.3 | 1.57x | 1.14x |
| Llama 3 8B | mlp.down input | 2048 | 14336 | exact | 35.10 | 61.30 | 1.75x | 2614.1 | 33.0 | — | — |
| Llama 3 8B | mlp.down input | 2048 | 14336 | fast | 19.45 | 51.46 | 2.65x | 4717.2 | 59.5 | 1.80x | 1.19x |
| Llama 3 70B | hidden-state input | 2048 | 8192 | exact | 22.50 | 38.40 | 1.71x | 2329.8 | 29.4 | — | — |
| Llama 3 70B | hidden-state input | 2048 | 8192 | fast | 13.52 | 32.86 | 2.43x | 3877.7 | 48.9 | 1.66x | 1.17x |
| Llama 3 70B | mlp.down input | 2048 | 28672 | exact | 69.38 | 113.02 | 1.63x | 2644.9 | 33.4 | — | — |
| Llama 3 70B | mlp.down input | 2048 | 28672 | fast | 36.39 | 92.91 | 2.55x | 5042.9 | 63.6 | 1.91x | 1.22x |

Fast math is worth far more to CuteDSL (1.57-1.91x) than to Triton (1.14-1.22x), and
worth more on the linear path than on the grouped one below (1.27-1.38x). CuteDSL at 28672
goes from 33.4% to 63.6% of peak bandwidth. The split is structural: fast math removes
the bfloat16 round-through of the RHT accumulator and the `div.rn` reciprocal, and those
are a much larger share of what the CuteDSL epilogue does once its other work is fused
(see **Columnwise scale-factor store packing** below, whose win also landed almost
entirely on fast math).

**Which to use: fast.** It is the default, and the trade is one-sided. It buys 1.57-1.91x
on the linear path and 1.27-1.38x grouped for CuteDSL, and costs 30.2-32.6 dB SQNR
against exact (30.4-31.9 grouped) — roughly 10 dB quieter than NVFP4's own ~20 dB
quantization noise, so the perturbation is well under the error the format already
carries. Both backends stay bitwise identical to each other and to TE in either mode, so
the choice does not fork `AUTO` against `TRITON`. The one property worth knowing is the
error's shape: skipping the bfloat16 round-through moves a value by ~2**-9, but an element
near an E2M1 midpoint then flips a whole FP4 step, so the difference is large and rare
(~1.2% of code bytes) rather than small and uniform. Set `use_fast_math=False` when
bisecting a numerics or loss regression through this branch, to take it off the table as a
variable — `NVFP4TrainingConfig` documents the exact config that recovers prior numerics.

### 2D Weight Quantize (`cutedsl_weight_quantize_2d` vs `triton_weight_quantize_2d`, no RHT)

Both kernels emit 2D 16x16 weight block scaling.
Requires `out_features % 128 == 0`.

| Model | Weight | M (out) | N (in) | cutedsl_kernel_us | triton_kernel_us | speedup | cutedsl_gbps |
|---|---|---:|---:|---:|---:|---:|---:|
| Llama 3 8B | mlp.gate/up | 14336 | 4096 | 56.60 | 154.39 | 2.73x | 3242.0 |
| Llama 3 8B | mlp.down | 4096 | 14336 | 57.03 | 154.32 | 2.71x | 3217.4 |
| Llama 3 70B | mlp.gate/up | 28672 | 8192 | 202.54 | 577.16 | 2.85x | 3624.0 |
| Llama 3 70B | mlp.down | 8192 | 28672 | 202.08 | 577.10 | 2.86x | 3632.2 |

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
| 671B | 256 | 64 | 4 | 7168 | 2048 |

Every benchmark below runs at `E = 4`: the target deployment is high expert parallelism,
so a rank holds a handful of experts and the per-model M/N at small `E` is the
representative shape. For 671B that is exactly the local expert count at its EP=64
training layout; for debugmodel and 16B it is smaller than the table's `E (local)`.
`--experts` overrides it where the script takes one.

Run environment for every table in this section: NVIDIA GB200, CUDA 13.4, PyTorch
2.15.0a0+git0f3e7e2, Triton 3.8.0, nvidia-cutlass-dsl 4.5.2. All of them were
re-measured together on that environment, at the current tree, as the median of three
full script passes. The **CuteDSL comparison vs TransformerEngine** subsection below is
the exception — see its provenance note.

### CuteDSL kernels — comparison vs Triton

Each grouped `bench_*` script runs **both backends** on the same shapes and reports the
speedup. Under RTNE the two produce **bitwise identical output** — codes and scale
factors — so the choice is purely a performance one. Under stochastic rounding the
CuteDSL kernel draws its Philox stream differently (one counter per 16-element block, all
four words consumed) and so produces different, statistically equivalent codes; it is
checked on reconstruction SQNR, unbiasedness, and reproducibility instead.

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
| debugmodel | gate/up (w1/w3) | 4 | 256 | 256 | 11.10 | 8.80 | 0.79x | 47.2 |
| debugmodel | down (w2) | 4 | 256 | 256 | 11.09 | 8.77 | 0.79x | 47.3 |
| 16B | gate/up (w1/w3) | 4 | 1408 | 2048 | 10.20 | 18.10 | 1.77x | 2261.7 |
| 16B | down (w2) | 4 | 2048 | 1408 | 9.82 | 18.12 | 1.85x | 2348.3 |
| 671B | gate/up (w1/w3) | 4 | 2048 | 7168 | 22.95 | 39.91 | 1.74x | 5117.1 |
| 671B | down (w2) | 4 | 7168 | 2048 | 22.42 | 40.54 | 1.81x | 5237.5 |

#### Grouped Hadamard Quantize Row+Col

The grouped analog of `rht_quantize_row_col`. Consumes the per-group amaxes produced
above and writes rowwise flat buffers plus columnwise per-group views over one flat
columnwise buffer. Bandwidth counts the bfloat16 input read plus rowwise and columnwise
FP4 codes and swizzled FP8 scales. Use `--rounding rtne` or `--rounding rs` for one
mode; the default is both.

Split by rounding mode, since a caller picks that once for the whole run. Within each
table `--math all` puts the `exact` and `fast` rows for a shape adjacent, because that
*is* the choice a caller makes per config; `fast/exact` is each backend against the exact
row directly above it, from the same run.

Round-to-nearest-even (`rtne`):

| model | projection | E | M | N | math | cutedsl_us | triton_us | speedup | cutedsl_gbps | pct_peak | cutedsl fast/exact | triton fast/exact |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| debugmodel | gate/up (w1/w3) | 4 | 256 | 256 | exact | 19.53 | 6.68 | 0.34x | 42.0 | 0.53 | — | — |
| debugmodel | gate/up (w1/w3) | 4 | 256 | 256 | fast | 14.83 | 5.23 | 0.35x | 55.2 | 0.70 | 1.32x | 1.28x |
| debugmodel | down (w2) | 4 | 256 | 256 | exact | 19.50 | 6.67 | 0.34x | 42.0 | 0.53 | — | — |
| debugmodel | down (w2) | 4 | 256 | 256 | fast | 14.82 | 5.23 | 0.35x | 55.3 | 0.70 | 1.32x | 1.28x |
| 16B | gate/up (w1/w3) | 4 | 1408 | 2048 | exact | 19.37 | 19.67 | 1.02x | 1861.1 | 23.47 | — | — |
| 16B | gate/up (w1/w3) | 4 | 1408 | 2048 | fast | 14.38 | 14.43 | 1.00x | 2506.7 | 31.62 | 1.35x | 1.36x |
| 16B | down (w2) | 4 | 2048 | 1408 | exact | 18.09 | 19.66 | 1.09x | 1993.1 | 25.14 | — | — |
| 16B | down (w2) | 4 | 2048 | 1408 | fast | 13.14 | 14.41 | 1.09x | 2742.6 | 34.59 | 1.38x | 1.36x |
| 671B | gate/up (w1/w3) | 4 | 2048 | 7168 | exact | 53.95 | 87.49 | 1.62x | 3401.1 | 42.90 | — | — |
| 671B | gate/up (w1/w3) | 4 | 2048 | 7168 | fast | 39.73 | 64.16 | 1.62x | 4618.4 | 58.25 | 1.36x | 1.36x |
| 671B | down (w2) | 4 | 7168 | 2048 | exact | 55.23 | 86.91 | 1.57x | 3322.3 | 41.91 | — | — |
| 671B | down (w2) | 4 | 7168 | 2048 | fast | 41.14 | 63.71 | 1.55x | 4460.6 | 56.26 | 1.34x | 1.36x |

Stochastic rounding (`rs`):

| model | projection | E | M | N | math | cutedsl_us | triton_us | speedup | cutedsl_gbps | pct_peak | cutedsl fast/exact | triton fast/exact |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| debugmodel | gate/up (w1/w3) | 4 | 256 | 256 | exact | 25.56 | 10.84 | 0.42x | 32.0 | 0.40 | — | — |
| debugmodel | gate/up (w1/w3) | 4 | 256 | 256 | fast | 20.03 | 9.48 | 0.47x | 40.9 | 0.52 | 1.28x | 1.14x |
| debugmodel | down (w2) | 4 | 256 | 256 | exact | 25.59 | 10.83 | 0.42x | 32.0 | 0.40 | — | — |
| debugmodel | down (w2) | 4 | 256 | 256 | fast | 20.03 | 9.47 | 0.47x | 40.9 | 0.52 | 1.28x | 1.14x |
| 16B | gate/up (w1/w3) | 4 | 1408 | 2048 | exact | 25.57 | 35.51 | 1.39x | 1409.4 | 17.78 | — | — |
| 16B | gate/up (w1/w3) | 4 | 1408 | 2048 | fast | 20.16 | 31.19 | 1.55x | 1788.3 | 22.56 | 1.27x | 1.14x |
| 16B | down (w2) | 4 | 2048 | 1408 | exact | 24.45 | 35.61 | 1.46x | 1474.5 | 18.60 | — | — |
| 16B | down (w2) | 4 | 2048 | 1408 | fast | 18.78 | 31.20 | 1.66x | 1919.4 | 24.21 | 1.30x | 1.14x |
| 671B | gate/up (w1/w3) | 4 | 2048 | 7168 | exact | 74.75 | 159.96 | 2.14x | 2454.8 | 30.96 | — | — |
| 671B | gate/up (w1/w3) | 4 | 2048 | 7168 | fast | 54.93 | 141.28 | 2.57x | 3340.7 | 42.14 | 1.36x | 1.13x |
| 671B | down (w2) | 4 | 7168 | 2048 | exact | 76.51 | 159.30 | 2.08x | 2398.3 | 30.25 | — | — |
| 671B | down (w2) | 4 | 7168 | 2048 | fast | 56.51 | 140.86 | 2.49x | 3247.3 | 40.96 | 1.35x | 1.13x |

Grouped fast math is flatter than linear: CuteDSL gains 1.27-1.38x across every shape and
both rounding modes, matching the "about 25% of the quantize stage" the grouped op's
docstring claims. Triton gains 1.28-1.36x under RTNE but only 1.13-1.14x under SR, where
the Philox work it still carries dominates what fast math removes.

Stochastic rounding now costs CuteDSL about 1.4x its own RTNE time against Triton's
1.8x, which is what turns a 1.6x RTNE lead into a 2.1x SR lead at 671B. Both run the
same 10-round Philox generator, but no longer the same counter stream: Triton draws one
word per packed byte and discards two of every four it computes, while CuteDSL draws one
counter per 16-element block and consumes all four words -- 34 multiplies where the
triton-compatible stride cost 124.

#### Grouped 2D Weight Quantize (`cutedsl_group_weight_quantize_2d` vs Triton)

Quantizes dense `(E, M, N)` BF16 expert weights with 2D 16x16 block scaling, emitting
rowwise and columnwise (`W.T`) FP4 codes and swizzled scale factors for every expert.
No RHT. Bandwidth accounts for the bfloat16 read plus both FP4 outputs and both scale
writes.

| model | projection | E | M (out) | N (in) | cutedsl_us | triton_us | speedup | cutedsl_gbps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| debugmodel | gate/up (w1/w3) | 4 | 256 | 256 | 6.76 | 6.33 | 0.94x | 121.1 |
| debugmodel | down (w2) | 4 | 256 | 256 | 6.77 | 6.31 | 0.93x | 121.0 |
| 16B | gate/up (w1/w3) | 4 | 1408 | 2048 | 18.72 | 25.25 | 1.35x | 1925.6 |
| 16B | down (w2) | 4 | 2048 | 1408 | 16.31 | 25.22 | 1.55x | 2209.5 |
| 671B | gate/up (w1/w3) | 4 | 2048 | 7168 | 59.76 | 109.31 | 1.83x | 3070.8 |
| 671B | down (w2) | 4 | 7168 | 2048 | 60.02 | 107.99 | 1.80x | 3057.5 |

The 16B `gate/up` row has `M = 1408`, which is `% 128` but not `% 256`, so it compiles
the 128-row CuteDSL supertile. It gets 1.35x where the same-size `down` shape, which is
`% 256` and takes the 256-row path, gets 1.55x: the no-MMA weight path has no
per-supertile pipeline for the shorter height to amortize.

### CuteDSL comparison vs TransformerEngine

DeepSeek-V3 671B FFN shapes, excluding attention GEMMs, at `E = 4`. Run environment:
NVIDIA GB200, CUDA 13.4, PyTorch 2.15.0a0+git0f3e7e2, nvidia-cutlass-dsl 4.5.2, and
TransformerEngine 2.19.0.dev0 built from
[NVIDIA/TransformerEngine@172bd93](https://github.com/NVIDIA/TransformerEngine/commit/172bd93773ad6ee4ba44b460b7f10ef42fc89d57)
("[Common] Ensure quantization kernels handle noop properly", #3271) — an untagged
main-branch commit, not a release. Both sides were measured together.

**Provenance.** No checked-in script reproduces this section: the benchmark modules cover
CuteDSL-vs-Triton only, and these shapes are the DeepSeek-V3 `(2048, 7168)` /
`(7168, 2048)` pair that `--shape-set` does not offer. Every table here is retained from
the run described above and **predates the last two optimization rounds** (bf16 widening
by shift, 64-bit row-code stores). The CuteDSL columns are therefore a lower bound —
the refreshed CuteDSL-vs-Triton tables above supersede them for torchao-side absolute
times. TE-relative ratios are still the best available, because both sides were timed in
the same run; re-measuring only one side would be worse than leaving the pair alone.

Times are CUDA kernel self-time in microseconds, with 15 warmups and 50 measured
iterations. Memcpy and memset events are excluded. The grouped activation comparison
times the complete post-RHT amax and row/column quantization pipeline: one
`cutedsl_group_rht_amax` followed by one `cutedsl_group_rht_quantize_row_col`, versus
TransformerEngine's `split_quantize` counterpart, which computes its post-RHT amax
internally.

| projection | E | M | N | math | rounding | CuteDSL pipeline (us) | TE pipeline (us) | TE speedup |
|---|---:|---:|---:|---|---|---:|---:|---:|
| gate/up (w1/w3) | 4 | 2048 | 7168 | standard | RTNE | 86.22 | 64.03 | 1.35x |
| gate/up (w1/w3) | 4 | 2048 | 7168 | standard | SR | 106.44 | 87.90 | 1.21x |
| gate/up (w1/w3) | 4 | 2048 | 7168 | fast | RTNE | 70.44 | 55.66 | 1.27x |
| gate/up (w1/w3) | 4 | 2048 | 7168 | fast | SR | 86.24 | 77.74 | 1.11x |
| down (w2) | 4 | 7168 | 2048 | standard | RTNE | 88.35 | 62.59 | 1.41x |
| down (w2) | 4 | 7168 | 2048 | standard | SR | 108.59 | 86.67 | 1.25x |
| down (w2) | 4 | 7168 | 2048 | fast | RTNE | 71.54 | 54.73 | 1.31x |
| down (w2) | 4 | 7168 | 2048 | fast | SR | 87.43 | 76.47 | 1.14x |

The linear (non-grouped) activation path, `(2048, 7168)`, measured the same way — one
`cutedsl_rht_amax` plus one `cutedsl_rht_quantize_row_col` against a single-tensor
`NVFP4Quantizer` call:

| math | rounding | CuteDSL pipeline (us) | TE pipeline (us) | TE speedup |
|---|---|---:|---:|---:|
| standard | RTNE | 29.82 | 18.61 | 1.60x |
| standard | SR | 36.69 | 31.33 | 1.17x |
| fast | RTNE | 21.75 | 15.95 | 1.36x |
| fast | SR | 28.24 | 24.63 | 1.15x |

The SR rows are the ones that moved. Before the optimization below, TE led the SR
pipeline by 2.06x (gate/up) and 2.12x (down) at standard math; it now leads by 1.21x
and 1.26x. The remaining gap is no longer SR-specific: SR costs CuteDSL 1.23x its own
RTNE pipeline against TE's 1.37x, so the SR path is now the more efficient of the two
relative to its own baseline, and what is left is the RTNE gap.

The corresponding standalone CuteDSL stage times:

| family | projection | math | rounding | amax (us) | quantize (us) |
|---|---|---|---|---:|---:|
| grouped | gate/up (w1/w3) | standard | RTNE | 23.34 | 57.04 |
| grouped | gate/up (w1/w3) | standard | SR | 23.30 | 75.91 |
| grouped | gate/up (w1/w3) | fast | RTNE | 23.32 | 42.67 |
| grouped | gate/up (w1/w3) | fast | SR | 23.31 | 57.24 |
| grouped | down (w2) | standard | RTNE | 23.40 | 58.69 |
| grouped | down (w2) | standard | SR | 23.29 | 77.68 |
| grouped | down (w2) | fast | RTNE | 23.26 | 43.77 |
| grouped | down (w2) | fast | SR | 23.33 | 58.75 |
| linear | (2048, 7168) | standard | RTNE | 9.63 | 19.50 |
| linear | (2048, 7168) | standard | SR | 9.69 | 25.93 |
| linear | (2048, 7168) | fast | RTNE | 9.64 | 11.49 |
| linear | (2048, 7168) | fast | SR | 9.70 | 17.96 |

Breaking TE's side out by kernel puts the remaining gap in one place. The grouped amax
**wins** (23.34 against TE's 25.81) and the linear amax trails at 9.63 against 4.47; the
quantize kernel is behind on both paths, 57.04 against 38.32 grouped and 19.50 against
12.98 linear. That quantize ratio -- 1.49x and 1.51x -- is nearly identical across the
two families, which is what one would expect from them sharing `_quant16_from_amax`.

TransformerEngine has no grouped 2D weight quantization kernel, so the weight
comparison is one `cutedsl_group_weight_quantize_2d` launch over all four experts
against four times TE's single-expert time, using
`NVFP4Quantizer(with_2d_quantization=True)` so both sides emit 16x16 block scaling.

**Read the TE column carefully.** A TE 2D call launches three kernels, not one:

| TE kernel | (2048, 7168) | (7168, 2048) |
|---|---:|---:|
| `quantize_transpose_kernel` | 13.80 | 13.76 |
| `amax_kernel` | 5.40 | 5.39 |
| `zero_amax_kernel` | 1.34 | 1.34 |
| total | 20.53 | 20.49 |

`cutedsl_weight_quantize_2d` and its grouped twin consume a **precomputed** amax, so the
apples-to-apples figure is TE's quantize kernel alone. Comparing against TE's full call
instead would credit CuteDSL with skipping a pass it never runs.

| kernel | projection | CuteDSL (us) | TE quantize only (us) | TE speedup |
|---|---|---:|---:|---:|
| 2D linear | gate/up | 16.33 | 13.80 | 1.18x |
| 2D linear | down | 16.36 | 13.76 | 1.19x |
| 2D grouped, E=4 | gate/up | 60.26 | 55.20 (x4) | 1.09x |
| 2D grouped, E=4 | down | 60.57 | 55.04 (x4) | 1.10x |

At the op level, where each side computes its own amax, CuteDSL wins instead: TE pays
6.74 us for `amax_kernel` plus `zero_amax_kernel` where torchao's weight amax is
cheaper (see **Grouped Weight Amax** below).

**Neither side has a 2D fast path**, and this is structural rather than an omission.
Fast math buys two things on the 1D path: skipping the bfloat16 round-through of the
tcgen05 RHT accumulator, and `rcp_approx` in place of `div.rn` for the encode
reciprocal. Without an RHT there is no accumulator, so the first cannot apply, and the
second is one instruction per 16-element block in a kernel that is otherwise load/store
bound. Measured, `NVTE_USE_FAST_MATH=1` moves TE's three kernels by at most 0.02 us
(13.78/5.39/1.34), and the public CuteDSL 2D wrapper exposes no `use_fast_math` at all
(`_cutedsl_kernels_impl.py` asserts `not (fast_math and not apply_rht)`).

#### 1D linear optimization sweep

The default-math RTNE `(2048, 7168)` sentinel measured the standalone
`cutedsl_rht_quantize_row_col` kernel with 15 warmups and 50 profiler iterations.
None of the first three bounded experiments met the approximately 2% acceptance
threshold, so all source changes were reverted.

| experiment | median (us) | change vs baseline | outcome |
|---|---:|---:|---|
| baseline, 256-row supertile | 23.6610 | — | retained |
| existing 128-row supertile | 23.6493 | 0.05% faster | reverted |
| packed FP32x2 multiply and FP4 conversion | 23.3460 | 1.33% faster | reverted |
| double-buffered column TMA stores | 23.7940 | 0.56% slower | reverted |

The packed conversion experiment confirmed the epilogue hypothesis directionally,
but its gain was too small to justify a shared primitive that would also broaden the
2D kernel's validation surface. The store experiment showed that, for the three
persistent iterations used by this sentinel, a second column staging buffer costs
more than the overlap saves. Remaining 1D linear work should focus on a larger
structural reduction in row-epilogue work rather than tile geometry or store staging.

#### 1D grouped optimization sweep

The default-math RTNE `E=4`, per-expert `(2048, 7168)` sentinel measured the
standalone `cutedsl_group_rht_quantize_row_col` kernel with the same 15/50 timing
parameters. The existing kernel already caches each binary group lookup across an
8-tile scheduler work item.

| experiment | time (us) | change vs baseline | outcome |
|---|---:|---:|---|
| baseline | 61.3872 | — | retained |
| aligned two-word row-code vector stores | 62.2391 | 1.39% slower | reverted |
| 16-tile scheduler work items | 77.6977 | 26.57% slower | reverted |

Autovectorizing the already aligned output pair introduced register/copy overhead.
Doubling scheduler work length reduced CLC and lookup work but severely damaged
persistent load balance. Remaining grouped work should preserve the 8-tile scheduler
and focus on eliminating scalar scale-factor scatters or capacity handling when the
logical packed length is smaller than capacity.

#### 2D linear and grouped optimization results

The 2D linear `(2048, 7168)` baseline was 24.7719 us. Selecting the existing
128-row/416-thread geometry produced 24.7887 us and was reverted. A shared exact-mode
RTNE primitive now scales eight BF16-origin values with four packed `mul.f32x2`
instructions and immediately converts them to four packed FP4 bytes. It removes the
intermediate 16-element scaled FP32 tensor when no RHT rounding or stochastic rounding
is required.

The primitive reduced the final 2D linear gate/up median to 23.2973 us (5.95%) and is
bitwise identical to Triton. Grouped 2D inherits the same implementation on both
output orientations: gate/up improved from 92.74 to 87.3549 us (5.81%), and down from
92.57 to 87.4361 us (5.55%). Relative to TE's single-expert time multiplied by four,
the grouped gap narrows from about 1.70x to 1.60x. The linear gap is about 1.71x.

#### Current standalone kernel medians

Each entry is the median of three samples; every sample uses 15 warmups and 50 timed
CUDA-profiler iterations and reports device kernel self-time in microseconds. The 1D
rows measure the fused quantize stage with precomputed amaxes; the 2D rows consume
precomputed weight amaxes. These supersede every before/after table in the optimization
sections below, which record what each round changed rather than where the kernels stand.

| family | projection | math | rounding | median (us) | source |
|---|---|---|---|---:|---|
| 1D linear | gate/up | standard | RTNE | 19.42 | out of band |
| 1D linear | gate/up | standard | SR | 25.82 | out of band |
| 1D linear | gate/up | fast | RTNE | 11.39 | out of band |
| 1D linear | gate/up | fast | SR | 17.77 | out of band |
| 1D grouped, E=4 | gate/up | standard | RTNE | 53.95 | script |
| 1D grouped, E=4 | gate/up | standard | SR | 74.75 | script |
| 1D grouped, E=4 | gate/up | fast | RTNE | 39.73 | script |
| 1D grouped, E=4 | gate/up | fast | SR | 54.93 | script |
| 1D grouped, E=4 | down | standard | RTNE | 55.23 | script |
| 1D grouped, E=4 | down | standard | SR | 76.51 | script |
| 1D grouped, E=4 | down | fast | RTNE | 41.14 | script |
| 1D grouped, E=4 | down | fast | SR | 56.51 | script |
| amax linear | (2048, 7168) | — | — | 9.63 | out of band |
| amax grouped, E=4 | gate/up | — | — | 22.95 | script |
| 2D linear | gate/up | — | RTNE | 16.33 | out of band |
| 2D linear | down | — | RTNE | 16.36 | out of band |
| 2D grouped, E=4 | gate/up | — | RTNE | 59.76 | script |
| 2D grouped, E=4 | down | — | RTNE | 60.02 | script |

`source = script` rows come from the commands below and are reproducible. The `1D linear`,
`amax linear` and `2D linear` rows are the DeepSeek-V3 `(2048, 7168)` / `(7168, 2048)`
shapes, which **no checked-in script emits** — `--shape-set` offers only `sweep` and
`representative-models`, and neither contains `N = 7168` at `M = 2048`. Those rows were
measured out of band and predate the last two optimization rounds (bf16 widening by shift
and the 64-bit row-code store), so treat them as a lower bound on current performance.
Giving the linear scripts a DeepSeek-V3 shape set would make the whole table reproducible.

The exact sentinel commands use the public callables the benchmark modules show:

```bash
# Run each three times for the reported medians (15 warmups / 50 iterations are the
# defaults in bench_utils.kernel_time_us).
python -m benchmarks.prototype.nvfp4_training.bench_hadamard_quantize_row_col --math all
python -m benchmarks.prototype.nvfp4_training.bench_group_rht_quantize_row_col --experts 4 --math all
python -m benchmarks.prototype.nvfp4_training.bench_quantize_2d
python -m benchmarks.prototype.nvfp4_training.bench_group_quantize_2d
```

`bench_group_quantize_2d` takes no arguments (`LOCAL_EXPERTS = 4` is hardcoded), and
`bench_hadamard_quantize_row_col` benchmarks RTNE only — `--rounding` exists on the
grouped script alone. Both RHT quantize scripts take `--math {exact,fast,all}` (default
`all`) and report a `math` column; the fast rows are now reproducible from the checked-in
scripts rather than measured out of band. `NVTE_USE_FAST_MATH` is a TransformerEngine
variable with no effect on a pure torchao run — the torchao equivalent is `--math fast`,
which passes `use_fast_math=True` to both backends.

### Epilogue SMEM-read and packing optimization

A second pass profiled the compiled SASS instead of guessing at geometry
(`CUTE_DSL_KEEP=cubin` + `cuobjdump -sass -res-usage`). The 2D weight kernel used
**49 registers and no shared memory beyond its dynamic allocation**, so occupancy was
never register-limited; instead **53% of its 3432 static instructions were address
arithmetic** feeding 256 scalar `LDS.U16`. That measurement, not the thread count,
explained the TE gap, and it identified three changes:

1. **Vectorized rowwise SMEM reads.** The A operand uses an `MN_SW128` atom, so the N
   grain is contiguous and each 16-element block is one vector copy. The row epilogue
   was re-deriving a swizzled address per element; it now uses the same
   `cute.autovec_copy` shape the grouped 1D row epilogue already used. 128 `LDS.U16`
   became 16 `LDS.128`, and the static instruction count fell to 3048.
2. **Paired columnwise SMEM reads.** A column thread now owns the adjacent N-row pair
   `(nrow, nrow+1)` and a quarter of the col-group blocks, rather than one row and a
   half. Because N is the contiguous mode the pair is one 32-bit load, so a warp moves
   the full 128 B instead of 64 B; and because both rows sit in the same 16x16 block
   they share one amax, one E4M3 scale and one exact reciprocal, halving the per-block
   scale work. 128 `LDS.U16` became 64 32-bit `LDS`; 2712 static instructions.
   The 16x16 amax butterfly correspondingly drops to the 8-lane offsets (4/2/1) after
   an in-register fold of the pair.
3. **Fused RHT-accumulator epilogue.** The 1D columnwise path was the one place the
   earlier `_mul_cvt_rn_e2m1x8_f32` win had not been applied: it rounded the tcgen05
   accumulator through bfloat16, re-widened by shift/mask, multiplied scalar-wise and
   clamped, then packed. `_mul_cvt_rn_e2m1x8_acc_f32` folds all of that into one asm
   block; the explicit `min.xorsign.abs` clamp is dropped because
   `cvt.rn.satfinite.e2m1x2.f32` already saturates at +-6, matching what TE's
   `mul_cvt_bf16_to_fp4_8x_round_to_nearest` relies on.

| family | projection | math | rounding | before (us) | after (us) | change |
|---|---|---|---|---:|---:|---:|
| 1D linear | gate/up | standard | RTNE | 23.2411 | 19.5444 | 15.9% faster |
| 1D linear | gate/up | standard | SR | 51.0504 | 46.4310 | 9.0% faster |
| 1D linear | gate/up | fast | RTNE | 18.8167 | 14.8403 | 21.1% faster |
| 1D linear | gate/up | fast | SR | 34.6773 | 33.0158 | 4.8% faster |
| 1D linear | down | standard | RTNE | 23.3231 | 19.5522 | 16.2% faster |
| 1D linear | down | standard | SR | 50.6500 | 46.3109 | 8.6% faster |
| 1D linear | down | fast | RTNE | 18.7982 | 14.8500 | 21.0% faster |
| 1D linear | down | fast | SR | 34.5102 | 32.9883 | 4.4% faster |
| 1D grouped, E=4 | gate/up | standard | RTNE | 61.6432 | 57.5612 | 6.6% faster |
| 1D grouped, E=4 | gate/up | standard | SR | 149.5823 | 150.9141 | 0.89% slower |
| 1D grouped, E=4 | gate/up | fast | RTNE | 45.3090 | 45.5933 | 0.63% slower |
| 1D grouped, E=4 | gate/up | fast | SR | 115.1210 | 114.9064 | 0.19% faster |
| 1D grouped, E=4 | down | standard | RTNE | 63.2194 | 58.4040 | 7.6% faster |
| 1D grouped, E=4 | down | standard | SR | 150.8909 | 151.1979 | 0.20% slower |
| 1D grouped, E=4 | down | fast | RTNE | 46.2625 | 46.1014 | 0.35% faster |
| 1D grouped, E=4 | down | fast | SR | 115.8542 | 115.8348 | 0.02% faster |
| 2D linear | gate/up | standard | RTNE | 23.2973 | 16.5665 | 28.9% faster |
| 2D linear | down | standard | RTNE | 23.2692 | 16.5806 | 28.7% faster |
| 2D grouped, E=4 | gate/up | standard | RTNE | 87.3549 | 60.5176 | 30.7% faster |
| 2D grouped, E=4 | down | standard | RTNE | 87.4361 | 60.7924 | 30.5% faster |

Aggregate device time across these 20 cases improved 8.3%; the largest regression was
0.89%, below the 2% limit. The aggregate is dominated by the four grouped SR rows,
which are unchanged because stochastic rounding takes neither new packing primitive
and the grouped 1D row epilogue already read SMEM with `cute.autovec_copy`.

Against TransformerEngine the 2D gap narrows substantially:

| projection | E | CuteDSL (us) | TE (us) | TE speedup before | TE speedup now |
|---|---:|---:|---:|---:|---:|
| 2D linear gate/up | 1 | 16.5665 | 13.64 | 1.71x | 1.21x |
| 2D grouped gate/up | 4 | 60.5176 | 54.56 (single x4) | 1.70x | 1.11x |
| 2D grouped down | 4 | 60.7924 | 54.62 (single x4) | 1.69x | 1.11x |

Every result above is bitwise identical to both oracles: the Triton backend (36 RTNE
cases) and the TE-derived PyTorch reference in
`test/prototype/moe_training/nvfp4_training/nvfp4_reference.py` (54 CuteDSL cases).
Stochastic rounding is exempt by design on both the linear and grouped paths — see the
notes above — and is covered instead by
`test_rht_quantize_rs_at_most_one_fp4_step_from_rtne` (every SR code within one FP4 step
of the bitwise-checked RTNE code), `test_group_rht_sr_reconstructs`,
`test_cutedsl_rht_quantize_sr_unbiased` / `test_group_rht_sr_unbiased`, and
`test_group_rht_rng_state_controls_stochastic_rounding`.

#### Rejected in this pass

| experiment | result | outcome |
|---|---:|---|
| hoisting the N-row slice out of the col inner loop | 3072 vs 3048 static instrs | reverted |
| two co-resident CTAs (128-row supertile, grid = 2x SMs) | 16.18 vs 16.48 us, 1.81% | reverted |

Registers (49 before, 56 after) and shared memory do permit two 416-thread CTAs per
SM, so the co-residency experiment ran as intended rather than being blocked — it
simply did not clear the 2% bar, and it would have forced the 128-row geometry on
every weight shape. This is distinct from the earlier 128-row supertile experiment,
which kept `GRID = min(NUM_SMS, num_super)` and therefore never changed occupancy at
all.

### Grouped stochastic-rounding optimization

A third pass again started from the SASS rather than a benchmark, diffing the grouped
kernel's SR and RTNE compilations directly (`_compile_group_fused_kernel(0, sr,
False)` under `CUTE_DSL_KEEP=cubin`, then `cuobjdump -res-usage -sass`). SR ran **2360
more static instructions than RTNE**, 5816 against 3456, and the mix said where they
were: `IMAD` + `LOP3` accounted for 1802 (76%) and `FMUL` + `FMNMX` for 384 (16%).
Both variants reported `REG:128` with no spill traffic, which killed a third
hypothesis — that SR was spilling against the shared `REG_COL`/`REG_ROW` budget — for
free, before any code was written.

1. **Fused SR multiply and clamp.** The SR path still materialized a 16-entry FP32
   register tensor plus 16 `mul.f32` and 16 `min.xorsign.abs.f32` before packing, the
   shape RTNE had already abandoned. `_mul_cvt_rs_e2m1x8_f32` and its `_acc` twin
   mirror the RTNE pair, replacing four `cvt.rn.satfinite.e2m1x2.f32` with two
   `cvt.rs.satfinite.e2m1x4.f32` that each consume one random word. The explicit clamp
   is dropped, relying on `.satfinite` as TE does; because `cvt.rs` perturbs the
   mantissa before saturating this was verified against the linear path's five
   triton SR bitwise-parity cases rather than assumed.
2. **Fused RTNE fast math.** The same `not sr and not fast_math` gate meant fast math
   was the last caller still materializing that tensor. Unifying the gate left six
   helpers unreachable (`_pack16`, both `cvt` pack4 primitives, `_cvt_rn_bf16x2_f32`,
   `_u32_as_f32`, `_min_xorsign_abs_f32`), removing 186 lines.
3. **One Philox draw per block.** `philox4` reproduces triton's per-packed-byte
   counter stride, running four full round schedules per 16-element block and keeping
   one word of each — 124 multiplies for 128 bits that a single draw produces.
   `philox4_all` keeps the existing launch-uniform `philox_prep` hoist (an advantage
   over TE, which recomputes its key schedule in-loop) and computes all four words in
   the last round: **34 multiplies instead of 124**. The counter is derived from tile
   coordinates rather than a running per-thread value, because this kernel's CLC
   scheduler is persistent and visit order is not fixed.

Change 3 is the first in this project that is not bitwise-preserving. Grouped SR codes
are now a different, equally valid stream; RTNE is untouched and remains bitwise
identical to both oracles.

Static instruction excess over the RTNE variant:

| | total | IMAD | LOP3 | FMUL | FMNMX |
|---|---:|---:|---:|---:|---:|
| before | +2360 | +1044 | +758 | +192 | +192 |
| after | +736 | +346 | +206 | 0 | 0 |

Sentinel, grouped `E=4` per-expert `(2048, 7168)`, median of three 15/50 samples:

| math | rounding | before (us) | after (us) | change |
|---|---|---:|---:|---:|
| standard | SR | 150.73 | 77.33 | **48.7% faster** |
| fast | SR | 116.45 | 57.77 | **50.4% faster** |
| standard | RTNE | 56.82 | 56.95 | unchanged |
| fast | RTNE | 45.60 | 42.55 | 6.7% faster |

### Linear stochastic-rounding optimization

The linear kernels shared `_quant16_from_amax` with the grouped ones, so they inherited
the fused SR multiply/clamp and the fused RTNE fast-math path automatically: linear SR
improved 10.5% and RTNE fast math 12.6% with no linear-specific change. A SASS diff then
showed the linear SR path was almost purely RNG-bound in its SR-specific work — `FMUL`
and `FMNMX` excess over RTNE was already zero, while `IMAD` + `LOP3` accounted for **91%**
of the 3872-instruction excess, a larger Philox burden than grouped carried before its
own fix.

Applying the same one-draw-per-block Philox to the linear path required dropping its
Triton SR bitwise parity, which was the project's only bitwise oracle for stochastic
rounding (there is no TE SR reference; TE's stream differs by construction). It was
dropped deliberately, and `test_cutedsl_vs_triton_stochastic_rounding_bitwise` was
replaced by extending `test_rht_quantize_rs_at_most_one_fp4_step_from_rtne` to both
backends and three tile geometries. That test is the stronger structural guard: it pins
every SR code to within one FP4 magnitude step of the RTNE code, and the RTNE code *is*
still bitwise-checked against both Triton and TE. `triton_tile_id`, which existed only to
invert Triton's `GROUP_SIZE_N=8` L2 swizzle for the SR counter, was deleted with it.

Static instruction excess over the RTNE variant, linear:

| | total | IMAD | LOP3 | FMUL | FMNMX |
|---|---:|---:|---:|---:|---:|
| before | +3872 | +2007 | +1530 | 0 | 0 |
| after | +1112 | +650 | +423 | 0 | 0 |

Linear `(2048, 7168)` gate/up sentinel, median of three 15/50 samples:

| math | rounding | before round | after fused mul/clamp | after one-draw Philox | total |
|---|---|---:|---:|---:|---:|
| standard | SR | 46.4310 | 41.5547 | **25.8337** | **44.4% faster** |
| fast | SR | 33.0158 | 31.4567 | **18.0922** | **45.2% faster** |
| standard | RTNE | 19.5444 | 19.5155 | 19.4503 | unchanged |
| fast | RTNE | 14.8403 | 12.9741 | 12.9274 | 12.9% faster |

### Linear RHT amax vectorization

The linear `cutedsl_rht_amax` was the worst-performing kernel in the project: 2.2-3.0x
behind TransformerEngine's `HadamardAmaxTmaKernel` and running at 2145 GB/s where its own
grouped sibling reached 5074 on the same bytes.

Two register/occupancy theories were tested first and **both were refuted**, which is
worth recording so they are not re-derived:

1. **Warp-specialized register redistribution.** The linear file contains zero
   `warpgroup_reg_alloc/dealloc` calls against the grouped file's 13, and the linear amax
   needs 98 registers where the grouped one needs 52. Adding the discipline (block padded
   to whole warpgroups, dealloc 32 / alloc 96 / alloc 64) changed performance by +-0.5% and
   left `REG:98` untouched. `setmaxnreg` redistributes registers *within* a CTA's existing
   allocation; it does not lower the compiled base, and the base is what sets blocks per SM.
2. **Supertile height plus a larger grid.** Compiling the 128-row supertile does drop the
   kernel to `REG:42`, which genuinely fits four blocks per SM against the 256-row config's
   one. Raising `GRID` past `NUM_SMS` to exploit that was **monotonically worse** at every
   shape, co-resident or not. The kernel is not warp-starved in a way more warps repair.

What worked was looking at the instruction mix instead. The row epilogue read SMEM one
element at a time, re-deriving a swizzled address per element:

| kernel | SMEM loads |
|---|---|
| linear amax, before | **128 x `LDS.U16`** |
| grouped amax | 13 x `LDS.128` |
| linear amax, after | **16 x `LDS.128`** |

This is the same pattern the epilogue round fixed for the 2D weight kernel and the fused 1D
row epilogue; the amax kernel does the same read for the same reason and was simply missed.
The corrected shape already existed one function away in the same file. Static instructions
fell 2088 -> 1704 (18.4%), with registers unchanged at 98 — so the address arithmetic was
not holding them either.

| shape | before (us) | after (us) | change | GB/s after |
|---|---:|---:|---:|---:|
| (512, 7168) | 8.60 | 7.31 | 15.0% faster | 1004 |
| (2048, 2048) | 8.71 | 7.42 | 14.9% faster | 1131 |
| (2048, 4096) | 11.18 | 8.42 | 24.7% faster | 1993 |
| (2048, 7168) | 13.69 | **9.67** | **29.4% faster** | 3036 |
| (4096, 7168) | 20.81 | 12.97 | 37.7% faster | 4526 |
| (8192, 2048) | 16.05 | 10.77 | 32.9% faster | 3114 |
| (8192, 7168) | 37.01 | **24.69** | **33.3% faster** | 4757 |

At (8192, 7168) the kernel now edges TE's `HadamardAmaxTmaKernel` (24.92 us). The complete
linear pipeline against TE:

| math | rounding | TE speedup before | TE speedup now |
|---|---|---:|---:|
| standard | RTNE | 1.82x | **1.60x** |
| standard | SR | 1.31x | **1.17x** |
| fast | RTNE | 1.71x | **1.46x** |
| fast | SR | 1.31x | **1.15x** |

Output is bitwise identical: the read order within a 16-element block changes, and max is
commutative.

### Columnwise scale-factor store packing

The swizzled columnwise SF write-view is `(1, sf_gcol, 32, 16)` with stride
`(.., SF_BLK, 16, 1)`, and the epilogue's group index `u` drives the **last, contiguous**
mode: `u` and `u+1` land in adjacent bytes. The store nevertheless wrote one byte at a
time, so every group paid its own `STS.U8` with its own four-index swizzle address, and
across a warp the lanes hit addresses 16 bytes apart. Staging four scale factors in a
register tile and committing them with one `autovec_copy` takes 25 `STS.U8` down to 9.

| math | rounding | before (us) | after (us) | change |
|---|---|---:|---:|---:|
| fast | RTNE | 12.9274 | **11.3856** | **11.9% faster** |
| fast | SR | 18.0922 | 17.7677 | 1.8% faster |
| standard | RTNE | 19.4503 | 19.4164 | unchanged |
| standard | SR | 25.8337 | 25.8176 | unchanged |

The win lands almost entirely on fast math, which is consistent: fast math drops the
`div.rn` reciprocal and the bfloat16 round-through, so the epilogue's store traffic is a
much larger share of what remains. Total static instruction count did not move (3608
either way) — the packing costs what the stores saved, so the gain is in issue efficiency
rather than instruction count, which caps how much more is available from store shaping.

This clears the aggregate acceptance clause (3.5% across the four cases, nothing
regressing) but not the >= 2% standard-math sentinel gate, which it misses at 0.17%. It
was retained on the strength of a reproducible 11.9% on a configuration the project ships
and tests byte-identically against TE. The same packing applied to the **rowwise** SF
store was measured and discarded: it takes 9 `STS.U8` to 1, but every case moved within
+-0.5%, so those lines did not earn their place.

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
  speedup survives: an L2-flushed 671B run measured 1.54x against its hot counterpart's
  1.52x, a 1.3% gap. Read the speedup column and treat bandwidth as an upper bound.

| model | projection | E | M | N | vector_norm_us | triton_us | speedup | triton_gbps |
|---|---|---:|---:|---:|---:|---:|---|---:|
| debugmodel | gate/up (w1/w3) | 4 | 256 | 256 | 9.079 | 3.833 | 2.37x | 136.9 |
| debugmodel | down (w2) | 4 | 256 | 256 | 9.089 | 3.947 | 2.30x | 132.7 |
| 16B | gate/up (w1/w3) | 4 | 1408 | 2048 | 11.452 | 5.845 | 1.96x | 3946.7 |
| 16B | down (w2) | 4 | 2048 | 1408 | 11.464 | 5.861 | 1.96x | 3935.9 |
| 671B | gate/up (w1/w3) | 4 | 2048 | 7168 | 33.678 | 20.591 | 1.64x | 5703.4 |
| 671B | down (w2) | 4 | 7168 | 2048 | 33.653 | 20.543 | 1.64x | 5716.7 |

`nvfp4_linear` uses this same op at `E = 1` on `W.unsqueeze(0)` — nothing in the kernel
is expert-specific beyond the `program_id(1)` base.
