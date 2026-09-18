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
are a much larger share of what the CuteDSL epilogue does once its other work is fused.

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
full script passes. The **CuteDSL kernels — comparison vs TransformerEngine** section at
the end of this file is the exception — see its provenance note.

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

## CuteDSL kernels — comparison vs TransformerEngine

DeepSeek-V3 671B FFN shapes, excluding attention GEMMs. The grouped kernels run at
`E = 4` local experts over per-expert `(2048, 7168)` gate/up and `(7168, 2048)` down;
the linear kernels run that same pair as plain 2D tensors.

Run environment: NVIDIA GB200, CUDA 13.4, PyTorch 2.15.0a0+git0f3e7e2,
nvidia-cutlass-dsl 4.5.2, and TransformerEngine 2.19.0.dev0 built from
[NVIDIA/TransformerEngine@172bd93](https://github.com/NVIDIA/TransformerEngine/commit/172bd93773ad6ee4ba44b460b7f10ef42fc89d57)
("[Common] Ensure quantization kernels handle noop properly", #3271) — an untagged
main-branch commit, not a release.

**Provenance.** No checked-in script reproduces this section: the benchmark modules
cover CuteDSL-vs-Triton only, and `--shape-set` does not offer the DeepSeek-V3 linear
pair. Every figure here was measured out of band, and the CuteDSL columns predate the
most recent optimization rounds — treat them as a lower bound. The TE-relative ratios
are still the best available, because both sides were timed in the same run;
re-measuring one side alone would be worse than leaving the pair intact.

### Methodology

- **Device kernel self-time** via `torch.profiler`, 15 warmups and 50 measured
  iterations, memcpy and memset events excluded — the same `bench_utils.kernel_time_us`
  the CuteDSL-vs-Triton tables use. Device time rather than wall clock because NVFP4
  training runs these under CUDA graphs / `torch.compile`.
- TE's `split_quantize` computes its post-RHT amax internally, so the apples-to-apples
  activation figure is the full CuteDSL pipeline: one `*_rht_amax` launch followed by one
  `*_rht_quantize_row_col` launch.
- A TE 2D weight call launches three kernels, not one. `cutedsl_weight_quantize_2d` and
  its grouped twin consume a **precomputed** amax, so the 2D comparison is against TE's
  quantize kernel alone. Comparing against TE's full call would credit CuteDSL with
  skipping a pass it never runs.
- Speedup columns are computed from the absolute columns beside them. TE's 2D quantize
  kernel measured 13.64-13.80 us across passes; each 2D row carries the figure from its
  own paired run.

### Hadamard Amax

`cutedsl_rht_amax` and `cutedsl_group_rht_amax` against TE's `HadamardAmaxTmaKernel`.
The grouped kernel has no standalone TE counterpart — TE folds the reduction into
`split_quantize` — so its TE column is that internal cost.

| family | shape | CuteDSL (us) | TE kernel | TE (us) | TE speedup |
|---|---|---:|---|---:|---:|
| linear | (2048, 7168) | 9.67 | `HadamardAmaxTmaKernel` | 4.47 | 2.16x |
| linear | (8192, 7168) | 24.69 | `HadamardAmaxTmaKernel` | 24.92 | 0.99x |
| grouped, E=4 | gate/up (2048, 7168) | 23.34 | internal to `split_quantize` | 25.81 | 0.90x |

The grouped kernel beats TE outright. The linear one trails badly at small M and closes
the gap as M grows, reaching parity at (8192, 7168) — it is bandwidth-bound and needs the
rows to amortize its epilogue.

Linear amax across shapes:

| shape | CuteDSL (us) | GB/s |
|---|---:|---:|
| (512, 7168) | 7.31 | 1004 |
| (2048, 2048) | 7.42 | 1131 |
| (2048, 4096) | 8.42 | 1993 |
| (2048, 7168) | 9.67 | 3036 |
| (4096, 7168) | 12.97 | 4526 |
| (8192, 2048) | 10.77 | 3114 |
| (8192, 7168) | 24.69 | 4757 |

### Hadamard Quantize Row+Col

Full pipeline, grouped — `cutedsl_group_rht_amax` plus
`cutedsl_group_rht_quantize_row_col` against TE's `split_quantize`:

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

Full pipeline, linear `(2048, 7168)` — `cutedsl_rht_amax` plus
`cutedsl_rht_quantize_row_col` against a single-tensor `NVFP4Quantizer` call:

| math | rounding | CuteDSL pipeline (us) | TE pipeline (us) | TE speedup |
|---|---|---:|---:|---:|
| standard | RTNE | 29.82 | 18.61 | 1.60x |
| standard | SR | 36.69 | 31.33 | 1.17x |
| fast | RTNE | 21.75 | 15.95 | 1.46x † |
| fast | SR | 28.24 | 24.63 | 1.15x |

† The later amax-vectorization run re-reported this ratio as 1.46x; the absolute columns
are from the earlier paired session, where they give 1.36x.

Quantize kernel alone, with TE's side broken out of `split_quantize`:

| family | CuteDSL (us) | TE (us) | TE speedup |
|---|---:|---:|---:|
| grouped, E=4 gate/up | 57.04 | 38.32 | 1.49x |
| linear (2048, 7168) | 19.50 | 12.98 | 1.50x |

The two ratios are nearly identical, which is what one would expect from the grouped and
linear kernels sharing `_quant16_from_amax`. Against the pipeline tables above, this is
where the remaining gap lives: the grouped amax wins and the linear amax closes with M,
so the quantize stage is the standing deficit on both paths.

### 2D Weight Quantize

A TE 2D call launches three kernels:

| TE kernel | (2048, 7168) | (7168, 2048) |
|---|---:|---:|
| `quantize_transpose_kernel` | 13.80 | 13.76 |
| `amax_kernel` | 5.40 | 5.39 |
| `zero_amax_kernel` | 1.34 | 1.34 |
| total | 20.53 | 20.49 |

`cutedsl_weight_quantize_2d` consumes a precomputed amax, so it is compared against
`quantize_transpose_kernel` alone. TransformerEngine has no grouped 2D weight kernel, so
the grouped rows put one `cutedsl_group_weight_quantize_2d` launch over all four experts
against four times TE's single-expert time, both sides emitting 16x16 block scaling via
`NVFP4Quantizer(with_2d_quantization=True)`.

| kernel | projection | E | CuteDSL (us) | TE quantize only (us) | TE speedup |
|---|---|---:|---:|---:|---:|
| 2D linear | gate/up | 1 | 16.5665 | 13.64 | 1.21x |
| 2D linear | down | 1 | 16.5806 | 13.76 | 1.20x |
| 2D grouped | gate/up | 4 | 60.5176 | 54.56 (x4) | 1.11x |
| 2D grouped | down | 4 | 60.7924 | 54.62 (x4) | 1.11x |

At the op level, where each side computes its own amax, CuteDSL wins instead: TE pays
6.74 us for `amax_kernel` plus `zero_amax_kernel`, where torchao's weight amax is cheaper
(see **Grouped Weight Amax** above).

**Neither side has a 2D fast path**, and this is structural rather than an omission. Fast
math buys two things on the 1D path: skipping the bfloat16 round-through of the tcgen05
RHT accumulator, and `rcp_approx` in place of `div.rn` for the encode reciprocal. Without
an RHT there is no accumulator, so the first cannot apply, and the second is one
instruction per 16-element block in a kernel that is otherwise load/store bound. Measured,
`NVTE_USE_FAST_MATH=1` moves TE's three kernels by at most 0.02 us (13.78/5.39/1.34), and
the public CuteDSL 2D wrapper exposes no `use_fast_math` at all (`_cutedsl_kernels_impl.py`
asserts `not (fast_math and not apply_rht)`).

### Standalone CuteDSL kernel medians

Each entry is the median of three samples; every sample uses 15 warmups and 50 timed
CUDA-profiler iterations and reports device kernel self-time in microseconds. The 1D rows
measure the fused quantize stage with precomputed amaxes; the 2D rows consume precomputed
weight amaxes.

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

`source = script` rows come from the commands below and are reproducible. The `out of
band` rows are the DeepSeek-V3 `(2048, 7168)` / `(7168, 2048)` linear shapes, which no
checked-in script emits — giving the linear scripts a DeepSeek-V3 shape set would make
this table fully reproducible. Those rows also predate the last two optimization rounds,
so treat them as a lower bound; the 2D linear entries here disagree with the paired 2D
table above (16.33/16.36 against 16.5665/16.5806) because the two were measured in
different passes.

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
`all`). `NVTE_USE_FAST_MATH` is a TransformerEngine variable with no effect on a pure
torchao run — the torchao equivalent is `--math fast`, which passes `use_fast_math=True`
to both backends.

### Numerical parity

Every RTNE result above is bitwise identical to both oracles: the Triton backend and the
TE-derived PyTorch reference in
`test/prototype/moe_training/nvfp4_training/nvfp4_reference.py`. Stochastic rounding is
exempt by design — the CuteDSL kernels draw one Philox counter per 16-element block and
consume all four words, rather than reproducing Triton's per-packed-byte stride, so their
SR codes are a different, equally valid stream. SR is covered instead by
`test_rht_quantize_rs_at_most_one_fp4_step_from_rtne` (every SR code within one FP4 step
of the bitwise-checked RTNE code), `test_group_rht_sr_reconstructs`,
`test_cutedsl_rht_quantize_sr_unbiased` / `test_group_rht_sr_unbiased`, and
`test_group_rht_rng_state_controls_stochastic_rounding`.
