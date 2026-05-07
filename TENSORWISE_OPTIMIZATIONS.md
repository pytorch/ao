# FP8 Tensorwise Grouped GEMM Notes

This branch implements `Float8TrainingRecipe.FP8_TENSORWISE` for MoE grouped
GEMM. The active entry point is `_to_fp8_tensorwise_then_scaled_grouped_mm` in
`torchao/prototype/moe_training/fp8_tensorwise_grouped_mm.py`.

## Active Implementation

The TensorWise path uses one scalar scale per whole tensor, following the
Primus-style tensorwise recipe:

- Forward quantizes the 2D activation tensor `A` with one tensorwise scale.
- Forward quantizes the 3D expert weight tensor `B_t` with one global scale
  across the full `(E, K, N)` tensor.
- Forward materializes the FP8 `B_t` layout used by the forward grouped GEMM.
- Forward also materializes the transposed FP8 `B` layout used by backward
  `grad_A`, because MoE training spends enough time in backward that caching
  this layout is worthwhile.
- Forward precomputes the col-major `A` layout and flat inverse-scale buffer
  used by backward `grad_B`.
- Backward quantizes `grad_output`, then runs two grouped GEMMs:
  `grad_A = grad_output @ B` and `grad_B = grad_output_t @ A`.

The important implementation files are:

- `torchao/prototype/moe_training/fp8_tensorwise_grouped_mm.py`
  - autograd wrapper and calls to `torch._scaled_grouped_mm`
  - saves backward-ready FP8 layouts from forward
  - owns the Python custom ops that wrap the 2D/3D Triton quantizers
- `torchao/prototype/moe_training/kernels/fp8_tensorwise_2d.py`
  - 2D tensorwise activation / `grad_output` quantization
  - current experimental version can materialize row-major and col-major FP8
    layouts from the same quantization pass
- `torchao/prototype/moe_training/kernels/fp8_tensorwise_3d.py`
  - low-level 3D dual-layout quantization kernel for expert weights
- `torchao/prototype/moe_training/config.py`
  - exposes `Float8TrainingRecipe.FP8_TENSORWISE`
  - on the ROCm 7.2 nightly used here, MI355X must use `torch.float8_e4m3fn`
    for `torch._scaled_grouped_mm`
- `torchao/prototype/moe_training/utils.py`
  - dispatches `ScalingGranularity.TENSORWISE` to the TensorWise grouped GEMM
    implementation

## Cleanup Commit

Committed as:

```text
b917f1345 [moe training] Clean up tensorwise FP8 benchmark path
```

That commit removes stale TensorWise code and adds benchmark infrastructure.

### Removed Stale Code

The current implementation no longer uses the old per-group TensorWise kernels,
so these were removed:

- `torchao/prototype/moe_training/kernels/fp8_tensorwise_per_group.py`
  - standalone per-group TensorWise quantization file
  - no active imports or call sites remained
- TensorWise per-group custom ops in
  `torchao/prototype/moe_training/kernels/jagged_float8_scales.py`
  - `triton_fp8_per_group_tensorwise_scales`
  - `triton_fp8_per_group_tensorwise_scales_col_major`
  - `triton_fp8_per_group_tensorwise_dual_col_major`
  - their private Triton kernels and fake implementations
- Unused public wrapper custom ops in
  `torchao/prototype/moe_training/kernels/fp8_tensorwise_3d.py`
  - the active path calls the low-level dual-layout kernel directly

The cleanup reduced review surface and removed multiple TensorWise paths that
were not exercised by the active grouped GEMM implementation.

### Benchmark Added

Added:

```text
benchmarks/prototype/moe_training/fp8_tensorwise/bench_fp8_tensorwise_grouped_mm.py
```

The benchmark reports:

- BF16 forward grouped GEMM time
- TorchAO TensorWise FP8 forward time
- TorchAO TensorWise FP8 backward-only time
- TorchAO TensorWise FP8 forward+backward time
- optional Primus Turbo TensorWise FP8 forward, backward, and forward+backward
  times via `--compare-turbo`

Example command:

```bash
cd /workspace/ao
HIP_VISIBLE_DEVICES=0 python benchmarks/prototype/moe_training/fp8_tensorwise/bench_fp8_tensorwise_grouped_mm.py \
  --shape 8,256,2048,2048 \
  --backward \
  --compare-turbo
```

Shape format is:

```text
experts,tokens_per_expert,k,n
```

## Benchmark Environment

Container:

```text
rocm/primus:v26.2
GPU: AMD Instinct MI355X
```

PyTorch was upgraded inside the container to a ROCm 7.2 nightly:

```text
torch: 2.13.0.dev20260428+rocm7.2
hip: 7.2.53211
```

TorchAO was installed from the local worktree with:

```bash
USE_CPP=0 pip install --no-build-isolation .
```

`USE_CPP=0` was needed because the TorchAO ROCm C++ extension did not build
against this nightly due to a removed/renamed
`HIPCachingAllocatorMasqueradingAsCUDA` symbol. The TensorWise path tested here
is Python/Triton plus `torch._scaled_grouped_mm`, so the C++ extension is not
required for these microbenchmarks.

Primus Turbo was rebuilt from:

```text
/workspace/alex_tensorwise_microbench/Primus-Turbo
```

The image copy at `/workspace/Primus-Turbo` was ABI-incompatible with the new
PyTorch. The mounted copy had stream API updates and built successfully:

```text
primus_turbo-0.3.0+ef5b58e
```

## Baseline Comparisons

For shape `8,256,2048,2048` on MI355X:

```text
M=2048, K=2048, N=2048, experts=8
```

After the cleanup commit, before later experiments, representative timing was:

```text
BF16 forward:             ~115 us
TorchAO TensorWise fwd:   ~192 us
TorchAO TensorWise bwd:   ~164 us
TorchAO TensorWise f+b:   ~516 us
Primus Turbo fwd:          ~96 us
Primus Turbo bwd:         ~137 us
Primus Turbo f+b:         ~353 us
```

The main gap is not only quantization. Primus Turbo uses its own grouped FP8
backend, while TorchAO currently calls `torch._scaled_grouped_mm`.

## Results Timeline

This section records the measured speedup from the work done after the cleanup
commit. All timings below use:

```text
shape: 8,256,2048,2048
M=2048, K=2048, N=2048, experts=8
GPU: MI355X
torch: 2.13.0.dev20260428+rocm7.2
hip: 7.2.53211
TorchAO install: USE_CPP=0 pip install --no-build-isolation .
```

### Before Today's Performance Work

After the cleanup/benchmark commit `b917f1345`, before the dual-layout and
fused inverse-scale experiments, the representative large-shape run was:

```text
BF16 forward:             114.66 us
TorchAO TensorWise fwd:   192.36 us
TorchAO TensorWise bwd:   163.92 us
TorchAO TensorWise f+b:   516.25 us
Primus Turbo fwd:          96.40 us
Primus Turbo bwd:         137.16 us
Primus Turbo f+b:         353.04 us
```

This is the practical baseline for the optimization work. The main observation
was that TorchAO forward was roughly 2x Primus Turbo forward, and TorchAO
forward+backward had a large remaining gap.

### After Fused Inverse-Scale Materialization

Committed as:

```text
9889e1e00 [moe training] Fuse tensorwise inverse scale materialization
```

After reinstalling that commit in the same container:

```text
torchao-0.18.0+git9889e1e00
```

Three repeated large-shape runs produced:

```text
Run  TorchAO fwd  TorchAO bwd  TorchAO f+b  Turbo fwd  Turbo bwd  Turbo f+b
1    144.56 us    143.80 us    430.98 us    96.40 us   132.88 us  338.60 us
2    135.42 us    147.58 us    444.92 us    96.44 us   138.80 us  352.28 us
3    143.48 us    144.56 us    430.80 us    96.40 us   132.56 us  337.48 us
```

Using the median-ish post-change run against the pre-change baseline:

```text
Metric                Before      After       Improvement
TorchAO fwd           192.36 us   ~143 us     ~25-30% faster
TorchAO bwd           163.92 us   ~145 us     ~10-13% faster
TorchAO fwd+bwd       516.25 us   ~431-445 us ~14-17% faster
```

The larger win is in forward because the 2D and 3D quantization kernels now
write the contiguous inverse-scale buffers required by `torch._scaled_grouped_mm`
directly. This removes post-kernel scalar reciprocal and expanded contiguous
scale materialization from the hot path.

### Current Gap to Primus Turbo

After the committed optimization, the large-shape gap is approximately:

```text
Metric          TorchAO      Primus Turbo   Remaining gap
forward         ~143 us      ~96 us         ~1.5x slower
backward        ~145 us      ~133-139 us    near parity
forward+backward~431-445 us  ~337-352 us    ~1.25-1.3x slower
```

This suggests the backward quant/layout overhead is now much closer to Primus
Turbo, while forward still has a substantial gap. The remaining forward gap is
likely a mix of quantization launch overhead and the grouped GEMM backend:
TorchAO calls `torch._scaled_grouped_mm`, while Primus Turbo dispatches through
its own grouped FP8 backend.

### Multi-Shape Snapshot After Commit `9889e1e00`

The same committed state was also measured across smaller shapes:

```text
experts  tokens/expert  M     K     N     TorchAO fwd  TorchAO bwd  TorchAO f+b  Turbo fwd  Turbo bwd  Turbo f+b
1        128            128   512   512   135.76 us    143.76 us    431.72 us    63.08 us   140.68 us  355.88 us
2        128            256   512   512   141.84 us    104.42 us    308.48 us    64.80 us    98.24 us  216.42 us
4        128            512   1024  1024  138.86 us    104.08 us    302.36 us    66.40 us    99.50 us  213.64 us
8        256            2048  2048  2048  140.70 us    143.56 us    318.36 us    93.64 us   131.68 us  225.80 us
```

These one-shot multi-shape numbers are useful for spotting trends, but the
larger-shape repeated runs above are the better apples-to-apples result for
tracking progress.

## Experiment 1: Defer Backward Layouts

Goal:

- reduce forward time by not precomputing backward-only layouts in forward
- build `B_rhs_fp8` and `A_col_major` only during backward

Result:

```text
Before:
TorchAO fwd:      192.16 us
TorchAO bwd:      146.52 us
TorchAO fwd+bwd:  385.80 us

After deferring layouts:
TorchAO fwd:      164.62 us
TorchAO bwd:      308.18 us
TorchAO fwd+bwd:  520.29 us
```

Conclusion:

- Forward improved, but backward got much worse.
- This is not good for TorchTitan MoE training because backward has more
  quantization/layout work than forward.
- The experiment was reverted.
- Keeping backward-ready layouts in forward is the better training trade-off.

## Experiment 2: Dual-Layout 2D Quantization

Goal:

- avoid separate `_copy_to_column_major` kernels for 2D tensors
- have 2D quantization write both row-major and col-major FP8 layouts directly

Implementation:

- Added `triton_fp8_tensorwise_quantize_2d_dual_layout` in
  `kernels/fp8_tensorwise_2d.py`.
- Added a `torch.library.custom_op` wrapper in
  `fp8_tensorwise_grouped_mm.py`.
- Used it for:
  - `A` in forward, producing `A_fp8` and `A_col_major`
  - `grad_output` in backward, producing `grad_output_fp8` and
    `grad_output_col_major`

Correctness:

```text
E=1: out=28.75 dB, grad_A=28.50 dB, grad_B=28.75 dB
E=2: out=28.62 dB, grad_A=28.25 dB, grad_B=28.75 dB
E=4: out=28.25 dB, grad_A=28.50 dB, grad_B=28.50 dB
```

Initial speed result:

```text
Before: fwd ~192 us, bwd ~164 us, fwd+bwd ~516 us
After:  fwd ~185-191 us, bwd ~153-166 us, fwd+bwd ~494-519 us
```

Conclusion:

- Correct but too close to benchmark noise by itself.
- Forward improved consistently, but backward and fwd+bwd were not consistently
  better in every run.

## Experiment 3: Fused Inverse-Scale Materialization

Goal:

- remove scalar reciprocal + expanded contiguous scale materialization after
  quantization
- have Triton quantization kernels write the final inverse-scale buffers that
  `torch._scaled_grouped_mm` requires

Important constraint:

`torch._scaled_grouped_mm` rejects stride-0 expanded scale views:

```text
RuntimeError: scale must be contiguous for arg 0
```

So the optimization cannot pass broadcasted views directly. The scale buffers
must still be contiguous, but the kernels can write them directly.

Implementation:

- 2D quantization now writes the `(M,)` inverse-scale buffer from inside the
  quantization kernel.
- 2D dual-layout quantization writes both FP8 layouts and the `(M,)` inverse
  scale in the same quantization pass.
- 3D dual-layout quantization now writes:
  - `(E, N)` forward inverse scales
  - `(E, K)` backward RHS inverse scales
  directly from the Triton kernel.
- This removes separate post-kernel work like `1.0 / scale_buf` and
  `expand(...).contiguous()` for the main TensorWise scale buffers.

Correctness:

```text
E=1: out=28.75 dB, grad_A=28.50 dB, grad_B=28.75 dB
E=2: out=28.62 dB, grad_A=28.25 dB, grad_B=28.75 dB
E=4: out=28.25 dB, grad_A=28.50 dB, grad_B=28.50 dB
```

Five repeated large-shape runs after this experiment:

```text
Run  fwd_us  bwd_us  fwd+bwd_us
1    142.00  144.78  471.08
2    144.24  143.72  431.68
3    145.02  147.84  447.64
4    140.22  143.50  434.90
5    140.10  143.78  434.92
```

Compared to the previous dual-layout-only experiment:

```text
Before: fwd ~185-191 us, bwd ~153-166 us, fwd+bwd ~494-519 us
After:  fwd ~140-145 us, bwd ~143-148 us, fwd+bwd ~432-471 us
```

Conclusion:

- This is a clear win.
- Forward improves substantially.
- Backward improves modestly.
- Forward+backward improves consistently across repeated runs.
- This optimization is worth keeping.

## Current State After Experiments

As of commit `9889e1e00`, the TensorWise path includes:

- dual-layout 2D quantization
- fused inverse-scale materialization in 2D and 3D quant kernels

This is committed because it was correct and measured faster on the target
MI355X ROCm 7.2 nightly setup.

The remaining gap to Primus Turbo on the large shape is roughly:

```text
Metric            TorchAO TensorWise   Primus Turbo
forward           ~135-145 us          ~96 us
backward          ~144-148 us          ~133-139 us
forward+backward  ~431-445 us          ~337-352 us
```

Forward still has the largest relative gap. Backward is much closer to Primus
Turbo after the inverse-scale materialization change.

## Remaining Optimization Ideas

### Test `use_fast_accum=False`

This is cheap and may affect ROCm grouped GEMM performance. It should be tested
with correctness/SQNR checks because accumulation mode can change numerics.

### Profile Kernel Breakdown

Before writing more Triton code, collect a kernel-level trace or per-section
timing for:

- 2D quant `A`
- 3D quant `B_t`
- forward grouped GEMM
- 2D quant `grad_output`
- `grad_A` grouped GEMM
- `grad_B` grouped GEMM

This will show whether the remaining gap is quant/layout overhead or the
`torch._scaled_grouped_mm` backend.

### Try Primus Turbo Backend With TorchAO Quantized Inputs

Turbo’s forward GEMM is much faster. If TorchAO’s quantized tensors and scales
can be fed into Primus Turbo’s lower-level grouped FP8 backend, that isolates
whether quantization is competitive and the main gap is the grouped GEMM backend.

### Optimize Flat Scale Buffers for `grad_B`

`grad_B` still needs flat contiguous inverse-scale buffers:

```text
go_inv_scales_flat = go_inv_scales[:1].expand(num_groups * N).contiguous()
A_inv_scales_flat = A_inv_scales[:1].expand(num_groups * K).contiguous()
```

If these can be written directly by the producer kernels, or avoided by a
backend that accepts scalar TensorWise scales, it may remove more overhead.

### Backend-Level Work

The largest remaining win may require calling a faster grouped FP8 backend than
`torch._scaled_grouped_mm`, or adding a transpose-aware/tensorwise-specialized
path to PyTorch/TorchAO.
