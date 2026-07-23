# PR Review — NVFP4 Grouped-GEMM MoE Training (`nvfp4_moe` branch)

## Review scope
- **Reviewed:** `git diff origin/main...HEAD` — 17 commits, 17 files, +3318/−50
- **Change:** New NVFP4 grouped-GEMM MoE-training stack — 3 Triton kernels (grouped RHT amax, grouped 2D weight quantize, grouped RHT row/col quantize), autograd orchestrator (`_NVFP4GroupedMM` / `_to_nvfp4_then_scaled_grouped_mm`), single-GPU example, tests, benchmarks.
- **Base:** `origin/main`
- **BLOCKING found:** 2 (2 shown)
- **NON-BLOCKING found:** 4 (3 shown, 1 collapsed)
- **Not run:** no kernels executed (no SM100 hardware). Numerics assessed by static analysis + 3 independent kernel reviewers + cross-check against the shipped non-grouped twins. Benchmarks / README / `deepseek_v3_shapes.py` treated as non-correctness scope.

---

## Findings

### [BLOCKING / HIGH] int32 overflow in per-expert base-pointer arithmetic
- **Location:** `torchao/prototype/moe_training/nvfp4_training/group_quantize_2d_triton.py:68-69,78`
- **Why it matters:** `expert = tl.program_id(2)` (int32) and `M`, `N` are int32 kernel args, so `expert * M * N` (and `expert * M * (N//2)`, `expert * N * (M//2)`) is evaluated in int32. Once `E*N*K > 2^31` the offset wraps negative → illegal memory access, or worse, silent reads/writes against the wrong expert's slab (corrupted codes/scales, no error). The largest shape in this PR (671B, `w1=(128,2048,7168)` → `127*14.68M ≈ 1.86e9`) sits ~14% **under** 2^31, so current tests/example do not catch it — but `E=256` local experts or larger hidden dims cross it. This is the only genuinely grouped-specific pointer math; the non-grouped twin uses TMA on a single 2D tensor and never hits it.
- **Suggested fix:** cast the expert index to int64 before the multiply on all four bases, e.g. `expert.to(tl.int64) * M * N`. Add a round-trip assertion at a config where `E*N*K > 2^31` (e.g. `E=64, N=K=8192`) checking the **last** expert specifically.

### [BLOCKING / LOW-MEDIUM] Padded-capacity masking path is untested
- **Location:** `test/prototype/moe_training/nvfp4_training/test_group_rht_quantize_row_col_triton.py` (and the amax test)
- **Why it matters:** the `logical_packed_length < packed_sequence_length` masking (rows beyond the logical length flush to zero) is the only substantively new logic vs. the 2D kernels, yet no unit test sets `logical_packed_length`. Static analysis confirms correctness (128-aligned ⇒ every tile is fully-valid or fully-padded; padded tiles produce zero codes/scales and write disjoint regions). **Mitigating:** the shipped `_NVFP4GroupedMM` always calls with `logical == packed`, so this path is currently unreachable from the autograd function. But it is a public kernel capability where a silent regression would ship undetected.
- **Suggested fix:** one unit case with `packed_sequence_length > offsets[-1]` (e.g. +128 padded rows) asserting valid ranges dequantize correctly and padded `qa/qd/sf` regions are zero.

### [NON-BLOCKING / LOW] `_validate_graph_amax` omits 1D/contiguity check
- **Location:** `torchao/prototype/moe_training/nvfp4_training/group_rht_quantize_row_col_triton.py:184-198`
- **Why it matters:** kernel does linear `global_amax_ptr + group_idx` indexing; a strided or 2D amax silently reads the wrong element. Caller-controlled today.
- **Suggested fix:** add `ndim == 1` + contiguous asserts.

### [NON-BLOCKING / LOW] Autotune relies on `atomic_max` idempotency, not `reset_to_zero`
- **Location:** `torchao/prototype/moe_training/nvfp4_training/group_hadamard_amax_triton.py:133`
- **Why it matters:** correct only because buffers are pre-zeroed and the reduction is monotone-max; a future refactor to an accumulating reduction (e.g. `atomic_add`) would let autotuning silently corrupt results.
- **Suggested fix:** one-line comment noting the reliance.

### [NON-BLOCKING / LOW] Pattern note (collapsed)
Inherited-from-2D-twin minor items, no action needed:
- Plain `/` (not `tl.div_rn`) for the 2D weight global-encode scale — ≤1-ULP FP8 tie difference vs TE, applied identically row/col so wgrad stays consistent.
- Philox counter `<< 32` truncation reduces SR entropy above ~8.6e9 packed elements/tile (no bias — seeds still differ).

---

## Test economy
Strong, high-signal, no coverage theater. `test_nvfp4_grouped_gemm_fwd_bwd` (fwd/dgrad/wgrad SQNR gates), `_compile_fwd_bwd` (fullgraph), `_unaligned_padding` (pad/unpad round-trip), and `_fake_tensor` (shape+stride) each target a distinct real failure mode. The modified `test_hadamard_quantize_row_col_triton.py` correctly tracks the intentional removal of the `FP8_E4M3_EPS` floor (TE zero-scale semantics) rather than papering over it. Only real gap: the padded-capacity unit case above.

## Diff economy
Clean. The three kernels are faithful grouped ports of shipped non-grouped twins reusing shared pack/swizzle/quantize helpers (no speculative abstractions). The `safe_global_amax * 0.0 + (...)` in `hadamard_utils.py:242` is a deliberate, commented anti-constant-fold hack for `tl.div_rn`. Real vs. fake output shapes match exactly for the 2D quantizer (no eager/compile divergence). No drive-by refactors or unrelated touches.

## Verification
- Inspected: full diff, `nvfp4_grouped_mm.py` orchestrator (fwd/dgrad/wgrad shape flow), all new tests, the TE-scale source+test change, real-vs-fake shape parity for the 2D quantizer.
- Directly confirmed the int32-overflow finding by reading `group_quantize_2d_triton.py:60-84`.
- Not run: no kernels executed (needs SM100). Numerical correctness rests on existing SQNR tests + static review.

## Summary
Merge-blocked on one HIGH bug: a trivial one-line int64 cast prevents silent cross-expert corruption on large MoE weights (`group_quantize_2d_triton.py:68-69,78`). Add the padded-capacity unit test (LOW-MEDIUM). Everything else — orchestration, autograd shapes, stochastic-rounding independence, packing/swizzle, and test coverage — checks out. The amax and rht row/col kernels are clean.

## Best next mode
Implementer — apply the int64 cast (+ large-`E*N*K` guard test) and the padded-capacity unit case; optionally the two LOW comments.
