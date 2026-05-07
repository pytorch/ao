# FP8 Tensorwise Grouped GEMM Notes

This branch implements `Float8TrainingRecipe.FP8_TENSORWISE` for MoE grouped
GEMM. The active path is `_to_fp8_tensorwise_then_scaled_grouped_mm` in
`torchao/prototype/moe_training/fp8_tensorwise_grouped_mm.py`.

Current implementation:
- Quantizes 2D activations with a single tensorwise scale.
- Quantizes 3D expert weights with one global Primus-like scale across the
  entire `(E, K, N)` tensor.
- Materializes both column-major RHS layouts for forward and backward reuse.
- Reuses cached FP8 RHS data in backward rather than re-quantizing weights.

Removed cleanup leftovers:
- Standalone per-group TensorWise kernels from
  `kernels/fp8_tensorwise_per_group.py`.
- Unused per-group TensorWise custom ops in `kernels/jagged_float8_scales.py`.
- Unused public 3D wrapper custom ops in `kernels/fp8_tensorwise_3d.py`; the
  grouped GEMM path calls the low-level dual-layout kernel directly.
