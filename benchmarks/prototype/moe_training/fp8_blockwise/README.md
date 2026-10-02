# Blockwise FP8 MoE Benchmarks

## Grouped-Kernel Benchmark

This benchmark measures each kernel in the blockwise FP8 MoE grouped-GEMM path
with the DeepGEMM backend. Quantization kernels are reported in GB/s against the
memory-bandwidth roofline. Grouped GEMMs are reported in TFLOP/s against the FP8
tensor-core roofline.

```bash
python -m benchmarks.prototype.moe_training.fp8_blockwise.bench_moe_grouped_kernels
```

The benchmark requires the optional `deep_gemm` dependency and an SM90+ GPU. It
times the following operations for each `(M, N, K, E)` shape:

- forward: `act_quant_lhs`, `weight_quant_forward_rhs`, `deepgemm_grouped_mm`
- dgrad: `act_quant_lhs(grad_out)`, `weight_quant_dgrad_rhs`,
  `deepgemm_grouped_mm_dgrad`
- wgrad: `wgrad_quant_lhs(grad_out)`, `wgrad_quant_rhs(A)`,
  `deepgemm_grouped_mm_wgrad`

Each quantization row reports the input read plus FP8 data and FP32 scale writes.
Each GEMM row reports compute throughput and modeled memory traffic. Offsets use
balanced, 128-aligned per-expert token counts by default. Pass `--jagged` for
skewed token distributions or `--shapes M,N,K,E ...` to override the default
DeepSeek-V3 FFN shapes.

### H100 Results

Balanced tokens with `M=32768`, `E=8`, `N=2048`, and `K=7168`:

| kernel | us | TFLOP/s | %ach_compute | GB/s | %ach_bw |
|---|--:|--:|--:|--:|--:|
| fwd: act_quant_lhs | 253.5 | - | - | 2809 | 91.1 |
| fwd: weight_quant_forward_rhs | 128.1 | - | - | 2750 | 89.2 |
| fwd: deepgemm_grouped_mm | 779.3 | 1234 | 80.0 | 634 | 20.5 |
| bwd: act_quant_lhs(grad_out) | 77.6 | - | - | 2621 | 85.0 |
| bwd: weight_quant_dgrad_rhs | 126.3 | - | - | 2790 | 90.5 |
| bwd: deepgemm_grouped_mm_dgrad | 778.6 | 1236 | 80.1 | 843 | 27.3 |
| bwd: wgrad_quant_lhs(grad_out) [transposed] | 85.1 | - | - | 2390 | 77.5 |
| bwd: wgrad_quant_rhs(A) [direct] | 275.3 | - | - | 2586 | 83.8 |
| bwd: deepgemm_grouped_mm_wgrad | 2105.4 | 457 | 29.6 | 594 | 19.3 |
