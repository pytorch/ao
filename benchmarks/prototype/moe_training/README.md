# MoE Training Benchmarks

## FP8 Blockwise Grouped GEMM Roofline

Benchmark the CuTeDSL FP8 blockwise 2D x 3D grouped GEMM kernel against BF16,
the emulated blockwise grouped path, and an FP8 grouped-GEMM roofline target:

```bash
python benchmarks/prototype/moe_training/bench_fp8_blockwise_grouped_gemm_roofline.py \
  --outfile /tmp/fp8_blockwise_grouped_gemm_roofline.csv
```

For a quick correctness and timing smoke run:

```bash
python benchmarks/prototype/moe_training/bench_fp8_blockwise_grouped_gemm_roofline.py \
  --shape smoke,2,256,256,256 \
  --warmup 1 \
  --iterations 3 \
  --rounds 1 \
  --check-correctness
```

Useful output columns:

- `cutedsl_us`: measured CuTeDSL grouped GEMM kernel time.
- `fp8_roofline_us`: modeled lower-bound time for an optimized FP8 grouped
  GEMM using the shared roofline GPU specs.
- `cutedsl_roofline_pct`: `fp8_roofline_us / cutedsl_us * 100`, where higher
  is closer to the roofline target.
- `emulated_us`: measured correctness-bridge implementation time.
- `cutedsl_speedup_vs_emulated`: measured `emulated_us / cutedsl_us`.
- `max_abs_diff_vs_emulated`: reported when `--check-correctness` is set.

Optimization notes and tuning results for the DeepSeek v3 FP8 blockwise CuTeDSL
grouped GEMM path are recorded in
[`fp8_blockwise_cutedsl_grouped_gemm_notes.md`](fp8_blockwise_cutedsl_grouped_gemm_notes.md).
