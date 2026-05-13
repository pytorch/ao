# NVFP4 Training Benchmarks

This directory contains benchmarking scripts for the NVFP4 training kernels
under `torchao.prototype.mx_formats`.

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
| DeepSeek-V3 671B | attn.wkv_b input | 4096 | 512 | 43.328 | 96.804 |
| DeepSeek-V3 671B | attn.wq_b input | 4096 | 1536 | 46.048 | 273.256 |
| DeepSeek-V3 671B | shared expert w2 input | 4096 | 2048 | 48.128 | 348.596 |
| DeepSeek-V3 671B | hidden-state input | 4096 | 7168 | 45.088 | 1302.350 |
| DeepSeek-V3 671B | attn.wo input | 4096 | 16384 | 64.512 | 2080.510 |
| DeepSeek-V3 671B | dense ffn.w2 input | 4096 | 18432 | 68.672 | 2198.780 |
| DeepSeek-V3 671B | avg routed expert w2 input | 128 | 2048 | 42.720 | 12.273 |
| DeepSeek-V3 671B | avg routed expert w1/w3 input | 128 | 7168 | 42.240 | 43.442 |
| Llama 3 8B | hidden-state input | 2048 | 4096 | 46.496 | 360.831 |
| Llama 3 8B | mlp.down input | 2048 | 14336 | 46.080 | 1274.310 |
| Llama 3 70B | hidden-state input | 2048 | 8192 | 44.032 | 762.047 |
| Llama 3 70B | mlp.down input | 2048 | 28672 | 58.368 | 2012.070 |
