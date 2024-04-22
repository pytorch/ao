## Fused `int4 / fp16` Quant Matmul

Fused gemm for asymmetric quantized weights. Tested and benchmarked for `HQQ` but could theoretically be used for any asymmetric quantization scheme.

The kernel packs `u8 / s8` weights and fuses dequantization with the matmul.

- tested for `float16 / bfloat16` activations, scales, and zeros
- autotuned for both compute-bound and io-bound configs
- assumes operand B of the `gemm` is is the quantized type.
- requires quantization along `in-features`, i.e., the `K` dimension, or `axis=1`, of `torch.linear.weight`.

### Performance

Initial benchmarking demonstrates promising results, scaling well across io-bound and compute-bound workloads:

|     | M    | N    | K    | group_size | dtype          | hqq_ref | triton | tinygemm |
| --- | ---- | ---- | ---- | ---------- | -------------- | ------- | ------ | -------- |
| 0   | 16   | 4096 | 4096 | 128        | torch.bfloat16 | 0.2675  | 0.0633 | 0.0382   |
| 1   | 32   | 4096 | 4096 | 128        | torch.bfloat16 | 0.2669  | 0.0704 | 0.0649   |
| 2   | 128  | 4096 | 4096 | 128        | torch.bfloat16 | 0.2689  | 0.0960 | 0.2523   |
| 3   | 256  | 4096 | 4096 | 128        | torch.bfloat16 | 0.3268  | 0.1355 | 0.5192   |
| 4   | 512  | 4096 | 4096 | 128        | torch.bfloat16 | 0.3628  | 0.2369 | 1.0892   |
| 5   | 1024 | 4096 | 4096 | 128        | torch.bfloat16 | 0.5133  | 0.4753 | 2.2016   |

- Times are in `ms`, see `benchmarks/benchmark_hqq.py`.
- `hqq_ref` is the base `HQQ_Linear` [module](https://github.com/mobiusml/hqq/blob/6d50eee4bcdd99cc10716f1297c5b2803d2b6da4/hqq/core/quantize.py#L349) that is unfused (dequantization followed by call to torch.matmul).
- `tinygemm` calls `torch.ops.aten._weight_int4pack_mm`. Implementation is a custom HQQLinear layer that wraps the preprocessing necessary for this kernel, adapted from a benchmark script posted by @mobicham from `CUDA-mode` Discord discussions.

GPU details:

```
_CudaDeviceProperties(name='NVIDIA RTX A6000', major=8, minor=6, total_memory=48676MB, multi_processor_count=84)
```

### NOTE

This implementation requires **`triton >= 3.0.0`**.

- Running tests / benchmarks requires installation of `hqq`:

  ```
  pip install hqq
  ```
