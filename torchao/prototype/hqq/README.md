## Fused `int4 / fp16` Quant Matmul

Fused kernel that combines asymmetric dequantization and gemm. Useful primarily for compute-bound (M > 16) scenarios and not for memory-bound / inference scenarios.

The kernel fuses two ops:

- Dequantization: upcasts `u4 / s4` weights to `float16 / bfloat16`, followed by groupwise scaling and shifting by scales / zeropoints
- GEMM: matmul on dequantized weights and activations.

Tested and benchmarked for `HQQ` but could theoretically be used for any asymmetric quantization scheme.

> **NOTE**: Benchmark below is only indicative of performance on consumer-grade `Ampere` GPUs (`A6000` specifically). When tested on `H100`, the performance is on par / marginally worse than native / compiled `torch`.  
> The intended use is thus for fine-tuning / training models on non-datacenter GPUs (`80 <= compute capability < 90`). If interested in optimizing the kernel for other architectures, please drop a note in the CUDA-MODE Discord channel.

### Usage

Typical workflow:

- quantize `float16 / bfloat16` weights to `s4 / u4` using a group-wise asymmetric quantization scheme, outputs are the quantized 4b weights stored as `torch.int8 / torch.uint8`
- pack weights using `pack_2xint4` such that 2 weights are packed per `torch.int8 / torch.uint8`.
- pass the packed weights, scales, and zeros to the kernel

If running transposed matmul (e.g., for backwards passes during training), there is no need to unpack / re-pack the weights, simply pass `transposed=True` to the kernel.

The pseudocode below explains the expected shapes and dtypes. Also see `test/hqq/test_triton_mm.py` for a concrete example of usage with `HQQ`.

```python

#The reason we use N x K is to match that shape of the weight for a torch.nn.Linear layer, where N -> out-features, K -> in-features
weights = torch.randn(N, K, dtype=torch.float16, device="cuda")

#Perform groupwise asymmetric quantization along axis=1 (in-features). E.g., `scales = Wq.reshape(-1, group_size).max(axis=1)`.
#Wq are `s4 / u4` stored as dtype = torch.int8 / torch.uint8, shape N x K
# scales and zeros are shape (N * K // group_size)
Wq, scales, zeros = quantize(weights) #Choose your favorite quantization library

#Pack i4 stored as i8 to packed 2xi4 i8.
#Note that we transpose W_q such that the packed shape is (K // 2) x N, and when unpacked K x N
packed_w = pack_2xint4(W_q.T)

#Reshape scales such that they can be broadcasted within kernel
scales = scales.reshape(N, -1)
zeros = zeros.reshape(N, -1)

#Sample input
x = torch.randn(M, K, dtype=torch.float16, device="cuda")

#Run fused dequant matmul
#If running transposed case such as for backwards pass,
#switch transposed to True
tt_out = triton_mixed_mm(
          x,
          packed_w,
          scales.T,
          zeros.T,
          transposed=False,
          group_size=group_size,
          fp8_fast_accum=False,
      )
```

### Implementation Details

- Bitpacking is simple row interleave, no need for extensive preprocessing (e.g., `tinygemm` or `fastertransformer`)
- Tested for `float16 / bfloat16` activations, scales, and zeros
- Autotuned for both compute-bound and memory-bound configs
- Assumes operand B of the `gemm` is is the quantized type.
- Requires quantization along `in-features`, i.e., the `K` dimension, or `axis=1`, of `torch.linear.weight`.
- Implementation handles both transposed and non-tranposed quantized weights, useful for forward / backward training passes.

### Performance

Initial benchmarking (on `A6000`) demonstrates promising results, scaling well for compute-bound workloads:

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
