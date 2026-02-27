# Quantized Inference

For inference, we support dynamic and weight-only quantization of `torch.nn.funtional.linear` across various dtype configurations.
The pseudocode is as follows:

```python
# high precision (baseline)
output_bf16 = input_bf16 @ weight_bf16.t()

# dynamic quantization (shown for fp8 rowwise)
output_bf16 = to_fp8(input_bf16) @ to_fp8(weight_fp8.t())

# weight-only quantization (shown for int4)
output_bf16 = input_bf16 @ weight_int4.t()
```

## Inference Workflows

Below are the stable and near-stable inference workflows in torchao:

| weight dtype | act dtype | summary |
|--------------|-----------|---------|
| float8 | float8 | {class}`~torchao.quantization.Float8DynamicActivationFloat8WeightConfig`: Applies float8 dynamic symmetric quantization to both activations and weights. Requires CUDA ≥8.9, AMD MI350+, or Intel XPU. Supports `PerTensor` and `PerRow` granularity. |
| float8 | bf16 | {class}`~torchao.quantization.Float8WeightOnlyConfig`: Applies float8 weight-only symmetric per-channel quantization. Matmul computed in original precision. |
| int8 | int8 | {class}`~torchao.quantization.Int8DynamicActivationInt8WeightConfig`: Applies int8 dynamic symmetric per-token activation and int8 per-channel weight quantization. |
| int8 | bf16 | {class}`~torchao.quantization.Int8WeightOnlyConfig`: Applies int8 weight-only symmetric per-channel quantization. |
| mxfp8 | mxfp8 | {class}`~torchao.prototype.mx_formats.MXDynamicActivationMXWeightConfig`(prototype): Applies mxfp8 or mxfp4 dynamic quantization to activations and weights. Requires NVIDIA SM100+ (Blackwell) or AMD MI350+. |
| int4 | bf16 | {class}`~torchao.quantization.Int4WeightOnlyConfig`: Applies int4 weight-only groupwise quantization. Supports group sizes 256, 128, 64, 32. |
| int4 | float8 | {class}`~torchao.quantization.Float8DynamicActivationInt4WeightConfig`: Applies float8 dynamic per-row activation and int4 per-group weight quantization. Group size 128 only. |
| nvfp4 | bf16 | {class}`~torchao.prototype.mx_formats.NVFP4WeightOnlyConfig`(prototype): Applies NVFP4 weight-only quantization. |
| nvfp4 | nvfp4 | {class}`~torchao.prototype.mx_formats.NVFP4DynamicActivationNVFP4WeightConfig`(prototype): Applies NVFP4 dynamic quantization to activations and weights with double quantization (per-tensor + per-block scales). Requires NVIDIA SM100+ (Blackwell). |
| mxfp4 | mxfp4 | {class}`~torchao.prototype.mx_formats.MXDynamicActivationMXWeightConfig`(prototype): Applies mxfp8 or mxfp4 dynamic quantization to activations and weights. Requires NVIDIA SM100+ (Blackwell) or AMD MI350+. |
| intx | bf16 | {class}`~torchao.quantization.IntxWeightOnlyConfig`: Applies intx (1-8 bit) weight-only quantization. Supports groupwise and per-channel. Works with Linear and Conv2D. |
| intx | int8 | {class}`~torchao.quantization.Int8DynamicActivationIntxWeightConfig`: Applies int8 dynamic per-token activation and intx (1-8 bit) weight quantization. CPU optimized. |


## Accuracy benchmarks

All the following benchmarks are for `meta-llama/Llama-3.1-8B` using `lm-eval`.

| weight | activation | wikitext-perplexity | winogrande | checkpoint size (GB) |
| --------- | ------------------- | ---------- | -------------------- | -------- |
| bfloat16 | bfloat16 | 7.3315 | 0.7380 | 16.1 |
| float8_rowwise | float8_rowwise | 7.4197 | 0.7388 | 9.1 |
| int8_rowwise | bfloat16 | 7.3451 | 0.7340 | 9.1 |
| int8_rowwise | int8_rowwise | 7.4535 | 0.7285 | 9.1 |
| mxfp8 | mxfp8 | 7.6034 | 0.7316 | 9.32 |
| nvfp4 | nvfp4 | 8.4459 | 0.7135 | 6.05 |

To reproduce, run the following command:

```bash
// on an H100
SKIP_VLLM=1 ./benchmarks/quantization/measure_accuracy_and_performance.sh h100
// on a B200
SKIP_VLLM=1 ./benchmarks/quantization/measure_accuracy_and_performance.sh b200
```

## Performance benchmarks

### e2e model level benchmarks

All the following benchmarks are for `meta-llama/Llama-3.1-8B` using `torch==2.9.0` and `vllm==0.13.0`.


#### NVIDIA B200

| weight | activation | prefill toks/s | decode toks/s | prefill_speedup | decode_speedup |
| ------ | ---------- | -------------- | ------------- | --------------- | -------------- |
| bfloat16 | bfloat16 | 59099.9 | 14380 | 1 | 1 |
| mxfp8 | mxfp8 | TODO(https://github.com/pytorch/ao/issues/3549) | - | - | - |
| nvfp4 | nvfp4 | 102786 | 15218.9 | 1.739 | 1.058 |
| float8_rowwise | float8_rowwise | 69313.7 | 15984 | 1.173 | 1.112 |

#### NVIDIA H100

| weight | activation | prefill toks/s | decode toks/s | prefill_speedup | decode_speedup |
| ------ | ---------- | -------------- | ------------- | --------------- | -------------- |
| bfloat16 | bfloat16 | 30946.5 | 6612 | 1 | 1 |
| float8_rowwise | float8_rowwise | 45312.5 | 8025.95 | 1.464 | 1.214 |
| int8_rowwwise | bfloat16 | 28231.9 | 4309.8 | 0.912 | 0.652 |
| int4 | float8_rowwise | TODO(https://github.com/pytorch/ao/issues/3550) | - | - | - |

To reproduce these benchmarks, run

```bash
// on an h100
SKIP_LM_EVAL=1 ./benchmarks/quantization/measure_accuracy_and_performance.sh h100
// on a b200
SKIP_LM_EVAL=1 ./benchmarks/quantization/measure_accuracy_and_performance.sh h100

// under the hood, the actual vllm benchmark is doing the following:
// 1. prefill
vllm bench throughput --num_prompts 32 --input_len 4096 --output_len 32 --max_model_len 4128
// 2. decode
vllm bench throughput --num_prompts 128 --input_len 32 --output_len 2048 --max_model_len 2080
```

### Microbenchmarks

The following set of microbenchmarks measures the roofline peak and observed execution time of
a `ReLU -> Linear` toy model swept across various (M, K, N) shapes, with the activation
shaped (M, K) and the weight shaped (K, N). This can be used to estimate expected speedup
of quantizing `torch.nn.Linear` layers with various recipes based on the activation and weight shapes.

#### NVIDIA B200

```bash
# `r_fp8_gemm_and_ovhd_spdp` is the roofline expected speedup of the
#    quantized ReLU -> Linear layer vs high precision version
# `b_fp8_e2e_spdp` is the observed speedup of the quantized
#    ReLU -> Linear layer vs high precision version

#
# mxfp8
#
> python benchmarks/float8/float8_inference_roofline.py --recipe_name mxfp8_cublas --enable_fusion_modeling True --skip_printing_detailed_metrics True
...
GPU                     NVIDIA B200
torch version           2.12.0.dev20260218+cu130
torchao version         0.17.0+git3075bb624
...
   fwd_M  fwd_K  fwd_N  r_fp8_gemm_and_ovhd_spdp  b_fp8_e2e_spdp
0   1024   1024   1024                      0.64            0.94
1   2048   2048   2048                      1.75            1.21
2   4096   4096   4096                      1.90            1.45
3   8192   8192   8192                      1.94            1.75
4  16384  16384  16384                      1.97            1.77

#
# nvfp4 with dynamic global scaling
#
> python benchmarks/float8/float8_inference_roofline.py --recipe_name nvfp4 --enable_fusion_modeling True --skip_printing_detailed_metrics True
...
GPU                     NVIDIA B200
torch version           2.12.0.dev20260218+cu130
torchao version         0.17.0+git3075bb624
...
   fwd_M  fwd_K  fwd_N  r_fp8_gemm_and_ovhd_spdp  b_fp8_e2e_spdp
0   1024   1024   1024                      0.64            0.37
1   2048   2048   2048                      2.39            0.74
2   4096   4096   4096                      2.92            1.19
3   8192   8192   8192                      3.34            1.78
4  16384  16384  16384                      3.63            2.57
```

## Other Available Quantization Techniques

### Int8DynamicActivationIntxWeightConfig Quantization
We have kernels that do 8-bit dynamic quantization of activations and uintx groupwise quantization of weights.  These kernels are experimental and can only be run on a device with an ARM CPU (e.g., a Mac computers with Apple silicon).  The benchmarks below were run on an M1 Mac Pro, with 8 perf cores, and 2 efficiency cores, and 32GB of RAM.  In all cases, torch.compile was used.

| Model         | Technique                                        | Tokens/Second | Memory Bandwidth (GB/s) | Peak Memory (GB) | Model Size (GB) |
| ------------- | -------------------------------------------------| --------------| ------------------------| ---------------- | ----------------|
| Llama-3.1-8B  | Base (bfloat16)                                  |  1.24         |  18.62                  |  NA              | 15.01           |
|               | int8_dynamic_activation_intx_weight-4-256-false  |  16.03        |  65.81                  |  NA              | 4.11            |
|               | int8_dynamic_activation_intx_weight-3-256-false  |  18.94        |  59.97                  |  NA              | 3.17            |

You can try out these apis with the `quantize_` api as above alongside the config `Int8DynamicActivationIntxWeightConfig`.  An example can be found in `torchao/_models/llama/generate.py`.

### Codebook Quantization
The benchmarks below were run on a single NVIDIA-A6000 GPU.

| Model       | Technique               | wikitext-perplexity | Tokens/Second | Memory Bandwidth (GB/s) | Peak Memory (GB) | Model Size (GB) |
| ----------- | ----------------------- | ------------------- | ------------- | ----------------------- | ---------------- | --------------- |
| Llama-3-8B  | Base (bfloat16)         |  7.590              |  32.36        |  485.71                 | 16.19            | 15.01           |
|             | codebook-4-64           |  9.533              |  1.73         |  8.62                   | 23.11            |  4.98           |
| Llama-3.1-8B| Base (bfloat16)         |  7.713              |  32.16        |  482.70                 | 16.35            | 15.01           |
|             | codebook-4-64           |  10.095             |  1.73         |  8.63                   | 23.11            |  4.98           |

You try can out these apis with the `quantize_` api as above alongside the config `CodebookWeightOnlyConfig` an example can be found in  in `torchao/_models/llama/generate.py`.
