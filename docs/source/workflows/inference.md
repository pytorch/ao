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

### Microbenchmarks and roofline model

The following set of microbenchmarks show the roofline expected and observed execution times of
a `ReLU -> Linear` toy model across a sweep of (M, K, N) shapes, with the activation
shaped (M, K) and the weight shaped (K, N). This can be used to estimate expected speedup
of quantizing `torch.nn.Linear` layers with various recipes based on shapes in your model
during inference.

Explanation: to see speedup from quantization of `activation -> gemm` during inference, we want

```
(bf16_activation_time + bf16_gemm_time) > (bf16_activation_and_quantize_tensor_time + fp8_gemm_time)
```

In a perfect world (and our roofline model), 
1. `bf16_activation_time > bf16_activation_and_quantize_tensor_time` is always true
because `bf16_activation` reads+writes `M*K*2 bytes` and `bf16_activation_and_quantize_tensor` is a single
fused kernel that reads+writes `M*K*1.5 bytes`.
2. `bf16_gemm_time` > `fp8_gemm_time` is always true as fp8 gemm has ~2x peak efficiency vs bf16 gemm

In the real world, both (1) and (2) are not always true due to kernel launch overhead, kernel efficiency,
lack of fusion for some recipes, etc. Therefore, the observed speedups are often significantly
below the roofline peak.  In general you should expect the observed speedup from inference quantization
to increase as MKN increases.

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
0   1024   1024   1024                      1.00            0.93
1   2048   2048   2048                      1.75            1.20
2   4096   4096   4096                      1.90            1.46
3   8192   8192   8192                      1.94            1.76
4  16384  16384  16384                      1.97            1.77

#
# nvfp4 with dynamic global scaling
#
> python benchmarks/float8/float8_inference_roofline.py --recipe_name nvfp4 --enable_fusion_modeling True --skip_printing_detailed_metrics True
...
GPU                     NVIDIA B200
torch version           2.12.0.dev20260312+cu130
torchao version         0.17.0+gitbd7717d20
...
   fwd_M  fwd_K  fwd_N  r_fp8_gemm_and_ovhd_spdp  b_fp8_e2e_spdp
0   1024   1024   1024                      1.00            0.46
1   2048   2048   2048                      2.36            0.76
2   4096   4096   4096                      2.89            1.37
3   8192   8192   8192                      3.32            1.97
4  16384  16384  16384                      3.62            2.77

#
# nvfp4 with static global scaling (user API in progress)
#
> python benchmarks/float8/float8_inference_roofline.py --recipe_name nvfp4_static --enable_fusion_modeling True --skip_printing_detailed_metrics True
...
GPU                     NVIDIA B200
torch version           2.12.0.dev20260312+cu130
torchao version         0.17.0+gitbd7717d20
...
   fwd_M  fwd_K  fwd_N  r_fp8_gemm_and_ovhd_spdp  b_fp8_e2e_spdp
0   1024   1024   1024                      1.00            0.55
1   2048   2048   2048                      2.74            0.95
2   4096   4096   4096                      3.42            1.69
3   8192   8192   8192                      3.67            2.29
4  16384  16384  16384                      3.82            2.98

```

## e2e flux-1.schnell benchmarks

These benchmarks compare accuracy and performance of torchao inference quantization on the
[flux-1.schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) model.

For accuracy, we measure the [LPIPS](https://github.com/richzhang/PerceptualSimilarity) score
between images generated by the quantized model and the high precision (bfloat16) baseline,
averaged over the prompts from the [sayakpaul/drawbench](https://huggingface.co/datasets/sayakpaul/drawbench) dataset —
lower is better, with 0 meaning identical.

Note that this benchmark optimizes for speed of iteration and does not represent 
the best possible metrics someone could achieve on this model. Instead, this is an
apples-to-apples comparison intended to compare different quantization recipes at a
high level, and measure performance improvements.

| experiment | lpips_avg | time_s_bsz_1 | speedup_bsz_1 | time_s_bsz_4 | speedup_bsz_4 |
| ---------- | --------- | ------------- | -------------- | ------------- | -------------- |
| bfloat16 | 0 | 0.4178 | 1.00 | 1.4914 | 1.00 |
| float8_rowwise | 0.1236| 0.3455 | 1.21 | 1.1986 | 1.24 |
| mxfp8 | 0.1260 | 0.3673 | 1.14 | 1.2820 | 1.16 |
| nvfp4 | 0.2694 | 0.3203 | 1.30 | 1.0913 | 1.37 |

To reproduce, run:

```bash
./benchmarks/quantization/eval_accuracy_and_perf_of_flux.sh
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

### Low-Precision FP8 Attention (Prototype)

FP8 low-precision attention for inference, built on Flash Attention backends. Currently supports FA3 on Hopper (SM90) and FA4 on Blackwell (SM100).

**Requirements:** PyTorch >= 2.11, Hopper or Blackwell GPU, Flash Attention 3 (`pip install flash-attn-3 --index-url=https://download.pytorch.org/whl/{cuda_version}`).

```{literalinclude} ../examples/prototype/low_precision_attention.py
:language: python
```

`apply_low_precision_attention` replaces all `F.scaled_dot_product_attention` calls with FP8 attention for eager execution. When combined with `torch.compile`, RoPE patterns are automatically detected and fused into a single kernel. KV caching should be disabled before calling for best results with `torch.compile`. See the {ref}`API reference <api_attention>` for details.
