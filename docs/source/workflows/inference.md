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

## Quantization Techniques

See the [API Reference documentation](https://docs.pytorch.org/ao/main/api_reference/api_ref_quantization.html) for code examples and detailed documentation for each quantization config:

- **float8 weight configs**: `Float8DynamicActivationFloat8WeightConfig`, `Float8WeightOnlyConfig`
- **int8 weight configs**: `Int8DynamicActivationInt8WeightConfig`, `Int8WeightOnlyConfig`
- **int4 weight configs**: `Int4WeightOnlyConfig`, `Float8DynamicActivationInt4WeightConfig`, `Int8DynamicActivationInt4WeightConfig`
- **intx weight configs**: `IntxWeightOnlyConfig`, `Int8DynamicActivationIntxWeightConfig`

Notes:
- The quantization error incurred by applying int4 quantization to your model can be fairly significant, so using external techniques like GPTQ may be necessary to obtain a usable model.
- Float8 quantization requires hardware with CUDA compute capability 8.9 or greater (e.g., H100).
- Third-party backend CI status:
  - Ascend NPU(requires torch_npu ≥ 2.7.1)
  [![Ascend NPU](https://github.com/Ascend/Ascend-CI/actions/workflows/torchao.yml/badge.svg)](https://github.com/Ascend/Ascend-CI/actions/workflows/torchao.yml)


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

All the following benchmarks are for `meta-llama/Llama-3.1-8B` using `torch==2.9.0` and `vllm==0.13.0`.


### NVIDIA B200

| weight | activation | prefill toks/s | decode toks/s | prefill_speedup | decode_speedup |
| ------ | ---------- | -------------- | ------------- | --------------- | -------------- |
| bfloat16 | bfloat16 | 59099.9 | 14380 | 1 | 1 |
| mxfp8 | mxfp8 | TODO(https://github.com/pytorch/ao/issues/3549) | - | - | - |
| nvfp4 | nvfp4 | 102786 | 15218.9 | 1.739 | 1.058 |
| float8_rowwise | float8_rowwise | 69313.7 | 15984 | 1.173 | 1.112 |

### NVIDIA H100

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
