# Workflows

This page provides an overview of the various workflows available in torchao:

1. by where quantization is applied in the model lifecycle (training vs QAT vs PTQ vs inference) <a href="#workflows-for-training-qat-inference">(link)</a>
2. by type of quantization being applied (float8, mxfp8, int4, etc) <a href="#workflows-status-by-dtype-hardware">(link)</a>

## Workflows for training/QAT/inference

### Training

* Our main training product is for float8 quantized training, the documentation is here: [(link)](https://github.com/pytorch/ao/blob/main/torchao/float8/README.md)
* We have three prototype quantized training workflows: mxfp8 dense [(link)](https://github.com/pytorch/ao/tree/main/torchao/prototype/mx_formats#mx-training),
  mxfp8 MoE [(link)](https://github.com/pytorch/ao/tree/main/torchao/prototype/moe_training#mxfp8-moe-training),
  int8 dense [(link)](https://github.com/pytorch/ao/tree/main/torchao/prototype/quantized_training)

### Quantization-Aware Training (QAT)

See the [QAT documentation](qat.md) for details on how to use quantization-aware training to improve model accuracy after quantization.

### Inference

See the [inference quantization documentation](https://github.com/pytorch/ao/blob/main/torchao/quantization/README.md) for an overview of quantization for inference workflows.

## Workflows status by dtype + hardware

🟢 = stable, 🟡 = prototype, 🟠 = planned, ⚪ = not supported

### NVIDIA CUDA

| recommended hardware | weight | activation | quantized training | QAT | PTQ data algorithms | quantized inference |
| -------- | ------ | ---------- | ------------------ | --- | ------------------- | ------------------- |
| H100, B200 | float8 rowwise | float8 rowwise | 🟢 [(link)](https://github.com/pytorch/ao/tree/main/torchao/float8) | 🟢 [(link)](qat.md) | ⚪ | 🟢 [(link)](https://github.com/pytorch/ao/tree/main/torchao/quantization#a8w8-float8-dynamic-quantization-with-rowwise-scaling) |
| H100 | int4 | float8 rowwise | ⚪ | 🟢 [(link)](qat.md) | 🟠 | 🟢 [(link)](https://github.com/pytorch/ao/blob/257d18ae1b41e8bd8d85849dd2bd43ad3885678e/torchao/quantization/quant_api.py#L1296) |
| A100 | int4 | bfloat16 | ⚪ | 🟢 [(link)](qat.md) | 🟡: [HQQ](https://github.com/pytorch/ao/tree/main/torchao/prototype/hqq/README.md), [AWQ](https://github.com/pytorch/ao/tree/main/torchao/prototype/awq) | 🟢 [(link)](https://github.com/pytorch/ao/tree/main/torchao/quantization#a16w4-weightonly-quantization) |
| A100 | int8 | bfloat16 | ⚪ | 🟢 [(link)](qat.md) | ⚪ | 🟢 [(link)](https://github.com/pytorch/ao/tree/main/torchao/quantization#a16w8-int8-weightonly-quantization) |
| A100 | int8 | int8 | 🟡 [(link)](https://github.com/pytorch/ao/tree/main/torchao/prototype/quantized_training) | 🟢 [(link)](qat.md) | ⚪ | 🟢 [(link)](https://github.com/pytorch/ao/tree/main/torchao/quantization#a8w8-int8-dynamic-quantization) |
| B200 | nvfp4 | nvfp4 | 🟠 | 🟡 [(link)](https://github.com/pytorch/ao/blob/main/torchao/prototype/qat/nvfp4.py) | ⚪ |  🟡 [(link)](https://github.com/pytorch/ao/tree/main/torchao/prototype/mx_formats#mx-inference) |
| B200 | mxfp8 | mxfp8 | 🟡 [(dense)](https://github.com/pytorch/ao/tree/main/torchao/prototype/mx_formats#mx-training), [(moe)](https://github.com/pytorch/ao/tree/main/torchao/prototype/moe_training) | ⚪ | ⚪ | 🟡 [(link)](https://github.com/pytorch/ao/tree/main/torchao/prototype/mx_formats#mx-inference) |
| B200 | mxfp4 | mxfp4 | ⚪ not supported | 🟠 | 🟠 | 🟡 [(link)](https://github.com/pytorch/ao/tree/main/torchao/prototype/mx_formats#mx-inference) |
| H100 | float8 128x128 (blockwise) | float8 1x128 | 🟠 | ⚪ | ⚪ | 🟡 |

### Edge

| recommended hardware | weight | activation | quantized training | QAT | PTQ data algorithms | quantized inference |
| -------- | ------ | ---------- | ------------------ | --- | ------------------- | ------------------- |
| edge | intx (1..7) | bfloat16 | ⚪ | 🟢 [(link)](qat.md) | ⚪ | 🟢 [(link)](https://github.com/pytorch/ao/blob/257d18ae1b41e8bd8d85849dd2bd43ad3885678e/torchao/quantization/quant_api.py#L2267) |

### ROCM

| recommended hardware | weight | activation | quantized training | QAT | PTQ data algorithms | quantized inference |
| -------- | ------ | ---------- | ------------------ | --- | ------------------- | ------------------- |
| MI350x | mxfp8 | mxfp8 | 🟡 [(dense)](https://github.com/pytorch/ao/tree/main/torchao/prototype/mx_formats#mx-training), [(moe)](https://github.com/pytorch/ao/tree/main/torchao/prototype/moe_training) | ⚪ | ⚪ | 🟡 [(link)](https://github.com/pytorch/ao/tree/main/torchao/prototype/mx_formats#mx-inference) |
| MI350x | mxfp4 | mxfp4 | ⚪ not supported | 🟠 | 🟠 | 🟡 [(link)](https://github.com/pytorch/ao/tree/main/torchao/prototype/mx_formats#mx-inference) |

### Intel

| recommended hardware | weight | activation | quantized training | QAT | PTQ data algorithms | quantized inference |
| -------- | ------ | ---------- | ------------------ | --- | ------------------- | ------------------- |
| Intel® BMG | float8 tensor/rowwise | float8 tensor/rowwise |🟠 | 🟢 [(link)](qat.md) | ⚪ | 🟢 [(link)](https://github.com/pytorch/ao/tree/main/torchao/quantization#a8w8-float8-dynamic-quantization-with-rowwise-scaling) |
| Intel® BMG | int4 | float16/bfloat16 | ⚪ | 🟢 [(link)](qat.md) | 🟡: [AWQ](https://github.com/pytorch/ao/tree/main/torchao/prototype/awq) | 🟢 [(link)](https://github.com/pytorch/ao/tree/main/torchao/quantization#a16w4-weightonly-quantization) |
| Intel® BMG | int8 | int8 | 🟠 | 🟢 [(link)](qat.md) | ⚪ | 🟢 [(link)](https://github.com/pytorch/ao/tree/main/torchao/quantization#a8w8-int8-dynamic-quantization) |

### Other
* [Sparsity README.md](https://github.com/pytorch/ao/tree/main/torchao/sparsity/README.md), includes different techniques such as 2:4 sparsity and block sparsity
* [the prototype folder](https://github.com/pytorch/ao/tree/main/torchao/prototype) for other prototype features

```{toctree}
:hidden:
:maxdepth: 1

qat
```
