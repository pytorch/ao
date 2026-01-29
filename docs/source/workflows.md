# Workflows

This page provides an overview of the quantization and training workflows available in torchao.

## Stable Workflows

🟢 = stable, 🟡 = prototype, 🟠 = planned, ⚪ = not supported

| recommended hardware | weight | activation | quantized training | QAT | PTQ data algorithms | quantized inference |
| -------- | ------ | ---------- | ------------------ | --- | ------------------- | ------------------- |
| H100, B200 GPUs | float8 rowwise | float8 rowwise | 🟢 [(link)](https://github.com/pytorch/ao/tree/main/torchao/float8) | 🟢 [(link)](https://github.com/pytorch/ao/tree/main/torchao/quantization/qat) | ⚪ | 🟢 [(link)](https://github.com/pytorch/ao/tree/main/torchao/quantization#a8w8-float8-dynamic-quantization-with-rowwise-scaling) |
| Intel® BMG GPUs | float8 tensor/rowwise | float8 tensor/rowwise |🟠 | 🟢 [(link)](https://github.com/pytorch/ao/tree/main/torchao/quantization/qat) | ⚪ | 🟢 [(link)](https://github.com/pytorch/ao/tree/main/torchao/quantization#a8w8-float8-dynamic-quantization-with-rowwise-scaling) |
| H100 GPUs | int4 | float8 rowwise | ⚪ | 🟢 [(link)](https://github.com/pytorch/ao/tree/main/torchao/quantization/qat) | 🟠 | 🟢 [(link)](https://github.com/pytorch/ao/blob/257d18ae1b41e8bd8d85849dd2bd43ad3885678e/torchao/quantization/quant_api.py#L1296) |
| A100 GPUs | int4 | bfloat16 | ⚪ | 🟢 [(link)](https://github.com/pytorch/ao/tree/main/torchao/quantization/qat) | 🟡: [HQQ](https://github.com/pytorch/ao/tree/main/torchao/prototype/hqq/README.md), [AWQ](https://github.com/pytorch/ao/tree/main/torchao/prototype/awq) | 🟢 [(link)](https://github.com/pytorch/ao/tree/main/torchao/quantization#a16w4-weightonly-quantization) |
| Intel® BMG GPUs | int4 | float16/bfloat16 | ⚪ | 🟢 [(link)](https://github.com/pytorch/ao/tree/main/torchao/quantization/qat) | 🟡: [AWQ](https://github.com/pytorch/ao/tree/main/torchao/prototype/awq) | 🟢 [(link)](https://github.com/pytorch/ao/tree/main/torchao/quantization#a16w4-weightonly-quantization) |
| A100 GPUs | int8 | bfloat16 | ⚪ | 🟢 [(link)](https://github.com/pytorch/ao/tree/main/torchao/quantization/qat) | ⚪ | 🟢 [(link)](https://github.com/pytorch/ao/tree/main/torchao/quantization#a16w8-int8-weightonly-quantization) |
| A100 GPUs | int8 | int8 | 🟡 [(link)](https://github.com/pytorch/ao/tree/main/torchao/prototype/quantized_training) | 🟢 [(link)](https://github.com/pytorch/ao/tree/main/torchao/quantization/qat) | ⚪ | 🟢 [(link)](https://github.com/pytorch/ao/tree/main/torchao/quantization#a8w8-int8-dynamic-quantization) |
| Intel® BMG GPUs | int8 | int8 | 🟠 | 🟢 [(link)](https://github.com/pytorch/ao/tree/main/torchao/quantization/qat) | ⚪ | 🟢 [(link)](https://github.com/pytorch/ao/tree/main/torchao/quantization#a8w8-int8-dynamic-quantization) |
| edge | intx (1..7) | bfloat16 | ⚪ | 🟢 [(link)](https://github.com/pytorch/ao/tree/main/torchao/quantization/qat) | ⚪ | 🟢 [(link)](https://github.com/pytorch/ao/blob/257d18ae1b41e8bd8d85849dd2bd43ad3885678e/torchao/quantization/quant_api.py#L2267) |

## Prototype Workflows

🟢 = stable, 🟡 = prototype, 🟠 = planned, ⚪ = not supported

| recommended hardware | weight | activation | quantized training | QAT | PTQ data algorithms | quantized inference |
| -------- | ------ | ---------- | ------------------ | --- | ------------------- | ------------------- |
| B200, MI350x GPUs | mxfp8 | mxfp8 | 🟡 [(dense)](https://github.com/pytorch/ao/tree/main/torchao/prototype/mx_formats#mx-training), [(moe)](https://github.com/pytorch/ao/tree/main/torchao/prototype/moe_training) | ⚪ | ⚪ | 🟡 [(link)](https://github.com/pytorch/ao/tree/main/torchao/prototype/mx_formats#mx-inference) |
| B200 GPUs | nvfp4 | nvfp4 | 🟠 | 🟡 [(link)](https://github.com/pytorch/ao/blob/main/torchao/prototype/qat/nvfp4.py) | ⚪ |  🟡 [(link)](https://github.com/pytorch/ao/tree/main/torchao/prototype/mx_formats#mx-inference) |
| B200, MI350x GPUs | mxfp4 | mxfp4 | ⚪ not supported | 🟠 | 🟠 | 🟡 [(link)](https://github.com/pytorch/ao/tree/main/torchao/prototype/mx_formats#mx-inference) |
| H100 | float8 128x128 (blockwise) | float8 1x128 | 🟠 | ⚪ | ⚪ | 🟡 |

## Other

* [Quantization-Aware Training (QAT) README.md](https://github.com/pytorch/ao/tree/main/torchao/quantization/qat/README.md)
* [Post-Training Quantization (PTQ) README.md](https://github.com/pytorch/ao/tree/main/torchao/quantization/README.md)
* [Sparsity README.md](https://github.com/pytorch/ao/tree/main/torchao/sparsity/README.md), includes different techniques such as 2:4 sparsity and block sparsity
* [the prototype folder](https://github.com/pytorch/ao/tree/main/torchao/prototype) for other prototype features
